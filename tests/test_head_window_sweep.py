"""head_window_sweep on a tiny random Qwen2.5-VL: the identities every window row rests on.

A full-coverage window installed at the last token must reproduce the image patch's
first answer token (head_trace's phase-1c identity carried through generation), an
empty window must reproduce the unhooked run, and a window below the patch layer
must be refused. Mechanics only -- not the scientific outcome on the 7B model.
"""
import pytest
import torch

from test_head_followups import tiny
from methods.head_trace import generate_with_patches, head_patches, image_patch, capture_head_outputs
from methods.head_window_sweep import Tally, enumerate_windows, parse_window, sweep_windows, window_label
from methods.ndm.verify_sites import generate_unhooked

PATCH_LAYER = 1
MAX_NEW = 3


def paired_batch(tiny, seed=7):
    """Two rows, each four merged image tokens at columns 1-4, last prompt token at column 8."""
    runner, _ = tiny
    torch.manual_seed(seed)
    ids = torch.tensor([[1, 2, 2, 2, 2, 3, 12, 13, 14]] * 2)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    b_img = {'rows': [{'row_index': 0}, {'row_index': 1}],
             'base_input_ids': ids, 'source_input_ids': ids.clone(), 'attention_mask': torch.ones_like(ids),
             'base_extra': {'pixel_values': torch.randn(32, 24) * 3, 'image_grid_thw': grid},
             'source_extra': {'pixel_values': torch.randn(32, 24) * 3, 'image_grid_thw': grid},
             'positions': torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])}
    b_last = dict(b_img, positions=torch.tensor([[8], [8]]))
    return runner, b_img, b_last


def with_golds(b_img, base_gen, source_gen):
    """Golds = what the unhooked / image-patched runs actually generate, so the
    expected rates are known exactly."""
    out = dict(b_img)
    out['base_gold_toks'], out['base_gold_len'] = base_gen[:, :3], torch.tensor([3, 3])
    out['source_gold_toks'], out['source_gold_len'] = source_gen[:, :3], torch.tensor([3, 3])
    return out


def test_full_window_reproduces_image_patch_first_token_and_empty_window_is_unhooked(tiny):
    adapter, model = tiny[0].adapter, tiny[0].model
    n_layers = len(adapter.get_decoder_layers(model))
    # A random tiny model's greedy answer rarely depends on one image token, so search for images
    # whose image patch flips the first token in BOTH rows; without that the test is vacuous.
    for seed in range(200):
        runner, b_img, b_last = paired_batch(tiny, seed)
        base_gen = generate_unhooked(model, b_img['base_input_ids'], b_img['attention_mask'],
                                     b_img['base_extra'], 0, MAX_NEW)
        src_gen = generate_with_patches(adapter, model, [image_patch(adapter, model, b_img, PATCH_LAYER)],
                                        b_img['base_input_ids'], b_img['attention_mask'], b_img['base_extra'],
                                        0, MAX_NEW)
        if (base_gen[:, 0] != src_gen[:, 0]).all():
            break
    else:
        pytest.fail('no seed gave an image patch that flips the first token; test is vacuous')
    b_img = with_golds(b_img, base_gen, src_gen)
    b_last = dict(b_last, **{k: b_img[k] for k in b_img if 'gold' in k})

    full = list(range(PATCH_LAYER, n_layers))
    rep = sweep_windows(adapter, model, [(b_img, b_last)], PATCH_LAYER, [full, [PATCH_LAYER]], 0, MAX_NEW,
                        log=lambda m: None)
    io, ref = rep['image_patch_only'], rep['windows'][0]
    assert rep['unhooked']['first_src'] == 0.0 and rep['unhooked']['base_kept'] == 1.0
    assert io['cause'] == 1.0 and io['first_src'] == 1.0
    # The identity is exact for the FIRST answer token only: later decode steps also attend to
    # the image tokens' K/V, which the image patch changes and the head install does not.
    assert ref['suff']['first_src'] == io['first_src']
    # Restoring every downstream head under the image patch must undo it at the first token.
    assert ref['nec']['first_base'] == 1.0
    assert set(rep['per_block_mass']) == {str(b) for b in full}


def test_empty_head_set_is_the_unhooked_run(tiny):
    runner, b_img, b_last = paired_batch(tiny)
    adapter, model = runner.adapter, runner.model
    blocks = [PATCH_LAYER]
    z, _ = capture_head_outputs(adapter, model, blocks, b_last['positions'], b_img['base_input_ids'],
                                b_img['attention_mask'], b_img['base_extra'],
                                patches=[image_patch(adapter, model, b_img, PATCH_LAYER)])
    patched = generate_with_patches(adapter, model, head_patches([], z, b_last['positions'], 32, 8, model.device),
                                    b_img['base_input_ids'], b_img['attention_mask'], b_img['base_extra'], 0,
                                    MAX_NEW)
    clean = generate_unhooked(model, b_img['base_input_ids'], b_img['attention_mask'], b_img['base_extra'], 0,
                              MAX_NEW)
    assert torch.equal(patched, clean)


def test_window_below_patch_layer_is_refused(tiny):
    runner, b_img, b_last = paired_batch(tiny)
    with pytest.raises(AssertionError, match='below'):
        sweep_windows(runner.adapter, runner.model, [(b_img, b_last)], 2, [[1, 2]], 0, MAX_NEW,
                      log=lambda m: None)


def test_window_parsing_and_enumeration():
    assert parse_window('21-23') == [21, 22, 23]
    assert parse_window('22') == [22]
    assert parse_window('21.23') == [21, 23]
    assert window_label([21, 22, 23]) == '21-23' and window_label([21, 23]) == '21.23'
    ws = enumerate_windows(21, 24, 2)
    assert [21] in ws and [23, 24] in ws and [21, 22, 23, 24] in ws and [21, 22, 23] not in ws
    assert len(ws) == 4 + 3 + 1


def test_first_token_rate_ignores_rows_with_a_shared_leading_token():
    t = Tally()
    batch = {'source_gold_toks': torch.tensor([[1, 5, 0], [7, 8, 0]]), 'source_gold_len': torch.tensor([2, 2]),
             'base_gold_toks': torch.tensor([[1, 6, 0], [9, 8, 0]]), 'base_gold_len': torch.tensor([2, 2])}
    t.add(torch.tensor([[1, 6], [7, 3]]), batch)
    r = t.rates()
    assert r['n_first'] == 1 and r['first_src'] == 1.0   # row 0 shares its first token: not counted
    assert r['cause'] == 0.0 and r['base_kept'] == 0.5 and r['other'] == 0.5
