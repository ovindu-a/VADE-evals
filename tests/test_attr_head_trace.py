"""Question-side head trace: mechanics checked on a real, randomly initialized tiny VLM.

These cover the parts that are new here -- the batch's gold/position bookkeeping,
the question-side patch, and the read-back identity phases 2-4 rest on. The phase
machinery itself is head_trace's and is exercised by its own tests.
"""
import pytest
import torch

from test_head_followups import tiny
from methods.attr_head_trace import main, question_patch, sample_rows, score_switch
from methods.attribute_switch_sweep import alignment
from methods.head_trace import capture_head_outputs, head_patches, probe_identity


def switch_batch(tiny):
    """Two rows over one shared image, differing only after the image span."""
    runner, _ = tiny
    ids = torch.tensor([[1, 2, 2, 2, 2, 3, 12, 13, 14], [1, 2, 2, 2, 2, 3, 12, 13, 14]])
    donor = ids.clone()
    donor[:, 7] = 15
    pad = 0
    return runner, {
        'rows': [{'row_index': 0, 'base': 'AA', 'donor_attribute': 'currency'},
                 {'row_index': 1, 'base': 'BB', 'donor_attribute': 'capital'}],
        'base_input_ids': ids, 'source_input_ids': donor, 'attention_mask': torch.ones_like(ids),
        'base_extra': {'pixel_values': torch.randn(32, 24),
                       'image_grid_thw': torch.tensor([[1, 4, 4], [1, 4, 4]])},
        'positions': torch.tensor([[6, 7], [6, 7]]),
        'last_positions': torch.tensor([[8], [8]]),
        'all_text_positions': alignment(ids[:1], donor[:1], 2, [1, 3]), 'first_divergence': 7,
        'base_gold_toks': torch.tensor([[20, 21, pad], [22, pad, pad]]),
        'base_gold_len': torch.tensor([2, 1]),
        'source_gold_toks': torch.tensor([[30, 31, pad], [32, pad, pad]]),
        'source_gold_len': torch.tensor([2, 1]),
    }


def test_score_first_is_not_score_full():
    """A multi-token gold whose FIRST token is right must count as first-match and
    not as full-match -- the whole reason both columns are reported."""
    batch = {'source_gold_toks': torch.tensor([[30, 31, 0]]), 'source_gold_len': torch.tensor([2]),
             'base_gold_toks': torch.tensor([[20, 21, 0]]), 'base_gold_len': torch.tensor([2])}
    first, full, _, _ = score_switch(torch.tensor([[30, 99, 0]]), batch)
    assert (first, full) == (1.0, 0.0)
    first, full, _, _ = score_switch(torch.tensor([[30, 31, 0]]), batch)
    assert (first, full) == (1.0, 1.0)
    first, full, _, _ = score_switch(torch.tensor([[99, 31, 0]]), batch)
    assert (first, full) == (0.0, 0.0)


def test_self_patch_at_question_positions_is_a_no_op(tiny):
    runner, batch = switch_batch(tiny)
    adapter, model = runner.adapter, runner.model
    layers = adapter.get_decoder_layers(model)
    with torch.no_grad():
        clean = model(input_ids=batch['base_input_ids'], attention_mask=batch['attention_mask'],
                      **batch['base_extra'], logits_to_keep=1).logits[:, -1]
    same = dict(batch, source_input_ids=batch['base_input_ids'])
    site, layer, fn = question_patch(adapter, model, same, 2)
    handles = site.register(adapter, model, layers, layer, fn)
    try:
        with torch.no_grad():
            patched = model(input_ids=batch['base_input_ids'], attention_mask=batch['attention_mask'],
                            **batch['base_extra'], logits_to_keep=1).logits[:, -1]
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(patched, clean)


def test_question_patch_actually_moves_the_read_column(tiny):
    """The patch is at question columns only, so it must reach the read column
    through attention and nowhere else -- and it must reach it at all."""
    runner, batch = switch_batch(tiny)
    adapter, model = runner.adapter, runner.model
    blocks = [2, 3]
    clean_z, _ = capture_head_outputs(adapter, model, blocks, batch['last_positions'],
                                      batch['base_input_ids'], batch['attention_mask'],
                                      batch['base_extra'])
    patched_z, _ = capture_head_outputs(adapter, model, blocks, batch['last_positions'],
                                        batch['base_input_ids'], batch['attention_mask'],
                                        batch['base_extra'],
                                        patches=[question_patch(adapter, model, batch, 2)])
    assert any((patched_z[b] - clean_z[b]).abs().max() > 0 for b in blocks)


def test_readback_identity_holds_with_full_downstream_coverage(tiny):
    """Phase 1c's closed identity, on a model small enough to check exactly.

    Installing every traced head's patched value at the read column must
    reproduce the question-patched run's final residual there, because the patch
    is at other positions, attention is the only cross-position operation, and
    MLPs are position-wise. If this fails, phases 2-4 of a real run are void.
    """
    runner, batch = switch_batch(tiny)
    adapter, model = runner.adapter, runner.model
    n_layers = len(adapter.get_decoder_layers(model))
    patch_layer = 1
    blocks = list(range(patch_layer, n_layers))
    hidden, n_heads = runner.hidden, runner.n_heads
    head_dim = hidden // n_heads
    pos = batch['last_positions']
    args = (adapter, model, blocks, blocks[0], pos, batch['base_input_ids'], batch['attention_mask'],
            batch['base_extra'])
    _, _, clean_entry, clean_final = probe_identity(*args)
    pat_z, _, pat_entry, pat_final = probe_identity(
        *args, patches=[question_patch(adapter, model, batch, patch_layer)])
    every = [(b, h) for b in blocks for h in range(n_heads)]
    _, _, _, inst_final = probe_identity(
        *args, patches=head_patches(every, pat_z, pos, hidden, head_dim, model.device))

    def rel(got, want):
        return (got.float() - want.float()).norm().item() / max(want.float().norm().item(), 1e-9)

    assert rel(pat_entry, clean_entry) < 1e-5, 'the patch reached the read column before the window'
    assert rel(pat_final, clean_final) > 1e-4, 'the question patch changed nothing; test is vacuous'
    assert rel(inst_final, pat_final) < 2e-2


def test_blocks_below_patch_layer_are_rejected():
    """A block whose attention runs before the patch cannot respond to it. Caught
    at argument time rather than reported as a row of zeros."""
    with pytest.raises(SystemExit):
        main(['--dry_run', '--patch_layer', '16', '--blocks', '15', '16'])


def test_rows_are_all_directed_attribute_pairs_over_one_image(tmp_path):
    items = {'AA': {'image': 'a.png', 'capital': 'X', 'currency': 'Y', 'language': 'Z',
                    'calling_code': '1'}}
    (tmp_path / 'a.png').write_bytes(b'')
    chosen, rows = sample_rows(tmp_path, items, ['capital', 'currency', 'language'], None, 1, 0)
    assert chosen == ['AA'] and len(rows) == 6
    assert all(r['base'] == r['source'] for r in rows)
    assert {(r['base_attribute'], r['donor_attribute']) for r in rows} == {
        ('capital', 'currency'), ('currency', 'capital'), ('capital', 'language'),
        ('language', 'capital'), ('currency', 'language'), ('language', 'currency')}
    assert all(r['base_label'] == items['AA'][r['base_attribute']] and
               r['source_label'] == items['AA'][r['donor_attribute']] for r in rows)
