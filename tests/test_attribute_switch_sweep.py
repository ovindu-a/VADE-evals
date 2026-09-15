"""Same-image prompt interventions on a real, randomly initialized tiny VLM."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_head_followups import tiny
from methods.attribute_switch_sweep import SwitchRunner, alignment, execute, sweep_arms, patch_positions
from methods.head_followup_common import Results


def setup(tiny):
    original, batch = tiny
    runner = SwitchRunner.__new__(SwitchRunner)
    runner.__dict__.update(original.__dict__)
    runner.layers = runner.adapter.get_decoder_layers(runner.model)
    batch['source_input_ids'][0, 4] = 15
    batch['paraphrase_ids'] = batch['base_input_ids'].clone()
    batch['paraphrase_ids'][0, 3] = 16
    batch['earlier_positions'] = alignment(batch['base_input_ids'], batch['source_input_ids'], 2, [1, 3])
    return runner, batch


def test_alignment_rejects_shift_and_image_context_changes():
    base = torch.tensor([[1, 2, 3, 12, 13, 14]])
    assert alignment(base, base.clone(), 2, [1, 3]) == [3, 4]
    with pytest.raises(ValueError, match='equal token lengths'):
        alignment(base, base[:, :-1], 2)
    changed = base.clone()
    changed[0, 0] = 10
    with pytest.raises(ValueError, match='after the image'):
        alignment(base, changed, 2)


def test_sweep_plan_baseline_only_and_span_alias():
    args = SimpleNamespace(sites=['last_residual'], block_spans=[1, 4], layers=[2],
                           modes=['prefill', 'continuous'], controls=['self'], baselines_only=False)
    arms = sweep_arms(args)
    assert len(arms) == 8
    assert {a[1] for a in arms} == {'last_residual', 'last_joint'}
    args.baselines_only = True
    assert sweep_arms(args) == []


@pytest.mark.parametrize('site', ['earlier_text', 'last_residual', 'last_attention', 'last_mlp',
                                 'last_joint', 'blocks:2', 'blocks:4'])
def test_self_patch_identity_all_sites(tiny, site):
    runner, batch = setup(tiny)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23]])), dim=1)
    baseline = runner.run(ids, batch)
    for mode in ['prefill', 'continuous']:
        logits, _ = runner.step(batch, site, 4, mode, 'self')(ids)
        torch.testing.assert_close(logits, baseline)
    assert not runner.layers[-1]._forward_hooks


def test_final_readout_positive_and_earlier_text_negative_controls(tiny):
    runner, batch = setup(tiny)
    ids = batch['base_input_ids']
    clean = runner.run(ids, batch)
    donor = runner.run(batch['source_input_ids'], batch)
    assert not torch.allclose(clean, donor)
    patched, _ = runner.step(batch, 'last_residual', 4, 'prefill')(ids)
    torch.testing.assert_close(patched, donor)
    null, _ = runner.step(batch, 'earlier_text', 4, 'prefill')(ids)
    torch.testing.assert_close(null, clean)
    # Same initial last-token embedding plus all block contributions gives donor residual.
    whole_span, _ = runner.step(batch, 'blocks:4', 4, 'prefill')(ids)
    torch.testing.assert_close(whole_span, donor)
    extended = torch.cat((ids, torch.tensor([[22, 23, 24]])), dim=1)
    donor_extended = torch.cat((batch['source_input_ids'], extended[:, ids.shape[1]:]), dim=1)
    continuous, _ = runner.step(batch, 'last_residual', 4, 'continuous')(extended)
    torch.testing.assert_close(continuous, runner.run(donor_extended, batch))


def test_joint_alias_and_donor_prefix_alignment(tiny):
    runner, batch = setup(tiny)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23]])), dim=1)
    joint, _ = runner.step(batch, 'last_joint', 3, 'continuous')(ids)
    span, _ = runner.step(batch, 'blocks:1', 3, 'continuous')(ids)
    torch.testing.assert_close(joint, span)
    calls = []
    original = runner.run
    def observed(ids, batch, hooks=()):
        calls.append(ids.clone())
        return original(ids, batch, hooks)
    runner.run = observed
    runner.step(batch, 'last_mlp', 3, 'continuous')(ids)
    assert torch.equal(calls[0][:, :6], batch['source_input_ids'])
    assert torch.equal(calls[0][:, 6:], ids[:, 6:])
    assert torch.equal(calls[1], ids)


@pytest.mark.parametrize('explicit_scopes', [False, True])
def test_end_to_end_records_conditioning_and_resume(tiny, tmp_path, explicit_scopes):
    runner, batch = setup(tiny)
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': [14] + ([20, 21, 22, 23] if 'Paris' in text else
                                       [24, 25, 26, 27] if 'EUR' in text else [])}
        def decode(self, tokens, **kwargs):
            return str(tokens)
        def convert_ids_to_tokens(self, tokens):
            return list(map(str, tokens))
    runner.processor = SimpleNamespace(tokenizer=Tokenizer())
    runner.batch = lambda image, base, donor: batch
    Image.new('RGB', (4, 4)).save(tmp_path / 'flag.png')
    args = SimpleNamespace(sites=['last_residual', 'earlier_text'], block_spans=[], layers=[4],
                           modes=['prefill', 'continuous'], controls=['self', 'paraphrase'], max_new_tokens=4)
    if explicit_scopes:
        args.scopes = ['earlier_text', 'last_token', 'all_text']
        args.attention_spans = [2]
    row = dict(row_index=0, base='FR', source='FR', base_attribute='capital', donor_attribute='currency',
               base_label='Paris', source_label='EUR', template_id='controlled')
    results = Results(tmp_path / 'results', {})
    execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, results)
    assert len(results.records) == 2 + len(sweep_arms(args))
    for record in results.records[2:]:
        assert record['blocks'] in ([3], [2, 3])
        assert 1 not in record['prompt_positions']
    cells = json.loads((results.path / 'switch_summary.json').read_text())
    assert all(c['all']['country_count'] == 1 for c in cells)
    assert all(c['both_clean_correct'] is None for c in cells)
    stamp = (results.path / 'rows.jsonl').stat().st_mtime_ns
    execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, Results(results.path, {}))
    assert (results.path / 'rows.jsonl').stat().st_mtime_ns == stamp


def test_explicit_scope_sweep_deduplicates_and_bounds_windows():
    args = SimpleNamespace(sites=['earlier_text', 'last_residual', 'last_attention', 'last_joint'],
                           scopes=['earlier_text', 'last_token', 'all_text'],
                           block_spans=[1, 2, 4], attention_spans=[1, 2, 4], layers=[2],
                           modes=['prefill', 'continuous'], controls=['self'])
    arms = sweep_arms(args)
    # Five unique sites, five scope/mode combinations, two donor kinds.
    assert len(arms) == 50
    assert len({a[0] for a in arms}) == len(arms)
    assert all(a[3] == 'prefill' for a in arms if a[1].startswith('earlier_text/'))
    assert not any(':4' in a[1] for a in arms)


@pytest.mark.parametrize('scope', ['earlier_text', 'last_token', 'all_text'])
@pytest.mark.parametrize('site', ['residual', 'attention', 'mlp', 'joint', 'blocks:2', 'attention_blocks:2'])
def test_explicit_scopes_self_identity_and_position_boundaries(tiny, scope, site):
    runner, batch = setup(tiny)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23]])), dim=1)
    clean = runner.run(ids, batch)
    for mode in ['prefill', 'continuous']:
        positions = patch_positions(batch, scope, ids.shape[1], mode)
        assert 1 not in positions  # The actual image token is never patched.
        assert 0 not in positions and 2 not in positions  # Vision delimiters.
        assert (7 in positions) == (scope != 'earlier_text' and mode == 'continuous')
        patched, _ = runner.step(batch, f'{scope}/{site}', 3, mode, 'self')(ids)
        torch.testing.assert_close(patched, clean)


def test_all_text_residual_and_attention_window_match_direct_hooks(tiny):
    runner, batch = setup(tiny)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23]])), dim=1)
    donor_ids = torch.cat((batch['source_input_ids'], ids[:, 6:]), dim=1)
    # Replacing all ordinary text states before the final block supplies the
    # donor trajectory: the remaining image/special states precede the change.
    patched, _ = runner.step(batch, 'all_text/residual', 3, 'continuous')(ids)
    torch.testing.assert_close(patched, runner.run(donor_ids, batch))

    # Independent module-hook oracle: capture and replace only attention outputs
    # in two blocks, preserving image/special positions and all MLP computations.
    donor_values, handles = {}, []
    for block in [1, 2]:
        def capture(module, inputs, output, b=block):
            donor_values[b] = output[0].detach().clone()
        handles.append(runner.adapter.get_attn_block(runner.model, block).register_forward_hook(capture))
    try:
        runner.run(donor_ids, batch)
    finally:
        for handle in handles:
            handle.remove()
    handles = []
    for block in [1, 2]:
        def replace(module, inputs, output, b=block):
            value = output[0].clone()
            value[:, [3, 4, 5, 6, 7]] = donor_values[b][:, [3, 4, 5, 6, 7]]
            return (value,) + output[1:]
        handles.append(runner.adapter.get_attn_block(runner.model, block).register_forward_hook(replace))
    try:
        expected = runner.run(ids, batch)
    finally:
        for handle in handles:
            handle.remove()
    actual, _ = runner.step(batch, 'all_text/attention_blocks:2', 3, 'continuous')(ids)
    torch.testing.assert_close(actual, expected)
    for scope in ['earlier_text', 'last_token', 'all_text']:
        single, _ = runner.step(batch, f'{scope}/attention', 3, 'continuous')(ids)
        window, _ = runner.step(batch, f'{scope}/attention_blocks:1', 3, 'continuous')(ids)
        torch.testing.assert_close(single, window)
