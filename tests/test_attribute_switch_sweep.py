"""Same-image prompt interventions on a real, randomly initialized tiny VLM."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_head_followups import tiny
from methods.attribute_switch_sweep import SwitchRunner, alignment, execute, sweep_arms
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


def test_end_to_end_records_conditioning_and_resume(tiny, tmp_path):
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
    row = dict(row_index=0, base='FR', source='FR', base_attribute='capital', donor_attribute='currency',
               base_label='Paris', source_label='EUR', template_id='controlled')
    results = Results(tmp_path / 'results', {})
    execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, results)
    assert len(results.records) == 11
    cells = json.loads((results.path / 'switch_summary.json').read_text())
    assert all(c['all']['country_count'] == 1 for c in cells)
    assert all(c['both_clean_correct'] is None for c in cells)
    stamp = (results.path / 'rows.jsonl').stat().st_mtime_ns
    execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, Results(results.path, {}))
    assert (results.path / 'rows.jsonl').stat().st_mtime_ns == stamp
