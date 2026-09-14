"""Offline checks on a tiny RANDOM Qwen2.5-VL, including real image inputs/GQA.

No model weights, tokenizer download, CUDA, pyvene, or VADE data required.
"""
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
from methods.head_decode_trace import replay_step
from methods.head_followup_common import Results, Runner, replace_heads, score_tokens, summarize
from methods.head_token_ablation import AttentionEdges, remove_edges
from methods.head_token_trace import knockout_arms


@pytest.fixture
def tiny():
    torch.manual_seed(31)
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'mrope_section': [1, 1, 2]},
                         pad_token_id=0, bos_token_id=5, eos_token_id=None),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, fullatt_block_indexes=[0], window_size=4),
        image_token_id=2, video_token_id=4, vision_start_token_id=1, vision_end_token_id=3,
        pad_token_id=0, eos_token_id=None)
    config._attn_implementation = 'eager'
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = None
    # Prevent a random tiny model generating structural image markers into a text-only suffix.
    with torch.no_grad():
        model.lm_head.weight[:5].zero_()
    runner = Runner.__new__(Runner)
    runner.args = SimpleNamespace(max_new_tokens=4)
    runner.trace = {'patch_layer': 1}
    runner.adapter, runner.model = Qwen25VLAdapter('tiny-Qwen2.5-VL'), model
    runner.n_heads, runner.head_dim, runner.hidden, runner.n_layers = 4, 8, 32, 4
    ids = torch.tensor([[1, 2, 3, 12, 13, 14]])
    batch = {'base_input_ids': ids, 'source_input_ids': ids.clone(),
             'attention_mask': torch.ones_like(ids), 'positions': torch.tensor([[1]]),
             'base_extra': {'pixel_values': torch.randn(4, 24), 'image_grid_thw': torch.tensor([[1, 2, 2]])},
             'source_extra': {'pixel_values': torch.randn(4, 24), 'image_grid_thw': torch.tensor([[1, 2, 2]])}}
    return runner, batch


def test_later_positions_get_their_own_donor_values():
    z = torch.zeros(1, 5, 8)
    donor = torch.arange(24).reshape(1, 3, 8).float()
    prefill = replace_heads(z, donor, [1], 4, 2, 'prefill')
    continuous = replace_heads(z, donor, [1], 4, 2, 'continuous')
    assert torch.equal(prefill[:, 2, 4:], donor[:, 0, 4:])
    assert not prefill[:, 3:].any()
    assert torch.equal(continuous[:, 2:, 4:], donor[:, :, 4:])
    assert not continuous[:, :2].any() and not continuous[:, :, :4].any()
    with pytest.raises(ValueError, match='prefix length'):
        replace_heads(z, donor[:, :1], [1], 4, 2, 'continuous')


def test_complete_answer_metrics_do_not_truncate_at_three():
    s = score_tokens([1, 2, 3, 9], [1, 2, 3, 4])
    assert s['prefix3_match'] and not s['full_match']
    assert s['per_token'] == [True, True, True, False]
    assert not score_tokens([1], [1, 2])['full_match']


def test_all_downstream_replay_identity_at_every_decode_step(tiny):
    runner, batch = tiny
    source = runner.source(batch)
    heads = [(b, h) for b in range(1, 4) for h in range(4)]
    # Include a shared arbitrary answer prefix to test later positions even when free rollouts agree.
    ids = torch.cat([batch['base_input_ids'], torch.tensor([[20, 21]])], dim=1)
    direct, _ = runner.forward(ids, batch, source)
    replayed, info = replay_step(runner, batch, source, heads, 'continuous', verify=True)(ids)
    torch.testing.assert_close(direct, replayed, atol=1e-6, rtol=1e-5)
    assert info['substituted_query_positions'] == 3
    expected, _ = runner.generate(batch, lambda ids: (runner.forward(ids, batch, source)[0], {}))
    got, telemetry = runner.generate(batch, replay_step(runner, batch, source, heads, 'continuous', verify=True))
    assert got == expected and len(telemetry) == 4


def test_prefill_recomputation_matches_cached_generation(tiny):
    runner, batch = tiny
    source = runner.source(batch)
    heads = [(1, 0), (2, 2)]
    actual, _ = runner.generate(batch, replay_step(runner, batch, source, heads, 'prefill'))
    _, donor = runner.forward(batch['base_input_ids'], batch, source, capture_blocks=[1, 2])
    handles = []
    try:
        for b, h in heads:
            def pre(module, inputs, block=b, head=h):
                if inputs[0].shape[1] == 1:
                    return inputs
                return (replace_heads(inputs[0], donor[block], [head], 8, 5, 'prefill'),) + inputs[1:]
            handles.append(runner.adapter.get_attn_head_output_module(runner.model, b).register_forward_pre_hook(pre))
        with torch.no_grad():
            expected = runner.model.generate(input_ids=batch['base_input_ids'], attention_mask=batch['attention_mask'],
                                              **batch['base_extra'], do_sample=False, max_new_tokens=4,
                                              pad_token_id=0, eos_token_id=None)
        assert actual == expected[0, 6:].tolist()
    finally:
        for h in handles:
            h.remove()


@pytest.mark.parametrize('scope', ['prefill', 'continuous'])
@pytest.mark.parametrize('renormalize', [False, True])
def test_edge_removal_matches_actual_attention_weight_edit(tiny, monkeypatch, scope, renormalize):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf
    runner, batch = tiny
    ids = torch.cat([batch['base_input_ids'], torch.tensor([[20, 21]])], dim=1)
    heads = [(1, 1), (1, 3)]  # different GQA value heads
    removals = {(1, 1): [0, 1], (1, 3): [2, 4]}
    obs = AttentionEdges(runner, heads, 6, removals, scope, renormalize, collect=True)
    actual, _ = runner.forward(ids, batch, observer=obs)
    assert obs.max_reconstruction_error < 1e-5
    original = hf.eager_attention_forward

    def reference(module, q, k, v, attention_mask, **kwargs):
        output, weights = original(module, q, k, v, attention_mask, **kwargs)
        if getattr(module, 'layer_idx', None) != 1:
            return output, weights
        weights = weights.clone()
        queries = [5] if scope == 'prefill' else [5, 6, 7]
        for (_, h), keys in removals.items():
            _, changed = remove_edges(weights[0, h, queries], torch.eye(weights.shape[-1]), keys, renormalize)
            weights[0, h, queries] = changed
        expanded_v = v.repeat_interleave(2, dim=1)
        return (weights @ expanded_v).transpose(1, 2).contiguous(), weights

    monkeypatch.setattr(hf, 'eager_attention_forward', reference)
    expected, _ = runner.forward(ids, batch)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_observation_and_empty_knockout_are_identity(tiny):
    runner, batch = tiny
    baseline, _ = runner.forward(batch['base_input_ids'], batch)
    obs = AttentionEdges(runner, [(1, 0), (2, 3)], 6, collect=True)
    observed, _ = runner.forward(batch['base_input_ids'], batch, observer=obs)
    assert torch.equal(baseline, observed)
    assert set(obs.snapshot) == {'1.0', '2.3'}
    assert abs(sum(obs.snapshot['1.0']['attention']) - 1) < 1e-6
    assert not obs.handles


def test_knockout_hooks_clean_up_on_error(tiny):
    runner, batch = tiny
    obs = AttentionEdges(runner, [(1, 0)], 6, {(1, 0): [999]})
    with pytest.raises(ValueError, match='prompt-token'):
        runner.forward(batch['base_input_ids'], batch, observer=obs)
    assert not obs.handles
    module = runner.adapter.get_attn_block(runner.model, 1)
    # Transformers may keep its own output-capture hook after the first forward.
    assert all(h.__name__ != 'attn_hook' for h in module._forward_hooks.values())
    assert not module.v_proj._forward_hooks and not module.o_proj._forward_pre_hooks


def test_removing_all_keys_handles_zero_remaining_mass():
    weights = torch.tensor([[0.2, 0.8]])
    for normalize in (False, True):
        z, a = remove_edges(weights, torch.ones(2, 3), [0, 1], normalize)
        assert torch.isfinite(z).all() and not z.any() and not a.any()


def test_knockout_uses_frozen_per_head_rankings():
    row = {'row_index': 42, 'heads': {'21.1': {'ranked_prompt_keys': [7, 4, 3]},
                                     '22.5': {'ranked_prompt_keys': [3, 7, 4]}}}
    arms = dict(knockout_arms(row, [1, 2], 2, 1, True, 0))
    assert arms['joint_top1_per_head'] == {(21, 1): [7], (22, 5): [3]}
    assert arms['head21.1_single_rank2'] == {(21, 1): [4]}
    for h in [(21, 1), (22, 5)]:
        assert arms['joint_random1_per_head_r0'][h] == arms['joint_random2_per_head_r0'][h][:1]
    assert row['heads']['21.1']['ranked_prompt_keys'] == [7, 4, 3]


def test_results_resume_and_reject_mixed_config(tmp_path):
    rec = {'row_index': 1, 'arm': 'image_patch', 'source_score': score_tokens([1], [1]),
           'base_score': score_tokens([1], [2]), 'budget_shorter_than_gold': False, 'generated_text': 'x'}
    results = Results(tmp_path, {'seed': 0})
    results.add(rec)
    results.finish()
    with (tmp_path / 'rows.jsonl').open('a') as f:
        f.write('{"row_index":')
    resumed = Results(tmp_path, {'seed': 0})
    assert resumed.has({'row_index': 1}, 'image_patch')
    assert len(resumed.records) == 1
    assert summarize(resumed.records)['image_patch']['full_match'] == 1
    with pytest.raises(ValueError, match='different configuration'):
        Results(tmp_path, {'seed': 1})


def test_identification_to_knockout_handoff(tiny, tmp_path, monkeypatch):
    """Exercise both CLIs and their on-disk contract with actual tiny-model forwards."""
    from methods import head_token_trace as cli
    runner, batch = tiny
    root = tmp_path / 'VADE'
    entity = root / 'data' / 'flags'
    entity.mkdir(parents=True)
    for name in ('a.png', 'b.png'):
        (entity / name).write_bytes(b'placeholder; runner fixture supplies pixel tensors')
    (entity / 'ground_truth.json').write_text(json.dumps({'countries': {
        'A': {'image': 'a.png'}, 'B': {'image': 'b.png'}}}))
    tuples = root / 'models' / 'Qwen2.5-VL-7B-Instruct' / 'flags' / 'tuples' / 'language'
    tuples.mkdir(parents=True)
    row = {'row_index': 7, 'base': 'A', 'source': 'B', 'rule': 'match_source',
           'queried': 'language', 'template_id': 'test', 'base_label': 'a', 'source_label': 'b'}
    (tuples / 'test.jsonl').write_text(json.dumps(row) + '\n')
    trace = {'entity': 'flags', 'attribute': 'language', 'patch_layer': 1, 'positions': 'full_image',
             'blocks': [1, 2], 'seed': 0, 'n_rows': 1, 'rank_by': 'delta_resid',
             'phase1': [{'block': 1, 'head': 0, 'delta_resid': 2.}, {'block': 2, 'head': 1, 'delta_resid': 1.}]}
    trace_path = tmp_path / 'trace.json'
    trace_path.write_text(json.dumps(trace))
    gold = {'source': [20, 21], 'base': [22]}
    runner.batch = lambda _: (batch, gold)
    runner.processor = SimpleNamespace(tokenizer=SimpleNamespace(
        all_special_ids=[0, 1, 2, 3, 4],
        convert_ids_to_tokens=lambda x: [str(t) for t in x] if isinstance(x, list) else str(x),
        decode=lambda ids, **kwargs: ' '.join(map(str, ids))))
    runner.assets = SimpleNamespace(object_location={
        'vlm_token_grid': {'n_tokens': 1, 'grid_rows': 1, 'grid_cols': 1},
        'object_token_indices': {'object_only': {'flat': [0]}}})

    def factory(args, trace):
        runner.args, runner.trace = args, trace
        return runner
    monkeypatch.setattr(cli, 'Runner', factory)
    identified = tmp_path / 'identify'
    cli.identify(['--trace', str(trace_path), '--vade_root', str(root), '--device', 'cpu',
                  '--head_k', '2', '--max_new_tokens', '2', '--out_dir', str(identified)])
    manifest = json.loads((identified / 'identification.json').read_text())
    assert manifest['complete'] and set(manifest['rows'][0]['heads']) == {'1.0', '2.1'}
    assert manifest['rows'][0]['tokens'][1]['group'] == 'object'
    assert (identified / 'tokens.md').is_file()
    out = tmp_path / 'knockout'
    cli.knockout(['--identification_dir', str(identified), '--device', 'cpu', '--knockout_ks', '1',
                  '--single_top_n', '1', '--query_scope', 'continuous', '--out_dir', str(out)])
    saved = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()]
    assert len(saved) == 5
    assert all(r['token_scores'] for r in saved)
    assert set(json.loads((out / 'summary.json').read_text())) == {r['arm'] for r in saved}
    # Re-running phase 2 resumes without appending duplicates.
    cli.knockout(['--identification_dir', str(identified), '--device', 'cpu', '--knockout_ks', '1',
                  '--single_top_n', '1', '--query_scope', 'continuous', '--out_dir', str(out)])
    assert len((out / 'rows.jsonl').read_text().splitlines()) == 5
