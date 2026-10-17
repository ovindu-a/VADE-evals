"""Tier-3 additions to the attribute-side scripts (FOLLOWUP_PLAN.md C6, C7, C8, C12).

All offline: a tiny RANDOM Qwen2.5-VL from test_head_followups and a fake VADE
root in tmp_path. Each new flag is checked where its answer is known exactly --
the reorder keeps this run's values, a direction from layer K is K's, a
continuous steer reaches decode steps, a gapped attention set covering a
contiguous span IS attention_blocks:N, VADE-template captures land on their
item-major grid rows -- and every default is checked to be unchanged.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test_head_followups import tiny
from test_attribute_switch_sweep import setup
from test_attribute_switch_cached import arm, spatial_setup
from methods import attr_capture, attr_head_trace, attr_steer, attribute_switch_cached
from methods import attribute_switch_sweep as sweep
from methods.attr_capture import capture_all_sites, run_order, vade_prompts
from methods.attr_directions import load as load_capture
from methods.attr_head_trace import order_from_trace, sample_rows
from methods.attr_steer import build_steer_patch, steer_hook
from methods.attribute_switch_cached import CachedPrefill
from methods.attribute_switch_sweep import (parse_attention_set, parts, patch_positions, sweep_arms,
                                          valid_scope)
from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
from methods.head_trace import generate_with_patches

ATTRS = ['capital', 'currency', 'language', 'calling_code']


def fake_vade(tmp_path, n_items=4, templates=None):
    """A VADE root with `n_items` flags (distinct solid-colour images) and optional templates."""
    entity = tmp_path / 'vade' / 'data' / 'flags'
    (entity / 'images').mkdir(parents=True)
    countries = {}
    for i in range(n_items):
        cid = f'C{i}'
        Image.new('RGB', (4, 4), (40 * i, 10, 200 - 30 * i)).save(entity / 'images' / f'{cid}.png')
        countries[cid] = {'image': f'images/{cid}.png', 'capital': f'Cap{i}', 'currency': f'CU{i}',
                          'language': f'Lang{i}', 'calling_code': str(100 + i)}
    (entity / 'ground_truth.json').write_text(json.dumps({'countries': countries}))
    if templates is not None:
        (entity / 'prompt_templates.json').write_text(json.dumps(templates))
    return tmp_path / 'vade', entity, countries


# ------------------------------------------------------------------ C6: attr_head_trace

def test_donor_attribute_filter_keeps_every_base_and_the_same_items(tmp_path):
    _, entity, items = fake_vade(tmp_path)
    chosen, every = sample_rows(entity, items, ATTRS, None, 3, 0)
    chosen_f, rows = sample_rows(entity, items, ATTRS, None, 3, 0, ['currency'])
    assert chosen_f == chosen
    assert len(rows) == len(chosen) * 3 and {r['donor_attribute'] for r in rows} == {'currency'}
    assert {r['base_attribute'] for r in rows} == {'capital', 'language', 'calling_code'}
    assert [r['row_index'] for r in rows] == list(range(len(rows)))
    assert [(r['base'], r['base_attribute']) for r in rows] == [
        (r['base'], r['base_attribute']) for r in every if r['donor_attribute'] == 'currency']
    # None (the default) is exactly the unfiltered call
    assert sample_rows(entity, items, ATTRS, None, 3, 0, None) == (chosen, every)


def _table(rng, blocks=(15, 16), n_heads=3):
    return [{'block': b, 'head': h, 'delta_z': float(rng.normal()), 'delta_resid': float(rng.normal()),
             'delta_dla': float(rng.normal())} for b in blocks for h in range(n_heads)]


def test_rank_from_takes_the_prior_order_and_keeps_this_runs_values():
    rng = np.random.default_rng(0)
    mine, prior = _table(rng), _table(rng)
    prior_trace = {'rank_by': 'delta_dla', 'phase1': prior}
    out = order_from_trace(mine, prior_trace)
    want = [(r['block'], r['head']) for r in sorted(prior, key=lambda r: -abs(r['delta_dla']))]
    assert [(r['block'], r['head']) for r in out] == want
    by_key = {(r['block'], r['head']): r for r in mine}
    assert all(r is by_key[(r['block'], r['head'])] for r in out)   # THIS run's records, unchanged
    assert [r['block'] for r in mine] == [15, 15, 15, 16, 16, 16]     # table itself not reordered
    # missing rank_by falls back to delta_resid, as attr_head_trace's own default
    out2 = order_from_trace(mine, {'phase1': prior})
    assert [(r['block'], r['head']) for r in out2] == [
        (r['block'], r['head']) for r in sorted(prior, key=lambda r: -abs(r['delta_resid']))]


def test_rank_from_refuses_a_different_head_table():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match='identical head table'):
        order_from_trace(_table(rng), {'rank_by': 'delta_resid', 'phase1': _table(rng, blocks=(15, 17))})
    with pytest.raises(ValueError, match='identical head table'):
        order_from_trace(_table(rng), {'rank_by': 'delta_resid', 'phase1': _table(rng)[:-1]})


def test_attr_head_trace_tag_and_validation(tmp_path, capsys):
    root, _, _ = fake_vade(tmp_path)
    base = ['--dry_run', '--vade_root', str(root), '--n_countries', '2', '--blocks', '15', '16']
    attr_head_trace.main(base)
    default = capsys.readouterr().out
    assert 'attr_head_trace_patch15_blocks15-16_all_text.json' in default and '_rankfrom' not in default
    prior = tmp_path / 'prior.json'
    prior.write_bytes(json.dumps({'experiment': 'attr_head_trace', 'blocks': [15, 16], 'rank_by': 'delta_resid',
                                  'phase1': _table(np.random.default_rng(2))}).encode())
    import hashlib
    sha = hashlib.sha256(prior.read_bytes()).hexdigest()
    attr_head_trace.main(base + ['--rank_from', str(prior), '--donor_attributes', 'capital'])
    out = capsys.readouterr().out
    assert f'all_text_donorcapital_rankfrom{sha[:8]}.json' in out
    assert '= 6 rows' in out                    # 2 items x 3 bases with donor capital
    with pytest.raises(SystemExit):            # traced blocks differ from the prior's
        attr_head_trace.main(base[:-2] + ['15', '16', '17', '--rank_from', str(prior)])
    with pytest.raises(SystemExit):            # donor not among --attributes
        attr_head_trace.main(base + ['--attributes', 'capital', 'currency', '--donor_attributes', 'language'])


# ------------------------------------------------------------------ C7: attr_steer

H = 32


def _rows(spec, bases=None):
    rows = [{'row_index': i, 'base': (bases or ['AA'] * len(spec))[i], 'base_attribute': b,
             'donor_attribute': d} for i, (b, d) in enumerate(spec)]
    return rows


def test_direction_layer_uses_means_at_k_but_hooks_layer_l():
    torch.manual_seed(0)
    means = {L: {a: torch.randn(H) for a in ATTRS[:2]} for L in (17, 22)}
    batch = {'rows': _rows([('capital', 'currency')]), 'steer_positions': torch.tensor([[1]])}
    (site, layer, fn), = build_steer_patch(means, batch, 22, ATTRS[:2], 'add', 1.0, 'cpu', direction_layer=17)
    assert layer == 22
    out = fn(torch.zeros(1, 3, H))
    torch.testing.assert_close(out[0, 1], means[17]['currency'] - means[17]['capital'])
    # default (None) and direction_layer == layer are the old arithmetic exactly
    for kw in ({}, {'direction_layer': 22}):
        for mode in ('add', 'project'):
            x = torch.randn(1, 3, H)
            old = build_steer_patch(means, batch, 22, ATTRS[:2], mode, 1.5, 'cpu')[0][2](x)
            new = build_steer_patch(means, batch, 22, ATTRS[:2], mode, 1.5, 'cpu', **kw)[0][2](x)
            assert torch.equal(old, new)


def test_continuous_steers_the_single_decode_column_and_default_does_not():
    d = torch.arange(H, dtype=torch.float32)
    fn = lambda x: x + d                       # noqa: E731
    pos = torch.tensor([[2], [2]])
    step = torch.zeros(2, 1, H)
    assert torch.equal(steer_hook(pos, fn)(step), step)                 # cache-aware no-op
    torch.testing.assert_close(steer_hook(pos, fn, True)(step), step + d)
    full = torch.zeros(2, 4, H)
    for cont in (False, True):                 # the prefill is the same either way
        out = steer_hook(pos, fn, cont)(full)
        torch.testing.assert_close(out[:, 2], torch.stack([d, d]))
        assert not out[:, [0, 1, 3]].any()


def test_continuous_reaches_decode_steps_in_real_generation(tiny):
    runner, batch = tiny
    adapter, model = runner.adapter, runner.model
    ids = batch['base_input_ids']
    seen = {False: [], True: []}
    for cont in (False, True):
        def fn(x, c=cont):
            seen[c].append(x.shape[1])
            return x
        patch = [(attr_steer.RESIDUAL_SITE, 2, steer_hook(torch.tensor([[ids.shape[1] - 1]]), fn, cont))]
        generate_with_patches(adapter, model, patch, ids, torch.ones_like(ids), batch['base_extra'], 0, 3)
    assert seen[False] == [1]                  # prefill only: fn sees the one steered column
    assert seen[True] == [1, 1, 1]             # prefill + both decode steps


def test_item_scope_uses_each_lanes_own_item_and_installs_its_coordinate():
    torch.manual_seed(4)
    means = {5: {a: torch.randn(H) for a in ATTRS[:2]}}
    item_means = {it: {a: torch.randn(H) for a in ATTRS[:2]} for it in ('AA', 'BB')}
    batch = {'rows': _rows([('capital', 'currency'), ('capital', 'currency')], ['AA', 'BB']),
             'steer_positions': torch.tensor([[0], [0]])}
    x = torch.randn(2, 2, H) * 5
    out = build_steer_patch(means, batch, 5, ATTRS[:2], 'project', 1.0, 'cpu', item_means=item_means)[0][2](x)
    for lane, it in enumerate(('AA', 'BB')):
        m = item_means[it]
        d = m['currency'] - m['capital']
        u = d / d.norm()
        target = (m['currency'] - (m['capital'] + m['currency']) / 2) @ u
        torch.testing.assert_close(out[lane, 0] @ u, target, rtol=1e-4, atol=1e-4)
    glob = build_steer_patch(means, batch, 5, ATTRS[:2], 'project', 1.0, 'cpu')[0][2](x)
    assert not torch.allclose(out, glob)


def test_attr_steer_flag_validation_and_tag(tmp_path, capsys):
    root, _, _ = fake_vade(tmp_path, n_items=3)
    base = ['--dry_run', '--vade_root', str(root), '--n_train_items', '2', '--n_test_items', '1',
            '--layers', '21', '22']
    attr_steer.main(base)
    assert 'attr_steer_last_token_L21-22_add-project_train2.json' in capsys.readouterr().out
    attr_steer.main(base + ['--mode', 'project', '--direction_layer', '17', '--continuous',
                            '--attn_impl', 'eager', '--direction_scope', 'item'])
    assert 'L21-22_project_train2_dir17_item_cont_eager.json' in capsys.readouterr().out
    attr_steer.main(base + ['--attn_impl', 'eager', '--image_trace', 'x.json'])   # eager is forced anyway
    assert 'train2.json' in capsys.readouterr().out
    for bad in (['--continuous', '--positions', 'question'],
                ['--direction_scope', 'item'],                      # mode both
                ['--direction_scope', 'item', '--mode', 'add'],
                ['--direction_layer', '0']):
        with pytest.raises(SystemExit):
            attr_steer.main(base + bad)


# ------------------------------------------------------------------ C8: attribute_switch_sweep

def test_scope_validator():
    for ok in ['earlier_text', 'last_token', 'all_text', 'field', 'tok:-1', 'tok:-12']:
        assert valid_scope(ok) == ok
    for bad in ['tok:-0', 'tok:1', 'tok:-', 'tok:-x', 'tok:-01', 'fields', 'last']:
        with pytest.raises(ValueError):
            valid_scope(bad)
    with pytest.raises(ValueError, match='Unknown position scope'):
        sweep.intervention('bogus/residual')


def test_field_and_tok_positions(tiny):
    runner, batch = setup(tiny)       # base [1,2,3,12,13,14]; donor differs at 4; paraphrase at 3
    p = batch['base_input_ids'].shape[1]
    assert patch_positions(batch, 'field', p + 3, 'continuous') == [4]
    assert patch_positions(batch, 'tok:-1', p + 3, 'continuous') == [p - 1]
    assert patch_positions(batch, 'tok:-1', p, 'prefill') == patch_positions(batch, 'last_token', p, 'prefill')
    assert patch_positions(batch, 'tok:-3', p, 'prefill') == [3]
    with pytest.raises(ValueError, match='outside'):
        patch_positions(batch, f'tok:-{p + 1}', p, 'prefill')
    two = dict(batch, source_input_ids=batch['source_input_ids'].clone())
    two['source_input_ids'][0, 3] = 16
    with pytest.raises(ValueError, match='exactly one differing'):
        patch_positions(two, 'field', p, 'prefill')


def test_field_and_tok_scopes_self_identity_and_field_embedding_swap_is_the_donor(tiny):
    runner, batch = setup(tiny)
    ids = batch['base_input_ids']
    clean = runner.run(ids, batch)
    for scope in ['field', 'tok:-1', 'tok:-2']:
        for site in ['residual', 'attention', 'mlp', 'joint', 'attention_set:1-2']:
            patched, _ = runner.step(batch, f'{scope}/{site}', 3, 'prefill', 'self')(ids)
            torch.testing.assert_close(patched, clean)
    # The field token is the ONLY difference between the prompts, so swapping its embedding
    # (residual layer 0) reproduces the donor run exactly.
    swapped, _ = runner.step(batch, 'field/residual', 0, 'prefill')(ids)
    torch.testing.assert_close(swapped, runner.run(batch['source_input_ids'], batch))
    # tok:-1 is last_token in prefill mode
    a, _ = runner.step(batch, 'tok:-1/residual', 3, 'prefill')(ids)
    b, _ = runner.step(batch, 'last_token/residual', 3, 'prefill')(ids)
    torch.testing.assert_close(a, b)


def test_attention_set_parsing():
    assert parse_attention_set('16-18.20-23') == 'attention_set:16-18.20-23'
    assert parse_attention_set('20-23.16-18') == 'attention_set:16-18.20-23'
    assert parse_attention_set('attention_set:19') == 'attention_set:19'
    assert parse_attention_set('16-17.18') == 'attention_set:16-18'
    for bad in ['3-1', '1-3.2', 'a-b', '', '1..2']:
        with pytest.raises(ValueError):
            parse_attention_set(bad)
    assert [(s.name, at) for s, at in parts('attention_set:0.2-3', 4)] == [
        ('attn_output', 1), ('attn_output', 3), ('attn_output', 4)]


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_contiguous_attention_set_is_exactly_attention_blocks(tiny, n):
    runner, batch = setup(tiny)
    L = 4
    span = f'{L - n}-{L - 1}' if n > 1 else f'{L - 1}'
    name = f'attention_set:{span}'
    assert [(s.name, at) for s, at in parts(name, L)] == \
        [(s.name, at) for s, at in parts(f'attention_blocks:{n}', L)]
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23]])), dim=1)
    for scope in ['last_token', 'all_text', 'field']:
        for mode in (['prefill'] if scope == 'field' else ['prefill', 'continuous']):
            got, _ = runner.step(batch, f'{scope}/{name}', L, mode)(ids)
            want, _ = runner.step(batch, f'{scope}/attention_blocks:{n}', L, mode)(ids)
            assert torch.equal(got, want)


def test_sweep_arms_new_scopes_are_prefill_only_and_sets_are_emitted_once():
    args = SimpleNamespace(sites=['last_residual'], block_spans=[], attention_spans=[2], layers=[2, 3, 4],
                           scopes=['field', 'tok:-2', 'last_token'], modes=['prefill', 'continuous'],
                           controls=['self'], attention_sets=['1-1.3'])
    arms = sweep_arms(args)
    assert len({a[0] for a in arms}) == len(arms)
    assert all(a[3] == 'prefill' for a in arms if a[1].split('/')[0] in ('field', 'tok:-2'))
    sets = [a for a in arms if 'attention_set' in a[1]]
    # one arm per (scope, mode, control), all labelled with the set's own end layer 4
    assert {a[2] for a in sets} == {4}
    assert len(sets) == 2 * 1 + 2 * 1 + 2 * 2      # field, tok:-2 prefill-only; last_token both modes
    assert sets[0][1] == 'field/attention_set:1.3'
    # attention_blocks:2 still skips layers it cannot end at; nothing raised parsing 'attention_set:...'
    assert not [a for a in arms if 'attention_blocks:2' in a[1] and a[2] < 2]


def test_sweep_arms_default_is_unchanged_without_new_options():
    args = SimpleNamespace(sites=sweep.SITES, block_spans=[2, 4, 8], attention_spans=[], layers=list(range(1, 29)),
                           modes=['prefill'], controls=['self', 'paraphrase'])
    legacy = sweep_arms(args)
    args.attention_sets = []
    assert sweep_arms(args) == legacy
    assert len(legacy) == 28 * 5 * 3 + 3 * (28 - 1 + 28 - 3 + 28 - 7)


@pytest.mark.parametrize('batch_size', [1, 3])
def test_cached_field_tok_and_attention_set_match_the_reference(tiny, batch_size):
    runner, batch = spatial_setup(tiny)      # donor differs at 7, paraphrase at 6 -> shared prefix 6
    arms = [arm(f'{scope}/{site}', 4, control)
            for scope in ['field', 'tok:-1', 'tok:-3', 'tok:-5']
            for site in ['residual', 'attention', 'joint', 'attention_set:1.3']
            for control in ['switch', 'self', 'paraphrase']]
    arms += [arm('last_token/attention_set:0.2-3', 4), arm('all_text/attention_set:2', 3)]
    cached = CachedPrefill(runner, batch, arms)
    assert cached.prefix_length == 6
    # tok:-5 is column 4 (inside the prefix: inert and dropped), field is column 7 (never dropped)
    assert cached.plan[arms[0][0]][1] == [7]
    assert cached.plan[arm('tok:-5/residual', 4)[0]][1] == []
    for start in range(0, len(arms), batch_size):
        cached.generate(arms[start:start + batch_size], verify=True)


def test_cached_field_column_inside_the_shared_prefix_is_refused(tiny, monkeypatch):
    runner, batch = spatial_setup(tiny)
    monkeypatch.setattr(attribute_switch_cached, 'shared_prefix_length', lambda b: 8)
    with pytest.raises(AssertionError, match='inside the shared prefix'):
        CachedPrefill(runner, batch, [arm('field/residual', 4)])
    CachedPrefill(runner, batch, [arm('tok:-3/residual', 4)])   # an inert tok column is fine


def test_end_to_end_with_field_scope_and_attention_set(tiny, tmp_path):
    runner, batch = setup(tiny)
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': [14] + ([20, 21] if 'Paris' in text else [24, 25] if 'EUR' in text else [])}
        def decode(self, tokens, **kwargs):
            return str(tokens)
        def convert_ids_to_tokens(self, tokens):
            return list(map(str, tokens))
    runner.processor = SimpleNamespace(tokenizer=Tokenizer())
    runner.batch = lambda image, base, donor: batch
    Image.new('RGB', (4, 4)).save(tmp_path / 'flag.png')
    from methods.head_followup_common import Results
    row = dict(row_index=0, base='FR', source='FR', base_attribute='capital', donor_attribute='currency',
               base_label='Paris', source_label='EUR', template_id='controlled')
    for execution in ['serial', 'cached']:
        args = SimpleNamespace(sites=['last_residual'], block_spans=[], layers=[2], scopes=['field', 'tok:-1'],
                               attention_sets=['1-2'], modes=['prefill', 'continuous'], controls=['self'],
                               max_new_tokens=3, execution=execution, batch_size=2, verify_cached=True)
        results = Results(tmp_path / execution, {})
        sweep.execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, results)
        recs = {r['arm']: r for r in results.records}
        assert len(recs) == 2 + len(sweep_arms(args))
        assert recs['field/residual/L2/prefill/switch']['prompt_positions'] == [4]
        assert recs['tok:-1/residual/L2/prefill/switch']['prompt_positions'] == [5]
        assert recs['field/attention_set:1-2/L3/prefill/switch']['blocks'] == [1, 2]
    with pytest.raises(ValueError, match='attention_set blocks'):
        args.attention_sets = ['3-4']
        sweep.execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path,
                      Results(tmp_path / 'bad', {}))


def test_switch_sweep_cli_accepts_new_scopes_and_sets(tmp_path, capsys):
    root, _, _ = fake_vade(tmp_path, n_items=2)
    base = ['--dry_run', '--vade_root', str(root), '--n_countries', '1', '--layers', '24']
    sweep.main(base + ['--scopes', 'field', 'tok:-2', 'last_token', '--sites', 'residual',
                       '--block_spans', '--attention_sets', '16-18.20-23', '20-23'])
    assert 'directed question pairs' in capsys.readouterr().out
    for bad in (['--scopes', 'tok:-0'], ['--attention_sets', '5-3']):
        with pytest.raises(SystemExit):
            sweep.main(base + bad)


# ------------------------------------------------------------------ C12: attr_capture

TEMPLATES = {
    'capital': [{'template_id': 'capital_v2', 'question': 'Name the capital, please now.', 'prefill': 'It is'},
                {'template_id': 'capital_v1', 'question': 'Capital?', 'prefill': 'The capital city is'}],
    'currency': [{'template_id': 'currency_v1', 'question': 'What is the currency code here?',
                  'prefill': 'The code is'}],
    'language': [{'template_id': 'language_v1', 'question': 'Language?', 'prefill': 'One language is'}],
    'calling_code': [{'template_id': 'calling_code_v1', 'question': 'Dialling code?',
                      'prefill': 'The code is +'}],
}


def test_vade_prompt_selection(tmp_path):
    _, entity, _ = fake_vade(tmp_path, templates=TEMPLATES)
    t0 = vade_prompts(entity, ATTRS, 0)
    assert t0['capital'] == ('capital_v1', 'Capital?', 'The capital city is')     # sorted ids, not file order
    assert vade_prompts(entity, ['capital'], 1)['capital'][0] == 'capital_v2'
    with pytest.raises(ValueError, match='out of range'):
        vade_prompts(entity, ATTRS, 1)                                            # currency has one


def test_run_order_controlled_is_contiguous_and_vade_is_attribute_major():
    chosen, A = ['a', 'b', 'c'], ATTRS
    ctrl = run_order(chosen, A, 'controlled', 5)
    assert ctrl == [list(range(s, min(s + 5, 12))) for s in range(0, 12, 5)]
    vade = run_order(chosen, A, 'vade', 2)
    assert sorted(r for c in vade for r in c) == list(range(12))
    for c in vade:
        assert len({r % len(A) for r in c}) == 1      # never spans two attributes
    assert vade[0] == [0, 4] and vade[1] == [8]


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [5 + ord(ch) % 50 for ch in text]}


class _FakeAdapter(Qwen25VLAdapter):
    """The real adapter's model accessors over the tiny model. Prompt length follows the
    PREFILL, so the controlled prompts (one prefill) share a length and the VADE templates do not."""
    def __init__(self, model):
        super().__init__('tiny-Qwen2.5-VL')
        self._model = model
        self.prompts_seen = []

    def load(self, device=None, dtype=None, attn_implementation=None):
        return self._model, SimpleNamespace(tokenizer=_Tok())

    def build_inputs(self, processor, image, question, prefill):
        self.prompts_seen.append((question, prefill))
        text = [12 + ord(ch) % 40 for ch in (question + '|' + prefill)][:6 + len(prefill) % 5]
        r, g, b = image.getpixel((0, 0))
        ids = torch.tensor([1, 2, 3] + text)
        return {'input_ids': ids,
                'extra': {'pixel_values': torch.full((4, 24), r / 255.) + torch.linspace(0, b / 255., 24),
                          'image_grid_thw': torch.tensor([[1, 2, 2]])}}


@pytest.mark.parametrize('mode', ['controlled', 'vade'])
def test_capture_writes_each_run_to_its_item_major_row(tiny, tmp_path, monkeypatch, mode):
    runner, _ = tiny
    model = runner.model
    root, entity, items = fake_vade(tmp_path, n_items=3, templates=TEMPLATES)
    fake = _FakeAdapter(model)
    monkeypatch.setattr(attr_capture, 'get_adapter', lambda model_id: fake)
    out = tmp_path / 'out'
    argv = ['--vade_root', str(root), '--n_items', '3', '--blocks', '1', '2', '--batch_size', '2',
            '--device', 'cpu', '--out_dir', str(out)]
    if mode == 'vade':
        argv += ['--prompts', 'vade', '--template_index', '0']
    attr_capture.main(argv)
    meta = json.loads((out / 'meta.json').read_text())
    index = [json.loads(line) for line in (out / 'index.jsonl').read_text().splitlines()]
    assert [r['row'] for r in index] == list(range(len(index)))
    assert [(r['item'], r['attribute']) for r in index] == [(i, a) for i in meta['items'] for a in ATTRS]
    prompts = (vade_prompts(entity, ATTRS, 0) if mode == 'vade' else
               {a: (None, sweep.question(a), sweep.PREFILL) for a in ATTRS})
    if mode == 'vade':
        assert meta['prompt_mode'] == 'vade' and meta['template_index'] == 0
        assert meta['prefill'] == {a: prompts[a][2] for a in ATTRS}
        assert len({r['prompt_length'] for r in index}) > 1     # the case the chunking exists for
    else:
        assert 'prompt_mode' not in meta and meta['prefill'] == sweep.PREFILL
    _, data = load_capture(out)
    acts = np.load(out / 'acts_residual.npy')
    for r in index:
        a = r['attribute']
        assert r['gold_ids'] == attr_capture.derive_gold_token_ids(_Tok(), prompts[a][2],
                                                                   items[r['item']][a])
        with Image.open(entity / items[r['item']]['image']) as img:
            built = fake.build_inputs(None, img.convert('RGB'), prompts[a][1], prompts[a][2])
        ids = built['input_ids'].unsqueeze(0)
        one = capture_all_sites(fake, model, [1, 2], torch.tensor([[ids.shape[1] - 1]]), ids,
                                torch.ones_like(ids), built['extra'])
        np.testing.assert_allclose(acts[r['row'], 1, 0], one['residual'][2][0, 0].numpy(), rtol=1e-4, atol=1e-5)
        i, j = meta['items'].index(r['item']), ATTRS.index(a)
        np.testing.assert_allclose(data['residual'][i, j, 0], acts[r['row'], 0, 0], rtol=1e-6)


def test_capture_cli_out_dir_and_validation(tmp_path, capsys):
    root, _, _ = fake_vade(tmp_path, n_items=2, templates=TEMPLATES)
    base = ['--dry_run', '--vade_root', str(root), '--n_items', '2']
    attr_capture.main(base)
    assert capsys.readouterr().out.split('-> ')[1].splitlines()[0].endswith('blocks15-22_n2')
    attr_capture.main(base + ['--prompts', 'vade', '--template_index', '0'])
    assert 'blocks15-22_n2_vade_t0' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        attr_capture.main(base + ['--template_index', '1'])
    with pytest.raises(ValueError, match='out of range'):
        attr_capture.main(base + ['--prompts', 'vade', '--template_index', '1'])
