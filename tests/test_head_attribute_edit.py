"""CPU checks of selective head editing with real tiny Qwen multimodal forwards."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_head_followups import tiny
from methods.head_attribute_edit import evaluate, prepare, train
from methods.head_followup_common import Results
from methods.head_attribute_mask import HeadMask, answer_loss, masked_logits
from methods.head_decode_trace import replay_step


def test_mask_endpoints_and_full_answer_gradients(tiny):
    runner, batch = tiny
    runner.model.requires_grad_(False)
    source = runner.source(batch)
    heads = [(1, 0), (2, 1), (3, 2)]
    mask = HeadMask(heads, runner.head_dim)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[22, 23, 24]])), dim=1)
    with torch.no_grad():
        mask.logits.fill_(-100)
        actual = masked_logits(runner, batch, source, ids, mask, hard=True)[:, -1]
        clean, _ = runner.forward(ids, batch)
        torch.testing.assert_close(actual, clean)
        mask.logits.fill_(100)
        actual = masked_logits(runner, batch, source, ids, mask, hard=True)[:, -1]
        expected, _ = replay_step(runner, batch, source, heads, 'continuous')(ids)
        torch.testing.assert_close(actual, expected)
        mask.logits.zero_()
    # Full four-token answer, beyond the old three-token cap.
    loss = answer_loss(runner, batch, source, [22, 23, 24, 25], mask, 1.)
    loss.backward()
    assert mask.logits.grad is not None and mask.logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in runner.model.parameters())
    assert all(not runner.adapter.get_attn_head_output_module(runner.model, b)._forward_pre_hooks
               for b, h in heads)


def test_joint_training_evaluation_and_resume(tiny, tmp_path, monkeypatch):
    runner, batch = tiny
    runner.model.requires_grad_(False)
    runner.processor = SimpleNamespace(tokenizer=SimpleNamespace(
        decode=lambda tokens, **kwargs: ' '.join(map(str, tokens)),
        convert_ids_to_tokens=lambda tokens: list(map(str, tokens))))
    runner.batch = lambda row: (batch, {'base': [20, 21, 22, 23], 'source': [24, 25, 26, 27]})
    rows = [dict(row_index=i, pair_id=0, queried=q, base='A', source='B', template_id=q + '_v1')
            for i, q in enumerate(['language', 'capital'])]
    args = SimpleNamespace(epochs=1, lr=.05, isolation_weight=1., sparsity=.01,
                           temperature=1., max_new_tokens=4, seed=0)
    heads = [(1, 0), (2, 1)]
    seen = []
    def observed_loss(runner, batch, source, gold, mask, temperature):
        seen.append((gold, id(mask)))
        return answer_loss(runner, batch, source, gold, mask, temperature)
    monkeypatch.setattr('methods.head_attribute_mask.answer_loss', observed_loss)
    train(runner, args, {'train': rows, 'test': rows}, heads, 'language', {}, tmp_path)
    assert [g for g, _ in seen] == [[24, 25, 26, 27], [20, 21, 22, 23]]
    assert len({m for _, m in seen}) == 1
    checkpoint = torch.load(tmp_path / 'mask.pt', weights_only=True)
    assert checkpoint['completed'] == 1
    assert checkpoint['mask']['logits'].abs().sum() > 0
    assert set(checkpoint['history'][0]['answer_ce']) == {'language', 'capital'}
    matrix = json.loads((tmp_path / 'eval/cross_attribute_matrix.json').read_text())
    assert len(matrix) == 10
    assert {r['objective'] for r in matrix} == {'cause', 'isolation'}
    stamp = (tmp_path / 'mask.pt').stat().st_mtime_ns
    train(runner, args, {'train': rows, 'test': rows}, heads, 'language', {}, tmp_path)
    assert (tmp_path / 'mask.pt').stat().st_mtime_ns == stamp
    matrix_results = Results(tmp_path / 'matrix', {})
    evaluate(runner, rows, matrix_results, sets={'language/top2': heads, 'capital/top2': heads[::-1]})
    assert len(matrix_results.records) == 8
    assert all({r['queried'] for r in matrix_results.records if r['arm'] == arm} == {'language', 'capital'}
               for arm in ['language/top2', 'capital/top2'])
    args.lr = .1
    with pytest.raises(ValueError, match='different configuration'):
        train(runner, args, {'train': rows, 'test': rows}, heads, 'language', {}, tmp_path)


def test_paired_data_freezes_queries_and_excludes_reverse_leakage(tmp_path):
    entity = tmp_path / 'data/flags'
    entity.mkdir(parents=True)
    (entity / 'image.png').touch()
    (entity / 'ground_truth.json').write_text(json.dumps({'countries': {
        c: {'image': 'image.png'} for c in 'ABCD'}}))
    paths = []
    for attr in ['language', 'capital']:
        trace = dict(attribute=attr, entity='flags', patch_layer=1, positions='full_image',
                     blocks=[1], n_heads=2, head_dim=8,
                     phase1=[dict(block=1, head=h, delta_resid=2-h) for h in range(2)])
        path = tmp_path / (attr + '.json')
        path.write_text(json.dumps(trace))
        paths.append(str(path))
        directory = tmp_path / 'models/tiny/flags/tuples' / attr
        directory.mkdir(parents=True)
        for split, pairs, suffix in [('train', [('A', 'B'), ('C', 'D')], 'v1'),
                                     ('test', [('B', 'A')], 'v2')]:
            rows = [dict(base=b, source=s, queried=attr, rule='match_source', base_label=b,
                         source_label=s, template_id=attr + '_' + suffix) for b, s in pairs]
            (directory / (split + '.jsonl')).write_text('\n'.join(map(json.dumps, rows)))
    args = SimpleNamespace(traces=paths, head_ks=[1], seed=0, train_templates=['v1'],
                           test_templates=['v2'], vade_root=str(tmp_path), model_id='tiny',
                           train_pairs=1, test_pairs=1)
    _, attrs, sets, rows, config = prepare(args)
    assert {(r['base'], r['source']) for r in rows['train']} == {('C', 'D')}
    assert {(r['base'], r['source']) for r in rows['test']} == {('B', 'A')}
    assert {r['queried'] for r in rows['test']} == set(attrs)
    assert len(sets) == 3
    assert prepare(args)[-1] == config
    args.test_templates = ['v1']
    with pytest.raises(ValueError, match='disjoint'):
        prepare(args)
