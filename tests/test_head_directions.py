"""Offline subspace and contrastive intervention checks on real tiny Qwen."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_head_followups import tiny
from methods.head_subspace import HeadSubspace, MatchedBlend
from methods.head_attribute_mask import masked_logits, answer_loss
from methods.head_decode_trace import replay_step
from methods.head_directions import entity_setup, fit_mdas, item_split, parser, selected_heads
from methods.head_contrastive import direction, random_direction, additive_step, run_contrastive


def test_projection_geometry_and_matched_control():
    model = HeadSubspace([(1, 0), (1, 2)], 4, 2)
    z, donor = torch.randn(1, 4, 12), torch.randn(1, 2, 12)
    patched = model.patch(z, donor, 1, 2)
    columns = model.groups[1]
    u = model.basis(1)
    torch.testing.assert_close(u.T @ u, torch.eye(2))
    torch.testing.assert_close(patched[:, 2:, columns] @ u, donor[:, :, columns] @ u)
    orthogonal = torch.eye(8) - u@u.T
    torch.testing.assert_close(patched[:, 2:, columns] @ orthogonal, z[:, 2:, columns] @ orthogonal)
    torch.testing.assert_close(patched[:, :2], z[:, :2])
    torch.testing.assert_close(patched[:, :, 4:8], z[:, :, 4:8])
    blended = MatchedBlend(model).patch(z, donor, 1, 2)
    torch.testing.assert_close((blended-z).norm(dim=-1), (patched-z).norm(dim=-1))


def test_full_rank_identity_and_low_rank_gradients(tiny):
    runner, batch = tiny
    runner.model.requires_grad_(False)
    heads = [(1, 0), (2, 1)]
    source = runner.source(batch)
    ids = torch.cat((batch['base_input_ids'], torch.tensor([[20, 21, 22]])), dim=1)
    for mode in ['prefill', 'continuous']:
        full = HeadSubspace(heads, 8, 8, mode=mode)
        actual = masked_logits(runner, batch, source, ids, full)[:, -1]
        expected, _ = replay_step(runner, batch, source, heads, mode)(ids)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    low = HeadSubspace(heads, 8, 2)
    answer_loss(runner, batch, source, [20, 21, 22, 23], low, 1.).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in low.parameters())
    assert all(p.grad is None for p in runner.model.parameters())


def test_mdas_training_evaluation_resume(tiny, tmp_path):
    runner, batch = tiny
    runner.model.requires_grad_(False)
    runner.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda x, **kw: str(x),
                                                                  convert_ids_to_tokens=lambda x: list(map(str, x))))
    runner.batch = lambda row: (batch, {'base': [20, 21, 22, 23], 'source': [24, 25, 26, 27]})
    rows = [dict(row_index=i, pair_id=0, queried=q, base='A', source='B', template_id=q+'_v1')
            for i,q in enumerate(['capital', 'currency'])]
    args = SimpleNamespace(seed=0, mode='continuous', lr=.01, epochs=1, grad_clip=1.,
                           isolation_weight=1., max_new_tokens=4)
    heads = [(1, 0), (2, 1)]
    fit_mdas(runner, args, {'train': rows, 'test': rows}, heads, 'currency', 2, {}, tmp_path)
    saved = torch.load(tmp_path/'subspace.pt', weights_only=True)
    initial = HeadSubspace(heads, 8, 2).state_dict()
    assert saved['completed'] == 1
    assert any(not torch.equal(v, initial[k]) for k,v in saved['model'].items())
    cells = json.loads((tmp_path/'eval/cross_attribute_matrix.json').read_text())
    assert len(cells) == 12
    assert len(json.loads((tmp_path/'eval/joint_success.json').read_text())) == 6
    stamp = (tmp_path/'subspace.pt').stat().st_mtime_ns
    fit_mdas(runner, args, {'train': rows, 'test': rows}, heads, 'currency', 2, {}, tmp_path)
    assert (tmp_path/'subspace.pt').stat().st_mtime_ns == stamp


def test_item_identity_holdout_and_head_selection():
    args = SimpleNamespace(items=None, item_limit=None, seed=0, train_fraction=.6, train_items=None, test_items=None)
    train, test = item_split(args, dict.fromkeys('ABCDEFGH'))
    assert not set(train)&set(test)
    assert item_split(args, dict.fromkeys('ABCDEFGH')) == (train, test)
    args.train_items, args.test_items = ['A'], ['A']
    with pytest.raises(ValueError, match='disjoint'):
        item_split(args, dict.fromkeys('ABCDEFGH'))
    sets = {'x/top2': [(1,0),(1,1)], 'y/top2': [(1,1),(2,0)]}
    assert selected_heads(sets, ['x','y'], 'x', 2, 'intersection') == [(1,1)]
    assert len(selected_heads(sets, ['x','y'], 'x', 2, 'union')) == 3


def test_other_entity_and_explicit_traces(tmp_path):
    directory=tmp_path/'data/animals'
    directory.mkdir(parents=True)
    Image.new('RGB',(4,4)).save(directory/'animal.png')
    (directory/'ground_truth.json').write_text(json.dumps(dict(attributes=['habitat','class'],
        species={i:dict(image='animal.png',habitat='land',**{'class':'mammal'}) for i in ['cat','dog','cow','fox']})))
    paths=[]
    for attribute in ['habitat','class']:
        path=tmp_path/f'{attribute}.json'
        path.write_text(json.dumps(dict(entity='animals',attribute=attribute,patch_layer=1,positions='full_image',
            blocks=[1],head_dim=8,n_heads=4,phase1=[dict(block=1,head=0,delta_resid=1.)])))
        paths.append(str(path))
    args=parser().parse_args(['contrastive','--entity','animals','--head_ks','1','--traces',*paths,
                             '--vade_root',str(tmp_path)])
    _,_,attrs,_,trace,_,train,test=entity_setup(args,'animals')
    assert attrs==['habitat','class'] and trace['entity']=='animals'
    assert set(train)|set(test)=={'cat','dog','cow','fox'}
    assert not set(train)&set(test)


def test_contrastive_arithmetic_and_zero_control(tiny):
    heads = [(1,0)]
    means = {'a': {1: torch.arange(32).float()}, 'b': {1: torch.arange(32).float()+2}}
    v = direction(means, 'a', 'b', heads, 8)
    torch.testing.assert_close(v[1][:8], torch.full((8,), 2.))
    assert not v[1][8:].any()
    rev = direction(means, 'b', 'a', heads, 8)
    torch.testing.assert_close(rev[1], -v[1])
    rand = random_direction(v, heads, 8, 10)
    torch.testing.assert_close(rand[1].norm(), v[1].norm())
    assert not rand[1][8:].any()
    runner, batch = tiny
    for mode in ['prefill', 'continuous']:
        ids = torch.cat((batch['base_input_ids'], torch.tensor([[20,21]])), dim=1)
        actual,_ = additive_step(runner, batch, v, 0., mode)(ids)
        torch.testing.assert_close(actual, runner.forward(ids,batch)[0])
    assert not runner.adapter.get_attn_head_output_module(runner.model,1)._forward_pre_hooks


def test_contrastive_end_to_end_and_resume(tiny, tmp_path, monkeypatch):
    runner, original = tiny
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': [14]+([20,21,22,23] if 'City' in text else [24,25,26,27] if 'CCC' in text else [])}
        def decode(self,x,**kw): return str(x)
        def convert_ids_to_tokens(self,x): return list(map(str,x))
    runner.processor = SimpleNamespace(tokenizer=Tokenizer())
    calls=[]
    def fake_batch(runner,image,entity,attribute,template,prefill):
        calls.append((image.getpixel((0,0))[0], attribute, template))
        batch=dict(original)
        batch['base_input_ids']=original['base_input_ids'].clone()
        batch['base_input_ids'][0,4]=15 if attribute=='currency' else 13
        return batch
    monkeypatch.setattr('methods.head_followup_common.Runner', lambda args,trace: runner)
    monkeypatch.setattr('methods.head_contrastive.batch_for', fake_batch)
    items={}
    for c,n in [('A',1),('B',2)]:
        Image.new('RGB',(4,4),(n,0,0)).save(tmp_path/f'{c}.png')
        items[c]=dict(image=f'{c}.png',capital='City',currency='CCC')
    args=SimpleNamespace(seed=0,contrast_train_limit=1,contrast_test_limit=1,train_templates=['v1'],
        test_templates=['v5'],targets=['currency'],from_attributes=['capital'],dry_run=False,out_dir=str(tmp_path),
        head_ks=[1],head_selection='target',prefill='Answer:',strengths=[1.],mode='continuous',max_new_tokens=4)
    trace=dict(entity='flags',head_dim=8,n_heads=4)
    sets={'capital/top1':[(1,0)],'currency/top1':[(1,0)]}
    run_contrastive(args,trace,sets,['capital','currency'],tmp_path,items,['A'],['B'],{})
    saved=torch.load(tmp_path/'flags/contrastive/directions.pt',weights_only=True)
    assert saved['config']['actual_train_items']==['A']
    assert calls[:2]==[(1,'capital','v1'),(1,'currency','v1')]
    assert calls[2:]==[(2,'capital','v5'),(2,'currency','v5')]
    assert len(json.loads((tmp_path/'flags/contrastive/switch_summary.json').read_text()))==5
    calls.clear()
    run_contrastive(args,trace,sets,['capital','currency'],tmp_path,items,['A'],['B'],{})
    assert calls==[]
