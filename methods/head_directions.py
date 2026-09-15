"""Small-rank multi-task DAS and same-image contrastive direction experiments."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import ROOT, Results, Runner
from methods.head_attribute_edit import atomic_json, digest, prepare, report


def item_split(args, items):
    selected = sorted(args.items or items)
    if len(set(selected)) != len(selected) or not set(selected) <= set(items):
        raise ValueError('Item IDs must be unique and belong to the selected entity')
    if args.item_limit:
        selected = random.Random(args.seed).sample(selected, min(args.item_limit, len(selected)))
    if args.train_items is not None or args.test_items is not None:
        if args.train_items is None or args.test_items is None:
            raise ValueError('Provide both --train_items and --test_items')
        train, test = args.train_items, args.test_items
        if not set(train + test) <= set(selected):
            raise ValueError('Train/test IDs must be in the selected item pool')
    else:
        random.Random(args.seed).shuffle(selected)
        cut = int(len(selected) * args.train_fraction)
        train, test = selected[:cut], selected[cut:]
    if not train or not test or set(train) & set(test):
        raise ValueError('Need nonempty, disjoint training and test item identities')
    return sorted(set(train)), sorted(set(test))


def entity_setup(args, entity):
    directory = Path(args.vade_root) / 'data' / entity
    gt = json.loads((directory / 'ground_truth.json').read_text())
    key = {'flags': 'countries', 'brands': 'brands', 'animals': 'species'}.get(entity)
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != 'coverage')
    items = gt[key]
    attributes = args.attributes or gt['attributes']
    if len(attributes) < 2 or len(set(attributes)) != len(attributes) or not set(attributes) <= set(gt['attributes']):
        raise ValueError(f'{entity}: select at least two distinct attributes from {gt["attributes"]}')
    if not set(args.targets or attributes) <= set(attributes):
        raise ValueError('Targets must be among --attributes; retain other attributes for isolation')
    paths = []
    if args.traces:
        for p in args.traces:
            t = json.loads(Path(p).read_text())
            if t['entity'] == entity and t['attribute'] in attributes:
                paths.append(p)
    else:
        for a in attributes:
            found = sorted((Path(args.trace_root) / entity / 'ndm' / a).glob(f'*/{args.trace_name}'))
            if len(found) != 1:
                raise ValueError(f'{entity}/{a}: expected one saved ranking, found {len(found)}. '
                                 'Supply compatible --traces or adjust --trace_root/--trace_name; heads are never invented.')
            paths.append(str(found[0]))
    traces = [json.loads(Path(p).read_text()) for p in paths]
    if sorted(t['attribute'] for t in traces) != sorted(attributes):
        raise ValueError(f'Need exactly one trace for each selected attribute: {attributes}')
    first = traces[0]
    for t in traces:
        if any(t[k] != first[k] for k in ['entity', 'patch_layer', 'positions', 'blocks', 'head_dim', 'n_heads']):
            raise ValueError('Selected traces must share patch site and dimensions')
    heads = {}
    for t in traces:
        ranked = [(r['block'], r['head']) for r in sorted(t['phase1'], key=lambda r: -abs(r[t.get('rank_by', 'delta_resid')]))]
        if len(set(ranked)) != len(ranked) or any(b < t['patch_layer'] or not 0 <= h < t['n_heads'] for b, h in ranked):
            raise ValueError('Invalid downstream head ranking')
        for k in args.head_ks:
            if k > len(ranked):
                raise ValueError('Requested head count exceeds ranking')
            heads[f'{t["attribute"]}/top{k}'] = ranked[:k]
    train, test = item_split(args, items)
    for i in train + test:
        if not (directory / items[i]['image']).is_file():
            raise FileNotFoundError(directory / items[i]['image'])
    return directory, items, attributes, paths, first, heads, train, test


def selected_heads(sets, attributes, target, k, selection):
    groups = [sets[f'{a}/top{k}'] for a in attributes]
    if selection == 'target':
        return sets[f'{target}/top{k}']
    merged = set.union(*(set(map(tuple, g)) for g in groups)) if selection == 'union' else set.intersection(
        *(set(map(tuple, g)) for g in groups))
    if not merged:
        raise ValueError('Head-set intersection is empty')
    return sorted(merged)


def joint_report(results, target):
    report(results, target)
    groups = {}
    for r in results.records:
        key = (r['arm'], r['pair_id'], r['template_id'].rsplit('_', 1)[-1])
        groups.setdefault(key, []).append(r)
    total_attributes = len({r['queried'] for r in results.records})
    scores = {}
    for (arm, pair, template), rows in groups.items():
        if len({r['queried'] for r in rows}) != total_attributes:
            continue
        scores.setdefault(arm, []).append(all(r['source_score' if r['queried'] == target else 'base_score']['full_match']
                                              for r in rows))
    atomic_json(results.path / 'joint_success.json', {a: dict(n=len(v), successes=sum(v), rate=sum(v)/len(v))
                                                   for a, v in scores.items()})


def evaluate_mdas(runner, rows, results, target, model, random_model):
    import torch
    from methods.head_attribute_mask import masked_logits
    from methods.head_subspace import MatchedBlend
    from methods.head_decode_trace import replay_step
    edits = {'mdas': model, 'random_subspace': random_model, 'matched_blend': MatchedBlend(model)}
    for row in rows:
        arms = ['clean', 'image_patch', 'full_heads'] + list(edits)
        if all(results.has(row, a) for a in arms):
            continue
        batch, gold = runner.batch(row)
        if any(not g or len(g) > runner.args.max_new_tokens for g in gold.values()):
            raise ValueError('Increase max_new_tokens to cover full nonempty answers')
        source = runner.source(batch)
        for arm in arms:
            if results.has(row, arm):
                continue
            if arm in ['clean', 'image_patch']:
                def step(ids, image=arm == 'image_patch'):
                    return runner.forward(ids, batch, source if image else None)[0], {}
            elif arm == 'full_heads':
                step = replay_step(runner, batch, source, model.heads, model.mode)
            else:
                def step(ids, edit=edits[arm]):
                    with torch.no_grad():
                        return masked_logits(runner, batch, source, ids, edit)[:, -1], {}
            generated, _ = runner.generate(batch, step, gold)
            results.add(runner.record(row, arm, generated, gold, queried=row['queried'],
                                      pair_id=row['pair_id'], target_attribute=target))
    joint_report(results, target)


def fit_mdas(runner, args, rows, heads, target, rank, config, path):
    import torch
    from methods.head_subspace import HeadSubspace
    from methods.head_attribute_mask import answer_loss
    config = json.loads(json.dumps(dict(config, heads=heads, target=target, rank=rank)))
    results = Results(path / 'eval', config)
    results.check_runtime(runner)
    subspace = HeadSubspace(heads, runner.head_dim, rank, args.seed, args.mode).to(runner.model.device)
    random_subspace = HeadSubspace(heads, runner.head_dim, rank, args.seed + 10000, args.mode).to(runner.model.device)
    optimizer = torch.optim.Adam(subspace.parameters(), lr=args.lr)
    checkpoint = path / 'subspace.pt'
    completed, history = 0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=runner.model.device, weights_only=True)
        if saved['config'] != config:
            raise ValueError('Checkpoint configuration mismatch')
        subspace.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        completed, history = saved['completed'], saved['history']
    bundles = {}
    for r in rows['train']:
        bundles.setdefault(r['pair_id'], []).append(r)
    schedule = []
    for epoch in range(args.epochs):
        order = sorted(bundles)
        random.Random(args.seed + epoch).shuffle(order)
        schedule.extend(order)
    for index in range(completed, len(schedule)):
        bundle = bundles[schedule[index]]
        queries = sorted({r['queried'] for r in bundle})
        counts = {q: sum(r['queried'] == q for r in bundle) for q in queries}
        losses = dict.fromkeys(queries, 0.)
        optimizer.zero_grad()
        for row in bundle:
            batch, gold = runner.batch(row)
            query = row['queried']
            answer = gold['source' if query == target else 'base']
            if len(answer) > args.max_new_tokens:
                raise ValueError('Increase max_new_tokens to cover training answers')
            loss = answer_loss(runner, batch, runner.source(batch), answer, subspace, 1.)
            weight = (1. if query == target else args.isolation_weight / (len(queries)-1)) / counts[query]
            (weight * loss).backward()
            losses[query] += loss.detach().item() / counts[query]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in subspace.parameters()):
            raise FloatingPointError('Invalid subspace gradient')
        torch.nn.utils.clip_grad_norm_(subspace.parameters(), args.grad_clip)
        optimizer.step()
        history.append(dict(step=index+1, pair_id=schedule[index], answer_ce=losses))
        print(f'{target}/rank{rank}: {index+1}/{len(schedule)} {losses}', flush=True)
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(config=config, model=subspace.state_dict(), optimizer=optimizer.state_dict(),
                        completed=index+1, history=history), temp)
        temp.replace(checkpoint)
    atomic_json(path / 'training_history.json', history)
    atomic_json(path / 'subspace.json', dict(rank_per_layer=rank, heads=heads,
        layers=[dict(block=b, width=len(cols), rank=rank) for b, cols in subspace.groups.items()]))
    evaluate_mdas(runner, rows['test'], results, target, subspace, random_subspace)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment', choices=['mdas', 'contrastive'])
    ap.add_argument('--entities', '--entity', nargs='+', default=['flags'])
    ap.add_argument('--attributes', nargs='+')
    ap.add_argument('--targets', '--target', nargs='+')
    ap.add_argument('--items', nargs='+', help='Restrict individual item IDs; single entity only')
    ap.add_argument('--item_limit', type=int)
    ap.add_argument('--train_items', nargs='+')
    ap.add_argument('--test_items', nargs='+')
    ap.add_argument('--train_fraction', type=float, default=.7)
    ap.add_argument('--traces', nargs='+')
    ap.add_argument('--trace_root', default='logs/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--trace_name', default='head_trace_patch21_blocks21-23.json')
    ap.add_argument('--head_ks', nargs='+', type=int, default=[8])
    ap.add_argument('--head_selection', choices=['target', 'union', 'intersection'], default='target')
    ap.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--train_pairs', type=int, default=32)
    ap.add_argument('--test_pairs', type=int, default=16)
    ap.add_argument('--train_templates', nargs='+', default=['v1', 'v2', 'v3', 'v4'])
    ap.add_argument('--test_templates', nargs='+', default=['v5', 'v6'])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=.01)
    ap.add_argument('--isolation_weight', type=float, default=1.)
    ap.add_argument('--grad_clip', type=float, default=1.)
    ap.add_argument('--mode', choices=['prefill', 'continuous'], default='continuous')
    ap.add_argument('--contrast_train_limit', type=int, default=32)
    ap.add_argument('--contrast_test_limit', type=int, default=16)
    ap.add_argument('--from_attributes', nargs='+')
    ap.add_argument('--strengths', nargs='+', type=float, default=[.25, .5, 1., 2.])
    ap.add_argument('--prefill', default='Answer:')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_new_tokens', type=int, default=12)
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--vade_root', default=os.environ.get('VADE_ROOT', str(ROOT.parent / 'VADE')))
    ap.add_argument('--out_dir', default='results/head_directions')
    ap.add_argument('--dry_run', action='store_true')
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    positive = args.ranks + args.head_ks + [args.train_pairs, args.test_pairs, args.epochs, args.lr,
        args.grad_clip, args.max_new_tokens, args.contrast_train_limit, args.contrast_test_limit]
    if (any(not math.isfinite(v) or v <= 0 for v in positive) or not 0 < args.train_fraction < 1
            or not math.isfinite(args.isolation_weight) or args.isolation_weight < 0
            or any(not math.isfinite(v) for v in args.strengths)
            or (args.item_limit is not None and args.item_limit < 2)):
        ap.error('Invalid numeric parameter')
    if set(args.train_templates) & set(args.test_templates):
        ap.error('Prompt template splits must be disjoint')
    if len(args.entities) > 1 and any([args.items, args.train_items, args.test_items]):
        ap.error('Explicit item IDs require a single --entity; run one command per entity')
    for entity in args.entities:
        directory, items, attributes, paths, trace, sets, train_items, test_items = entity_setup(args, entity)
        if args.from_attributes and not set(args.from_attributes) <= set(attributes):
            ap.error('from_attributes must be selected attributes')
        config = {k: v for k, v in vars(args).items() if k not in ['dry_run', 'out_dir']}
        config.update(entity=entity, attributes=attributes, train_item_ids=train_items, test_item_ids=test_items,
            trace_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            metadata_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.json')},
            image_hashes={i: hashlib.sha256((directory/items[i]['image']).read_bytes()).hexdigest()
                          for i in train_items + test_items},
            implementation_hash=hashlib.sha256(b''.join((ROOT/'methods'/p).read_bytes() for p in
                ['head_directions.py', 'head_subspace.py', 'head_contrastive.py', 'head_attribute_edit.py',
                 'head_attribute_mask.py', 'head_followup_common.py', 'head_decode_trace.py',
                 'common/sites.py', 'common/hooks.py', 'common/entities.py', 'common/targets.py',
                 'common/position_sets.py', 'adapters/qwen2_5_vl.py'])).hexdigest())
        print(f'{entity}: {len(train_items)} train / {len(test_items)} held-out items, attributes={attributes}', flush=True)
        if args.experiment == 'contrastive':
            from methods.head_contrastive import run_contrastive
            run_contrastive(args, trace, sets, attributes, directory, items, train_items, test_items, config)
            continue
        local_args = argparse.Namespace(**vars(args))
        local_args.traces, local_args.train_items, local_args.test_items = paths, train_items, test_items
        _, _, _, rows, manifest = prepare(local_args)
        config['manifest'] = manifest
        for target in args.targets or attributes:
            for k in args.head_ks:
                heads = selected_heads(sets, attributes, target, k, args.head_selection)
                widths = [sum(b == block for b, h in heads)*trace['head_dim'] for block in {b for b,h in heads}]
                if max(args.ranks) > min(widths):
                    raise ValueError(f'Ranks exceed selected layer width {min(widths)}')
        if args.dry_run:
            continue
        runner = Runner(args, trace)
        runner.model.eval().requires_grad_(False)
        if runner.head_dim != trace['head_dim'] or runner.n_heads != trace['n_heads']:
            raise ValueError('Trace dimensions differ from loaded model')
        for target in args.targets or attributes:
            for k in args.head_ks:
                heads = selected_heads(sets, attributes, target, k, args.head_selection)
                for rank in args.ranks:
                    fit_mdas(runner, args, rows, heads, target, rank, config,
                        Path(args.out_dir)/entity/'mdas'/target/f'{args.head_selection}{k}'/f'rank{rank}')
        del runner


if __name__ == '__main__':
    main()
