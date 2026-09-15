"""Phase 1: cross-attribute head matrix. Phase 2: learn selective head masks.

Run as a module or script; --dry_run validates data without importing torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import ROOT, Results, Runner


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(args):
    traces = [json.loads(Path(p).read_text()) for p in args.traces]
    attributes = [t['attribute'] for t in traces]
    if len(set(attributes)) != len(attributes) or len(attributes) < 2:
        raise ValueError('Supply one trace per attribute, at least two')
    trace = traces[0]
    for t in traces:
        if any(t[k] != trace[k] for k in ('entity', 'patch_layer', 'positions', 'blocks', 'n_heads', 'head_dim')):
            raise ValueError('Traces must share entity, patch site, window and model dimensions')
    sets = {}
    for t in traces:
        ranked = [(r['block'], r['head']) for r in sorted(
            t['phase1'], key=lambda r: -abs(r[t.get('rank_by', 'delta_resid')]))]
        if (t['patch_layer'] < 1 or len(set(ranked)) != len(ranked)
                or any(b < t['patch_layer'] or b not in t['blocks'] or not 0 <= h < t['n_heads']
                       for b, h in ranked)):
            raise ValueError('Invalid or duplicate downstream heads')
        for k in args.head_ks:
            if k < 1 or k > len(ranked):
                raise ValueError('head_ks exceeds ranking size')
            sets[f'{t["attribute"]}/top{k}'] = ranked[:k]
    universe = [(b, h) for b in trace['blocks'] for h in range(trace['n_heads'])]
    for k in args.head_ks:
        sets[f'random/top{k}'] = random.Random(args.seed + k).sample(universe, k)
    if set(args.train_templates) & set(args.test_templates):
        raise ValueError('Train and test prompt template suffixes must be disjoint')
    root = Path(args.vade_root) / 'models' / args.model_id.split('/')[-1] / trace['entity'] / 'tuples'
    data, hashes = {}, {}
    for split, suffixes in [('train', args.train_templates), ('test', args.test_templates)]:
        by_attr = {}
        for attr in attributes:
            path = root / attr / f'{split}.jsonl'
            raw = path.read_bytes()
            hashes[f'{attr}/{split}'] = hashlib.sha256(raw).hexdigest()
            groups = {}
            for line in raw.splitlines():
                r = json.loads(line)
                allowed_items = getattr(args, f'{split}_items', None)
                if allowed_items is not None and (r['base'] not in allowed_items or r['source'] not in allowed_items):
                    continue
                if (r['queried'] == attr and r['rule'] == 'match_source'
                        and r['base_label'] != r['source_label']
                        and any(r['template_id'].endswith('_' + s) for s in suffixes)):
                    groups.setdefault((r['base'], r['source']), {})[r['template_id']] = r
            # Require every requested prompt for every attribute on each pair.
            by_attr[attr] = {p: rows for p, rows in groups.items() if all(
                any(t.endswith('_' + s) for t in rows) for s in suffixes)}
        pairs = set.intersection(*(set(d) for d in by_attr.values()))
        data[split] = (by_attr, pairs)
    # Exclude even reverse-direction leakage, before sampling either split.
    test_pairs = {tuple(sorted(p)) for p in data['test'][1]}
    data['train'][1].difference_update(p for p in list(data['train'][1]) if tuple(sorted(p)) in test_pairs)
    rows = {}
    for split, count in [('train', args.train_pairs), ('test', args.test_pairs)]:
        by_attr, pool = data[split]
        if count < 1 or len(pool) < count:
            raise ValueError(f'{split}: requested {count} pairs, only {len(pool)} eligible')
        chosen = random.Random(args.seed).sample(sorted(pool), count)
        selected = []
        for pair_id, pair in enumerate(chosen):
            for attr in attributes:
                for template, r in sorted(by_attr[attr][pair].items()):
                    selected.append(dict(r, row_index=len(selected), pair_id=pair_id))
        rows[split] = selected
    entity_dir = Path(args.vade_root) / 'data' / trace['entity']
    gt_path = entity_dir / 'ground_truth.json'
    gt = json.loads(gt_path.read_text())
    key = {'flags': 'countries', 'brands': 'brands', 'animals': 'species'}.get(trace['entity'])
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != 'coverage')
    for r in rows['train'] + rows['test']:
        for role in ('base', 'source'):
            if not (entity_dir / gt[key][r[role]]['image']).is_file():
                raise FileNotFoundError(entity_dir / gt[key][r[role]]['image'])
    # Freeze both phases to the exact same data, rankings and implementation.
    config = dict(schema_version=1, model_id=args.model_id, traces=traces, sets=sets, rows=rows,
                  tuple_hashes=hashes, seed=args.seed,
                  metadata_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in entity_dir.glob('*.json')},
                  implementation_hash=hashlib.sha256(b''.join((ROOT / 'methods' / f).read_bytes() for f in
                      ['head_attribute_edit.py', 'head_attribute_mask.py', 'head_followup_common.py',
                       'head_decode_trace.py'])).hexdigest())
    config = json.loads(json.dumps(config))
    print(f'Paired data: {len(rows["train"])} train / {len(rows["test"])} test questions; '
          f'{len(sets)} frozen head sets. Manifest {digest(config)[:12]}', flush=True)
    return trace, attributes, sets, rows, config


def report(results, target=None):
    """Separate source transfer from base preservation, including clean-correct isolation."""
    results.finish()
    clean = {r['row_index']: r for r in results.records if r['arm'] == 'clean'}
    cells = {}
    for r in results.records:
        cells.setdefault((r['arm'], r['queried']), []).append(r)
    output = []
    for (arm, query), rr in sorted(cells.items()):
        baseline_correct = [r for r in rr if clean.get(r['row_index'], {}).get('base_score', {}).get('full_match')]
        output.append(dict(arm=arm, queried=query, n=len(rr),
                           source_full_match=sum(r['source_score']['full_match'] for r in rr) / len(rr),
                           base_full_match=sum(r['base_score']['full_match'] for r in rr) / len(rr),
                           clean_correct_n=len(baseline_correct),
                           base_preservation_given_clean_correct=(sum(r['base_score']['full_match'] for r in
                               baseline_correct) / len(baseline_correct) if baseline_correct else None),
                           objective=('cause' if query == target else 'isolation') if target else None))
    atomic_json(results.path / 'cross_attribute_matrix.json', output)


def evaluate(runner, rows, results, sets=None, mask=None, temperature=1., target=None):
    from methods.head_decode_trace import replay_step
    for row in rows:
        arms = ['clean', 'image_patch'] + (list(sets) if sets is not None else ['full_heads', 'mask_soft', 'mask_hard'])
        if all(results.has(row, a) for a in arms):
            continue
        batch, gold = runner.batch(row)
        if any(len(g) > runner.args.max_new_tokens for g in gold.values()):
            raise ValueError('max_new_tokens is shorter than a full gold answer; increase the budget')
        source = runner.source(batch)
        for arm in arms:
            if results.has(row, arm):
                continue
            if arm in ('clean', 'image_patch'):
                def step(ids, image=arm == 'image_patch'):
                    logits, _ = runner.forward(ids, batch, source if image else None)
                    return logits, {}
            elif sets is not None:
                step = replay_step(runner, batch, source, sets[arm], 'continuous')
            elif arm == 'full_heads':
                step = replay_step(runner, batch, source, mask.heads, 'continuous')
            else:
                from methods.head_attribute_mask import masked_logits
                def step(ids, hard=arm == 'mask_hard'):
                    import torch
                    with torch.no_grad():
                        return masked_logits(runner, batch, source, ids, mask, temperature, hard)[:, -1], {}
            generated, _ = runner.generate(batch, step, gold)
            results.add(runner.record(row, arm, generated, gold, queried=row['queried'],
                                      pair_id=row['pair_id'], target_attribute=target))
    report(results, target)


def train(runner, args, rows, heads, target, config, directory):
    import torch
    from methods.head_attribute_mask import HeadMask, answer_loss
    directory.mkdir(parents=True, exist_ok=True)
    training = dict(manifest_hash=digest(config), target=target, heads=heads, epochs=args.epochs,
                    lr=args.lr, isolation_weight=args.isolation_weight, sparsity=args.sparsity,
                    temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    training = json.loads(json.dumps(training))
    results = Results(directory / 'eval', training)
    results.check_runtime(runner)
    mask = HeadMask(heads, runner.head_dim).to(runner.model.device)
    optimizer = torch.optim.Adam([mask.logits], lr=args.lr)
    checkpoint = directory / 'mask.pt'
    completed = 0
    history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=runner.model.device, weights_only=True)
        if saved['config'] != training:
            raise ValueError('Checkpoint configuration differs')
        mask.load_state_dict(saved['mask'])
        optimizer.load_state_dict(saved['optimizer'])
        completed, history = saved['completed'], saved['history']
    bundles = {}
    for row in rows['train']:
        bundles.setdefault(row['pair_id'], []).append(row)
    schedule = []
    for epoch in range(args.epochs):
        order = sorted(bundles)
        random.Random(args.seed + epoch).shuffle(order)
        schedule.extend((epoch, p) for p in order)
    for index in range(completed, len(schedule)):
        epoch, pair = schedule[index]
        bundle = bundles[pair]
        counts = {q: sum(r['queried'] == q for r in bundle) for q in {r['queried'] for r in bundle}}
        optimizer.zero_grad()
        losses = {q: 0. for q in counts}
        for row in bundle:
            batch, gold = runner.batch(row)
            query = row['queried']
            answer = gold['source' if query == target else 'base']
            if len(answer) > args.max_new_tokens:
                raise ValueError('Increase max_new_tokens to cover the full training answer')
            loss = answer_loss(runner, batch, runner.source(batch), answer, mask, args.temperature)
            weight = (1. if query == target else args.isolation_weight / (len(counts) - 1)) / counts[query]
            (weight * loss).backward()
            losses[query] += loss.detach().item() / counts[query]
        penalty = mask.gates(args.temperature).mean()
        (args.sparsity * penalty).backward()
        if not torch.isfinite(mask.logits.grad).all():
            raise FloatingPointError('Non-finite mask gradient')
        optimizer.step()
        history.append(dict(step=index + 1, epoch=epoch, pair_id=pair, answer_ce=losses,
                            mean_gate=penalty.detach().item()))
        print(f'{target}: step {index + 1}/{len(schedule)} CE={losses} gate={penalty.item():.4f}', flush=True)
        tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(config=training, mask=mask.state_dict(), optimizer=optimizer.state_dict(),
                        completed=index + 1, history=history), tmp)
        tmp.replace(checkpoint)
    gates = mask.gates(args.temperature).detach().cpu()
    atomic_json(directory / 'mask.json', dict(target=target, head_dim=runner.head_dim,
        active_coordinates=int((gates >= .5).sum()), total_coordinates=gates.numel(),
        per_head=[dict(block=b, head=h, soft_gate=gates[i].tolist(),
                       active_coordinates=int((gates[i] >= .5).sum())) for i, (b, h) in enumerate(heads)]))
    atomic_json(directory / 'training_history.json', history)
    evaluate(runner, rows['test'], results, mask=mask, temperature=args.temperature, target=target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['matrix', 'train'])
    ap.add_argument('--traces', nargs='+', required=True)
    ap.add_argument('--head_ks', nargs='+', type=int, default=[8, 16])
    ap.add_argument('--target', nargs='+', help='Phase 2 target attributes; default all trace attributes')
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--vade_root', default=os.environ.get('VADE_ROOT', str(ROOT.parent / 'VADE')))
    ap.add_argument('--out_dir', default='results/head_attribute_edit/flags')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_pairs', type=int, default=32)
    ap.add_argument('--test_pairs', type=int, default=16)
    ap.add_argument('--train_templates', nargs='+', default=['v1', 'v2', 'v3', 'v4'])
    ap.add_argument('--test_templates', nargs='+', default=['v5', 'v6'])
    ap.add_argument('--max_new_tokens', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=.05)
    ap.add_argument('--isolation_weight', type=float, default=1.)
    ap.add_argument('--sparsity', type=float, default=.01)
    ap.add_argument('--temperature', type=float, default=1.)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args(argv)
    if (args.epochs < 1 or args.lr <= 0 or args.temperature <= 0 or args.max_new_tokens < 1
            or args.isolation_weight < 0 or args.sparsity < 0):
        ap.error('Invalid training/decode hyperparameters')
    trace, attributes, sets, rows, config = prepare(args)
    if args.target and not set(args.target) <= set(attributes):
        ap.error('Targets must be attributes represented by the traces')
    if args.dry_run:
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Data/rankings/code changed; choose a fresh out_dir or restore the original configuration')
    atomic_json(manifest, config)
    runner = Runner(args, trace)
    if runner.n_heads != trace['n_heads'] or runner.head_dim != trace['head_dim']:
        raise ValueError('Saved trace dimensions do not match the loaded model')
    runner.model.requires_grad_(False)
    if args.phase == 'matrix':
        results = Results(out / 'matrix', dict(manifest_hash=digest(config), max_new_tokens=args.max_new_tokens))
        results.check_runtime(runner)
        evaluate(runner, rows['test'], results, sets=sets)
    else:
        for target in args.target or attributes:
            for k in args.head_ks:
                train(runner, args, rows, sets[f'{target}/top{k}'], target, config,
                      out / 'masks' / target / f'top{k}')


if __name__ == '__main__':
    main()
