"""Two separate phases: identify each head's token reads, then knock them out.

  python methods/head_token_trace.py identify --trace TRACE.json --out_dir IDENTIFY_DIR
  python methods/head_token_trace.py knockout --identification_dir IDENTIFY_DIR

Identification saves clean/image-patched attention maps and fixed per-head prompt
token rankings. Knockout consumes that artifact; it never re-ranks after removing
a token. Knocks out attention EDGES in the selected heads, not token embeddings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import Results, Runner, parser, prepare


def write_identification(directory, results, config, heads):
    records = [r for r in results.records if r['arm'] == 'image_patch']
    manifest = {'schema_version': 1, 'complete': len(records) == len(config['row_indices']),
                'trace_sha256': config['trace_sha256'], 'heads': heads,
                'ranking_context': 'image_patch', 'ranking_query': 'last_prompt_token',
                'token_rank_by': config['token_rank_by'], 'key_scope': config['key_scope'],
                'rows': [{'row_index': r['row_index'], 'tuple': r['tuple'], 'tokens': r['prompt_tokens'],
                          'heads': r['attention_steps'][0], 'reference_generated_ids': r['generated_ids']}
                         for r in records]}
    tmp = directory / 'identification.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    tmp.replace(directory / 'identification.json')
    lines = ['# Token identification', '',
             'Rankings are frozen from the image-patched last prompt token. Image coordinates are zero-based.',
             'Attention is softmax probability; contribution is the norm of A[q,k] V[k] W_O for this head.', '']
    for row in manifest['rows']:
        r = row['tuple']
        lines += [f'## Row {r["row_index"]}: {r["base"]} → {r["source"]} ({r["template_id"]})', '']
        for head, rec in row['heads'].items():
            lines += [f'### Head {head}', '', '| Position | Token / image patch | Group | Attention | Contribution |',
                      '|---|---|---|---:|---:|']
            for t in rec['top_tokens'][:8]:
                label = (f'image[{t["grid_row"]},{t["grid_col"]}]' if 'image_index' in t else repr(t['text']))
                label = label.replace('|', '\\|').replace('\n', ' ')
                lines.append(f'| {t["position"]} | {label} | {t["group"]} | {t["attention"]:.5f} '
                             f'| {t["weighted_value_residual_norm"]:.5f} |')
            lines.append('')
    (directory / 'tokens.md').write_text('\n'.join(lines) + '\n')


def identify(argv):
    ap = parser('Phase 1: identify tokens read by each selected head. No knockouts.')
    ap.add_argument('--head_k', type=int, default=8)
    ap.add_argument('--key_scope', choices=['all', 'image', 'object', 'background', 'text'], default='all')
    ap.add_argument('--token_rank_by', choices=['attention', 'contribution'], default='attention')
    args = ap.parse_args(argv)
    if args.head_k < 1:
        ap.error('head_k must be positive')
    trace, ranked, rows, config, out_dir = prepare(args, 'token_identify')
    heads = ranked[:args.head_k]
    if args.dry_run:
        print(f'Phase 1: identify {len(heads)} heads on {len(rows)} rows. No model loaded or output written.')
        return
    from methods.head_token_ablation import AttentionEdges, annotate_snapshot, candidate_positions, describe_tokens
    results = Results(out_dir, config)
    (out_dir / 'trace.json').write_bytes(Path(args.trace).read_bytes())
    runner = Runner(args, trace)
    results.check_runtime(runner)
    try:
        for ri, row in enumerate(rows):
            print(f'Identify row {ri + 1}/{len(rows)}', flush=True)
            if all(results.has(row, a) for a in ('clean', 'image_patch')):
                continue
            batch, gold = runner.batch(row)
            source = runner.source(batch)
            prompt_len = batch['base_input_ids'].shape[1]
            prompt_tokens = describe_tokens(runner, batch['base_input_ids'][0].tolist())
            candidates = candidate_positions(prompt_tokens, args.key_scope)
            if not candidates:
                raise ValueError(f'No candidate tokens in scope {args.key_scope}')
            for arm, src in [('clean', None), ('image_patch', source)]:
                if results.has(row, arm):
                    continue
                def step(ids, s=src):
                    obs = AttentionEdges(runner, heads, prompt_len, collect=True)
                    logits, _ = runner.forward(ids, batch, s, observer=obs)
                    tokens = describe_tokens(runner, ids[0].tolist())
                    for t in tokens[prompt_len:]:
                        t['group'] = 'answer_context'
                    annotated = annotate_snapshot(obs.snapshot, tokens, candidates, args.token_rank_by)
                    return logits, {'heads': annotated, 'reconstruction_error': obs.max_reconstruction_error}
                gen, info = runner.generate(batch, step, gold=gold)
                results.add(runner.record(row, arm, gen, gold, tuple=row, heads=heads,
                                         prompt_tokens=prompt_tokens,
                                         attention_steps=[i['heads'] for i in info],
                                         token_scores=[i['gold_token_scores'] for i in info],
                                         reconstruction_errors=[i['reconstruction_error'] for i in info]))
            results.finish()
            write_identification(out_dir, results, config, heads)
    finally:
        results.finish()
        write_identification(out_dir, results, config, heads)
    print(f'Phase 1 complete: inspect {out_dir / "tokens.md"}; phase 2 consumes identification.json.')


def knockout_arms(identified_row, ks, single_top_n, repeats, per_head_curves, seed):
    """Frozen rankings, paired random orders. Each joint top-K removes K keys PER HEAD."""
    rankings = {tuple(map(int, name.split('.'))): rec['ranked_prompt_keys']
                for name, rec in identified_row['heads'].items()}
    arms = [('no_knockout', {})]
    for head, keys in rankings.items():
        for i, key in enumerate(keys[:single_top_n]):
            arms.append((f'head{head[0]}.{head[1]}_single_rank{i + 1}', {head: [key]}))
    ks = sorted({min(k, min(map(len, rankings.values()))) for k in ks})
    for k in ks:
        arms.append((f'joint_top{k}_per_head', {h: keys[:k] for h, keys in rankings.items()}))
        if per_head_curves:
            for h, keys in rankings.items():
                arms.append((f'head{h[0]}.{h[1]}_top{k}', {h: keys[:k]}))
    for repeat in range(repeats):
        # One shuffle per head/repeat; all K use nested prefixes, not independent draws.
        orders = {}
        for h, keys in rankings.items():
            order = list(keys)
            rng = random.Random(f'{seed}:{identified_row["row_index"]}:{h}:{repeat}')
            rng.shuffle(order)
            orders[h] = order
        for k in ks:
            arms.append((f'joint_random{k}_per_head_r{repeat}', {h: keys[:k] for h, keys in orders.items()}))
            if per_head_curves:
                for h, keys in orders.items():
                    arms.append((f'head{h[0]}.{h[1]}_random{k}_r{repeat}', {h: keys[:k]}))
    return arms


def knockout(argv):
    ap = argparse.ArgumentParser(description='Phase 2: knock out frozen token rankings from phase 1.')
    ap.add_argument('--identification_dir', required=True)
    ap.add_argument('--query_scope', choices=['prefill', 'continuous'], default='prefill',
                    help='Ablate at last prompt query only, or at every answer readout query.')
    ap.add_argument('--knockout_ks', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--single_top_n', type=int, default=3, help='Test these top tokens individually for EACH head.')
    ap.add_argument('--per_head_curves', action='store_true', help='Also run cumulative/random curves for each head alone.')
    ap.add_argument('--random_repeats', type=int, default=1)
    ap.add_argument('--ablation_mode', choices=['zero', 'renormalize'], default='zero')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--vade_root', default=None, help='Override only if moving artifacts to another machine.')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--dry_run', action='store_true')
    opts = ap.parse_args(argv)
    if any(k < 1 for k in opts.knockout_ks) or opts.single_top_n < 0 or opts.random_repeats < 0:
        ap.error('K must be positive; single_top_n/random_repeats must be nonnegative')
    directory = Path(opts.identification_dir)
    original = json.loads((directory / 'config.json').read_text())
    raw = (directory / 'identification.json').read_bytes()
    identified = json.loads(raw)
    if not identified['complete']:
        raise ValueError('Phase 1 is incomplete; resume identification before running knockouts')
    common_keys = ['model_id', 'vade_root', 'split', 'allow_unpruned', 'sample_seed', 'limit', 'max_new_tokens']
    args = argparse.Namespace(**{k: original[k] for k in common_keys})
    args.__dict__.update(vars(opts))
    args.vade_root = opts.vade_root or original['vade_root']
    args.trace = str(directory / 'trace.json')
    args.identification_sha256 = hashlib.sha256(raw).hexdigest()
    trace, _, rows, config, out_dir = prepare(args, 'token_knockout')
    if config['trace_sha256'] != identified['trace_sha256'] or config['rows_sha256'] != original['rows_sha256']:
        raise ValueError('Phase 1 and knockout inputs differ; do not mix rankings across traces/tuples')
    by_row = {r['row_index']: r for r in identified['rows']}
    heads = [tuple(h) for h in identified['heads']]
    plans = {r['row_index']: knockout_arms(by_row[r['row_index']], args.knockout_ks, args.single_top_n,
                                         args.random_repeats, args.per_head_curves, config['effective_seed']) for r in rows}
    print(f'Phase 2: {sum(map(len, plans.values()))} row/arm generations; scope={args.query_scope}; '
          f'zeroing={args.ablation_mode}; no token re-ranking.', flush=True)
    if args.dry_run:
        return
    from methods.head_token_ablation import AttentionEdges, describe_tokens
    results = Results(out_dir, config)
    runner = Runner(args, trace)
    results.check_runtime(runner)
    if json.loads((directory / 'runtime.json').read_text()) != json.loads((out_dir / 'runtime.json').read_text()):
        raise ValueError('Model/runtime changed since identification; use the phase-1 runtime or re-identify tokens')
    try:
        for ri, row in enumerate(rows):
            print(f'Knockout row {ri + 1}/{len(rows)}', flush=True)
            if all(results.has(row, a) for a, _ in plans[row['row_index']]):
                continue
            batch, gold = runner.batch(row)
            # A different tokenizer/template must fail before applying saved absolute columns.
            prompt_tokens = describe_tokens(runner, batch['base_input_ids'][0].tolist())
            if prompt_tokens != by_row[row['row_index']]['tokens']:
                raise ValueError('Tokenization/geometry changed since identification; saved columns are invalid')
            source = runner.source(batch)
            for arm, removals in plans[row['row_index']]:
                if results.has(row, arm):
                    continue
                def step(ids, rm=removals):
                    obs = AttentionEdges(runner, heads, len(prompt_tokens), rm, args.query_scope,
                                         args.ablation_mode == 'renormalize')
                    logits, _ = runner.forward(ids, batch, source, observer=obs)
                    return logits, {'reconstruction_error': obs.max_reconstruction_error}
                gen, info = runner.generate(batch, step, gold=gold)
                if arm == 'no_knockout' and gen != by_row[row['row_index']]['reference_generated_ids']:
                    raise AssertionError('No-knockout generation differs from phase 1; check model/runtime before interpreting curves')
                results.add(runner.record(row, arm, gen, gold,
                                         removed_keys={f'{b}.{h}': keys for (b, h), keys in removals.items()},
                                         removed_tokens={f'{b}.{h}': [prompt_tokens[k] for k in keys]
                                                         for (b, h), keys in removals.items()},
                                         telemetry=info, token_scores=[i['gold_token_scores'] for i in info]))
            results.finish()
    finally:
        results.finish()
    print(f'Phase 2 complete: {out_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('identify', 'knockout'):
        raise SystemExit('Usage: python methods/head_token_trace.py {identify|knockout} --help')
    (identify if sys.argv[1] == 'identify' else knockout)(sys.argv[2:])
