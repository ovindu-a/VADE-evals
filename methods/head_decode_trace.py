"""Test whether head-trace failures come from stopping substitution at prefill.

Uses a saved ranking and row sample. Compares clean/image baselines with
prefill-only and continuous head substitution, in both sufficiency and necessity
directions. Donors are recomputed on the recipient's freely generated prefix.
No gold tokens are supplied to the model. Full-prefix recomputation keeps all
earlier substitutions active and avoids reusing the prefill vector at later steps.
"""
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import Results, Runner, grouped_heads, parser, prepare


def replay_step(runner, batch, source, heads, mode, necessity=False, verify=False):
    groups = grouped_heads(heads)

    def step(ids):
        import torch
        # Both forwards consume identical tokens. Only the donor's image intervention
        # differs; conditioning on its independent rollout would confound the comparison.
        donor_logits, donor = runner.forward(ids, batch, None if necessity else source,
                                              capture_blocks=list(groups))
        patches = {b: (donor[b], hs, mode) for b, hs in groups.items()}
        logits, _ = runner.forward(ids, batch, source if necessity else None, patches=patches)
        info = {'substituted_query_positions': 1 if mode == 'prefill' else
                ids.shape[1] - batch['base_input_ids'].shape[1] + 1}
        if verify:
            # All downstream heads, at every readout position, must reproduce the
            # image-patched logits. This tests content, position, and decode alignment.
            error = float((logits.float() - donor_logits.float()).abs().max())
            scale = max(float(donor_logits.float().abs().max()), 1e-8)
            info['all_downstream_logit_relative_error'] = error / scale
            tolerance = 1e-5 if logits.dtype == torch.float32 else 0.02
            if error / scale > tolerance:
                raise AssertionError(f'All-downstream replay identity failed: relative error {error / scale}')
        return logits, info
    return step


def main():
    ap = parser(__doc__)
    ap.add_argument('--head_ks', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--random_repeats', type=int, default=1)
    ap.add_argument('--skip_necessity', action='store_true')
    args = ap.parse_args()
    if any(k < 1 for k in args.head_ks) or args.random_repeats < 0:
        ap.error('head_ks must be positive; random_repeats must be nonnegative')
    trace, ranked, rows, config, out_dir = prepare(args, 'decode')
    if args.dry_run:
        print('Dry run passed. No model loaded or output written.')
        return
    results = Results(out_dir, config)
    runner = Runner(args, trace)
    results.check_runtime(runner)
    ks = sorted({min(k, len(ranked)) for k in args.head_ks})
    selections = [(f'top{k}', ranked[:k]) for k in ks]
    # Paired random controls use the same sets for both modes/directions.
    for k in ks:
        for repeat in range(args.random_repeats):
            rng = random.Random(config['effective_seed'] + 1000 + 100 * k + repeat)
            selections.append((f'random{k}_r{repeat}', rng.sample(ranked, k)))
    all_downstream = [(b, h) for b in range(trace['patch_layer'], runner.n_layers)
                      for h in range(runner.n_heads)]
    try:
        for ri, row in enumerate(rows):
            print(f'Row {ri + 1}/{len(rows)}', flush=True)
            expected = ['clean', 'image_patch', 'all_downstream_continuous'] + [
                f'{direction}_{label}_{mode}' for label, _ in selections
                for direction in (['sufficiency'] if args.skip_necessity else ['sufficiency', 'necessity'])
                for mode in ('prefill', 'continuous')]
            if all(results.has(row, arm) for arm in expected):
                continue
            batch, gold = runner.batch(row)
            source = runner.source(batch)
            for arm, src in [('clean', None), ('image_patch', source)]:
                if not results.has(row, arm):
                    gen, info = runner.generate(batch, lambda ids, s=src: (runner.forward(ids, batch, s)[0], {}), gold=gold)
                    results.add(runner.record(row, arm, gen, gold, token_scores=[i['gold_token_scores'] for i in info]))
            if not results.has(row, 'all_downstream_continuous'):
                gen, telemetry = runner.generate(batch, replay_step(
                    runner, batch, source, all_downstream, 'continuous', verify=True), gold=gold)
                reference = next(r for r in results.records
                                 if r['row_index'] == row['row_index'] and r['arm'] == 'image_patch')
                results.add(runner.record(row, 'all_downstream_continuous', gen, gold,
                                         heads=all_downstream, telemetry=telemetry,
                                         token_scores=[i['gold_token_scores'] for i in telemetry],
                                         identical_to_image_patch=gen == reference['generated_ids']))
            for label, heads in selections:
                for necessity in ([False] if args.skip_necessity else [False, True]):
                    for mode in ('prefill', 'continuous'):
                        arm = f'{"necessity" if necessity else "sufficiency"}_{label}_{mode}'
                        if results.has(row, arm):
                            continue
                        gen, telemetry = runner.generate(batch, replay_step(
                            runner, batch, source, heads, mode, necessity=necessity), gold=gold)
                        results.add(runner.record(row, arm, gen, gold, heads=heads, telemetry=telemetry,
                                                 token_scores=[i['gold_token_scores'] for i in telemetry]))
            results.finish()
    finally:
        results.finish()
    print(f'Wrote {out_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
