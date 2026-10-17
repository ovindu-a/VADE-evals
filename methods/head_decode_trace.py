"""Test whether head-trace failures come from stopping substitution at prefill.

Uses a saved ranking and row sample. Compares clean/image baselines with
prefill-only and continuous head substitution, in both sufficiency and necessity
directions. Donors are recomputed on the recipient's freely generated prefix.
No gold tokens are supplied to the model. Full-prefix recomputation keeps all
earlier substitutions active and avoids reusing the prefill vector at later steps.

--heads B.H,B.H,... adds an explicit head set (e.g. exactly common10) as the
selection `custom<k>` next to the trace's own top-k, with a random<k> control of
the same size drawn the same way (reused when --head_ks already has k).
"""
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import Results, Runner, grouped_heads, parser, prepare
from methods.head_swap_vade import parse_heads


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


def build_selections(ranked, head_ks, random_repeats, seed, custom=None, patch_layer=None):
    """-> [(label, heads)]: the trace's top-k per k, an optional explicit `custom`
    set (label custom<len>), then the paired random controls, one size-k draw per
    (k, repeat) seeded by (seed, k, repeat) alone. The custom set's size joins the
    random sizes, so it gets a control drawn exactly like top-k's -- and when k is
    already a --head_ks size, that control IS the existing random<k>, not a copy."""
    ks = sorted({min(k, len(ranked)) for k in head_ks})
    selections = [(f'top{k}', ranked[:k]) for k in ks]
    sizes = list(ks)
    if custom:
        custom = sorted(custom)
        if len(set(custom)) != len(custom):
            raise ValueError(f'--heads repeats a head: {custom}')
        if patch_layer is not None and any(b < patch_layer for b, _ in custom):
            raise ValueError(f'--heads must sit at/after the image patch layer {patch_layer}: {custom}')
        if len(custom) > len(ranked):
            raise ValueError(f'--heads has {len(custom)} heads; the random control draws from the '
                             f'{len(ranked)}-head ranking')
        selections.append((f'custom{len(custom)}', custom))
        if len(custom) not in sizes:
            sizes.append(len(custom))
    # Paired random controls use the same sets for both modes/directions.
    for k in sizes:
        for repeat in range(random_repeats):
            rng = random.Random(seed + 1000 + 100 * k + repeat)
            selections.append((f'random{k}_r{repeat}', rng.sample(ranked, k)))
    return selections


def build_parser():
    ap = parser(__doc__)
    ap.add_argument('--head_ks', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--random_repeats', type=int, default=1)
    ap.add_argument('--skip_necessity', action='store_true')
    ap.add_argument('--heads', default=None, metavar='B.H,B.H,...',
                    help='Also run this explicit head set (selection custom<k>, e.g. common10) next to '
                         "the trace's top-k, with a same-size random control. Part of config.json.")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if any(k < 1 for k in args.head_ks) or args.random_repeats < 0:
        ap.error('head_ks must be positive; random_repeats must be nonnegative')
    custom = parse_heads(args.heads) if args.heads else None
    trace, ranked, rows, config, out_dir = prepare(args, 'decode')
    selections = build_selections(ranked, args.head_ks, args.random_repeats, config['effective_seed'],
                                  custom=custom, patch_layer=trace['patch_layer'])
    if custom:
        print(f'Explicit head set custom{len(custom)}: {custom}', flush=True)
    if args.dry_run:
        print('Dry run passed. No model loaded or output written.')
        return
    results = Results(out_dir, config)
    runner = Runner(args, trace)
    results.check_runtime(runner)
    if custom and any(b >= runner.n_layers or h >= runner.n_heads for b, h in custom):
        raise ValueError(f'--heads outside {runner.n_layers} blocks x {runner.n_heads} heads: {custom}')
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
