"""Aggregate attribute_switch_sweep runs into the tables the findings rest on.

Reads `switch_summary.json` (plus `rows.jsonl` for the clean baselines) and
pools the per-(arm, base, donor) cells into per-(scope, site, layer) rates.

Two things this enforces that hand-aggregation gets wrong:

  * It pools `both_clean_correct` cells, weighted by each cell's own n. Pooling
    the 12 directed attribute pairs unweighted would let a pair with 6 eligible
    rows count as much as one with 8.
  * It reports DONOR_FIRST by default, not DONOR_FULL. A `last_token`
    intervention is structurally unable to steer an answer past its first token
    (the KV cache below it still belongs to the base -- see
    methods/ndm/swap_trace.py), and `text_continuous_cached_b2` measures exactly
    that: residual@L21+ reads 100% first / 69% full under `prefill` and
    100% / 100% under `continuous`. The `full` deficit is answer length, not a
    failed intervention, so `first` is the comparable number across sites.

LAYER vs BLOCK. `--layer L` in the sweep means the residual stream AFTER decoder
block L-1, and a sublayer site at L means block L-1's sublayer (common/sites.py's
convention). Tables here print BOTH so a head index is never off by one.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ARM = re.compile(r'^(?P<scope>[^/]+)/(?P<site>.+)/L(?P<layer>\d+)/(?P<mode>[^/]+)/(?P<control>[^/]+)$')
METRICS = {'first': 'donor_first', 'full': 'donor_full', 'base': 'original_full', 'neither': 'neither_full'}


def parse_arm(name):
    m = ARM.match(name)
    return m.groupdict() if m else None


def load_cells(run_dir, population='both_clean_correct'):
    """-> [(parsed arm, base_attribute, donor_attribute, scores)] for one run."""
    cells = []
    for c in json.loads((Path(run_dir) / 'switch_summary.json').read_text()):
        arm = parse_arm(c['arm'])
        if arm is None or not c[population]:
            continue          # 'clean'/'donor_clean' have no site; empty cells have no eligible rows
        cells.append((arm, c['base_attribute'], c['donor_attribute'], c[population]))
    return cells


def pool(cells, key, metrics=METRICS):
    """Weighted pooling over whatever `key(arm, base, donor)` groups together."""
    agg = defaultdict(lambda: defaultdict(float))
    for arm, base, donor, s in cells:
        k = key(arm, base, donor)
        if k is None:
            continue
        agg[k]['n'] += s['n']
        for short, field in metrics.items():
            agg[k][short] += s[field] * s['n']
    return {k: {'n': v['n'], **{m: v[m] / v['n'] for m in metrics}} for k, v in agg.items()}


def baselines(run_dir):
    """Clean accuracy and eligibility, read off rows.jsonl rather than the summary.

    Also reports how many of the configured rows were actually written: an
    interrupted sweep resumes at arm granularity, so a partial run looks
    complete in switch_summary.json except for the cells' n.
    """
    clean, donor, covered = {}, {}, set()
    for line in (Path(run_dir) / 'rows.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        covered.add(r['row_index'])
        if r['arm'] == 'clean':
            clean[r['row_index']] = r
        elif r['arm'] == 'donor_clean':
            donor[r['row_index']] = r
    configured = len(json.loads((Path(run_dir) / 'config.json').read_text())['rows'])
    per_attribute = defaultdict(lambda: [0, 0, 0])
    for i, c in clean.items():
        a = per_attribute[c['base_attribute']]
        a[0] += c['base_score']['full_match']
        a[1] += c['base_score']['first_token']
        a[2] += 1
    eligible = sum(1 for i in clean
                   if clean[i]['base_score']['full_match'] and donor[i]['source_score']['full_match'])
    return {'rows_configured': configured, 'rows_covered': len(covered), 'rows_with_baselines': len(clean),
            'eligible': eligible, 'clean_by_attribute': {k: {'full': v[0] / v[2], 'first': v[1] / v[2],
                                                             'n': v[2]} for k, v in per_attribute.items()}}


def crossover(cells, metric='first', scope_write='earlier_text', scope_read='last_token'):
    """The handoff table: where editing the QUESTION stops working and editing
    the READOUT starts. Their crossing is the read, and the attention column is
    the mechanism (attention is the only cross-position operation there is)."""
    write = pool([c for c in cells if c[0]['scope'] == scope_write and c[0]['control'] == 'switch'],
                 lambda a, b, d: (a['site'], int(a['layer'])))
    read = pool([c for c in cells if c[0]['scope'] == scope_read and c[0]['control'] == 'switch'],
                lambda a, b, d: (a['site'], int(a['layer'])))
    layers = sorted({L for _, L in read} | {L for _, L in write})
    out = []
    for L in layers:
        out.append({'layer': L, 'block': L - 1,
                    'question_residual': write.get(('residual', L), {}).get(metric),
                    'last_residual': read.get(('residual', L), {}).get(metric),
                    'last_attention': read.get(('attention', L), {}).get(metric),
                    'last_mlp': read.get(('mlp', L), {}).get(metric)})
    return out


def spans(cells, scope='last_token', family='attention_blocks', metric='first'):
    """Span table: site `family:N` ending at layer L covers blocks L-N..L-1.

    Reading it by COVERED BLOCK SET rather than by (L, N) is the point -- it is
    what separates "block b is needed" from "depth N is needed".
    """
    def key(a, b, d):
        if a['scope'] != scope or a['control'] != 'switch':
            return None
        site = a['site']
        single = {'attention_blocks': 'attention', 'blocks': 'joint'}[family]
        if site == single:
            width = 1
        elif site.startswith(family + ':'):
            width = int(site.split(':')[1])
        else:
            return None
        return int(a['layer']), width
    p = pool(cells, key)
    return [{'layer': L, 'width': w, 'blocks': list(range(L - w, L)), metric: v[metric], 'n': v['n']}
            for (L, w), v in sorted(p.items())]


def per_attribute(cells, scope='last_token', site='attention', metric='first', by='donor'):
    idx = {'donor': 2, 'base': 1}[by]
    p = pool(cells, lambda a, b, d: ((a['site'], int(a['layer']), (b, d)[idx - 1]))
             if a['scope'] == scope and a['control'] == 'switch' and a['site'] == site else None)
    return {(L, attr): v for (_, L, attr), v in p.items()}


def controls(cells):
    """`self` must be an exact no-op and `paraphrase` a near one. Both are
    scored against the DONOR attribute's label, so donor_* near zero and
    original_full near one is the pass condition."""
    return pool(cells, lambda a, b, d: a['control'])


def _fmt(v, width=7):
    return f'{"-":>{width}}' if v is None else f'{100 * v:{width}.1f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', nargs='+', help='results/attribute_switch/<run> directories')
    ap.add_argument('--metric', default='first', choices=list(METRICS))
    ap.add_argument('--population', default='both_clean_correct', choices=['both_clean_correct', 'all'])
    ap.add_argument('--json_out', default=None, help='Write every table to one JSON for later plotting')
    args = ap.parse_args(argv)

    everything = {}
    for run in args.run_dir:
        cells = load_cells(run, args.population)
        base = baselines(run)
        print(f'\n{"=" * 92}\n{run}\n{"=" * 92}')
        status = ('COMPLETE' if base['rows_covered'] == base['rows_configured']
                  else f'PARTIAL {base["rows_covered"]}/{base["rows_configured"]} rows')
        print(f'  {status}; {base["eligible"]}/{base["rows_with_baselines"]} rows eligible '
              f'(both clean answers correct)')
        print('  clean accuracy on the controlled prompts, by queried attribute:')
        for a, v in sorted(base['clean_by_attribute'].items()):
            print(f'    {a:13s} full={v["full"]:.3f}  first={v["first"]:.3f}  (n={v["n"]})')
        ctl = controls(cells)
        for name in ('self', 'paraphrase'):
            if name in ctl:
                c = ctl[name]
                print(f'  control {name:11s} donor_first={c["first"]:.4f} donor_full={c["full"]:.4f} '
                      f'base_kept={c["base"]:.4f}   (pass: donor ~0, base ~1)')

        cross = crossover(cells, args.metric)
        if any(r['last_residual'] is not None for r in cross):
            print(f'\n  -- the handoff ({METRICS[args.metric]}, %) --')
            print(f'  {"layer":>5} {"block":>5} | {"question resid":>14} {"last resid":>11} '
                  f'{"last attn":>10} {"last mlp":>9}')
            for r in cross:
                print(f'  {r["layer"]:>5} {r["block"]:>5} | {_fmt(r["question_residual"], 14)} '
                      f'{_fmt(r["last_residual"], 11)} {_fmt(r["last_attention"], 10)} '
                      f'{_fmt(r["last_mlp"], 9)}')

        sp = spans(cells, 'last_token', 'attention_blocks', args.metric)
        if sp:
            widths = sorted({r['width'] for r in sp})
            layers = sorted({r['layer'] for r in sp})
            lookup = {(r['layer'], r['width']): r[args.metric] for r in sp}
            print(f'\n  -- last-token attention spans ({METRICS[args.metric]}, %); '
                  f'a span at L of width N covers blocks L-N..L-1 --')
            print(f'  {"L":>3} {"blocks":>9} |' + ''.join(f'{f"w={w}":>8}' for w in widths))
            for L in layers:
                cov = f'{L - max(widths)}..{L - 1}'
                print(f'  {L:>3} {cov:>9} |' + ''.join(f'{_fmt(lookup.get((L, w)), 8)}' for w in widths))

        everything[run] = {'baselines': base, 'crossover': cross, 'attention_spans': sp,
                           'controls': {k: v for k, v in ctl.items()}}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(everything, indent=2, default=str) + '\n')
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
