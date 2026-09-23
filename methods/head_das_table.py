#!/usr/bin/env python3
"""One table over every head_das arm: cause, iso, final_score, SLACK, width.

`final_score` alone hides the result. Every arm at this site lies on a monotone
cause<->iso trade-off, so arms with very different behaviour (cause 46.6/iso 67.7
vs cause 73.2/iso 39.9) score within noise of each other. `slack` is the
informative column: how far an arm's iso sits ABOVE the straight line joining
the clean run to the full-swap ceiling, i.e. how much better it is than a coin
flip between "swap everything" and "swap nothing". A probabilistic full swap has
slack 0 by construction.

`width` is the arm's effective dimensionality -- summed `subspace_dim` for
das_fixed, summed `mask_sum` (the SOFT count, sum of sigmoid(m/T)) for the
masked methods -- so a rotated arm can be compared against a fixed arm that
spends the same number of dimensions. That comparison is the only one that
separates "the rotation helps" from "this arm just sits further along the same
curve".

  python methods/head_das_table.py                       # every arm, flags/language
  python methods/head_das_table.py --attribute currency
"""
import argparse, glob, json, math, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find(o, key):
    """First dict anywhere in a nested json carrying `key`."""
    if isinstance(o, dict):
        if key in o:
            return o
        for v in o.values():
            r = _find(v, key)
            if r:
                return r
    if isinstance(o, list):
        for v in o:
            r = _find(v, key)
            if r:
                return r
    return None


def _queried_rows(o):
    if isinstance(o, dict):
        if "queried" in o and "pct_accuracy" in o:
            return [o]
        return [r for v in o.values() for r in _queried_rows(v)]
    if isinstance(o, list):
        return [r for v in o for r in _queried_rows(v)]
    return []


def arm_width(run_dir):
    """Effective dims: subspace_dim (das_fixed) or mask_sum (masked methods)."""
    p = os.path.join(run_dir, "summary.json")
    if not os.path.exists(p):
        return float("nan")
    iv = _find(json.load(open(p)), "kind")
    if iv is None:
        return float("nan")
    blocks = _find(json.load(open(p)), "21") or {}
    tot = 0.0
    for b, e in blocks.items():
        if isinstance(e, dict):
            tot += e.get("mask_sum", e.get("subspace_dim", 0))
    return tot


def se_final(run_dir, design_effect):
    """Binomial SE on 1/2(cause + mean(iso)), inflated by `design_effect`.

    Rows cluster by item (VADE splits by item), so the plain binomial SE is a
    floor; pass 2.0 for a conservative read before calling two arms different."""
    rows = _queried_rows(json.load(open(os.path.join(run_dir, "predictions_summary.json"))))
    cause = [r for r in rows if r["is_cause"]]
    iso = [r for r in rows if not r["is_cause"]]
    if not cause or not iso:
        return float("nan")
    var = lambda r: (r["pct_accuracy"] / 100) * (1 - r["pct_accuracy"] / 100) / max(r["n"], 1)
    v = 0.25 * (var(cause[0]) + sum(var(r) for r in iso) / len(iso) ** 2)
    return 100 * math.sqrt(v * design_effect)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results", "head_das", "flags"))
    ap.add_argument("--attribute", default="language")
    ap.add_argument("--clean_iso", type=float, default=94.2,
                    help="iso of the unintervened run -- one end of the slack line.")
    ap.add_argument("--ceiling", nargs=2, type=float, default=(94.7, 1.7), metavar=("CAUSE", "ISO"),
                    help="cause/iso of the full-swap arm -- the other end.")
    ap.add_argument("--design_effect", type=float, default=2.0,
                    help="Multiplier on the binomial variance for item clustering.")
    args = ap.parse_args()

    c_cause, c_iso = args.ceiling
    arms, skipped = [], []
    for d in sorted(glob.glob(os.path.join(args.results_dir, f"{args.attribute}_das_*"))):
        p = os.path.join(d, "predictions_summary.json")
        if not os.path.exists(p):
            skipped.append(os.path.basename(d))
            continue
        ov = _find(json.load(open(p)), "final_score")
        cause, iso = ov["cause_accuracy"], ov["iso_mean_accuracy"]
        # iso a probabilistic full swap would reach at this cause
        line = args.clean_iso + (c_iso - args.clean_iso) * (cause / c_cause)
        arms.append((os.path.basename(d).replace(f"{args.attribute}_das_", ""),
                     cause, iso, ov["final_score"], iso - line, arm_width(d),
                     se_final(d, args.design_effect)))
    arms.sort(key=lambda a: -a[4])                       # by SLACK, not final_score

    print(f"{'arm':>32} {'cause':>7} {'iso':>7} {'final':>7} {'+/-':>5} {'SLACK':>7} {'width':>7}")
    for n, c, i, f, sl, w, se in arms:
        print(f"{n:>32} {c:6.1f}% {i:6.1f}% {f:6.1f}% {se:5.1f} {sl:+6.1f} {w:7.0f}")
    if arms:
        print(f"\n  slack = iso - [the line from clean (iso {args.clean_iso}) to ceiling "
              f"(cause {c_cause}, iso {c_iso})]. 0 = no better than a coin flip between them.")
        print(f"  +/- is 1 SE on final_score at design_effect {args.design_effect:g}; compare arms "
              f"against ~2x that, and prefer the SLACK column at MATCHED width.")
    if skipped:
        print(f"\n  no predictions (incomplete runs): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
