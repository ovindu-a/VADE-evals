"""Do the per-attribute DBM masks select the SAME units?

Reads the `mask_selected.json` that head_das.py writes for `--method dbm`
(one per attribute run) and reports, per block, how many units each attribute
kept and how much every pair of attributes overlaps, against the overlap two
independent random masks of those sizes would have (|A| |B| / d).

A mask can only isolate an attribute if the units it keeps are ones the
other attributes do not need, so this is the mask-level version of
readout_jacobian.py's cross-energy: overlap ~ chance says the attributes are
looked up by different neurons; overlap >> chance says the masks converged on
one shared "which country" set. Read it next to each run's VADE score -- a
mask that scores ~50% (the no-selectivity null) is not evidence either way.

Usage
-----
    python methods/mask_overlap.py results/head_das/flags/*_dbm_*_mlp23-26/mask_selected.json
"""
import argparse
import json
from itertools import combinations


def overlap_table(runs):
    """runs = {attribute: {"selected": {block: [idx]}, "width": {block: d}}}
    -> {block: {"n": {attr: k}, "d": d, "pairs": {"a|b": {...}}}}."""
    blocks = sorted({int(b) for r in runs.values() for b in r["selected"]})
    out = {}
    for b in blocks:
        sel = {a: set(r["selected"].get(str(b), r["selected"].get(b, []))) for a, r in runs.items()}
        d = next(int(r["width"].get(str(b), r["width"].get(b))) for r in runs.values())
        pairs = {}
        for x, y in combinations(sorted(sel), 2):
            inter, union = len(sel[x] & sel[y]), len(sel[x] | sel[y])
            chance = len(sel[x]) * len(sel[y]) / d
            pairs[f"{x}|{y}"] = {"intersection": inter, "jaccard": inter / union if union else float("nan"),
                                 "chance_intersection": chance,
                                 "ratio_to_chance": inter / chance if chance else float("nan")}
        out[b] = {"n": {a: len(s) for a, s in sel.items()}, "d": d, "pairs": pairs}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="mask_selected.json files, one per attribute")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()
    runs = {}
    for path in args.files:
        r = json.load(open(path))
        assert r["attribute"] not in runs, f"two runs for {r['attribute']} -- pass one file per attribute"
        runs[r["attribute"]] = r
    sites = {r["site"] for r in runs.values()}
    assert len(sites) == 1, f"mixing sites {sites}"
    table = overlap_table(runs)
    for b, t in table.items():
        print(f"\n=== block {b} (d={t['d']}) selected: "
              + ", ".join(f"{a} {k}" for a, k in t["n"].items()))
        for pair, p in t["pairs"].items():
            print(f"  {pair:>26}  shared {p['intersection']:6d}  jaccard {p['jaccard']:.3f}  "
                  f"x chance {p['ratio_to_chance']:.2f}")
    if args.out:
        json.dump(table, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
