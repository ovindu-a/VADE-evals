#!/usr/bin/env python3
"""Where does an activation's variance live: the ENTITY, the QUESTION, or their binding?

Reads `attr_capture.py`'s crossed grid. No model, no GPU, seconds to run.

The grid is balanced -- exactly one row per (item, attribute) cell -- and the
activations are deterministic, one forward pass per cell with no replication. So
the two-way decomposition

    SS_total = SS_item + SS_attr + SS_interaction

is EXACT and the residual term IS the interaction, not noise. That matters: in a
normal ANOVA the residual is error and the interaction needs replication to
separate from it. Here there is no error term to confuse it with.

The interaction is the quantity VADE is about. `item` alone is "which country",
`attribute` alone is "which question", and only `item x attribute` is *this
country's this attribute* -- the only thing a selective intervention could edit.
An intervention at a site where the interaction is ~3% of the variance has almost
nothing of the target quantity in its input, whatever it is trained to do.

  python methods/attr_variance.py                                  # R13.1
  python methods/attr_variance.py --heads 21.1,21.5,22.13          # a head subset
  python methods/attr_variance.py --per_head --blocks 21 22 23     # rank heads
"""
import argparse, json, os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAPTURE = os.path.join(REPO_ROOT, "results", "attr_capture", "flags",
                               "blocks15-27_n84")
COMMON10 = "21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"


def load_grid(capture_dir, site):
    """-> ([n_items, n_attrs, n_blocks, hidden] float32, meta). Item-major."""
    meta = json.load(open(os.path.join(capture_dir, "meta.json")))
    rows = [json.loads(l) for l in open(os.path.join(capture_dir, "index.jsonl"))]
    items, attrs = meta["items"], meta["attributes"]
    I = {x: i for i, x in enumerate(items)}
    A = {x: i for i, x in enumerate(attrs)}
    acts = np.load(os.path.join(capture_dir, f"acts_{site}.npy"), mmap_mode="r")
    hidden = acts.shape[-1]
    G = np.zeros((len(items), len(attrs), len(meta["blocks"]), hidden), dtype=np.float32)
    seen = np.zeros((len(items), len(attrs)), dtype=bool)
    for r in rows:
        if r.get("condition", "clean") != "clean":
            continue
        G[I[r["item"]], A[r["attribute"]]] = acts[r["row"], :, 0, :]
        seen[I[r["item"]], A[r["attribute"]]] = True
    assert seen.all(), (f"grid is not balanced -- {int((~seen).sum())} empty (item, attribute) "
                        f"cells. The exact decomposition below assumes one row per cell.")
    return G, meta


def decompose(X):
    """[n_item, n_attr, d] -> (item %, attr %, interaction %). Sums to 100."""
    g = X.mean((0, 1), keepdims=True)
    mi, ma = X.mean(1, keepdims=True), X.mean(0, keepdims=True)
    ss = lambda Z: float((Z ** 2).sum())
    tot = ss(X - g)
    if tot == 0:
        return float("nan"), float("nan"), float("nan")
    return (ss(mi - g) * X.shape[1] / tot * 100,
            ss(ma - g) * X.shape[0] / tot * 100,
            ss(X - mi - ma + g) / tot * 100)


def head_cols(heads, block, head_dim):
    hs = sorted(h for b, h in heads if b == block)
    return np.concatenate([np.arange(h * head_dim, (h + 1) * head_dim) for h in hs]) if hs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", default=DEFAULT_CAPTURE)
    ap.add_argument("--site", default="residual",
                    choices=["residual", "attn_output", "mlp_output", "attn_head_output"])
    ap.add_argument("--heads", default=None, metavar="B.H,B.H",
                    help="Restrict to these heads' columns (attn_head_output only). "
                         f"'common10' expands to {COMMON10}.")
    ap.add_argument("--blocks", nargs="+", type=int, default=None)
    ap.add_argument("--per_head", action="store_true",
                    help="One row per (block, head), ranked by interaction mass actually landing "
                         "in the residual -- interaction SS scaled by ||W_O,h||^2, since a head "
                         "whose projection writes nothing cannot matter however structured it is.")
    args = ap.parse_args()

    G, meta = load_grid(args.capture_dir, args.site)
    blocks = meta["blocks"]
    head_dim = meta.get("head_dim", 128)
    want = args.blocks or blocks
    heads = None
    if args.heads:
        spec = COMMON10 if args.heads == "common10" else args.heads
        heads = sorted(tuple(int(x) for x in h.split(".")) for h in spec.replace(" ", ",").split(",") if h)
        assert args.site == "attn_head_output", "--heads only means anything for attn_head_output"

    print(f"{args.site} @ {meta['positions']} -- {len(meta['items'])} items x "
          f"{len(meta['attributes'])} attributes, balanced")
    if args.per_head:
        norms = meta.get("head_proj_norms", {})
        rows = []
        for bi, b in enumerate(blocks):
            if b not in want:
                continue
            for h in range(meta.get("n_heads", 28)):
                X = G[:, :, bi, h * head_dim:(h + 1) * head_dim]
                it, at, inter = decompose(X)
                w = float(norms.get(str(b), [1.0] * 28)[h]) ** 2 if norms else 1.0
                rows.append((b, h, it, at, inter, inter * float((X - X.mean((0, 1))) ** 2).sum() * w))
        rows.sort(key=lambda r: -r[5])
        print(f"\n{'head':>8} {'item':>8} {'attr':>8} {'inter':>8}   (ranked by interaction mass)")
        for b, h, it, at, inter, _ in rows[:25]:
            print(f"{b:>5}.{h:<2} {it:7.1f}% {at:7.1f}% {inter:7.1f}%")
        return

    print(f"\n{'block':>6} {'ITEM (entity)':>15} {'ATTR (question)':>17} {'ITEM x ATTR':>13}")
    for bi, b in enumerate(blocks):
        if b not in want:
            continue
        X = G[:, :, bi, :]
        if heads is not None:
            cols = head_cols(heads, b, head_dim)
            if cols is None:
                continue
            X = X[:, :, cols]
        it, at, inter = decompose(X)
        print(f"{b:>6} {it:14.1f}% {at:16.1f}% {inter:12.1f}%")
    print("\n  Only ITEM x ATTR identifies 'this item's this attribute'. A site where it is\n"
          "  small has almost none of what a selective intervention would need to read.")


if __name__ == "__main__":
    main()
