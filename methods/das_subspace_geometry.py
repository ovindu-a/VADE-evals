#!/usr/bin/env python3
"""What subspace did `das_rotated` actually learn? -- read off the checkpoints.

`final_score` cannot answer this and neither can `slack`. Every arm at the head
site lies on one cause<->iso trade-off, so two arms that found completely
different geometry can score identically. These checkpoints carry the rotation
and the mask, so the geometry is directly inspectable.

Four analyses, in the order they have to be read (R14.3-R14.4):

  1. SATURATION  -- is the mask binary (a hard subspace) or fractional? Only a
     binary mask is comparable to das_fixed's hard K.
  2. NESTING     -- is a sparse arm's span contained in a denser arm's? Reported
     against the random-subspace expectation b/D, which is large when b is, so a
     raw overlap of 0.8 can be *below* chance.
  3. CONTROLS    -- the two ways nesting could be an artefact. If every arm
     shares --seed the rotations start identical, so agreement would prove
     nothing; and if the rotation never trains, the mask is choosing among fixed
     random axes. Both are checked: mean |cos| between same-index axes against
     sqrt(2/pi*D), and the Jaccard of selected coordinate INDICES against chance.
  4. ALIGNMENT   -- is the learned span the ENTITY directions or the ATTRIBUTE
     directions? Needs attr_capture's grid. Note the attribute-mean subspace has
     rank (n_attributes - 1), so its overlap is capped at rank/k however perfect
     the alignment; the report gives that ceiling alongside chance.

  python methods/das_subspace_geometry.py --checkpoints ~/Downloads/checkpoints
  python methods/das_subspace_geometry.py --checkpoints DIR --skip_alignment
"""
import argparse, glob, math, os, re
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS = {21: [1, 5], 22: [13, 15, 17, 19], 23: [3, 4, 6, 17]}


def arm_label(path):
    m = re.search(r"_m([0-9.eE+-]+?)(?:_|\.pt)", os.path.basename(path))
    return float(m.group(1)) if m else float("nan")


def load_arm(path, vade_root):
    """-> {block: (selected_rows [k, d] orthonormal, mask_sigmoid [d], R [d, d])}"""
    import sys
    sys.path.insert(0, REPO_ROOT)
    from methods.head_das import load_das_module
    das = load_das_module(vade_root)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in ck.items():
        if not str(k).isdigit():
            continue
        iv = das.RotatedSpaceIntervention(v["masks"].shape[0])
        iv.load_state_dict(v, strict=False)
        R = iv.rotate_layer.weight.detach()
        s = torch.sigmoid(v["masks"] / float(v["temperature"]))
        out[int(k)] = (R[s > 0.5], s, R)
    assert out, f"{path} has no integer block keys -- not a head_das checkpoint?"
    return out


def span_overlap(A, B):
    """Fraction of orthonormal A's span captured by orthonormal B, and its chance value."""
    return float(((A @ B.T) ** 2).sum()) / A.shape[0], B.shape[0] / A.shape[1]


def pooled(arms, a, b, fn):
    """fn per block, pooled weighting each block by the sparse arm's dim there."""
    num = den = chn = 0.0
    for blk in arms[a]:
        o, c = fn(arms[a][blk][0], arms[b][blk][0])
        k = arms[a][blk][0].shape[0]
        num += o * k; chn += c * k; den += k
    return num / den, chn / den


def basis(M, k):
    M = M - M.mean(0, keepdims=True)
    return np.linalg.svd(M, full_matrices=False)[2][:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help="Directory of *.pt written by head_das.py --method das_rotated.")
    ap.add_argument("--glob", default="*mlr4*.pt",
                    help="Only compare arms that differ in ONE knob; the default takes the "
                         "--mask_lr 4.0 sweep and leaves the rescaled arm out.")
    ap.add_argument("--vade_root", default=os.path.join(os.path.dirname(REPO_ROOT), "VADE"))
    ap.add_argument("--capture_dir", default=os.path.join(REPO_ROOT, "results", "attr_capture",
                                                          "flags", "blocks15-27_n84"))
    ap.add_argument("--skip_alignment", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.checkpoints), args.glob)),
                   key=arm_label)
    assert paths, f"no checkpoints matching {args.glob}"
    arms = {arm_label(p): load_arm(p, args.vade_root) for p in paths}
    cs = sorted(arms)
    blocks = sorted(arms[cs[0]])

    print("== 1. SATURATION: hard subspace, or fractional transfer? ==")
    print(f"{'coef':>8} {'blk':>4} {'sum':>7} {'width':>6} | {'<.01':>6} {'.01-.5':>7} "
          f"{'.5-.99':>7} {'>.99':>6}")
    for c in cs:
        for b in blocks:
            s = arms[c][b][1]
            n = lambda lo, hi: int(((s >= lo) & (s < hi)).sum())
            print(f"{c:>8g} {b:>4} {float(s.sum()):7.1f} {len(s):6} | {n(0,.01):6} "
                  f"{n(.01,.5):7} {n(.5,.99):7} {n(.99,1.01):6}")

    print("\n== 2. NESTING: sparse arm's span inside the denser arm's (observed / chance) ==")
    print(f"{'sparse':>8}{'dims':>6} | " + "".join(f"{c:>16g}" for c in cs[:-1]))
    for a in sorted(cs, reverse=True):
        da = sum(arms[a][b][0].shape[0] for b in blocks)
        row = []
        for b in cs[:-1]:
            if b >= a:
                row.append(f"{'-':>16}"); continue
            o, ch = pooled(arms, a, b, span_overlap)
            row.append(f"{o:7.3f} /{ch:6.3f}".rjust(16))
        print(f"{a:>8g}{da:>6} | " + "".join(row))

    print("\n== 3a. CONTROL: did the ROTATION train, or do all arms share an init? ==")
    print(f"{'pair':>18} " + "".join(f"{b:>9}" for b in blocks) + "   expected if random")
    for i, a in enumerate(cs[:-1]):
        b = cs[i + 1]
        cells = [f"{float((arms[a][blk][2] * arms[b][blk][2]).sum(1).abs().mean()):9.3f}"
                 for blk in blocks]
        exp = ", ".join(f"{math.sqrt(2/(math.pi*arms[a][blk][2].shape[0])):.3f}" for blk in blocks)
        print(f"{f'{a:g} vs {b:g}':>18} " + "".join(cells) + f"   {exp}")

    print("\n== 3b. CONTROL: do the masks keep the same COORDINATE INDICES? ==")
    for i, a in enumerate(cs[:-1]):
        for b in cs[i + 1:]:
            inter = un = 0; chn = chd = 0.0
            for blk in blocks:
                ka, kb = arms[a][blk][1] > 0.5, arms[b][blk][1] > 0.5
                D = len(ka); na, nb = int(ka.sum()), int(kb.sum()); e = na * nb / D
                inter += int((ka & kb).sum()); un += int((ka | kb).sum())
                chn += e; chd += na + nb - e
            print(f"  {a:>7g} vs {b:<7g}  Jaccard {inter/un:6.3f}   chance {chn/chd:6.3f}")

    if args.skip_alignment:
        return
    import sys
    sys.path.insert(0, REPO_ROOT)
    from methods.attr_variance import load_grid
    G, meta = load_grid(args.capture_dir, "attn_head_output")
    hd = meta.get("head_dim", 128)
    print("\n== 4. ALIGNMENT: is the learned span ENTITY or ATTRIBUTE structure? ==")
    print(f"{'coef':>7} {'blk':>4} {'dims':>5} {'top-k PCA':>10} {'ITEM means':>11} "
          f"{'ATTR means':>11} {'chance':>8} {'ATTR ceil':>10}")
    for c in cs:
        for b in blocks:
            cols = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in HEADS[b]])
            X = G[:, :, meta["blocks"].index(b), :][:, :, cols]
            d, k = X.shape[2], arms[c][b][0].shape[0]
            Q = arms[c][b][0].numpy()
            f = lambda B: float(((Q @ B.T) ** 2).sum()) / k
            r_at = min(k, X.shape[1] - 1)
            print(f"{c:>7g} {b:>4} {k:>5} {f(basis(X.reshape(-1, d), k)):10.3f} "
                  f"{f(basis(X.mean(1), min(k, X.shape[0]-1))):11.3f} "
                  f"{f(basis(X.mean(0), r_at)):11.3f} {k/d:8.3f} {r_at/k:10.3f}")
    print("\n  ATTR ceil = the most of its own span a k-dim arm COULD take from a rank-(n_attr-1)\n"
          "  subspace. Read ATTR means against both that and chance before calling it non-zero.")


if __name__ == "__main__":
    main()
