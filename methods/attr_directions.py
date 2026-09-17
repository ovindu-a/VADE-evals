"""Read methods/attr_capture.py's grid: are there common directions per attribute,
and do the image heads carry the same image data whatever is asked?

Everything here is linear algebra over a fully crossed [items x attributes] grid
of CLEAN activations. Nothing is trained, nothing is patched, and no model is
loaded -- run it on a laptop against the captured arrays.

THE CENTERING IS THE METHOD. Raw activations are dominated by which flag is in
the image; an "attribute direction" read off them would mostly be item identity.
Subtracting each ITEM's own mean over its four questions removes the item main
effect exactly, and what survives is what the question changed:

    Y[c,a] = X[c,a] - mean_a' X[c,a']        m_a = mean_c Y[c,a]

Because sum_a Y[c,a] = 0 by construction, the four centroids sum to zero and
span at most A-1 dimensions -- so a reported rank of 3 for four attributes is the
CEILING, not a finding, while a rank of 1 would be.

WHAT EACH ANALYSIS ANSWERS

  directions   Is there a consistent per-attribute direction at all? Reported as
               `attr_var_explained` (the fraction of item-demeaned variance the
               attribute centroids account for) and a leave-one-ITEM-out
               nearest-centroid accuracy. Held out by item, never by row: the
               four rows of one item are dependent after centering, so holding
               out single rows would leak.
  geometry     Cosines between the four centroids and their singular-value
               spectrum. Four directions that are near-orthogonal is a different
               claim from one axis the attributes sit along.
  heads        The same separability per (block, head) on `attn_head_output`,
               giving a CORRELATIONAL head ranking. Pass --trace to compare it
               against attr_head_trace's CAUSAL one; the two disagreeing is
               informative, not a bug in either -- a head can carry an attribute
               linearly without the model reading it out that way.
  image_heads  The item-4 question. For the top-k heads of an IMAGE head_trace,
               decompose each head's variance over the grid into

                   item main effect + attribute main effect + interaction

               (an exact orthogonal split on a balanced grid). A head that only
               ferries the picture is nearly all ITEM. A head whose INTERACTION
               term is large reads the same image differently depending on what
               was asked -- which is "only the data relevant to the attribute".

Scale caveat, handled: every separability statistic above is scale-invariant, so
a head can look highly separable while writing almost nothing into the residual
stream. `landed_scale` multiplies each head's RMS by ||W_O_h||_F from meta.json
(o_proj weights heads very differently -- the same reason head_trace ranks by
delta_resid rather than delta_z). Read the two columns together.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(capture_dir):
    """-> (meta, {site: X[n_items, n_attributes, n_blocks, width]}).

    The capture writes the grid item-major/attribute-minor; that is re-derived
    from index.jsonl rather than assumed, so a reshape can never silently
    transpose the two factors.
    """
    d = Path(capture_dir)
    meta = json.loads((d / "meta.json").read_text())
    index = [json.loads(line) for line in (d / "index.jsonl").read_text().splitlines() if line.strip()]
    items, attributes = meta["items"], meta["attributes"]
    slot = {(i, a): None for i in items for a in attributes}
    for r in index:
        if r["condition"] != "clean":
            continue
        key = (r["item"], r["attribute"])
        if key not in slot:
            raise ValueError(f"index row {r['row']} is outside the declared grid: {key}")
        if slot[key] is not None:
            raise ValueError(f"duplicate grid cell {key}")
        slot[key] = r["row"]
    missing = [k for k, v in slot.items() if v is None]
    if missing:
        raise ValueError(f"grid is incomplete ({len(missing)} cells missing, e.g. {missing[:3]}); "
                         f"every analysis here assumes a balanced design")
    order = np.array([[slot[(i, a)] for a in attributes] for i in items])   # [n_items, n_attributes]
    out = {}
    for site in meta["sites"]:
        path = d / f"acts_{site}.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        out[site] = np.asarray(arr[order.reshape(-1)], dtype=np.float64) \
            .reshape(len(items), len(attributes), arr.shape[1], arr.shape[3])
    return meta, out


def separability(X):
    """X: [n_items, n_attributes, width] -> attribute separability after removing
    each item's own mean.

    attr_var_explained is between/(between+within) on the item-demeaned data: 0
    means the four questions move this representation in no consistent way, 1
    means the attribute fully determines it once the item is factored out.
    """
    C, A, _ = X.shape
    Y = X - X.mean(axis=1, keepdims=True)
    centroids = Y.mean(axis=0)                                    # [A, width]
    between = C * float((centroids ** 2).sum())
    within = float(((Y - centroids[None]) ** 2).sum())
    total = between + within
    # Leave-one-ITEM-out nearest centroid. Recomputing centroids without the held-out item is what
    # keeps this from being a description of the training set.
    hits = 0
    for c in range(C):
        rest = np.delete(Y, c, axis=0).mean(axis=0)               # [A, width]
        d = ((Y[c][:, None, :] - rest[None]) ** 2).sum(-1)        # [A true, A candidate]
        hits += int((d.argmin(axis=1) == np.arange(A)).sum())
    return {"attr_var_explained": between / total if total else 0.0,
            "loo_accuracy": hits / (C * A), "chance": 1.0 / A,
            "rms": float(np.sqrt((X ** 2).mean()))}


def geometry(X):
    """Cosines between the attribute centroids and their singular values."""
    Y = X - X.mean(axis=1, keepdims=True)
    m = Y.mean(axis=0)
    norms = np.linalg.norm(m, axis=1)
    unit = m / np.maximum(norms, 1e-12)[:, None]
    sv = np.linalg.svd(m, compute_uv=False)
    return {"norms": norms.tolist(), "cosines": (unit @ unit.T).tolist(),
            "singular_values": sv.tolist(),
            "effective_rank": float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12))}


def variance_decomposition(X):
    """X: [n_items, n_attributes, width] -> exact orthogonal split of the total
    sum of squares on a BALANCED grid:

        X[c,a] = m + alpha_c + beta_a + gamma[c,a]

    SS_total = A*sum_c||alpha_c||^2 + C*sum_a||beta_a||^2 + sum||gamma||^2.
    Returned as fractions, so a head is described by where its variation lives
    rather than by how big it is (see `landed_scale` for that).
    """
    C, A, _ = X.shape
    m = X.mean(axis=(0, 1))
    alpha = X.mean(axis=1) - m
    beta = X.mean(axis=0) - m
    gamma = X - m - alpha[:, None, :] - beta[None]
    ss_item = A * float((alpha ** 2).sum())
    ss_attr = C * float((beta ** 2).sum())
    ss_int = float((gamma ** 2).sum())
    total = ss_item + ss_attr + ss_int
    if total <= 0:
        return {"item": 0.0, "attribute": 0.0, "interaction": 0.0, "ss_total": 0.0}
    return {"item": ss_item / total, "attribute": ss_attr / total,
            "interaction": ss_int / total, "ss_total": total}


def head_slices(width, n_heads):
    head_dim = width // n_heads
    return [slice(h * head_dim, (h + 1) * head_dim) for h in range(n_heads)]


def ranked_heads_from_trace(path, rank_by=None, top_k=16):
    trace = json.loads(Path(path).read_text())
    field = rank_by or trace.get("rank_by", "delta_resid")
    ranked = sorted(trace["phase1"], key=lambda r: -abs(r[field]))
    return [(int(r["block"]), int(r["head"])) for r in ranked[:top_k]], field, trace


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", help="A methods/attr_capture.py output directory")
    ap.add_argument("--sites", nargs="+", default=None, help="Default: every site in the capture")
    ap.add_argument("--trace", default=None,
                    help="attr_head_trace JSON: compare this correlational head ranking with its "
                         "causal one (rank correlation + top-k overlap)")
    ap.add_argument("--image_trace", default=None,
                    help="head_trace JSON from the IMAGE experiment: run the item-4 variance "
                         "decomposition on its top --top_k heads")
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--rank_by", default=None, choices=[None, "delta_resid", "delta_z", "delta_dla"])
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args(argv)

    meta, data = load(args.capture_dir)
    blocks, attributes = meta["blocks"], meta["attributes"]
    sites = args.sites or [s for s in meta["sites"] if s in data]
    n_heads = meta["n_heads"]
    report = {"capture": str(args.capture_dir), "blocks": blocks, "attributes": attributes,
              "n_items": len(meta["items"])}
    print(f"{args.capture_dir}\n  {len(meta['items'])} items x {len(attributes)} attributes "
          f"{attributes}, blocks {blocks[0]}-{blocks[-1]}, sites {sites}")

    # ---- is there a per-attribute direction, and where ----
    print(f"\n{'=' * 78}\n=== attribute separability after removing each item's own mean\n{'=' * 78}")
    print(f"  chance LOO accuracy = {1 / len(attributes):.3f}")
    print(f"  {'block':>5} " + "".join(f"{s:>26}" for s in sites))
    print(f"  {'':>5} " + "".join(f"{'var_expl / LOO acc':>26}" for _ in sites))
    report["separability"] = {}
    for bi, b in enumerate(blocks):
        cells = []
        for site in sites:
            s = separability(data[site][:, :, bi, :])
            report["separability"].setdefault(site, {})[str(b)] = s
            cells.append(f"{s['attr_var_explained']:11.3f} /{s['loo_accuracy']:12.3f}")
        print(f"  {b:>5} " + "".join(f"{c:>26}" for c in cells))

    # ---- geometry of the four directions, at the most separable residual block ----
    ref_site = "residual" if "residual" in sites else sites[0]
    best = max(range(len(blocks)),
               key=lambda i: report["separability"][ref_site][str(blocks[i])]["attr_var_explained"])
    print(f"\n{'=' * 78}\n=== direction geometry: {ref_site} at block {blocks[best]} "
          f"(most separable)\n{'=' * 78}")
    g = geometry(data[ref_site][:, :, best, :])
    report["geometry"] = {"site": ref_site, "block": blocks[best], **g}
    print(f"  {'':>13}" + "".join(f"{a:>14}" for a in attributes))
    for i, a in enumerate(attributes):
        print(f"  {a:>13}" + "".join(f"{g['cosines'][i][j]:14.3f}" for j in range(len(attributes))))
    print(f"  centroid norms: " + ", ".join(f"{a}={n:.2f}" for a, n in zip(attributes, g["norms"])))
    print(f"  singular values: " + ", ".join(f"{v:.2f}" for v in g["singular_values"]))
    print(f"  effective rank {g['effective_rank']:.2f} of at most {len(attributes) - 1} "
          f"(centering forces the centroids to sum to zero, so {len(attributes) - 1} is the ceiling, "
          f"not a finding; well below it means the attributes share one axis)")

    # ---- correlational head ranking ----
    if "attn_head_output" in data:
        print(f"\n{'=' * 78}\n=== per-head attribute separability (attn_head_output, correlational)"
              f"\n{'=' * 78}")
        X = data["attn_head_output"]
        width = X.shape[-1]
        slices = head_slices(width, n_heads)
        proj = meta.get("head_proj_norms", {})
        rows = []
        for bi, b in enumerate(blocks):
            for h, sl in enumerate(slices):
                s = separability(X[:, :, bi, sl])
                scale = proj.get(str(b), [1.0] * n_heads)[h]
                rows.append({"block": b, "head": h, "landed_scale": s["rms"] * scale, **s})
        rows.sort(key=lambda r: -r["attr_var_explained"])
        report["head_separability"] = rows
        print(f"  {'block.head':>12} {'var_expl':>10} {'LOO acc':>9} {'landed_scale':>13}")
        for r in rows[:args.top_k]:
            print(f"  {r['block']}.{r['head']:<10} {r['attr_var_explained']:10.3f} "
                  f"{r['loo_accuracy']:9.3f} {r['landed_scale']:13.2f}")
        per_block = {}
        for r in rows:
            per_block.setdefault(r["block"], []).append(r["attr_var_explained"])
        print(f"\n  mean var_expl per block: " +
              ", ".join(f"{b}={np.mean(v):.3f}" for b, v in sorted(per_block.items())))

        if args.trace:
            causal, field, _ = ranked_heads_from_trace(args.trace, args.rank_by, top_k=10 ** 6)
            pos_causal = {bh: i for i, bh in enumerate(causal)}
            shared = [r for r in rows if (r["block"], r["head"]) in pos_causal]
            if shared:
                mine = np.array([i for i, r in enumerate(rows) if (r["block"], r["head"]) in pos_causal])
                theirs = np.array([pos_causal[(r["block"], r["head"])] for r in shared])
                # Spearman without scipy: Pearson on the two rank vectors.
                rho = float(np.corrcoef(mine.argsort().argsort(), theirs.argsort().argsort())[0, 1])
                top_mine = {(r["block"], r["head"]) for r in rows[:args.top_k]}
                top_theirs = set(causal[:args.top_k])
                print(f"\n  vs the CAUSAL ranking in {Path(args.trace).name} (by {field}): "
                      f"rank correlation {rho:+.3f} over {len(shared)} shared heads; "
                      f"top-{args.top_k} overlap {len(top_mine & top_theirs)}/{args.top_k}")
                print(f"  (low overlap is a result, not an error: linear separability is not the same "
                      f"claim as the model reading the head out -- the same gap RESULTS.md records "
                      f"between the Phase B proxy and real interventions.)")
                report["vs_causal"] = {"trace": args.trace, "rank_by": field, "rank_correlation": rho,
                                       "top_k_overlap": len(top_mine & top_theirs),
                                       "top_k": args.top_k, "n_shared": len(shared)}

    # ---- item 4: what do the IMAGE heads carry when the question changes ----
    if args.image_trace:
        print(f"\n{'=' * 78}\n=== image heads under a changing question (variance decomposition)"
              f"\n{'=' * 78}")
        heads, field, trace = ranked_heads_from_trace(args.image_trace, args.rank_by, args.top_k)
        usable = [(b, h) for b, h in heads if b in blocks]
        dropped = [(b, h) for b, h in heads if b not in blocks]
        print(f"  top-{args.top_k} heads of {Path(args.image_trace).name} by {field} "
              f"(image->text trace, patch_layer={trace.get('patch_layer')}, "
              f"attribute={trace.get('attribute')})")
        if dropped:
            print(f"  !! {len(dropped)} of them sit outside the captured blocks {blocks[0]}-{blocks[-1]} "
                  f"and are skipped: {dropped}. Re-capture with --blocks covering them for the full set.")
        X = data["attn_head_output"]
        slices = head_slices(X.shape[-1], n_heads)
        print(f"\n  {'block.head':>12} {'item':>8} {'attribute':>10} {'interaction':>12} "
              f"{'landed_scale':>13}")
        proj = meta.get("head_proj_norms", {})
        rows = []
        for b, h in usable:
            bi = blocks.index(b)
            sub = X[:, :, bi, slices[h]]
            d = variance_decomposition(sub)
            scale = proj.get(str(b), [1.0] * n_heads)[h]
            rows.append({"block": b, "head": h, "landed_scale": float(np.sqrt((sub ** 2).mean())) * scale,
                         **d})
            print(f"  {b}.{h:<10} {d['item']:8.3f} {d['attribute']:10.3f} {d['interaction']:12.3f} "
                  f"{rows[-1]['landed_scale']:13.2f}")
        if rows:
            mean_item = float(np.mean([r["item"] for r in rows]))
            mean_attr = float(np.mean([r["attribute"] for r in rows]))
            mean_int = float(np.mean([r["interaction"] for r in rows]))
            print(f"\n  mean over these heads: item={mean_item:.3f} attribute={mean_attr:.3f} "
                  f"interaction={mean_int:.3f}")
            print(f"  Reading: item ~1 means these heads ferry the SAME image content whatever is "
                  f"asked. A large ATTRIBUTE term means they carry a question-dependent component "
                  f"that is the same for every flag (so: not image data at all). A large "
                  f"INTERACTION term is the interesting one -- the head reads THIS flag differently "
                  f"depending on what was asked, which is attribute-selective image reading.")
            # The control that makes the number mean something: the same decomposition over every
            # head, so "item=0.9" can be read against what an arbitrary head does here.
            allr = [variance_decomposition(X[:, :, bi, sl])
                    for bi in range(len(blocks)) for sl in slices]
            print(f"  ALL {len(allr)} captured heads, for reference: item={np.mean([r['item'] for r in allr]):.3f} "
                  f"attribute={np.mean([r['attribute'] for r in allr]):.3f} "
                  f"interaction={np.mean([r['interaction'] for r in allr]):.3f}")
            report["image_heads"] = {"trace": args.image_trace, "rank_by": field, "heads": rows,
                                     "skipped_outside_blocks": dropped,
                                     "mean": {"item": mean_item, "attribute": mean_attr,
                                              "interaction": mean_int},
                                     "all_heads_mean": {
                                         "item": float(np.mean([r["item"] for r in allr])),
                                         "attribute": float(np.mean([r["attribute"] for r in allr])),
                                         "interaction": float(np.mean([r["interaction"] for r in allr]))}}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
