"""Head-set overlap against chance: across entities (E7) and against published lists (E10).

No model, no GPU -- reads head lists off disk and runs permutation nulls.

THE QUESTION

The flags conduit (`common10`, blocks 21-23) reappears in the brands and
celebrities traces by eye (ATTRIBUTE_HEAD_EXPERIMENTS.md, image_head_pipeline
SUMMARY.md files). "By eye" is not a statistic: every one of those traces ranks
heads INSIDE a narrow window of 3-5 blocks, so two top-8 lists drawn from the same
84-140 heads overlap a little by chance alone, and more if both concentrate on the
same block. This script puts a null under every overlap.

  E7  each entity/attribute's top-k image->text heads (head_trace phase-1 ranking)
      against flags' common10, against each other, and the CONSENSUS count (heads
      in >= m sets) against sets re-drawn at random.
  E10 our movers (common10) and routers (router8) against published Qwen2.5-VL-7B
      head lists -- Park et al.'s Visual Retrieval Heads (arXiv 2608.27417) and
      Lietzow et al.'s knowledge-promoting / routing / writing heads (arXiv
      2606.28273) -- read from methods/published_heads.json, which records each
      list's provenance and index base. Converted to this repo's 0-based
      (block, head) before comparison; a list with an unconfirmed index base is
      compared under BOTH conventions and flagged.

TWO NULLS, BECAUSE "CHANCE" DEPENDS ON WHAT YOU HOLD FIXED

  pool      B redrawn uniformly from the heads it could have been chosen from (its
            trace's own block window; every head of the model for a published
            list). Asks: are these the same heads, given the windows?
  block     B redrawn keeping its per-block head counts. Asks: given that both sets
            sit in the same blocks, do they pick the same heads WITHIN a block?
            Strictly harder to pass; it is the null that rules out "both lists just
            live in block 23".

p = P(null overlap >= observed), Monte Carlo with --n_perm draws (add-one
smoothed, so p is never reported as 0). The consensus statistic redraws EVERY set
under the chosen null simultaneously.

Usage
-----
    python methods/head_overlap.py                         # all traces found + published lists
    python methods/head_overlap.py --k 8 --n_perm 100000
    python methods/head_overlap.py --model_id Qwen/Qwen2.5-VL-3B-Instruct \\
        --pipeline_root results/image_head_pipeline_Qwen2.5-VL-3B-Instruct --n_layers 36 --n_heads 16
"""
import argparse
import glob
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import COMMON10, MODEL_ID, TRACE_GLOB, parse_heads  # noqa: E402
from methods.head_vocab_projection import ROUTER8  # noqa: E402

PUBLISHED = os.path.join(REPO_ROOT, "methods", "published_heads.json")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def trace_is_live(trace):
    """head_swap_vade.heads_from_trace's discriminator, non-fatal: a trace whose
    patch-all-heads arm left >= 75% of base answers intact ranked heads the readout
    never reads (pre-518cf31). Unknown (no phase-3) counts as live."""
    p3 = trace.get("phase3_sufficiency") or {}
    all_arm = next((a for a in p3.get("arms", []) if a.get("kind") == "all"), None)
    if all_arm is None or all_arm.get("base_kept") is None:
        return True
    return 1.0 - float(all_arm["base_kept"]) >= 0.25


def load_trace_sets(pipeline_root, flags_glob, k, include_inert=False):
    """-> {name: {"heads": [(b, h)], "pool_blocks": [b...], "source": path}} for every
    head_trace JSON found: the image_head_pipeline outputs (one per entity/attribute)
    plus flags' hand-run blocks 21-23 traces."""
    paths = sorted(glob.glob(os.path.join(pipeline_root, "*", "*", "head_trace_*.json")))
    paths += sorted(glob.glob(flags_glob.replace("{entity}", "flags").replace("{attribute}", "*")))
    out, skipped = {}, []
    for p in paths:
        d = json.load(open(p))
        name = f"{d['entity']}/{d['attribute']}"
        if name in out:            # a re-run of the same attribute: keep the first (pipeline) one
            continue
        if not include_inert and not trace_is_live(d):
            skipped.append(name)
            continue
        ranked = [tuple(h) for h in d["phase1_ranked"]]
        out[name] = {"heads": sorted(ranked[:k]), "pool_blocks": sorted(d["blocks"]), "source": p,
                     "n_heads": d.get("n_heads")}
    return out, skipped


def to_zero_based(heads, layer_base, head_base):
    return sorted((b - layer_base, h - head_base) for b, h in heads)


def load_published(path, model_id):
    """-> {name: {"heads", "pool_blocks": None, "source", "index_note"}} for this model.

    published_heads.json schema (one entry per list):
      {"name": "park_vrh", "paper": "...", "url": "...", "model": "<HF id or label>",
       "applies_to": ["Qwen/Qwen2.5-VL-7B-Instruct"], "layer_base": 0|1|null,
       "head_base": 0|1|null, "heads": [[layer, head], ...], "verbatim": true|false, "note": "..."}
    A null base means the paper does not say; the list is then emitted twice, as
    '<name>' (0-based) and '<name>@1based', so a reader sees both."""
    if not os.path.exists(path):
        return {}
    out = {}
    for e in json.load(open(path))["lists"]:
        if model_id not in e.get("applies_to", []) or not e.get("heads"):
            continue
        heads = [tuple(h) for h in e["heads"]]
        lb, hb = e.get("layer_base"), e.get("head_base")
        variants = [(e["name"], lb or 0, hb or 0)]
        if lb is None or hb is None:
            variants.append((e["name"] + "@1based", 1 if lb is None else lb, 1 if hb is None else hb))
        for name, l0, h0 in variants:
            out[name] = {"heads": to_zero_based(heads, l0, h0), "pool_blocks": None,
                         "source": e.get("url", ""), "verbatim": bool(e.get("verbatim")),
                         "index_note": "index base unconfirmed" if (lb is None or hb is None) else ""}
    return out


# ---------------------------------------------------------------------------
# Nulls (pure; unit-tested)
# ---------------------------------------------------------------------------

def draw(rng, spec, n_layers, n_heads, null):
    """One random replacement for a head set under `null` -> set of (b, h).
    `rng` is a random.Random (stdlib only, so this runs where numpy is absent)."""
    heads = spec["heads"]
    if null == "block":
        out = set()
        by_block = {}
        for b, _ in heads:
            by_block[b] = by_block.get(b, 0) + 1
        for b, c in by_block.items():
            out |= {(b, h) for h in rng.sample(range(n_heads), c)}
        return out
    blocks = spec["pool_blocks"] if spec["pool_blocks"] is not None else list(range(n_layers))
    pool = [(b, h) for b in blocks for h in range(n_heads)]
    return set(rng.sample(pool, len(heads)))


def _p_value(sims, obs):
    return (1 + sum(x >= obs for x in sims)) / (1 + len(sims))


def overlap_test(a, b, n_layers, n_heads, null, n_perm, seed=0):
    """A held fixed, B redrawn. -> {"observed", "null_mean", "p", "shared"}."""
    rng = random.Random(seed)
    sa = set(a["heads"])
    obs = len(sa & set(b["heads"]))
    sims = [len(sa & draw(rng, b, n_layers, n_heads, null)) for _ in range(n_perm)]
    return {"observed": obs, "null_mean": sum(sims) / n_perm, "p": _p_value(sims, obs),
            "shared": [f"{x}.{y}" for x, y in sorted(sa & set(b["heads"]))]}


def multiplicity(sets):
    """-> {(b, h): number of sets containing it}."""
    counts = {}
    for heads in sets:
        for x in set(heads):
            counts[x] = counts.get(x, 0) + 1
    return counts


def consensus_test(specs, m, n_layers, n_heads, null, n_perm, seed=0):
    """Number of heads present in >= m of the sets, against every set redrawn at once."""
    rng = random.Random(seed)
    obs_counts = multiplicity([s["heads"] for s in specs])
    obs = sum(c >= m for c in obs_counts.values())
    sims = []
    for _ in range(n_perm):
        counts = multiplicity([draw(rng, s, n_layers, n_heads, null) for s in specs])
        sims.append(sum(c >= m for c in counts.values()))
    return {"m": m, "observed": obs, "null_mean": sum(sims) / n_perm, "p": _p_value(sims, obs),
            "heads": sorted(f"{b}.{h}" for (b, h), c in obs_counts.items() if c >= m)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--pipeline_root", default=os.path.join(REPO_ROOT, "results", "image_head_pipeline"))
    ap.add_argument("--flags_trace_glob", default=os.path.join(REPO_ROOT, TRACE_GLOB),
                    help="flags' hand-run blocks 21-23 traces ({entity}/{attribute} placeholders).")
    ap.add_argument("--published", default=PUBLISHED)
    ap.add_argument("--k", type=int, default=8, help="Top-k of each trace's ranking.")
    ap.add_argument("--n_layers", type=int, default=28)
    ap.add_argument("--n_heads", type=int, default=28)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--consensus_m", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--include_inert", action="store_true")
    ap.add_argument("--compare_pipeline_root", default=None,
                    help="Another model's image_head_pipeline root (same n_layers x n_heads, e.g. InternVL3-8B "
                         "vs Qwen2.5-VL-7B, both a 28x28 Qwen2.5-7B text stack): each entity/attribute's trace is "
                         "compared with the SAME entity/attribute here. Only same-shape models make sense.")
    ap.add_argument("--compare_label", default="other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    slug = args.model_id.split("/")[-1]
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "head_overlap",
                                           "" if slug == MODEL_ID.split("/")[-1] else slug)
    traces, skipped = load_trace_sets(args.pipeline_root, args.flags_trace_glob, args.k, args.include_inert)
    ours = {}
    if slug == MODEL_ID.split("/")[-1]:
        # Fixed lists only mean anything on the model they were found on.
        ours = {"flags/common10": {"heads": parse_heads(COMMON10), "pool_blocks": [21, 22, 23], "source": "R8/R10"},
                "flags/router8": {"heads": parse_heads(ROUTER8), "pool_blocks": [19, 20, 21], "source": "R1"}}
    published = load_published(args.published, args.model_id)
    compare = {}
    if args.compare_pipeline_root:
        other, _ = load_trace_sets(args.compare_pipeline_root, "", args.k, args.include_inert)
        compare = {f"{args.compare_label}:{n}": v for n, v in other.items() if n in traces}
    for name, s in list(traces.items()) + list(ours.items()) + list(published.items()) + list(compare.items()):
        bad = [x for x in s["heads"] if not (0 <= x[0] < args.n_layers and 0 <= x[1] < args.n_heads)]
        assert not bad, f"{name}: heads {bad} outside a {args.n_layers}x{args.n_heads} model -- wrong --n_layers/--n_heads or index base"

    print(f"[head_overlap] {args.model_id}: {len(traces)} traces (top-{args.k}), {len(ours)} fixed sets, "
          f"{len(published)} published lists; nulls: pool, block; n_perm={args.n_perm}")
    if skipped:
        print(f"  skipped inert traces: {skipped}")
    if not published:
        print(f"  no published lists for this model in {args.published} -- E10 rows omitted")
    for name, s in {**traces, **ours, **published}.items():
        print(f"  {name:<34} {' '.join(f'{b}.{h}' for b, h in s['heads'])}")

    results = {"model_id": args.model_id, "k": args.k, "n_perm": args.n_perm, "skipped_inert": skipped,
               "sets": {n: {**s, "heads": [list(x) for x in s["heads"]]}
                        for n, s in {**traces, **ours, **published, **compare}.items()},
               "pairs": [], "consensus": []}
    anchors = list(ours) + list(traces)
    pairs = []
    # E7: every set against flags/common10, and every entity trace against every other.
    for a in anchors:
        for b in list(traces) + list(published):
            if a != b and (b, a) not in pairs and (a, b) not in pairs:
                pairs.append((a, b))
    for b in published:
        for a in ours:
            if (a, b) not in pairs:
                pairs.append((a, b))
    for c in compare:
        pairs.append((c.split(":", 1)[1], c))
    allsets = {**traces, **ours, **published, **compare}
    rows = []
    for a, b in pairs:
        rec = {"a": a, "b": b, "n_a": len(allsets[a]["heads"]), "n_b": len(allsets[b]["heads"])}
        for null in ("pool", "block"):
            rec[null] = overlap_test(allsets[a], allsets[b], args.n_layers, args.n_heads, null, args.n_perm,
                                     args.seed)
        rows.append(rec)
    results["pairs"] = rows

    entity_specs = [traces[n] for n in traces]
    if len(entity_specs) >= 3:
        for m in args.consensus_m:
            for null in ("pool", "block"):
                c = consensus_test(entity_specs, m, args.n_layers, args.n_heads, null,
                                   max(1000, args.n_perm // 10), args.seed)
                results["consensus"].append({**c, "null": null, "n_sets": len(entity_specs)})

    os.makedirs(out_dir, exist_ok=True)
    lines = [f"# Head-set overlap ({args.model_id}, top-{args.k})", "",
             "Generated by `methods/head_overlap.py`. p = P(random overlap >= observed). `pool` redraws B from "
             "its own trace window (all heads for a published list); `block` keeps B's per-block counts.", "",
             "`(derived)`: common10 was BUILT from the flags traces (R8: heads in every flags attribute's "
             "top-16), so those rows are circular and confirm nothing; they are kept as a sanity anchor only.", "",
             "| A | B | |A| | |B| | overlap | pool mean | pool p | block mean | block p | shared |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["block"]["p"]):
        r["derived"] = r["a"] == "flags/common10" and r["b"].startswith("flags/")
        lines.append(f"| {r['a']} | {r['b']}{' (derived)' if r['derived'] else ''} | {r['n_a']} | {r['n_b']} | {r['pool']['observed']} | "
                     f"{r['pool']['null_mean']:.2f} | {r['pool']['p']:.2g} | {r['block']['null_mean']:.2f} | "
                     f"{r['block']['p']:.2g} | {' '.join(r['pool']['shared'])} |")
    if results["consensus"]:
        lines += ["", f"## Consensus across the {len(entity_specs)} entity/attribute traces", "",
                  "| m | null | heads in >= m sets | null mean | p | heads |", "|---|---|---|---|---|---|"]
        for c in results["consensus"]:
            lines.append(f"| {c['m']} | {c['null']} | {c['observed']} | {c['null_mean']:.2f} | {c['p']:.2g} | "
                         f"{' '.join(c['heads'])} |")
    if any(s.get("index_note") for s in published.values()):
        lines += ["", "Published lists marked `@1based` are the same list read with a 1-based index: the paper "
                      "does not state its convention. Only one reading can be right; check before citing."]
    with open(os.path.join(out_dir, "overlap.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "OVERLAP.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir}/overlap.json and OVERLAP.md")


if __name__ == "__main__":
    main()
