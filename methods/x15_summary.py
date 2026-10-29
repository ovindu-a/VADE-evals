"""X15 summary (CPU): how concentrated is each model's image->text entity bottleneck?

Reads what the X15 notebook writes per model,

    results/x15_cross_model_heads/<model slug>/
        pipeline/<entity>/<attribute>/pipeline.json       methods/image_head_pipeline.py's state
        pipeline/<entity>/<attribute>/head_trace_*.json   stage 3 (head_trace.py, --null_ks)
        accuracy/<entity>_accuracy.json                   methods/own_accuracy_prune.py

and writes, per model, summary.json + SUMMARY.md next to them, and a cross-model table at
results/x15_cross_model_heads/SUMMARY.md. The reported number, per (model, entity, attribute):

  k90   the smallest k whose top-k head INSTALL (head_trace phase 3: image NOT patched, only those
        heads' last-token outputs set to their image-patched values) reaches 90% of the full image
        swap's own cause (phase 2's image-patch-only arm, same rows, same scoring). Reported with
        the block-matched random-head null at the same k (head_trace --null_ks: same per-block
        counts as the top-k), and the top-k heads themselves.

k90 is read off head_trace's k grid (the notebook passes a fine one), so it is an upper bound on
the true smallest k. A last-prompt-column install cannot steer past the first answer token
(CLAUDE.md, swap_trace), so for attributes with multi-token answers k90 can be undefined while the
heads still carry the entity: read `frac_at_k` and the stage-2 first-token columns too.

`role_checks` is a placeholder for the later steps (name-writing rank, cross-modal swap,
mean-ablation necessity); each is null until its run exists.

    python methods/x15_summary.py                     # every model under results/x15_cross_model_heads
    python methods/x15_summary.py --models gemma-3-12b-it pixtral-12b
"""
import argparse
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "results", "x15_cross_model_heads")
ROLE_CHECKS = ("name_rank", "crossmodal", "mean_ablation")


def load(path):
    with open(path) as f:
        return json.load(f)


def resolve(state_path, rel_path):
    """pipeline.json stores stage outputs relative to the repo the pipeline RAN in (molab's clone);
    fall back to the same file name next to pipeline.json once the folder has been copied here."""
    if not rel_path:
        return None
    for p in (os.path.join(REPO_ROOT, rel_path), os.path.join(os.path.dirname(state_path), os.path.basename(rel_path))):
        if os.path.exists(p):
            return p
    return None


def span(blocks):
    if not blocks:
        return None
    return str(blocks[0]) if len(blocks) == 1 else f"{blocks[0]}-{blocks[-1]}"


def concentration(trace, frac=0.9):
    """-> the k-curve of top-k sufficiency against the image swap, with the block-matched null."""
    image = trace.get("phase2", {}).get("image_patch_only_cause")
    p3 = trace.get("phase3_sufficiency", {})
    arms = p3.get("arms", [])
    top = {a["k"]: a for a in arms if a["kind"] == "top"}
    null = {}
    for a in arms:
        if a["kind"] == "random_block":
            null.setdefault(a["k"], []).append(a["cause"])
    curve = []
    for k in sorted(top):
        n = null.get(k)
        curve.append({"k": k, "top": top[k]["cause"],
                      "frac_of_image": top[k]["cause"] / image if image else None,
                      "null_mean": sum(n) / len(n) if n else None, "null_max": max(n) if n else None,
                      "null_draws": len(n) if n else 0})
    k90 = next((c["k"] for c in curve if image and c["top"] >= frac * image - 1e-9), None)
    ranked = [f"{b}.{h}" for b, h in trace.get("phase1_ranked", [])]
    at = next((c for c in curve if c["k"] == k90), None)
    n_traced = next((a["k"] for a in arms if a["kind"] == "all"), None)
    return {"image_swap_cause": image, "all_traced_heads_cause": p3.get("ceiling_all_traced_heads"),
            "n_traced_heads": n_traced, "frac": frac, "k90": k90,
            "k90_heads": ranked[:k90] if k90 else None, "top8_heads": ranked[:8],
            "null_at_k90": {"mean": at["null_mean"], "max": at["null_max"], "draws": at["null_draws"]} if at else None,
            "curve": curve, "n_rows": trace.get("n_rows"), "blocks": trace.get("blocks")}


def summarize_model(model_dir):
    slug = os.path.basename(os.path.normpath(model_dir))
    out = {"model": slug, "attributes": {}}
    accuracy = {}
    for p in glob.glob(os.path.join(model_dir, "accuracy", "*_accuracy.json")):
        a = load(p)
        accuracy[a["entity"]] = {k: v for k, v in a["by"].items() if "/" not in k}
    out["accuracy"] = accuracy
    for sp in sorted(glob.glob(os.path.join(model_dir, "pipeline", "*", "*", "pipeline.json"))):
        st = load(sp)
        entity, attribute = st["entity"], st["attribute"]
        h = st.get("handoff", {})
        n_layers = max(map(int, h["image_curve"])) if h.get("image_curve") else None
        hb = h.get("handoff_blocks") or []
        win = st.get("window") or {}
        rec = {"scoring": st.get("scoring"), "skipped": st.get("skipped"),
               "source_read": st.get("scoring_check", {}).get("source_read"),
               "n_layers": n_layers, "patch_layer": h.get("patch_layer"), "read_end": h.get("read_end"),
               "handoff_blocks": span(hb),
               "handoff_depth": f"{hb[0] / n_layers:.2f}-{(hb[-1] + 1) / n_layers:.2f}" if hb and n_layers else None,
               "window": win.get("label"), "window_fallback": win.get("fallback"),
               "warnings": h.get("warnings", []) + st.get("scoring_check", {}).get("warnings", []),
               "role_checks": {k: None for k in ROLE_CHECKS}}
        wp = resolve(sp, st.get("windows_json"))
        if wp:
            wj = load(wp)
            w = next((w for w in wj["windows"] if w["blocks"] == win.get("blocks")), None)
            io = wj["image_patch_only"]
            rec["window_first_src"] = w["suff"]["first_src"] if w else None
            rec["image_first_src"] = io["first_src"]
            rec["n_first"] = io.get("n_first")
            rec["heads_per_block"] = wj.get("n_heads")
        tp = resolve(sp, st.get("head_trace_json"))
        if tp:
            rec["heads"] = concentration(load(tp))
            hp = rec.get("heads_per_block")
            if rec["heads"]["k90"] and hp and n_layers:
                rec["heads"]["k90_frac_of_model_heads"] = rec["heads"]["k90"] / (hp * n_layers)
        out["attributes"][f"{entity}/{attribute}"] = rec
    return out


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}%"


def model_markdown(s):
    L = [f"# X15: {s['model']}", "",
         "Generated by `methods/x15_summary.py` from the X15 notebook's outputs. k90 = smallest k (on "
         "head_trace's grid) whose top-k head install reaches 90% of the full image swap's cause; the "
         "null is block-matched random heads at the same k.", ""]
    if s["accuracy"]:
        L += ["## Clean accuracy (own_accuracy_prune.py, VADE text match)", ""]
        for e, by in s["accuracy"].items():
            L.append(f"- {e}: " + ", ".join(f"{a} {pct(v['accuracy'])} (n={v['n']})" for a, v in by.items()))
        L.append("")
    L += ["| entity/attribute | source_read | handoff (depth) | window | image swap | k90 | top-k90 | null@k90 (mean/max) | k90 heads |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, r in s["attributes"].items():
        hd = r.get("heads") or {}
        at = next((c for c in hd.get("curve", []) if c["k"] == hd.get("k90")), None)
        nl = hd.get("null_at_k90") or {}
        L.append(f"| {key} | {pct(r['source_read'])} | {r['handoff_blocks'] or '--'} ({r['handoff_depth'] or '--'}) | "
                 f"{r['window'] or '--'} | {pct(hd.get('image_swap_cause'))} | {hd.get('k90') or 'not reached'} | "
                 f"{pct(at['top']) if at else '--'} | {pct(nl.get('mean'))} / {pct(nl.get('max'))} | "
                 f"`{' '.join(hd.get('k90_heads') or hd.get('top8_heads') or [])}` |")
    L.append("")
    for key, r in s["attributes"].items():
        hd = r.get("heads")
        if not hd:
            continue
        L += [f"## {key}: k-curve (blocks {span(hd['blocks'])}, n={hd['n_rows']})", "",
              "| k | top-k cause | / image swap | block-matched null mean | null max |", "|---|---|---|---|---|"]
        for c in hd["curve"]:
            L.append(f"| {c['k']} | {pct(c['top'])} | {pct(c['frac_of_image'])} | {pct(c['null_mean'])} | {pct(c['null_max'])} |")
        for w in r["warnings"]:
            L.append(f"\n> warning: {w}")
        L.append("")
    return "\n".join(L) + "\n"


def cross_markdown(summaries):
    L = ["# X15: cross-model entity-mover bottleneck", "",
         "One row per (model, entity, attribute). Same rule-based procedure on every model "
         "(`methods/image_head_pipeline.py` stage 1 -> 2 -> 3); k90 is the concentration number.", "",
         "| model | entity/attribute | handoff blocks | depth | window | image swap | k90 | k90 / all heads | null@k90 | top-8 heads |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for key, r in s["attributes"].items():
            hd = r.get("heads") or {}
            nl = hd.get("null_at_k90") or {}
            L.append(f"| {s['model']} | {key} | {r['handoff_blocks'] or '--'} | {r['handoff_depth'] or '--'} | "
                     f"{r['window'] or '--'} | {pct(hd.get('image_swap_cause'))} | {hd.get('k90') or 'not reached'} | "
                     f"{pct(hd.get('k90_frac_of_model_heads'))} | {pct(nl.get('mean'))} | "
                     f"`{' '.join(hd.get('top8_heads') or [])}` |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--models", nargs="+", default=None, help="Model slugs (default: every dir under --root).")
    args = ap.parse_args(argv)
    slugs = args.models or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d, "pipeline")))
    summaries = []
    for slug in slugs:
        d = os.path.join(args.root, slug)
        s = summarize_model(d)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        with open(os.path.join(d, "SUMMARY.md"), "w") as f:
            f.write(model_markdown(s))
        summaries.append(s)
        print(f"wrote {os.path.join(d, 'SUMMARY.md')}")
    if summaries:
        with open(os.path.join(args.root, "SUMMARY.md"), "w") as f:
            f.write(cross_markdown(summaries))
        print(f"wrote {os.path.join(args.root, 'SUMMARY.md')}")
    return summaries


if __name__ == "__main__":
    main()
