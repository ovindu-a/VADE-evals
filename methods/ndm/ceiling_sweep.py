"""Training-free CAUSAL-HEADROOM probe: for each (site, layer), swap in
100% of the source's activation and measure how far the model's answer
actually moves. No mask, no training, no optimizer -- just the maximum
effect that site/layer can possibly have.

WHY THIS EXISTS. A learned mask selects a SUBSET of the dimensions a full
swap uses, so its `cause` can never exceed the full swap's. That makes the
full swap a hard UPPER BOUND on what any amount of mask training at that
(site, layer) could achieve -- and it costs a couple of forward passes
instead of a training run. Discovered the hard way: NDM's first real run
(flags/language, layer 14, mlp_hidden) scored cause=2.4%, and the obvious
reading was that the mask had failed to train (it had -- ce_loss was flat
across all 480 optimizer steps). But the full-swap ceiling at that
(site, layer) turned out to be ~0 as well: swapping ALL 18944 neurons at all
24 object positions did not change the answer at all. So no amount of
temperature/l1/lr tuning could ever have helped there, and the training run
was wasted GPU. Run this FIRST to pick a layer, then train only where there
is headroom.

WHAT IT REPORTS, per (site, layer), using the same token-level gold matching
the trainer supervises against (common/targets.py's exact_match, up to
MAX_ANSWER_TOKENS):

  cause_ceiling  fraction of CAUSE rows (rule == match_source) whose full-swap
                 generation equals the SOURCE's gold answer. The upper bound
                 on `cause` for any mask at this site/layer.
  iso_floor      fraction of ISO rows whose full-swap generation still equals
                 the BASE's gold answer. A full swap is the maximum-damage
                 intervention, so this is roughly a LOWER bound on `iso` --
                 a mask touching fewer dimensions should do no worse.
  base_kept      fraction of CAUSE rows still answering the BASE's value after
                 the full swap. High base_kept alongside low cause_ceiling is
                 the signature of a site with no causal purchase: the swap
                 landed (verify_sites.py proves the hook works) but the model
                 ignored it.

Reading the output: a site/layer whose cause_ceiling is near zero is DEAD --
do not train there. A site/layer with high cause_ceiling AND high iso_floor
is the interesting one. High cause_ceiling with low iso_floor means the swap
works by replacing the entity wholesale rather than by isolating the
attribute (what `residual` does at layer 14: it flips `capital` too), which
a sparse mask might still be able to disentangle -- that IS worth training.

The default --sites list is chosen so one run answers three questions at
once, each holding the others fixed (see --sites' help): global-vs-local at
matched width, which sublayer, and narrow-vs-wide pre-projection space. The
layer-14 result that motivated this file had residual alive while BOTH
mlp_output and mlp_hidden were dead, which isolated LOCALITY rather than the
basis as the limiting variable -- but with only MLP sites probed it could not
distinguish "MLPs specifically don't carry this" from "no single sublayer
does". attn_output settles that; attn_head_output additionally tests the one
site here whose units are both privileged (per-head softmax + OV circuit)
and known to be functionally specialized in VLMs, and which this repo's own
methods/attention_maps.py already ranks correlationally -- so a ceiling there
causally tests heads that probe has already flagged.

This is a filter, not a result: the real numbers still come from
methods/ndm/eval.py + VADE's eval/score.py. Token-level exact_match here is a
cheap proxy for score.py's text normalization, chosen so this stays fast
enough to sweep every layer.

Usage:
    python methods/ndm/ceiling_sweep.py --entity flags --attribute language \\
        --layers 2 6 10 14 18 22 26 --sites mlp_hidden mlp_output residual
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import (  # noqa: E402
    BuildBatchCache, build_batch, load_entity_assets, load_tuples, require_pruned_tuples,
)
from methods.common.hooks import make_cache_aware_patch_hook  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.sites import SITES, InterventionSite  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS, exact_match  # noqa: E402
from methods.ndm.config import METHOD_NAME, ndm_logs_dir  # noqa: E402
from methods.ndm.verify_sites import generate_unhooked, hard_mask  # noqa: E402


def full_swap_generation(site, adapter, model, layers, layer, batch, pad_token_id, max_new_tokens):
    """Generation with sigma(m/T)=1 on EVERY dimension -- the whole source
    activation replaces the whole base activation at this site's positions."""
    positions = batch["positions"]
    source_act = site.capture(adapter, model, layer, batch["source_input_ids"], batch["attention_mask"],
                               batch["source_extra"], positions)
    one = hard_mask(site.width(adapter, model), 1.0, model.device)
    patch_fn = make_cache_aware_patch_hook(positions, lambda base_vals: one(base_vals, source_act))
    return site.generate_patched(adapter, model, layers, layer, patch_fn, batch["base_input_ids"],
                                  batch["attention_mask"], batch["base_extra"], pad_token_id, max_new_tokens)


def score_generation(gen_toks, batch):
    """-> (matches_source_rate, matches_base_rate) over this batch, using the
    trainer's own token-level gold matching."""
    pred = gen_toks[:, :MAX_ANSWER_TOKENS]
    ms = exact_match(pred, batch["source_gold_toks"], batch["source_gold_len"])
    mb = exact_match(pred, batch["base_gold_toks"], batch["base_gold_len"])
    return ms.float().mean().item(), mb.float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                     help="Layers to probe. MLP sites need >=1 (layer L addresses block L-1's MLP).")
    ap.add_argument("--sites", nargs="+", choices=list(SITES),
                     default=["residual", "attn_output", "mlp_output", "attn_head_output", "mlp_hidden"],
                     help="Default probes all five, ordered so three comparisons fall out of one run: "
                          "GLOBAL vs LOCAL (residual vs attn_output/mlp_output -- all width hidden_size, so "
                          "width is controlled); WHICH SUBLAYER (attn_output vs mlp_output -- is a dead local "
                          "site MLP-specific, or is any single sublayer too local?); and NARROW vs WIDE "
                          "PRE-PROJECTION space (attn_output vs attn_head_output, mlp_output vs mlp_hidden -- "
                          "the privileged-basis axis, since privilege does not survive o_proj/down_proj).")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n_rows", type=int, default=32,
                     help="Rows per pool (cause and iso probed separately), drawn as a SEEDED RANDOM sample "
                          "rather than the first n (the files are row_index-ordered, which groups by base "
                          "entity -- see the sampling comment in main). Default 32 -- enough for a filter; "
                          "raise it if two layers come out close and you need to separate them.")
    ap.add_argument("--seed", type=int, default=0, help="Row-sampling seed. Same seed = same rows across runs, "
                                                          "so two sites/layers are always compared on identical rows.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_ANSWER_TOKENS + 2)
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--out", default=None,
                     help="JSON output path. Defaults to ceiling_sweep_layers<tag>.json under this entity/"
                          "attribute's NDM logs dir (it's a diagnostic, not a scored result).")
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    layers_tag = "-".join(str(l) for l in args.layers)
    # Logs dir, not results: this is a diagnostic filter, and its numbers are a token-level proxy rather
    # than score.py output -- keeping it out of results/ avoids it being mistaken for a scored run.
    log_dir = ndm_logs_dir(model_slug, args.entity, args.attribute, 0.0, 0.0, 0.0, 0.0,
                            args.positions, "mlp_hidden", pruned)
    out_path = args.out or os.path.join(log_dir, f"ceiling_sweep_layers{layers_tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with tee_to_log(os.path.join(log_dir, f"ceiling_sweep_layers{layers_tag}.log")):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)
        layers_stack = adapter.get_decoder_layers(model)
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        batch_cache = BuildBatchCache()

        rows = load_tuples(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir)
        for r in rows:
            r.setdefault("target_attribute", args.attribute)
        # SEEDED RANDOM sample, not rows[:n]. The tuples files are ordered by row_index, which groups
        # rows by base entity -- so the first 32 cause rows of flags/language are ALL base=AO (Angola),
        # six source flags between them. A "ceiling" measured on one base entity is not a ceiling for
        # the entity set, and the bias is invisible in the output. Verified: rows[:32] gave
        # Counter({'AO': 32}) for the base column.
        all_cause = [r for r in rows if r["rule"] == "match_source"]
        all_iso = [r for r in rows if r["rule"] != "match_source"]
        rng = random.Random(args.seed)
        cause_rows = rng.sample(all_cause, min(args.n_rows, len(all_cause)))
        iso_rows = rng.sample(all_iso, min(args.n_rows, len(all_iso)))
        n_bases = len({r["base"] for r in cause_rows})
        print(f"[{METHOD_NAME}/ceiling_sweep] sampled {len(cause_rows)} cause rows (seed={args.seed}) "
              f"spanning {n_bases} distinct base entities of {len({r['base'] for r in all_cause})} available")
        assert cause_rows, "no cause rows (rule == match_source) found"
        print(f"[{METHOD_NAME}/ceiling_sweep] entity={args.entity} attribute={args.attribute} "
              f"layers={args.layers} sites={args.sites} positions={args.positions} "
              f"cause_rows={len(cause_rows)} iso_rows={len(iso_rows)}")

        def batches(pool):
            for i in range(0, len(pool), args.batch_size):
                yield build_batch(pool[i:i + args.batch_size], entity_assets, adapter, model, processor,
                                   args.positions, batch_cache=batch_cache)

        cause_batches = list(batches(cause_rows))
        iso_batches = list(batches(iso_rows)) if iso_rows else []

        # Unhooked baseline -- independent of site/layer, so computed once.
        def unhooked_rates(bs):
            ms, mb, n = 0.0, 0.0, 0
            for b in bs:
                gen = generate_unhooked(model, b["base_input_ids"], b["attention_mask"], b["base_extra"],
                                         pad_token_id, args.max_new_tokens)
                s, t = score_generation(gen, b)
                k = len(b["rows"])
                ms, mb, n = ms + s * k, mb + t * k, n + k
            return (ms / n, mb / n) if n else (float("nan"), float("nan"))

        base_cause_ms, base_cause_mb = unhooked_rates(cause_batches)
        print(f"  unhooked baseline on cause rows: matches_source={base_cause_ms:.1%} "
              f"matches_base={base_cause_mb:.1%}  (matches_source here is the 'already correct by "
              f"accident' floor -- subtract it mentally when reading cause_ceiling)")

        results = {}
        for site_name in args.sites:
            site = InterventionSite(site_name)
            for layer in args.layers:
                if site_name != "residual" and layer < 1:
                    print(f"  skip {site_name} layer {layer}: MLP sites need layer >= 1")
                    continue
                cs, cb, n = 0.0, 0.0, 0
                for b in cause_batches:
                    gen = full_swap_generation(site, adapter, model, layers_stack, layer, b, pad_token_id,
                                                args.max_new_tokens)
                    s, t = score_generation(gen, b)
                    k = len(b["rows"])
                    cs, cb, n = cs + s * k, cb + t * k, n + k
                cause_ceiling, base_kept = cs / n, cb / n

                iso_floor = float("nan")
                if iso_batches:
                    ib, m = 0.0, 0
                    for b in iso_batches:
                        gen = full_swap_generation(site, adapter, model, layers_stack, layer, b, pad_token_id,
                                                    args.max_new_tokens)
                        _, t = score_generation(gen, b)
                        k = len(b["rows"])
                        ib, m = ib + t * k, m + k
                    iso_floor = ib / m

                results[f"{site_name}/{layer}"] = {
                    "site": site_name, "layer": layer, "cause_ceiling": cause_ceiling,
                    "base_kept": base_kept, "iso_floor": iso_floor,
                    "width": site.width(adapter, model), "n_cause": n,
                }
                print(f"  {site_name:>11} layer {layer:>2}: cause_ceiling={cause_ceiling:6.1%} "
                      f"base_kept={base_kept:6.1%} iso_floor={iso_floor:6.1%}", flush=True)

        print("\n=== ceiling sweep summary (cause_ceiling = upper bound on `cause` for ANY mask) ===")
        print(f"{'site':>11} {'layer':>6} {'width':>7} {'cause_ceiling':>14} {'base_kept':>10} {'iso_floor':>10}")
        for r in results.values():
            print(f"{r['site']:>11} {r['layer']:>6} {r['width']:>7} {r['cause_ceiling']:>13.1%} "
                  f"{r['base_kept']:>9.1%} {r['iso_floor']:>9.1%}")
        live = [r for r in results.values() if r["cause_ceiling"] > max(0.10, base_cause_ms + 0.05)]
        if live:
            best = max(live, key=lambda r: r["cause_ceiling"])
            print(f"\nMost headroom: {best['site']} layer {best['layer']} "
                  f"(cause_ceiling={best['cause_ceiling']:.1%}) -- train there.")
        else:
            print(f"\nNo (site, layer) probed here has meaningful headroom above the "
                  f"{base_cause_ms:.1%} unhooked floor. Training any of them cannot produce a real "
                  f"`cause`; widen --layers or reconsider the site/positions before spending GPU.")

        with open(out_path, "w") as f:
            json.dump({"entity": args.entity, "attribute": args.attribute, "positions": args.positions,
                       "split": args.split, "layers": args.layers, "sites": args.sites,
                       "unhooked_cause_matches_source": base_cause_ms,
                       "unhooked_cause_matches_base": base_cause_mb,
                       "by_site_layer": results}, f, ensure_ascii=False, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
