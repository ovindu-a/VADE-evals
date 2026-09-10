"""Train + evaluate DBM across several layers for one (entity, attribute),
loading the model once and looping layers in-process -- no repeated
model-load per layer. Near-verbatim port of VADE's own
methods/das/layer_sweep.py; see that file's module docstring for the
run_one_layer/run_sweep split rationale (unchanged here).

Usage:
    python methods/dbm/layer_sweep.py --entity flags --attribute currency \\
        --layers 6 10 14 18 22 --positions flag_ring1 --num_epochs 1
"""
import argparse
import gc
import json
import os
import sys
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods.adapters.registry import get_adapter
from methods.common.entities import load_entity_assets, require_pruned_tuples
from methods.common.run_logging import tee_to_log
from methods.common.source_cache import get_or_build_source_cache
from methods.dbm.run_layer import run_one_layer
from methods.dbm.train import (
    GRAD_CLIP_NORM, LR, MIN_LR_RATIO, DBM_L1_COEF, NUM_EPOCHS, TEMP_END, TEMP_START, dbm_logs_dir, dbm_results_dir,
)


def run_sweep(adapter, model, processor, entity_assets, attribute, layers, out_dir, vade_root,
               source_cache_for_layer=None, **run_kwargs):
    """Loops run_one_layer across `layers`, isolating failures so one bad
    layer doesn't take down the rest. Returns {layer: overall_dict_or_None}.

    source_cache_for_layer (optional callable layer -> cache): for methods
    whose source cache is PER-LAYER rather than one file covering every
    layer. DBM's residual cache is the latter (built once up front and
    passed straight through run_kwargs as `source_cache`), so DBM leaves
    this None and nothing changes. NDM's MLP-site cache is the former --
    one file per (site, positions, layer), since MLP internals can't be
    captured for every layer in a single pass -- so it passes a callable
    here instead and the per-layer cache is built lazily inside the loop.
    See common/site_source_cache.py."""
    results = {}
    for layer in layers:
        print(f"\n=== layer {layer} ===", flush=True)
        try:
            layer_kwargs = dict(run_kwargs)
            if source_cache_for_layer is not None:
                layer_kwargs["source_cache"] = source_cache_for_layer(layer)
            overall = run_one_layer(adapter, model, processor, entity_assets, attribute, layer, out_dir, vade_root,
                                     **layer_kwargs)
            results[layer] = overall
            print(f"layer {layer}: cause={overall.get('cause_accuracy')}% "
                  f"iso_mean={overall.get('iso_mean_accuracy')}% final_score={overall.get('final_score')}%")
        except Exception:
            print(f"LAYER {layer} FAILED:")
            traceback.print_exc()
            results[layer] = None
        finally:
            # See VADE's das/layer_sweep.py's identical comment: many layers share one process/CUDA
            # context with no cache-clearing between them -- each layer's intervention/optimizer are
            # locals dropped on return, but the caching allocator holds their memory as "reserved"
            # rather than returning it to the driver, which left unchecked ratchets up across layers
            # until an allocation fails despite nominal free memory existing (fragmentation, not a leak).
            gc.collect()
            torch.cuda.empty_cache()
    return results


def print_summary(layers, results):
    print("\n=== sweep summary ===")
    for layer in layers:
        r = results[layer]
        if r is None:
            print(f"layer {layer}: FAILED")
        else:
            print(f"layer {layer}: cause={r.get('cause_accuracy')}% iso_mean={r.get('iso_mean_accuracy')}% "
                  f"final_score={r.get('final_score')}%")


def write_sweep_summary(out_dir, layers, results, run_config, method_label="DBM"):
    """Writes sweep_layers<tag>_summary.{json,md} to out_dir -- the sweep-level
    counterpart to what score.py's score_file() already writes PER layer
    (layer{L}_predictions_{split}_summary.{json,md}, from run_one_layer's own
    call to it). Without this, comparing layers side by side or picking the
    winning one meant re-reading print_summary's plain-text output back out
    of the log file -- not a structured, easily-reloaded result. run_config:
    a dict of the sweep's own hyperparameters (entity/attribute/positions/
    l1_coef/temperature_start/temperature_end/pruned/eval_split/layers), so
    the summary file is self-describing without needing its own filename or
    the log file to know what produced it.

    Also folds in each scored layer's n_selected/embed_dim/epsilon from its own
    layer{L}_mask_stats.json (written by train.py right after the checkpoint --
    see methods/dbm/intervention.py's mask_stats()), so a layer that ties or
    nearly ties on final_score but selects far fewer dimensions (more sparse,
    arguably more interpretable) is visible here rather than only inside each
    layer's own separate mask_stats.json. Missing file (e.g. a layer trained
    before this existed) is tolerated -- that layer's row just omits it.

    Returns (json_path, md_path)."""
    layers_tag = "-".join(str(l) for l in layers)

    def load_mask_stats(layer):
        path = os.path.join(out_dir, f"layer{layer}_mask_stats.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            stats = json.load(f)
        return {"n_selected": stats["n_selected"], "embed_dim": stats["embed_dim"], "epsilon": stats["epsilon"]}

    by_layer = {
        str(layer): ({"status": "failed"} if results[layer] is None
                      else {"status": "ok", **results[layer], **load_mask_stats(layer)})
        for layer in layers
    }
    scored = [(layer, r) for layer, r in results.items() if r is not None and "final_score" in r]
    best_layer, best_score = (None, None)
    if scored:
        best_layer, best = max(scored, key=lambda lr: lr[1]["final_score"])
        best_score = best["final_score"]

    json_out = {**run_config, "layers": layers, "by_layer": by_layer,
                "best_layer": best_layer, "best_final_score": best_score}
    json_path = os.path.join(out_dir, f"sweep_layers{layers_tag}_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    # `site` appears in run_config only for methods that have one (NDM); DBM's own summary files stay
    # byte-identical to what they were before sites existed.
    site_str = f"site={run_config['site']} " if "site" in run_config else ""
    md_lines = [f"# {method_label} layer sweep -- entity={run_config['entity']} "
                f"attribute={run_config['attribute']}", "",
                f"positions={run_config['positions']} {site_str}l1_coef={run_config['l1_coef']} "
                f"temperature={run_config['temperature_start']}->{run_config['temperature_end']} "
                f"lr={run_config['lr']} min_lr_ratio={run_config.get('min_lr_ratio', MIN_LR_RATIO)} "
                f"grad_clip_norm={run_config.get('grad_clip_norm', GRAD_CLIP_NORM)} "
                f"pruned={run_config['pruned']} eval_split={run_config['eval_split']}", ""]
    if best_layer is not None:
        md_lines.append(f"**best layer: {best_layer} (final_score={best_score}%)**")
        md_lines.append("")
    md_lines += ["| layer | status | cause | iso_mean | final_score | n | dims selected |",
                 "|---|---|---|---|---|---|---|"]
    for layer in layers:
        r = by_layer[str(layer)]
        if r["status"] == "failed":
            md_lines.append(f"| {layer} | FAILED | - | - | - | - | - |")
        else:
            dims = f"{r['n_selected']}/{r['embed_dim']} (eps={r['epsilon']})" if "n_selected" in r else "-"
            md_lines.append(f"| {layer} | ok | {r.get('cause_accuracy', '-')}% | {r.get('iso_mean_accuracy', '-')}% | "
                             f"{r.get('final_score', '-')}% | {r.get('n', '-')} | {dims} |")
    md_path = os.path.join(out_dir, f"sweep_layers{layers_tag}_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return json_path, md_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--l1_coef", type=float, default=DBM_L1_COEF)
    ap.add_argument("--temperature_start", type=float, default=TEMP_START)
    ap.add_argument("--temperature_end", type=float, default=TEMP_END)
    ap.add_argument("--lr", type=float, default=LR, help="See train.py --lr.")
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=None, help="Defaults to train.py's BATCH_SIZE constant.")
    ap.add_argument("--grad_accum_steps", type=int, default=None, help="Defaults to train.py's GRAD_ACCUM_STEPS.")
    ap.add_argument("--min_lr_ratio", type=float, default=MIN_LR_RATIO, help="See train.py --min_lr_ratio.")
    ap.add_argument("--grad_clip_norm", type=float, default=GRAD_CLIP_NORM, help="See train.py --grad_clip_norm.")
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--randomize_positions", action="store_true")
    ap.add_argument("--allow_unpruned", action="store_true",
                     help="See train.py --allow_unpruned -- pruned tuples are required by default here too.")
    ap.add_argument("--keep_checkpoint", action="store_true",
                     help="By default each layer's resume-state checkpoint is deleted once training finishes "
                          "successfully. Pass this if you might later raise --num_epochs on these exact layers.")
    ap.add_argument("--eval_split", default="test", choices=["test", "train"])
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--no_source_cache", action="store_true",
                     help="Disable the per-entity source-activation cache (on by default, built once up front and "
                          "reused across every layer in the sweep -- see common/source_cache.py). This is the "
                          "single biggest win for the cache, since a sweep otherwise redoes the exact same source "
                          "forward passes once per layer. Ignored either way for positions=last_token.")
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    layers_tag = "-".join(str(l) for l in args.layers)
    log_path = os.path.join(dbm_logs_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                          args.temperature_start, args.temperature_end, args.lr,
                                          args.positions, pruned, args.min_lr_ratio, args.grad_clip_norm),
                             f"sweep_layers{layers_tag}.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)
        out_dir = dbm_results_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                   args.temperature_start, args.temperature_end, args.lr,
                                   args.positions, pruned, args.min_lr_ratio, args.grad_clip_norm)

        print(f"[layer_sweep] entity={args.entity} attribute={args.attribute} layers={args.layers} "
              f"l1_coef={args.l1_coef} positions={args.positions} pruned={pruned} -> {out_dir}")

        source_cache = None
        if not args.no_source_cache:
            source_cache = get_or_build_source_cache(adapter, model, processor, entity_assets,
                                                       args.vade_root, model_slug)

        results = run_sweep(
            adapter, model, processor, entity_assets, args.attribute, args.layers, out_dir, args.vade_root,
            positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
            temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps, min_lr_ratio=args.min_lr_ratio,
            grad_clip_norm=args.grad_clip_norm, cause_only=args.cause_only,
            randomize_positions=args.randomize_positions, tuples_dir=tuples_dir,
            eval_split=args.eval_split, eval_batch_size=args.eval_batch_size,
            cleanup_checkpoint=not args.keep_checkpoint, source_cache=source_cache,
        )
        print_summary(args.layers, results)
        write_sweep_summary(out_dir, args.layers, results, run_config={
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "l1_coef": args.l1_coef, "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end, "lr": args.lr, "min_lr_ratio": args.min_lr_ratio,
            "grad_clip_norm": args.grad_clip_norm, "pruned": pruned,
            "eval_split": args.eval_split,
        })


if __name__ == "__main__":
    main()
