"""Train one DBM layer, then immediately eval it, then score it -- the
one-layer contract methods/dbm/layer_sweep.py's per-layer loop and a single
ad-hoc layer run both share. Near-verbatim port of VADE's own
methods/das/run_layer.py (see that file's docstring for the split
rationale); only the intervention-specific args differ (--l1_coef/
--temperature_* instead of --subspace_dim).

Usage:
    python methods/dbm/run_layer.py --entity flags --attribute capital --layer 14 \\
        --positions flag_ring1
"""
import argparse
import gc
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods.adapters.registry import get_adapter
from methods.common.entities import load_entity_assets, require_pruned_tuples
from methods.common.run_logging import tee_to_log
from methods.common.sites import RESIDUAL_SITE
from methods.common.source_cache import get_or_build_source_cache
from methods.dbm.eval import eval_layer
from methods.dbm.train import (
    GRAD_CLIP_NORM, LR, MIN_LR_RATIO, DBM_L1_COEF, NUM_EPOCHS, TEMP_END, TEMP_START, dbm_logs_dir, dbm_results_dir,
    train_layer,
)


def run_one_layer(adapter, model, processor, entity_assets, attribute, layer, out_dir, vade_root,
                   positions="flag_ring1", l1_coef=DBM_L1_COEF, temperature_start=TEMP_START,
                   temperature_end=TEMP_END, lr=LR, num_epochs=NUM_EPOCHS, batch_size=None, grad_accum_steps=None,
                   min_lr_ratio=MIN_LR_RATIO, grad_clip_norm=GRAD_CLIP_NORM,
                   cause_only=False, randomize_positions=False, tuples_dir=None,
                   eval_split="test", eval_batch_size=16, cleanup_checkpoint=True, source_cache=None,
                   site=None, method_label="dbm"):
    """Train layer, then immediately eval it, then score. Raises on failure
    (no try/except here -- that's the sweep loop's job, see layer_sweep.py's
    run_sweep). Returns score.py's `overall` dict for this attribute
    (cause_accuracy/iso_mean_accuracy/final_score/...).

    See methods/das/run_layer.py's run_one_layer -- identical shape, only
    the intervention-specific kwargs (l1_coef/temperature_*) differ from
    DAS's subspace_dim. vade_root: needed here (unlike DAS's version) only
    to locate score.py -- see eval.py's module docstring for why this
    import can't happen at module load time the way DAS's does."""
    from methods.dbm.train import BATCH_SIZE, GRAD_ACCUM_STEPS
    batch_size = batch_size or BATCH_SIZE
    grad_accum_steps = grad_accum_steps or GRAD_ACCUM_STEPS
    site = site or RESIDUAL_SITE

    ckpt_path = train_layer(
        adapter, model, processor, entity_assets, attribute, layer, out_dir,
        positions=positions, l1_coef=l1_coef, temperature_start=temperature_start, temperature_end=temperature_end,
        lr=lr, num_epochs=num_epochs, batch_size=batch_size, grad_accum_steps=grad_accum_steps,
        min_lr_ratio=min_lr_ratio, grad_clip_norm=grad_clip_norm,
        cause_only=cause_only, randomize_positions=randomize_positions, tuples_dir=tuples_dir,
        cleanup_checkpoint=cleanup_checkpoint, source_cache=source_cache, site=site, method_label=method_label,
    )
    # See VADE's das/run_layer.py's identical comment: training's/eval's allocation regimes differ
    # (backprop activations vs. larger-batch no-grad generation) -- clear the caching allocator's
    # reserved-but-unused blocks between them so eval can request cleanly instead of fragmenting.
    gc.collect()
    torch.cuda.empty_cache()

    predictions_path = os.path.join(out_dir, f"layer{layer}_predictions_{eval_split}.jsonl")
    eval_layer(
        adapter, model, processor, entity_assets, attribute, layer, site.width(adapter, model), ckpt_path,
        predictions_path, positions=positions, split=eval_split, batch_size=eval_batch_size, tuples_dir=tuples_dir,
        source_cache=source_cache, site=site, method_label=method_label,
    )

    sys.path.insert(0, os.path.join(vade_root, "eval"))
    from score import score_file  # noqa: E402
    _, _, by_target_attr = score_file(
        predictions_path, entity=entity_assets.entity, tuples_dir=tuples_dir, attribute=attribute, split=eval_split,
        method_name=f"layer{layer}_predictions_{eval_split}", out_dir=out_dir,
    )
    return by_target_attr[attribute][1]  # overall dict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layer", type=int, required=True)
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
                          "successfully. Pass this if you might later raise --num_epochs on this exact run.")
    ap.add_argument("--eval_split", default="test", choices=["test", "train"])
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--no_source_cache", action="store_true",
                     help="See train.py --no_source_cache -- reused across every future run on this entity.")
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_path = os.path.join(dbm_logs_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                          args.temperature_start, args.temperature_end, args.lr,
                                          args.positions, pruned, args.min_lr_ratio, args.grad_clip_norm),
                             f"layer{args.layer}_run.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)
        out_dir = dbm_results_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                   args.temperature_start, args.temperature_end, args.lr,
                                   args.positions, pruned, args.min_lr_ratio, args.grad_clip_norm)

        print(f"[run_layer] entity={args.entity} attribute={args.attribute} layer={args.layer} "
              f"l1_coef={args.l1_coef} positions={args.positions} pruned={pruned} -> {out_dir}")

        source_cache = None
        if not args.no_source_cache:
            source_cache = get_or_build_source_cache(adapter, model, processor, entity_assets,
                                                       args.vade_root, model_slug)

        overall = run_one_layer(
            adapter, model, processor, entity_assets, args.attribute, args.layer, out_dir, args.vade_root,
            positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
            temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs, batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps, min_lr_ratio=args.min_lr_ratio,
            grad_clip_norm=args.grad_clip_norm, cause_only=args.cause_only,
            randomize_positions=args.randomize_positions, tuples_dir=tuples_dir,
            eval_split=args.eval_split, eval_batch_size=args.eval_batch_size,
            cleanup_checkpoint=not args.keep_checkpoint, source_cache=source_cache,
        )
        print(f"layer {args.layer}: cause={overall.get('cause_accuracy')}% "
              f"iso_mean={overall.get('iso_mean_accuracy')}% final_score={overall.get('final_score')}%")


if __name__ == "__main__":
    main()
