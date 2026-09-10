"""NDM: train one layer, eval it, score it -- the one-layer contract, as a
thin wrapper over methods/dbm/run_layer.py's run_one_layer (which is itself
train_layer + eval_layer + score.py's score_file). See methods/ndm/config.py
for what NDM is.

Usage:
    python methods/ndm/run_layer.py --entity flags --attribute language --layer 16 \\
        --positions flag_ring1 --site mlp_hidden
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import load_entity_assets  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.dbm.run_layer import run_one_layer  # noqa: E402
from methods.dbm.train import BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS  # noqa: E402
from methods.ndm.config import METHOD_NAME, add_shared_args, resolve_run, source_cache_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared_args(ap)
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=None,
                     help=f"Defaults to methods/dbm/train.py's BATCH_SIZE ({BATCH_SIZE}) -- see "
                          "methods/ndm/train.py --batch_size for the extra memory an mlp_hidden run costs.")
    ap.add_argument("--grad_accum_steps", type=int, default=None,
                     help=f"Defaults to methods/dbm/train.py's GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}).")
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--randomize_positions", action="store_true")
    ap.add_argument("--keep_checkpoint", action="store_true")
    ap.add_argument("--eval_split", default="test", choices=["test", "train"])
    ap.add_argument("--eval_batch_size", type=int, default=16)
    args = ap.parse_args()

    model_slug, pruned, tuples_dir, out_dir, log_dir, site = resolve_run(args)
    log_path = os.path.join(log_dir, f"layer{args.layer}_run.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)

        print(f"[{METHOD_NAME}/run_layer] entity={args.entity} attribute={args.attribute} layer={args.layer} "
              f"site={site.name} l1_coef={args.l1_coef} positions={args.positions} pruned={pruned} -> {out_dir}")

        source_cache = source_cache_for(args, adapter, model, processor, entity_assets, site, args.layer,
                                         model_slug)

        overall = run_one_layer(
            adapter, model, processor, entity_assets, args.attribute, args.layer, out_dir, args.vade_root,
            positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
            temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs,
            batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, cause_only=args.cause_only,
            randomize_positions=args.randomize_positions, tuples_dir=tuples_dir, eval_split=args.eval_split,
            eval_batch_size=args.eval_batch_size, cleanup_checkpoint=not args.keep_checkpoint,
            source_cache=source_cache, site=site, method_label=METHOD_NAME,
        )
        print(f"layer {args.layer} ({site.name}): cause={overall.get('cause_accuracy')}% "
              f"iso_mean={overall.get('iso_mean_accuracy')}% final_score={overall.get('final_score')}%")


if __name__ == "__main__":
    main()
