"""NDM layer sweep: train+eval+score several layers in ONE model load, then
write a cross-layer comparison. Thin wrapper over methods/dbm/layer_sweep.py's
run_sweep/print_summary/write_sweep_summary; see methods/ndm/config.py for
what NDM is.

One real difference from DBM's sweep, and the reason this passes a callable
rather than a cache object: NDM's source-activation cache is per-(site,
positions, LAYER) -- MLP internals can't be captured for every layer in one
pass the way output_hidden_states hands the residual stream over for free
(common/site_source_cache.py explains the shape choice). So the cache is
built lazily inside the loop via run_sweep's source_cache_for_layer hook,
whereas DBM builds one file up front and reuses it for every layer.

The sweep-level summary lands in sweep_layers<tag>_summary.{json,md} next to
the per-layer summary files, with `site` recorded in it -- so the comparison
table says which site produced it and two sites' sweeps can't be confused
after the fact.

Usage:
    python methods/ndm/layer_sweep.py --entity flags --attribute language \\
        --layers 8 10 14 16 20 22 --positions flag_ring1 --site mlp_hidden
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import load_entity_assets  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.dbm.layer_sweep import print_summary, run_sweep, write_sweep_summary  # noqa: E402
from methods.dbm.train import BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS  # noqa: E402
from methods.ndm.config import METHOD_NAME, add_shared_args, resolve_run, source_cache_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared_args(ap)
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                     help="Layers to sweep. Each must be >=1 (layer L addresses block L-1's MLP).")
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=None,
                     help=f"Defaults to methods/dbm/train.py's BATCH_SIZE ({BATCH_SIZE}).")
    ap.add_argument("--grad_accum_steps", type=int, default=None,
                     help=f"Defaults to methods/dbm/train.py's GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}).")
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--randomize_positions", action="store_true")
    ap.add_argument("--keep_checkpoint", action="store_true")
    ap.add_argument("--eval_split", default="test", choices=["test", "train"])
    ap.add_argument("--eval_batch_size", type=int, default=16)
    args = ap.parse_args()

    # add_shared_args supplies a single required --layer, which a sweep replaces with --layers. Bind
    # args.layer to the first swept layer purely so resolve_run's shared path/site resolution (and its
    # layer>=1 assertion) work unchanged -- the results dir is per-config, not per-layer, so which
    # layer it sees does not affect out_dir.
    assert all(l >= 1 for l in args.layers), (
        f"--layers must all be >=1 for an NDM site (layer L addresses block L-1's MLP; L=0 is the "
        f"embedding output, which has no MLP). Got {args.layers}.")
    args.layer = args.layers[0]

    model_slug, pruned, tuples_dir, out_dir, log_dir, site = resolve_run(args)
    layers_tag = "-".join(str(l) for l in args.layers)
    log_path = os.path.join(log_dir, f"sweep_layers{layers_tag}.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)

        print(f"[{METHOD_NAME}/layer_sweep] entity={args.entity} attribute={args.attribute} "
              f"layers={args.layers} site={site.name} l1_coef={args.l1_coef} positions={args.positions} "
              f"pruned={pruned} -> {out_dir}")

        results = run_sweep(
            adapter, model, processor, entity_assets, args.attribute, args.layers, out_dir, args.vade_root,
            source_cache_for_layer=lambda layer: source_cache_for(
                args, adapter, model, processor, entity_assets, site, layer, model_slug),
            positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
            temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs,
            batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, cause_only=args.cause_only,
            randomize_positions=args.randomize_positions, tuples_dir=tuples_dir,
            eval_split=args.eval_split, eval_batch_size=args.eval_batch_size,
            cleanup_checkpoint=not args.keep_checkpoint, site=site, method_label=METHOD_NAME,
        )
        print_summary(args.layers, results)
        write_sweep_summary(out_dir, args.layers, results, run_config={
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "site": site.name,
            "l1_coef": args.l1_coef, "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end, "lr": args.lr, "pruned": pruned,
            "eval_split": args.eval_split,
        }, method_label=METHOD_NAME.upper())


if __name__ == "__main__":
    main()
