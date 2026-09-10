"""NDM (Native Dictionary Masking) training CLI -- learns a sigmoid mask over
a decoder block's MLP hidden state (the post-SwiGLU neuron vector) instead of
over the residual stream. See methods/ndm/config.py's module docstring for
what NDM is and why it isn't just DBM at another site.

This file is deliberately thin: the training loop is methods/dbm/train.py's
train_layer, parameterized by a common/sites.py InterventionSite. Nothing
about the loop, the L1 term, the temperature anneal, checkpoint/resume or
mask_stats is site-specific.

Usage:
    python methods/ndm/train.py --entity flags --attribute language --layer 16 \\
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
from methods.dbm.train import BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, train_layer  # noqa: E402
from methods.ndm.config import METHOD_NAME, add_shared_args, resolve_run, source_cache_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared_args(ap)
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                     help=f"Default {BATCH_SIZE}, inherited from DBM/DAS. NOTE: an mlp_hidden run holds a mask and "
                          f"source-activation slices 5.3x wider than DBM's, and takes an EXTRA source forward pass "
                          f"per micro-batch (no source cache -- see --no_source_cache), so watch VRAM before "
                          f"raising this.")
    ap.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--randomize_positions", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=None, help="Debug/smoke-test: cap number of training rows.")
    ap.add_argument("--keep_checkpoint", action="store_true", help="See methods/dbm/train.py --keep_checkpoint.")
    args = ap.parse_args()

    model_slug, pruned, tuples_dir, out_dir, log_dir, site = resolve_run(args)
    log_path = os.path.join(log_dir, f"layer{args.layer}_train.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)

        print(f"[{METHOD_NAME}/train] site={site.name} -> {out_dir}")

        # NDM's own cache, not the one DBM/DAS share -- see methods/ndm/config.py's source_cache_for
        # and common/site_source_cache.py.
        source_cache = source_cache_for(args, adapter, model, processor, entity_assets, site, args.layer,
                                         model_slug)

        train_layer(adapter, model, processor, entity_assets, args.attribute, args.layer, out_dir,
                    positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
                    temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs,
                    batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
                    cause_only=args.cause_only, randomize_positions=args.randomize_positions,
                    limit_rows=args.limit_rows, tuples_dir=tuples_dir,
                    cleanup_checkpoint=not args.keep_checkpoint, source_cache=source_cache,
                    site=site, method_label=METHOD_NAME)


if __name__ == "__main__":
    main()
