"""NDM evaluation CLI -- patches a trained NDM mask at its trained
(layer, site, positions), generates freely, writes predictions in
VADE/eval/score.py's format and grades them with that same scorer. Thin
wrapper over methods/dbm/eval.py's eval_layer; see methods/ndm/config.py for
what NDM is.

Usage:
    python methods/ndm/eval.py --entity flags --attribute language --layer 16 \\
        --positions flag_ring1 --site mlp_hidden --split test
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import load_entity_assets  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.dbm.eval import eval_layer  # noqa: E402
from methods.ndm.config import METHOD_NAME, add_shared_args, resolve_run, source_cache_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared_args(ap)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=None, help="Debug: cap number of eval rows.")
    ap.add_argument("--ckpt_path", default=None,
                     help="Defaults to layer{layer}_intervention.pt under this config's results dir. Every "
                          "hyperparameter above must match the training run, since they derive that path -- and "
                          "--site additionally sets the mask width, so a mismatch fails loudly on load_state_dict "
                          "rather than mis-patching silently.")
    args = ap.parse_args()

    model_slug, pruned, tuples_dir, out_dir, log_dir, site = resolve_run(args)
    log_path = os.path.join(log_dir, f"layer{args.layer}_eval_{args.split}.log")

    with tee_to_log(log_path):
        # Lazy, post-argparse: --vade_root is only known now (see methods/dbm/eval.py's own note).
        sys.path.insert(0, os.path.join(args.vade_root, "eval"))
        from score import score_file  # noqa: E402

        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)

        ckpt_path = args.ckpt_path or os.path.join(out_dir, f"layer{args.layer}_intervention.pt")
        out_path = os.path.join(out_dir, f"layer{args.layer}_predictions_{args.split}.jsonl")

        source_cache = source_cache_for(args, adapter, model, processor, entity_assets, site, args.layer,
                                         model_slug)

        eval_layer(adapter, model, processor, entity_assets, args.attribute, args.layer,
                   site.width(adapter, model), ckpt_path, out_path, positions=args.positions, split=args.split,
                   batch_size=args.batch_size, cause_only=args.cause_only, max_new_tokens=args.max_new_tokens,
                   tuples_dir=tuples_dir, limit_rows=args.limit_rows, source_cache=source_cache,
                   site=site, method_label=METHOD_NAME)

        score_file(out_path, entity=args.entity, tuples_dir=tuples_dir, attribute=args.attribute, split=args.split,
                   method_name=f"layer{args.layer}_predictions_{args.split}", out_dir=out_dir)


if __name__ == "__main__":
    main()
