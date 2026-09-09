"""DBM evaluation: patches a trained SigmoidMaskIntervention checkpoint at
its trained layer/positions, runs free-running generation, writes
predictions in VADE/eval/score.py's exact format, then calls that same
score_file() to grade them -- no DBM-specific summarizer, identical to how
VADE's own methods/das/eval.py (this file's near-verbatim template) is
graded. score.py itself is imported from --vade_root (a sibling VADE
checkout) at runtime, not vendored here -- see module docstring of
methods/dbm/train.py for why this repo's own results/logs stay separate
from VADE's data/tuples/source-cache.

Usage:
    python methods/dbm/eval.py --entity flags --attribute capital --layer 14 \\
        --positions flag_ring1 --split test
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch

from methods.adapters.registry import get_adapter
from methods.common.entities import BuildBatchCache, build_batch, load_entity_assets, load_tuples, require_pruned_tuples
from methods.common.hooks import cache_layer_hidden, generate_patched, make_cache_aware_patch_hook
from methods.common.run_logging import tee_to_log
from methods.common.source_cache import get_or_build_source_cache, lookup_source_hidden
from methods.dbm.intervention import SigmoidMaskIntervention
from methods.dbm.train import DBM_L1_COEF, TEMP_END, TEMP_START, dbm_logs_dir, dbm_results_dir


def eval_layer(adapter, model, processor, entity_assets, attribute, layer, hidden_size, ckpt_path, out_path,
               positions="flag_ring1", split="test", batch_size=16, cause_only=False, max_new_tokens=16,
               tuples_dir=None, limit_rows=None, source_cache=None):
    """Writes predictions to out_path (score.py format), resumable -- skips
    (attribute, row_index) pairs already present. Returns out_path.

    source_cache (optional): see train_layer's docstring / common/source_cache.py."""
    layers = adapter.get_decoder_layers(model)
    batch_cache = BuildBatchCache()
    intervention = SigmoidMaskIntervention(embed_dim=hidden_size).to(model.device)
    intervention.load_state_dict(torch.load(ckpt_path, map_location=model.device))
    intervention.eval()
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    rows = load_tuples(entity_assets, attribute, split, tuples_dir=tuples_dir)
    for r in rows:
        r.setdefault("target_attribute", attribute)
    if cause_only:
        rows = [r for r in rows if r["queried"] == r["target_attribute"]]
    if limit_rows:
        rows = rows[:limit_rows]

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line)
                done.add((rec["attribute"], rec["row_index"]))
            except Exception:
                continue
    todo = [r for r in rows if (r["target_attribute"], r["row_index"]) not in done]
    print(f"[dbm/eval] entity={entity_assets.entity} attribute={attribute} layer={layer} "
          f"positions={positions} split={split}: {len(todo)} rows remaining of {len(rows)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_f = open(out_path, "a")
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch_rows = todo[bi * batch_size:(bi + 1) * batch_size]
        batch = build_batch(batch_rows, entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
        if source_cache is not None and not batch["is_last_token"]:
            source_hidden = lookup_source_hidden(source_cache, batch, layer, model.device, model.dtype)
        else:
            source_hidden = cache_layer_hidden(model, batch["source_input_ids"], batch["attention_mask"],
                                                batch["source_extra"], batch["positions"], layer)
        patch_fn = make_cache_aware_patch_hook(batch["positions"], lambda base_vals: intervention(base_vals, source_hidden))
        gen_toks = generate_patched(model, layers, layer, patch_fn, batch["base_input_ids"], batch["attention_mask"],
                                     batch["base_extra"], pad_token_id, max_new_tokens)
        for i, row in enumerate(batch["rows"]):
            text = processor.tokenizer.decode(gen_toks[i].tolist(), skip_special_tokens=True)
            rec = {"attribute": row["target_attribute"], "row_index": row["row_index"], "generated_text": text}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        if (bi + 1) % 20 == 0:
            print(f"  batch {bi + 1}/{n_batches}", flush=True)
    out_f.close()
    print(f"DONE. wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--l1_coef", type=float, default=DBM_L1_COEF, help="Must match the value train.py was run with "
                                                                         "-- only used to derive the default out_dir.")
    ap.add_argument("--temperature_start", type=float, default=TEMP_START,
                     help="Must match the value train.py was run with -- only used to derive the default out_dir.")
    ap.add_argument("--temperature_end", type=float, default=TEMP_END,
                     help="Must match the value train.py was run with -- only used to derive the default out_dir.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--allow_unpruned", action="store_true",
                     help="See train.py --allow_unpruned -- pruned tuples are required by default here too.")
    ap.add_argument("--ckpt_path", default=None, help="Defaults to layer{layer}_intervention.pt under the "
                                                        "standard results dir for this config.")
    ap.add_argument("--limit_rows", type=int, default=None, help="Debug: cap number of eval rows")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no_source_cache", action="store_true",
                     help="See train.py --no_source_cache -- same cache, shared across train/eval.")
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_path = os.path.join(dbm_logs_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                          args.temperature_start, args.temperature_end, args.positions, pruned),
                             f"layer{args.layer}_eval_{args.split}.log")

    with tee_to_log(log_path):
        # Imported lazily (after argparse) because --vade_root is only known now -- unlike VADE's own
        # das/eval.py, this repo's vade_root is a configurable sibling path, not a fixed same-repo one.
        sys.path.insert(0, os.path.join(args.vade_root, "eval"))
        from score import score_file  # noqa: E402 -- VADE's own shared scorer, no per-method reimplementation

        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()

        entity_assets = load_entity_assets(args.vade_root, args.entity)

        out_dir = args.out_dir or dbm_results_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                                    args.temperature_start, args.temperature_end,
                                                    args.positions, pruned)
        ckpt_path = args.ckpt_path or os.path.join(out_dir, f"layer{args.layer}_intervention.pt")
        out_path = os.path.join(out_dir, f"layer{args.layer}_predictions_{args.split}.jsonl")

        source_cache = None
        if not args.no_source_cache:
            source_cache = get_or_build_source_cache(adapter, model, processor, entity_assets,
                                                       args.vade_root, model_slug)

        eval_layer(adapter, model, processor, entity_assets, args.attribute, args.layer, adapter.hidden_size(model),
                   ckpt_path, out_path, positions=args.positions, split=args.split, batch_size=args.batch_size,
                   cause_only=args.cause_only, max_new_tokens=args.max_new_tokens, tuples_dir=tuples_dir,
                   limit_rows=args.limit_rows, source_cache=source_cache)

        score_file(out_path, entity=args.entity, tuples_dir=tuples_dir, attribute=args.attribute, split=args.split,
                   method_name=f"layer{args.layer}_predictions_{args.split}", out_dir=out_dir)


if __name__ == "__main__":
    main()
