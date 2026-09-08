"""Small, ad hoc logit-lens probe for Qwen2.5-VL on real VADE prompts --
same spirit as methods/attention_maps.py (a handful of example rows, not a
sweep): "at which decoder layer does the correct answer already dominate
the model's own vocabulary-space prediction."

At each layer L (0 = embedding output .. n_layers = the real output, using
the forward pass's own out.logits there -- see below) and each of the up
to MAX_ANSWER_TOKENS teacher-forced answer positions, projects that
layer's residual stream through the model's own final norm + lm_head
(adapters.qwen2_5_vl.Qwen25VLAdapter.unembed) and records: the gold
token's rank in that layer's predicted distribution, its probability, and
the layer's actual top-1 token. Aggregated into a "when does the answer
emerge" curve (top1-match-rate and mean gold-rank per layer).

Why the FINAL layer (L == n_layers) uses out.logits directly instead of
calling unembed() on hidden_states[-1]: this project's installed
transformers ties hidden_states[-1] to last_hidden_state (the POST-final-
norm residual stream -- see transformers.utils.output_capturing's
capture_outputs(tie_last_hidden_states=True), traced through this exact
version's Qwen2_5_VLTextModel.forward), so calling unembed() there would
apply the final RMSNorm a second time and silently corrupt the result.
Every layer BEFORE the last is the raw (pre-final-norm) decoder-layer
output, which is exactly what unembed() expects. Using out.logits for the
last layer sidesteps needing that tied/untied distinction to be correct at
all -- it's definitionally the model's real prediction, not a
recomputation of it.

Deliberately built on methods/probe_common.py's real question+prefill
machinery, not sae.py's dummy-question activations -- see that module's
docstring for why.

Usage:
    python methods/logit_lens.py --entity flags --attribute language --limit 5
    python methods/logit_lens.py --entity flags --attribute language --limit 5 --layers 0 4 10 14 18 24 28
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch

from methods.adapters.registry import get_adapter
from methods.common.entities import load_entity_assets, require_pruned_tuples
from methods.common.targets import MAX_ANSWER_TOKENS
from methods.probe_common import build_probe_batch, load_probe_rows, teacher_forced_forward


def score_rows(model, processor, adapter, entity_assets, rows, positions, layers=None, batch_cache=None):
    """Runs each row through the model ONE AT A TIME (consistent with
    attention_maps.py -- an ad hoc probe over a handful of rows, not worth
    batching). Returns (per_row_records, summary):
      per_row_records: list of {attribute, row_index, base, template_id,
        gold_label, gold_len, per_layer: [{layer, positions: [{j, gold_id,
        gold_rank, gold_prob, top1_id, top1_text, top1_prob}, ...]}, ...]}
      summary: {layer: {n, top1_match_rate, mean_gold_rank}} -- averaged
        over every valid (row, answer-position) pair at that layer.
    layers: defaults to EVERY layer (0..n_layers inclusive) -- logit lens
    is cheap (one norm+lm_head matmul per layer), no reason to subsample
    the way attention_maps.py's --layers does for eager-attention memory.
    """
    n_layers = len(adapter.get_decoder_layers(model))
    if layers is None:
        layers = list(range(n_layers + 1))
    assert all(0 <= l <= n_layers for l in layers), \
        f"--layers must be in [0, {n_layers}] for this model (n_layers={n_layers}), got {layers}"

    per_row_records = []
    agg = {l: {"top1_hits": 0, "n_valid": 0, "rank_sum": 0} for l in layers}

    for row in rows:
        batch = build_probe_batch([row], entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
        out = teacher_forced_forward(model, batch, output_hidden_states=True, logits_to_keep=MAX_ANSWER_TOKENS)
        gold_toks = batch["base_gold_toks"][0]      # [K]
        gold_len = int(batch["base_gold_len"][0].item())
        readout_start = batch["readout_start_col"]

        row_record = {
            "attribute": row["target_attribute"], "row_index": row["row_index"], "base": row["base"],
            "template_id": row["template_id"], "gold_label": row["base_label"], "gold_len": gold_len,
            "per_layer": [],
        }

        for l in layers:
            if l == n_layers:
                logits_lk = out.logits[0].float()  # [K, V] -- logits_to_keep=K already truncates to this window
            else:
                hs_lk = out.hidden_states[l][0, readout_start:readout_start + MAX_ANSWER_TOKENS, :]  # [K, H]
                logits_lk = adapter.unembed(model, hs_lk).float()  # [K, V]
            probs = torch.softmax(logits_lk, dim=-1)

            layer_positions = []
            for j in range(gold_len):
                gold_id = int(gold_toks[j].item())
                logit_row = logits_lk[j]
                rank = int((logit_row > logit_row[gold_id]).sum().item()) + 1  # 1 = top prediction
                top1_id = int(logit_row.argmax().item())
                layer_positions.append({
                    "j": j, "gold_id": gold_id, "gold_rank": rank, "gold_prob": probs[j, gold_id].item(),
                    "top1_id": top1_id, "top1_text": processor.tokenizer.decode([top1_id]),
                    "top1_prob": probs[j, top1_id].item(),
                })
                agg[l]["n_valid"] += 1
                agg[l]["rank_sum"] += rank
                agg[l]["top1_hits"] += int(top1_id == gold_id)
            row_record["per_layer"].append({"layer": l, "positions": layer_positions})

        per_row_records.append(row_record)

    summary = {
        l: {
            "n": agg[l]["n_valid"],
            "top1_match_rate": agg[l]["top1_hits"] / agg[l]["n_valid"] if agg[l]["n_valid"] else None,
            "mean_gold_rank": agg[l]["rank_sum"] / agg[l]["n_valid"] if agg[l]["n_valid"] else None,
        }
        for l in layers
    }
    return per_row_records, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attribute", default="language")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--limit", type=int, default=5, help="Number of example rows -- this is an ad hoc "
                     "probe, not a sweep; keep this small.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Decoder layer indices to report (0=embedding output .. n_layers=the real "
                          "output; 28 for Qwen2.5-VL-7B). Defaults to EVERY layer -- unlike "
                          "attention_maps.py's --layers, this is cheap enough not to subsample.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Validate rows/positions resolve without "
                     "loading the 7B model -- no GPU needed.")
    ap.add_argument("--out", default=None, help="Defaults to methods/logit_lens/"
                     "<entity>_<attribute>_<positions>_report.jsonl (per-row) + _summary.json")
    args = ap.parse_args()

    entity_assets = load_entity_assets(args.vade_root, args.entity)
    model_slug = args.model_id.split("/")[-1]
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if not args.allow_unpruned else None)
    rows = load_probe_rows(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir, limit=args.limit,
                            one_per_image=True)
    print(f"[logit_lens] entity={args.entity} attribute={args.attribute} positions={args.positions}: "
          f"{len(rows)} example row(s)")
    for r in rows:
        print(f"  row_index={r['row_index']} base={r['base']} template={r['template_id']} "
              f"gold={r['base_label']!r}")

    if args.dry_run:
        from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
        from methods.common.entities import resolve_position_set
        from transformers import AutoConfig

        class _ConfigOnlyModel:
            def __init__(self, config):
                self.config = config

        adapter = Qwen25VLAdapter(args.model_id)
        shim = _ConfigOnlyModel(AutoConfig.from_pretrained(args.model_id))
        flat_indices, n_image_tokens, _is_last_token = resolve_position_set(args.positions, entity_assets, adapter, shim)
        for r in rows:
            img_path = os.path.join(entity_assets.entity_dir, entity_assets.items[r["base"]]["image"])
            assert os.path.exists(img_path), f"missing image {img_path}"
        n_obj = len(flat_indices) if flat_indices is not None else "last_token"
        print(f"[dry_run] OK -- positions={args.positions!r} resolves to {n_obj} of {n_image_tokens} "
              f"image tokens; all {len(rows)} row images found on disk. "
              f"(Model not loaded -- rerun without --dry_run, on a GPU box, to actually run the logit lens.)")
        return

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()  # default sdpa/flash -- no eager attention needed for this probe

    per_row_records, summary = score_rows(model, processor, adapter, entity_assets, rows, args.positions, args.layers)

    print("\n=== per-layer curve (averaged over all rows and valid answer positions) ===")
    for l in sorted(summary):
        s = summary[l]
        if s["n"] == 0:
            continue
        print(f"layer {l:2d}: top1_match_rate={s['top1_match_rate']:.3f}  mean_gold_rank={s['mean_gold_rank']:.1f}"
              f"  (n={s['n']})")

    out_stem = args.out or os.path.join(REPO_ROOT, "methods", "logit_lens",
                                         f"{args.entity}_{args.attribute}_{args.positions}_report")
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    with open(out_stem + ".jsonl", "w") as f:
        for rec in per_row_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_stem + "_summary.json", "w") as f:
        json.dump({
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "n_rows": len(rows), "per_layer": summary,
        }, f, indent=2)
    print(f"\nwrote {out_stem}.jsonl and {out_stem}_summary.json")


if __name__ == "__main__":
    main()
