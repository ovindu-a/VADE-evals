"""Small, ad hoc logit-lens probe for Qwen2.5-VL on real VADE prompts --
same spirit as methods/attention_maps.py (a handful of example rows, not a
sweep): "at which decoder layer does the correct answer already dominate
the model's own vocabulary-space prediction."

At each layer L (0 = embedding output .. n_layers = the real output, using
the forward pass's own out.logits there -- see below) and each of the up
to MAX_ANSWER_TOKENS teacher-forced answer positions, projects that
layer's residual stream through the model's own final norm + lm_head
(adapters.qwen2_5_vl.Qwen25VLAdapter.unembed) and records: the gold
token's rank in that layer's predicted distribution, its probability, the
layer's top-K predicted tokens (--top_k, default 5), and -- unless
--skip_other_attributes -- where every OTHER scored attribute's own
ground-truth value would rank at that SAME layer/position (see
probe_common.other_attribute_gold_toks) -- "is the model already carrying
capital/currency/calling_code information at this position even though
only language was asked, or does only the queried attribute ever surface."
Aggregated into a "when does the answer emerge" curve (top1-match-rate and
mean gold-rank per layer), plus a parallel curve per other attribute.

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

score_one_row() takes an ALREADY-COMPUTED forward pass -- see
methods/mech_probe.py, which runs logit_lens.py and attention_maps.py's
scorers off a SINGLE forward pass per row (output_hidden_states=True AND
output_attentions=True together) instead of the two separate passes this
script's own standalone score_rows()/main() still does when run alone.

Usage:
    python methods/logit_lens.py --entity flags --attribute language --limit 12 --questions_per_image 6
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
from methods.probe_common import build_probe_batch, load_probe_rows, other_attribute_gold_toks, teacher_forced_forward

DEFAULT_TOP_K = 5


def _layer_logits(model, adapter, out, layer, n_layers, readout_start):
    """[K, V] float logits at layer `layer`'s residual stream, read out at columns
    [readout_start : readout_start+K] -- see module docstring for why layer==n_layers
    is special-cased onto out.logits rather than unembed(hidden_states[-1])."""
    if layer == n_layers:
        return out.logits[0].float()  # logits_to_keep=K already truncates to this exact window
    hs_lk = out.hidden_states[layer][0, readout_start:readout_start + MAX_ANSWER_TOKENS, :]  # [K, H]
    return adapter.unembed(model, hs_lk).float()


def score_one_row(model, processor, adapter, entity_assets, row, batch, out, layers, n_layers,
                   top_k=DEFAULT_TOP_K, track_other_attributes=True):
    """Scores ONE row against an already-computed forward pass `out` (from
    teacher_forced_forward(..., output_hidden_states=True, logits_to_keep=MAX_ANSWER_TOKENS) over
    `batch` -- see probe_common.build_probe_batch). Pure w.r.t. `out`: does no forwarding itself, so
    methods/mech_probe.py can call this against a forward pass it's ALSO handing to
    attention_maps.score_one_row, instead of forwarding twice.

    Returns (row_record, layer_deltas):
      row_record: {attribute, row_index, base, template_id, gold_label, gold_len,
        per_layer: [{layer, positions: [{j, gold_id, gold_rank, gold_prob,
          top_k: [{token_id, text, prob}, ...] sorted by rank}, ...],
          other_attributes: {attr: {gold_rank, gold_prob}} at j=0 only -- see module docstring}]}
      layer_deltas: {layer: {"top1_hits": int, "n_valid": int, "rank_sum": int,
        "other": {attr: {"rank_sum": int, "n": int}}}} -- accumulator deltas from JUST this row, for
        the caller to fold into a running aggregate (see score_rows below).
    """
    gold_toks = batch["base_gold_toks"][0]      # [K]
    gold_len = int(batch["base_gold_len"][0].item())
    readout_start = batch["readout_start_col"]

    other_gold = other_attribute_gold_toks(entity_assets, processor.tokenizer, row) if track_other_attributes else {}

    row_record = {
        "attribute": row["target_attribute"], "row_index": row["row_index"], "base": row["base"],
        "template_id": row["template_id"], "gold_label": row["base_label"], "gold_len": gold_len,
        "per_layer": [],
    }
    layer_deltas = {l: {"top1_hits": 0, "n_valid": 0, "rank_sum": 0,
                         "other": {a: {"rank_sum": 0, "n": 0} for a in other_gold}} for l in layers}

    for l in layers:
        logits_lk = _layer_logits(model, adapter, out, l, n_layers, readout_start)
        probs = torch.softmax(logits_lk, dim=-1)

        layer_positions = []
        for j in range(gold_len):
            gold_id = int(gold_toks[j].item())
            logit_row = logits_lk[j]
            rank = int((logit_row > logit_row[gold_id]).sum().item()) + 1  # 1 = top prediction
            _, top_ids = torch.topk(logit_row, k=top_k)
            top_k_entries = [
                {"token_id": int(tid), "text": processor.tokenizer.decode([int(tid)]), "prob": probs[j, tid].item()}
                for tid in top_ids.tolist()
            ]
            layer_positions.append({
                "j": j, "gold_id": gold_id, "gold_rank": rank, "gold_prob": probs[j, gold_id].item(),
                "top_k": top_k_entries,
            })
            layer_deltas[l]["n_valid"] += 1
            layer_deltas[l]["rank_sum"] += rank
            layer_deltas[l]["top1_hits"] += int(top_k_entries[0]["token_id"] == gold_id)

        other_attrs_out = {}
        if gold_len > 0:  # other-attribute ranks only computed at j=0, same convention as the main gold token
            for attr, other_ids in other_gold.items():
                other_id0 = int(other_ids[0])
                rank0 = int((logits_lk[0] > logits_lk[0, other_id0]).sum().item()) + 1
                other_attrs_out[attr] = {"gold_rank": rank0, "gold_prob": probs[0, other_id0].item()}
                layer_deltas[l]["other"][attr]["rank_sum"] += rank0
                layer_deltas[l]["other"][attr]["n"] += 1

        row_record["per_layer"].append({"layer": l, "positions": layer_positions, "other_attributes": other_attrs_out})

    return row_record, layer_deltas


def score_rows(model, processor, adapter, entity_assets, rows, positions, layers=None, top_k=DEFAULT_TOP_K,
               track_other_attributes=True, batch_cache=None):
    """Standalone-script entry point: builds the batch and runs the forward pass itself (one row at a
    time), then calls score_one_row. See mech_probe.py for the single-forward-pass version that
    shares its `out` with attention_maps.py's scorer instead.

    Returns (per_row_records, summary) -- summary: {layer: {n, top1_match_rate, mean_gold_rank,
      other_attributes: {attr: {n, mean_gold_rank}}}}, averaged over every valid (row, position) pair
    (other_attributes: over every row that had that attribute registered, at j=0 only).

    layers: defaults to EVERY layer (0..n_layers inclusive) -- logit lens is cheap (one norm+lm_head
    matmul per layer), no reason to subsample the way attention_maps.py's --layers does for eager-
    attention memory.
    """
    n_layers = len(adapter.get_decoder_layers(model))
    if layers is None:
        layers = list(range(n_layers + 1))
    assert all(0 <= l <= n_layers for l in layers), \
        f"--layers must be in [0, {n_layers}] for this model (n_layers={n_layers}), got {layers}"

    per_row_records = []
    agg = init_agg(layers)

    for row in rows:
        batch = build_probe_batch([row], entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
        out = teacher_forced_forward(model, batch, output_hidden_states=True, logits_to_keep=MAX_ANSWER_TOKENS)
        row_record, layer_deltas = score_one_row(model, processor, adapter, entity_assets, row, batch, out, layers,
                                                   n_layers, top_k=top_k, track_other_attributes=track_other_attributes)
        per_row_records.append(row_record)
        fold_deltas(agg, layers, layer_deltas)

    return per_row_records, finalize_summary(agg, layers)


def init_agg(layers):
    """A fresh per-layer accumulator for fold_deltas() -- one instance per run, shared across every
    row via fold_deltas (score_rows above, or mech_probe.py's merged loop)."""
    return {l: {"top1_hits": 0, "n_valid": 0, "rank_sum": 0, "other": {}} for l in layers}


def fold_deltas(agg, layers, layer_deltas):
    """Adds one row's layer_deltas (from score_one_row) into a running agg (from init_agg) IN PLACE
    -- factored out of score_rows so mech_probe.py's merged per-row loop can call this without
    duplicating the accumulation logic."""
    for l in layers:
        d = layer_deltas[l]
        agg[l]["top1_hits"] += d["top1_hits"]
        agg[l]["n_valid"] += d["n_valid"]
        agg[l]["rank_sum"] += d["rank_sum"]
        for attr, od in d["other"].items():
            a = agg[l]["other"].setdefault(attr, {"rank_sum": 0, "n": 0})
            a["rank_sum"] += od["rank_sum"]
            a["n"] += od["n"]


def finalize_summary(agg, layers):
    """agg (from init_agg, folded via fold_deltas) -> the final {layer: {n, top1_match_rate,
    mean_gold_rank, other_attributes}} summary dict."""
    return {
        l: {
            "n": agg[l]["n_valid"],
            "top1_match_rate": agg[l]["top1_hits"] / agg[l]["n_valid"] if agg[l]["n_valid"] else None,
            "mean_gold_rank": agg[l]["rank_sum"] / agg[l]["n_valid"] if agg[l]["n_valid"] else None,
            "other_attributes": {
                attr: {"n": od["n"], "mean_gold_rank": od["rank_sum"] / od["n"] if od["n"] else None}
                for attr, od in agg[l]["other"].items()
            },
        }
        for l in layers
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attribute", default="language")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--limit", type=int, default=12, help="Number of DISTINCT example images -- this is "
                     "an ad hoc probe, not a sweep; keep this small.")
    ap.add_argument("--questions_per_image", type=int, default=6, help="Number of differently-worded "
                     "questions (template_ids) to run per selected image, instead of just one -- total "
                     "rows run is up to --limit * --questions_per_image.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Decoder layer indices to report (0=embedding output .. n_layers=the real "
                          "output; 28 for Qwen2.5-VL-7B). Defaults to EVERY layer -- unlike "
                          "attention_maps.py's --layers, this is cheap enough not to subsample.")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="How many top-ranked tokens to "
                     "record per layer/position (not just the top-1).")
    ap.add_argument("--skip_other_attributes", action="store_true", help="Don't track where OTHER "
                     "scored attributes' ground-truth values would rank at each layer -- on by "
                     "default (see module docstring); this only saves a bit of bookkeeping.")
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
                            one_per_image=True, templates_per_image=args.questions_per_image)
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

    per_row_records, summary = score_rows(model, processor, adapter, entity_assets, rows, args.positions, args.layers,
                                           top_k=args.top_k, track_other_attributes=not args.skip_other_attributes)

    print("\n=== per-layer curve (averaged over all rows and valid answer positions) ===")
    for l in sorted(summary):
        s = summary[l]
        if s["n"] == 0:
            continue
        others = ", ".join(f"{a}={od['mean_gold_rank']:.0f}" for a, od in s["other_attributes"].items() if od["n"])
        print(f"layer {l:2d}: top1_match_rate={s['top1_match_rate']:.3f}  mean_gold_rank={s['mean_gold_rank']:.1f}"
              f"  (n={s['n']})" + (f"  other_attr_rank[{others}]" if others else ""))

    out_stem = args.out or os.path.join(REPO_ROOT, "methods", "logit_lens",
                                         f"{args.entity}_{args.attribute}_{args.positions}_report")
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    with open(out_stem + ".jsonl", "w") as f:
        for rec in per_row_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_stem + "_summary.json", "w") as f:
        json.dump({
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "n_rows": len(rows), "top_k": args.top_k, "per_layer": summary,
        }, f, indent=2)
    print(f"\nwrote {out_stem}.jsonl and {out_stem}_summary.json")


if __name__ == "__main__":
    main()
