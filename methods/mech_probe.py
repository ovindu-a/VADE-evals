"""Runs logit_lens.py's and attention_maps.py's scorers off a SINGLE
forward pass per row, instead of running each script separately (two
forward passes over the same rows). Both probes already read from the
same teacher-forced batch (see probe_common.build_probe_batch) -- the
only thing stopping them sharing a pass was that each script's own
score_rows() called teacher_forced_forward() itself. This script calls it
ONCE per row with BOTH output_hidden_states=True and output_attentions=True,
and hands the same `out` to logit_lens.score_one_row() and
attention_maps.score_one_row().

Always loads with attn_implementation="eager" (required for
attention_maps' scorer to get real attn_weights back -- see that module's
docstring) -- harmless for logit_lens' scorer, which only reads
out.hidden_states/out.logits and doesn't care which attention kernel
produced them.

Writes the exact same two outputs the standalone scripts would (same
paths, same shapes -- methods/logit_lens/<...>.jsonl + _summary.json and
methods/attention_maps/<...>_report.json), each tagged
"single_forward_pass": true for provenance, so downstream consumers don't
need to know which script produced a given report.

Layer selection is reconciled between the two probes' different defaults
(logit lens wants every layer -- cheap; attention wants a handful --
eager-attention memory): --layers here defaults to EVERY decoder layer
(matching the "all layer attention map" run this project has already done
manually) and is used for BOTH -- attention stats over exactly --layers
(clamped to valid decoder-layer indices, 0..n_layers-1), logit lens over
--layers PLUS the final pseudo-layer (n_layers, the model's real output --
always included, since it's free/definitional, not an extra forward
pass). Pass a smaller --layers explicitly to cut attention's memory cost
back down to the old 5-layer default if a full sweep isn't needed.

Usage:
    python methods/mech_probe.py --entity flags --attribute language --limit 12 --questions_per_image 6
    python methods/mech_probe.py --entity flags --attribute language --limit 5 \\
        --layers 4 10 14 18 24 --dump_raw_rows 3
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods import attention_maps, logit_lens
from methods.adapters.registry import get_adapter
from methods.common.entities import load_entity_assets, require_pruned_tuples, resolve_position_set
from methods.common.targets import MAX_ANSWER_TOKENS
from methods.probe_common import build_probe_batch, load_probe_rows, teacher_forced_forward


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attribute", default="language")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--limit", type=int, default=12, help="Number of DISTINCT example images.")
    ap.add_argument("--questions_per_image", type=int, default=6, help="Differently-worded questions "
                     "(template_ids) per selected image -- total rows run is up to "
                     "--limit * --questions_per_image.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Decoder layer indices, used by BOTH probes -- see module docstring for how "
                          "each interprets it. Defaults to every layer.")
    ap.add_argument("--threshold", type=float, default=0.2,
                     help="attention_maps: text->image score above which a head is flagged.")
    ap.add_argument("--top_n_heads", type=int, default=attention_maps.DEFAULT_TOP_N_HEADS,
                     help="attention_maps: size of the ranked image-attending-heads list.")
    ap.add_argument("--top_k", type=int, default=logit_lens.DEFAULT_TOP_K,
                     help="logit_lens: how many top-ranked tokens to record per layer/position.")
    ap.add_argument("--skip_other_attributes", action="store_true",
                     help="logit_lens: don't track other attributes' rank at each layer.")
    ap.add_argument("--dump_raw_rows", type=int, default=0,
                     help="attention_maps: save full uint8-quantized attention weights for the first "
                          "N rows -- see attention_maps.py's module docstring. 0 (default) = off.")
    ap.add_argument("--dump_raw_layers", type=int, nargs="+", default=None,
                     help="Layers to include in --dump_raw_rows dumps -- defaults to --layers.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Validate rows/positions resolve without "
                     "loading the 7B model -- no GPU needed.")
    args = ap.parse_args()

    entity_assets = load_entity_assets(args.vade_root, args.entity)
    model_slug = args.model_id.split("/")[-1]
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if not args.allow_unpruned else None)
    rows = load_probe_rows(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir, limit=args.limit,
                            one_per_image=True, templates_per_image=args.questions_per_image)
    print(f"[mech_probe] entity={args.entity} attribute={args.attribute} positions={args.positions}: "
          f"{len(rows)} example row(s)")
    for r in rows:
        print(f"  row_index={r['row_index']} base={r['base']} template={r['template_id']} "
              f"gold={r['base_label']!r}")

    if args.dry_run:
        from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
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
              f"(Model not loaded -- rerun without --dry_run, on a GPU box, to actually run both probes.)")
        return

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(attn_implementation="eager")  # required by attention_maps; harmless for logit_lens

    n_layers = len(adapter.get_decoder_layers(model))
    attn_layers = [l for l in args.layers if l < n_layers] if args.layers is not None else list(range(n_layers))
    ll_layers = sorted(set(args.layers) | {n_layers}) if args.layers is not None else list(range(n_layers + 1))

    image_token_id = adapter.image_token_id(model, processor)
    n_heads = model.config.text_config.num_attention_heads
    _, n_image_tokens, _ = resolve_position_set(args.positions, entity_assets, adapter, model)
    dump_raw_layers = args.dump_raw_layers or attn_layers
    raw_out_dir = os.path.join(REPO_ROOT, "methods", "attention_maps", "raw",
                                f"{args.entity}_{args.attribute}_{args.positions}")

    ll_per_row_records = []
    ll_agg = logit_lens.init_agg(ll_layers)
    am_text_to_image, am_breakdown = attention_maps.init_accumulators(attn_layers, n_heads)
    am_n_mention_found = 0
    am_dumped_paths = []
    batch_cache = None  # each script's own BuildBatchCache type is identical (entities.BuildBatchCache);
    # not shared here since neither probe's per-row cost benefits enough to bother -- one forward pass
    # per row is already the whole point of this script, not a batching optimization.

    for row in rows:
        batch = build_probe_batch([row], entity_assets, adapter, model, processor, args.positions,
                                   batch_cache=batch_cache)
        out = teacher_forced_forward(model, batch, output_hidden_states=True, output_attentions=True,
                                      logits_to_keep=MAX_ANSWER_TOKENS)

        row_record, layer_deltas = logit_lens.score_one_row(
            model, processor, adapter, entity_assets, row, batch, out, ll_layers, n_layers,
            top_k=args.top_k, track_other_attributes=not args.skip_other_attributes)
        ll_per_row_records.append(row_record)
        logit_lens.fold_deltas(ll_agg, ll_layers, layer_deltas)

        if len(am_dumped_paths) < args.dump_raw_rows:
            am_dumped_paths.append(attention_maps.dump_raw_attention(out, dump_raw_layers, row, batch, raw_out_dir))
        text_to_image_row, breakdown_row, mention_found = attention_maps.score_one_row(
            model, processor, adapter, entity_assets, row, batch, out, attn_layers, image_token_id,
            n_image_tokens, n_heads)
        am_n_mention_found += int(mention_found)
        attention_maps.fold_row(am_text_to_image, am_breakdown, attn_layers, n_heads, text_to_image_row, breakdown_row)

    # --- logit_lens output ---
    ll_summary = logit_lens.finalize_summary(ll_agg, ll_layers)
    print("\n=== logit lens: per-layer curve ===")
    for l in sorted(ll_summary):
        s = ll_summary[l]
        if s["n"] == 0:
            continue
        others = ", ".join(f"{a}={od['mean_gold_rank']:.0f}" for a, od in s["other_attributes"].items() if od["n"])
        print(f"layer {l:2d}: top1_match_rate={s['top1_match_rate']:.3f}  mean_gold_rank={s['mean_gold_rank']:.1f}"
              f"  (n={s['n']})" + (f"  other_attr_rank[{others}]" if others else ""))

    ll_out_stem = os.path.join(REPO_ROOT, "methods", "logit_lens", f"{args.entity}_{args.attribute}_{args.positions}_report")
    os.makedirs(os.path.dirname(ll_out_stem), exist_ok=True)
    with open(ll_out_stem + ".jsonl", "w") as f:
        for rec in ll_per_row_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(ll_out_stem + "_summary.json", "w") as f:
        json.dump({
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "n_rows": len(rows), "top_k": args.top_k, "single_forward_pass": True, "per_layer": ll_summary,
        }, f, indent=2)
    print(f"wrote {ll_out_stem}.jsonl and {ll_out_stem}_summary.json")

    # --- attention_maps output ---
    print(f"\n=== attention maps: text->image flow heads (score > {args.threshold}) ===")
    flagged = attention_maps.flagged_heads(am_text_to_image, attn_layers, n_heads, args.threshold)
    print(f"flagged: {flagged}")
    ranked = attention_maps.ranked_image_attending_heads(am_text_to_image, attn_layers, n_heads, args.top_n_heads)
    print(f"top {args.top_n_heads} by text->image attention (ungated): "
          + ", ".join(f"{d['layer']}.{d['head']}={d['score']:.3f}" for d in ranked))

    am_answer_summary = attention_maps.summarize_breakdown(am_breakdown["answer"], attn_layers, n_heads)
    am_mention_summary = (attention_maps.summarize_breakdown(am_breakdown["attribute_mention"], attn_layers, n_heads)
                           if am_n_mention_found else None)
    if am_dumped_paths:
        print(f"wrote {len(am_dumped_paths)} raw quantized attention dump(s) under {raw_out_dir}/")

    am_out_path = os.path.join(REPO_ROOT, "methods", "attention_maps",
                                f"{args.entity}_{args.attribute}_{args.positions}_report.json")
    os.makedirs(os.path.dirname(am_out_path), exist_ok=True)
    with open(am_out_path, "w") as f:
        json.dump({
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "layers": attn_layers, "threshold": args.threshold, "n_rows": len(rows),
            "n_mention_found": am_n_mention_found, "single_forward_pass": True,
            "rows": [{"row_index": r["row_index"], "base": r["base"], "template_id": r["template_id"]} for r in rows],
            "flagged_text_to_image_heads": flagged,
            "ranked_image_attending_heads": ranked,
            "answer_breakdown_summary": am_answer_summary,
            "attribute_mention_breakdown_summary": am_mention_summary,
            "raw_attention_dumps": am_dumped_paths,
        }, f, indent=2)
    print(f"wrote {am_out_path}")


if __name__ == "__main__":
    main()
