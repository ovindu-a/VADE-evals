"""Small, ad hoc attention-flow probe for Qwen2.5-VL on real VADE prompts --
NOT a full sweep (see methods/probe_common.py's module docstring for why
this is a separate concern from PCA/SAE/DAS/DBM): just "for a handful of
example rows, which heads carry image information into the text stream,
and what does the model mainly attend to when it answers."

Three per-(layer, head) scores, computed the same shape as a classic
induction-head detector (loop layers/heads, reduce one attention pattern
to a scalar, threshold, collect "layer.head" strings) -- just with the
reduction redefined for cross-modal flow instead of an offset diagonal:

  1. text_to_image: mean attention mass TEXT-token query positions place
     on IMAGE-token keys. This is the only causally valid direction to ask
     "does image information reach the text stream" in -- Qwen2.5-VL's
     image tokens sit BEFORE the question text in the sequence, so under
     causal masking they can never attend forward into it; only text/
     answer positions can look back at the image. A head with a high
     score here is a candidate image-to-text conduit (cf. "Interpreting
     Attention Heads for Image-to-Text Information Flow", arXiv
     2509.17588).
  2. answer breakdown: at the first answer-prediction position, mean
     attention mass split across {object, image_other, text, answer_ctx}
     token groups (see probe_common.token_groups_for_row) -- "what the
     model mainly looks at when it answers."
  3. attribute-mention breakdown: same 4-group split, but queried FROM the
     specific token that NAMES the attribute in the question itself (e.g.
     the " language" token in "What is the official language..." --
     found via probe_common.attribute_mention_col/adapters' own
     find_last_phrase_token_col, verified against every real flags
     template before being trusted here) -- "when the question first
     mentions the attribute, where does that token look." Skipped for a
     row whose template's phrasing isn't in probe_common.ATTRIBUTE_KEYWORDS
     yet, and for --positions last_token (no image-grid reasoning applies
     there -- see resolve_position_set).

Requires attn_implementation="eager" -- sdpa/flash-attention return None
for attention weights even with output_attentions=True (verified against
this project's installed transformers: Qwen2_5_VLAttention.forward picks
its kernel via ALL_ATTENTION_FUNCTIONS.get_interface(config.
_attn_implementation, eager_attention_forward), and only the eager path
actually returns attn_weights). This is separate from every other
script's model load (which stays on sdpa/flash for speed) -- eager
attention is only ever used here.

Deliberately small-scale: a handful of rows (--limit, default 5), one at
a time (no batching, to keep eager attention's O(seq^2) memory low), a
handful of layers (--layers, default 4 10 14 18 24 -- the same spread the
README's DBM layer-sweep example already uses, so results are directly
comparable to existing runs). Not resumable, not append-and-write like
intervene.py's predictions -- cheap enough to just rerun.

Usage:
    python methods/attention_maps.py --entity flags --attribute language --limit 5
    python methods/attention_maps.py --entity flags --attribute language \\
        --layers 4 10 14 18 24 --positions flag_ring1 --threshold 0.2
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
from methods.common.entities import load_entity_assets, require_pruned_tuples, resolve_position_set
from methods.probe_common import (
    attribute_mention_col, build_probe_batch, load_probe_rows, teacher_forced_forward, token_groups_for_row,
)

DEFAULT_LAYERS = [4, 10, 14, 18, 24]
GROUPS = ("object", "image_other", "text", "answer_ctx")
QUERY_NAMES = ("answer", "attribute_mention")


def _group_mass(attn_row_head, cols):
    """attn_row_head: [Lseq] attention distribution for one (row, layer, head, query
    position). cols: LongTensor of key-column indices. Returns the summed mass over
    those columns, or 0.0 if the group is empty (e.g. answer_ctx for a single-token
    gold answer)."""
    if cols.numel() == 0:
        return 0.0
    return attn_row_head[cols].sum().item()


def score_rows(model, processor, adapter, entity_assets, rows, positions, layers, batch_cache=None):
    """Runs each row through the model ONE AT A TIME (see module docstring --
    eager attention's memory cost scales with seq^2, not worth batching for
    a handful of rows), and returns:
      text_to_image[layer][head]              -- list of per-row scores (#1)
      breakdown[qname][layer][group][head]     -- list of per-row fractions (#2/#3),
        qname in QUERY_NAMES ("answer", "attribute_mention")
      n_mention_found                          -- how many rows had a registered
        attribute-mention phrase actually present in their template's wording
        (out of len(rows) -- see probe_common.ATTRIBUTE_KEYWORDS)
    All averaged by the caller, not here, so it can also report spread across rows.
    """
    image_token_id = adapter.image_token_id(model, processor)
    n_layers = len(adapter.get_decoder_layers(model))
    assert all(0 <= l < n_layers for l in layers), f"--layers must be in [0, {n_layers}) for this model, got {layers}"
    n_heads = model.config.text_config.num_attention_heads
    # Real image-token count for THIS entity, independent of `positions` -- needed to locate the
    # attribute-mention token even though it has nothing to do with the object's own token set (see
    # entities.py's resolve_position_set: only "last_token" mode returns None here).
    _, n_image_tokens, _ = resolve_position_set(positions, entity_assets, adapter, model)

    text_to_image = {l: [[] for _ in range(n_heads)] for l in layers}
    breakdown = {qname: {l: {g: [[] for _ in range(n_heads)] for g in GROUPS} for l in layers}
                 for qname in QUERY_NAMES}
    n_mention_found = 0

    for row in rows:
        batch = build_probe_batch([row], entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
        out = teacher_forced_forward(model, batch, output_attentions=True, logits_to_keep=1)
        seq_len = batch["ext_input_ids"].shape[1]
        object_cols = batch["positions"][0]  # this row's object-token columns, absolute (see probe_common)
        groups = token_groups_for_row(seq_len, batch["ext_attention_mask"][0], image_token_id,
                                       batch["ext_input_ids"][0], object_cols, batch["readout_start_col"])
        image_cols = torch.cat([groups["object"], groups["image_other"]])
        text_cols = groups["text"]

        query_positions = {"answer": batch["readout_start_col"]}  # first answer-prediction position (j=0)
        if n_image_tokens is not None:
            tmpl = entity_assets.template_lookup[row["queried"]][row["template_id"]]
            # build_probe_batch used batch size 1 -> build_batch's left-pad is 0 for this row, so the
            # unpadded column find_last_phrase_token_col returns IS the absolute column in ext_input_ids.
            mention_col = attribute_mention_col(adapter, processor, row["target_attribute"],
                                                 tmpl["question"], tmpl["prefill"], n_image_tokens)
            if mention_col is not None:
                query_positions["attribute_mention"] = mention_col
                n_mention_found += 1

        for l in layers:
            attn_l = out.attentions[l][0]  # [H, Lseq, Lseq] -- batch size 1
            for h in range(n_heads):
                if text_cols.numel() > 0:
                    per_text_query_mass = attn_l[h, text_cols][:, image_cols].sum(dim=-1)  # [n_text_q]
                    text_to_image[l][h].append(per_text_query_mass.mean().item())
                for qname, qpos in query_positions.items():
                    for g in GROUPS:
                        breakdown[qname][l][g][h].append(_group_mass(attn_l[h, qpos], groups[g]))

    return text_to_image, breakdown, n_heads, n_mention_found


def flagged_heads(text_to_image, layers, n_heads, threshold):
    """Same shape as a classic induction_attn_detector: loop layers/heads,
    average the per-row scores, collect "layer.head" strings over threshold."""
    flagged = []
    for l in layers:
        for h in range(n_heads):
            scores = text_to_image[l][h]
            if not scores:
                continue
            mean_score = sum(scores) / len(scores)
            if mean_score > threshold:
                print(f"text->image score for {l}.{h} = {mean_score:.3f}")
                flagged.append(f"{l}.{h}")
    return flagged


def summarize_breakdown(breakdown_for_qname, layers, n_heads):
    """Per layer: the head with the largest OBJECT-group share (the head most
    focused on the entity's own tokens FROM whichever query position this
    breakdown was computed at -- the answer position, or the attribute-
    mention token), plus the layer-mean share of each group averaged over
    heads and rows. Shared by both query positions (see QUERY_NAMES) --
    pass breakdown[qname], not the whole breakdown dict."""
    summary = {}
    for l in layers:
        head_object_means = [
            sum(breakdown_for_qname[l]["object"][h]) / len(breakdown_for_qname[l]["object"][h])
            if breakdown_for_qname[l]["object"][h] else 0.0
            for h in range(n_heads)
        ]
        top_head = max(range(n_heads), key=lambda h: head_object_means[h])
        layer_means = {
            g: sum(sum(breakdown_for_qname[l][g][h]) / len(breakdown_for_qname[l][g][h])
                   if breakdown_for_qname[l][g][h] else 0.0 for h in range(n_heads)) / n_heads
            for g in GROUPS
        }
        summary[l] = {
            "layer_mean_group_share": layer_means,
            "top_object_attending_head": top_head,
            "top_object_attending_head_score": head_object_means[top_head],
        }
    return summary


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
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                     help="Decoder layer indices (0-27 for Qwen2.5-VL-7B) -- default matches the "
                          "README's existing DBM layer-sweep spread.")
    ap.add_argument("--threshold", type=float, default=0.2,
                     help="text->image score above which a head is flagged as a candidate conduit.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Validate rows/positions resolve without "
                     "loading the 7B model -- no GPU needed.")
    ap.add_argument("--out", default=None, help="Defaults to methods/attention_maps/"
                     "<entity>_<attribute>_<positions>_report.json")
    args = ap.parse_args()

    entity_assets = load_entity_assets(args.vade_root, args.entity)
    model_slug = args.model_id.split("/")[-1]
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if not args.allow_unpruned else None)
    rows = load_probe_rows(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir, limit=args.limit,
                            one_per_image=True)
    print(f"[attention_maps] entity={args.entity} attribute={args.attribute} positions={args.positions}: "
          f"{len(rows)} example row(s), layers={args.layers}")
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
              f"(Model not loaded -- rerun without --dry_run, on a GPU box, to actually score attention.)")
        return

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(attn_implementation="eager")

    text_to_image, breakdown, n_heads, n_mention_found = score_rows(
        model, processor, adapter, entity_assets, rows, args.positions, args.layers)

    print(f"\n=== text->image flow heads (score > {args.threshold}) ===")
    flagged = flagged_heads(text_to_image, args.layers, n_heads, args.threshold)
    print(f"flagged: {flagged}")

    def _print_breakdown(title, summary):
        print(f"\n=== {title} ===")
        for l in args.layers:
            s = summary[l]
            shares = ", ".join(f"{g}={v:.3f}" for g, v in s["layer_mean_group_share"].items())
            print(f"layer {l}: mean group share [{shares}]; head most focused on the entity's own tokens = "
                  f"head {s['top_object_attending_head']} (share={s['top_object_attending_head_score']:.3f})")

    answer_summary = summarize_breakdown(breakdown["answer"], args.layers, n_heads)
    _print_breakdown("answer-position attention breakdown", answer_summary)

    mention_summary = None
    print(f"\n=== attribute-mention-token attention breakdown "
          f"({n_mention_found}/{len(rows)} rows matched a registered phrasing) ===")
    if n_mention_found:
        mention_summary = summarize_breakdown(breakdown["attribute_mention"], args.layers, n_heads)
        for l in args.layers:
            s = mention_summary[l]
            shares = ", ".join(f"{g}={v:.3f}" for g, v in s["layer_mean_group_share"].items())
            print(f"layer {l}: mean group share [{shares}]; head most focused on the entity's own tokens = "
                  f"head {s['top_object_attending_head']} (share={s['top_object_attending_head_score']:.3f})")
    else:
        print("(no rows matched -- either --positions last_token, or this attribute has no entry / no "
              "matching phrasing yet in probe_common.ATTRIBUTE_KEYWORDS)")

    out_path = args.out or os.path.join(REPO_ROOT, "methods", "attention_maps",
                                         f"{args.entity}_{args.attribute}_{args.positions}_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "entity": args.entity, "attribute": args.attribute, "positions": args.positions,
            "layers": args.layers, "threshold": args.threshold, "n_rows": len(rows),
            "n_mention_found": n_mention_found,
            "rows": [{"row_index": r["row_index"], "base": r["base"], "template_id": r["template_id"]} for r in rows],
            "flagged_text_to_image_heads": flagged,
            "answer_breakdown_summary": answer_summary,
            "attribute_mention_breakdown_summary": mention_summary,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
