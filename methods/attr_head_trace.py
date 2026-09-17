"""TRACE the question->readout handoff: which attention heads carry WHICH
ATTRIBUTE was asked into the position that answers, and which are load-bearing.

WHAT IS DIFFERENT FROM head_trace.py. That script holds the question fixed and
swaps the IMAGE, so it traces the image->text read: which heads carry the
ENTITY's identity. This one holds the image fixed and swaps the QUESTION, so it
traces which heads carry the ATTRIBUTE SELECTION. Same entity, same pixels, same
answer position -- only "report the capital" vs "report the currency" differs.
The two are different circuits and their windows are different claims.

Everything else is deliberately the same machinery: phases 1/1b/1c/2/3/4, the
saturated-mask head patch, the shuffled-donor control, and the read-back
identity are IMPORTED from head_trace rather than reimplemented, so a fix to
either lands in both and the two results stay comparable.

WHERE THE WINDOW COMES FROM (see ATTRIBUTE_SWITCH_FINDINGS.md). A full residual
swap in methods/attribute_switch_sweep.py reads, as donor-first-token rate:

    block           <=15    17     18     19     20     21    22+
    question text   100%    79%    29%    13%     2%     2%    2%
    last token        2%    13%    50%    66%    98%   100%  100%
    last attention    2%     2%    10%    19%    14%    10%    2%

The two residual columns cross in blocks 17-20: before it, editing the question
propagates; after it, editing the question is too late and editing the readout
is decisive. MLP is at the 2% floor across the whole crossover, and attention is
the only cross-position operation a transformer has, so the entire transfer is
some set of (block, head) in that window. Attention SPANS pin it tighter: any
span covering blocks 19-21 reads 96.6-100%, while blocks 20..27 -- eight blocks,
missing only 19 -- caps at 79.8%. Blocks 16-18 carry ~45% on their own, which is
the representation being assembled before it is read.

So --blocks defaults to 15..22 and --patch_layer to 15.

    --patch_layer L patches the residual at the QUESTION tokens at layer L,
    i.e. the output of block L-1. Block L-1's attention has already run, so a
    traced block must satisfy block >= patch_layer or its heads cannot possibly
    respond to the patch. Asserted, not documented-and-hoped.

WHY SUFFICIENCY (phase 3) IS THE PHASE TO READ. The span table above says this
mechanism is redundant across depth: dropping block 21 costs almost nothing when
enough earlier blocks remain. Cumulative knockout is exactly the measurement
redundancy defeats, so expect a flat phase-2 curve even for real conduits. A
steep phase-3 curve is localization regardless -- and phase 4 is what decides
whether it is localization or an artefact of perturbing the site.

AND: report cause_first, not cause. A last-token intervention is structurally
unable to steer an answer past its first token; the KV cache below it still
belongs to the base. `capital` and `calling_code` golds are 2-3 tokens, so their
full-match rate is capped by answer length rather than by the intervention.
Both are printed; the first-token column is the one that compares across
attributes.

Usage:
    python methods/attr_head_trace.py --entity flags --patch_layer 15 \
        --blocks 15 16 17 18 19 20 21 22 --n_countries 24
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.attribute_switch_sweep import ATTRIBUTES, PREFILL, alignment, question  # noqa: E402
from methods.common.hooks import make_cache_aware_patch_hook  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.sites import RESIDUAL_SITE  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS, derive_gold_token_ids, exact_match, pad_gold_toks  # noqa: E402
# The phase machinery itself: identical code for the image trace and this one.
from methods.head_trace import (  # noqa: E402
    RANK_BY, capture_head_outputs, generate_with_patches, head_patches, per_head_delta, probe_identity,
)
from methods.ndm.verify_sites import generate_unhooked, hard_mask  # noqa: E402

ENTITY_KEYS = {"flags": "countries", "animals": "species", "brands": "brands"}


# ---------------------------------------------------------------- batch build

def build_switch_batch(rows, entity_dir, items, adapter, model, processor, question_positions):
    """One batch of same-image/different-question rows.

    Every row contributes a BASE prompt (its own attribute) and a DONOR prompt
    (the attribute being switched in) over the SAME image, so `base_extra` is
    the vision input for both -- there is no source image. `alignment` enforces
    equal token length and that all differences fall after the image, which is
    what makes one shared position set valid for the whole batch.
    """
    image_id = adapter.image_token_id(model, processor)
    special = processor.tokenizer.all_special_ids
    base_ids, donor_ids, pixels, grids, earlier = [], [], [], [], None
    base_gold, source_gold = [], []
    for row in rows:
        with Image.open(entity_dir / items[row["base"]]["image"]) as img:
            picture = img.convert("RGB")
            base = adapter.build_inputs(processor, picture, question(row["base_attribute"]), PREFILL)
            donor = adapter.build_inputs(processor, picture, question(row["donor_attribute"]), PREFILL)
        b, d = base["input_ids"].unsqueeze(0), donor["input_ids"].unsqueeze(0)
        positions = alignment(b, d, image_id, special)
        if earlier is None:
            earlier = positions
        elif positions != earlier:
            raise ValueError("Rows disagree on the text positions; batch only prompts of one shape")
        base_ids.append(b)
        donor_ids.append(d)
        pixels.append(base["extra"]["pixel_values"])
        grids.append(base["extra"]["image_grid_thw"])
        base_gold.append(derive_gold_token_ids(processor.tokenizer, PREFILL, row["base_label"]))
        source_gold.append(derive_gold_token_ids(processor.tokenizer, PREFILL, row["source_label"]))
    if len({x.shape[1] for x in base_ids}) != 1:
        raise ValueError("Prompts in a batch must share a token length; split the batch by shape")
    ids = torch.cat(base_ids)
    donor = torch.cat(donor_ids)
    # The divergent tail is the only part of `earlier` that can differ between the two prompts:
    # below the first differing token the two prompts ARE the same tokens, so under causal
    # attention their activations are identical and patching them is provably a no-op. Offering
    # the restricted set is not an approximation -- it is the same intervention, cheaper.
    first_diff = int((ids[0] != donor[0]).nonzero()[0])
    if question_positions == "divergent":
        cols = [p for p in earlier if p >= first_diff]
    else:
        cols = list(earlier)
    pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    bg, bl = pad_gold_toks(base_gold, pad)
    sg, sl = pad_gold_toks(source_gold, pad)
    B = len(rows)
    return {
        "rows": rows,
        "base_input_ids": ids, "source_input_ids": donor,
        "attention_mask": torch.ones_like(ids),
        "base_extra": {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.cat(grids)},
        "positions": torch.tensor([cols] * B, dtype=torch.long),
        "last_positions": torch.full((B, 1), ids.shape[1] - 1, dtype=torch.long),
        "all_text_positions": earlier, "first_divergence": first_diff,
        "base_gold_toks": bg, "base_gold_len": bl, "source_gold_toks": sg, "source_gold_len": sl,
    }


def score_switch(gen_toks, batch):
    """-> (donor_first, donor_full, base_first, base_full) with the trainer's own matcher.

    `*_first` is `exact_match` with every gold length clamped to 1 -- the same
    code path, so the two columns cannot drift apart in tokenization or padding.
    """
    pred = gen_toks[:, :MAX_ANSWER_TOKENS]
    one = torch.ones_like(batch["source_gold_len"])
    return (exact_match(pred, batch["source_gold_toks"], one).float().mean().item(),
            exact_match(pred, batch["source_gold_toks"], batch["source_gold_len"]).float().mean().item(),
            exact_match(pred, batch["base_gold_toks"], one).float().mean().item(),
            exact_match(pred, batch["base_gold_toks"], batch["base_gold_len"]).float().mean().item())


# ------------------------------------------------------- the question-side patch

def capture_question_source(adapter, model, batch, patch_layer):
    """The DONOR question's residual stream at the question positions.

    The image is the same in both prompts, so this is the exact analogue of
    head_trace's `capture_image_source` with the roles of the two inputs
    swapped: there, the pixels change and the text is fixed.
    """
    return RESIDUAL_SITE.capture(adapter, model, patch_layer, batch["source_input_ids"],
                                 batch["attention_mask"], batch["base_extra"], batch["positions"])


def question_patch(adapter, model, batch, patch_layer, src=None):
    """-> (site, layer, patch_fn) replacing the base's residual at the question
    positions with the donor question's -- the intervention the sweep measures
    at ~100% donor-first below block 16, and the effect every phase traces."""
    if src is None:
        src = capture_question_source(adapter, model, batch, patch_layer)
    one = hard_mask(RESIDUAL_SITE.width(adapter, model), 1.0, model.device)
    fn = make_cache_aware_patch_hook(batch["positions"], lambda base_vals: one(base_vals, src))
    return (RESIDUAL_SITE, patch_layer, fn)


# ------------------------------------------------------------------------ rows

def sample_rows(entity_dir, items, attributes, countries, n_countries, seed):
    chosen = countries or random.Random(seed).sample(sorted(items), min(n_countries, len(items)))
    if len(set(chosen)) != len(chosen):
        raise ValueError("Item IDs must be unique")
    rows = []
    for item_id in chosen:
        item = items[item_id]
        if not (entity_dir / item["image"]).is_file():
            raise FileNotFoundError(entity_dir / item["image"])
        for base, donor in itertools.permutations(attributes, 2):
            rows.append({"row_index": len(rows), "base": item_id, "source": item_id,
                         "base_attribute": base, "donor_attribute": donor,
                         "base_label": item[base], "source_label": item[donor],
                         "template_id": "controlled_report_v1"})
    return chosen, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attributes", nargs="+", choices=ATTRIBUTES, default=ATTRIBUTES)
    ap.add_argument("--patch_layer", type=int, default=15,
                    help="Layer to patch the QUESTION-token residual at (output of block L-1). Must sit "
                         "BELOW the handoff, where the sweep's question-residual column is still ~100%%; "
                         "on flags that is any L<=16. Every traced block must be >= this.")
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(15, 23)),
                    help="Decoder blocks to trace heads in. Default 15-22: 19-21 is the read, 16-18 is "
                         "where the representation is assembled (ATTRIBUTE_SWITCH_FINDINGS.md).")
    ap.add_argument("--question_positions", choices=["all_text", "divergent"], default="all_text",
                    help="Which question columns to patch. 'divergent' keeps only columns at/after the "
                         "first token where the two prompts differ; the rest are provably inert under "
                         "causal attention, so the two choices are the same intervention.")
    ap.add_argument("--rank_by", default="delta_resid", choices=list(RANK_BY))
    ap.add_argument("--knockout_ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n_random", type=int, default=8,
                    help="Size of the RANDOM-head null. 0 disables it, which makes phase 2 unreadable.")
    ap.add_argument("--control_k", type=int, nargs="+", default=[8])
    ap.add_argument("--skip_knockout", action="store_true", help="Phase 1 + 1c only.")
    ap.add_argument("--skip_sufficiency", action="store_true")
    ap.add_argument("--per_attribute", action="store_true",
                    help="Also rank heads separately per DONOR attribute. The sweep shows the "
                         "single-block peak moves with the attribute (block 19 for currency and "
                         "calling_code, 20 for capital, nowhere for language), so a pooled ranking "
                         "can hide an attribute-specific head. Costs no extra forward passes.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--countries", nargs="+", help="Explicit item IDs; otherwise a deterministic sample.")
    ap.add_argument("--n_countries", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_ANSWER_TOKENS + 2)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true", help="Validate rows/images without torch or a model.")
    args = ap.parse_args(argv)

    from pathlib import Path
    entity_dir = Path(args.vade_root) / "data" / args.entity
    raw = (entity_dir / "ground_truth.json").read_bytes()
    gt = json.loads(raw)
    key = ENTITY_KEYS.get(args.entity)
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != "coverage")
    items = gt[key]
    if len(set(args.attributes)) < 2:
        ap.error("Need at least two attributes to switch between")
    blocks = sorted(set(args.blocks))
    if blocks[0] < args.patch_layer:
        ap.error(f"block {blocks[0]} runs its attention BEFORE the patch at layer {args.patch_layer} "
                 f"(which writes block {args.patch_layer - 1}'s output), so its heads cannot respond to "
                 f"it. Use --patch_layer {blocks[0]} or start --blocks at {args.patch_layer}.")
    chosen, rows = sample_rows(entity_dir, items, args.attributes, args.countries,
                               args.n_countries, args.seed)

    out_dir = Path(REPO_ROOT) / "logs" / "attr_head_trace" / args.entity
    out_dir.mkdir(parents=True, exist_ok=True)
    span = f"{blocks[0]}-{blocks[-1]}" if blocks == list(range(blocks[0], blocks[-1] + 1)) \
        else ".".join(map(str, blocks))
    tag = f"attr_head_trace_patch{args.patch_layer}_blocks{span}_{args.question_positions}"
    out_path = args.out or str(out_dir / f"{tag}.json")
    print(f"[attr_head_trace] entity={args.entity} attributes={args.attributes} "
          f"patch_layer={args.patch_layer} blocks={span} positions={args.question_positions} "
          f"{len(chosen)} items x {len(args.attributes) * (len(args.attributes) - 1)} directed pairs "
          f"= {len(rows)} rows -> {out_path}")
    if args.dry_run:
        print("Metadata valid. Token alignment and clean accuracy require a model run.")
        return

    with tee_to_log(str(out_dir / f"{tag}.log")):
        adapter = get_adapter(args.model_id)
        # sdpa, not eager: nothing here reads attention weights (per_head_delta works off
        # attn_head_output, which is o_proj's INPUT, not the attention matrix).
        model, processor = adapter.load(device=args.device, dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
        model.eval().requires_grad_(False)
        n_layers = len(adapter.get_decoder_layers(model))
        n_heads = adapter.n_attention_heads(model)
        hidden = adapter.hidden_size(model)
        assert hidden % n_heads == 0, f"hidden_size {hidden} is not divisible by {n_heads} heads"
        head_dim = hidden // n_heads
        assert all(0 <= b < n_layers for b in blocks), f"blocks must be in 0..{n_layers - 1}"
        assert 1 <= args.patch_layer <= n_layers, f"patch_layer must be in 1..{n_layers}"
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        batches = []
        for i in range(0, len(rows), args.batch_size):
            b = build_switch_batch(rows[i:i + args.batch_size], entity_dir, items, adapter, model,
                                   processor, args.question_positions)
            # The read column must not be one of the patched columns: patching the position being
            # read would write the answer directly into what phase 1 then reports as a head's doing.
            overlap = sorted(set(b["last_positions"][0].tolist()) & set(b["positions"][0].tolist()))
            assert not overlap, f"the read column {overlap} is inside the patched question span"
            batches.append(b)
        print(f"  {len(batches)} batch(es); question columns={batches[0]['positions'].shape[1]} "
              f"of {len(batches[0]['all_text_positions'])} text columns "
              f"(first divergence at token {batches[0]['first_divergence']}), "
              f"prompt length={batches[0]['base_input_ids'].shape[1]}")

        # ---------------- phase 0: the population ----------------
        # Restricting to rows the clean model answers correctly is not cosmetic: a row whose base
        # answer is already wrong cannot show base preservation, and one whose donor answer the
        # model does not know cannot show transfer. The sweep conditions on exactly this.
        print(f"\n{'=' * 78}\n=== phase 0: clean baselines\n{'=' * 78}")
        clean_first = clean_full = donor_first = donor_full = 0.0
        seen = 0
        for b in batches:
            gen = generate_unhooked(model, b["base_input_ids"], b["attention_mask"], b["base_extra"],
                                    pad_token_id, args.max_new_tokens)
            _, _, bf, bu = score_switch(gen, b)
            gen_d = generate_unhooked(model, b["source_input_ids"], b["attention_mask"], b["base_extra"],
                                      pad_token_id, args.max_new_tokens)
            df, du, _, _ = score_switch(gen_d, b)
            k = len(b["rows"])
            clean_first, clean_full = clean_first + bf * k, clean_full + bu * k
            donor_first, donor_full = donor_first + df * k, donor_full + du * k
            seen += k
        print(f"  base question answered correctly:  first={clean_first / seen:6.1%} "
              f"full={clean_full / seen:6.1%}")
        print(f"  donor question answered correctly: first={donor_first / seen:6.1%} "
              f"full={donor_full / seen:6.1%}")

        # ---------------- phase 1: differential head trace ----------------
        print(f"\n{'=' * 78}\n=== phase 1: differential head trace (2 forwards per batch, no generation)"
              f"\n{'=' * 78}")
        agg = {(b, h): {"delta_z": 0.0, "delta_resid": 0.0, "delta_dla": 0.0, "n": 0}
               for b in blocks for h in range(n_heads)}
        per_attr = {}
        base_z_per_batch, patched_z_per_batch, q_src = [], [], []
        for bi, batch in enumerate(batches):
            last_pos = batch["last_positions"]
            q_src.append(capture_question_source(adapter, model, batch, args.patch_layer))
            patch = question_patch(adapter, model, batch, args.patch_layer, src=q_src[-1])
            base_z, base_final = capture_head_outputs(
                adapter, model, blocks, last_pos, batch["base_input_ids"], batch["attention_mask"],
                batch["base_extra"])
            patched_z, patched_final = capture_head_outputs(
                adapter, model, blocks, last_pos, batch["base_input_ids"], batch["attention_mask"],
                batch["base_extra"], patches=[patch])
            # base gold minus donor gold at the first answer position: the exact axis the switch
            # moves along, kept PER ROW because each row has its own pair of answers.
            gold_b = batch["base_gold_toks"][:, 0].to(model.device)
            gold_s = batch["source_gold_toks"][:, 0].to(model.device)
            d = adapter.logit_direction(model, gold_b) - adapter.logit_direction(model, gold_s)
            scale_base = adapter.final_norm_scale(model, base_final.float())
            scale_patched = adapter.final_norm_scale(model, patched_final.float())
            base_z_per_batch.append(base_z)
            patched_z_per_batch.append(patched_z)
            for b in blocks:
                for h, dz, dr, dd in per_head_delta(adapter, model, base_z[b], patched_z[b], b, n_heads,
                                                    head_dim, d, scale_base, scale_patched):
                    a = agg[(b, h)]
                    a["delta_z"] += dz
                    a["delta_resid"] += dr
                    a["delta_dla"] += dd
                    a["n"] += 1
            if args.per_attribute:
                # Same captures, re-reduced over the rows of one donor attribute. Free: the
                # expensive part (two forwards) has already happened.
                for attribute in sorted({r["donor_attribute"] for r in batch["rows"]}):
                    idx = [i for i, r in enumerate(batch["rows"]) if r["donor_attribute"] == attribute]
                    if not idx:
                        continue
                    sel = torch.tensor(idx, device=base_z[blocks[0]].device)
                    slot = per_attr.setdefault(attribute, {(b, h): [0.0, 0] for b in blocks
                                                           for h in range(n_heads)})
                    for b in blocks:
                        for h, dz, dr, dd in per_head_delta(
                                adapter, model, base_z[b][sel], patched_z[b][sel], b, n_heads, head_dim,
                                d[sel], scale_base[sel], scale_patched[sel]):
                            rec = slot[(b, h)]
                            rec[0] += {"delta_z": dz, "delta_resid": dr, "delta_dla": dd}[args.rank_by]
                            rec[1] += 1
            print(f"  batch {bi + 1}/{len(batches)} traced", flush=True)

        table = [{"block": b, "head": h, "delta_z": a["delta_z"] / a["n"],
                  "delta_resid": a["delta_resid"] / a["n"], "delta_dla": a["delta_dla"] / a["n"]}
                 for (b, h), a in agg.items() if a["n"]]
        ranked = sorted(table, key=lambda r: -abs(r[args.rank_by]))
        print(f"\ntop {args.top_n} heads by |{args.rank_by}| (how much this head's write to the "
              f"answer position changed when the QUESTION was swapped):")
        print(f"{'block.head':>12} {'delta_resid':>12} {'delta_z':>10} {'delta_dla':>11}")
        for r in ranked[:args.top_n]:
            print(f"{r['block']}.{r['head']:<10} {r['delta_resid']:>12.4f} {r['delta_z']:>10.4f} "
                  f"{r['delta_dla']:>11.4f}")
        by_block = {}
        for r in table:
            by_block[r["block"]] = by_block.get(r["block"], 0.0) + abs(r[args.rank_by])
        print(f"\nper-block total |{args.rank_by}| (where the attribute selection is written):")
        for b in blocks:
            print(f"  block {b:>2}: {by_block[b]:>10.4f}")
        for k in sorted({min(kk, len(ranked)) for kk in args.knockout_ks}):
            comp = {}
            for r in ranked[:k]:
                comp[r["block"]] = comp.get(r["block"], 0) + 1
            print(f"  top-{k:<3} heads sit in blocks: "
                  + ", ".join(f"{b}x{c}" for b, c in sorted(comp.items())))

        report = {"experiment": "attr_head_trace", "entity": args.entity,
                  "attribute": "+".join(args.attributes), "attributes": args.attributes,
                  "patch_layer": args.patch_layer, "positions": f"question:{args.question_positions}",
                  "blocks": blocks, "n_heads": n_heads, "head_dim": head_dim, "n_rows": len(rows),
                  "items": chosen, "rank_by": args.rank_by, "seed": args.seed,
                  "clean": {"base_first": clean_first / seen, "base_full": clean_full / seen,
                            "donor_first": donor_first / seen, "donor_full": donor_full / seen},
                  "phase1": table, "phase1_ranked": [(r["block"], r["head"]) for r in ranked[:64]],
                  "ground_truth_sha256": hashlib.sha256(raw).hexdigest()}

        if args.per_attribute:
            print(f"\nper-attribute top-8 by |{args.rank_by}| (a pooled ranking can hide a head that "
                  f"only fires for one attribute):")
            report["phase1_per_attribute"] = {}
            for attribute, slot in sorted(per_attr.items()):
                rows_a = sorted(({"block": b, "head": h, args.rank_by: v[0] / v[1]}
                                 for (b, h), v in slot.items() if v[1]),
                                key=lambda r: -abs(r[args.rank_by]))
                report["phase1_per_attribute"][attribute] = rows_a
                print(f"  {attribute:13s} " + "  ".join(f"{r['block']}.{r['head']}" for r in rows_a[:8]))

        # ---------------- phase 1c: read-back identity ----------------
        # Imported wholesale from head_trace: install every traced head's patched value at the read
        # column, read the same site back, and compare the final residual against the question-patched
        # run. With every downstream block traced this is a closed identity, not an approximation --
        # the patch is at QUESTION columns, so the read column's residual entering the first traced
        # block is identical either way; attention is the only cross-position operation; MLPs are
        # position-wise. Anything short of a match there is a fault, not a ceiling.
        print(f"\n{'=' * 78}\n=== phase 1c: read-back identity (batch 1, 3 forwards, no generation)"
              f"\n{'=' * 78}")
        b0 = batches[0]
        pos0, ids0, mask0, extra0 = (b0["last_positions"], b0["base_input_ids"], b0["attention_mask"],
                                     b0["base_extra"])
        every_head = [(b, h) for b in blocks for h in range(n_heads)]
        clean_z, clean_o, clean_entry, clean_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0)
        pat_z, pat_o, pat_entry, pat_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0,
            patches=[question_patch(adapter, model, b0, args.patch_layer, src=q_src[0])])
        inst_z, inst_o, inst_entry, inst_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0,
            patches=head_patches(every_head, pat_z, pos0, hidden, head_dim, model.device))

        def rel(got, want):
            want = want.float()
            return (got.float() - want).norm().item() / max(want.norm().item(), 1e-9)

        def o_proj_rel(run_z, run_o, b):
            mod = adapter.get_attn_head_output_module(model, b)
            expect = run_z[b].float() @ mod.weight.float().T
            if mod.bias is not None:
                expect = expect + mod.bias.float()
            return rel(run_o[b], expect)

        entry_rel = rel(pat_entry, clean_entry)
        print(f"  residual entering block {blocks[0]} at the read column, question-patched vs clean: "
              f"{entry_rel:.3%}")
        if blocks[0] == args.patch_layer:
            print("    -> must be ~0: nothing upstream of the window can have carried the edit here yet"
                  if entry_rel < 0.01 else
                  "    !! should be ~0 -- the patch reaches the read column before the first traced "
                  "block, so the identity does not apply and nothing below is interpretable")
        else:
            print(f"    -> expected nonzero: blocks {args.patch_layer}..{blocks[0] - 1} run before the "
                  f"window. Read it as how much of the question edit has already reached the read "
                  f"column by block {blocks[0]}.")
        worst, worst_b = 0.0, blocks[0]
        for b in blocks:
            r = rel(inst_z[b], pat_z[b])
            if r >= worst:
                worst, worst_b = r, b
        cons_clean = {b: o_proj_rel(clean_z, clean_o, b) for b in blocks}
        cons_inst = {b: o_proj_rel(inst_z, inst_o, b) for b in blocks}
        print(f"\n  o_proj INPUT reads back what was installed: worst {worst:.2e} (block {worst_b})")
        print(f"  o_proj OUTPUT vs W_O @ its captured INPUT: clean max {max(cons_clean.values()):.2%}, "
              f"installed max {max(cons_inst.values()):.2%}")
        to_patched, to_clean = rel(inst_final, pat_final), rel(inst_final, clean_final)
        missing = sorted(set(range(args.patch_layer, n_layers)) - set(blocks))
        print(f"  final pre-norm residual at the read column, installed-heads run vs:")
        print(f"    the question-patched run: {to_patched:.3%}   <-- the identity; ~0 with full coverage")
        print(f"    the CLEAN run:            {to_clean:.3%}   <-- ~0 would mean the patch changed nothing")
        if max(cons_clean.values()) >= 0.05:
            print("  -> the CLEAN control fails, so this check is itself wrong. Fix it first.")
        elif max(cons_inst.values()) >= 0.05:
            print("  -> FOUND IT: o_proj's input reads back but its OUTPUT does not match W_O @ that "
                  "input. The block is not consuming the rewrite; phases 2-4 are void.")
        elif to_clean < 0.01:
            print("  -> the patch is consumed yet the final residual is unchanged from clean; look for "
                  "the head patch being removed before the blocks that matter run.")
        elif to_patched < 0.02:
            print("  -> MATCHES. Phases 2 and 3's k=all arms MUST reproduce the question patch's cause.")
        elif missing:
            print(f"  -> diverges, and --blocks omits downstream block(s) {missing}, which explains it: "
                  f"their attention at the read column still sees the clean question. Phase 3's ALL arm "
                  f"is a PARTIAL ceiling, not the identity.")
        else:
            print("  -> diverges with complete coverage and a consumed patch. Phases 2-4 are void.")
        report["phase1c"] = {"entry_residual_rel": entry_rel, "readback_worst_rel": worst,
                             "o_proj_consumption_clean": cons_clean,
                             "o_proj_consumption_installed": cons_inst,
                             "final_vs_patched_rel": to_patched, "final_vs_clean_rel": to_clean,
                             "blocks_missing_downstream": missing}

        if args.skip_knockout:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nwrote {out_path}  (generation phases skipped)")
            return

        def run_cause(selected_heads, z_per_batch, with_question_patch, telemetry=None, roll=False):
            """One function for every generation arm, so no two curves can drift
            apart in rows, masks, scoring or generation.

              with_question_patch=True,  z = base_z    -> KNOCKOUT (necessity)
              with_question_patch=False, z = patched_z -> SUFFICIENCY
              roll=True                                -> shuffled-donor control

            Returns first/full donor rates plus base_kept, and under `roll` also
            the rate against the DONOR ROW's gold -- the number that separates
            real transfer from a generic perturbation.
            """
            acc = {"first": 0.0, "full": 0.0, "kept": 0.0, "roll_first": 0.0, "n": 0}
            for batch, z, src in zip(batches, z_per_batch, q_src):
                if roll and len(batch["rows"]) < 2:
                    continue                     # a 1-row batch rolls onto itself
                values = {b: torch.roll(v, shifts=1, dims=0) for b, v in z.items()} if roll else z
                patches = ([question_patch(adapter, model, batch, args.patch_layer, src=src)]
                           if with_question_patch else [])
                patches += head_patches(selected_heads, values, batch["last_positions"], hidden,
                                        head_dim, model.device, telemetry=telemetry)
                gen = generate_with_patches(adapter, model, patches, batch["base_input_ids"],
                                            batch["attention_mask"], batch["base_extra"], pad_token_id,
                                            args.max_new_tokens)
                f, u, _, kept = score_switch(gen, batch)
                k = len(batch["rows"])
                acc["first"] += f * k
                acc["full"] += u * k
                acc["kept"] += kept * k
                if roll:
                    rolled = {**batch,
                              "source_gold_toks": torch.roll(batch["source_gold_toks"], 1, dims=0),
                              "source_gold_len": torch.roll(batch["source_gold_len"], 1, dims=0)}
                    acc["roll_first"] += score_switch(gen, rolled)[0] * k
                acc["n"] += k
            n = max(acc["n"], 1)
            return {k: v / n for k, v in acc.items() if k != "n"} | {"n": acc["n"]}

        def line(label, r, extra=""):
            print(f"  {label:<34} first={r['first']:6.1%} full={r['full']:6.1%} "
                  f"base_kept={r['kept']:6.1%}{extra}", flush=True)

        # ---------------- phase 1b: connectivity + block-everything ----------------
        print(f"\n{'=' * 78}\n=== phase 1b: connectivity + block-everything\n{'=' * 78}")
        tel = {}
        zero_z = [{b: z[b] * 0.0 for b in blocks} for z in base_z_per_batch]
        blocked = run_cause(every_head, zero_z, True, telemetry=tel)
        for b in blocks:
            r = tel.get(b)
            print(f"    block {b:>2}: " + ("PATCH FN NEVER CALLED" if r is None else
                  f"prefill_calls={r['prefill_calls']} decode_skips={r['decode_skips']} "
                  f"seq_len={r['seq_len']} max|delta|={r['max_delta']:.4f}"))
        assert sum(r["prefill_calls"] for r in tel.values()) > 0, \
            "HEAD PATCH NEVER REACHED on a multi-token tensor; phases 2-4 are void."
        wrote = sum(1 for r in tel.values() if r["max_delta"] > 0)
        assert wrote == len(tel), \
            f"HEAD PATCH WROTE NOTHING at {len(tel) - wrote} of {len(tel)} blocks; phases 2-4 are void."
        line(f"question patch + ALL {len(every_head)} heads ZEROED", blocked)
        print(f"  (zeroing every traced head at the read column severs every path by which the patched "
              f"question can reach it after block {blocks[0]}, so this SHOULD fall to the clean floor. "
              f"If it does not, --blocks is too narrow or the read does not go through this position's "
              f"attention -- and that would be the finding.)")
        report["phase1b"] = {"telemetry": {str(b): tel.get(b) for b in blocks}, "blocked_all": blocked}

        # ---------------- phase 2: cumulative knockout ----------------
        print(f"\n{'=' * 78}\n=== phase 2: cumulative knockout under the question patch\n{'=' * 78}")
        unhooked = run_cause([], base_z_per_batch, False)
        line("unhooked", unhooked)
        full = run_cause([], base_z_per_batch, True)
        line("question patch only (k=0)", full, "   <-- the effect being traced")
        if full["first"] < 0.10:
            print(f"  !! the question patch at layer {args.patch_layer} barely moves the answer, so "
                  f"there is nothing downstream to trace. Pick a --patch_layer BELOW the handoff -- "
                  f"the sweep's question-residual column is ~100% for L<=16 on flags.")
        rand_heads = (random.Random(args.seed + 1)
                      .sample([(r["block"], r["head"]) for r in table], min(args.n_random, len(table)))
                      if args.n_random else [])
        knock = []
        for k in args.knockout_ks:
            if k > len(ranked):
                continue
            heads = [(r["block"], r["head"]) for r in ranked[:k]]
            r = run_cause(heads, base_z_per_batch, True)
            knock.append({"k": k, "kind": "top", "heads": heads, **r})
            line(f"restore top-{k} heads", r,
                 f"  (removed {max(full['first'] - r['first'], 0) / max(full['first'], 1e-9):5.1%} "
                 f"of the effect)")
        if args.n_random:
            r = run_cause(rand_heads, base_z_per_batch, True)
            knock.append({"k": args.n_random, "kind": "random", "heads": rand_heads, **r})
            line(f"restore {args.n_random} RANDOM heads (null)", r)
            same = next((x for x in knock if x["kind"] == "top" and x["k"] == args.n_random), None)
            if same is not None:
                print(f"  top-{args.n_random} vs random-{args.n_random} gap: "
                      f"{r['first'] - same['first']:+.1%}. Near zero means the ranking carries no "
                      f"information and this curve says nothing.")
        print("  (expect this curve to be FLAT even for real conduits: the sweep's span table shows "
              "the mechanism is redundant across blocks, and redundancy is exactly what a necessity "
              "measurement cannot see. Phase 3 is the one to read.)")
        report["phase2"] = {"unhooked": unhooked, "question_patch_only": full, "knockouts": knock}

        # ---------------- phase 3: sufficiency ----------------
        if not args.skip_sufficiency:
            print(f"\n{'=' * 78}\n=== phase 3: sufficiency -- patch ONLY these heads, no question patch"
                  f"\n{'=' * 78}")
            suff = []
            z0 = run_cause([], patched_z_per_batch, False)
            line("nothing patched (k=0)", z0, "   <-- must match the unhooked row")
            if abs(z0["first"] - unhooked["first"]) > 1e-9:
                print(f"  !! k=0 differs from unhooked -- a patch is leaking when no heads are "
                      f"selected; every number below is suspect.")
            ceiling = run_cause(every_head, patched_z_per_batch, False)
            if missing:
                line(f"ALL {len(every_head)} traced heads", ceiling,
                     f"   <-- PARTIAL ceiling: blocks {missing} untraced")
            else:
                line(f"ALL {len(every_head)} traced heads", ceiling,
                     f"   <-- full coverage, so this MUST equal {full['first']:.1%}")
                if abs(ceiling["first"] - full["first"]) > 0.05:
                    print(f"  !! it does not ({ceiling['first']:.1%} vs {full['first']:.1%}); every "
                          f"top-k row below is meaningless until that is fixed.")
            suff.append({"k": len(every_head), "kind": "all", **ceiling})
            for k in args.knockout_ks:
                if k > len(ranked):
                    continue
                heads = [(r["block"], r["head"]) for r in ranked[:k]]
                r = run_cause(heads, patched_z_per_batch, False)
                suff.append({"k": k, "kind": "top", "heads": heads, **r})
                line(f"patch top-{k} heads", r,
                     f"  ({r['first'] / max(ceiling['first'], 1e-9):5.1%} of this arm's ceiling)")
            if args.n_random:
                r = run_cause(rand_heads, patched_z_per_batch, False)
                suff.append({"k": args.n_random, "kind": "random", "heads": rand_heads, **r})
                line(f"patch {args.n_random} RANDOM heads (null)", r, "   <-- phase 2's null head set")
            report["phase3_sufficiency"] = {"ceiling": ceiling, "k0": z0, "arms": suff}

            # ---------------- phase 4: shuffled-donor control ----------------
            print(f"\n{'=' * 78}\n=== phase 4: shuffled-donor control (is the content row-specific?)"
                  f"\n{'=' * 78}")
            print("  Each row receives the head outputs ANOTHER row's DONOR QUESTION produced, and is "
                  "scored against its own donor gold (`own`) and against the donor row's (`donor`). "
                  "Every artefactual way phase 3 could succeed -- a mask not really restricted to k "
                  "heads, a perturbation knocking the residual off-distribution, a scorer counting "
                  "the wrong thing -- is indifferent to WHICH row supplied the values. Real transfer "
                  "is not: `own` must collapse and `donor` must rise.")
            ks = sorted({min(k, len(ranked)) for k in args.control_k if k > 0})
            arms = [(f"top-{k}", [(r["block"], r["head"]) for r in ranked[:k]]) for k in ks]
            arms.append((f"ALL {len(every_head)}", every_head))
            ctrl, seen_sets = [], set()
            for label, heads in arms:
                key_set = frozenset(heads)
                if not heads or key_set in seen_sets:
                    continue
                seen_sets.add(key_set)
                r = run_cause(heads, patched_z_per_batch, False, roll=True)
                straight = next((x["first"] for x in suff
                                 if x.get("k") == len(heads) and x["kind"] in ("top", "all")), None)
                print(f"  {label:>10} shuffled:  own={r['first']:6.1%} donor={r['roll_first']:6.1%} "
                      f"kept={r['kept']:6.1%}"
                      + (f"   (unshuffled first was {straight:.1%})" if straight is not None else ""))
                ctrl.append({"label": label, "n_heads": len(heads), "own": r["first"],
                             "donor": r["roll_first"], "base_kept": r["kept"],
                             "unshuffled_first": straight})
            confirmed = next((c for c in ctrl if c["donor"] > 0.5 and c["own"] < 0.2), None)
            if ctrl:
                if confirmed is not None:
                    print(f"\n  -> CONFIRMED at {confirmed['label']}: the answer follows the donor "
                          f"(own={confirmed['own']:.1%}, donor={confirmed['donor']:.1%}). THIS k is "
                          f"the localization number worth quoting, not the smallest k whose phase-3 "
                          f"number moved.")
                elif ctrl[0]["own"] > 0.5:
                    print(f"\n  !! the answer did NOT follow the donor -- `own` stayed at "
                          f"{ctrl[0]['own']:.1%} while receiving another row's values. Phase 3's curve "
                          f"is an artefact of perturbing this site, not transfer.")
                else:
                    best = max(ctrl, key=lambda c: c["donor"])
                    print(f"\n  -> partial: best donor rate {best['donor']:.1%} at {best['label']} "
                          f"(own={best['own']:.1%}). Report as 'carries the selection', not 'sufficient'.")
            report["phase4_shuffled_donor"] = ctrl

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
