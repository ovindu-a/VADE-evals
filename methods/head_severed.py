"""Severed test: does the entity the conduit heads deliver get LOOKED UP by a
downstream MLP, or do the heads already carry the answer?

ROME's "severed" causal tracing (Meng et al. 2022, rome.baulab.info), moved
onto the blocks 21-23 conduit of ATTRIBUTE_HEAD_EXPERIMENTS.md R8-R11.

THE DESIGN

R10 installs the SOURCE image's `common10` head outputs into the BASE run at the
last column of every forward (head_swap_vade.py's continuous patch) and the base
then answers about the source country at 88-98%. This script repeats exactly
that install, and additionally FREEZES chosen downstream components at the last
column to the values they had in the CLEAN BASE run:

    heads                 R10's arm, the reference (no freeze)
    heads+mlp[22-27]      every late MLP held at its base value
    heads+mlp[b]          one block at a time
    heads+attn[24-27]     the mirror: late ATTENTION held instead
    heads+attn[b]         (optional singles)
    full_image(+mlp[..])  the source residual at every image token at
                          --patch_layer, with/without the freeze: what the same
                          freeze does to a whole-entity swap

A frozen component can no longer react to the new entity. So:

  * cause collapses under heads+mlp[span]  -> the heads deliver something a late
    MLP has to turn into the answer: ENTITY in, lookup downstream.
  * cause survives                         -> the answer is already in the
    residual after the install: the heads (or attention reading them) carry it.
  * different attributes collapse under DIFFERENT single blocks -> the attribute
    is localized to a block, which no head-level experiment can show.

Freezing attn[24-27] is the mirror: if the late attention heads, not the MLPs,
are what reads the conduit's output, that arm collapses instead.

WHAT "FROZEN AT THE BASE VALUE" MEANS AT STEP t

Step 0 (the prefill's last column, i.e. the first answer token) is exact: the
base value is the clean base run's value there. After step 0 the patched run has
generated different tokens, so there is no base run with the same prefix.
`--freeze_steps all` (default) installs the clean base run's step-t value at
step t -- the same step-matched replay head_swap_vade uses for the donor, holding
the last step if the patched run outlasts it. `--freeze_steps first` freezes at
step 0 only and lets later steps compute freely. The FIRST-TOKEN columns are
exact under both and are the primary readout; full-answer text matching (VADE's
own matcher) is reported next to them.

WHICH VALUE A COMPONENT IS FROZEN TO (--freeze_values, any subset)

A component frozen at the BASE value does two things at once: it stops reacting
to the new entity, and it keeps writing the base country's own lookup. The
flags run (2026-09-24) cannot tell those apart -- under heads+mlp[25],
calling_code goes back to the base answer on 100% of pairs. Two more values
separate them:

  base    the clean base run's value (ROME's severed test; the default)
  mean    leave-one-out mean of that component over the attribute's OTHER base
          items asking the same template -- "stop reacting" with no single
          country's answer in it (mean ablation)
  other   a THIRD country's clean value (same template, base and label both
          different from this pair's), with that country's gold scored too

  * mean ~ base (both collapse)  -> the block is NEEDED: without its reaction
    to the new entity there is no transfer, whoever's answer it writes.
  * mean recovers, base does not -> the collapse was the base's answer being
    re-written, not a missing lookup.
  * other lands on the THIRD country's value -> the block WRITES the answer
    itself (a value writer, ROME's "MLP recall"), rather than enabling a
    lookup done elsewhere.

Arms frozen to mean/other are named `<arm>@mean` / `<arm>@other`; base keeps
the plain name so earlier summaries stay comparable.

SELF-CHECKS (run on the first batch, before any scored generation)

  1. read-back   the head install reads back what was installed (verify_readback).
  2. identity    freezing EVERY component any arm freezes, at its own base value,
                 on the clean base run, must reproduce the clean generation
                 BIT-FOR-BIT. Anything else means the freeze is addressing the
                 wrong column or step, and every number below is void.
  3. liveness    freezing to a DIFFERENT row's base values must move the
                 last-column logits. Logits, not text: a thresholded readout
                 absorbs perturbations (CLAUDE.md's verify_sites note).

Usage
-----
    python methods/head_severed.py --dry_run
    python methods/head_severed.py --n_pairs 16 --attributes language    # smoke
    python methods/head_severed.py --n_pairs 64                          # all attributes
    python methods/head_severed.py --freeze_attn_singles 24 25 26 27     # + attention singles
    python methods/head_severed.py --freeze_values base mean other       # what the frozen block writes
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import (COMMON10, DEFAULT_VADE_ROOT, MODEL_ID, build_batch,  # noqa: E402
                                    capture_donor, decode, entity_attributes, full_image_patch, load_assets,
                                    parse_heads, patched_generate, to_device, verify_readback)

KINDS = {"mlp": "mlp_output", "attn": "attn_output"}
FREEZE_VALUES = ("base", "mean", "other")


def parse_span(s):
    """'22-27' -> [22..27]; '24' -> [24]."""
    if "-" in str(s):
        lo, hi = str(s).split("-")
        out = list(range(int(lo), int(hi) + 1))
    else:
        out = [int(s)]
    assert out, f"empty span {s!r}"
    return out


def build_arms(args, head_blocks):
    """-> ordered list of (arm name, uses_heads, uses_full_image, [(kind, block), ...],
    freeze value mode). Every freezing arm is repeated once per --freeze_values."""
    arms = [("clean", False, False, []), ("heads", True, False, [])]
    mlp_span = parse_span(args.freeze_mlp_span) if args.freeze_mlp_span else []
    attn_span = parse_span(args.freeze_attn_span) if args.freeze_attn_span else []
    if mlp_span:
        arms.append((f"heads+mlp[{args.freeze_mlp_span}]", True, False, [("mlp", b) for b in mlp_span]))
    for b in args.freeze_mlp_singles:
        arms.append((f"heads+mlp[{b}]", True, False, [("mlp", b)]))
    if attn_span:
        arms.append((f"heads+attn[{args.freeze_attn_span}]", True, False, [("attn", b) for b in attn_span]))
    for b in args.freeze_attn_singles:
        arms.append((f"heads+attn[{b}]", True, False, [("attn", b)]))
    if args.full_image:
        arms.append(("full_image", False, True, []))
        if mlp_span:
            arms.append((f"full_image+mlp[{args.freeze_mlp_span}]", False, True, [("mlp", b) for b in mlp_span]))
    for name, uses_heads, _, freezes in arms:
        bad = [b for k, b in freezes if k == "attn" and b in head_blocks and uses_heads]
        assert not bad, (
            f"arm {name!r} freezes attention in block(s) {bad}, which hold installed heads -- the "
            f"attn_output freeze would overwrite the install itself. Freeze attention downstream only.")
    modes = list(getattr(args, "freeze_values", None) or ["base"])
    assert modes and all(m in FREEZE_VALUES for m in modes), f"--freeze_values must be from {FREEZE_VALUES}"
    out = []
    for name, uses_heads, uses_img, freezes in arms:
        if not freezes:
            out.append((name, uses_heads, uses_img, freezes, None))
            continue
        for m in modes:
            out.append((name if m == "base" else f"{name}@{m}", uses_heads, uses_img, freezes, m))
    return out


def pad_steps(t, n_steps):
    """[T, H] -> [n_steps, H], holding the last step (freeze_patches' own rule)."""
    import torch
    if t.shape[0] >= n_steps:
        return t[:n_steps]
    return torch.cat([t, t[-1:].expand(n_steps - t.shape[0], -1)])


def choose_thirds(rows, seed):
    """-> {row_index: (third row or None, same_template)} for the `other` freeze.

    The third must be a different country from both the base and the source,
    with a gold label different from both, so a first token matching it is
    unambiguous. Same template first (its last column is the same prompt text
    over a different image); any template of the attribute as a fallback."""
    out = {}
    for r in rows:
        rng = random.Random(seed * 1_000_003 + int(r["row_index"]))
        ok = [c for c in rows
              if c["base"] not in (r["base"], r["source"])
              and str(c["base_label"]) not in (str(r["base_label"]), str(r["source_label"]))]
        same = [c for c in ok if c["template_id"] == r["template_id"]]
        pool = same or ok
        out[r["row_index"]] = (rng.choice(pool) if pool else None, bool(same))
    return out


def mean_groups(rows, min_size=4):
    """-> {row_index: [row_index, ...]} for the `mean` freeze: every OTHER base
    item asking the same template (leave-one-out, so a row's own base never
    enters its mean). Below `min_size` such items the "mean" would be one or two
    specific countries -- an `other` freeze in disguise -- so it falls back to
    every other base item of the attribute, across templates."""
    out = {}
    for r in rows:
        others = [c for c in rows if c["base"] != r["base"]]
        same = [c["row_index"] for c in others if c["template_id"] == r["template_id"]]
        out[r["row_index"]] = same if len(same) >= min_size else [c["row_index"] for c in others]
        assert out[r["row_index"]], f"row {r['row_index']}: no other base item to average over"
    return out


def freeze_values_for(mode, chunk, store, thirds, means):
    """-> {(kind, block): [B, T, H]} frozen values for one batch.

    store[row_index][(kind, block)] = that row's clean [T, H]; means is the
    same shape, precomputed per row."""
    import torch
    keys = list(store[chunk[0]["row_index"]])
    src = []
    for r in chunk:
        ri = r["row_index"]
        if mode == "base":
            src.append(store[ri])
        elif mode == "mean":
            src.append(means[ri])
        elif mode == "other":
            third = thirds[ri][0]
            src.append(store[third["row_index"]] if third is not None else store[ri])
        else:
            raise ValueError(mode)
    n = max(v.shape[0] for d in src for v in d.values())
    return {k: torch.stack([pad_steps(d[k], n) for d in src]) for k in keys}


def row_means(rows, store, groups):
    """-> {row_index: {(kind, block): [T_all, H]}}: the leave-one-out mean, over
    each row's group, of the members' clean values padded to a common length."""
    import torch
    n = max(v.shape[0] for d in store.values() for v in d.values())
    out = {}
    for r in rows:
        members = groups[r["row_index"]]
        out[r["row_index"]] = {
            k: torch.stack([pad_steps(store[m][k], n).float() for m in members]).mean(0).to(v.dtype)
            for k, v in store[r["row_index"]].items()}
    return out


def capture_components(adapter, model, need, ids, mask, extra, max_new_tokens, pad_id):
    """Clean greedy generation that ALSO records, at the last column of every
    forward, each (kind, block) component's output. -> (generated [B, T_new],
    {(kind, block): [B, n_steps, H]}). One call gives the `clean` arm and every
    base value the freezes need."""
    import torch
    from methods.common.sites import InterventionSite
    sinks = {key: [] for key in need}
    handles = []
    for kind, b in need:
        module, _ = InterventionSite(KINDS[kind])._module_and_hook_kind(adapter, model, b + 1)

        def grab(mod, inputs, output, _sink=sinks[(kind, b)]):
            t = output[0] if isinstance(output, tuple) else output
            _sink.append(t[:, -1, :].detach())
        handles.append(module.register_forward_hook(grab))
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                 **to_device(extra, model.device, model.dtype),
                                 max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    finally:
        for h in handles:
            h.remove()
    return out[:, ids.shape[1]:].cpu(), {k: torch.stack(v, dim=1) for k, v in sinks.items()}


def freeze_patches(freezes, base_vals, freeze_steps="all"):
    """-> extra_patches for patched_generate: each (kind, block) component's
    last column is overwritten with its base value at the matching step.

    Uses common/sites.py's own post-hook registration (tuple-aware for self_attn),
    so the tensor frozen here is the one every other experiment reads and patches."""
    from methods.common.sites import InterventionSite
    patches = []
    for kind, b in freezes:
        vals = base_vals[(kind, b)]
        counter = [0]

        def fn(t, _vals=vals, _counter=counter):
            step = _counter[0]
            _counter[0] += 1
            if freeze_steps == "first" and step > 0:
                return t
            idx = min(step, _vals.shape[1] - 1)
            out = t.clone()
            out[:, -1, :] = _vals[:, idx, :].to(device=t.device, dtype=t.dtype)
            return out
        patches.append((InterventionSite(KINDS[kind]), b + 1, fn))
    return patches


def last_logits(adapter, model, ids, mask, extra, extra_patches=()):
    """One prefill forward with the given patches -> last-column logits [B, V]."""
    import torch
    handles = []
    for site, layer_idx, fn in extra_patches:
        handles.extend(site.register(adapter, model, adapter.get_decoder_layers(model), layer_idx, fn))
    try:
        with torch.no_grad():
            out = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                        **to_device(extra, model.device, model.dtype), logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return out.logits[:, -1].float().cpu()


def self_checks(adapter, model, batch, heads, head_dim, arms, max_new_tokens, pad_id, freeze_steps):
    """The three gates in the module docstring. Returns a dict of what was measured."""
    import torch
    need = sorted({f for a in arms for f in a[3]})
    clean, base_vals = capture_components(adapter, model, need, batch["base_ids"], batch["base_mask"],
                                          batch["base_extra"], max_new_tokens, pad_id)
    report = {}
    if any(a[1] for a in arms):
        blocks = sorted({b for b, _ in heads})
        z = capture_donor(adapter, model, blocks, batch["donor_ids"], batch["donor_mask"],
                          batch["donor_extra"], max_new_tokens, pad_id)
        report["readback_worst"] = verify_readback(adapter, model, heads, z, batch["base_ids"],
                                                   batch["base_mask"], batch["base_extra"],
                                                   max_new_tokens, pad_id, head_dim)
    if need:
        frozen = patched_generate(adapter, model, [], {}, batch["base_ids"], batch["base_mask"],
                                  batch["base_extra"], max_new_tokens, pad_id, head_dim,
                                  extra_patches=freeze_patches(need, base_vals, freeze_steps))
        assert torch.equal(frozen, clean), (
            "IDENTITY FAILED: freezing every component at its own base value changed the clean "
            "generation. The freeze is not addressing the column/step it captured -- every arm below "
            "would be void.")
        report["identity"] = "bit-exact"
        if batch["base_ids"].shape[0] > 1:
            rolled = {k: v.roll(1, dims=0) for k, v in base_vals.items()}
            ref = last_logits(adapter, model, batch["base_ids"], batch["base_mask"], batch["base_extra"])
            moved = last_logits(adapter, model, batch["base_ids"], batch["base_mask"], batch["base_extra"],
                                freeze_patches(need, rolled, freeze_steps))
            delta = float((moved - ref).abs().max())
            assert delta > 1e-3, (
                f"LIVENESS FAILED: freezing to another row's values moved the logits by only {delta:.2e}; "
                f"the freeze hooks are not connected")
            report["liveness_max_logit_delta"] = delta
    return report


# ---------------------------------------------------------------------------
# Rows and scoring
# ---------------------------------------------------------------------------

def load_cause_rows(vade_root, model_slug, entity, attribute, split, n_pairs, seed, allow_unpruned):
    """Cause rows (rule == match_source) for one attribute, one row per sampled
    (base, source) pair. Pruned tuples (the model reads the SOURCE correctly)
    unless --allow_unpruned."""
    pruned = os.path.join(vade_root, "models", model_slug, entity, "tuples", attribute, f"{split}.jsonl")
    raw = os.path.join(vade_root, "data", entity, "tuples", attribute, f"{split}.jsonl")
    path = pruned if os.path.exists(pruned) else (raw if allow_unpruned else None)
    assert path, f"no pruned tuples at {pruned}; pass --allow_unpruned to use {raw}"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r["rule"] == "match_source" and r["queried"] == attribute]
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r["base"], r["source"])].append(r)
    rng = random.Random(seed)
    pairs = sorted(by_pair)
    rng.shuffle(pairs)
    picked = [rng.choice(by_pair[p]) for p in pairs[:n_pairs]]
    return sorted(picked, key=lambda r: r["row_index"]), path


def as_job(r):
    """head_swap_vade.build_batch's job shape, donor asking the SAME question."""
    return {"base": r["base"], "source": r["source"], "queried": r["queried"], "template_id": r["template_id"],
            "donor_attribute": r["queried"], "donor_template_id": r["template_id"],
            "rows": [(r["target_attribute"], r["row_index"])]}


def score(texts, toks, rows, lookup, tokenizer, matcher, thirds=None):
    """Per row: first-token vs source/base gold (exact) and VADE text match.
    `thirds` (one label or None per row) adds the same two checks against a
    third country's gold, for the `other` freeze."""
    from methods.common.targets import derive_gold_token_ids
    out = []
    thirds = thirds or [None] * len(rows)
    for r, text, t, third in zip(rows, texts, toks.tolist(), thirds):
        prefill = lookup[r["queried"]][r["template_id"]]["prefill"]
        sg = derive_gold_token_ids(tokenizer, prefill, str(r["source_label"]))
        bg = derive_gold_token_ids(tokenizer, prefill, str(r["base_label"]))
        s_txt, b_txt = matcher(text, str(r["source_label"])), matcher(text, str(r["base_label"]))
        rec = {"distinct_first": bool(sg and bg and sg[0] != bg[0]),
               "src_first": bool(sg) and t[0] == sg[0], "base_first": bool(bg) and t[0] == bg[0],
               "src_text": s_txt, "base_text": b_txt, "other_text": not (s_txt or b_txt)}
        if third is not None:
            tg = derive_gold_token_ids(tokenizer, prefill, str(third))
            t_txt = matcher(text, str(third))
            rec.update({"third_label": str(third),
                        "distinct3_first": bool(rec["distinct_first"] and tg and tg[0] not in (sg[0], bg[0])),
                        "third_first": bool(tg) and t[0] == tg[0], "third_text": t_txt,
                        "other_text": not (s_txt or b_txt or t_txt)})
        out.append(rec)
    return out


def aggregate(scored):
    """[score dicts] -> rates. First-token rates are over rows whose base/source
    golds differ in their first token (otherwise a hit is uninformative)."""
    d = [s for s in scored if s["distinct_first"]]
    n = len(scored)
    out = {"n": n, "n_distinct_first": len(d),
           "src_first": sum(s["src_first"] for s in d) / max(len(d), 1),
           "base_first": sum(s["base_first"] for s in d) / max(len(d), 1),
           "src_text": sum(s["src_text"] for s in scored) / max(n, 1),
           "base_text": sum(s["base_text"] for s in scored) / max(n, 1),
           "other_text": sum(s["other_text"] for s in scored) / max(n, 1)}
    t3 = [s for s in scored if "third_first" in s]
    if t3:
        # Third-country first-token rate over rows whose three golds all differ at token 0.
        d3 = [s for s in t3 if s["distinct3_first"]]
        out.update({"n_distinct3_first": len(d3),
                    "third_first": sum(s["third_first"] for s in d3) / max(len(d3), 1),
                    "third_text": sum(s["third_text"] for s in t3) / max(len(t3), 1)})
    return out


def print_table(summary):
    for attr, arms in summary.items():
        ref = arms.get("heads", {}).get("src_first") or 0.0
        print(f"\n=== {attr} ===")
        print(f"{'arm':>34} {'n':>4} {'src 1st':>8} {'base 1st':>9} {'kept':>6} "
              f"{'src txt':>8} {'base txt':>9} {'other':>6} {'3rd 1st':>8} {'3rd txt':>8}")
        for arm, s in arms.items():
            kept = s["src_first"] / ref if ref else float("nan")
            third = (f"{s['third_first']:8.1%} {s['third_text']:8.1%}" if "third_first" in s
                     else f"{'-':>8} {'-':>8}")
            print(f"{arm:>34} {s['n']:>4} {s['src_first']:8.1%} {s['base_first']:9.1%} {kept:6.2f} "
                  f"{s['src_text']:8.1%} {s['base_text']:9.1%} {s['other_text']:6.1%} {third}")
    print("\n  kept = src 1st / src 1st of `heads`. <<1 under heads+mlp[span] = the lookup is downstream.")
    print("  @mean ~ base -> the block is needed; @mean recovers -> base's answer was being re-written;")
    print("  @other lands on the 3rd country -> the block writes the answer itself. 'other' excludes 3rd.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--heads", default=COMMON10)
    ap.add_argument("--attributes", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--n_pairs", type=int, default=64, help="Cause rows per attribute, one per (base, source).")
    ap.add_argument("--freeze_mlp_span", default="22-27", help="'' to skip the all-at-once MLP arm.")
    ap.add_argument("--freeze_mlp_singles", type=int, nargs="*", default=[22, 23, 24, 25, 26, 27])
    ap.add_argument("--freeze_attn_span", default="24-27", help="'' to skip the attention mirror.")
    ap.add_argument("--freeze_attn_singles", type=int, nargs="*", default=[])
    ap.add_argument("--full_image", action="store_true", help="Add full_image and full_image+mlp[span].")
    ap.add_argument("--patch_layer", type=int, default=21, help="full_image arms only.")
    ap.add_argument("--freeze_steps", choices=["all", "first"], default="all")
    ap.add_argument("--freeze_values", nargs="+", choices=list(FREEZE_VALUES), default=["base"],
                    help="What a frozen component is held at: the clean base value (ROME), the "
                         "leave-one-out mean over other base items (mean ablation), or a third "
                         "country's value. Any subset; each freezing arm runs once per value.")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    head_blocks = sorted({b for b, _ in heads})
    arms = build_arms(args, head_blocks)
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    attributes = args.attributes or [a for a in entity_attributes(gt) if a in lookup]
    model_slug = args.model_id.split("/")[-1]
    rows = {}
    for a in attributes:
        rows[a], path = load_cause_rows(args.vade_root, model_slug, args.entity, a, args.split, args.n_pairs,
                                        args.seed, args.allow_unpruned)
        print(f"  {a}: {len(rows[a])} cause rows from {path}")
    default_dir = args.entity if args.freeze_values == ["base"] else \
        f"{args.entity}__freeze-{'-'.join(args.freeze_values)}"
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "head_severed", default_dir)
    n_gen = sum(len(v) for v in rows.values()) * len(arms)
    print(f"[head_severed] {args.entity}: heads {', '.join(f'{b}.{h}' for b, h in heads)}")
    print(f"  arms ({len(arms)}): {[a[0] for a in arms]}")
    print(f"  freeze_steps={args.freeze_steps}; ~{n_gen} scored generations -> {out_dir}")
    if args.dry_run:
        for a in attributes:
            for r in rows[a][:2]:
                print(f"    {a}: {r['base']}->{r['source']} {r['template_id']} "
                      f"{r['base_label']!r}->{r['source_label']!r}")
            if "mean" in args.freeze_values:
                sizes = [len(g) for g in mean_groups(rows[a]).values()]
                print(f"    {a}: mean freeze averages {min(sizes)}-{max(sizes)} other base items per row")
            if "other" in args.freeze_values:
                th = choose_thirds(rows[a], args.seed)
                found = [t for t in th.values() if t[0] is not None]
                print(f"    {a}: third country found for {len(found)}/{len(th)} rows "
                      f"({sum(t[1] for t in found)} from the same template)")
        print("Grid valid.")
        return

    import torch
    from methods.adapters.registry import get_adapter
    from methods.head_cross import vade_matcher

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    head_dim = adapter.hidden_size(model) // adapter.n_attention_heads(model)
    n_layers = len(adapter.get_decoder_layers(model))
    for a in arms:
        assert all(0 <= b < n_layers for _, b in a[3]), f"freeze blocks must be in 0..{n_layers - 1}"
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    image_token_id = adapter.image_token_id(model, processor)
    matcher = vade_matcher(args.vade_root)

    probe = build_batch([as_job(r) for r in rows[attributes[0]][:4]], processor, items, entity_dir, lookup, pad_id)
    checks = self_checks(adapter, model, probe, heads, head_dim, arms, args.max_new_tokens, pad_id,
                         args.freeze_steps)
    print(f"  self-checks: {checks}")

    os.makedirs(out_dir, exist_ok=True)
    scored = defaultdict(lambda: defaultdict(list))
    need = sorted({f for a in arms for f in a[3]})
    modes = sorted({a[4] for a in arms if a[4]})
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
        for attr in attributes:
            batches = [rows[attr][i:i + args.batch_size] for i in range(0, len(rows[attr]), args.batch_size)]
            # PASS 1: every row's clean generation and clean component values. The mean
            # and third-country freezes need OTHER rows' values, so these come first.
            clean_toks, store = {}, {}
            for chunk in batches:
                batch = build_batch([as_job(r) for r in chunk], processor, items, entity_dir, lookup, pad_id)
                clean, base_vals = capture_components(adapter, model, need, batch["base_ids"], batch["base_mask"],
                                                      batch["base_extra"], args.max_new_tokens, pad_id)
                for i, r in enumerate(chunk):
                    clean_toks[r["row_index"]] = clean[i]
                    store[r["row_index"]] = {k: v[i].cpu() for k, v in base_vals.items()}
            thirds = choose_thirds(rows[attr], args.seed) if "other" in modes else {}
            means = row_means(rows[attr], store, mean_groups(rows[attr])) if ("mean" in modes and need) else {}
            if thirds:
                n_same = sum(t[1] for t in thirds.values() if t[0] is not None)
                print(f"  {attr}: third country from the same template for {n_same}/{len(thirds)} rows", flush=True)

            # PASS 2: the arms.
            for start, chunk in zip(range(0, len(rows[attr]), args.batch_size), batches):
                batch = build_batch([as_job(r) for r in chunk], processor, items, entity_dir, lookup, pad_id)
                donor_z = capture_donor(adapter, model, head_blocks, batch["donor_ids"], batch["donor_mask"],
                                        batch["donor_extra"], args.max_new_tokens, pad_id)
                vals = {m: freeze_values_for(m, chunk, store, thirds, means) for m in modes} if need else {}
                img_patches = None
                for name, uses_heads, uses_img, freezes, mode in arms:
                    if name == "clean":
                        n = max(clean_toks[r["row_index"]].shape[0] for r in chunk)
                        toks = torch.stack([torch.nn.functional.pad(clean_toks[r["row_index"]],
                                                                    (0, n - clean_toks[r["row_index"]].shape[0]),
                                                                    value=pad_id) for r in chunk])
                    else:
                        extra = freeze_patches(freezes, vals[mode], args.freeze_steps) if freezes else []
                        if uses_img:
                            img_patches = img_patches or full_image_patch(adapter, model, batch, args.patch_layer,
                                                                          image_token_id)
                            extra = extra + img_patches
                        toks = patched_generate(adapter, model, heads if uses_heads else [], donor_z,
                                                batch["base_ids"], batch["base_mask"], batch["base_extra"],
                                                args.max_new_tokens, pad_id, head_dim, extra_patches=extra)
                    texts = decode(processor, toks)
                    third_rows = [thirds[r["row_index"]][0] if mode == "other" else None for r in chunk]
                    sc = score(texts, toks, chunk, lookup, processor.tokenizer, matcher,
                               thirds=[t["base_label"] if t is not None else None for t in third_rows])
                    for r, text, s_, t in zip(chunk, texts, sc, third_rows):
                        scored[attr][name].append(s_)
                        fh.write(json.dumps({"attribute": attr, "row_index": r["row_index"], "arm": name,
                                             "freeze_value": mode, "base": r["base"], "source": r["source"],
                                             "third": None if t is None else t["base"],
                                             "template_id": r["template_id"], "generated_text": text,
                                             **s_}, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {attr}: {min(start + args.batch_size, len(rows[attr]))}/{len(rows[attr])}", flush=True)

    summary = {a: {arm: aggregate(v) for arm, v in arms_.items()} for a, arms_ in scored.items()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"heads": args.heads, "freeze_steps": args.freeze_steps,
                   "freeze_values": args.freeze_values, "self_checks": checks,
                   "arms": [a[0] for a in arms], "by_attribute": summary}, f, indent=2)
    print_table(summary)
    print(f"\nwrote {out_dir}/rows.jsonl and summary.json")


if __name__ == "__main__":
    main()
