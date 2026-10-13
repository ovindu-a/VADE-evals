"""E9: are Park et al.'s Visual Retrieval Heads the stage BEFORE the entity movers?

Park et al., "Retrieval Heads Meet Vision" (arXiv 2608.27417) find ~20 heads in
Qwen2.5-VL-7B layers 14-19 whose attention from the generated box tokens lands
inside a referred object's bounding box, and whose zeroing collapses grounding
(86.2% -> 7.5%). Our entity movers (common10, blocks 21-23) read identity mostly
off BACKGROUND image tokens (R16.1). This asks whether entity recall from a flag
depends on the VRHs at all.

THREE MEASUREMENTS, ONE MODEL LOAD

  0. INDEX CHECK (before anything is ablated). Their list is only usable here if
     their (layer, head) indices mean the same heads as ours. Re-measure their
     score in THIS code: a grounding prompt on our flags ("locate the flag ...
     bbox"), greedy box generation, then one eager forward over prompt + box with
     attention captured, and per head the attention mass from the generated
     positions onto the flag's own image tokens (flag_only), averaged. If the
     indices line up, their heads sit in the top percentiles of all 784; under
     the wrong convention (off by one layer and/or head) they look like random
     heads. Every list variant in published_heads.json is scored, so a 0- vs
     1-based ambiguity is decided by data rather than assumed.

  1. RECALL. Each VADE template question on every flag, generated greedily, scored
     with VADE's own matcher against the stored label -- clean, and with each
     ablated set's outputs ZEROED (o_proj input columns, every position, every
     step: Park's intervention).

  2. THE ENTITY AT THE MOVERS. In the same generation's prefill, the summed direct
     write of the ten movers at the last prompt column, projected on every
     flag's name token (dla.py's direction; the frozen norm scale is a positive
     per-row scalar, so ranks do not need it). Rank of THIS flag's name among all
     84 (1 = best). If the VRHs feed the movers, this rank degrades under their
     ablation; if identity is read some other way, it does not.

ARMS
    clean
    zero:<list>          each published list variant for this model
    zero:random<i>       --n_random block-matched draws per list (same number of
                         heads in each block, never a head of the list itself)
    zero:common10        our own movers -- a reference for what "the entity path
                         is cut" looks like on these two readouts. Necessity, so
                         expect redundancy to soften it (R10 is a sufficiency
                         result); the mover-rank column is undefined for it.

READING IT
    recall and mover rank both collapse   -> grounding is upstream of the entity
                                             path: Park's heads are stage 1 here
    both survive (vs random)              -> identifying a flag does not need region
                                             retrieval: consistent with movers reading
                                             the background (R16.1)
    recall drops, mover rank survives     -> the VRHs matter downstream of identity
                                             (e.g. for reading the question), not as
                                             its source

Usage
-----
    python methods/vrh_ablation.py --dry_run
    python methods/vrh_ablation.py --limit_items 8 --n_templates 1 --index_items 4    # smoke
    python methods/vrh_ablation.py                                                     # full
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

from methods.head_overlap import PUBLISHED, load_published  # noqa: E402
from methods.head_swap_vade import (COMMON10, DEFAULT_VADE_ROOT, MODEL_ID, entity_attributes,  # noqa: E402
                                    load_assets, parse_heads)

# Qwen2.5-VL's own grounding phrasing (its model card): absolute-coordinate JSON boxes.
GROUNDING_PROMPT = "Locate the flag in this image and output its bbox coordinates in JSON format."


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------

def block_matched_random(heads, n_heads, rng, exclude=()):
    """Same number of heads per block as `heads`, none of `heads` or `exclude`."""
    banned = set(heads) | set(exclude)
    by_block = defaultdict(int)
    for b, _ in heads:
        by_block[b] += 1
    out = []
    for b, c in sorted(by_block.items()):
        free = [h for h in range(n_heads) if (b, h) not in banned]
        assert len(free) >= c, f"block {b}: only {len(free)} heads left to draw {c} from"
        out += [(b, h) for h in rng.sample(free, c)]
    return sorted(out)


def zero_columns(t, cols):
    """t [B, T, W] with `cols` zeroed at every position (a copy)."""
    out = t.clone()
    out[..., cols] = 0
    return out


def head_cols(heads, block, head_dim):
    return [c for b, h in heads if b == block for c in range(h * head_dim, (h + 1) * head_dim)]


def box_mass(attn, query_cols, box_cols):
    """attn [n_heads, Q_all, K] (one layer, one row) -> [n_heads] mean over
    `query_cols` of the attention mass on `box_cols`."""
    a = attn[:, query_cols][:, :, box_cols]
    return a.sum(-1).mean(-1)


def percentile_of(scores, heads):
    """scores {(b, h): s} -> {head: percentile (100 = highest)} and the mean."""
    ordered = sorted(scores.values())
    n = len(ordered)

    def pct(v):
        return 100.0 * sum(x <= v for x in ordered) / n
    per = {f"{b}.{h}": pct(scores[(b, h)]) for b, h in heads if (b, h) in scores}
    return per, (sum(per.values()) / len(per) if per else None)


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

def register_zero(adapter, model, heads, head_dim):
    """Forward pre-hooks on o_proj zeroing `heads` at every position and step."""
    import torch
    handles = []
    for b in sorted({b for b, _ in heads}):
        cols = torch.tensor(head_cols(heads, b, head_dim), dtype=torch.long)

        def pre(_m, args, _cols=cols):
            return (zero_columns(args[0], _cols.to(args[0].device)),) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(pre))
    return handles


def register_mover_capture(adapter, model, movers, sink):
    """Pre-hooks (registered AFTER any zeroing hook, so they read what o_proj gets)
    recording the movers' blocks' o_proj input at the last column of the FIRST
    forward only -- the prefill's readout column."""
    handles = []
    for b in sorted({b for b, _ in movers}):
        def grab(_m, args, _b=b):
            if _b not in sink:
                sink[_b] = args[0][:, -1].detach().float().cpu()
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(grab))
    return handles


def mover_write(adapter, model, z_by_block, movers, head_dim):
    """Summed direct write of the movers -> [B, H] float32 (z_h @ W_O[:, h]^T)."""
    import torch
    total = None
    for b, z in z_by_block.items():
        cols = torch.tensor(head_cols(movers, b, head_dim), dtype=torch.long)
        w_o = adapter.get_attn_head_output_module(model, b).weight.detach().float().cpu()   # [H, H_in]
        c = z[:, cols] @ w_o[:, cols].T
        total = c if total is None else total + c
    return total


def pack_left(seqs, pad_id):
    import torch
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = s
        mask[i, n - len(s):] = 1
    return ids, mask


def index_check(adapter, model, processor, items, keys, entity_dir, box_flat, n_layers, n_heads,
                max_new_tokens=48):
    """-> ({(b, h): mean box mass}, example generations). See module docstring (0)."""
    import torch
    from PIL import Image
    from methods.head_swap_vade import build_prompt, to_device
    image_token_id = adapter.image_token_id(model, processor)
    totals = defaultdict(float)
    examples = []
    for key in keys:
        with Image.open(os.path.join(entity_dir, items[key]["image"])) as im:
            p = build_prompt(processor, im.convert("RGB"), GROUNDING_PROMPT, "")
        ids = p["input_ids"]
        extra = {"pixel_values": p["pixel_values"], "image_grid_thw": p["image_grid_thw"]}
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(model.device), **to_device(extra, model.device, model.dtype),
                                 max_new_tokens=max_new_tokens, do_sample=False)
            full = gen[:, :]
            out = model(input_ids=full, **to_device(extra, model.device, model.dtype),
                        output_attentions=True, use_cache=False, logits_to_keep=1)
        n_prompt = ids.shape[1]
        query_cols = list(range(n_prompt, full.shape[1])) or [n_prompt - 1]
        img_cols = (ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
        box_cols = [img_cols[i] for i in box_flat]
        assert len(out.attentions) == n_layers, "expected one attention map per decoder block"
        for b, a in enumerate(out.attentions):
            m = box_mass(a[0].float().cpu(), query_cols, box_cols)
            for h in range(n_heads):
                totals[(b, h)] += float(m[h]) / len(keys)
        examples.append({"item": key, "generated": processor.tokenizer.decode(gen[0, n_prompt:],
                                                                             skip_special_tokens=True)})
    return dict(totals), examples


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--published", default=PUBLISHED)
    ap.add_argument("--lists", nargs="+", default=None,
                    help="Published list names to ablate (default: every list for this model whose name "
                         "starts with 'park_vrh').")
    ap.add_argument("--movers", default=COMMON10)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--attributes", nargs="+", default=None)
    ap.add_argument("--n_templates", type=int, default=2, help="VADE templates per attribute (sorted ids).")
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--index_items", type=int, default=16, help="Flags used for the index check (0 = skip).")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    published = load_published(args.published, args.model_id)
    names = args.lists or [n for n in published if n.startswith("park_vrh")]
    missing = [n for n in names if n not in published]
    assert not missing, f"lists {missing} not in {args.published} for {args.model_id}; have {list(published)}"
    if not names:
        raise SystemExit(f"No published head list for {args.model_id} in {args.published} -- fill in park_vrh "
                         f"first (see that file's schema in methods/head_overlap.py).")
    movers = parse_heads(args.movers)
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    attributes = args.attributes or [a for a in entity_attributes(gt) if a in lookup]
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    box_flat = min(loc["object_token_indices"].values(), key=lambda v: len(v["flat"]))["flat"]
    templates = {a: sorted(lookup[a])[:args.n_templates] for a in attributes}
    jobs = [(k, a, t) for k in keys for a in attributes for t in templates[a]]

    rng = random.Random(args.seed)
    n_heads_guess = 28
    arms = [("clean", [])]
    for n in names:
        arms.append((f"zero:{n}", published[n]["heads"]))
        for i in range(args.n_random):
            arms.append((f"zero:{n}:random{i}", block_matched_random(published[n]["heads"], n_heads_guess, rng,
                                                                     exclude=movers)))
    arms.append(("zero:common10", movers))
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "vrh_ablation", args.entity)
    print(f"[vrh_ablation] {args.entity}: {len(keys)} items x {len(attributes)} attributes x "
          f"{args.n_templates} templates = {len(jobs)} generations per arm, {len(arms)} arms -> {out_dir}")
    for name, hs in arms:
        print(f"  {name:<28} ({len(hs):>2}) {' '.join(f'{b}.{h}' for b, h in hs)}")
    if args.dry_run:
        for k in keys:
            assert os.path.exists(os.path.join(entity_dir, items[k]["image"])), f"missing image for {k}"
        print(f"  object footprint for the index check: {len(box_flat)} image tokens")
        print("Grid valid.")
        return

    import torch
    from PIL import Image
    from methods.adapters.registry import get_adapter
    from methods.head_cross import vade_matcher
    from methods.head_swap_vade import build_prompt, decode, to_device
    from methods.head_vocab_projection import build_candidates, candidate_rank

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(device=args.device, attn_implementation="eager")
    tok = processor.tokenizer
    n_layers = len(adapter.get_decoder_layers(model))
    n_heads = adapter.n_attention_heads(model)
    assert n_heads == n_heads_guess, f"random draws assumed {n_heads_guess} heads; model has {n_heads}"
    head_dim = adapter.hidden_size(model) // n_heads
    pad_id = tok.pad_token_id or tok.eos_token_id
    matcher = vade_matcher(args.vade_root)
    os.makedirs(out_dir, exist_ok=True)
    report = {"model_id": args.model_id, "entity": args.entity, "arms": {n: [f"{b}.{h}" for b, h in hs]
                                                                         for n, hs in arms}}

    # 0. Index check.
    if args.index_items:
        scores, examples = index_check(adapter, model, processor, items, keys[:args.index_items], entity_dir,
                                       box_flat, n_layers, n_heads)
        report["index_check"] = {"examples": examples, "lists": {}}
        print("\n  INDEX CHECK: percentile of each list's heads by box-attention mass (100 = top of all "
              f"{n_layers * n_heads}); a correctly indexed VRH list should sit near the top")
        for n in published:
            per, mean = percentile_of(scores, published[n]["heads"])
            report["index_check"]["lists"][n] = {"per_head_percentile": per, "mean_percentile": mean}
            print(f"    {n:<24} mean percentile {mean:5.1f}")
        top = sorted(scores, key=lambda x: -scores[x])[:20]
        report["index_check"]["our_top20"] = [f"{b}.{h}" for b, h in top]
        print(f"    our top-20 by box mass: {' '.join(f'{b}.{h}' for b, h in top)}")
        for e in examples[:3]:
            print(f"    e.g. {e['item']}: {e['generated']!r}")

    # 1-2. Recall + mover rank, per arm.
    ent_ids, _, _ = build_candidates(tok, {k: items[k] for k in keys}, [], lookup, 0)
    all_ent = sorted({i for v in ent_ids.values() for i in v})
    mover_blocks = sorted({b for b, _ in movers})
    with torch.no_grad():
        name_dirs = adapter.logit_direction(model, torch.tensor(all_ent, device=model.device)).cpu()   # [N, H]
    images = {}
    for k in keys:
        with Image.open(os.path.join(entity_dir, items[k]["image"])) as im:
            images[k] = im.convert("RGB")
    agg = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
        for arm, heads in arms:
            for start in range(0, len(jobs), args.batch_size):
                chunk = jobs[start:start + args.batch_size]
                prompts = [build_prompt(processor, images[k], lookup[a][t]["question"], lookup[a][t]["prefill"])
                           for k, a, t in chunk]
                ids, mask = pack_left([p["input_ids"][0] for p in prompts], pad_id)
                extra = {"pixel_values": torch.cat([p["pixel_values"] for p in prompts]),
                         "image_grid_thw": torch.cat([p["image_grid_thw"] for p in prompts])}
                sink = {}
                handles = register_zero(adapter, model, heads, head_dim) if heads else []
                handles += register_mover_capture(adapter, model, movers, sink)
                try:
                    gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                         **to_device(extra, model.device, model.dtype),
                                         max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad_id)
                finally:
                    for h in handles:
                        h.remove()
                texts = decode(processor, gen[:, ids.shape[1]:].cpu())
                assert set(sink) == set(mover_blocks), "mover capture missed a block"
                scores = mover_write(adapter, model, sink, movers, head_dim) @ name_dirs.T          # [B, N]
                for i, ((k, a, t), text) in enumerate(zip(chunk, texts)):
                    correct = bool(matcher(text, str(items[k][a])))
                    rank = (None if arm == "zero:common10" else
                            candidate_rank(scores[i].new_zeros(max(all_ent) + 1).index_copy(
                                0, torch.tensor(all_ent), scores[i]), ent_ids[k], all_ent)["rank"])
                    agg[arm][a].append((correct, rank))
                    fh.write(json.dumps({"arm": arm, "item": k, "attribute": a, "template_id": t,
                                         "generated_text": text, "correct": correct, "mover_name_rank": rank},
                                        ensure_ascii=False) + "\n")
            print(f"  {arm}: done", flush=True)

    summary = {}
    for arm, by_attr in agg.items():
        rows = [x for v in by_attr.values() for x in v]
        ranks = [r for _, r in rows if r is not None]
        summary[arm] = {"accuracy": sum(c for c, _ in rows) / len(rows),
                        "by_attribute": {a: sum(c for c, _ in v) / len(v) for a, v in by_attr.items()},
                        "mover_rank_mean": (sum(ranks) / len(ranks)) if ranks else None,
                        "mover_rank1": (sum(r == 1 for r in ranks) / len(ranks)) if ranks else None,
                        "n": len(rows)}
    report["summary"] = summary
    report["n_names"] = len(all_ent)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'arm':<28} {'acc':>6} " + " ".join(f"{a[:10]:>10}" for a in attributes)
          + f" {'mover rank':>11} {'rank=1':>7}")
    for arm, s in summary.items():
        r = f"{s['mover_rank_mean']:11.2f} {s['mover_rank1']:7.1%}" if s["mover_rank_mean"] is not None \
            else f"{'-':>11} {'-':>7}"
        print(f"{arm:<28} {s['accuracy']:6.1%} " + " ".join(f"{s['by_attribute'][a]:10.1%}" for a in attributes)
              + f" {r}")
    print(f"\n  mover rank: this flag's name among {len(all_ent)} name tokens in the movers' summed direct write "
          f"(1 = best, chance {(len(all_ent) + 1) / 2:.1f}).")
    print(f"wrote {out_dir}/rows.jsonl and summary.json")


if __name__ == "__main__":
    main()
