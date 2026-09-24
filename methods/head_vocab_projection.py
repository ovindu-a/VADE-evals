"""What does each selected head WRITE, in vocabulary space -- and from WHERE?

Geva et al. 2023 ("Dissecting recall", arXiv:2304.14767) and Chughtai et al.
2024 ("Summing up the facts", arXiv:2402.07321), applied to the heads R1/R8 of
ATTRIBUTE_HEAD_EXPERIMENTS.md localized. Read-only, one forward pass per row, no
patching.

THE QUESTION

R9/R11 say the blocks 21-23 conduit (`common10`) carries the ENTITY, not the
answer, and R1 says the blocks 19-21 router carries WHICH ATTRIBUTE was asked
with ~0 direct logit effect. Both claims were measured causally or through
variance. This asks the complementary, mechanism-level question: what does each
head's write look like when read through the model's OWN unembedding?

  entity mover     its write, unembedded, promotes THIS item's NAME ('Mexico'),
                   whichever question was asked
  attribute        its write promotes the value of the QUERIED attribute
  extractor        ('Spanish' when asked the language, 'Mexico City' when asked
                   the capital) -- Geva's "extraction" event, and the thing a
                   selective VADE edit would need
  relation head    reads the QUESTION tokens rather than the image
  (Chughtai)

WHAT IS COMPUTED, PER (row, head)

The head's contribution to the residual stream at the last prompt column is
exactly  c_h = z_h @ W_O[:, h]^T  (o_proj has no bias -- asserted). Pushed
through the final norm with dla.py's FROZEN scale (the rsqrt of the real final
residual, so the map is linear and exact) and the unembedding:

    logits_h = s * W_U diag(norm.weight) c_h                      [vocab]

From that:

  top tokens         the full-vocab top-k, decoded. Geva's "extraction rate" is
                     whether the top-1 is the attribute value.
  entity rank        rank of this item's name token among every item's name
                     token (1 = the head promotes THIS entity more than any
                     other). Chance = (n+1)/2. Reported with a z-score.
  value rank [attr]  the same, among the distinct values of EACH attribute --
                     for all attributes, not just the queried one, so a head that
                     promotes the capital while the language was asked is visible.
  top-1 class        own_value(queried) > own_entity > own_value(other attr) >
                     other_entity > other_value > other.

and the SOURCE-POSITION split (Chughtai's per-source DLA): z_h at the last query
is exactly sum_k A[h, q, k] V_h[k], so the write splits into one term per key
group, which sum back to c_h by construction:

    system | object | image_background | question | assistant_prefix

For each group: attention mass, ||write||, and its direct logit along the
entity direction (own name minus the mean of the other names) and along the
queried value direction (own value minus the mean of the other values).
Reconstruction of A@V against o_proj's actual input is ASSERTED per head, the
same check head_token_ablation.py uses -- a wrong GQA mapping would otherwise
produce plausible groups.

READING IT

This is DIRECT paths only, like dla.py: a head that matters by being read by a
later MLP (the entity-then-lookup hypothesis) shows its entity in vocab space and
little of the value. That is the point of running it on the conduit -- if
`common10` promotes names and not values, and the late MLPs are where the value
appears, the lookup is downstream (head_severed.py tests that causally).

Known limit: first-token candidates collide ('South' for South Africa / South
Korea; calling codes share leading digits, 10 candidates at most). Collisions
are reported, and the rank is taken over DISTINCT token ids.

Usage
-----
    python methods/head_vocab_projection.py --dry_run
    python methods/head_vocab_projection.py --limit_items 12          # smoke
    python methods/head_vocab_projection.py                           # all items x attributes
    python methods/head_vocab_projection.py --head_sets "c10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import (COMMON10, DEFAULT_VADE_ROOT, MODEL_ID, entity_attributes,  # noqa: E402
                                    load_assets, resolve_head_sets)

# R1's top-8 router heads (attr_head_trace, blocks 19-21). 21.5 is in both sets.
ROUTER8 = "21.0,20.17,20.18,19.21,21.25,21.4,21.5,19.27"
DEFAULT_SETS = [f"common10={COMMON10}", f"router8={ROUTER8}"]
GROUPS = ("system", "object", "image_background", "question", "assistant_prefix")
_VOCAB_CHUNK = 16384


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested without a model)
# ---------------------------------------------------------------------------

def item_name(item):
    """Display name of a VADE item, whatever the entity calls it."""
    for key in ("name", "common_name"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"item has no name field: {sorted(item)}")


def label_variants(label):
    """The label as stored, plus title case for entities that store lowercase
    labels ('united states' is written ' United States' by the model)."""
    out = [str(label)]
    if str(label).lower() == str(label) and str(label).title() != str(label):
        out.append(str(label).title())
    return out


def token_groups(ids, image_token_id, im_start_id, footprint):
    """-> list of group names, one per position.

    `footprint` is the set of IMAGE-RELATIVE indices of the object's tokens.
    `question` runs from the first text token after the image to the last
    <|im_start|> (exclusive); `assistant_prefix` is that <|im_start|> onward --
    the assistant header plus the prefill. Anything before the image is `system`."""
    ids = list(ids)
    img = [i for i, t in enumerate(ids) if t == image_token_id]
    assert img, "no image tokens in the prompt"
    starts = [i for i, t in enumerate(ids) if t == im_start_id]
    assert starts and starts[-1] > img[-1], "no <|im_start|> after the image -- not a chat prompt?"
    groups, k = [], 0
    for i, t in enumerate(ids):
        if t == image_token_id:
            groups.append("object" if k in footprint else "image_background")
            k += 1
        elif i < img[0]:
            groups.append("system")
        elif i < starts[-1]:
            groups.append("question")
        else:
            groups.append("assistant_prefix")
    return groups


def candidate_rank(logits, own_ids, cand_ids):
    """Rank of the item's own token among DISTINCT candidate tokens (1 = best),
    plus a z-score. `own_ids` may hold several variants; the best one counts.

    Ties rank optimistically for nobody: rank = 1 + #candidates strictly above."""
    cand = sorted(set(int(c) for c in cand_ids))
    own = sorted(set(int(o) for o in own_ids))
    lc = logits[cand].float()
    lo = logits[own].float().max()
    rank = 1 + int((lc > lo).sum())
    sd = float(lc.std()) if len(cand) > 1 else 0.0
    z = float((lo - lc.mean()) / sd) if sd > 0 else 0.0
    return {"rank": rank, "n": len(cand), "z": z, "chance": (len(cand) + 1) / 2}


def classify_top(tok, queried, own_entity, own_values, all_entity, all_values):
    """Priority classification of a top-1 token id. See module docstring."""
    if tok in own_values.get(queried, ()):
        return "own_value_queried"
    if tok in own_entity:
        return "own_entity"
    if any(tok in v for a, v in own_values.items() if a != queried):
        return "own_value_other_attr"
    if tok in all_entity:
        return "other_entity"
    if any(tok in v for v in all_values.values()):
        return "other_value"
    return "other"


def direction_minus_mean(unembed_dirs, own_ids, cand_ids):
    """Mean direction of the item's own token(s) minus the mean over the OTHER
    distinct candidates. unembed_dirs: fn(ids tensor) -> [n, H] float."""
    import torch
    own = sorted(set(int(o) for o in own_ids))
    others = sorted(set(int(c) for c in cand_ids) - set(own))
    d = unembed_dirs(torch.tensor(own)).mean(0)
    if others:
        d = d - unembed_dirs(torch.tensor(others)).mean(0)
    return d


def split_head_by_groups(weights, values, w_o, group_index, n_groups):
    """One head at one query. weights [K], values [K, D], w_o [H, D].
    -> (z [D], per-group write [G, H]); the group writes sum to z @ w_o^T."""
    import torch
    z = weights @ values
    per = torch.zeros(n_groups, values.shape[1], dtype=values.dtype, device=values.device)
    per.index_add_(0, group_index, weights.unsqueeze(1) * values)
    return z, per @ w_o.T


# ---------------------------------------------------------------------------
# The forward pass
# ---------------------------------------------------------------------------

def capture_forward(adapter, model, ids, extra, blocks):
    """ONE unpadded, eager forward. -> dict with, per block, v_proj output, o_proj
    input and the last query's attention row, plus the pre-norm final residual and
    the model's own logits at the last column."""
    import torch
    from methods.common.hooks import extra_to_device
    layers = adapter.get_decoder_layers(model)
    store = {"v": {}, "z": {}, "a": {}}
    handles = []
    for b in blocks:
        attn = adapter.get_attn_block(model, b)
        handles.append(attn.v_proj.register_forward_hook(
            lambda m, i, o, _b=b: store["v"].__setitem__(_b, o[0].detach().float())))
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(
            lambda m, a, _b=b: store["z"].__setitem__(_b, a[0][0, -1].detach().float())))

        def grab_attn(m, i, o, _b=b):
            assert isinstance(o, tuple) and len(o) > 1 and o[1] is not None, (
                "attention weights unavailable -- load with attn_implementation='eager' and pass "
                "output_attentions=True")
            store["a"][_b] = o[1][0, :, -1, :].detach().float()          # [n_heads, K]
        handles.append(attn.register_forward_hook(grab_attn))
    handles.append(layers[-1].register_forward_hook(
        lambda m, i, o: store.__setitem__("resid", (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float())))
    try:
        with torch.no_grad():
            out = model(input_ids=ids.to(model.device), attention_mask=torch.ones_like(ids).to(model.device),
                        **extra_to_device(extra, model.device, model.dtype),
                        output_attentions=True, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    store["logits"] = out.logits[0, -1].detach().float()
    return store


def vocab_logits(adapter, model, writes):
    """logit_direction(all tokens) @ writes for writes [H, n] -> [V, n].

    Chunked (a float32 [V, H] is ~2GB), and ONE pass over the vocabulary for
    every head at once: streaming the unembedding per head would cost ~18x the
    memory traffic for identical numbers."""
    import torch
    V = model.lm_head.weight.shape[0]
    out = torch.empty(V, writes.shape[1], device=writes.device)
    for lo in range(0, V, _VOCAB_CHUNK):
        ids = torch.arange(lo, min(lo + _VOCAB_CHUNK, V), device=writes.device)
        out[lo:lo + len(ids)] = adapter.logit_direction(model, ids) @ writes
    return out


def analyze_heads(adapter, model, store, heads, head_dim, groups_per_pos, cands, top_k, tokenizer=None,
                  tolerance=0.02):
    """Per-head vocab projection + per-group split, off one captured forward.

    cands = {"entity": (own_ids, all_ids), "values": {attr: (own_ids, all_ids)},
             "queried": attr}. Returns {"b.h": record}."""
    import torch
    with torch.no_grad():
        return _analyze_heads(adapter, model, store, heads, head_dim, groups_per_pos, cands, top_k,
                              tokenizer, tolerance)


def _analyze_heads(adapter, model, store, heads, head_dim, groups_per_pos, cands, top_k, tokenizer, tolerance):
    import torch
    device = store["logits"].device
    assert getattr(adapter.get_attn_head_output_module(model, heads[0][0]), "bias", None) is None, (
        "o_proj has a bias -- per-head writes would not sum to attn_output; add it as its own term")
    scale = adapter.final_norm_scale(model, store["resid"].unsqueeze(0))[0, 0]      # scalar
    # End-to-end check of the frozen-scale algebra on the model's own top token.
    top = int(store["logits"].argmax())
    recon = float(scale * (adapter.logit_direction(model, torch.tensor([top], device=device))[0]
                           @ store["resid"]))
    rel = abs(recon - float(store["logits"][top])) / max(abs(float(store["logits"][top])), 1.0)
    assert rel <= tolerance, (
        f"frozen-scale logit {recon:.3f} vs model logit {float(store['logits'][top]):.3f} ({rel:.2%}) -- "
        f"the final-norm scale or logit_direction is wrong, every projection below would be off")

    gidx = torch.tensor([GROUPS.index(g) for g in groups_per_pos], device=device)
    udirs = lambda ids: adapter.logit_direction(model, ids.to(device))
    ent_own, ent_all = cands["entity"]
    q = cands["queried"]
    d_ent = direction_minus_mean(udirs, ent_own, ent_all)
    d_val = direction_minus_mean(udirs, *cands["values"][q])
    own_values = {a: set(o) for a, (o, _) in cands["values"].items()}
    all_values = {a: set(c) for a, (_, c) in cands["values"].items()}

    n_heads = adapter.n_attention_heads(model)
    parts = []
    for b, h in heads:
        a = store["a"][b][h]                                                   # [K]
        v_raw = store["v"][b]                                                  # [K, kv*D]
        kv = v_raw.shape[-1] // head_dim
        assert v_raw.shape[-1] % head_dim == 0 and n_heads % kv == 0, "unexpected GQA layout"
        vh = v_raw.view(v_raw.shape[0], kv, head_dim)[:, h // (n_heads // kv)]
        w_o = adapter.get_attn_head_output_module(model, b).weight[:, h * head_dim:(h + 1) * head_dim].float()
        z_pred, per_group = split_head_by_groups(a, vh, w_o, gidx, len(GROUPS))
        z_real = store["z"][b][h * head_dim:(h + 1) * head_dim]
        err = float((z_pred - z_real).norm() / z_real.norm().clamp_min(1e-8))
        assert err <= tolerance, (
            f"A@V != o_proj input at head {b}.{h} ({err:.2%}) -- wrong tensor or GQA mapping; "
            f"the per-group split would be meaningless")
        parts.append((b, h, a, err, z_real @ w_o.T, per_group))              # z_real @ w_o.T: the exact write

    all_logits = scale * vocab_logits(adapter, model, torch.stack([p[4] for p in parts], dim=1))
    records = {}
    for n, (b, h, a, err, write, per_group) in enumerate(parts):
        logits = all_logits[:, n]
        centered = logits - logits.mean()
        topv, topi = centered.topk(top_k)
        rec = {
            "reconstruction_rel_err": err,
            "write_norm": float(write.norm()),
            "top": [{"id": int(i), "logit": float(v),
                     **({"token": tokenizer.decode([int(i)])} if tokenizer is not None else {})}
                    for v, i in zip(topv, topi)],
            "top1_class": classify_top(int(topi[0]), q, set(ent_own), own_values, set(ent_all), all_values),
            "entity": candidate_rank(logits, ent_own, ent_all),
            "values": {attr: candidate_rank(logits, o, c) for attr, (o, c) in cands["values"].items()},
            "dla_entity": float(scale * (write @ d_ent)),
            "dla_value_queried": float(scale * (write @ d_val)),
            "groups": {},
        }
        for gi, g in enumerate(GROUPS):
            mask = gidx == gi
            rec["groups"][g] = {
                "attention": float(a[mask].sum()),
                "write_norm": float(per_group[gi].norm()),
                "dla_entity": float(scale * (per_group[gi] @ d_ent)),
                "dla_value_queried": float(scale * (per_group[gi] @ d_val)),
            }
        records[f"{b}.{h}"] = rec
    return records


# ---------------------------------------------------------------------------
# Candidates, rows, summary
# ---------------------------------------------------------------------------

def build_candidates(tokenizer, items, attributes, lookup, template_index):
    """-> (entity_ids {item: [ids]}, value_ids {attr: {item: [ids]}}, collisions).

    Entity: first token of ' <name>' and '<name>'. Values: the first gold token
    under that attribute's own prefill, exactly as targets.py derives golds, for
    every stored-label variant."""
    from methods.common.targets import derive_gold_token_ids
    entity = {}
    for key, it in items.items():
        name = item_name(it)
        ids = {tokenizer(" " + name, add_special_tokens=False)["input_ids"][0],
               tokenizer(name, add_special_tokens=False)["input_ids"][0]}
        entity[key] = sorted(ids)
    values = {}
    for attr in attributes:
        tmpl = lookup[attr][sorted(lookup[attr])[template_index]]
        values[attr] = {}
        for key, it in items.items():
            if it.get(attr) in (None, ""):
                continue
            ids = set()
            for lab in label_variants(it[attr]):
                toks = derive_gold_token_ids(tokenizer, tmpl["prefill"], lab)
                if toks:
                    ids.add(toks[0])
            values[attr][key] = sorted(ids)
    collisions = {"entity": _collisions(entity)}
    collisions.update({a: _collisions(v) for a, v in values.items()})
    return entity, values, collisions


def _collisions(by_item):
    """Items whose first token is shared with a DIFFERENT item's -- where the
    rank measures 'this family of names', not this item."""
    owners = defaultdict(set)
    for k, ids in by_item.items():
        for i in ids:
            owners[i].add(k)
    return sorted({k for s in owners.values() if len(s) > 1 for k in s})


def summarize(records, attributes):
    """records: flat list of {head_set, head, queried, item, rec}. -> nested summary."""
    import statistics as st
    out = {}
    by = defaultdict(list)
    for r in records:
        by[(r["head_set"], r["head"])].append(r)
    for (hs, head), rows in sorted(by.items()):
        ent_rank = [r["rec"]["entity"]["rank"] for r in rows]
        q_rank = [r["rec"]["values"][r["queried"]]["rank"] for r in rows if r["queried"] in r["rec"]["values"]]
        other_rank = [r["rec"]["values"][a]["rank"] for r in rows for a in r["rec"]["values"] if a != r["queried"]]
        cls = Counter(r["rec"]["top1_class"] for r in rows)
        # Question invariance: same top-1 token for an item across every question asked.
        tops = defaultdict(set)
        for r in rows:
            tops[r["item"]].add(r["rec"]["top"][0]["id"])
        grp = {g: {k: st.mean(r["rec"]["groups"][g][k] for r in rows)
                   for k in ("attention", "write_norm", "dla_entity", "dla_value_queried")} for g in GROUPS}
        out.setdefault(hs, {})[head] = {
            "n": len(rows),
            "entity_rank_mean": st.mean(ent_rank), "entity_rank_chance": rows[0]["rec"]["entity"]["chance"],
            "entity_top1_rate": sum(x == 1 for x in ent_rank) / len(ent_rank),
            "entity_z_mean": st.mean(r["rec"]["entity"]["z"] for r in rows),
            "queried_value_rank_mean": st.mean(q_rank) if q_rank else None,
            "queried_value_z_mean": st.mean(r["rec"]["values"][r["queried"]]["z"] for r in rows
                                            if r["queried"] in r["rec"]["values"]),
            "other_value_rank_mean": st.mean(other_rank) if other_rank else None,
            "other_value_z_mean": st.mean(r["rec"]["values"][a]["z"] for r in rows for a in r["rec"]["values"]
                                          if a != r["queried"]) if other_rank else None,
            "dla_entity_mean": st.mean(r["rec"]["dla_entity"] for r in rows),
            "dla_value_queried_mean": st.mean(r["rec"]["dla_value_queried"] for r in rows),
            "top1_class": {k: v / len(rows) for k, v in cls.items()},
            "top1_question_invariant_rate": sum(len(s) == 1 for s in tops.values()) / max(len(tops), 1),
            "groups": grp,
        }
    return out


def label_head(s):
    """A coarse verdict from the summary numbers, printed next to them -- never a
    substitute for reading them."""
    ez, qz, oz = s["entity_z_mean"], s["queried_value_z_mean"], s["other_value_z_mean"] or 0.0
    g = s["groups"]
    img = g["object"]["attention"] + g["image_background"]["attention"]
    txt = g["question"]["attention"]
    if qz > 1.0 and qz > oz + 0.75:
        return "attribute-extractor-like"
    if ez > 1.0 and ez >= qz:
        return "entity-mover-like" + ("" if img >= txt else " (reads text)")
    if txt > img and abs(s["dla_entity_mean"]) < 0.05:
        return "relation/question-reader-like"
    return "unclear"


def print_summary(summary):
    for hs, heads in summary.items():
        print(f"\n=== head set: {hs} ===")
        print(f"{'head':>6} {'ent rank':>9} {'ent z':>6} {'q-val z':>8} {'oth-val z':>9} {'dla ent':>8} "
              f"{'dla qval':>9} {'img attn':>9} {'q attn':>7} {'top1 inv':>8}  top-1 class / verdict")
        for head, s in heads.items():
            g = s["groups"]
            img = g["object"]["attention"] + g["image_background"]["attention"]
            top_cls = max(s["top1_class"].items(), key=lambda kv: kv[1])
            print(f"{head:>6} {s['entity_rank_mean']:6.1f}/{s['entity_rank_chance']:<3.0f}"
                  f"{s['entity_z_mean']:6.2f} {s['queried_value_z_mean']:8.2f} "
                  f"{(s['other_value_z_mean'] or 0):9.2f} {s['dla_entity_mean']:8.3f} "
                  f"{s['dla_value_queried_mean']:9.3f} {img:9.2f} {g['question']['attention']:7.2f} "
                  f"{s['top1_question_invariant_rate']:8.2f}  {top_cls[0]} {top_cls[1]:.0%} | {label_head(s)}")
        print("  dla_entity by source group (mean over rows): ")
        for head, s in heads.items():
            print(f"    {head:>6} " + "  ".join(f"{g}={s['groups'][g]['dla_entity']:+.3f}" for g in GROUPS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--head_sets", nargs="+", default=DEFAULT_SETS, metavar="NAME=SPEC",
                    help="Same syntax as head_swap_vade.py. Default: common10 + R1's router8.")
    ap.add_argument("--attributes", nargs="+", default=None, help="Default: every entity attribute.")
    ap.add_argument("--template_index", type=int, default=0,
                    help="Which template (by sorted id) to ask each attribute with.")
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    head_sets = resolve_head_sets(args.head_sets, args.entity, 8)
    for name, hs in head_sets.items():
        assert not isinstance(hs, dict), f"{name}: per-attribute head sets are not supported here"
    all_heads = sorted({x for hs in head_sets.values() for x in hs})
    blocks = sorted({b for b, _ in all_heads})
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    attributes = args.attributes or [a for a in entity_attributes(gt) if a in lookup]
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    footprint = set(min(loc["object_token_indices"].values(), key=lambda v: len(v["flat"]))["flat"])
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "head_vocab_projection", args.entity)

    print(f"[head_vocab_projection] {args.entity}: {len(keys)} items x {len(attributes)} attributes = "
          f"{len(keys) * len(attributes)} forwards; object footprint {len(footprint)} image tokens")
    for name, hs in head_sets.items():
        print(f"  {name} ({len(hs)}): " + ", ".join(f"{b}.{h}" for b, h in hs))
    print(f"  -> {out_dir}")
    if args.dry_run:
        for k in keys:
            assert os.path.exists(os.path.join(entity_dir, items[k]["image"])), f"missing image for {k}"
            item_name(items[k])
        print("Grid valid.")
        return

    import torch
    from PIL import Image
    from methods.adapters.registry import get_adapter
    from methods.head_swap_vade import build_prompt

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(device=args.device, attn_implementation="eager")
    tok = processor.tokenizer
    head_dim = adapter.hidden_size(model) // adapter.n_attention_heads(model)
    image_token_id = adapter.image_token_id(model, processor)
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    ent_ids, val_ids, collisions = build_candidates(tok, items, attributes, lookup, args.template_index)
    all_ent = sorted({i for v in ent_ids.values() for i in v})
    print("  first-token collisions (rank measures a name FAMILY for these): "
          + "; ".join(f"{k}: {len(v)}" for k, v in collisions.items()))

    os.makedirs(out_dir, exist_ok=True)
    records = []
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
        for n, key in enumerate(keys):
            with Image.open(os.path.join(entity_dir, items[key]["image"])) as im:
                image = im.convert("RGB")
            for attr in attributes:
                tmpl = lookup[attr][sorted(lookup[attr])[args.template_index]]
                p = build_prompt(processor, image, tmpl["question"], tmpl["prefill"])
                ids = p["input_ids"]
                groups = token_groups(ids[0].tolist(), image_token_id, im_start_id, footprint)
                store = capture_forward(adapter, model, ids, {"pixel_values": p["pixel_values"],
                                                              "image_grid_thw": p["image_grid_thw"]}, blocks)
                cands = {"entity": (ent_ids[key], all_ent), "queried": attr,
                         "values": {a: (val_ids[a][key], sorted({i for v in val_ids[a].values() for i in v}))
                                    for a in attributes if key in val_ids[a]}}
                recs = analyze_heads(adapter, model, store, all_heads, head_dim, groups, cands, args.top_k, tok)
                pred = int(store["logits"].argmax())
                for hs_name, hs in head_sets.items():
                    for b, h in hs:
                        r = {"head_set": hs_name, "head": f"{b}.{h}", "item": key, "queried": attr,
                             "template_id": tmpl["template_id"], "model_top1": tok.decode([pred]),
                             "rec": recs[f"{b}.{h}"]}
                        records.append(r)
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {n + 1}/{len(keys)} items", flush=True)

    summary = summarize(records, attributes)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"entity": args.entity, "attributes": attributes, "template_index": args.template_index,
                   "n_items": len(keys), "collisions": collisions, "by_head": summary}, f, indent=2)
    print_summary(summary)
    print(f"\nwrote {out_dir}/rows.jsonl and summary.json")


if __name__ == "__main__":
    main()
