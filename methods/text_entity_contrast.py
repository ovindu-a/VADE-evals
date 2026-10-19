"""E6: the same circuit probes with the entity NAMED in text instead of SHOWN.

The paper's contrast experiment. Cohen et al. (ACL 2025) and Venhoff et al. (NeurIPS
2025) show VLMs recall facts worse about an entity shown as an image than named in
text, and attribute it to the image entity being resolved too late. This script asks
the component-level version of that question on one model, one prompt shape, and
the same (base, source, attribute) pairs, changing only how the entity arrives:

    image   "<image> For the country shown ... Report the capital of this country."
    text    "For the country named ... The country is France. Report the capital ..."

Both conditions put the entity BEFORE the question, so "entity position" means the
same thing causally: the image tokens, or the name's tokens.

PHASES (--phases, any subset; each is cheap, forward-only except `accuracy`)

  accuracy  greedy answers for every (country, attribute) in both conditions, VADE's
            matcher. The recall gap on this model and these items.
  curves    residual swap from the source at layer L, two sites per condition:
              entity   image: every image column | text: the name's LAST token
              readout  the last prompt column + the appended divergence prefix
            -> the handoff curve per condition. Where entity-site transfer dies and
            readout-site transfer starts IS the handoff; the claim under test is that
            it sits later for images than for text.
  heads     patch the entity site at L_e (top of its plateau), rank every head of
            blocks >= L_e by how much its write at the readout column changes
            (delta_resid, head_trace's measure), then install the top-k heads alone
            into a clean run (sufficiency). Also installs the IMAGE movers (R8) into
            the text condition and each condition's top-10 into the other. Same heads
            -> one shared entity-moving mechanism; different heads -> different routes.
            --mover_subsets adds every single mover, every unordered pair and every
            leave-one-out set of the image movers (A1: a true conjunction vs an
            additive effect below a generation threshold, read off `ld`), and writes
            the per-row records to heads_rows.jsonl.
  freeze    entity patch at L_e, plus MLP outputs frozen at the base's own values
            (`base`) or the leave-one-out batch mean over other countries asked the
            same attribute (`mean`, which writes no single country's answer):
              readout:A-B  the last-token MLPs, blocks A..B (R16.2's lookup site)
              entity:+a-+b the entity position's MLPs, blocks L_e+a..L_e+b
            The R16.2 prediction for images: readout:23-26 kills transfer. If the text
            condition is untouched by it but hurt by entity:+..., text facts are looked
            up at the subject position (Geva/Meng) and images use a different, later
            site -- the mechanistic content of "too late to recall".
  crossmodal  (opt-in; not in the default --phases) do the mover heads carry the SAME
            thing in both conditions? `heads` shows the same heads are USED; every one of
            its installs is within-condition, so it cannot say whether their content is
            shared. Two parts, no --entity_layer needed:
              representation  clean runs of every (country, attribute) in both conditions;
                at the last prompt column, each mover's o_proj input, the movers' joint
                residual write, and the residual at every layer (plus the entity site:
                image tokens mean-pooled vs the name). Per layer / per head: cross-modal
                retrieval -- is image-run France nearest to text-run France among all
                countries asked the same question? -- after centering each condition on
                its own mean (a modality offset otherwise dominates every cosine), and
                again through a ridge map fitted on half the COUNTRIES and tested on the
                other half. The layer curve answers "when does the image run become the
                text run"; chance top-1 is 1/n_countries.
              install  the causal version: the movers' values from a CLEAN source run in
                the OTHER condition, installed at the readout columns of a clean base run
                (text-named B's mover outputs into image-of-A, and the reverse), against
                the same-condition donor, a per-head norm-matched cross donor, a random-10
                head set from the mover blocks, and a `self` donor (the BASE country in
                the other condition), which must keep the base answer: it separates
                "carries the entity" from "any foreign-modality vector breaks the run".
            cross ~= same -> the moved payload is modality-invariant; cross ~= 0 with
            normmatch recovering it -> same direction, different scale; cross ~= 0 and
            self also breaks -> the two conditions' mover spaces are not compatible.
            Arrays are saved to crossmodal_reps.pt for CPU re-analysis.

READOUT. Divergence-point transfer / base_kept / normalized logit difference, as in
methods/circuit_common.py. Text prompts differ in length between countries, so text
batches are left-padded; readout columns are aligned from the end.

Usage
-----
    python methods/text_entity_contrast.py --dry_run
    python methods/text_entity_contrast.py --n_pairs 8 --out_dir results/text_entity_contrast/smoke
    python methods/text_entity_contrast.py                     # 64 pairs x 4 attributes, all phases
    python methods/text_entity_contrast.py --phases crossmodal \
        --out_dir results/text_entity_contrast/flags_crossmodal  # mover content, image vs text
"""
import argparse
import itertools
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import (ATTRIBUTES, DEFAULT_VADE_ROOT, MODEL_ID, MOVERS, clean_eligible,  # noqa: E402
                                    load_flags, normalized_ld, parse_heads, print_table, sample_pairs,
                                    summarize_arms, text_question, write_json)

CONDITIONS = ("image", "text")
PHASES = ("accuracy", "curves", "heads", "freeze")
OPT_IN_PHASES = ("crossmodal",)                     # never in the default --phases
DEFAULT_FREEZE = ["readout:16-22", "readout:23-26", "readout:22-27", "entity:+0-+5"]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def chunk_rows(rows, batch_size):
    """Group by (prefix length, attribute): equal appended length, and a batch's
    leave-one-out mean then averages over countries asked the SAME question."""
    groups = {}
    for r in rows:
        groups.setdefault((len(r["prefix"]), r["attribute"]), []).append(r)
    for key in sorted(groups):
        g = groups[key]
        for i in range(0, len(g), batch_size):
            yield g[i:i + batch_size]


def build_side(m, cond, entity_dir, items, rows, who, text_site="last"):
    """-> dict(ids, mask, extra, entity [B, n_e], readout [B, k+1]) for the base or
    source side of a batch in one condition.

    Text entity columns are stored as offsets from the prompt's END, so left padding
    cannot shift them. With text_site='all' every name token is a column; rows with
    shorter names repeat their last column (a duplicate write of the same value is a
    no-op), which keeps the position tensor rectangular. That requires base and source
    names of EQUAL token length per row -- main() filters the pairs to guarantee it."""
    import torch
    from methods.circuit_common import cat_extra, left_pack
    k = len(rows[0]["prefix"])
    seqs, extras, ent_offs = [], [], []
    for r in rows:
        item = items[r[who]]
        tail = torch.tensor(r["prefix"], dtype=torch.long)
        if cond == "image":
            ids, extra = m.image_prompt(entity_dir, item, r["attribute"])
            extras.append(extra)
        else:
            ids, cols = m.text_prompt(item["name"], r["attribute"])
            use = cols if text_site == "all" else cols[-1:]
            ent_offs.append([c - len(ids) for c in use])  # offsets from the prompt's end, padding-proof
        seqs.append(torch.cat([ids, tail]))
    ids, mask = left_pack(seqs, m.pad_id)
    T = ids.shape[1]
    readout = torch.tensor([list(range(T - 1 - k, T))] * len(rows))
    if cond == "image":
        img = ids == m.image_id
        assert torch.equal(img, img[:1].expand_as(img)), "image rows must share one layout"
        entity = torch.tensor([img[0].nonzero().flatten().tolist()] * len(rows))
    else:
        width = max(len(o) for o in ent_offs)
        entity = torch.tensor([[T - k + off for off in o + o[-1:] * (width - len(o))] for o in ent_offs])
    return {"ids": ids, "mask": mask, "extra": cat_extra(extras) if extras else {},
            "entity": entity, "readout": readout}


def fmt_head(h):
    return f"{h[0]}.{h[1]}"


def mover_subsets(heads):
    """-> {name: heads} for every single head, every unordered pair and every
    leave-one-out set of `heads` (10 movers -> 10 + 45 + 10 sets). Names are
    `single:B.H`, `pair:B.H+B.H` (in `heads` order) and `loo:-B.H`."""
    heads = list(heads)
    out = {f"single:{fmt_head(h)}": [h] for h in heads}
    for a, b in itertools.combinations(heads, 2):
        out[f"pair:{fmt_head(a)}+{fmt_head(b)}"] = [a, b]
    for h in heads:
        out[f"loo:-{fmt_head(h)}"] = [x for x in heads if x != h]
    return out


# ---------------------------------------------------------------------------
# Cross-modal comparison (pure tensor functions; tested offline)
# ---------------------------------------------------------------------------

def retrieval(A, B, groups):
    """A, B [N, d]: row i of A and row i of B are the same entity in two conditions.
    Within each group (attribute asked), each condition is centered on its own group
    mean and unit-normalized; row i of A is then ranked against every row of B in its
    group. -> {n, top1, mean_rank, same_cos, diff_cos, chance_top1}."""
    import torch.nn.functional as F
    acc = {"n": 0, "top1": 0.0, "mean_rank": 0.0, "same_cos": 0.0, "diff_cos": 0.0, "chance_top1": 0.0}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        n = len(idx)
        if n < 2:
            continue
        a, b = A[idx].float(), B[idx].float()
        a = F.normalize(a - a.mean(0), dim=-1)
        b = F.normalize(b - b.mean(0), dim=-1)
        S = a @ b.T
        d = S.diagonal()
        # Ties count against a hit and take the average rank: identical inputs (e.g. the
        # same last token at layer 0, centered to zero) must read chance, not top-1 = 100%.
        above = (S > d[:, None]).sum(1)
        ties = (S == d[:, None]).sum(1) - 1
        rank = above + 1 + ties / 2
        acc["n"] += n
        acc["top1"] += float(((above == 0) & (ties == 0)).sum())
        acc["mean_rank"] += float(rank.sum())
        acc["same_cos"] += float(d.sum())
        acc["diff_cos"] += float(((S.sum(1) - d) / (n - 1)).sum())
        acc["chance_top1"] += 1.0                                  # n rows x 1/n
    if acc["n"]:
        for k in ("top1", "mean_rank", "same_cos", "diff_cos", "chance_top1"):
            acc[k] /= acc["n"]
    return acc


def ridge_retrieval(A, B, entities, groups, alpha=0.1, folds=2, seed=0):
    """Held-out linear map A -> B: ridge fitted on the rows of all but one fold of
    ENTITIES (so no country is on both sides), applied to the held-out fold, then
    `retrieval` there. Dual-form solve, so d >> n is fine; lambda = alpha x mean
    squared row norm. -> retrieval's dict, pooled over folds."""
    import torch
    uniq = sorted(set(entities))
    random.Random(seed).shuffle(uniq)
    fold_of = {e: i % folds for i, e in enumerate(uniq)}
    parts = []
    for f in range(folds):
        tr = [i for i, e in enumerate(entities) if fold_of[e] != f]
        te = [i for i, e in enumerate(entities) if fold_of[e] == f]
        At, Bt = A[tr].float(), B[tr].float()
        mu_a, mu_b = At.mean(0), Bt.mean(0)
        Ac = At - mu_a
        K = Ac @ Ac.T
        lam = alpha * K.diagonal().mean()
        W = Ac.T @ torch.linalg.solve(K + lam * torch.eye(len(tr), dtype=K.dtype), Bt - mu_b)
        parts.append(retrieval((A[te].float() - mu_a) @ W + mu_b, B[te], [groups[i] for i in te]))
    n = sum(p["n"] for p in parts)
    return {k: (n if k == "n" else sum(p[k] * p["n"] for p in parts) / max(n, 1)) for k in parts[0]}


def norm_match(z_donor, z_ref, heads, head_dim):
    """-> copy of z_donor ({b: [B, n, hidden]}) with each listed head's slice rescaled,
    per row and column, to the norm of the same slice in z_ref."""
    out = {b: v.clone() for b, v in z_donor.items()}
    for b, h in heads:
        sl = slice(h * head_dim, (h + 1) * head_dim)
        d, r = out[b][..., sl], z_ref[b][..., sl]
        out[b][..., sl] = d * (r.float().norm(dim=-1, keepdim=True)
                               / d.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(d.dtype)
    return out


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_accuracy(m, args, entity_dir, items):
    """Clean greedy answers for every (country, attribute), both conditions."""
    from methods.circuit_common import cat_extra, left_pack
    from methods.common.text_match import contains_label
    jobs = [(c, a) for c in sorted(items) for a in args.attributes]
    acc = {cond: {a: [0, 0] for a in args.attributes} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for i in range(0, len(jobs), args.batch_size):
            chunk = jobs[i:i + args.batch_size]
            seqs, extras = [], []
            for c, a in chunk:
                if cond == "image":
                    ids, x = m.image_prompt(entity_dir, items[c], a)
                    extras.append(x)
                else:
                    ids, _ = m.text_prompt(items[c]["name"], a)
                seqs.append(ids)
            ids, mask = left_pack(seqs, m.pad_id)
            texts = m.decode(m.generate(ids, mask, cat_extra(extras) if extras else {}, args.max_new_tokens))
            for (c, a), t in zip(chunk, texts):
                acc[cond][a][0] += contains_label(t, str(items[c][a]))
                acc[cond][a][1] += 1
        print(f"  accuracy[{cond}] done", flush=True)
    out = {cond: {a: v[0] / v[1] for a, v in d.items()} for cond, d in acc.items()}
    print("\nclean accuracy (VADE matcher)")
    print(f"  {'attribute':<14}{'image':>8}{'text':>8}{'gap':>8}")
    for a in args.attributes:
        print(f"  {a:<14}{out['image'][a]:>7.1%}{out['text'][a]:>8.1%}{out['text'][a] - out['image'][a]:>+8.1%}")
    return out


def run_curves(m, args, batch_sides, rows):
    """-> records for entity@L / readout@L, plus clean / source_clean, for one batch."""
    from methods.circuit_common import gather_cols, readout_metrics, residual_patch
    recs = []

    def emit(cond, arm, logits):
        for r, met in zip(rows, readout_metrics(logits, rows)):
            recs.append({"row_index": r["row_index"], "attribute": r["attribute"], "cond": cond, "arm": arm, **met})

    for cond in CONDITIONS:
        base, src = batch_sides[cond]
        out_s = m.forward(src["ids"], src["mask"], src["extra"], output_hidden_states=True)
        hs = out_s.hidden_states
        emit(cond, "source_clean", out_s.logits[:, -1])
        emit(cond, "clean", m.forward(base["ids"], base["mask"], base["extra"]).logits[:, -1])
        for L in args.layers:
            for site in ("entity", "readout"):
                vals = gather_cols(hs[L], src[site])
                hook = residual_patch(L, base[site], vals)
                emit(cond, f"{site}@{L}", m.forward(base["ids"], base["mask"], base["extra"], hooks=[hook]).logits[:, -1])
        del out_s, hs
    return recs


def pick_entity_layer(curve, layers, frac=0.9, onset=0.1):
    """L_e = the deepest layer that is BOTH on the entity-site plateau (>= frac of its
    max) AND before the readout-site curve starts (< onset of its max).

    The second condition matters for `heads`: the patch must precede the whole handoff
    window, or the window's first blocks run before the edit and cannot show up in the
    ranking -- the deepest plateau layer alone can land INSIDE the window (on flags the
    image plateau runs to ~L22 while the movers start at block 21)."""
    ent = {L: curve.get(f"entity@{L}", {}).get("transfer", 0.0) for L in layers}
    rd = {L: curve.get(f"readout@{L}", {}).get("transfer", 0.0) for L in layers}
    top, rtop = max(ent.values(), default=0.0), max(rd.values(), default=0.0)
    if top <= 0:
        return None
    plateau = [L for L in layers if ent[L] >= frac * top]
    before = [L for L in layers if rtop <= 0 or all(rd[x] < onset * rtop for x in layers if x <= L)]
    both = sorted(set(plateau) & set(before))
    return max(both) if both else max(plateau)


def head_capture(m, side, blocks, hooks=()):
    """o_proj inputs at the readout columns for `blocks`, one forward -> {b: [B, k+1, hidden]}."""
    from methods.circuit_common import gather_cols
    sinks, handles = {}, []
    for b in blocks:
        mod = m.adapter.get_attn_head_output_module(m.model, b)
        handles.append(mod.register_forward_pre_hook(
            lambda _m, inp, _b=b: sinks.__setitem__(_b, gather_cols(inp[0], side["readout"].to(inp[0].device)).detach())))
    try:
        logits = m.forward(side["ids"], side["mask"], side["extra"], hooks=hooks).logits[:, -1]
    finally:
        for h in handles:
            h.remove()
    return sinks, logits


def head_install(m, side, heads, z):
    """Clean forward with `heads` replaced at the readout columns by z[b] -> logits."""
    from methods.head_followup_common import grouped_heads
    handles = []
    for b, hs in grouped_heads(heads).items():
        cols = side["readout"]

        def pre(_m, inp, _hs=hs, _z=z[b]):
            t = inp[0].clone()
            for i in range(t.shape[0]):
                for h in _hs:
                    sl = slice(h * m.head_dim, (h + 1) * m.head_dim)
                    t[i, cols[i], sl] = _z[i, :, sl].to(t)
            return (t,) + tuple(inp[1:])
        handles.append(m.adapter.get_attn_head_output_module(m.model, b).register_forward_pre_hook(pre))
    try:
        return m.forward(side["ids"], side["mask"], side["extra"]).logits[:, -1]
    finally:
        for h in handles:
            h.remove()


def delta_resid(m, zc, zp, blocks):
    """-> {(b, h): ||(zp - zc)_h W_O_h^T|| at the LAST readout column, summed over rows}."""
    out = {}
    for b in blocks:
        W = m.adapter.get_attn_head_output_module(m.model, b).weight.float()     # [hidden, H*d]
        dz = (zp[b][:, -1] - zc[b][:, -1]).float()                               # [B, H*d]
        for h in range(m.n_heads):
            sl = slice(h * m.head_dim, (h + 1) * m.head_dim)
            out[(b, h)] = float((dz[:, sl] @ W[:, sl].T).norm(dim=-1).sum())
    return out


def parse_freeze(spec, L_e, n_layers):
    where, span = spec.split(":", 1)
    assert where in ("readout", "entity"), f"freeze spec {spec!r}: readout:A-B or entity:+a-+b"
    a, b = span.split("-")
    rel = lambda x: L_e + int(x[1:]) if x.startswith("+") else int(x)  # noqa: E731
    # Only blocks at/after the patch can react to it. Freezing an earlier one at `base` is a
    # no-op and at `mean` is pure ablation damage, which would contaminate the comparison.
    blocks = [x for x in range(rel(a), rel(b) + 1) if L_e <= x < n_layers]
    return where, blocks


def mlp_freeze_hooks(m, side, where, blocks, values):
    """values: {b: [B, n_pos, hidden]} written over the mlp_output at side[where]."""
    from methods.common.hooks import make_cache_aware_patch_hook
    from methods.common.sites import InterventionSite
    site = InterventionSite("mlp_output")
    return [(site, b + 1, make_cache_aware_patch_hook(side[where], lambda bv, _v=values[b]: _v.to(bv.dtype)))
            for b in blocks]


def mlp_capture(m, side, where, blocks, hooks=()):
    from methods.circuit_common import gather_cols
    sinks, handles = {}, []
    for b in blocks:
        mod = m.adapter.get_mlp_block(m.model, b)
        handles.append(mod.register_forward_hook(
            lambda _m, i, o, _b=b: sinks.__setitem__(_b, gather_cols(o, side[where].to(o.device)).detach())))
    try:
        m.forward(side["ids"], side["mask"], side["extra"], hooks=hooks)
    finally:
        for h in handles:
            h.remove()
    return sinks


def loo_mean(v):
    """Leave-one-out mean over the batch (rows = other countries, same attribute)."""
    B = v.shape[0]
    assert B >= 2, "a leave-one-out mean needs >= 2 rows per batch"
    return (v.sum(0, keepdim=True) - v) / (B - 1)


# ---------------------------------------------------------------------------

def crossmodal_capture(m, side, movers):
    """One clean forward -> (z [B, n_movers, head_dim], write [B, hidden], resid
    [B, n_layers, hidden], entity [B, n_layers, hidden]) at the LAST readout column.
    `entity` mean-pools the side's entity columns (every image column, or the name)."""
    import torch
    from methods.circuit_common import gather_cols
    blocks = sorted({b for b, _ in movers})
    z, _ = head_capture(m, side, blocks)
    out = m.forward(side["ids"], side["mask"], side["extra"], output_hidden_states=True)
    zs = torch.stack([z[b][:, -1, h * m.head_dim:(h + 1) * m.head_dim] for b, h in movers], 1)
    write = torch.stack([
        z[b][:, -1, h * m.head_dim:(h + 1) * m.head_dim].float()
        @ m.adapter.get_attn_head_output_module(m.model, b).weight[:, h * m.head_dim:(h + 1) * m.head_dim].float().T
        for b, h in movers]).sum(0)
    last = side["readout"][:, -1:]
    hs = out.hidden_states[:m.n_layers]               # the last entry is post-final-norm (gotcha 2b)
    resid = torch.stack([gather_cols(h, last.to(h.device))[:, 0] for h in hs], 1)
    ent = torch.stack([gather_cols(h, side["entity"].to(h.device)).float().mean(1) for h in hs], 1)
    return zs.half().cpu(), write.half().cpu(), resid.half().cpu(), ent.half().cpu()


def phase_crossmodal(m, args, entity_dir, items, chunks, sides, readout_metrics):
    import torch
    movers = parse_heads(args.movers)
    report: dict = {"movers": [fmt_head(h) for h in movers]}

    # -- representation: every (country, attribute), both conditions -------------------
    rep_rows = [{"row_index": j, "base": c, "source": c, "attribute": a, "prefix": []}
                for j, (a, c) in enumerate((a, c) for a in args.attributes for c in sorted(items))]
    reps = {c: {"z": [], "write": [], "resid": [], "entity": []} for c in CONDITIONS}
    order = []
    for i, chunk in enumerate(list(chunk_rows(rep_rows, args.batch_size))):
        order += chunk
        for cond in CONDITIONS:
            side = build_side(m, cond, entity_dir, items, chunk, "base", args.text_entity_site)
            for key, v in zip(("z", "write", "resid", "entity"), crossmodal_capture(m, side, movers)):
                reps[cond][key].append(v)
        print(f"  crossmodal/capture {i + 1}", flush=True)
    reps = {c: {k: torch.cat(v) for k, v in d.items()} for c, d in reps.items()}
    ents = [r["base"] for r in order]
    groups = [r["attribute"] for r in order]
    torch.save({"rows": [(r["base"], r["attribute"]) for r in order], "movers": movers, **{
        f"{c}_{k}": v for c, d in reps.items() for k, v in d.items()}},
        os.path.join(args.out_dir, "crossmodal_reps.pt"))

    def both(a, b):
        return {"centered": retrieval(a, b, groups),
                "ridge_img2txt": ridge_retrieval(a, b, ents, groups, seed=args.seed),
                "ridge_txt2img": ridge_retrieval(b, a, ents, groups, seed=args.seed)}

    img, txt = reps["image"], reps["text"]
    rep = {"n_rows": len(order), "n_countries": len(set(ents)),
           "resid_by_layer": {L: both(img["resid"][:, L], txt["resid"][:, L]) for L in range(m.n_layers)},
           "entity_by_layer": {L: both(img["entity"][:, L], txt["entity"][:, L]) for L in range(m.n_layers)},
           "movers_concat": both(img["z"].flatten(1), txt["z"].flatten(1)),
           "movers_write": both(img["write"], txt["write"]),
           "per_head": {fmt_head(h): {**both(img["z"][:, j], txt["z"][:, j]),
                                      "norm_ratio_img_over_txt": float(img["z"][:, j].float().norm(dim=-1).mean()
                                                                       / txt["z"][:, j].float().norm(dim=-1).mean())}
                        for j, h in enumerate(movers)}}
    report["representation"] = rep
    print(f"\nCROSSMODAL representation: n={rep['n_rows']} rows, chance top-1 "
          f"{rep['movers_concat']['centered']['chance_top1']:.1%}")
    print(f"  {'site':<22} {'top1 ctr':>9} {'top1 ridge i>t':>15} {'same-diff cos':>14}")

    def line(name, d):
        c = d["centered"]
        print(f"  {name:<22} {c['top1']:>8.1%} {d['ridge_img2txt']['top1']:>15.1%} "
              f"{c['same_cos'] - c['diff_cos']:>14.3f}")
    for L in range(m.n_layers):
        line(f"resid@{L}", rep["resid_by_layer"][L])
    for L in range(m.n_layers):
        line(f"entity@{L}", rep["entity_by_layer"][L])
    line("movers (concat z)", rep["movers_concat"])
    line("movers (joint write)", rep["movers_write"])
    for name, d in rep["per_head"].items():
        line(f"head {name}", d)

    # -- install: cross-condition donors into clean base runs ---------------------------
    blocks = sorted({b for b, _ in movers})
    rng = random.Random(args.seed + 11)
    random10 = sorted(rng.sample([(b, h) for b in blocks for h in range(m.n_heads) if (b, h) not in set(movers)], 10))
    recs = []
    for i, chunk in enumerate(chunks):
        sd = sides(chunk)
        cap = {}
        for cond in CONDITIONS:
            base, src = sd[cond]
            cap[cond] = {"base": head_capture(m, base, blocks), "src": head_capture(m, src, blocks)}
        for r_cond in CONDITIONS:
            base, _ = sd[r_cond]
            dev = lambda z: {b: v.to(m.model.device) for b, v in z.items()}  # noqa: E731
            zc, lc = cap[r_cond]["base"]
            arms = [("clean", lc), ("source_clean", cap[r_cond]["src"][1])]
            for d_cond in CONDITIONS:
                tag = "same" if d_cond == r_cond else "cross"
                z_src = cap[d_cond]["src"][0]
                arms.append((f"install:movers<-{tag}", head_install(m, base, movers, dev(z_src))))
                arms.append((f"install:random10<-{tag}", head_install(m, base, random10, dev(z_src))))
                if tag == "cross":
                    arms.append(("install:movers<-cross:normmatch",
                                 head_install(m, base, movers, dev(norm_match(z_src, zc, movers, m.head_dim)))))
                    arms.append(("install:movers<-cross:self",
                                 head_install(m, base, movers, dev(cap[d_cond]["base"][0]))))
            for arm, logits in arms:
                for r, met in zip(chunk, readout_metrics(logits.cpu(), chunk)):
                    recs.append({"row_index": r["row_index"], "attribute": r["attribute"], "cond": r_cond,
                                 "arm": arm, **met})
        print(f"  crossmodal/install {i + 1}/{len(chunks)}", flush=True)
    inst: dict = {"random10": [fmt_head(h) for h in random10]}
    report["install"] = inst
    for cond in CONDITIONS:
        rr = [r for r in recs if r["cond"] == cond]
        elig = clean_eligible(rr)
        summ = summarize_arms(rr, elig)
        inst[cond] = {"n_eligible": len(elig), "arms": summ, "arms_all_rows": summarize_arms(rr),
                                   "norm_ld": {a: normalized_ld(summ, a) for a in summ}}
        print_table(summ, f"CROSSMODAL install into the {cond} condition (donor = cross/same condition), "
                          f"n={len(elig)} eligible")
    return report, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--attributes", nargs="+", default=list(ATTRIBUTES), choices=list(ATTRIBUTES))
    ap.add_argument("--phases", nargs="+", default=list(PHASES), choices=list(PHASES) + list(OPT_IN_PHASES))
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(0, 28)),
                    help="Residual layers for `curves` (0 = embeddings). Stops at n_layers-1: the last "
                         "hidden_states entry is post-final-norm (CLAUDE.md gotcha 2b).")
    ap.add_argument("--entity_layer", type=int, nargs=2, default=None, metavar=("IMAGE", "TEXT"),
                    help="L_e per condition for `heads`/`freeze`. Default: picked from `curves` (deepest "
                         "entity-plateau layer that precedes the readout curve's onset); required if "
                         "`curves` is not run.")
    ap.add_argument("--text_entity_site", choices=["last", "all"], default="last",
                    help="Text entity columns: the name's LAST token (Geva/Meng's subject position, "
                         "default) or EVERY name token -- the closer analogue of patching all image "
                         "tokens, at the cost of keeping only pairs whose names tokenize to equal length.")
    ap.add_argument("--head_ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--movers", default=MOVERS)
    ap.add_argument("--mover_subsets", action="store_true",
                    help="`heads`: also install every single image mover, every pair and every leave-one-out "
                         "set (single:B.H, pair:B.H+B.H, loo:-B.H), and write per-row records to "
                         "heads_rows.jsonl.")
    ap.add_argument("--freeze", nargs="+", default=DEFAULT_FREEZE)
    ap.add_argument("--freeze_values", nargs="+", default=["base", "mean"], choices=["base", "mean"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "text_entity_contrast", "flags"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    entity_dir, items, gt_sha = load_flags(args.vade_root)
    pairs = sample_pairs(items, args.n_pairs, args.seed)
    if ("heads" in args.phases or "freeze" in args.phases) and "curves" not in args.phases:
        assert args.entity_layer, "--entity_layer IMAGE TEXT is required when `curves` is not run"
    print(f"[text_entity_contrast] phases={args.phases}; {len(pairs)} pairs x {len(args.attributes)} attributes")
    print(f"  example text prompt: {text_question('capital', 'France')!r}")
    if "crossmodal" in args.phases:
        print(f"  crossmodal: {len(items)} countries x {len(args.attributes)} attributes per condition for the "
              f"representation part; movers {args.movers}")
    if args.dry_run:
        print("Metadata valid.")
        return

    from methods.circuit_common import Model, pair_rows, readout_metrics
    m = Model(args.model_id, args.device, attn_impl="sdpa", vade_root=args.vade_root)
    assert all(0 <= L < m.n_layers for L in args.layers)
    os.makedirs(args.out_dir, exist_ok=True)
    # mover_subsets is left out of the config when off, so default runs write the same summary.json as before.
    report = {"config": {k: v for k, v in vars(args).items()
                         if k != "dry_run" and not (k == "mover_subsets" and not v)}, "ground_truth_sha256": gt_sha}

    if "accuracy" in args.phases:
        report["accuracy"] = phase_accuracy(m, args, entity_dir, items)
        write_json(os.path.join(args.out_dir, "summary.json"), report)

    rows, dropped = pair_rows(items, m.tokenizer, pairs, args.attributes)
    print(f"  {len(rows)} pair rows ({dropped} dropped: golds never diverge)")
    if args.text_entity_site == "all":
        n_tok = {c: len(m.text_prompt(items[c]["name"], "capital")[1]) for c in items}
        before = len(rows)
        rows = [r for r in rows if n_tok[r["base"]] == n_tok[r["source"]]]
        print(f"  text_entity_site=all: kept {len(rows)}/{before} rows whose names tokenize to equal length")
    chunks = list(chunk_rows(rows, args.batch_size))

    def sides(chunk):
        return {c: (build_side(m, c, entity_dir, items, chunk, "base", args.text_entity_site),
                    build_side(m, c, entity_dir, items, chunk, "source", args.text_entity_site))
                for c in CONDITIONS}

    records = []
    if "curves" in args.phases:
        for i, chunk in enumerate(chunks):
            records += run_curves(m, args, sides(chunk), chunk)
            print(f"  curves {i + 1}/{len(chunks)}", flush=True)
        report["curves"] = {}
        for cond in CONDITIONS:
            rr = [r for r in records if r["cond"] == cond]
            elig = clean_eligible(rr)
            summ = summarize_arms(rr, elig)
            report["curves"][cond] = {"n_eligible": len(elig), "arms": summ, "arms_all_rows": summarize_arms(rr),
                                      "norm_ld": {a: normalized_ld(summ, a) for a in summ}}
        print(f"\n{'L':>4} " + "".join(f"{c + ':' + s:>16}" for c in CONDITIONS for s in ("entity", "readout")))
        for L in args.layers:
            cells = []
            for c in CONDITIONS:
                for s in ("entity", "readout"):
                    v = report["curves"][c]["arms"].get(f"{s}@{L}", {}).get("transfer")
                    cells.append(f"{v:>15.1%} " if v is not None else f"{'-':>16}")
            print(f"{L:>4} " + "".join(cells))
        write_json(os.path.join(args.out_dir, "summary.json"), report)

    L_e = {}
    for j, cond in enumerate(CONDITIONS if ("heads" in args.phases or "freeze" in args.phases) else ()):
        if args.entity_layer:
            L_e[cond] = args.entity_layer[j]
        else:
            L_e[cond] = pick_entity_layer(report["curves"][cond]["arms"], args.layers)
            assert L_e[cond] is not None, f"{cond}: the entity-site curve never transfers; nothing to trace"
    if "heads" in args.phases or "freeze" in args.phases:
        print(f"\n  entity patch layer L_e: {L_e}")
        report["entity_layer"] = L_e
        late = sorted({b for b, _ in parse_heads(args.movers) if b < L_e["image"]})
        if late:
            print(f"  WARNING: image L_e={L_e['image']} is after mover block(s) {late}; those heads run before "
                  f"the patch and cannot appear in the image ranking. Pass --entity_layer to move it earlier.")

    def entity_patch(side_b, side_s, L):
        from methods.circuit_common import gather_cols, residual_patch
        hs = m.forward(side_s["ids"], side_s["mask"], side_s["extra"], output_hidden_states=True).hidden_states
        return residual_patch(L, side_b["entity"], gather_cols(hs[L], side_s["entity"]))

    if "heads" in args.phases:
        movers = parse_heads(args.movers)
        rank_recs, cache = {}, []
        for i, chunk in enumerate(chunks):
            sd = sides(chunk)
            entry = {}
            for cond in CONDITIONS:
                base, src = sd[cond]
                blocks = list(range(L_e[cond], m.n_layers))
                patch = entity_patch(base, src, L_e[cond])
                zc, lc = head_capture(m, base, blocks)
                zp, lp = head_capture(m, base, blocks, hooks=[patch])
                for key, v in delta_resid(m, zc, zp, blocks).items():
                    rank_recs.setdefault(cond, {}).setdefault(key, 0.0)
                    rank_recs[cond][key] += v
                entry[cond] = ({b: v.cpu() for b, v in zp.items()}, lc.cpu(), lp.cpu(),
                               m.forward(src["ids"], src["mask"], src["extra"]).logits[:, -1].cpu())
            cache.append(entry)
            print(f"  heads/rank {i + 1}/{len(chunks)}", flush=True)
        ranked = {c: [h for h, _ in sorted(rank_recs[c].items(), key=lambda kv: -kv[1])] for c in CONDITIONS}
        rng = random.Random(args.seed + 7)
        head_sets = {}
        for cond in CONDITIONS:
            other = CONDITIONS[1 - CONDITIONS.index(cond)]
            sets = {f"top{k}": ranked[cond][:k] for k in args.head_ks}
            sets["image_movers"] = [h for h in movers if h[0] >= L_e[cond]]
            sets[f"{other}_top10"] = [h for h in ranked[other][:10] if h[0] >= L_e[cond]]
            pool = sorted({b for b, _ in ranked[cond][:10]})
            sets["random10"] = sorted(rng.sample([(b, h) for b in pool for h in range(m.n_heads)
                                                  if (b, h) not in set(ranked[cond][:10])], 10))
            if args.mover_subsets:
                # Appended after every existing set, so those arms keep their order and names.
                sets.update(mover_subsets(sets["image_movers"]))
            head_sets[cond] = sets
        head_recs = []
        for chunk, entry in zip(chunks, cache):
            sd = sides(chunk)
            for cond in CONDITIONS:
                base, _ = sd[cond]
                zp, lc, lp, ls = entry[cond]
                for arm, logits in (("clean", lc), ("source_clean", ls), ("entity_patch", lp)):
                    for r, met in zip(chunk, readout_metrics(logits, chunk)):
                        head_recs.append({"row_index": r["row_index"], "attribute": r["attribute"], "cond": cond,
                                          "arm": arm, **met})
                for name, heads in head_sets[cond].items():
                    if not heads:
                        continue
                    logits = head_install(m, base, heads, {b: v.to(m.model.device) for b, v in zp.items()})
                    for r, met in zip(chunk, readout_metrics(logits.cpu(), chunk)):
                        head_recs.append({"row_index": r["row_index"], "attribute": r["attribute"], "cond": cond,
                                          "arm": f"install:{name}", **met})
        report["heads"] = {}
        for cond in CONDITIONS:
            rr = [r for r in head_recs if r["cond"] == cond]
            elig = clean_eligible(rr)
            top = ranked[cond][:20]
            report["heads"][cond] = {
                "L_e": L_e[cond], "top20": top, "sets": head_sets[cond],
                "movers_in_top10": sorted(set(movers) & set(ranked[cond][:10])),
                "movers_in_top20": sorted(set(movers) & set(top)),
                "arms": summarize_arms(rr, elig), "arms_all_rows": summarize_arms(rr), "n_eligible": len(elig)}
            print_table(report["heads"][cond]["arms"], f"HEADS [{cond}] L_e={L_e[cond]}, n={len(elig)} eligible")
            print(f"  top-20: {', '.join(f'{b}.{h}' for b, h in top)}")
            print(f"  image movers in this top-10: {report['heads'][cond]['movers_in_top10']}")
        shared = set(ranked["image"][:10]) & set(ranked["text"][:10])
        report["heads"]["top10_overlap"] = sorted(shared)
        print(f"\n  image top-10 ∩ text top-10: {sorted(shared)} ({len(shared)}/10)")
        write_json(os.path.join(args.out_dir, "summary.json"), report)
        if args.mover_subsets:
            # Per-row ld/src_hit for every arm (summary.json holds only the aggregates); a
            # conjunction vs additivity reading needs the row-level differences.
            with open(os.path.join(args.out_dir, "heads_rows.jsonl"), "w") as fh:
                for r in head_recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"  wrote {args.out_dir}/heads_rows.jsonl ({len(head_recs)} records)")

    if "freeze" in args.phases:
        frz_recs = []
        for i, chunk in enumerate(chunks):
            if len(chunk) < 2 and "mean" in args.freeze_values:
                continue                                    # no leave-one-out mean from one row
            sd = sides(chunk)
            for cond in CONDITIONS:
                base, src = sd[cond]
                patch = entity_patch(base, src, L_e[cond])
                emit = lambda arm, lg: frz_recs.extend(  # noqa: E731
                    {"row_index": r["row_index"], "cond": cond, "arm": arm, **met}
                    for r, met in zip(chunk, readout_metrics(lg, chunk)))
                emit("clean", m.forward(base["ids"], base["mask"], base["extra"]).logits[:, -1])
                emit("source_clean", m.forward(src["ids"], src["mask"], src["extra"]).logits[:, -1])
                emit("entity_patch", m.forward(base["ids"], base["mask"], base["extra"], hooks=[patch]).logits[:, -1])
                for spec in args.freeze:
                    where, blocks = parse_freeze(spec, L_e[cond], m.n_layers)
                    if not blocks:
                        continue
                    clean_vals = mlp_capture(m, base, where, blocks)
                    for mode in args.freeze_values:
                        vals = clean_vals if mode == "base" else {b: loo_mean(v) for b, v in clean_vals.items()}
                        hooks = [patch] + mlp_freeze_hooks(m, base, where, blocks, vals)
                        lg = m.forward(base["ids"], base["mask"], base["extra"], hooks=hooks).logits[:, -1]
                        emit(f"freeze:{where}:{blocks[0]}-{blocks[-1]}@{mode}", lg)
            print(f"  freeze {i + 1}/{len(chunks)}", flush=True)
        report["freeze"] = {}
        for cond in CONDITIONS:
            rr = [r for r in frz_recs if r["cond"] == cond]
            elig = clean_eligible(rr)
            report["freeze"][cond] = {"L_e": L_e[cond], "n_eligible": len(elig), "arms": summarize_arms(rr, elig),
                                      "arms_all_rows": summarize_arms(rr)}
            print_table(report["freeze"][cond]["arms"], f"FREEZE [{cond}] entity patch at L_e={L_e[cond]}")
        write_json(os.path.join(args.out_dir, "summary.json"), report)

    if "crossmodal" in args.phases:
        report["crossmodal"], cm_recs = phase_crossmodal(m, args, entity_dir, items, chunks, sides, readout_metrics)
        write_json(os.path.join(args.out_dir, "summary.json"), report)
        with open(os.path.join(args.out_dir, "crossmodal_rows.jsonl"), "w") as fh:
            for r in cm_recs:
                fh.write(json.dumps(r) + "\n")

    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
