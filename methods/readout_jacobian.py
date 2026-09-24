"""Do different attributes READ different subspaces of the entity heads' output?

A direct test of R13.2 of ATTRIBUTE_HEAD_EXPERIMENTS.md, via linear relation
decoding (Hernandez et al. 2024, arXiv:2308.09124).

WHAT R13.2 CLAIMS, AND THE GAP IN IT

R13.2: an intervention at the `common10` heads is one fixed function of (base,
donor) activations, those activations are 77-89% entity and 3-15% question, so a
question-blind edit there cannot move the queried attribute without moving the
others. The gap: selectivity does not need the WRITER to see the question. If
the downstream computation that answers "capital" reads a different subspace of
the heads' output than the one answering "language", then an edit confined to
the capital-readout subspace moves the capital and leaves the language alone,
while never looking at which question is asked. Whether such subspaces exist is
a property of the downstream Jacobians, and it can be measured.

PHASE fit -- the readout Jacobians

For attribute r, block b, and a clean base row asking r's question, the readout
Jacobian is the derivative of the FIRST-ANSWER-TOKEN logits over r's candidate
values (every item's value, first token, under that template's prefill) with
respect to block b's selected-head columns at the last prompt column:

    J_r^b = d logits[cand_r] / d z_b        [n_cand x d_b]

It is a TOTAL derivative: block 21's includes every path through blocks 22-27,
including the other selected heads. Rather than one backward per candidate, it
is SKETCHED: each probe backprops a random zero-mean unit combination w of the
candidate logits (zero-mean, so the shared "raise every candidate" direction is
not what gets measured), giving one row w^T J. Stacking probes over items,
templates and seeds gives G_r^b, whose row space estimates what attribute r's
readout is sensitive to. Probes are packed one per batch row, each row carrying
its own zero perturbation leaf, so ONE backward returns a whole batch of rows.

PHASE geometry -- no model

    U_r^b      top right-singular vectors of G_r^b (--energy of its energy,
               capped at --max_rank)
    angles     principal-angle cosines between U_r^b and U_s^b
    cross      ||G_r U_s||^2 / ||G_r||^2 -- the fraction of r's readout
               sensitivity lying inside s's readout subspace (chance k_s/d)
    shared     how much of each attribute's sensitivity sits in ONE common
               subspace (Christ et al. 2025 predict a large shared "which
               country" component)
    reach      for an edit operator M (delta = (donor - base) M, row
               convention), reach[r_edit][s] = ||G_s M^T||^2 / ||G_s||^2 is the
               fraction of s's readout the edit can touch. A selective edit has
               a large diagonal and a small off-diagonal.

The edit operators are the pasted design, delta = P_perp(others) J_r^+ (target -
current), with "target - current" read as the readout change the DONOR would
cause, J_r (z_src - z_base). Then J_r^+ J_r is the projector onto J_r's row
space, P_r = U_r U_r^T, and the edit is

    readout    delta = P_r (z_src - z_base)
    isolated   delta = P_perp(others) P_r (z_src - z_base)
    random     delta = Q Q^T (z_src - z_base), Q a random rank-k_r basis

All three are source-derived and question-blind (the donor asks the SAME
question as the base, head_swap_vade's `queried`), so they are legitimate VADE
interventions; the target attribute only chooses WHICH operator is applied.

MAKELOV CONTROL (--das_checkpoint). Makelov et al. 2024 (arXiv:2311.17030):
a subspace patch can work through a direction downstream layers never read. For
a head_das.py das_rotated checkpoint, report how much of each attribute's
readout energy its learned core captures (against chance k_v/d) and how much of
the core lies inside the union of readout subspaces. A core that carries
entity-transfer but little readout energy is the dormant-pathway signature.

PHASE eval -- VADE-scored

For each target attribute, every row of its test tuple file (cause AND iso),
generated with head_swap_vade.patched_generate (continuous, the R10 path) under
clean / full / readout / isolated / random, written as VADE predictions
(<arm>.jsonl, one file per arm holding all targets) plus an in-script
cause/iso/final summary with VADE's own matcher. score.py remains the number of
record.

READING IT

  * cross ~ 1 and isolated reach ~ 0  -> the attributes read the same subspace.
    R13.2's cap is then PROVEN for linear edits, not just argued.
  * cross well below 1, isolated diagonal large, off-diagonal small -> a
    question-blind isolating edit exists; the eval phase is its test, and DAS
    (R12/R14) simply did not find it.
  * The first-token restriction is real: candidates are FIRST answer tokens, so
    this measures what determines the first token. The eval phase patches every
    step, so the full answer is still scored.

Usage
-----
    python methods/readout_jacobian.py --dry_run
    python methods/readout_jacobian.py --phase fit --items_limit 8 --probes 2     # smoke
    python methods/readout_jacobian.py --phase fit
    python methods/readout_jacobian.py --phase geometry --das_checkpoint path/to/arm.pt
    python methods/readout_jacobian.py --phase eval --limit_pairs 20
    python methods/readout_jacobian.py --phase all
"""
import argparse
import json
import os
import sys
import zlib
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import (COMMON10, DEFAULT_VADE_ROOT, MODEL_ID, entity_attributes,  # noqa: E402
                                    load_assets, parse_heads)

EDIT_ARMS = ("readout", "isolated", "random")


# ---------------------------------------------------------------------------
# Geometry (pure numpy; unit-tested on synthetic readouts)
# ---------------------------------------------------------------------------

def energy_rank(s, energy, max_rank):
    """Smallest k whose top-k squared singular values reach `energy` of the total."""
    import numpy as np
    e = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-30)
    k = int(np.searchsorted(e, energy) + 1)
    return max(1, min(k, max_rank, len(s)))


def readout_basis(G, energy, max_rank):
    """-> (U [d, k] orthonormal, singular values)."""
    import numpy as np
    _, s, Vt = np.linalg.svd(np.asarray(G, dtype=np.float64), full_matrices=False)
    k = energy_rank(s, energy, max_rank)
    return Vt[:k].T, s


def orth(M, tol=1e-8):
    """Orthonormal basis of span(columns of M)."""
    import numpy as np
    if M.shape[1] == 0:
        return M
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol * max(float(s[0]), 1e-30)]


def complement_projector(bases, d):
    """I - Q Q^T for Q spanning every basis in `bases`."""
    import numpy as np
    if not bases:
        return np.eye(d)
    Q = orth(np.concatenate(bases, axis=1))
    return np.eye(d) - Q @ Q.T


def energy_in(G, U):
    """||G U||^2 / ||G||^2: fraction of G's row energy inside span(U)."""
    import numpy as np
    tot = float((G ** 2).sum())
    return float(((G @ U) ** 2).sum()) / tot if tot > 0 else float("nan")


def reach(G, M):
    """||G M^T||^2 / ||G||^2 for a row-convention edit operator M."""
    tot = float((G ** 2).sum())
    return float(((G @ M.T) ** 2).sum()) / tot if tot > 0 else float("nan")


def principal_cosines(U, V):
    import numpy as np
    return np.linalg.svd(U.T @ V, compute_uv=False)


def random_basis(d, k, seed):
    import numpy as np
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return Q


def edit_operators(U, target, seed=0):
    """-> {arm: M [d, d]} for one block. U = {attr: basis}."""
    d = next(iter(U.values())).shape[0]
    P = U[target] @ U[target].T
    Pperp = complement_projector([U[a] for a in U if a != target], d)
    # Seeded per TARGET as well as per block, so the four targets' nulls are four
    # different random subspaces rather than one reused draw.
    Q = random_basis(d, U[target].shape[1], seed + zlib.crc32(target.encode()))
    return {"readout": P, "isolated": P @ Pperp, "random": Q @ Q.T}


def block_geometry(G, energy, max_rank, seed=0, G_eval=None):
    """G = {attr: [n_rows, d]} for ONE block -> (U, report dict).

    Subspaces are always fitted on G. Every energy/reach number is measured on
    G_eval when given (rows from HELD-OUT items), otherwise on G itself. The
    in-sample version is biased toward selectivity -- a subspace fitted to a
    finite sketch captures that sketch's own noise -- which is exactly the
    direction that would fake a positive result here."""
    import numpy as np
    E = G if G_eval is None else G_eval
    attrs = sorted(G)
    U, spec = {}, {}
    for a in attrs:
        U[a], s = readout_basis(G[a], energy, max_rank)
        spec[a] = (s ** 2 / max(float((s ** 2).sum()), 1e-30))[:max_rank].tolist()
    d = U[attrs[0]].shape[0]
    rep = {"d": d, "rank": {a: int(U[a].shape[1]) for a in attrs}, "spectrum": spec,
           "cross": {}, "cross_chance": {}, "cos_mean_sq": {}, "cos_max": {}}
    for r in attrs:
        for s_ in attrs:
            if r == s_:
                continue
            rep["cross"][f"{r}->{s_}"] = energy_in(E[r], U[s_])
            rep["cross_chance"][f"{r}->{s_}"] = U[s_].shape[1] / d
            c = principal_cosines(U[r], U[s_])
            rep["cos_mean_sq"][f"{r}|{s_}"] = float((c ** 2).mean())
            rep["cos_max"][f"{r}|{s_}"] = float(c.max())
    # One subspace shared by all: SVD of the stacked, per-attribute-normalized sketches.
    k_shared = min(U[a].shape[1] for a in attrs)
    stacked = np.concatenate([G[a] / max(np.linalg.norm(G[a]), 1e-30) for a in attrs])
    _, _, Vt = np.linalg.svd(stacked, full_matrices=False)
    S = Vt[:k_shared].T
    rep["shared_k"] = k_shared
    rep["shared_energy"] = {a: energy_in(E[a], S) for a in attrs}
    rep["self_energy"] = {a: energy_in(E[a], U[a]) for a in attrs}
    rep["reach"] = {}
    for t in attrs:
        ops = edit_operators(U, t, seed)
        rep["reach"][t] = {arm: {s_: reach(E[s_], M) for s_ in attrs} for arm, M in ops.items()}
    rep["measured_on"] = "in-sample" if G_eval is None else "held-out items"
    return U, rep


def selectivity(reach_row, target):
    """reach on the target / mean reach on the others (inf if the others are ~0)."""
    others = [v for a, v in reach_row.items() if a != target]
    off = sum(others) / len(others) if others else 0.0
    return reach_row[target] / off if off > 1e-12 else float("inf")


def item_folds(row_items, attributes):
    """Two folds by ITEM (alternating over the sorted item list), so no item's
    sketch rows land on both sides. -> {"items": [set, set], "rows": {attr: [idx0, idx1]}}."""
    import numpy as np
    items = sorted({k for a in attributes for k in row_items[a]})
    halves = [set(items[0::2]), set(items[1::2])]
    rows = {a: [np.array([i for i, k in enumerate(row_items[a]) if k in h], dtype=int) for h in halves]
            for a in attributes}
    for a in attributes:
        assert all(len(r) for r in rows[a]), f"{a}: a fold has no rows -- need >= 2 fit items"
    return {"items": halves, "rows": rows}


def average_reports(reps):
    """Leaf-wise mean of numeric fields across fold reports; non-numeric fields
    (and integer ranks) from the first, with ranks kept per fold for reference."""
    def avg(xs):
        x0 = xs[0]
        if isinstance(x0, dict):
            return {k: avg([x[k] for x in xs]) for k in x0}
        if isinstance(x0, float):
            return sum(xs) / len(xs)
        return x0
    out = avg(reps)
    out["rank_by_fold"] = [r["rank"] for r in reps]
    out["measured_on"] = "held-out items (2-fold mean)"
    return out


def das_control(G, U, V):
    """Makelov check for one block. V [k_v, d] orthonormal rows (a DAS core)."""
    import numpy as np
    d = V.shape[1]
    Vb = V.T
    union = orth(np.concatenate(list(U.values()), axis=1))
    return {"k_v": int(V.shape[0]), "chance": V.shape[0] / d,
            "readout_energy_in_core": {a: energy_in(G[a], Vb) for a in G},
            "core_in_union_readout": float(((V @ union) ** 2).sum()) / V.shape[0],
            "core_in_union_chance": union.shape[1] / d}


# ---------------------------------------------------------------------------
# Fit: the Jacobian sketch
# ---------------------------------------------------------------------------

def probe_weights(n, seed):
    """Zero-mean, unit-norm Gaussian weights over n candidates."""
    import numpy as np
    w = np.random.default_rng(seed).standard_normal(n)
    w -= w.mean()
    return w / max(np.linalg.norm(w), 1e-12)


def jacobian_rows(adapter, model, ids, mask, extra, heads, head_dim, cand_lists, weights):
    """One forward + one backward for a LEFT-padded batch. Row i gets its own
    zero leaf on each block's selected-head columns at the last column, and its
    own probe weights over its own candidates, so the backward yields one sketch
    row per batch row. -> ({block: [B, d_b] float32 CPU}, clean logits [B, V])."""
    import torch
    from methods.head_swap_vade import head_columns, to_device
    blocks = sorted({b for b, _ in heads})
    B = ids.shape[0]
    leaves, handles = {}, []
    for b in blocks:
        cols = head_columns(heads, b, head_dim).to(model.device)
        leaves[b] = torch.zeros(B, len(cols), device=model.device, dtype=torch.float32, requires_grad=True)

        def pre(_m, args, _cols=cols, _eps=leaves[b]):
            t = args[0]
            add = torch.zeros(t.shape[0], t.shape[-1], device=t.device, dtype=_eps.dtype)
            add = add.index_copy(1, _cols, _eps)
            last = t[:, -1, :] + add.to(t.dtype)
            return (torch.cat([t[:, :-1, :], last.unsqueeze(1)], dim=1),) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(pre))
    try:
        with torch.enable_grad():
            out = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                        **to_device(extra, model.device, model.dtype), use_cache=False, logits_to_keep=1)
            logits = out.logits[:, -1].float()
            loss = logits.new_zeros(())
            for i in range(B):
                c = torch.tensor(cand_lists[i], device=logits.device)
                w = torch.tensor(weights[i], device=logits.device, dtype=logits.dtype)
                loss = loss + (logits[i, c] * w).sum()
            loss.backward()
    finally:
        for h in handles:
            h.remove()
    return {b: leaves[b].grad.detach().float().cpu() for b in blocks}, logits.detach().cpu()


def fit_units(items, attributes, lookup, templates, probes, seed):
    """-> [(attribute, item, template_id, probe_seed)], deterministic."""
    units = []
    for a in attributes:
        tids = [t for t in sorted(lookup[a]) if t.rsplit("_", 1)[-1] in templates]
        assert tids, f"{a}: no template among {sorted(lookup[a])} has a suffix in {templates}"
        for k in items:
            for t in tids:
                for p in range(probes):
                    # zlib, not hash(): str hashes are salted per process, which would
                    # silently change every probe (and the fitted subspaces) between runs.
                    units.append((a, k, t, zlib.crc32(f"{seed}|{a}|{k}|{t}|{p}".encode())))
    return units


def candidates(tokenizer, items, attributes, lookup):
    """-> {(attr, template_id): (distinct first-token ids, {item: own first token})}."""
    from methods.common.targets import derive_gold_token_ids
    out = {}
    for a in attributes:
        for tid, t in lookup[a].items():
            own = {}
            for k, it in items.items():
                if it.get(a) in (None, ""):
                    continue
                toks = derive_gold_token_ids(tokenizer, t["prefill"], str(it[a]))
                if toks:
                    own[k] = toks[0]
            out[(a, tid)] = (sorted(set(own.values())), own)
    return out


def run_fit(args, heads, items, entity_dir, lookup, attributes, fit_items):
    import numpy as np
    import torch
    from PIL import Image
    from methods.adapters.registry import get_adapter
    from methods.head_swap_vade import build_prompt

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(device=args.device)
    tok = processor.tokenizer
    head_dim = adapter.hidden_size(model) // adapter.n_attention_heads(model)
    pad_id = tok.pad_token_id or tok.eos_token_id
    cands = candidates(tok, items, attributes, lookup)
    units = fit_units(fit_items, attributes, lookup, args.fit_templates, args.probes, args.seed)
    blocks = sorted({b for b, _ in heads})
    G = {a: {b: [] for b in blocks} for a in attributes}
    row_items = {a: [] for a in attributes}
    acc = defaultdict(lambda: [0, 0])
    image_cache = {}

    def prompt(k, tid, a):
        if k not in image_cache:
            with Image.open(os.path.join(entity_dir, items[k]["image"])) as im:
                image_cache[k] = im.convert("RGB")
        t = lookup[a][tid]
        return build_prompt(processor, image_cache[k], t["question"], t["prefill"])

    print(f"  {len(units)} probe rows over {len(fit_items)} items; batch {args.jac_batch}", flush=True)
    for start in range(0, len(units), args.jac_batch):
        chunk = units[start:start + args.jac_batch]
        ps = [prompt(k, t, a) for a, k, t, _ in chunk]
        n = max(p["input_ids"].shape[1] for p in ps)
        ids = torch.full((len(ps), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ps), n), dtype=torch.long)
        for i, p in enumerate(ps):
            L = p["input_ids"].shape[1]
            ids[i, n - L:] = p["input_ids"][0]
            mask[i, n - L:] = 1
        extra = {"pixel_values": torch.cat([p["pixel_values"] for p in ps]),
                 "image_grid_thw": torch.cat([p["image_grid_thw"] for p in ps])}
        cl = [cands[(a, t)][0] for a, _, t, _ in chunk]
        ws = [probe_weights(len(c), s) for c, (_, _, _, s) in zip(cl, chunk)]
        rows, logits = jacobian_rows(adapter, model, ids, mask, extra, heads, head_dim, cl, ws)
        for i, (a, k, t, _) in enumerate(chunk):
            row_items[a].append(k)
            for b in blocks:
                G[a][b].append(rows[b][i].numpy())
            own = cands[(a, t)][1].get(k)
            if own is not None:
                c = cands[(a, t)][0]
                acc[a][0] += int(c[int(logits[i, c].argmax())] == own)
                acc[a][1] += 1
        print(f"    {min(start + args.jac_batch, len(units))}/{len(units)}", flush=True)
    G = {a: {b: np.stack(v) for b, v in bb.items()} for a, bb in G.items()}
    clean_acc = {a: acc[a][0] / max(acc[a][1], 1) for a in attributes}
    print(f"  clean first-token accuracy among candidates: "
          + ", ".join(f"{a} {v:.1%}" for a, v in clean_acc.items()))
    return G, row_items, clean_acc


# ---------------------------------------------------------------------------
# Eval: VADE-scored edits through head_swap_vade's continuous patch
# ---------------------------------------------------------------------------

def transform_for(M_by_block, device):
    """-> head_swap_vade `transform`: have + (want - have) @ M, per block."""
    import torch
    out = {}
    for b, M in M_by_block.items():
        Mt = torch.as_tensor(M, dtype=torch.float32, device=device)

        def fn(have, want, _M=Mt):
            return (have.float() + (want.float() - have.float()) @ _M).to(have.dtype)
        out[b] = fn
    return out


def quick_scores(preds, tuple_rows, matcher):
    """VADE-shaped cause / iso / final from (target, row_index) -> text."""
    cause, iso = [], defaultdict(list)
    for (target, ri), text in preds.items():
        r = tuple_rows[(target, ri)]
        if r["rule"] == "match_source":
            cause.append(matcher(text, str(r["source_label"])))
        else:
            iso[r["queried"]].append(matcher(text, str(r["base_label"])))
    c = sum(cause) / max(len(cause), 1)
    keeps = {q: sum(v) / len(v) for q, v in iso.items() if v}
    iso_mean = sum(keeps.values()) / len(keeps) if keeps else float("nan")
    return {"cause": c, "iso_by_queried": keeps, "iso_mean": iso_mean, "final": 0.5 * (c + iso_mean),
            "n_cause": len(cause), "n_iso": sum(len(v) for v in iso.values())}


def run_eval(args, heads, items, entity_dir, lookup, attributes, fit):
    import torch
    from methods.adapters.registry import get_adapter
    from methods.head_cross import vade_matcher
    from methods.head_swap_vade import (batches_by_donor_attribute, build_batch, capture_donor, decode,
                                        load_jobs, patched_generate, plain_generate)

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(device=args.device)
    head_dim = adapter.hidden_size(model) // adapter.n_attention_heads(model)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    matcher = vade_matcher(args.vade_root)
    blocks = sorted({b for b, _ in heads})
    U = {a: {b: fit["U"][a][b].numpy() for b in blocks} for a in attributes}
    arms = ["clean", "full"] + list(EDIT_ARMS)
    os.makedirs(args.out_dir, exist_ok=True)
    files = {arm: open(os.path.join(args.out_dir, f"{arm}.jsonl"), "w") for arm in arms}
    summary = {}
    try:
        for target in args.eval_attributes or attributes:
            ops = {b: edit_operators({a: U[a][b] for a in attributes}, target, args.seed + b) for b in blocks}
            trans = {arm: transform_for({b: ops[b][arm] for b in blocks}, model.device) for arm in EDIT_ARMS}
            jobs, _ = load_jobs(args.vade_root, args.entity, args.split, [target], "queried",
                                args.limit_pairs, args.seed)
            path = os.path.join(args.vade_root, "data", args.entity, "tuples", target, f"{args.split}.jsonl")
            tuple_rows = {(target, r["row_index"]): r for r in map(json.loads, open(path))}
            preds = {arm: {} for arm in arms}
            checked = False
            for _, chunk in batches_by_donor_attribute(jobs, args.batch_size):
                batch = build_batch(chunk, processor, items, entity_dir, lookup, pad_id)
                z = capture_donor(adapter, model, blocks, batch["donor_ids"], batch["donor_mask"],
                                  batch["donor_extra"], args.max_new_tokens, pad_id)
                gen = lambda tr=None: patched_generate(adapter, model, heads, z, batch["base_ids"],
                                                       batch["base_mask"], batch["base_extra"],
                                                       args.max_new_tokens, pad_id, head_dim, transform=tr)
                outs = {"clean": plain_generate(model, batch["base_ids"], batch["base_mask"], batch["base_extra"],
                                                args.max_new_tokens, pad_id),
                        "full": gen()}
                if not checked:
                    # Gate: M = 0 must be the clean run exactly; M = I must reproduce the plain
                    # head patch (up to bf16 rounding of have + (want - have)).
                    zero = gen(transform_for({b: 0 * ops[b]["readout"] for b in blocks}, model.device))
                    assert torch.equal(zero, outs["clean"]), (
                        "M=0 transform changed the clean generation -- the transform path is not a no-op")
                    import numpy as np
                    ident = gen(transform_for({b: np.eye(ops[b]["readout"].shape[0]) for b in blocks},
                                              model.device))
                    agree = float((ident == outs["full"]).all(1).float().mean())
                    print(f"  [{target}] transform gate: M=0 == clean (exact); M=I == full on {agree:.0%} of rows")
                    assert agree >= 0.75, "M=I transform disagrees with the plain head patch on most rows"
                    checked = True
                for arm in EDIT_ARMS:
                    outs[arm] = gen(trans[arm])
                for arm, toks in outs.items():
                    for j, text in zip(chunk, decode(processor, toks)):
                        for t_attr, ri in j["rows"]:
                            preds[arm][(t_attr, ri)] = text
                            files[arm].write(json.dumps({"attribute": t_attr, "row_index": ri,
                                                         "generated_text": text}, ensure_ascii=False) + "\n")
                    files[arm].flush()
                print(f"  [{target}] {sum(1 for _ in preds['clean'])} rows", flush=True)
            summary[target] = {arm: quick_scores(preds[arm], tuple_rows, matcher) for arm in arms}
    finally:
        for f in files.values():
            f.close()
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_geometry(geo, attributes):
    for b, rep in geo["blocks"].items():
        print(f"\n=== block {b} (d={rep['d']}) ranks per fold: {rep['rank_by_fold']}  [{rep['measured_on']}]")
        ins = geo["in_sample"][b]["self_energy"]
        print("  own-subspace energy held-out vs in-sample (a large gap = the subspace is fitting noise): "
              + ", ".join(f"{a} {rep['self_energy'][a]:.3f}/{ins[a]:.3f}" for a in attributes))
        print("  cross-energy r->s = share of r's readout inside s's subspace (chance in brackets):")
        for r in attributes:
            cells = []
            for s in attributes:
                if r == s:
                    cells.append(f"{'--':>13}")
                else:
                    cells.append(f"{rep['cross'][f'{r}->{s}']:6.3f}[{rep['cross_chance'][f'{r}->{s}']:.3f}]".rjust(13))
            print(f"    {r:>13} " + "".join(cells))
        print(f"  shared subspace (k={rep['shared_k']}) energy: "
              + ", ".join(f"{a} {v:.3f}" for a, v in rep["shared_energy"].items()))
        for arm in EDIT_ARMS:
            print(f"  reach[{arm}] (row = edited target, col = readout touched) / selectivity:")
            for t in attributes:
                row = rep["reach"][t][arm]
                print(f"    {t:>13} " + "".join(f"{row[s]:8.3f}" for s in attributes)
                      + f"   sel {selectivity(row, t):6.2f}")
        if "das_control" in rep:
            dc = rep["das_control"]
            print(f"  DAS core k={dc['k_v']}: readout energy captured (chance {dc['chance']:.3f}): "
                  + ", ".join(f"{a} {v:.3f}" for a, v in dc["readout_energy_in_core"].items())
                  + f"; core inside union readout {dc['core_in_union_readout']:.3f} "
                    f"(chance {dc['core_in_union_chance']:.3f})")


def print_eval(summary):
    for target, arms in summary.items():
        print(f"\n=== target {target} ===")
        print(f"{'arm':>9} {'cause':>7} {'iso':>7} {'final':>7}   iso by queried")
        for arm, s in arms.items():
            print(f"{arm:>9} {s['cause']:7.1%} {s['iso_mean']:7.1%} {s['final']:7.1%}   "
                  + ", ".join(f"{q} {v:.0%}" for q, v in s["iso_by_queried"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fit", "geometry", "eval", "all"], default="all")
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--heads", default=COMMON10)
    ap.add_argument("--attributes", nargs="+", default=None)
    ap.add_argument("--fit_split", default="train", help="Items are taken from this split's tuples.")
    ap.add_argument("--fit_templates", nargs="+", default=["v1", "v2", "v3", "v4"],
                    help="Template-id suffixes used to fit; keep them disjoint from the eval templates you "
                         "care about (VADE's test rows use all six).")
    ap.add_argument("--items_limit", type=int, default=0)
    ap.add_argument("--probes", type=int, default=4, help="Random candidate combinations per (item, template).")
    ap.add_argument("--jac_batch", type=int, default=8)
    ap.add_argument("--energy", type=float, default=0.9)
    ap.add_argument("--max_rank", type=int, default=64)
    ap.add_argument("--das_checkpoint", default=None, help="head_das.py das_rotated .pt for the Makelov control.")
    ap.add_argument("--split", default="test", help="eval phase: which tuples to score.")
    ap.add_argument("--eval_attributes", nargs="+", default=None)
    ap.add_argument("--limit_pairs", type=int, default=60,
                    help="eval phase: distinct (base, source) pairs per target (~24 rows each). 0 = the full "
                         "split, which is 14,052 rows x 5 arms x 4 targets ~ 281k generations on flags.")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    attributes = args.attributes or [a for a in entity_attributes(gt) if a in lookup]
    args.out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "readout_jacobian", args.entity)
    fit_items = sorted({json.loads(l)["base"] for a in attributes
                        for l in open(os.path.join(entity_dir, "tuples", a, f"{args.fit_split}.jsonl"))})
    if args.items_limit:
        fit_items = fit_items[:args.items_limit]
    units = fit_units(fit_items, attributes, lookup, args.fit_templates, args.probes, args.seed)
    fit_path = os.path.join(args.out_dir, "readout_fit.pt")
    print(f"[readout_jacobian] {args.entity} phase={args.phase}: heads "
          + ", ".join(f"{b}.{h}" for b, h in heads))
    print(f"  attributes: {attributes}; fit on {len(fit_items)} {args.fit_split} items x templates "
          f"{args.fit_templates} x {args.probes} probes = {len(units)} sketch rows")
    print(f"  -> {args.out_dir}")
    if args.dry_run:
        for k in fit_items:
            assert os.path.exists(os.path.join(entity_dir, items[k]["image"])), f"missing image for {k}"
        print("Grid valid.")
        return

    import torch
    os.makedirs(args.out_dir, exist_ok=True)
    if args.phase in ("fit", "all"):
        G, row_items, clean_acc = run_fit(args, heads, items, entity_dir, lookup, attributes, fit_items)
        torch.save({"G": {a: {b: torch.from_numpy(m) for b, m in bb.items()} for a, bb in G.items()},
                    "row_items": row_items,
                    "heads": heads, "attributes": attributes, "fit_items": fit_items,
                    "fit_templates": args.fit_templates, "probes": args.probes, "clean_acc": clean_acc},
                   fit_path)
        print(f"  wrote {fit_path}")

    if args.phase in ("geometry", "eval", "all"):
        fit = torch.load(fit_path, weights_only=False)
        assert [tuple(h) for h in fit["heads"]] == heads, "readout_fit.pt was fitted on a different head set"
        G = {a: {b: m.numpy() for b, m in bb.items()} for a, bb in fit["G"].items()}
        blocks = sorted({b for b, _ in heads})
        das = None
        if args.das_checkpoint:
            from methods.das_subspace_geometry import load_arm
            das = load_arm(args.das_checkpoint, args.vade_root)
        folds = item_folds(fit["row_items"], attributes)
        geo = {"energy": args.energy, "max_rank": args.max_rank, "blocks": {}, "in_sample": {},
               "folds": {"n_items": [len(f) for f in folds["items"]]}}
        fit["U"] = {a: {} for a in attributes}
        for b in blocks:
            Gb = {a: G[a][b] for a in attributes}
            # Subspaces for the EDITS use every row; the numbers REPORTED as the result are
            # two-fold item-held-out (fit on one half of the items, measure on the other).
            U, insample = block_geometry(Gb, args.energy, args.max_rank, args.seed + b)
            halves = [{a: Gb[a][folds["rows"][a][i]] for a in attributes} for i in (0, 1)]
            rep = average_reports([block_geometry(halves[i], args.energy, args.max_rank, args.seed + b,
                                                  G_eval=halves[1 - i])[1] for i in (0, 1)])
            if das is not None:
                V = das[b][0].numpy().astype("float64")
                assert V.shape[1] == rep["d"], f"DAS core width {V.shape[1]} != block {b} width {rep['d']}"
                rep["das_control"] = das_control(Gb, U, V)
            geo["blocks"][b] = rep
            geo["in_sample"][b] = insample
            for a in attributes:
                fit["U"][a][b] = torch.from_numpy(U[a])
        with open(os.path.join(args.out_dir, "geometry.json"), "w") as f:
            json.dump(geo, f, indent=2)
        print_geometry(geo, attributes)

        if args.phase in ("eval", "all"):
            summary = run_eval(args, heads, items, entity_dir, lookup, attributes, fit)
            with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print_eval(summary)
            print("\nScore each arm with VADE's own scorer (the number of record):")
            for arm in ["clean", "full"] + list(EDIT_ARMS):
                print(f"  python {os.path.join(args.vade_root, 'eval', 'score.py')} --entity {args.entity} "
                      f"--attribute all --predictions {os.path.join(args.out_dir, arm + '.jsonl')}")


if __name__ == "__main__":
    main()
