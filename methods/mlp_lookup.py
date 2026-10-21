"""X7: how the asked attribute and the entity combine into the answer in the late MLPs.

The see -> move -> recall account says the movers (blocks 21-23) write WHICH COUNTRY into the
answer column and the routers (19-21) write WHICH ATTRIBUTE WAS ASKED, and that last-token MLPs
23-26 turn the two into the fact. R13.1 already measured the shape of that from the crossed
grid: the residual entering the MLPs is ADDITIVE in (item, attribute) -- interaction <= 3.4% of
variance through block 22 -- while mlp_output's interaction share rises to 0.40 by block 26.
This script tests it causally and at neuron level, on flags with the controlled prompts
(attribute_switch_sweep: every flag x attribute prompt has one token length, so batches are
unpadded and the last prompt column is the same column everywhere).

PHASES
    capture   ONE forward per (item, attribute) cell of the crossed grid (default all 84 flags x
              4 attributes = 336 runs) plus one clean greedy generation per cell. At the last
              prompt column it stores:
                resid.npy        [cells, len(--resid_layers), hidden]  the residual AFTER block L-1
                                 (layer convention of hooks.register_patch_hook / residual_patch)
                mlp_hidden.npy   [cells, len(--mlp_blocks), intermediate]  down_proj's input, i.e.
                                 the post-SwiGLU neurons of block b (sites.py's mlp_hidden)
                mlp_output.npy   [cells, len(--mlp_blocks), hidden]  what block b's MLP writes
              index.jsonl carries item, attribute, the clean answer and whether it is correct.
    compose   Generation. Base prompt (image C, asked X); at layer L the last prompt column's
              residual is REPLACED by a vector built from the grid, with no run of the target cell:
                clean            nothing                                         -> C.X
                actual           R[B, A]: the real residual of the (B, A) run    -> B.A (ceiling)
                additive         rowmean_{-A}(B) + colmean_{-B}(A) - mean_{-B,-A} -> B.A
                swap_entity      R[C, X] + rowmean_{-X}(B) - rowmean_{-X}(C)       -> B.X
                swap_question    R[C, X] + colmean_{-C}(A) - colmean_{-C}(X)       -> C.A
                random_offset    R[C, X] + a random vector of ||additive - R[C,X]|| -> C.X
              Every estimate leaves out the cell it predicts, so `additive` is built ONLY from runs
              that show B with another question and other flags asked A. If `additive` gives B.A,
              the late blocks turn a plain SUM of an entity vector and a question vector into the
              specific fact: the lookup is a function of (entity + attribute).
              Readout: the first content token of the greedy answer against the four golds' first
              content tokens (rows are kept only when all four differ and the clean model answers
              all four cells correctly), plus a VADE-matcher text hit. A prefill patch at the last
              column controls the FIRST answer token only (CLAUDE.md, swap_trace), so `first` is the
              primary column; `text` is reported for single-token attributes' sake.
              Late attention still reads image C and question X from the context, which is why
              `actual` is reported per layer: it is the ceiling each layer's construction can reach.
    neurons   CPU only, off capture's arrays:
                * the exact two-way decomposition SS_total = SS_item + SS_attr + SS_interaction
                  (balanced grid, one deterministic run per cell, so the residual IS the
                  interaction) for the residual at every captured layer and for MLP neurons at
                  every captured block -- R13.1 extended to neuron space;
                * per-neuron shares and a count of item / attribute / interaction neurons among
                  the --top_k highest-variance neurons of each block;
                * the VALUE test: for attributes whose values are SHARED across countries
                  (language, currency, calling-code zone = first digit), how much of a neuron's
                  variance across countries the VALUE explains (eta^2) when that attribute is asked
                  vs when another is asked. A lookup neuron keyed by the answer should group
                  Spain/Mexico/Argentina when asked language, and not when asked capital. Mean
                  difference over the block's top-k interaction neurons, against a permutation null
                  (values shuffled across countries);
                * heatmap.npz: the top interaction neurons of --heatmap_block, z-scored, for the
                  notebook.

Usage
-----
    python methods/mlp_lookup.py --phase capture --dry_run
    python methods/mlp_lookup.py --phase all --limit_items 12 --n_rows 48      # smoke
    python methods/mlp_lookup.py --phase all                                   # full
    python methods/mlp_lookup.py --phase neurons                               # CPU re-analysis
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from methods.circuit_common import DEFAULT_VADE_ROOT, MODEL_ID, load_entity  # noqa: E402
from methods.attribute_switch_sweep import ATTRIBUTES  # noqa: E402

ARMS = ("clean", "actual", "additive", "swap_entity", "swap_question", "random_offset")
EXPECT = {"clean": "C.X", "actual": "B.A", "additive": "B.A", "swap_entity": "B.X",
          "swap_question": "C.A", "random_offset": "C.X"}
TARGETS = ("B.A", "C.X", "B.X", "C.A")
SHARED_VALUE_ATTRS = ("language", "currency", "calling_code")


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------

def two_way(X):
    """X [I, A, D] (balanced, one value per cell) -> dict of per-dimension sums of squares:
    item, attr, inter, total (each [D]). Exact: total == item + attr + inter."""
    X = np.asarray(X, dtype=np.float64)
    I, A = X.shape[:2]
    mu = X.mean(axis=(0, 1))
    ri = X.mean(axis=1) - mu
    ca = X.mean(axis=0) - mu
    inter = X - mu - ri[:, None] - ca[None]
    return {"item": A * (ri ** 2).sum(0), "attr": I * (ca ** 2).sum(0), "inter": (inter ** 2).sum((0, 1)),
            "total": ((X - mu) ** 2).sum((0, 1))}


def shares(ss):
    """Aggregate shares over all dimensions."""
    t = float(ss["total"].sum())
    return {k: float(ss[k].sum()) / t if t else 0.0 for k in ("item", "attr", "inter")}


def row_mean_excl(R, i, a):
    """Mean of item i's cells over every attribute except a. R [I, A, ...]."""
    keep = [x for x in range(R.shape[1]) if x != a]
    return R[i, keep].mean(0)


def col_mean_excl(R, i, a):
    """Mean of attribute a's cells over every item except i."""
    keep = [x for x in range(R.shape[0]) if x != i]
    return R[keep, a].mean(0)


def additive_loo(R, i, a):
    """Additive prediction of cell (i, a) from the grid WITHOUT that cell's own row or column
    entry: rowmean_{-a}(i) + colmean_{-i}(a) - mean over cells (i' != i, a' != a).
    Exact for a perfectly additive grid."""
    rest = np.delete(np.delete(R, i, axis=0), a, axis=1)
    return row_mean_excl(R, i, a) + col_mean_excl(R, i, a) - rest.mean((0, 1))


def arm_vector(arm, R, b_i, a_i, c_i, x_i, rng=None):
    """The residual installed at the base (C, X) column for `arm`. R [I, A, H] at one layer."""
    base = R[c_i, x_i]
    if arm == "actual":
        return R[b_i, a_i]
    if arm == "additive":
        return additive_loo(R, b_i, a_i)
    if arm == "swap_entity":
        return base + row_mean_excl(R, b_i, x_i) - row_mean_excl(R, c_i, x_i)
    if arm == "swap_question":
        return base + col_mean_excl(R, c_i, a_i) - col_mean_excl(R, c_i, x_i)
    if arm == "random_offset":
        d = additive_loo(R, b_i, a_i) - base
        g = (rng or np.random.default_rng(0)).standard_normal(base.shape)
        return base + g / np.linalg.norm(g) * np.linalg.norm(d)
    raise ValueError(arm)


def sample_compose_rows(items_order, attributes, first_tok, correct, n_rows, seed):
    """-> [(C, X, B, A)] index tuples. A row needs C != B, X != A, all four cells answered
    correctly by the clean model, and four distinct gold first tokens (so the first answer token
    says which cell the model is reading)."""
    I, A = len(items_order), len(attributes)
    cand = [(c, x, b, a) for c in range(I) for b in range(I) if b != c
            for x in range(A) for a in range(A) if a != x]
    random.Random(seed).shuffle(cand)
    out = []
    for c, x, b, a in cand:
        cells = [(b, a), (c, x), (b, x), (c, a)]
        if not all(correct[cell] for cell in cells):
            continue
        toks = [first_tok[cell] for cell in cells]
        if None in toks or len(set(toks)) < 4:
            continue
        out.append((c, x, b, a))
        if len(out) >= n_rows:
            break
    return out


def value_groups(values):
    """values [I] (str) -> (kept item indices, group labels) for items whose value is shared
    with at least one other item."""
    cnt = Counter(values)
    kept = [i for i, v in enumerate(values) if cnt[v] >= 2]
    return kept, [values[i] for i in kept]


def eta_sq(Y, labels):
    """Y [n, D], labels [n] -> eta^2 per dimension (between-group SS / total SS)."""
    Y = np.asarray(Y, dtype=np.float64)
    mu = Y.mean(0)
    tot = ((Y - mu) ** 2).sum(0)
    between = np.zeros(Y.shape[1])
    for g in set(labels):
        idx = [i for i, lab in enumerate(labels) if lab == g]
        between += len(idx) * (Y[idx].mean(0) - mu) ** 2
    return np.divide(between, tot, out=np.zeros_like(tot), where=tot > 0)


def value_test(Y, values, asked_idx, neurons, n_perm=200, seed=0):
    """Y [I, A, D] neuron grid; values [I] of attribute `asked_idx`. Mean over `neurons` of
    eta^2(value | that attribute asked) - mean_{other attributes} eta^2(value | other asked),
    with a permutation null that shuffles values across the kept items."""
    kept, labels = value_groups(values)
    if len(set(labels)) < 2:
        return None
    Yk = Y[kept][:, :, neurons]
    A = Y.shape[1]

    def diff(lab):
        asked = eta_sq(Yk[:, asked_idx], lab)
        other = np.mean([eta_sq(Yk[:, a], lab) for a in range(A) if a != asked_idx], axis=0)
        return asked - other, asked, other

    d, asked, other = diff(labels)
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        lab = labels[:]
        rng.shuffle(lab)
        null.append(float(diff(lab)[0].mean()))
    obs = float(d.mean())
    return {"n_items": len(kept), "n_values": len(set(labels)), "mean_diff": obs,
            "mean_eta_asked": float(asked.mean()), "mean_eta_other": float(other.mean()),
            "frac_asked_gt_other": float((d > 0).mean()),
            "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)),
            "p_value": (1 + sum(n >= obs for n in null)) / (1 + len(null))}


def attribute_values(items, keys, attribute):
    v = [str(items[k][attribute]) for k in keys]
    if attribute == "calling_code":
        v = [s.strip().lstrip("+")[:1] for s in v]        # ITU zone = first digit
    if attribute == "currency":
        v = ["EUR" if s == "EUR" else "other" for s in v]  # EUR is flags' only shared currency
    return v


def summarize_compose(rows):
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["layer"])].append(r)
    for (arm, layer), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], ARMS.index(kv[0][0]))):
        n = len(rs)
        first = Counter(r["first_hit"] for r in rs)
        text = Counter(r["text_hit"] for r in rs)
        out[str(layer)][arm] = {"n": n, "expect": EXPECT[arm],
                                **{f"first_{t}": first.get(t, 0) / n for t in TARGETS},
                                "first_other": first.get("other", 0) / n,
                                **{f"text_{t}": text.get(t, 0) / n for t in TARGETS},
                                "text_other": text.get("other", 0) / n}
    return dict(out)


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

def block_output_hook(layers, block, positions, sink):
    """Appends decoder block `block`'s OUTPUT (the pre-norm residual entering block+1) at
    `positions` to `sink`. Same as head_trace._residual_hook, kept local so this script does not
    import head_trace (and with it pyvene), which molab's environment does not install. Read
    through a hook, not out.hidden_states, whose last entry is post-final-norm."""
    import torch
    B = positions.shape[0]

    def post(mod, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        sink.append(torch.stack([t[i, positions[i]] for i in range(B)]).detach())

    return layers[block].register_forward_hook(post)


def grid_cells(keys, attributes):
    return [(k, a) for k in keys for a in attributes]          # item-major, attribute-minor


def run_capture(m, entity_dir, items, keys, attributes, resid_layers, mlp_blocks, batch_size,
                max_new_tokens, out_dir):
    import torch
    from methods.circuit_common import cat_extra, first_content_token
    from methods.common.sites import InterventionSite
    from methods.common.text_match import contains_label

    cells = grid_cells(keys, attributes)
    inter = m.adapter.intermediate_size(m.model)
    resid = np.lib.format.open_memmap(os.path.join(out_dir, "resid.npy"), mode="w+", dtype=np.float32,
                                      shape=(len(cells), len(resid_layers), m.hidden))
    neur = np.lib.format.open_memmap(os.path.join(out_dir, "mlp_hidden.npy"), mode="w+", dtype=np.float32,
                                     shape=(len(cells), len(mlp_blocks), inter))
    mout = np.lib.format.open_memmap(os.path.join(out_dir, "mlp_output.npy"), mode="w+", dtype=np.float32,
                                     shape=(len(cells), len(mlp_blocks), m.hidden))
    index = []
    site, out_site = InterventionSite("mlp_hidden"), InterventionSite("mlp_output")
    for s in range(0, len(cells), batch_size):
        chunk = cells[s:s + batch_size]
        built = [m.image_prompt(entity_dir, items[k], a) for k, a in chunk]
        assert len({b[0].shape[0] for b in built}) == 1, "controlled prompts must share one length"
        ids = torch.stack([b[0] for b in built])
        mask = torch.ones_like(ids)
        extra = cat_extra([b[1] for b in built])
        pos = torch.full((len(chunk), 1), ids.shape[1] - 1, dtype=torch.long)
        r_sink = {L: [] for L in resid_layers}
        n_sink = {b: [] for b in mlp_blocks}
        o_sink = {b: [] for b in mlp_blocks}
        handles = [block_output_hook(m.layers, L - 1, pos, r_sink[L]) for L in resid_layers]
        handles += [site.register_capture(m.adapter, m.model, b + 1, pos, n_sink[b]) for b in mlp_blocks]
        handles += [out_site.register_capture(m.adapter, m.model, b + 1, pos, o_sink[b]) for b in mlp_blocks]
        try:
            m.forward(ids, mask, extra)
        finally:
            for h in handles:
                h.remove()
        for j, L in enumerate(resid_layers):
            assert len(r_sink[L]) == 1
            resid[s:s + len(chunk), j] = r_sink[L][0][:, 0].float().cpu().numpy()
        for j, b in enumerate(mlp_blocks):
            assert len(n_sink[b]) == 1
            neur[s:s + len(chunk), j] = n_sink[b][0][:, 0].float().cpu().numpy()
            assert len(o_sink[b]) == 1
            mout[s:s + len(chunk), j] = o_sink[b][0][:, 0].float().cpu().numpy()
        gen = m.generate(ids, mask, extra, max_new_tokens)
        for (k, a), g, text in zip(chunk, gen, m.decode(gen)):
            ft = first_content_token(m.tokenizer, g.tolist())
            index.append({"item": k, "attribute": a, "clean": text, "first_tok": ft,
                          "correct": bool(contains_label(text, str(items[k][a])))})
        print(f"  captured {s + len(chunk)}/{len(cells)}", flush=True)
    for a in (resid, neur, mout):
        a.flush()
    with open(os.path.join(out_dir, "index.jsonl"), "w") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(r["correct"] for r in index) / len(index)
    print(f"clean accuracy over the grid: {acc:.1%}")
    return index


def run_compose(m, entity_dir, items, keys, attributes, meta, layers, n_rows, batch_size,
                max_new_tokens, seed, out_dir):
    import torch
    from methods.circuit_common import cat_extra, first_content_token, gold_ids, residual_patch
    from methods.common.text_match import contains_label

    index = [json.loads(l) for l in open(os.path.join(out_dir, "index.jsonl"))]
    I, A = len(keys), len(attributes)
    resid = np.load(os.path.join(out_dir, "resid.npy"), mmap_mode="r")
    resid_layers = meta["resid_layers"]
    missing = [L for L in layers if L not in resid_layers]
    assert not missing, f"--layers {missing} were not captured (captured: {resid_layers})"
    grid = {L: np.asarray(resid[:, resid_layers.index(L)], dtype=np.float64).reshape(I, A, -1) for L in layers}

    # Gold first tokens from the stored labels (flags labels are cased as the model writes them).
    first_tok, correct = {}, {}
    for n, r in enumerate(index):
        cell = (n // A, n % A)
        first_tok[cell] = first_content_token(m.tokenizer, gold_ids(m.tokenizer, items[r["item"]][r["attribute"]]))
        correct[cell] = r["correct"]
    rows = sample_compose_rows(keys, attributes, first_tok, correct, n_rows, seed)
    print(f"[compose] {len(rows)} informative rows (requested {n_rows}); layers {layers}")
    if not rows:
        raise RuntimeError("no informative rows -- is the clean model answering the grid correctly?")

    rng = np.random.default_rng(seed)
    out_rows = []
    fh = open(os.path.join(out_dir, "compose_rows.jsonl"), "w")
    try:
        for L in layers:
            R = grid[L]
            for arm in ARMS:
                for s in range(0, len(rows), batch_size):
                    chunk = rows[s:s + batch_size]
                    built = [m.image_prompt(entity_dir, items[keys[c]], attributes[x]) for c, x, _, _ in chunk]
                    ids = torch.stack([b[0] for b in built])
                    mask = torch.ones_like(ids)
                    extra = cat_extra([b[1] for b in built])
                    handles = []
                    if arm != "clean":
                        vals = np.stack([arm_vector(arm, R, b, a, c, x, rng) for c, x, b, a in chunk])
                        pos = torch.full((len(chunk), 1), ids.shape[1] - 1, dtype=torch.long)
                        site, layer, fn = residual_patch(
                            L, pos, torch.tensor(vals[:, None], dtype=torch.float32, device=m.model.device))
                        handles = site.register(m.adapter, m.model, m.layers, layer, fn)
                    try:
                        gen = m.generate(ids, mask, extra, max_new_tokens)
                    finally:
                        for h in handles:
                            h.remove()
                    for (c, x, b, a), g, text in zip(chunk, gen, m.decode(gen)):
                        cells = {"B.A": (b, a), "C.X": (c, x), "B.X": (b, x), "C.A": (c, a)}
                        ft = first_content_token(m.tokenizer, g.tolist())
                        first_hit = next((t for t, cell in cells.items() if first_tok[cell] == ft), "other")
                        text_hit = next((t for t, (i, j) in cells.items()
                                         if contains_label(text, str(items[keys[i]][attributes[j]]))), "other")
                        r = {"layer": L, "arm": arm, "C": keys[c], "X": attributes[x], "B": keys[b],
                             "A": attributes[a], "generated": text, "first_hit": first_hit, "text_hit": text_hit}
                        out_rows.append(r)
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  layer {L}: done", flush=True)
    finally:
        fh.close()
    return summarize_compose(out_rows)


# ---------------------------------------------------------------------------
# CPU analysis
# ---------------------------------------------------------------------------

def analyze_neurons(out_dir, items, keys, attributes, meta, top_k, heatmap_block, n_perm, seed):
    I, A = len(keys), len(attributes)
    resid = np.load(os.path.join(out_dir, "resid.npy"), mmap_mode="r")
    neur = np.load(os.path.join(out_dir, "mlp_hidden.npy"), mmap_mode="r")
    mout = np.load(os.path.join(out_dir, "mlp_output.npy"), mmap_mode="r")
    report = {"resid_shares": {}, "mlp_hidden_shares": {}, "mlp_output_shares": {}, "neuron_classes": {},
              "value_test": {}}
    for j, L in enumerate(meta["resid_layers"]):
        report["resid_shares"][str(L)] = shares(two_way(np.asarray(resid[:, j]).reshape(I, A, -1)))
    heat = None
    for j, b in enumerate(meta["mlp_blocks"]):
        Y = np.asarray(neur[:, j], dtype=np.float64).reshape(I, A, -1)
        ss = two_way(Y)
        report["mlp_hidden_shares"][str(b)] = shares(ss)
        report["mlp_output_shares"][str(b)] = shares(two_way(np.asarray(mout[:, j]).reshape(I, A, -1)))
        top = np.argsort(-ss["total"])[:top_k]
        parts = np.stack([ss["item"][top], ss["attr"][top], ss["inter"][top]])
        report["neuron_classes"][str(b)] = dict(zip(("item", "attr", "inter"),
                                                    np.bincount(parts.argmax(0), minlength=3).tolist()))
        top_inter = np.argsort(-ss["inter"])[:top_k]
        report["value_test"][str(b)] = {}
        for attr in SHARED_VALUE_ATTRS:
            if attr in attributes:
                report["value_test"][str(b)][attr] = value_test(
                    Y, attribute_values(items, keys, attr), attributes.index(attr), top_inter, n_perm, seed)
        if b == heatmap_block:
            sel = top_inter[:40]
            Z = Y[:, :, sel]
            Z = (Z - Z.mean((0, 1))) / (Z.std((0, 1)) + 1e-8)
            order = sorted(range(I), key=lambda i: (str(items[keys[i]].get("language", "")), keys[i]))
            heat = {"z": Z[order].astype(np.float32), "neurons": sel, "items": np.array([keys[i] for i in order]),
                    "languages": np.array([str(items[keys[i]].get("language", "")) for i in order]),
                    "attributes": np.array(attributes), "block": b}
    if heat is not None:
        np.savez(os.path.join(out_dir, "heatmap.npz"), **heat)
    return report


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["capture", "compose", "neurons", "all"], default="all")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    ap.add_argument("--limit_items", type=int, default=0, help="0 = all 84 flags")
    ap.add_argument("--resid_layers", type=int, nargs="+", default=list(range(18, 29)),
                    help="Residual AFTER block L-1 (register_patch_hook's convention); 28 = last block's "
                         "pre-norm output.")
    ap.add_argument("--mlp_blocks", type=int, nargs="+", default=[21, 22, 23, 24, 25, 26, 27])
    ap.add_argument("--layers", type=int, nargs="+", default=[22, 23, 24, 25, 26, 27],
                    help="compose: layers at which the constructed residual is installed")
    ap.add_argument("--n_rows", type=int, default=384)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--heatmap_block", type=int, default=25)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "mlp_lookup", "flags"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    entity_dir, items, gt_sha = load_entity(args.vade_root, "flags")
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    attributes = list(args.attributes)
    phases = ["capture", "compose", "neurons"] if args.phase == "all" else [args.phase]
    n_cells = len(keys) * len(attributes)
    print(f"[mlp_lookup] {len(keys)} flags x {len(attributes)} attributes = {n_cells} cells; phases {phases}; "
          f"-> {args.out_dir}")
    if args.dry_run:
        mb = n_cells * (len(args.resid_layers) * 3584 + len(args.mlp_blocks) * (18944 + 3584)) * 4 / 1e6
        assert all(L in args.resid_layers for L in args.layers), "--layers must be a subset of --resid_layers"
        for a in attributes:
            assert all(a in items[k] for k in keys), f"grid incomplete for {a}"
        print(f"Grid valid. capture writes ~{mb:.0f} MB; compose = {len(args.layers)} layers x {len(ARMS)} arms "
              f"x <= {args.n_rows} rows.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    meta = {"model_id": args.model_id, "items": keys, "attributes": attributes,
            "resid_layers": args.resid_layers, "mlp_blocks": args.mlp_blocks, "ground_truth_sha256": gt_sha,
            "prompts": "controlled (attribute_switch_sweep), prefill 'Answer:'", "position": "last prompt column"}
    if "capture" not in phases:
        meta = json.load(open(meta_path))
        assert meta["items"] == keys and meta["attributes"] == attributes, \
            "capture in --out_dir was made with other items/attributes; pass the same --limit_items/--attributes"

    report_path = os.path.join(args.out_dir, "summary.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    report.update(config=vars(args), meta=meta)

    m = None
    if "capture" in phases or "compose" in phases:
        from methods.circuit_common import Model
        m = Model(args.model_id, device=args.device, attn_impl=args.attn_impl, vade_root=args.vade_root)
    if "capture" in phases:
        json.dump(meta, open(meta_path, "w"), indent=2)
        index = run_capture(m, entity_dir, items, keys, attributes, args.resid_layers, args.mlp_blocks,
                            args.batch_size, args.max_new_tokens, args.out_dir)
        report["clean_accuracy"] = {a: float(np.mean([r["correct"] for r in index if r["attribute"] == a]))
                                    for a in attributes}
    if "compose" in phases:
        report["compose"] = run_compose(m, entity_dir, items, keys, attributes, meta, args.layers, args.n_rows,
                                        args.batch_size, args.max_new_tokens, args.seed, args.out_dir)
    if "neurons" in phases:
        report["neurons"] = analyze_neurons(args.out_dir, items, keys, attributes, meta, args.top_k,
                                            args.heatmap_block, args.n_perm, args.seed)
    json.dump(report, open(report_path, "w"), indent=2)

    if "compose" in report:
        print("\ncompose (first answer token; rows with four distinct golds)")
        for L, arms in report["compose"].items():
            for arm, s in arms.items():
                print(f"  L{L:<3} {arm:<14} n={s['n']:>4}  expect {s['expect']}  " +
                      "  ".join(f"{t} {s['first_' + t]:6.1%}" for t in TARGETS) + f"  other {s['first_other']:6.1%}")
    if "neurons" in report:
        nr = report["neurons"]
        print("\nresidual variance shares (item / attr / inter)")
        for L, s in nr["resid_shares"].items():
            print(f"  L{L:<3} {s['item']:.3f} {s['attr']:.3f} {s['inter']:.3f}")
        print("MLP neurons / MLP output: variance shares (item / attr / inter); top-k neuron classes")
        for b, s in nr["mlp_hidden_shares"].items():
            o = nr["mlp_output_shares"][b]
            print(f"  block {b:<3} {s['item']:.3f} {s['attr']:.3f} {s['inter']:.3f}  |  {o['item']:.3f} {o['attr']:.3f} "
                  f"{o['inter']:.3f}   {nr['neuron_classes'][b]}")
        print("value test (eta^2 asked - other, top interaction neurons; p from permutation)")
        for b, d in nr["value_test"].items():
            for attr, v in d.items():
                if v:
                    print(f"  block {b:<3} {attr:<13} diff {v['mean_diff']:+.3f}  null95 {v['null_p95']:+.3f}  "
                          f"p {v['p_value']:.3f}  ({v['n_items']} items, {v['n_values']} values)")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
