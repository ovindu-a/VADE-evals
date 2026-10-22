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
# Structured nulls (follow-up 3): each keeps something about the additive vector and breaks one thing.
CONTROL_ARMS = ("no_entity", "scrambled_entity", "entity_other_layer")
# Cross-modal (follow-up 4): the entity component comes from runs where the country is NAMED in text.
XM_ARMS = ("xm_swap_entity", "xm_additive")
ALL_ARMS = ARMS + CONTROL_ARMS + XM_ARMS
EXPECT = {"clean": "C.X", "actual": "B.A", "additive": "B.A", "swap_entity": "B.X",
          "swap_question": "C.A", "random_offset": "C.X",
          # no entity component at all: does late attention refill C from the image (-> C.A)?
          "no_entity": "C.A",
          # B's entity component with its coordinates permuted (norm and value distribution kept)
          "scrambled_entity": "B.A",
          # B's entity component taken from layer L-2 instead of L, question component from L
          "entity_other_layer": "B.A",
          "xm_swap_entity": "B.X", "xm_additive": "B.A"}
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


def rest_mean(R, i, a):
    """Mean over every cell (i' != i, a' != a)."""
    return np.delete(np.delete(R, i, axis=0), a, axis=1).mean((0, 1))


def entity_component(R, i, a):
    """Item i's deviation from the grid, estimated without attribute a's column (so a target
    cell (i, a) is never used): rowmean_{-a}(i) - mean over cells (i' != i, a' != a)."""
    return row_mean_excl(R, i, a) - rest_mean(R, i, a)


def entity_dev(R, i, a):
    """Item i's deviation from the SAME reference for every item: rowmean_{-a}(i) minus the mean
    of rowmean_{-a} over all items. Differences of two items' devs are exact on an additive grid
    (entity_component's leave-out reference differs per item and would scale them by I/(I-1)), so
    the swap arms use this; column a is still never read."""
    keep = [x for x in range(R.shape[1]) if x != a]
    return R[i, keep].mean(0) - R[:, keep].mean((0, 1))


def scramble(v, seed=0):
    """A fixed coordinate permutation: keeps the norm and every coordinate's magnitude
    (outlier dimensions included) but destroys the direction."""
    return v[np.random.default_rng(seed).permutation(v.shape[-1])]


def arm_vector(arm, R, b_i, a_i, c_i, x_i, rng=None, R_other=None, T=None):
    """The residual installed at the base (C, X) column for `arm`. R [I, A, H] at one layer.
    R_other: the same grid at another layer (entity_other_layer). T: the TEXT-named grid at the
    same layer, items and attributes in the same order (xm_* arms)."""
    base = R[c_i, x_i]
    if arm == "no_entity":
        return col_mean_excl(R, b_i, a_i)                       # mu + q_A, no item component
    if arm == "scrambled_entity":
        return col_mean_excl(R, b_i, a_i) + scramble(entity_component(R, b_i, a_i))
    if arm == "entity_other_layer":
        assert R_other is not None, "entity_other_layer needs the grid at another layer"
        return col_mean_excl(R, b_i, a_i) + entity_component(R_other, b_i, a_i)
    if arm == "xm_additive":
        assert T is not None, "xm arms need the text-named grid"
        return col_mean_excl(R, b_i, a_i) + entity_component(T, b_i, a_i)
    if arm == "xm_swap_entity":
        assert T is not None, "xm arms need the text-named grid"
        return base + entity_dev(T, b_i, x_i) - entity_dev(R, c_i, x_i)
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
    for (arm, layer), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], ALL_ARMS.index(kv[0][0]))):
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


def load_grid(out_dir, name, I, A, layers, resid_layers):
    arr = np.load(os.path.join(out_dir, name), mmap_mode="r")
    return {L: np.asarray(arr[:, resid_layers.index(L)], dtype=np.float64).reshape(I, A, -1) for L in layers}


def run_compose(m, entity_dir, items, keys, attributes, meta, layers, n_rows, batch_size,
                max_new_tokens, seed, out_dir, arms=ARMS, other_layer_offset=-2):
    """Appends to compose_rows.jsonl and skips every (layer, arm) already in it, so new arms can
    be added to a finished run without redoing the old ones. Rows are the same seeded sample
    whatever the arms (it depends only on the IMAGE grid's clean answers), so arms run at
    different times are compared on identical rows."""
    import torch
    from methods.circuit_common import cat_extra, first_content_token, gold_ids, residual_patch
    from methods.common.text_match import contains_label

    index = [json.loads(l) for l in open(os.path.join(out_dir, "index.jsonl"))]
    I, A = len(keys), len(attributes)
    resid_layers = meta["resid_layers"]
    missing = [L for L in layers if L not in resid_layers]
    assert not missing, f"--layers {missing} were not captured (captured: {resid_layers})"
    grid = load_grid(out_dir, "resid.npy", I, A, layers, resid_layers)
    other = {}
    if "entity_other_layer" in arms:
        need = sorted({L + other_layer_offset for L in layers})
        bad = [L for L in need if L not in resid_layers]
        assert not bad, f"entity_other_layer needs layers {bad}, which were not captured"
        other = load_grid(out_dir, "resid.npy", I, A, need, resid_layers)
    text, text_ok = {}, None
    if any(a in XM_ARMS for a in arms):
        assert os.path.exists(os.path.join(out_dir, "resid_text.npy")), "xm arms need --phase text_capture first"
        text = load_grid(out_dir, "resid_text.npy", I, A, layers, resid_layers)
        tindex = [json.loads(l) for l in open(os.path.join(out_dir, "index_text.jsonl"))]
        text_ok = {(n // A, n % A): r["correct"] for n, r in enumerate(tindex)}

    # Gold first tokens from the stored labels (flags labels are cased as the model writes them).
    first_tok, correct = {}, {}
    for n, r in enumerate(index):
        cell = (n // A, n % A)
        first_tok[cell] = first_content_token(m.tokenizer, gold_ids(m.tokenizer, items[r["item"]][r["attribute"]]))
        correct[cell] = r["correct"]
    rows = sample_compose_rows(keys, attributes, first_tok, correct, n_rows, seed)
    print(f"[compose] {len(rows)} informative rows (requested {n_rows}); layers {layers}; arms {list(arms)}")
    if not rows:
        raise RuntimeError("no informative rows -- is the clean model answering the grid correctly?")

    path = os.path.join(out_dir, "compose_rows.jsonl")
    out_rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    done = {(r["layer"], r["arm"]) for r in out_rows}
    rng = np.random.default_rng(seed)
    with open(path, "a") as fh:
        for L in layers:
            R = grid[L]
            for arm in arms:
                if (L, arm) in done:
                    print(f"  layer {L} {arm}: already in {os.path.basename(path)}, skipped")
                    continue
                for s in range(0, len(rows), batch_size):
                    chunk = rows[s:s + batch_size]
                    built = [m.image_prompt(entity_dir, items[keys[c]], attributes[x]) for c, x, _, _ in chunk]
                    ids = torch.stack([b[0] for b in built])
                    mask = torch.ones_like(ids)
                    extra = cat_extra([b[1] for b in built])
                    handles = []
                    if arm != "clean":
                        vals = np.stack([arm_vector(arm, R, b, a, c, x, rng, R_other=other.get(L + other_layer_offset),
                                                    T=text.get(L)) for c, x, b, a in chunk])
                        pos = torch.full((len(chunk), 1), ids.shape[1] - 1, dtype=torch.long)
                        site, layer, fn = residual_patch(
                            L, pos, torch.tensor(vals[:, None], dtype=torch.float32, device=m.model.device))
                        handles = site.register(m.adapter, m.model, m.layers, layer, fn)
                    try:
                        gen = m.generate(ids, mask, extra, max_new_tokens)
                    finally:
                        for h in handles:
                            h.remove()
                    for (c, x, b, a), g, txt in zip(chunk, gen, m.decode(gen)):
                        cells = {"B.A": (b, a), "C.X": (c, x), "B.X": (b, x), "C.A": (c, a)}
                        ft = first_content_token(m.tokenizer, g.tolist())
                        first_hit = next((t for t, cell in cells.items() if first_tok[cell] == ft), "other")
                        text_hit = next((t for t, (i, j) in cells.items()
                                         if contains_label(txt, str(items[keys[i]][attributes[j]]))), "other")
                        r = {"layer": L, "arm": arm, "C": keys[c], "X": attributes[x], "B": keys[b],
                             "A": attributes[a], "generated": txt, "first_hit": first_hit, "text_hit": text_hit}
                        if text_ok is not None:
                            r["text_ok"] = bool(text_ok[(b, a)] and text_ok[(b, x)])
                        out_rows.append(r)
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  layer {L} {arm}: done", flush=True)
    return summarize_compose(out_rows)


def run_text_capture(m, items, keys, attributes, resid_layers, batch_size, max_new_tokens, out_dir):
    """The crossed grid with the country NAMED in text (circuit_common.text_prompt: same
    instruction, same question and 'Answer:' prefill, no image). Residual at the last prompt
    column at every --resid_layers layer, plus a clean generation per cell for eligibility."""
    import torch
    from methods.circuit_common import left_pack
    from methods.common.text_match import contains_label

    cells = grid_cells(keys, attributes)
    resid = np.lib.format.open_memmap(os.path.join(out_dir, "resid_text.npy"), mode="w+", dtype=np.float32,
                                      shape=(len(cells), len(resid_layers), m.hidden))
    index = []
    for s in range(0, len(cells), batch_size):
        chunk = cells[s:s + batch_size]
        ids, mask = left_pack([m.text_prompt(items[k]["name"], a)[0] for k, a in chunk], m.pad_id)
        pos = torch.full((len(chunk), 1), ids.shape[1] - 1, dtype=torch.long)   # left-padded: last column
        sink = {L: [] for L in resid_layers}
        handles = [block_output_hook(m.layers, L - 1, pos, sink[L]) for L in resid_layers]
        try:
            m.forward(ids, mask, {})
        finally:
            for h in handles:
                h.remove()
        for j, L in enumerate(resid_layers):
            assert len(sink[L]) == 1
            resid[s:s + len(chunk), j] = sink[L][0][:, 0].float().cpu().numpy()
        gen = m.generate(ids, mask, {}, max_new_tokens)
        for (k, a), text in zip(chunk, m.decode(gen)):
            index.append({"item": k, "attribute": a, "clean": text,
                          "correct": bool(contains_label(text, str(items[k][a])))})
        print(f"  text captured {s + len(chunk)}/{len(cells)}", flush=True)
    resid.flush()
    with open(os.path.join(out_dir, "index_text.jsonl"), "w") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(r["correct"] for r in index) / len(index)
    print(f"clean accuracy over the TEXT grid: {acc:.1%}")
    return {a: float(np.mean([r["correct"] for r in index if r["attribute"] == a])) for a in attributes}


# ---------------------------------------------------------------------------
# CPU analysis
# ---------------------------------------------------------------------------

def ridge_loo_acc(X, y, lams=(1e-1, 1e0, 1e1, 1e2, 1e3)):
    """Leave-one-item-out accuracy of a one-vs-rest linear ridge classifier, exact and closed
    form: for kernel ridge with K = X X^T, the LOO prediction is y - (I - H)^-1-diag correction,
    H = K (K + lam I)^-1. X [n, d] (standardized by the caller), y [n] labels. -> best accuracy over
    `lams` (the permutation null must take the same max)."""
    X = np.asarray(X, dtype=np.float64)
    labels = sorted(set(y))
    Y = np.array([[1.0 if v == c else 0.0 for c in labels] for v in y])
    Y = Y - Y.mean(0)
    K = X @ X.T
    n = len(y)
    best = 0.0
    for lam in lams:
        H = K @ np.linalg.solve(K + lam * np.trace(K) / n * np.eye(n), np.eye(n))
        pred = H @ Y
        loo = (pred - np.diag(H)[:, None] * Y) / (1 - np.diag(H))[:, None]
        acc = float(np.mean([labels[j] == v for j, v in zip(loo.argmax(1), y)]))
        best = max(best, acc)
    return best


def standardize(X):
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def region_baseline(y, keys, region, fallback):
    """LOO accuracy of predicting an item's value as the majority value among the OTHER items in
    its region (falling back to the coarser region, then to the global majority)."""
    hits = 0
    for i, k in enumerate(keys):
        pred = None
        for reg in (region, fallback):
            pool = [y[j] for j, k2 in enumerate(keys) if j != i and reg.get(k2) == reg.get(k)]
            if pool:
                pred = Counter(pool).most_common(1)[0][0]
                break
        if pred is None:
            pred = Counter(y[:i] + y[i + 1:]).most_common(1)[0][0]
        hits += pred == y[i]
    return hits / len(y)


def bundle_probes(out_dir, items, keys, attributes, meta, n_perm=100, seed=0, grid_file="resid.npy"):
    """Is each attribute's VALUE already linearly readable from the entity component built
    WITHOUT the question that asks for it? For attribute a, features = item i's residual
    averaged over the OTHER questions (rowmean_{-a}(i)); labels = i's value of a (language, EUR,
    calling zone; continent as a reference for how regional the code is). Leave-one-country-out
    ridge accuracy at every captured layer, against a permutation null (labels shuffled, same max
    over the ridge penalty) and the REGION BASELINE: majority value among the other countries of
    the same UN subregion. A bundle predicts accuracy above the region baseline already at L24,
    before MLPs 24-26; a lookup predicts it appears only after them."""
    from methods.cpu_followups import CONTINENT, SUBREGION
    I, A = len(keys), len(attributes)
    grids = load_grid(out_dir, grid_file, I, A, meta["resid_layers"], meta["resid_layers"])
    targets = {}
    for attr in SHARED_VALUE_ATTRS:
        if attr in attributes:
            v = attribute_values(items, keys, attr)
            kept, labels = value_groups(v) if attr == "language" else (list(range(I)), v)
            targets[attr] = (attributes.index(attr), kept, labels)
    targets["continent (reference)"] = (None, list(range(I)), [CONTINENT.get(k, "?") for k in keys])
    rng = random.Random(seed)
    out = {}
    for name, (a_idx, kept, labels) in targets.items():
        kk = [keys[i] for i in kept]
        res = {"n_items": len(kept), "n_values": len(set(labels)),
               "majority": Counter(labels).most_common(1)[0][1] / len(labels),
               "region_baseline": region_baseline(labels, kk, SUBREGION, CONTINENT), "by_layer": {}}
        nulls = None
        for L, R in grids.items():
            feats = np.stack([row_mean_excl(R, i, a_idx) if a_idx is not None else R[i].mean(0) for i in kept])
            Xs = standardize(feats)
            acc = ridge_loo_acc(Xs, labels)
            if nulls is None:            # the null is layer-independent in distribution; draw it once
                nulls = []
                for _ in range(n_perm):
                    lab = labels[:]
                    rng.shuffle(lab)
                    nulls.append(ridge_loo_acc(Xs, lab))
            res["by_layer"][str(L)] = acc
        res["null_mean"], res["null_p95"] = float(np.mean(nulls)), float(np.percentile(nulls, 95))
        out[name] = res
    return out


def bundle_lens(m, out_dir, items, keys, attributes, meta):
    """Direct unembedding of three vectors per (item, attribute) cell and layer: the real
    residual R[i, a], the additive estimate, and the QUESTION-FREE entity vector rowmean_{-a}(i)
    (the residual averaged over the other questions). Score = rank of item i's own value
    (first content token) among the distinct first tokens of every item's value for a; 1 = top.
    The null is the same rank for a random other item's value. Unembedding direction only (the
    final norm's scalar does not change a rank); calling_code is skipped (its first tokens are
    single digits shared by many countries)."""
    import torch
    from methods.circuit_common import first_content_token, gold_ids
    I, A = len(keys), len(attributes)
    grids = load_grid(out_dir, "resid.npy", I, A, meta["resid_layers"], meta["resid_layers"])
    out = {}
    rng = random.Random(0)
    for attr in attributes:
        if attr == "calling_code":
            continue
        a = attributes.index(attr)
        tok = [first_content_token(m.tokenizer, gold_ids(m.tokenizer, str(items[k][attr]))) for k in keys]
        cand = sorted(set(t for t in tok if t is not None))
        with torch.no_grad():
            W = m.adapter.logit_direction(m.model, torch.tensor(cand, device=m.model.device)).float().cpu().numpy()
        col = {t: j for j, t in enumerate(cand)}
        others = [rng.choice([j for j in range(I) if tok[j] != tok[i] and tok[j] is not None]) for i in range(I)]
        res = {"n_candidates": len(cand), "chance_rank": (len(cand) + 1) / 2, "by_layer": {}}
        for L, R in grids.items():
            vecs = {"actual": np.stack([R[i, a] for i in range(I)]),
                    "additive": np.stack([additive_loo(R, i, a) for i in range(I)]),
                    "question_free": np.stack([row_mean_excl(R, i, a) for i in range(I)])}
            row = {}
            for name, V in vecs.items():
                S = V @ W.T                                            # [I, n_cand]
                own = [1 + int((S[i] > S[i, col[tok[i]]]).sum()) for i in range(I) if tok[i] is not None]
                oth = [1 + int((S[i] > S[i, col[tok[others[i]]]]).sum()) for i in range(I) if tok[i] is not None]
                row[name] = {"own_rank": float(np.mean(own)), "other_rank": float(np.mean(oth)),
                             "own_top1": float(np.mean([r == 1 for r in own]))}
            res["by_layer"][str(L)] = row
        out[attr] = res
    return out


def rank_rows(S, own_cols):
    """S [n, n_cand] scores, own_cols [n] -> rank (1 = top) of each row's own candidate."""
    own = S[np.arange(len(own_cols)), own_cols]
    return 1 + (S > own[:, None]).sum(1)


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_stats(m, out_dir, items, keys, attributes, meta, layers=None, n_boot=2000):
    """Per RUN (not per averaged vector): for every item i, layer L and attribute a, the
    direct-unembedding rank of i's own value of a among a's candidate values (distinct first
    content tokens), in the run that ASKED a and in each of the three runs that asked another
    question. Reports mean / median rank, top-1 / top-5 rates, mean reciprocal rank, the paired
    per-item difference (asked - mean over not-asked) with a country-bootstrap CI, and an
    asked-question matrix (rank of a's value when question q was asked). calling_code is skipped
    (its first content tokens are single digits shared across countries)."""
    import torch
    from methods.circuit_common import first_content_token, gold_ids
    I, A = len(keys), len(attributes)
    layers = layers or meta["resid_layers"]
    grids = load_grid(out_dir, "resid.npy", I, A, layers, meta["resid_layers"])
    out = {}
    for attr in attributes:
        if attr == "calling_code":
            continue
        a = attributes.index(attr)
        tok = [first_content_token(m.tokenizer, gold_ids(m.tokenizer, str(items[k][attr]))) for k in keys]
        ok = [i for i in range(I) if tok[i] is not None]
        cand = sorted(set(tok[i] for i in ok))
        col = {t: j for j, t in enumerate(cand)}
        with torch.no_grad():
            W = m.adapter.logit_direction(m.model, torch.tensor(cand, device=m.model.device)).float().cpu().numpy()
        own = np.array([col[tok[i]] for i in ok])
        res = {"n_items": len(ok), "n_candidates": len(cand), "chance_mean_rank": (len(cand) + 1) / 2,
               "by_layer": {}}
        for L in layers:
            R = grids[L]
            ranks = np.stack([rank_rows(R[ok, q] @ W.T, own) for q in range(A)], axis=1)   # [items, asked q]
            asked = ranks[:, a]
            other_q = [q for q in range(A) if q != a]
            notasked = ranks[:, other_q]                                                       # [items, 3]
            diff = asked - notasked.mean(1)

            def summ(r):
                r = np.asarray(r, dtype=np.float64).ravel()
                return {"mean_rank": float(r.mean()), "median_rank": float(np.median(r)),
                        "top1": float((r == 1).mean()), "top5": float((r <= 5).mean()), "mrr": float((1 / r).mean())}
            res["by_layer"][str(L)] = {
                "asked": {**summ(asked), "mean_rank_ci": boot_ci(asked, n_boot)},
                "not_asked": {**summ(notasked), "mean_rank_ci": boot_ci(notasked.mean(1), n_boot)},
                "paired_diff_asked_minus_notasked": {"mean": float(diff.mean()), "ci": boot_ci(diff, n_boot),
                                                     "frac_items_asked_better": float((diff < 0).mean()),
                                                     "frac_items_notasked_better": float((diff > 0).mean())},
                "by_asked_question": {attributes[q]: summ(ranks[:, q]) for q in range(A)}}
        out[attr] = res
    return out


def compose_report(out_dir, items, attributes):
    """CPU, off compose_rows.jsonl: (1) each arm's expected-target first-token rate per target
    attribute A; (2) what the 'other' first tokens actually said, typed against the arm's expected
    cell with gate_knockout.classify_fact (the full text right after all / refusal / this item's
    other attribute / another country's value / other); (3) for the cross-modal arms, the same
    rates on the rows where the TEXT model also knows B's A and X, next to their image twins."""
    from methods.gate_knockout import classify_fact
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "compose_rows.jsonl"))]
    by_attr = defaultdict(lambda: defaultdict(list))
    other_types = defaultdict(Counter)
    for r in rows:
        exp = EXPECT[r["arm"]]
        hit = r["first_hit"] == exp
        by_attr[(r["layer"], r["arm"])][r["A"]].append(hit)
        if r["first_hit"] == "other":
            item = {"B": r["B"], "C": r["C"]}[exp[0]]
            attr = {"A": r["A"], "X": r["X"]}[exp[2]]
            other_types[(r["layer"], r["arm"])][classify_fact(r["generated"], item, attr, items, attributes)] += 1
    out = {"by_target_attribute": {}, "other_breakdown": {}, "text_ok_subset": {}}
    for (L, arm), d in sorted(by_attr.items()):
        out["by_target_attribute"].setdefault(str(L), {})[arm] = {a: {"n": len(v), "hit": float(np.mean(v))}
                                                                    for a, v in sorted(d.items())}
    for (L, arm), c in sorted(other_types.items()):
        n = sum(c.values())
        out["other_breakdown"].setdefault(str(L), {})[arm] = {"n_other": n, **{k: v / n for k, v in c.items()}}
    sub = [r for r in rows if r.get("text_ok")]
    if sub:
        keyset = {(r["C"], r["X"], r["B"], r["A"]) for r in sub}
        twin = [r for r in rows if (r["C"], r["X"], r["B"], r["A"]) in keyset]
        out["text_ok_subset"] = summarize_compose(twin)
    return out



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
    ap.add_argument("--phase", nargs="+", default=["all"],
                    choices=["capture", "compose", "neurons", "text_capture", "bundle", "report", "rankstats", "all",
                             "followups"],
                    help="all = capture compose neurons. followups = text_capture compose bundle report, with "
                         "--arms defaulting to the control and cross-modal arms (run after `all`).")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=None,
                    help="compose arms. Default: the six original arms (or, with --phase followups, the "
                         "control + cross-modal arms). Arms already in compose_rows.jsonl are skipped.")
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
    phases = []
    for ph in args.phase:
        phases += {"all": ["capture", "compose", "neurons"],
                   "followups": ["text_capture", "compose", "bundle", "report", "rankstats"]}.get(ph, [ph])
    phases = list(dict.fromkeys(phases))
    arms = args.arms or (list(CONTROL_ARMS + XM_ARMS) if "followups" in args.phase else list(ARMS))
    n_cells = len(keys) * len(attributes)
    print(f"[mlp_lookup] {len(keys)} flags x {len(attributes)} attributes = {n_cells} cells; phases {phases}; "
          f"-> {args.out_dir}")
    if args.dry_run:
        mb = n_cells * (len(args.resid_layers) * 3584 + len(args.mlp_blocks) * (18944 + 3584)) * 4 / 1e6
        assert all(L in args.resid_layers for L in args.layers), "--layers must be a subset of --resid_layers"
        for a in attributes:
            assert all(a in items[k] for k in keys), f"grid incomplete for {a}"
        print(f"Grid valid. capture writes ~{mb:.0f} MB; compose = {len(args.layers)} layers x {len(arms)} arms "
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
    if {"capture", "compose", "text_capture", "bundle", "rankstats"} & set(phases):
        from methods.circuit_common import Model
        m = Model(args.model_id, device=args.device, attn_impl=args.attn_impl, vade_root=args.vade_root)
    if "capture" in phases:
        json.dump(meta, open(meta_path, "w"), indent=2)
        index = run_capture(m, entity_dir, items, keys, attributes, args.resid_layers, args.mlp_blocks,
                            args.batch_size, args.max_new_tokens, args.out_dir)
        report["clean_accuracy"] = {a: float(np.mean([r["correct"] for r in index if r["attribute"] == a]))
                                    for a in attributes}
    if "text_capture" in phases:
        report["clean_accuracy_text"] = run_text_capture(m, items, keys, attributes, meta["resid_layers"],
                                                         args.batch_size, args.max_new_tokens, args.out_dir)
    if "compose" in phases:
        report["compose"] = run_compose(m, entity_dir, items, keys, attributes, meta, args.layers, args.n_rows,
                                        args.batch_size, args.max_new_tokens, args.seed, args.out_dir, arms=arms)
    if "bundle" in phases:
        report["bundle"] = {"probes": bundle_probes(args.out_dir, items, keys, attributes, meta, seed=args.seed),
                            "lens": bundle_lens(m, args.out_dir, items, keys, attributes, meta)}
        if os.path.exists(os.path.join(args.out_dir, "resid_text.npy")):
            report["bundle"]["probes_text"] = bundle_probes(args.out_dir, items, keys, attributes, meta,
                                                            seed=args.seed, grid_file="resid_text.npy")
    if "report" in phases:
        report["compose_report"] = compose_report(args.out_dir, items, attributes)
    if "rankstats" in phases:
        report["rank_stats"] = rank_stats(m, args.out_dir, items, keys, attributes, meta)
    if "neurons" in phases:
        report["neurons"] = analyze_neurons(args.out_dir, items, keys, attributes, meta, args.top_k,
                                            args.heatmap_block, args.n_perm, args.seed)
    json.dump(report, open(report_path, "w"), indent=2)

    if "compose" in report:
        print("\ncompose (first answer token; rows with four distinct golds)")
        for L, arms in report["compose"].items():
            for arm, s in arms.items():
                print(f"  L{L:<3} {arm:<18} n={s['n']:>4}  expect {s['expect']}  " +
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
    if "bundle" in report:
        for title, key in (("IMAGE", "probes"), ("TEXT", "probes_text")):
            if key not in report["bundle"]:
                continue
            print(f"\nbundle probes ({title}): LOO-country ridge accuracy of the value from rowmean over the OTHER questions")
            for attr, r in report["bundle"][key].items():
                print(f"  {attr:<22} n={r['n_items']:>3} values={r['n_values']:>2} majority {r['majority']:.2f} "
                      f"region {r['region_baseline']:.2f} null95 {r['null_p95']:.2f} | " +
                      " ".join(f"L{L}:{v:.2f}" for L, v in r["by_layer"].items()))
        print("\nbundle lens: mean rank of the item's own value among candidates (own / random other), chance in ()")
        for attr, r in report["bundle"]["lens"].items():
            print(f"  {attr} ({r['n_candidates']} candidates, chance {r['chance_rank']:.1f})")
            for L, row in r["by_layer"].items():
                print(f"    L{L:<3} " + "  ".join(f"{k} {v['own_rank']:5.1f}/{v['other_rank']:5.1f} top1 {v['own_top1']:.2f}"
                                               for k, v in row.items()))
    if "rank_stats" in report and "rankstats" in phases:
        print("\nrank of the item's own value token (1 = top), per run: ASKED vs NOT ASKED")
        for attr, r in report["rank_stats"].items():
            print(f"  {attr}: {r['n_items']} items, {r['n_candidates']} candidates, chance mean rank {r['chance_mean_rank']:.1f}")
            print("    L    asked: mean [95% CI]   median top1  top5   | not asked: mean [95% CI]   median top1  top5   "
                  "| paired diff [95% CI]  items asked better / not-asked better")
            for L, d in r["by_layer"].items():
                a_, n_, p_ = d["asked"], d["not_asked"], d["paired_diff_asked_minus_notasked"]
                print(f"    L{L:<3} {a_['mean_rank']:6.1f} [{a_['mean_rank_ci'][0]:5.1f},{a_['mean_rank_ci'][1]:5.1f}] "
                      f"{a_['median_rank']:5.1f} {a_['top1']:5.2f} {a_['top5']:5.2f} | "
                      f"{n_['mean_rank']:6.1f} [{n_['mean_rank_ci'][0]:5.1f},{n_['mean_rank_ci'][1]:5.1f}] "
                      f"{n_['median_rank']:5.1f} {n_['top1']:5.2f} {n_['top5']:5.2f} | "
                      f"{p_['mean']:+6.1f} [{p_['ci'][0]:+5.1f},{p_['ci'][1]:+5.1f}]  "
                      f"{p_['frac_items_asked_better']:.2f} / {p_['frac_items_notasked_better']:.2f}")
            print("    by asked question (mean rank / top1):")
            for L in ("22", "24", "25", "26", "28"):
                if L in r["by_layer"]:
                    bq = r["by_layer"][L]["by_asked_question"]
                    print(f"      L{L:<3} " + "  ".join(f"{q}: {v['mean_rank']:5.1f}/{v['top1']:.2f}" for q, v in bq.items()))
    if "compose_report" in report:
        cr = report["compose_report"]
        print("\ncompose by target attribute A (expected-target first-token rate)")
        for L, arms_ in cr["by_target_attribute"].items():
            for arm, d in arms_.items():
                print(f"  L{L:<3} {arm:<18} " + "  ".join(f"{a} {v['hit']:5.1%} (n={v['n']})" for a, v in d.items()))
        print("\nwhat the 'other' first tokens said (typed against the arm's expected cell)")
        for L, arms_ in cr["other_breakdown"].items():
            for arm, d in arms_.items():
                print(f"  L{L:<3} {arm:<18} n_other={d['n_other']:>4}  " +
                      "  ".join(f"{k} {v:5.1%}" for k, v in d.items() if k != "n_other"))
        if cr.get("text_ok_subset"):
            print("\ncross-modal arms vs their image twins, on rows the TEXT model also answers (B's A and X)")
            for L, arms_ in cr["text_ok_subset"].items():
                for arm in ("additive", "xm_additive", "swap_entity", "xm_swap_entity"):
                    if arm in arms_:
                        s_ = arms_[arm]
                        print(f"  L{L:<3} {arm:<16} n={s_['n']:>4} expect {s_['expect']} hit {s_['first_' + s_['expect']]:6.1%}")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
