"""Train a subspace intervention ON the entity heads, scored by VADE.

WHY THIS EXISTS

R10/R11 of ATTRIBUTE_HEAD_EXPERIMENTS.md establish two things about the ten
blocks-21-23 heads (21.1, 21.5, 22.13, 22.15, 22.17, 22.19, 23.3, 23.4, 23.6,
23.17):

  * swapping their FULL 128-dim columns reproduces a whole-image swap
    (88.5% mean cause vs full_image's 93.0%, random heads 0.0%), and
  * what crosses is the ENTITY, not the answer -- `head_cross`'s `both` cell
    returns the donor's flag answered for the RECEIVING prompt's question,
    with the donor's own answer at exactly 0.0%.

So the conduit moves every attribute together, which is precisely what VADE's
`final_score` punishes: cause ~100 / iso ~0 averages to the 50% null. The open
question is whether the entity payload DECOMPOSES -- whether some subspace of
those 1,280 columns carries "which language" separably from "which country".

R11.5 shows the obvious shortcut fails: a 7-dim subspace read off the value
centroids' variance transfers 0.0%, less than a random-head control, despite
decoding `language` at 100%. Linear decodability is not causal sufficiency --
the same failure RESULTS.md records for the Phase B selection proxy.

This script does it the only way left: learn the subspace AGAINST THE
GENERATION OBJECTIVE, which is what DAS is.

WHAT IS TRAINED

One intervention per (entity, attribute) -- a `language` rotation is a
different object from a `capital` rotation, and must be, since the two make
opposite demands of the same activations. Each is block-diagonal by necessity:
blocks run sequentially, so patching block 21 changes what block 22 computes and
a single joint rotation across all three is not expressible in one hook. Widths
are n_heads_in_block * head_dim (common10: 256 / 512 / 512).

    das_fixed     FixedSubspaceIntervention -- a D x K semi-orthogonal R,
                  K fixed up front. output = base + R^T (R source - R base).
    das_rotated   RotatedSpaceIntervention -- a full D x D learned rotation
                  plus a sigmoid mask annealed over training, so K is learned.
    dbm           SigmoidMaskIntervention -- no rotation at all, an
                  axis-aligned mask over the raw head dimensions, + L1.
                  The privileged-basis hypothesis: are head dims themselves
                  the right coordinates?

All three are the SAME loop with a different nn.Module in the middle; the
modules are imported, not reimplemented (DAS's from the sibling VADE repo,
DBM's from methods/dbm/, which wraps pyvene's own class).

THE OBJECTIVE IS ALREADY VADE'S METRIC

`targets.target_gold_toks_and_len` supervises each row toward the SOURCE gold
when `rule == match_source` (the cause pool) and toward the BASE gold when
`rule == match_base` (the iso pool). VADE's train tuples for an attribute
contain both, so cross-entropy over them IS `final_score` made differentiable.
A cause-only objective would simply relearn the full swap -- R10 shows cause is
free at this site and iso is the entire difficulty.

  NOTE ON POOL BALANCE. The train split is ~57% cause / 43% iso, while the
  metric weights cause 1/2 and the iso MEAN 1/2. `--iso_weight` rescales the
  iso rows' loss to correct for that; 1.0 leaves the natural mixture.

THE PATCH COVERS EVERY ANSWER TOKEN, NOT JUST THE LAST PROMPT TOKEN

A last-prompt-token patch cannot steer past the first answer token (see
head_swap_vade.py's header and ndm/swap_trace.py), which is fatal for
`calling_code` and `capital`. `build_teacher_forced_extension` appends the true
answer tokens so that the last MAX_ANSWER_TOKENS columns of one forward are
exactly [last prompt token, answer tok 0, ... answer tok K-2], predicting answer
tokens 0..K-1. The hook patches ALL of those columns, with donor column j
aligned to donor answer step j -- the differentiable equivalent of the
step-by-step replay that `patched_generate` does at eval time.

The donor's own trajectory is captured the same way: one teacher-forced forward
of the SOURCE image asking the SAME question, extended by the SOURCE's gold, so
its columns are "the source emitting its own correct answer" step for step. That
is one forward instead of K, and it removes a train/eval mismatch -- pass
`--donor_capture generate` to reproduce head_swap_vade's generation-time capture
instead.

WHAT A RESULT MEANS

  final_score > 50%     the entity payload decomposes: a subspace moves the
                        queried attribute without dragging the country along.
                        This is the result the whole head line of work is for.
  final_score ~ 50%     it does not decompose at this site, under this
                        hypothesis class. Check WHICH corner: cause ~100/iso ~0
                        means the rotation just relearned the full swap
                        (raise --iso_weight, lower K); cause ~0/iso ~100 means
                        it learned to do nothing (K too small, or lr/epochs).
  cause ~0 AND iso ~0   broken, not a finding. Check the read-back gate.

Train on VADE's `train` split, evaluate on `test`. These are split by ITEM, not
by row: on flags, 59 countries train and a disjoint 25 test (0 overlap, asserted
in --dry_run). So a rotation that scores here has generalized to flags it never
saw, which is the claim worth making -- and a much harder one than a row split
would support. `--train_rows`/`--eval_rows` cap each independently; subsampling
is stratified by cause/iso, never by item, so the disjointness survives it.

Usage
-----
    # sanity: no model, validates rows/heads/widths/splits
    python methods/head_das.py --attribute language --dry_run

    # train + evaluate, K=8 per block
    python methods/head_das.py --attribute language --method das_fixed \
        --subspace_dim 8 --train_rows 4000 --eval_rows 4000

    # the learned-K variant, and the axis-aligned control
    python methods/head_das.py --attribute language --method das_rotated --train_rows 4000
    python methods/head_das.py --attribute language --method dbm --l1_coef 1e-3 --train_rows 4000

    # score (sibling VADE repo)
    python ../VADE/eval/score.py --entity flags --attribute language \
        --predictions results/head_das/flags/language_das_fixed_k8/predictions.jsonl
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import (  # noqa: E402
    ATTRIBUTES, COMMON10, DEFAULT_VADE_ROOT, MODEL_ID, build_prompt, capture_donor, decode,
    head_columns, load_assets, parse_heads, patched_generate, to_device,
)
from methods.common.targets import (  # noqa: E402
    MAX_ANSWER_TOKENS, build_teacher_forced_extension, derive_gold_token_ids,
    gold_labels_from_lens, pad_gold_toks,
)

METHODS = ("das_fixed", "das_rotated", "dbm")


# ---------------------------------------------------------------------------
# Interventions -- imported, never reimplemented
# ---------------------------------------------------------------------------

def load_das_module(vade_root):
    """VADE's own das/intervention.py, loaded by path.

    It is pure nn.Module code on hidden-state tensors with zero model or entity
    dependency (its own docstring says so), but it lives in the sibling repo and
    this one pins no dependency on VADE's package layout -- so load it the way
    head_cross.py loads VADE's `contains_label`, rather than copying a second
    divergent copy into this tree."""
    import importlib.util
    path = os.path.join(vade_root, "methods", "das", "intervention.py")
    assert os.path.exists(path), f"VADE DAS intervention not found at {path} (--vade_root)"
    spec = importlib.util.spec_from_file_location("_vade_das_intervention", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_intervention(method, dim, subspace_dim, vade_root, mask_init=None):
    """One intervention for one block's head columns."""
    if method == "dbm":
        from methods.dbm.intervention import SigmoidMaskIntervention
        return SigmoidMaskIntervention(embed_dim=dim)
    das = load_das_module(vade_root)
    if method == "das_fixed":
        k = min(subspace_dim, dim)
        assert k > 0, "--subspace_dim must be > 0 for das_fixed"
        return das.FixedSubspaceIntervention(dim, k)
    if method == "das_rotated":
        return (das.RotatedSpaceIntervention(dim) if mask_init is None
                else das.RotatedSpaceIntervention(dim, mask_init=float(mask_init)))
    raise ValueError(f"unknown method {method!r} (expected one of {METHODS})")


def parse_subspace_spec(spec, n_blocks):
    """`--subspace_dim` -> one width per block, in sorted-block order.

    Three forms, because K is NOT naturally one number here: the traced heads
    are unevenly distributed over blocks (common10 is 2/4/4 heads = 256/512/512
    columns), so a single K gives block 21 twice the fraction of its space that
    22 and 23 get.

      "128"         every block 128 (clamped to its own width)
      "128,64,64"   one per block, in sorted-block order
      "full"        each block's own width -- the ceiling arm

    `None` in the returned list means "this block's full width", resolved once
    the real column counts are known."""
    spec = str(spec).strip()
    if spec.lower() == "full":
        return [None] * n_blocks
    parts = [x.strip() for x in spec.split(",")]
    assert len(parts) in (1, n_blocks), (
        f"--subspace_dim {spec!r} has {len(parts)} values but there are {n_blocks} blocks; "
        f"pass one value for all blocks, {n_blocks} values, or 'full'")
    out = []
    for x in parts:
        assert x.lower() == "full" or (x.lstrip("-").isdigit()), \
            f"--subspace_dim component {x!r} is neither an integer nor 'full'"
        if x.lower() == "full":
            out.append(None)
        else:
            k = int(x)
            assert k > 0, f"--subspace_dim component {k} must be > 0"
            out.append(k)
    return out * n_blocks if len(parts) == 1 else out


def subspace_dof(k, dim):
    """Effective degrees of freedom of a das_fixed block: dim Gr(k, dim).

    `FixedSubspaceIntervention`'s output depends on R only through R^T R, so two
    R's spanning the same subspace are the SAME function -- the class is the
    Grassmannian, of dimension k(dim-k), not the k*dim stored parameters. It is
    0 at k == dim: there R^T R = I identically, the intervention IS the full
    swap whatever the weights say, and every gradient is pure gauge. A run like
    that is a ceiling measurement, not training."""
    k = min(k, dim)
    return k * (dim - k)


def resolve_subspace_dims(spec, blocks, widths):
    """-> {block: k}, with None/oversized entries clamped to the block's width."""
    vals = parse_subspace_spec(spec, len(blocks))
    return {b: (widths[b] if v is None else min(v, widths[b]))
            for b, v in zip(sorted(blocks), vals)}


def mask_travel_budget(mask_init, temps, mask_lr, eps=1e-6):
    """Can the mask actually move? -> (n_live_steps, travel, need).

    `das_rotated`'s mask enters the loss only as sigmoid(m / T), so every
    gradient reaching `masks` is scaled by sigmoid\'(m/T)/T = s(1-s)/T. Once
    m/T is large that factor underflows to EXACTLY 0 in float32 and the mask is
    frozen for the rest of training -- silently, with the run still producing
    fluent text and a plausible score. That is what made the first three
    --mask_coef arms byte-identical (R12.3).

    Adam normalizes by gradient magnitude, so while the factor is non-zero the
    mask moves ~mask_lr per step; `travel` is that budget and `need` is the
    distance from mask_init to 0 (where sigmoid = 0.5, i.e. the mask can still
    express "drop this dimension"). travel < need means the arm is PINNED at
    its initialization whatever the optimizer is told."""
    import torch
    m = float(mask_init)
    live = 0
    for t in temps:
        s = torch.sigmoid(torch.tensor(m / float(t), dtype=torch.float32))
        if float(s * (1 - s)) / float(t) > eps:
            live += 1
    return live, live * float(mask_lr), abs(m)


def build_interventions(method, colmap, sub_dims, vade_root, device, mask_init=None):
    """One intervention per block, on that block's head columns."""
    return {b: make_intervention(method, len(colmap[b]), sub_dims[b], vade_root, mask_init).to(device)
            for b in sorted(colmap)}


def temperature_schedule_for(method, n_steps, vade_root, t_start=None, t_end=None):
    """-> [n_steps] temperatures, or None for a method with no mask to anneal.

    `t_start`/`t_end` override the method's own defaults. They exist because the
    anneal and `mask_init` are a MATCHED PAIR -- the mask enters the loss only
    through sigmoid(m/T), so what the optimizer sees is the RATIO. VADE's
    (mask_init 150, T 50->0.1) puts that ratio at 3.0 for one step and past 15
    -- numerically saturated -- within a quarter of training. See
    `mask_travel_budget`."""
    if method == "dbm":
        from methods.dbm.intervention import temperature_schedule
        return temperature_schedule(n_steps)                 # RAVEL's 1e-2 -> 1e-7
    if method == "das_rotated":
        sched = load_das_module(vade_root).temperature_schedule            # 50 -> 0.1
        kw = {}
        if t_start is not None: kw["temperature_start"] = float(t_start)
        if t_end is not None: kw["temperature_end"] = float(t_end)
        return sched(n_steps, **kw)
    return None


def intervention_stats(method, interventions):
    """JSON-able summary of what training actually selected, per block."""
    import torch
    out = {}
    for b, iv in sorted(interventions.items()):
        with torch.no_grad():
            if method == "das_fixed":
                dim, k = int(iv.proj.weight.shape[1]), int(iv.subspace_dim)
                out[b] = {"kind": "fixed_subspace", "dim": dim, "subspace_dim": k,
                          # k(dim-k), not k*dim: see subspace_dof(). 0 means this
                          # block was an untrainable full swap.
                          "dof": subspace_dof(k, dim)}
            elif method == "das_rotated":
                m = torch.sigmoid(iv.masks / iv.temperature)
                out[b] = {"kind": "rotated_mask", "dim": int(iv.masks.shape[0]),
                          "temperature": float(iv.temperature),
                          "mask_sum": float(m.sum()), "n_above_0.5": int((m > 0.5).sum())}
            else:
                from methods.dbm.intervention import mask_stats
                s = mask_stats(iv)
                s.pop("selected_indices", None)              # can be thousands of ints
                out[b] = {"kind": "sigmoid_mask", **s}
    return out


# ---------------------------------------------------------------------------
# Rows and batches
# ---------------------------------------------------------------------------

def load_rows(vade_root, entity, attribute, split, n_rows, seed):
    """VADE tuple rows for ONE attribute (that file's `target_attribute`).

    Subsampling is stratified by `rule` so a small --train_rows never drops one
    of the two pools entirely -- which would silently turn this into the
    cause-only training the docstring warns about."""
    path = os.path.join(vade_root, "data", entity, "tuples", attribute, f"{split}.jsonl")
    assert os.path.exists(path), f"no tuples at {path}"
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    if n_rows and n_rows < len(rows):
        rng = random.Random(seed)
        cause = [r for r in rows if r["rule"] == "match_source"]
        iso = [r for r in rows if r["rule"] != "match_source"]
        frac = n_rows / len(rows)
        keep = (rng.sample(cause, max(1, round(len(cause) * frac)))
                + rng.sample(iso, max(1, round(len(iso) * frac))))
        rows = sorted(keep, key=lambda r: r["row_index"])
    return rows


class PromptCache:
    """Collapses build_batch's three pure-but-repeated computations.

    An entity has ~59 train images backing thousands of tuple rows (each image
    is somebody's base in some rows and somebody's source in others) and 24
    (queried, template_id) prompts, so building every row from scratch redoes
    the same PIL decode + vision preprocess ~200x and the same tokenization
    ~500x per epoch. Same reasoning -- and the same three keys -- as
    common/entities.py's BuildBatchCache, which is why VADE's own DAS trainer is
    much faster than this script was without it.

      image[item]                    -> (pixel_values, image_grid_thw)
      ids[(queried, template_id)]    -> input_ids
      gold[(prefill, label)]         -> gold token ids

    Splitting image from ids is what keeps this at 59 + 24 entries instead of
    their 1,416-entry cross product. It is valid because every image in a VADE
    entity is one fixed canvas render, so the fully tokenized prompt -- image
    placeholder tokens included -- depends only on the template, never on which
    image fills it. That is an assumption, not a guarantee, so `verify` checks
    it once per template against a SECOND item and fails loudly: a silent
    mismatch here would mean patching one prompt's columns while reading
    another's, the same class of bug that voided a whole head_trace result set
    (see common/entities.py's BuildBatchCache docstring)."""

    def __init__(self):
        self.image, self.ids, self.gold = {}, {}, {}
        self.builds = 0

    def _build(self, processor, entity_dir, items, item, tmpl):
        from PIL import Image
        self.builds += 1
        with Image.open(os.path.join(entity_dir, items[item]["image"])) as im:
            return build_prompt(processor, im.convert("RGB"), tmpl["question"], tmpl["prefill"])

    def prompt(self, processor, entity_dir, items, item, queried, template_id, tmpl):
        """-> (input_ids[0], pixel_values, image_grid_thw).

        A build happens only when this item's pixels or this template's ids are
        missing, so the total is bounded by n_items + n_templates rather than
        n_rows. Whenever a build lands on a template already in the cache -- the
        normal case as new items appear -- the fresh ids are COMPARED against
        the stored ones rather than discarded, so the fixed-canvas assumption is
        verified continuously at zero extra cost."""
        import torch
        key = (queried, template_id)
        if item not in self.image or key not in self.ids:
            built = self._build(processor, entity_dir, items, item, tmpl)
            ids = built["input_ids"][0]
            if key in self.ids:
                assert torch.equal(ids, self.ids[key]), (
                    f"prompt ids for {key} differ between items -- the fixed-canvas assumption "
                    f"this cache rests on does not hold for {item!r} (got {tuple(ids.shape)} vs "
                    f"cached {tuple(self.ids[key].shape)}); rerun with --no_prompt_cache")
            else:
                self.ids[key] = ids
            self.image.setdefault(item, (built["pixel_values"], built["image_grid_thw"]))
        px, thw = self.image[item]
        return self.ids[key], px, thw

    def gold_ids(self, tokenizer, prefill, label):
        key = (prefill, label)
        if key not in self.gold:
            self.gold[key] = derive_gold_token_ids(tokenizer, prefill, label)
        return self.gold[key]


def build_batch(rows, processor, items, entity_dir, lookup, pad_id, cache=None):
    """Base and donor prompts plus both golds, for one chunk.

    The donor asks the SAME question as the base (head_swap_vade's
    `donor_question=queried`): the edit must not be told which attribute the
    VADE row targets, or the intervention could satisfy iso by reading the
    label rather than by isolating a subspace.

    Pass ONE PromptCache across every call of a run to get the reuse; a fresh
    one per call just reproduces the uncached behavior."""
    import torch
    from PIL import Image

    tok = processor.tokenizer
    cache = cache if cache is not None else PromptCache()
    base_seqs, donor_seqs = [], []
    base_px, donor_px, base_thw, donor_thw = [], [], [], []
    base_golds, source_golds = [], []
    for r in rows:
        t = lookup[r["queried"]][r["template_id"]]
        bi, bpx, bthw = cache.prompt(processor, entity_dir, items, r["base"],
                                     r["queried"], r["template_id"], t)
        di, dpx, dthw = cache.prompt(processor, entity_dir, items, r["source"],
                                     r["queried"], r["template_id"], t)
        base_seqs.append(bi); donor_seqs.append(di)
        base_px.append(bpx); donor_px.append(dpx)
        base_thw.append(bthw); donor_thw.append(dthw)
        base_golds.append(cache.gold_ids(tok, t["prefill"], str(r["base_label"])))
        source_golds.append(cache.gold_ids(tok, t["prefill"], str(r["source_label"])))

    def pack(seqs):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = s
            mask[i, n - len(s):] = 1
        return ids, mask

    base_ids, base_mask = pack(base_seqs)
    donor_ids, donor_mask = pack(donor_seqs)
    bg, bl = pad_gold_toks(base_golds, pad_id)
    sg, sl = pad_gold_toks(source_golds, pad_id)
    is_cause = torch.tensor([r["rule"] == "match_source" for r in rows])
    return {
        "rows": rows,
        "base_ids": base_ids, "base_mask": base_mask,
        "base_extra": {"pixel_values": torch.cat(base_px), "image_grid_thw": torch.cat(base_thw)},
        "donor_ids": donor_ids, "donor_mask": donor_mask,
        "donor_extra": {"pixel_values": torch.cat(donor_px), "image_grid_thw": torch.cat(donor_thw)},
        "base_gold_toks": bg, "base_gold_len": bl,
        "source_gold_toks": sg, "source_gold_len": sl,
        "is_cause": is_cause,
        # Supervision target: source gold on cause rows, base gold on iso rows.
        "target_toks": torch.where(is_cause.unsqueeze(1), sg, bg),
        "target_len": torch.where(is_cause, sl, bl),
    }


# ---------------------------------------------------------------------------
# The donor's answer-token trajectory
# ---------------------------------------------------------------------------

def capture_donor_columns(adapter, model, blocks, batch, pad_id):
    """-> {block: [B, K, hidden]} of attn_head_output at the LAST K columns of
    ONE teacher-forced donor forward.

    The donor is extended by its OWN gold (the SOURCE's label), so column j is
    the source's state while emitting its answer token j -- the same alignment
    `capture_donor`'s step j has, at 1 forward instead of K."""
    import torch
    ext_ids, ext_mask = build_teacher_forced_extension(
        batch["donor_ids"], batch["donor_mask"], batch["source_gold_toks"], batch["source_gold_len"])
    sinks, handles = {}, []
    for b in blocks:
        sinks[b] = []

        def grab(_mod, args, _sink=sinks[b]):
            _sink.append(args[0][:, -MAX_ANSWER_TOKENS:, :].detach().float())
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(grab))
    try:
        with torch.no_grad():
            model(input_ids=ext_ids.to(model.device), attention_mask=ext_mask.to(model.device),
                  **to_device(batch["donor_extra"], model.device, model.dtype),
                  logits_to_keep=MAX_ANSWER_TOKENS)
    finally:
        for h in handles:
            h.remove()
    for b, v in sinks.items():
        assert len(v) == 1, f"block {b} o_proj fired {len(v)} times in one forward, expected 1"
    return {b: v[0] for b, v in sinks.items()}


def donor_key(row):
    """What the donor's head columns are a pure function of.

    The donor prompt is (source image + the QUERIED attribute's question at this
    template), teacher-forced by the source's own gold for that attribute -- so
    these three fields determine it completely, and nothing about the base does.

    Note this is a WIDER key than common/source_cache.py's item-only one, and it
    has to be: that cache is sound only for IMAGE positions, where causal
    attention makes the hidden state independent of the question that follows.
    These columns sit at the last prompt token and the answer tokens, which are
    downstream of the question, so caching them per item alone would be wrong."""
    return (row["source"], row["queried"], row["template_id"])


def donor_columns(adapter, model, blocks, colmap, batch, pad_id, cache=None):
    """-> {block: [B, K, W]}, computing only the rows not already cached.

    6,000 train rows carry ~1,330 distinct donor keys, so after the first pass
    most batches need no donor forward at all.

    W is n_cols_b when `colmap` is given (the training path, which feeds
    `intervened_logits` directly) and the full hidden size when it is None (the
    eval path: `patched_generate` does its own column selection, so handing it
    pre-sliced values would index the wrong columns). Slicing shrinks a cached
    entry from ~129KB to ~15KB, so the training cache is the one worth it."""
    import torch

    def cut(t, b):
        return t if colmap is None else t.index_select(-1, colmap[b].to(t.device))

    if cache is None:
        z = capture_donor_columns(adapter, model, blocks, batch, pad_id)
        return {b: cut(z[b], b) for b in blocks}

    keys = [donor_key(r) for r in batch["rows"]]
    need = sorted({k for k in keys if k not in cache}, key=str)
    if need:
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        idx = torch.tensor([first[k] for k in need])
        sub = {"donor_ids": batch["donor_ids"][idx], "donor_mask": batch["donor_mask"][idx],
               "donor_extra": _slice_extra(batch["donor_extra"], batch["donor_ids"].shape[0], idx),
               "source_gold_toks": batch["source_gold_toks"][idx],
               "source_gold_len": batch["source_gold_len"][idx]}
        z = capture_donor_columns(adapter, model, blocks, sub, pad_id)
        for j, k in enumerate(need):
            cache[k] = {b: cut(z[b][j], b).cpu() for b in blocks}
    return {b: torch.stack([cache[k][b] for k in keys]).to(model.device) for b in blocks}


def _slice_extra(extra, n_rows, idx):
    """Select rows out of a batch's vision inputs.

    pixel_values is NOT one row per image -- Qwen2.5-VL concatenates each
    image's patches along dim 0, so the rows must be split by each image's own
    patch count (the product of its grid_thw) before they can be indexed."""
    import torch
    thw = extra["image_grid_thw"]
    assert thw.shape[0] == n_rows, f"expected one grid per row, got {thw.shape[0]} for {n_rows}"
    counts = thw.prod(dim=1).tolist()
    parts = list(torch.split(extra["pixel_values"], counts, dim=0))
    return {"pixel_values": torch.cat([parts[i] for i in idx.tolist()]),
            "image_grid_thw": thw.index_select(0, idx)}


# ---------------------------------------------------------------------------
# The intervened teacher-forced forward
# ---------------------------------------------------------------------------

def intervened_logits(adapter, model, interventions, colmap, donor_cols, batch):
    """Teacher-forced forward with every block's intervention live at the last
    MAX_ANSWER_TOKENS columns. Returns [B, K, vocab] aligned 1:1 with
    target_toks. Differentiable through the interventions."""
    import torch
    ext_ids, ext_mask = build_teacher_forced_extension(
        batch["base_ids"], batch["base_mask"], batch["target_toks"], batch["target_len"])
    handles = []
    for b, iv in interventions.items():
        cols = colmap[b].to(model.device)
        z = donor_cols[b].to(model.device)                   # [B, K, d_b], already sliced

        def patch(_mod, args, _iv=iv, _cols=cols, _z=z):
            t = args[0]
            have = t[:, -MAX_ANSWER_TOKENS:, :].index_select(-1, _cols)
            new = _iv(have, _z.to(have.dtype))
            patched = t.clone()
            patched[:, -MAX_ANSWER_TOKENS:, _cols] = new.to(t.dtype)
            return (patched,) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(patch))
    try:
        out = model(input_ids=ext_ids.to(model.device), attention_mask=ext_mask.to(model.device),
                    **to_device(batch["base_extra"], model.device, model.dtype),
                    logits_to_keep=MAX_ANSWER_TOKENS)
    finally:
        for h in handles:
            h.remove()
    return out.logits[:, -MAX_ANSWER_TOKENS:, :]


def weighted_ce(logits, batch, iso_weight):
    """Per-row CE against each row's own gold, with the iso pool rescaled.

    VADE weights `cause` 1/2 and the MEAN over iso attributes 1/2, but the train
    split is cause-heavy, so the natural mixture over-weights the objective that
    is already free at this site."""
    import torch
    labels = gold_labels_from_lens(batch["target_toks"], batch["target_len"]).to(logits.device)
    per_tok = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
        ignore_index=-100, reduction="none").view(labels.shape)
    valid = (labels != -100).float()
    per_row = (per_tok * valid).sum(1) / valid.sum(1).clamp(min=1)
    w = torch.where(batch["is_cause"].to(logits.device), 1.0, float(iso_weight))
    return (per_row * w).sum() / w.sum()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def sparsity_term(method, interventions, args):
    """-> (name, raw, tensor_or_None): the pressure that shrinks the edit.

    `das_fixed` needs none -- K is fixed by construction, so the subspace cannot
    grow. The other two learn their own width and therefore need a cost, or the
    cause objective (free at this site -- R10) simply keeps every dimension and
    reproduces the full swap:

      dbm          l1_coef * ||m||_1 on the raw mask, RAVEL's own formulation.
      das_rotated  mask_coef * sum(sigmoid(m / T)), i.e. the SOFT DIMENSION
                   COUNT of the learned rotation -- Boundless DAS's boundary
                   penalty. VADE's own trainer never instantiates
                   RotatedSpaceIntervention and so has no equivalent term;
                   without one the mask starts at sigmoid(150/50) ~ 0.95 on
                   every dimension and has no reason to ever come down.

    `raw` is the interpretable number to watch (mask magnitude / effective K);
    the loss gets coefficient * raw."""
    if method == "dbm" and args.l1_coef:
        from methods.dbm.intervention import l1_penalty
        t = sum(l1_penalty(iv) for iv in interventions.values())
        return "l1", t, args.l1_coef
    if method == "das_rotated" and args.mask_coef:
        t = sum(iv.mask_sum for iv in interventions.values())
        return "maskK", t, args.mask_coef
    return None, None, 0.0


def fmt_hms(seconds):
    """-> 'H:MM:SS'. Used for both elapsed and ETA so the two are comparable at
    a glance; ETA is a plain linear extrapolation from the mean batch so far,
    which is honest here because every batch does the same two forwards and one
    backward on a fixed-size grid."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def progress(done, total, t0):
    """-> (elapsed_str, eta_str, rate_per_unit). `done` counts completed units."""
    import time
    elapsed = time.time() - t0
    rate = elapsed / max(done, 1)
    return fmt_hms(elapsed), fmt_hms(rate * (total - done)), rate


def accum_group(i, n_batches, grad_accum_steps):
    """-> (index within the accumulation group, real size of that group).

    The final group of an epoch is usually SHORT (n_batches is rarely a multiple
    of grad_accum_steps). Dividing it by grad_accum_steps anyway scales the
    tail's gradient down by group_size/grad_accum_steps -- the same tail-flush
    bug methods/dbm/train.py fixed in 77470b4 (see CLAUDE.md). Returning the real
    size keeps every optimizer step a true mean over the batches it saw."""
    start = (i // grad_accum_steps) * grad_accum_steps
    return i - start, min(grad_accum_steps, n_batches - start)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def train(adapter, model, processor, args, heads, colmap, items, entity_dir, lookup, pad_id,
          log_path, interventions):
    import torch

    blocks = sorted(colmap)
    params = [p for iv in interventions.values() for p in iv.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    # `masks` gets its own group: it needs a step size set by the DISTANCE it must
    # travel (mask_init -> 0) over the few steps its gradient survives, which has
    # nothing to do with the rotation's step size. One shared lr cannot serve both.
    mask_ps = [p for iv in interventions.values() for n, p in iv.named_parameters()
               if n == "masks" and p.requires_grad]
    if mask_ps and args.mask_lr is not None:
        ids = {id(p) for p in mask_ps}
        opt = torch.optim.Adam([{"params": [p for p in params if id(p) not in ids], "lr": args.lr},
                                {"params": mask_ps, "lr": args.mask_lr}])
        print(f"  param groups: rotation lr={args.lr:g}, masks lr={args.mask_lr:g}")
    else:
        opt = torch.optim.Adam(params, lr=args.lr)

    rows = load_rows(args.vade_root, args.entity, args.attribute, args.train_split,
                     args.train_rows, args.seed)
    batches = list(chunks(rows, args.batch_size))
    n_opt_steps = args.epochs * max(1, (len(batches) + args.grad_accum_steps - 1) // args.grad_accum_steps)
    temps = temperature_schedule_for(args.method, n_opt_steps, args.vade_root,
                                     args.temperature_start, args.temperature_end)
    if args.method == "das_rotated":
        mi = 150.0 if args.mask_init is None else float(args.mask_init)
        lr_m = args.lr if args.mask_lr is None else args.mask_lr
        live, travel, need = mask_travel_budget(mi, temps, lr_m)
        print(f"  mask anneal T {float(temps[0]):g} -> {float(temps[-1]):g}; mask_init={mi:g}")
        print(f"  mask gradient survives {live}/{len(temps)} steps -> travel budget "
              f"{travel:.1f} vs {need:.1f} needed to reach sigmoid=0.5")
        if travel < need:
            print(f"  WARNING: the mask CANNOT reach 0.5 -- it is pinned near "
                  f"sigmoid({mi:g}/{float(temps[0]):g})={float(torch.sigmoid(torch.tensor(mi/float(temps[0])))):.4f} "
                  f"and this arm is an (almost) FULL SWAP whatever --mask_coef says. "
                  f"Raise --mask_lr to >= {need / max(live, 1):.3g}, or lower --mask_init "
                  f"and --temperature_start together (their RATIO is what the sigmoid sees).")
    n_cause = sum(r["rule"] == "match_source" for r in rows)
    tail = len(batches) % args.grad_accum_steps
    print(f"  train: {len(rows)} rows ({n_cause} cause / {len(rows) - n_cause} iso), "
          f"{len(batches)} batches x {args.epochs} epochs = {n_opt_steps} opt steps")
    print(f"  effective batch: {args.batch_size} x {args.grad_accum_steps} accum = "
          f"{args.batch_size * args.grad_accum_steps} rows/step"
          + (f" (final group of each epoch is {tail} batches)" if tail else ""))
    print(f"  params: {n_params:,} over blocks {blocks} "
          f"(dims {[len(colmap[b]) for b in blocks]})")

    prompt_cache = None if args.no_prompt_cache else PromptCache()
    donor_cache = None if args.no_donor_cache else {}

    log = open(log_path, "w")
    step = 0
    import time
    t0 = time.time()
    total_batches = args.epochs * len(batches)
    for epoch in range(args.epochs):
        accum, accum_ce, accum_reg = 0.0, 0.0, 0.0
        for i, chunk in enumerate(batches):
            in_group, group_size = accum_group(i, len(batches), args.grad_accum_steps)
            batch = build_batch(chunk, processor, items, entity_dir, lookup, pad_id, prompt_cache)
            donor_cols = donor_columns(adapter, model, blocks, colmap, batch, pad_id, donor_cache)
            logits = intervened_logits(adapter, model, interventions, colmap, donor_cols, batch)
            ce = weighted_ce(logits, batch, args.iso_weight)
            reg_name, reg_t, reg_coef = sparsity_term(args.method, interventions, args)
            reg = 0.0 if reg_t is None else float(reg_t)
            loss = ce if reg_t is None else ce + reg_coef * reg_t
            (loss / group_size).backward()
            accum += float(loss); accum_ce += float(ce); accum_reg += reg

            n_cause_b = int(batch["is_cause"].sum())
            done = epoch * len(batches) + i + 1
            elapsed, eta, rate = progress(done, total_batches, t0)
            losses = f"loss={float(loss):.4f} ce={float(ce):.4f}"
            if reg_name:
                losses += f" {reg_name}={reg:.1f} ({reg_coef * reg:.4f})"
            if args.log_every and (i % args.log_every == 0 or i == len(batches) - 1):
                print(f"    e{epoch} batch {i + 1}/{len(batches)} "
                      f"[{in_group + 1}/{group_size} of step {step + 1}/{n_opt_steps}] "
                      f"rows={len(chunk)} ({n_cause_b}c/{len(chunk) - n_cause_b}i) "
                      f"{losses} | {elapsed} eta {eta} ({rate:.2f}s/batch)", flush=True)
            log.write(json.dumps({"kind": "batch", "epoch": epoch, "batch": i,
                                  "opt_step": step + 1, "rows": len(chunk),
                                  "n_cause": n_cause_b, "loss": float(loss), "ce": float(ce),
                                  "reg": reg, "reg_name": reg_name, "elapsed_s": round(time.time() - t0, 1)}) + "\n")

            if in_group + 1 == group_size:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                opt.step(); opt.zero_grad(set_to_none=True)
                if temps is not None:
                    t = float(temps[min(step, len(temps) - 1)])
                    for iv in interventions.values():
                        iv.set_temperature(t)
                step += 1
                rec = {"kind": "opt_step", "epoch": epoch, "opt_step": step,
                       "group_size": group_size, "loss": accum / group_size,
                       "ce": accum_ce / group_size, "reg": accum_reg / group_size,
                       "elapsed_s": round(time.time() - t0, 1),
                       "temperature": None if temps is None
                       else float(temps[min(step - 1, len(temps) - 1)])}
                log.write(json.dumps(rec) + "\n"); log.flush()
                e_s, eta_s, _ = progress(epoch * len(batches) + i + 1, total_batches, t0)
                print(f"  >> epoch {epoch} step {step}/{n_opt_steps} "
                      f"({group_size} batches = {group_size * args.batch_size} rows) "
                      f"loss={accum / group_size:.4f} ce={accum_ce / group_size:.4f}"
                      + (f" {reg_name}={accum_reg / group_size:.1f}" if reg_name else "")
                      + f" | {e_s} eta {eta_s}"
                      + ("" if donor_cache is None else f" | donor cache {len(donor_cache)}"),
                      flush=True)
                accum, accum_ce, accum_reg = 0.0, 0.0, 0.0
    log.close()
    if prompt_cache is not None:
        print(f"  prompt cache: {prompt_cache.builds} prompt builds for "
              f"{2 * len(rows) * args.epochs} row-sides "
              f"({len(prompt_cache.image)} images, {len(prompt_cache.ids)} templates, "
              f"{len(prompt_cache.gold)} golds)")
    if donor_cache is not None:
        print(f"  donor cache: {len(donor_cache)} distinct donor states for "
              f"{len(rows) * args.epochs} rows")
    return interventions


# ---------------------------------------------------------------------------
# Evaluate -- free generation, scored by VADE
# ---------------------------------------------------------------------------

def evaluate(adapter, model, processor, args, heads, colmap, items, entity_dir, lookup,
             pad_id, head_dim, interventions, out_path):
    """Writes predictions in VADE/eval/score.py's format.

    Generation goes through head_swap_vade.patched_generate with
    `transform=`, NOT a local copy: the generate path that produced R10 and R11
    is the one this is measured on, so a difference between arms cannot be a
    difference between two generation loops."""
    import torch

    blocks = sorted(colmap)
    for iv in interventions.values():
        iv.eval()
    rows = load_rows(args.vade_root, args.entity, args.attribute, args.eval_split,
                     args.eval_rows, args.seed + 1)
    print(f"  eval: {len(rows)} rows from {args.eval_split} -> {out_path}")

    def transform_for(b):
        iv = interventions[b]

        def fn(have, want):
            with torch.no_grad():
                return iv(have, want.to(have.dtype))
        return fn
    transform = {b: transform_for(b) for b in blocks}

    import time
    t0 = time.time()
    prompt_cache = None if args.no_prompt_cache else PromptCache()
    # NOT shared with training: eval is a disjoint item set, and a stale entry
    # from the train split could only ever be a bug. `generate` capture is
    # uncached -- its step count depends on when the donor hits EOS.
    donor_cache = None if (args.no_donor_cache or args.donor_capture == "generate") else {}
    n = 0
    with open(out_path, "w") as f:
        for chunk in chunks(rows, args.batch_size):
            batch = build_batch(chunk, processor, items, entity_dir, lookup, pad_id, prompt_cache)
            if args.donor_capture == "generate":
                dz = capture_donor(adapter, model, blocks, batch["donor_ids"], batch["donor_mask"],
                                   batch["donor_extra"], args.max_new_tokens, pad_id)
            else:
                # colmap=None: patched_generate selects the head columns itself.
                dz = {b: v.cpu() for b, v in
                      donor_columns(adapter, model, blocks, None, batch, pad_id, donor_cache).items()}
            toks = patched_generate(adapter, model, heads, dz, batch["base_ids"], batch["base_mask"],
                                    batch["base_extra"], args.max_new_tokens, pad_id, head_dim,
                                    transform=transform)
            for r, text in zip(chunk, decode(processor, toks)):
                f.write(json.dumps({"attribute": r["target_attribute"],
                                    "row_index": r["row_index"],
                                    "generated_text": text}) + "\n")
                n += 1
            f.flush()
            elapsed, eta, rate = progress(n, len(rows), t0)
            print(f"    {n}/{len(rows)} | {elapsed} eta {eta} ({rate:.2f}s/row)", flush=True)
    print(f"  eval done in {fmt_hms(time.time() - t0)}")
    return n


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attribute", default="language", choices=list(ATTRIBUTES),
                    help="One intervention is trained PER attribute -- a `language` rotation is a "
                         "different object from a `capital` one and must be.")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--method", default="das_fixed", choices=list(METHODS))
    ap.add_argument("--subspace_dim", default="8", metavar="K",
                    help="das_fixed only: the FIXED subspace width. One value for every block "
                         "(\"128\"), one PER block in sorted-block order (\"128,64,64\"), or "
                         "\"full\" for each block's own width -- the ceiling arm, which has zero "
                         "trainable degrees of freedom and is a measurement, not a run. Per-block "
                         "values matter because common10's heads are spread 2/4/4 over blocks "
                         "21/22/23, so a single K is a different FRACTION of each block's space. "
                         "das_rotated and dbm learn their own width via an annealed mask.")
    ap.add_argument("--heads", default=COMMON10,
                    help="BLOCK.HEAD list. Default is R8/R10's common10.")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--train_rows", type=int, default=0, metavar="N",
                    help="Cap training rows (0 = the whole split, ~33-36k). Subsampling is "
                         "stratified by cause/iso so a small N keeps both pools.")
    ap.add_argument("--eval_rows", type=int, default=0, metavar="N",
                    help="Cap eval rows (0 = the whole test split, 14,052). score.py EXCLUDES "
                         "missing rows rather than scoring them wrong, so a capped eval is a "
                         "valid partial score, not a penalized one.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask_lr", type=float, default=None, metavar="LR",
                    help="das_rotated/dbm only: separate Adam lr for the `masks` parameter. "
                         "Default (unset) reuses --lr, which is what produced the VOID R12.3 "
                         "runs: the mask enters the loss only as sigmoid(m/T), so its gradient "
                         "underflows to exactly 0 once m/T saturates, and Adam moves it ~lr per "
                         "step until then -- ~150,000 steps at 1e-3 to travel mask_init=150. The "
                         "run prints a travel budget and warns when the mask cannot move.")
    ap.add_argument("--mask_init", type=float, default=None, metavar="M",
                    help="das_rotated only: initial value of every mask logit (VADE default 150). "
                         "Only the RATIO mask_init/temperature matters to the sigmoid, so change "
                         "this together with --temperature_start.")
    ap.add_argument("--temperature_start", type=float, default=None)
    ap.add_argument("--temperature_end", type=float, default=None,
                    help="Override the method's anneal endpoints (das_rotated: 50 -> 0.1).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=16, metavar="G",
                    help="Batches accumulated per optimizer step. EFFECTIVE BATCH is "
                         "--batch_size * G: raise G to train at a large effective batch on a card "
                         "that only fits a small --batch_size, at no extra memory. A short final "
                         "group is divided by its own real size, not by G.")
    ap.add_argument("--log_every", type=int, default=1, metavar="N",
                    help="Print one line every N batches (0 = only on optimizer steps). Each line "
                         "carries that batch's own ce and its cause/iso split; the '>>' lines are "
                         "completed optimizer steps.")
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--iso_weight", type=float, default=1.0,
                    help="Multiplier on iso rows' loss. The train split is ~57%% cause but the "
                         "metric weights cause 1/2 and the iso mean 1/2; raise this if training "
                         "lands in the cause~100/iso~0 corner (i.e. relearned the full swap).")
    ap.add_argument("--l1_coef", type=float, default=1e-3, metavar="C",
                    help="dbm only: coefficient on ||m||_1 (RAVEL's reported optimum). 0 disables "
                         "the term, which lets the mask keep every dimension.")
    ap.add_argument("--mask_coef", type=float, default=1e-3, metavar="C",
                    help="das_rotated only: coefficient on sum(sigmoid(m/T)), the soft dimension "
                         "count -- Boundless DAS's boundary penalty. This is the ONLY thing pulling "
                         "the learned K down: the mask inits near a full swap (sigmoid(150/50) ~ "
                         "0.95 everywhere) and cause is free at this site, so at 0 the run will "
                         "simply reproduce R10's full head swap. At 1e-3 a fully-open mask costs "
                         "~1.28, comparable to the CE term; sweep it and read `maskK`.")
    ap.add_argument("--no_prompt_cache", action="store_true",
                    help="Rebuild every row's image preprocessing and tokenization from scratch. "
                         "~200x slower; use only to rule the cache out as a suspect.")
    ap.add_argument("--no_donor_cache", action="store_true",
                    help="Re-run the donor forward every batch instead of reusing it per "
                         "(source, queried, template_id). ~4.5x more donor compute.")
    ap.add_argument("--donor_capture", choices=["teacher_forced", "generate"], default="teacher_forced",
                    help="teacher_forced matches training exactly (1 forward); generate reproduces "
                         "head_swap_vade's R10 capture (K forwards).")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--skip_train", action="store_true",
                    help="Build the interventions and go straight to eval. Only meaningful where "
                         "the untrained module is already the thing you want to measure -- "
                         "`das_fixed --subspace_dim full`, which is the full head swap by "
                         "construction.")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    blocks = sorted({b for b, _ in heads})
    # The tag must encode every knob that changes the RESULT, or a sweep
    # overwrites itself into one directory -- the gotcha CLAUDE.md records for
    # select_features.py's --dictionaries_dir. das_fixed varies in K;
    # das_rotated and dbm learn their own width, so theirs is the coefficient
    # that decides how far it shrinks.
    if args.method == "das_fixed":
        suffix = f"_k{str(args.subspace_dim).replace(',', '-').strip()}"
    elif args.method == "das_rotated":
        # every knob that CHANGES the run goes in the tag -- R12.3's three arms
        # overwrote each other because only --mask_coef was encoded
        suffix = (f"_m{args.mask_coef:g}"
                  + (f"_mlr{args.mask_lr:g}" if args.mask_lr is not None else "")
                  + (f"_mi{args.mask_init:g}" if args.mask_init is not None else "")
                  + (f"_T{args.temperature_start:g}" if args.temperature_start is not None else ""))
    else:
        suffix = f"_l1{args.l1_coef:g}"
    tag = f"{args.attribute}_{args.method}{suffix}"
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "head_das", args.entity, tag)
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)

    train_rows = load_rows(args.vade_root, args.entity, args.attribute, args.train_split,
                           args.train_rows, args.seed)
    eval_rows = load_rows(args.vade_root, args.entity, args.attribute, args.eval_split,
                          args.eval_rows, args.seed + 1)
    n_cause = sum(r["rule"] == "match_source" for r in train_rows)
    print(f"[head_das] {args.entity}/{args.attribute} method={args.method}"
          + (f" K={args.subspace_dim}" if args.method == "das_fixed" else ""))
    print(f"  heads ({len(heads)}): " + ", ".join(f"{b}.{h}" for b, h in heads))
    print(f"  train {args.train_split}: {len(train_rows)} rows "
          f"({n_cause} cause / {len(train_rows) - n_cause} iso), iso_weight={args.iso_weight}")
    print(f"  eval  {args.eval_split}: {len(eval_rows)} rows")
    print(f"  effective batch: {args.batch_size} x {args.grad_accum_steps} accum = "
          f"{args.batch_size * args.grad_accum_steps} rows/optimizer step")
    print(f"  patch columns per forward: last {MAX_ANSWER_TOKENS} (answer tokens 0..{MAX_ANSWER_TOKENS - 1})")
    print(f"  -> {out_dir}")

    if args.dry_run:
        assert not (set(r["base"] for r in train_rows + eval_rows) - set(items)), "unknown base item"
        assert not (set(r["source"] for r in train_rows + eval_rows) - set(items)), "unknown source item"
        tr_items = {c for r in train_rows for c in (r["base"], r["source"])}
        ev_items = {c for r in eval_rows for c in (r["base"], r["source"])}
        assert n_cause and n_cause < len(train_rows), "train rows lost one of the two pools"
        print(f"    items: {len(tr_items)} train / {len(ev_items)} eval, "
              f"overlap {len(tr_items & ev_items)} -- VADE splits by ITEM, so eval is on flags "
              f"the rotation has never seen")
        assert not (tr_items & ev_items), (
            "train and eval share items -- the generalization claim below does not hold; "
            "check --train_split/--eval_split")
        widths = {b: 128 * sum(1 for x, _ in heads if x == b) for b in blocks}
        print(f"    blocks {blocks}; head columns per block {widths} (assuming head_dim=128)")
        if args.method == "das_fixed":
            dims = resolve_subspace_dims(args.subspace_dim, blocks, widths)
            print(f"    subspace per block {dims}; effective DOF "
                  f"{ {b: subspace_dof(k, widths[b]) for b, k in dims.items()} }")
        print("Grid valid.")
        return

    from methods.adapters.registry import get_adapter

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    model.requires_grad_(False)                # only the intervention trains
    hidden = adapter.hidden_size(model)
    n_heads = adapter.n_attention_heads(model)
    head_dim = hidden // n_heads
    n_layers = len(adapter.get_decoder_layers(model))
    assert all(0 <= b < n_layers for b in blocks), f"blocks must be in 0..{n_layers - 1}"
    assert all(0 <= h < n_heads for _, h in heads), f"head index out of 0..{n_heads - 1}"
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    colmap = {b: head_columns(heads, b, head_dim) for b in blocks}

    widths = {b: len(colmap[b]) for b in blocks}
    sub_dims = (resolve_subspace_dims(args.subspace_dim, blocks, widths)
                if args.method == "das_fixed" else {b: 0 for b in blocks})
    skip_train = args.skip_train
    if args.method == "das_fixed":
        dof = {b: subspace_dof(k, widths[b]) for b, k in sub_dims.items()}
        print(f"  subspace per block {sub_dims} of {widths}; effective DOF {dof}")
        dead = [b for b, d in dof.items() if d == 0]
        for b in dead:
            print(f"  WARNING: block {b} has K == its full width ({widths[b]}), so R^T R = I and "
                  f"this block is an UNTRAINABLE full swap whatever the optimizer does.")
        if dead and len(dead) < len(dof):
            print("  ^ this arm is MIXED: some blocks train, some are pinned to the full swap. "
                  "Its point on a K curve is not comparable to the arms where every block trains.")
        if len(dead) == len(dof) and not skip_train:
            print("  every block has zero degrees of freedom -- this is the CEILING arm, identical "
                  "to head_swap_vade's full head swap by construction. Skipping training: the "
                  "gradients are pure gauge and the optimizer steps would be a no-op.")
            skip_train = True

    os.makedirs(out_dir, exist_ok=True)
    interventions = build_interventions(args.method, colmap, sub_dims, args.vade_root,
                                        model.device, args.mask_init)
    if not skip_train:
        train(adapter, model, processor, args, heads, colmap, items, entity_dir,
              lookup, pad_id, os.path.join(out_dir, "train_log.jsonl"), interventions)

    import torch
    torch.save({b: iv.state_dict() for b, iv in interventions.items()},
               os.path.join(out_dir, "intervention.pt"))
    stats = intervention_stats(args.method, interventions)
    json.dump({"args": vars(args), "heads": [[b, h] for b, h in heads], "stats": stats},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print("  " + json.dumps(stats))

    if not args.skip_eval:
        out_path = os.path.join(out_dir, "predictions.jsonl")
        n = evaluate(adapter, model, processor, args, heads, colmap, items, entity_dir, lookup,
                     pad_id, head_dim, interventions, out_path)
        print(f"wrote {n} predictions -> {out_path}")
        print(f"score with:\n  python {os.path.join(args.vade_root, 'eval', 'score.py')} "
              f"--entity {args.entity} --attribute {args.attribute} --predictions {out_path}")


if __name__ == "__main__":
    main()
