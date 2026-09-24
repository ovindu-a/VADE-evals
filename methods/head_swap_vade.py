"""VADE-scored entity-head swap, with the patch held live across generation.

WHAT THIS IS

head_trace.py's phase-3 sufficiency arm, moved onto the benchmark. It installs a
chosen set of (block, head) outputs from a SOURCE image's run into a BASE
image's run and lets the model generate, then writes predictions in
VADE/eval/score.py's format so the result lands on `cause` / `iso` /
`final_score` instead of an internal flip rate.

The head set to use is R8 of ATTRIBUTE_HEAD_EXPERIMENTS.md: the five heads that
sit in all four flags attributes' top-8 of the blocks 21-23 image trace --

    21.1  22.19  23.3  23.4  23.6

which reach 100% cause on `language` at k=8 with a passing shuffled-donor
control. R9 measures those same heads as `item`-dominated (0.61-0.69 of their
variance is "which flag", against a 0.296 all-head baseline), i.e. an ENTITY
CONDUIT rather than an attribute-selective reader.

WHY CONTINUOUS PATCHING IS REQUIRED HERE, NOT OPTIONAL

head_trace patches only the last PROMPT column: common/hooks.py's
make_cache_aware_patch_hook returns the tensor untouched once generation
collapses to one column. That caps the intervention at the first answer token.
It is invisible on `language` (78.6% of its golds are one token) and fatal on
`calling_code` (0.0%, mean 2.68 tokens), which is why head_trace's per-attribute
ceilings rank-order exactly with single-token fraction -- 4 of 4 -- and mean
nothing as a statement about which attributes the window carries.

VADE's scorer matches a whole label anywhere in the generated text, so a
first-token-only flip produces NEITHER label: it scores as a miss on the cause
row AND as a false success on the iso rows. A last-token patch cannot be scored
by VADE honestly. So this script captures the source run's head outputs at EVERY
generated step and replays them step-by-step into the base run.

    donor pass   generate with the SOURCE image, recording z[block] at the last
                 column of every forward -> [B, n_steps, hidden]
    base pass    generate with the BASE image, and at step t overwrite the
                 chosen heads' 128-dim slices at the last column with the
                 donor's step-t values

The hook targets column -1 in both passes, which is correct for both: batches
are LEFT-padded so the last real prompt token is the final column, and a decode
step has exactly one column. That single rule is what makes prefill and decode
the same code path.

WHAT THE NUMBER MEANS -- READ THIS BEFORE INTERPRETING A RESULT

VADE builds one intervention per `target_attribute` and then asks ALL four
questions under it: the cause rows (queried == target) must flip to source, the
iso rows (queried != target) must stay at base. So:

  * An edit with NO attribute selectivity is pinned near final_score = 50%.
    It either transplants the country -- cause ~100%, iso ~0% -- or does nothing
    -- cause 0%, iso ~100%. Both average to 50%. Beating 50% REQUIRES the edit to
    move the queried attribute more than the un-queried ones.
  * R9 predicts this head set does NOT beat it: an item-dominated conduit moves
    every attribute together. A ~50% final_score here is the expected result and
    a real finding about where disentanglement can be done -- not a failed run.
    The informative quantities are the CORNER it lands in (cause vs iso) and the
    distance from 50%, not the 50% itself.

`--donor_question` chooses which of the two questions the donor is captured
under, and they are different experiments:

    queried (default)  the donor runs the SAME question being asked. A pure
                       entity edit that knows nothing about target_attribute.
                       Every target_attribute file gets the same generation, so
                       the whole 56,208-row test set costs 14,052 generations.
                       Measures how completely these heads transplant the entity.
    target             the donor runs the TARGET attribute's question, and that
                       one frozen vector is installed for all four queries. The
                       only arm that CAN beat 50%, because the installed value is
                       now conditioned on the target attribute. 4x the cost.

ARMS (all share one donor pass, so the controls are nearly free)

    clean         no patch. cause ~0 / iso ~100. Confirms the floor and that the
                  scorer is wired up.
    heads         the experiment.
    random_heads  as many random heads from the same blocks, drawn ONCE and
                  reused for every batch. If this moves, the result is about
                  perturbing the site, not about these heads.
    full_image    the source's residual at every image token at --patch_layer.
                  The unselective ceiling: what "the whole country changed"
                  looks like on this metric, in this prompt set.

Usage
-----
    # Sizing and grid validation. No torch, no model.
    python methods/head_swap_vade.py --dry_run

    # Smoke test: 20 pairs, all arms.
    python methods/head_swap_vade.py --limit_pairs 20 --out_dir results/head_swap_vade/smoke

    # The real run: entire flags test split, the two attribute-independent sets.
    python methods/head_swap_vade.py --batch_size 16

    # ALSO the per-attribute top-8 and top-16 straight off the traces. Four head
    # sets, ONE donor capture -- they all live in blocks 21-23.
    T='logs/Qwen2.5-VL-7B-Instruct/{entity}/ndm/{attribute}/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json'
    python methods/head_swap_vade.py --batch_size 16 \\
        --head_sets "common5=21.1,22.19,23.3,23.4,23.6" "top8=$T#8" "top16=$T#16"

    # The arm that can beat the null.
    python methods/head_swap_vade.py --donor_question target --arms clean heads

    # Score (sibling VADE repo). One file holds all four target attributes.
    python ../VADE/eval/score.py --entity flags --attribute all \\
        --predictions results/head_swap_vade/flags/heads.jsonl
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# R8's two attribute-INDEPENDENT sets, from the blocks 21-23 image trace.
COMMON5 = "21.1,22.19,23.3,23.4,23.6"                      # in all four attributes' top-8
COMMON10 = "21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"   # ... top-16
DEFAULT_SETS = [f"common5={COMMON5}", f"common10={COMMON10}"]
ATTRIBUTES = ("capital", "currency", "language", "calling_code")

# Where head_trace.py's valid per-attribute traces live. '{attribute}' in a
# --head_sets path expands to each of ATTRIBUTES, giving a per-attribute set.
TRACE_GLOB = ("logs/Qwen2.5-VL-7B-Instruct/{entity}/ndm/{attribute}/"
              "L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/"
              "head_trace_patch21_blocks21-23.json")


def parse_heads(spec):
    """'21.1,22.19' -> [(21, 1), (22, 19)]. Rejects anything else loudly: a
    typo'd head silently becomes a different head, and every number downstream
    still looks plausible."""
    out = []
    for tok in str(spec).replace(" ", ",").split(","):
        if not tok:
            continue
        assert tok.count(".") == 1, f"head spec {tok!r} must be BLOCK.HEAD, e.g. 23.4"
        b, h = tok.split(".")
        out.append((int(b), int(h)))
    assert out, "no heads parsed"
    assert len(set(out)) == len(out), f"duplicate heads in {spec!r}"
    return sorted(out)


def heads_from_trace(path, top_k):
    """The top-k of a head_trace.py run's phase-1 ranking.

    REFUSES a trace whose ranking is inert. Two of the 55 traces on disk predate
    commit 518cf31 (BuildBatchCache returning another spec's positions): they
    ranked heads traced at IMAGE columns while every printed line said last
    token, so the ranking points at heads the readout never reads.

    The discriminator is whether patching EVERY traced head DISLODGED the base
    answer -- not whether it transferred the source's. Those are different
    failures and only the first one indicts the ranking:

      void trace          cause 1.6%,  base_kept 95.3%  -> patch did nothing
      valid calling_code  cause 6.2%,  base_kept 26.6%  -> patch destroyed the
                          answer without transferring; that is the ANSWER-LENGTH
                          artefact (0% of its golds are single-token and
                          head_trace scores full matches), a live site whose
                          ceiling is capped by the metric
      valid language      cause 98.4%, base_kept  0.0%  -> full transfer

    Scoring on `cause` alone would reject calling_code's perfectly good trace."""
    d = json.load(open(path))
    p3 = d.get("phase3_sufficiency") or {}
    all_arm = next((a for a in p3.get("arms", []) if a.get("kind") == "all"), None)
    if all_arm is not None and all_arm.get("base_kept") is not None:
        moved = 1.0 - float(all_arm["base_kept"])
        if moved < 0.25:
            raise SystemExit(
                f"REFUSING {path}: patching ALL {all_arm.get('k')} traced heads left "
                f"{all_arm['base_kept']:.1%} of base answers intact (cause {all_arm.get('cause', 0):.1%}). "
                f"The ranking is inert -- the signature of a pre-518cf31 trace "
                f"(see ATTRIBUTE_HEAD_EXPERIMENTS.md R5/R8). Use a blocks21-23 trace.")
    return [tuple(h) for h in d["phase1_ranked"][:top_k]]


def build_value_subspace(capture_dir, heads, attribute, dim, vade_root, entity):
    """-> {block: [n_cols_b, dim]} orthonormal, spanning the centroids of
    `attribute`'s VALUES in that block's selected-head columns.

    BLOCK-DIAGONAL by necessity, not by preference: patch hooks fire per block in
    forward order, so at block 21 the base's block-22 activations do not exist
    yet and a subspace mixing the two cannot be applied during a forward pass.
    Measured cost of the restriction: none worth worrying about -- per-block
    7-dim subspaces decode language at 93.9-100%, and their union at 100%.

    Only an attribute whose values REPEAT across items defines a value centroid
    at all; with one item per value the centroids are the items and the subspace
    is just an entity subspace under another name. On flags that means
    `language` (8 classes over 49 countries) and nothing else."""
    import numpy as np
    import torch
    meta = json.load(open(os.path.join(capture_dir, "meta.json")))
    rows = [json.loads(l) for l in open(os.path.join(capture_dir, "index.jsonl"))]
    attrs, blocks_cap, head_dim = meta["attributes"], meta["blocks"], meta["head_dim"]
    acts = np.memmap(os.path.join(capture_dir, "acts_attn_head_output.npy"),
                     dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
    gt = json.load(open(os.path.join(vade_root, "data", entity, "ground_truth.json")))
    truth = gt["countries"] if "countries" in gt else gt["items"]

    items = sorted({r["item"] for r in rows})
    idx = {(r["item"], r["attribute"]): r["row"] for r in rows}
    value = {c: truth[c][attribute] for c in items}
    from collections import Counter
    counts = Counter(value.values())
    keep = [c for c in items if counts[value[c]] >= 2]
    classes = sorted({value[c] for c in keep})
    assert len(classes) >= 2, (
        f"--subspace_attribute {attribute!r} has {len(classes)} value(s) repeated across items; "
        f"its centroids would BE the items, making this an entity subspace, not an attribute one.")

    out = {}
    for b in sorted({blk for blk, _ in heads}):
        hs = [h for blk, h in heads if blk == b]
        bi = blocks_cap.index(b)
        Z = np.array([[np.concatenate([acts[idx[(c, a)], bi, 0, h * head_dim:(h + 1) * head_dim]
                                       for h in hs]) for a in attrs] for c in keep], dtype=np.float64)
        Zc = Z - Z.mean((0, 1))
        cent = np.array([Zc[[i for i, c in enumerate(keep) if value[c] == L]].mean((0, 1))
                         for L in classes])
        U, _, _ = np.linalg.svd((cent - cent.mean(0)).T, full_matrices=False)
        k = min(dim, len(classes) - 1, U.shape[1])
        out[b] = torch.from_numpy(np.ascontiguousarray(U[:, :k]))
    return out, len(classes), len(keep)


def resolve_head_sets(specs, entity, top_k):
    """['name=21.1,23.4', 'name=path/to/trace.json#16'] -> {name: heads}.

    `heads` is either a flat list, or a {attribute: list} dict when the path
    contained '{attribute}' -- a PER-ATTRIBUTE set, keyed by the attribute the
    donor was captured under. Several sets run in ONE invocation because they
    all live in blocks 21-23 and therefore share a single donor capture: adding
    k=8 and k=16 costs extra generations, not extra donor passes."""
    out = {}
    for spec in specs:
        assert "=" in spec, f"--head_sets entry {spec!r} must be NAME=SPEC"
        name, body = spec.split("=", 1)
        assert name not in out, f"duplicate head-set name {name!r}"
        if ".json" not in body:
            out[name] = parse_heads(body)
            continue
        path, _, k = body.partition("#")
        k = int(k) if k else top_k
        if "{attribute}" in path:
            out[name] = {a: heads_from_trace(path.format(entity=entity, attribute=a), k)
                         for a in ATTRIBUTES}
        else:
            out[name] = heads_from_trace(path.format(entity=entity), k)
    return out


def heads_for(head_set, attribute):
    """A per-attribute set is keyed by the attribute the DONOR was captured
    under -- `queried` under donor_question='queried', `target_attribute` under
    'target'. One rule, so the head set always matches the question that
    produced the values being installed."""
    return head_set[attribute] if isinstance(head_set, dict) else head_set


def all_blocks(head_sets):
    """Union over every set. The donor capture must cover all of them or a set
    would silently index a block that was never recorded."""
    blocks = set()
    for hs in head_sets.values():
        for heads in (hs.values() if isinstance(hs, dict) else [hs]):
            blocks |= {b for b, _ in heads}
    return sorted(blocks)


# ---------------------------------------------------------------------------
# VADE assets and the job grid
# ---------------------------------------------------------------------------

def entity_items(gt):
    """The item dict out of a VADE ground_truth.json, whatever it is called.

    Each entity names its own collection -- flags `countries`, brands `brands`,
    animals `species` -- so a hardcoded key silently restricts every script here
    to flags. Identify it by SHAPE instead: the one top-level dict whose values
    are dicts carrying an `image`. Fails loudly rather than guessing, because a
    wrong pick would produce plausible-looking runs on the wrong objects."""
    cand = [k for k, v in gt.items()
            if isinstance(v, dict) and v and all(isinstance(x, dict) and "image" in x
                                                 for x in v.values())]
    assert len(cand) == 1, (
        f"ground_truth.json must have exactly one item collection (a dict of dicts with "
        f"'image'); found {cand or 'none'} among {list(gt)}")
    return gt[cand[0]]


def entity_attributes(gt):
    """The entity's scored attributes, in the order it declares them.

    NOT the keys of prompt_templates.json -- that also carries
    `recognition_freeform`, which is not a scored attribute."""
    attrs = gt.get("attributes")
    assert isinstance(attrs, list) and attrs, f"ground_truth.json has no `attributes` list"
    return tuple(attrs)


def load_assets(vade_root, entity):
    entity_dir = os.path.join(vade_root, "data", entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    items = entity_items(gt)
    templates = json.load(open(os.path.join(entity_dir, "prompt_templates.json")))
    lookup = {attr: {t["template_id"]: t for t in tl} for attr, tl in templates.items()}
    return entity_dir, items, lookup


def load_jobs(vade_root, entity, split, attributes, donor_question, limit_pairs, seed):
    """-> (jobs, n_scored_rows).

    A job is ONE generation plus the list of (target_attribute, row_index) rows
    it answers. With donor_question='queried' the edit does not depend on
    target_attribute, so a single generation serves one row in each of the four
    target files -- 56,208 scored rows collapse to 14,052 generations. With
    'target' the donor differs per target and the collapse does not apply."""
    tuples_dir = os.path.join(vade_root, "data", entity, "tuples")
    by_key = {}
    n_rows = 0
    for target in attributes:
        path = os.path.join(tuples_dir, target, f"{split}.jsonl")
        for line in open(path):
            r = json.loads(line)
            n_rows += 1
            key = (r["base"], r["source"], r["queried"], r["template_id"])
            if donor_question == "target":
                key += (r["target_attribute"],)
            job = by_key.setdefault(key, {
                "base": r["base"], "source": r["source"], "queried": r["queried"],
                "template_id": r["template_id"],
                # Which question the DONOR prompt asks. Under 'queried' this is the
                # same prompt as the base run; under 'target' it is the attribute the
                # VADE row is targeting, so one frozen read serves all four queries.
                "donor_attribute": r["target_attribute"] if donor_question == "target" else r["queried"],
                "donor_template_id": r["template_id"] if donor_question == "queried" else None,
                "rows": [],
            })
            job["rows"].append((r["target_attribute"], r["row_index"]))

    jobs = list(by_key.values())
    if donor_question == "target":
        # The donor asks a different attribute's question, so it needs one of THAT
        # attribute's templates. Use the same version index so the pairing is a
        # deterministic function of the row, not of iteration order.
        for j in jobs:
            j["donor_template_id"] = None  # resolved at build time from the version suffix

    jobs.sort(key=lambda j: (j["base"], j["source"], j["queried"], j["template_id"],
                             j.get("donor_attribute") or ""))
    if limit_pairs:
        pairs = sorted({(j["base"], j["source"]) for j in jobs})
        keep = set(random.Random(seed).sample(pairs, min(limit_pairs, len(pairs))))
        jobs = [j for j in jobs if (j["base"], j["source"]) in keep]
    return jobs, n_rows


def donor_template(lookup, job):
    """The donor prompt's template. Same one as the base under
    donor_question='queried'; otherwise the matching version of the target
    attribute's template list."""
    if job["donor_template_id"]:
        return lookup[job["donor_attribute"]][job["donor_template_id"]]
    attr = job["donor_attribute"]
    suffix = job["template_id"].rsplit("_", 1)[-1]          # 'v3'
    by_suffix = {tid.rsplit("_", 1)[-1]: t for tid, t in lookup[attr].items()}
    return by_suffix.get(suffix) or next(iter(lookup[attr].values()))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def build_prompt(processor, image, question, prefill):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt")


def build_batch(jobs, processor, items, entity_dir, lookup, pad_id):
    """Base and donor prompts for one chunk, each LEFT-padded to its own max.

    Left padding is what lets every hook in this file address column -1: the
    last real prompt token is the final column for every row regardless of
    length, and a decode step has one column. Base and donor are padded
    independently because under donor_question='target' they are different
    questions and genuinely differ in length."""
    import torch
    from PIL import Image

    base_seqs, donor_seqs, base_px, donor_px, base_thw, donor_thw = [], [], [], [], [], []
    for j in jobs:
        bt = lookup[j["queried"]][j["template_id"]]
        dt = donor_template(lookup, j)
        with Image.open(os.path.join(entity_dir, items[j["base"]]["image"])) as im:
            bp = build_prompt(processor, im.convert("RGB"), bt["question"], bt["prefill"])
        with Image.open(os.path.join(entity_dir, items[j["source"]]["image"])) as im:
            dp = build_prompt(processor, im.convert("RGB"), dt["question"], dt["prefill"])
        base_seqs.append(bp["input_ids"][0]); donor_seqs.append(dp["input_ids"][0])
        base_px.append(bp["pixel_values"]); donor_px.append(dp["pixel_values"])
        base_thw.append(bp["image_grid_thw"]); donor_thw.append(dp["image_grid_thw"])

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
    return {
        "jobs": jobs,
        "base_ids": base_ids, "base_mask": base_mask,
        "base_extra": {"pixel_values": torch.cat(base_px), "image_grid_thw": torch.cat(base_thw)},
        "donor_ids": donor_ids, "donor_mask": donor_mask,
        "donor_extra": {"pixel_values": torch.cat(donor_px), "image_grid_thw": torch.cat(donor_thw)},
    }


# ---------------------------------------------------------------------------
# The continuous head capture / patch
# ---------------------------------------------------------------------------

def head_columns(heads, block, head_dim):
    """Column indices of `block`'s heads inside o_proj's 28*128 input."""
    import torch
    cols = []
    for b, h in heads:
        if b == block:
            cols.extend(range(h * head_dim, (h + 1) * head_dim))
    return torch.tensor(sorted(cols), dtype=torch.long)


# Where the continuous capture/patch lives. Both are the INPUT of a bias-free
# linear map, read and written by a forward pre-hook, so everything below is the
# same code for either:
#   attn_head_output  o_proj's input, 28 heads x 128 (the default; R10/R11)
#   mlp_hidden        down_proj's input, the 18944 post-SwiGLU neurons
# For mlp_hidden a "head" is a whole block: pass units = [(block, 0), ...] and
# unit_dim = intermediate_size (see `site_units`), and head_columns then selects
# every neuron of that block.
UNIT_SITES = ("attn_head_output", "mlp_hidden")


def site_module(adapter, model, block, site="attn_head_output"):
    if site == "attn_head_output":
        return adapter.get_attn_head_output_module(model, block)
    if site == "mlp_hidden":
        return adapter.get_mlp_hidden_module(model, block)
    raise ValueError(f"unknown site {site!r}; expected one of {UNIT_SITES}")


def site_units(adapter, model, site, heads=None, blocks=None):
    """-> (units, unit_dim): what to pass as `heads` / `head_dim` for `site`."""
    if site == "attn_head_output":
        assert heads, "attn_head_output needs a head list"
        return heads, adapter.hidden_size(model) // adapter.n_attention_heads(model)
    assert blocks, "mlp_hidden needs a block list"
    return [(b, 0) for b in sorted(blocks)], adapter.intermediate_size(model)


def capture_donor(adapter, model, blocks, ids, mask, extra, max_new_tokens, pad_id, site="attn_head_output"):
    """-> {block: [B, n_steps, width]} of `site` (default attn_head_output) at
    the LAST column of every forward of the donor's own generation.

    Step 0 is the prefill (the readout column); step t>0 is generated token t.
    One pre-hook per block on o_proj, each appending its own column -- a block's
    hook fires exactly once per forward, so list position IS the step index."""
    import torch
    sinks = {b: [] for b in blocks}
    handles = []
    for b in blocks:
        module = site_module(adapter, model, b, site)

        def grab(_mod, args, _sink=sinks[b]):
            _sink.append(args[0][:, -1, :].detach().float().cpu())
        handles.append(module.register_forward_pre_hook(grab))
    try:
        with torch.no_grad():
            model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                           **to_device(extra, model.device, model.dtype),
                           max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    finally:
        for h in handles:
            h.remove()
    return {b: torch.stack(v, dim=1) for b, v in sinks.items()}      # [B, steps, hidden]


ALIGN_MODES = ("matched", "hold0", "step0")


def donor_index(align, step, n_steps):
    """Which donor step to install at base step `step`; None = leave the base alone.

    `matched` is the R10/R11 intervention -- donor step t -> base step t, holding
    the donor's last step if the base outlasts it. The other two exist to LOCALIZE
    a multi-token effect to step 0 versus the steps after it. A step-matched replay
    from a donor that was asked a DIFFERENT question installs, at t>=1, that donor's
    state while emitting ITS OWN token t, which is off-manifold for the answer the
    base is producing -- the calling_code split in R11.2b, where `both` emits the
    source's first digit 98.5% of the time and then drifts.

      matched   idx = min(t, T-1)           step-for-step replay
      hold0     idx = 0                     the donor's readout column, held
      step0     idx = 0 at t=0, else None   patch once, then free-run

    All three are IDENTICAL at t=0 by construction, so the FIRST generated token
    must agree across modes for any given row -- a free self-test on the plumbing.
    `step0` is the falsifier for the alignment story: it makes `flag` and `both`
    differ only in the donor's prompt, with no answer token in either donor's
    context yet, so the two cells must converge if the gap really lives in t>=1."""
    if align == "matched":
        return min(step, n_steps - 1)
    if align == "hold0":
        return 0
    if align == "step0":
        return 0 if step == 0 else None
    raise ValueError(f"unknown align mode {align!r}; expected one of {ALIGN_MODES}")


def patched_generate(adapter, model, heads, donor_z, ids, mask, extra, max_new_tokens, pad_id,
                     head_dim, extra_patches=(), stats=None, observers=(), subspace=None,
                     transform=None, align="matched", site="attn_head_output"):
    """Greedy generation with `heads` overwritten at the last column of EVERY
    forward from the donor step `align` selects (see `donor_index`).

    Under the default `matched`, if the base run outlasts the donor's recorded
    steps (the donor hit EOS earlier), the last donor step is held and the event
    is counted rather than silently indexed out of range. That counter is
    meaningless under the other modes, which never index past step 0, so it is
    only kept for `matched`.

    `transform` generalizes what gets written: {block: fn(base_slice, donor_slice)
    -> new_slice}, each [B, 1, n_cols_b]. None (the default) writes the donor's
    values verbatim -- the R10 intervention. `subspace` is the fixed-projection
    special case. A trained intervention (head_das.py) passes its own module
    here rather than forking this function, so the generation path that produced
    R10 and R11 is the one every later result is measured on. Mutually exclusive
    with `subspace`.

    `site` (see UNIT_SITES) moves the whole patch from o_proj's input to
    down_proj's; `heads`/`head_dim` then come from `site_units`."""
    assert transform is None or subspace is None, "pass `transform` or `subspace`, not both"
    import torch
    blocks = sorted({b for b, _ in heads})
    counters = {b: 0 for b in blocks}
    handles = []
    for extra_site, layer_idx, fn in extra_patches:     # NOT `site`: that names this call's unit site
        handles.extend(extra_site.register(adapter, model, adapter.get_decoder_layers(model), layer_idx, fn))
    for b in blocks:
        cols = head_columns(heads, b, head_dim).to(model.device)
        z = donor_z[b].to(model.device)                              # [B, steps, hidden]
        P = None if subspace is None else subspace[b].to(model.device).float()
        fn = None if transform is None else transform[b]

        def patch(_mod, args, _b=b, _cols=cols, _z=z, _P=P, _fn=fn):
            t = args[0]
            step = counters[_b]
            counters[_b] += 1
            idx = donor_index(align, step, _z.shape[1])
            if idx is None:
                return None            # a pre-hook returning None leaves the input untouched,
                                       # which is exactly "free-run from here" for `step0`
            if stats is not None and align == "matched" and step >= _z.shape[1]:
                stats["steps_beyond_donor"] = stats.get("steps_beyond_donor", 0) + 1
            patched = t.clone()
            want = _z[:, idx, :].index_select(-1, _cols).to(t.dtype)
            if _fn is not None:
                # Same [B, 1, n_cols] shape the training hook sees, so a module
                # trained on teacher-forced columns is applied identically here.
                have = t[:, -1:, :].index_select(-1, _cols)
                patched[:, -1:, _cols] = _fn(have, want.unsqueeze(1)).to(t.dtype)
            elif _P is None:
                patched[:, -1, _cols] = want
            else:
                # Replace ONLY the component inside the subspace, leaving the
                # orthogonal complement as the base produced it. P is orthonormal,
                # so this is base + P (P^T donor - P^T base).
                have = t[:, -1, _cols]
                shift = ((want.float() - have.float()) @ _P) @ _P.T
                patched[:, -1, _cols] = (have.float() + shift).to(t.dtype)
            return (patched,) + tuple(args[1:])
        handles.append(site_module(adapter, model, b, site)
                       .register_forward_pre_hook(patch, with_kwargs=False))
    # Registered LAST on purpose. PyTorch runs a module's forward pre-hooks in
    # registration order, so an observer added before the patch reads the very
    # tensor the patch is about to overwrite -- head_trace.py documents the same
    # trap for its phase-1c read-back.
    for b, fn in observers:
        def watch(_m, args, _f=fn):
            _f(args[0])
            return None            # a pre-hook's return REPLACES the input; an observer
                                   # that happens to return something must not corrupt it
        handles.append(site_module(adapter, model, b, site).register_forward_pre_hook(watch))
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                 **to_device(extra, model.device, model.dtype),
                                 max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    finally:
        for h in handles:
            h.remove()
    return out[:, ids.shape[1]:].cpu()


def verify_readback(adapter, model, heads, donor_z, ids, mask, extra, max_new_tokens, pad_id,
                    head_dim, tol=1e-3, rel_tol=0.01, subspace=None, align="matched",
                    site="attn_head_output"):
    """Read the patched site back through a hook registered AFTER the patch and
    assert it returns what was installed, at EVERY step.

    This is head_trace.py's phase-1c applied to the continuous case, and it is
    the check that licenses the rest: if the patch and the read disagree, the
    hook is not addressing o_proj's input at the column it claims and every
    benchmark number downstream is void. Costs one generation.

    Tolerance is `max(tol, rel_tol * |want|)`, not a bare absolute `tol`: the
    subspace path (`--subspace_dim`) adds a `have.float() + shift` float32
    accumulation before the final downcast to the model's bf16 dtype, so its
    round-trip error scales with the activation's own magnitude rather than
    sitting at a fixed floor. Measured on flags/language common10 sub7: worst
    absolute error 4e-3 at |want|_max 2.2, i.e. ~0.18% relative -- consistent
    with bf16's ~0.4% epsilon (dla.py's docstring notes the same ~1% floor for
    bf16 accumulation), not an addressing bug. An addressing bug produces `g`
    and `w` reading DIFFERENT tensors, which lands at O(1) relative error, so
    1% still catches that case."""
    import torch
    seen = {b: [] for b in sorted({b for b, _ in heads})}
    observers = [(b, (lambda t, _b=b: seen[_b].append(t[:, -1, :].detach().float().cpu())))
                 for b in seen]
    patched_generate(adapter, model, heads, donor_z, ids, mask, extra, max_new_tokens, pad_id,
                     head_dim, observers=observers, subspace=subspace, align=align, site=site)
    worst, worst_allowed = 0.0, tol
    for b, steps in seen.items():
        cols = head_columns(heads, b, head_dim)
        for t, got in enumerate(steps):
            idx = donor_index(align, t, donor_z[b].shape[1])
            if idx is None:
                continue               # nothing was installed at this step, so there is
                                       # nothing to read back -- `step0` past t=0
            want = donor_z[b][:, idx, :]
            g, w = got[:, cols], want[:, cols]
            if subspace is not None:
                # Only the IN-SUBSPACE component was installed, so only it must
                # read back. Checking the full vector would fail by construction.
                P = subspace[b].cpu().to(g.dtype)
                g, w = g @ P, w @ P
            err = float((g - w).abs().max())
            allowed = max(tol, rel_tol * float(w.abs().max()))
            if err - allowed > worst - worst_allowed:
                worst, worst_allowed = err, allowed
    assert worst <= worst_allowed, (
        f"read-back mismatch {worst:.3e} > {worst_allowed:.3e} (tol={tol}, rel_tol={rel_tol}): the patch "
        f"and the capture are not addressing the same tensor/column -- every result below would be void")
    return worst


def plain_generate(model, ids, mask, extra, max_new_tokens, pad_id):
    import torch
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                             **to_device(extra, model.device, model.dtype),
                             max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    return out[:, ids.shape[1]:].cpu()


def to_device(extra, device, dtype):
    import torch
    out = {}
    for k, v in extra.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    return out


def full_image_patch(adapter, model, batch, patch_layer, image_token_id):
    """The source residual at EVERY image token -- the unselective ceiling arm.

    Patches only during prefill (image columns do not exist in a decode step),
    which is exactly right: an image-position edit is baked into the KV cache
    for every downstream position, which is why it does NOT need the continuous
    treatment the head patch does."""
    import torch
    from methods.common.sites import InterventionSite
    from methods.common.hooks import make_cache_aware_patch_hook
    site = InterventionSite("residual")
    cols = (batch["donor_ids"] == image_token_id)
    counts = cols.sum(dim=1)
    assert int(counts.min()) > 0, "no image tokens found; wrong image_token_id"
    assert int(counts.min()) == int(counts.max()), "rows disagree on image-token count"
    base_cols = (batch["base_ids"] == image_token_id)
    assert torch.equal(base_cols.sum(1), counts), "base and donor disagree on image-token count"
    donor_pos = torch.stack([torch.nonzero(r, as_tuple=True)[0] for r in cols])
    base_pos = torch.stack([torch.nonzero(r, as_tuple=True)[0] for r in base_cols])
    src = site.capture(adapter, model, patch_layer, batch["donor_ids"], batch["donor_mask"],
                       to_device(batch["donor_extra"], model.device, model.dtype), donor_pos)
    fn = make_cache_aware_patch_hook(base_pos, lambda base_vals: src.to(base_vals.dtype))
    return [(site, patch_layer, fn)]


# ---------------------------------------------------------------------------

def decode(processor, toks):
    return [processor.tokenizer.decode(t, skip_special_tokens=True) for t in toks]


def batches_by_donor_attribute(jobs, batch_size):
    """Chunk within one donor_attribute at a time.

    Required, not cosmetic: a PER-ATTRIBUTE head set installs a different mask
    per attribute, and the hook applies one mask to the whole batch. Grouping
    makes a mixed batch impossible rather than silently patching one attribute's
    heads with another's."""
    groups = {}
    for j in jobs:
        groups.setdefault(j["donor_attribute"], []).append(j)
    for attribute in sorted(groups):
        rows = groups[attribute]
        for start in range(0, len(rows), batch_size):
            yield attribute, rows[start:start + batch_size]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--head_sets", nargs="+", default=DEFAULT_SETS, metavar="NAME=SPEC",
                    help="One or more named head sets, each scored as its own arm and its own "
                         "predictions file. SPEC is either a BLOCK.HEAD list (21.1,23.4) or a "
                         "head_trace.py JSON path with an optional '#k' for its top-k. A path "
                         "containing '{attribute}' expands per attribute, giving a per-attribute "
                         "set. Every set shares ONE donor capture, so extra sets cost generations "
                         f"only. Default: {' '.join(DEFAULT_SETS)}")
    ap.add_argument("--top_k", type=int, default=8, help="Default k for a trace SPEC with no '#k'.")
    ap.add_argument("--subspace_dim", type=int, default=0, metavar="K",
                    help="If >0, patch only the K-dimensional value-centroid subspace of "
                         "--subspace_attribute inside the selected heads, instead of the whole head. "
                         "This is the DAS hypothesis without the training: does a subspace that "
                         "carries the attribute move it WITHOUT dragging the entity along? K is "
                         "per block (the patch is block-diagonal by necessity) and is capped at "
                         "n_values-1. Arms are renamed sub<K>_<set>.")
    ap.add_argument("--subspace_attribute", default="language",
                    help="Which attribute's value centroids define the subspace. Needs values that "
                         "REPEAT across items; on flags only `language` qualifies.")
    ap.add_argument("--capture_dir", default=os.path.join(REPO_ROOT, "results", "attr_capture",
                                                          "flags", "blocks15-27_n84"),
                    help="attr_capture grid the subspace is estimated from.")
    ap.add_argument("--donor_question", choices=["queried", "target"], default="queried",
                    help="Which question the donor runs. 'queried' is the attribute-agnostic entity "
                         "edit (4x cheaper, pinned near 50%% by construction); 'target' freezes the "
                         "target attribute's read and is the only arm that can beat the null.")
    ap.add_argument("--arms", nargs="+", default=["clean", "heads", "random_heads", "full_image"],
                    choices=["clean", "heads", "random_heads", "full_image"],
                    help="'heads' and 'random_heads' each expand to one arm PER head set.")
    ap.add_argument("--patch_layer", type=int, default=21, help="full_image arm only.")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit_pairs", type=int, default=0, help="Subsample distinct (base, source) "
                                                                "pairs. score.py EXCLUDES missing rows "
                                                                "rather than scoring them wrong.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    head_sets = resolve_head_sets(args.head_sets, args.entity, args.top_k)
    blocks = all_blocks(head_sets)
    subspaces, sub_tag = {}, ""
    if args.subspace_dim:
        sub_tag = f"sub{args.subspace_dim}_"
        for name, hs in head_sets.items():
            assert not isinstance(hs, dict), (
                f"--subspace_dim with the per-attribute set {name!r}: the subspace is estimated once "
                f"from a fixed column layout, so the head set must not vary by attribute.")
            subspaces[name], n_val, n_it = build_value_subspace(
                args.capture_dir, hs, args.subspace_attribute, args.subspace_dim,
                args.vade_root, args.entity)
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "head_swap_vade", args.entity)

    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    jobs, n_scored = load_jobs(args.vade_root, args.entity, args.split, ATTRIBUTES,
                               args.donor_question, args.limit_pairs, args.seed)
    n_emitted = sum(len(j["rows"]) for j in jobs)

    arm_names = []
    for arm in args.arms:
        arm_names += ([f"{sub_tag}{arm}_{n}" for n in head_sets]
                      if arm in ("heads", "random_heads") else [arm])

    print(f"[head_swap_vade] {args.entity}/{args.split}: {len(jobs)} generations -> {n_emitted} scored "
          f"rows (full split is {n_scored}); donor_question={args.donor_question}")
    for name, hs in head_sets.items():
        if isinstance(hs, dict):
            sizes = {a: len(v) for a, v in hs.items()}
            print(f"  {name}: per-attribute, {sizes}")
            for a, v in sorted(hs.items()):
                print(f"      {a:13} " + ", ".join(f"{b}.{h}" for b, h in v))
        else:
            print(f"  {name} ({len(hs)}): " + ", ".join(f"{b}.{h}" for b, h in hs))
    if args.subspace_dim:
        for name, sp in subspaces.items():
            dims = {b: int(P.shape[1]) for b, P in sp.items()}
            print(f"  subspace [{name}]: {args.subspace_attribute} value-centroids, dims per block "
                  f"{dims} of {{b: P.shape[0] for b, P in sp.items()}} columns"
                  .replace("{b: P.shape[0] for b, P in sp.items()}",
                           str({b: int(P.shape[0]) for b, P in sp.items()})))
    print(f"  blocks captured: {blocks}")
    print(f"  arms ({len(arm_names)}): {arm_names}")
    print(f"  {len(jobs)} donor passes + {len(jobs) * len(arm_names)} scored generations "
          f"at batch {args.batch_size}")
    print(f"  -> {out_dir}")
    if args.dry_run:
        missing = [j for j in jobs if j["base"] not in items or j["source"] not in items]
        assert not missing, f"{len(missing)} jobs reference an unknown item, e.g. {missing[0]}"
        sizes = [len(c) for _, c in batches_by_donor_attribute(jobs, args.batch_size)]
        print(f"    {len(sizes)} batches, none mixing donor attributes (max {max(sizes)} rows)")
        for j in jobs[:3]:
            bt = lookup[j["queried"]][j["template_id"]]
            print(f"    {j['base']}->{j['source']} q={j['queried']} donor_q={j['donor_attribute']} "
                  f"| {bt['prefill']!r} | serves {len(j['rows'])} rows")
        print("Grid valid.")
        return

    from methods.adapters.registry import get_adapter

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    hidden = adapter.hidden_size(model)
    n_heads = adapter.n_attention_heads(model)
    assert hidden % n_heads == 0
    head_dim = hidden // n_heads
    n_layers = len(adapter.get_decoder_layers(model))
    assert all(0 <= b < n_layers for b in blocks), f"blocks must be in 0..{n_layers - 1}"
    for name, hs in head_sets.items():
        for heads in (hs.values() if isinstance(hs, dict) else [hs]):
            assert all(0 <= h < n_heads for _, h in heads), f"{name}: head index out of 0..{n_heads - 1}"
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    image_token_id = adapter.image_token_id(model, processor)

    # One null PER head set, matched in size: 16 random heads is a bigger
    # perturbation than 5, so a single shared null would under-control the
    # larger sets and over-control the smaller ones.
    rng = random.Random(args.seed)
    pool = [(b, h) for b in blocks for h in range(n_heads)]
    nulls = {}
    for name, hs in head_sets.items():
        k = max(len(heads) for heads in (hs.values() if isinstance(hs, dict) else [hs]))
        used = {x for heads in (hs.values() if isinstance(hs, dict) else [hs]) for x in heads}
        nulls[name] = sorted(rng.sample([x for x in pool if x not in used], k))
        print(f"  null for {name} ({k}): " + ", ".join(f"{b}.{h}" for b, h in nulls[name]))

    os.makedirs(out_dir, exist_ok=True)
    files = {arm: open(os.path.join(out_dir, f"{arm}.jsonl"), "w") for arm in arm_names}
    stats = {}

    # The gate. If the patch and the read disagree, every number below is void --
    # so pay one generation to find out before paying len(jobs).
    if "heads" in args.arms:
        first_attr, probe_jobs = next(batches_by_donor_attribute(jobs, 4))
        probe = build_batch(probe_jobs, processor, items, entity_dir, lookup, pad_id)
        pz = capture_donor(adapter, model, blocks, probe["donor_ids"], probe["donor_mask"],
                           probe["donor_extra"], args.max_new_tokens, pad_id)
        for name, hs in head_sets.items():
            worst = verify_readback(adapter, model, heads_for(hs, first_attr), pz, probe["base_ids"],
                                    probe["base_mask"], probe["base_extra"], args.max_new_tokens,
                                    pad_id, head_dim, subspace=subspaces.get(name))
            scope = "in-subspace component" if args.subspace_dim else "installed"
            print(f"  read-back check [{name}]: worst |{scope} - read| = {worst:.2e} over "
                  f"{args.max_new_tokens} steps")

    done = 0
    try:
        for attribute, chunk in batches_by_donor_attribute(jobs, args.batch_size):
            batch = build_batch(chunk, processor, items, entity_dir, lookup, pad_id)
            need_donor = any(a in ("heads", "random_heads") for a in args.arms)
            donor_z = (capture_donor(adapter, model, blocks, batch["donor_ids"], batch["donor_mask"],
                                     batch["donor_extra"], args.max_new_tokens, pad_id)
                       if need_donor else None)

            def emit(arm, toks):
                for j, text in zip(chunk, decode(processor, toks)):
                    for target, row_index in j["rows"]:
                        files[arm].write(json.dumps({"attribute": target, "row_index": row_index,
                                                     "generated_text": text}) + "\n")
                files[arm].flush()

            for arm in args.arms:
                if arm == "clean":
                    emit(arm, plain_generate(model, batch["base_ids"], batch["base_mask"],
                                             batch["base_extra"], args.max_new_tokens, pad_id))
                elif arm == "full_image":
                    patches = full_image_patch(adapter, model, batch, args.patch_layer, image_token_id)
                    emit(arm, patched_generate(adapter, model, [], {}, batch["base_ids"],
                                               batch["base_mask"], batch["base_extra"],
                                               args.max_new_tokens, pad_id, head_dim,
                                               extra_patches=patches))
                else:
                    for name, hs in head_sets.items():
                        use = heads_for(hs, attribute) if arm == "heads" else nulls[name]
                        sp = subspaces.get(name) if arm == "heads" else None
                        emit(f"{sub_tag}{arm}_{name}",
                             patched_generate(adapter, model, use, donor_z, batch["base_ids"],
                                              batch["base_mask"], batch["base_extra"],
                                              args.max_new_tokens, pad_id, head_dim, stats=stats,
                                              subspace=sp))
            done += len(chunk)
            print(f"  {done}/{len(jobs)} generations ({attribute})", flush=True)
    finally:
        for f in files.values():
            f.close()

    if stats.get("steps_beyond_donor"):
        print(f"  note: {stats['steps_beyond_donor']} hook calls ran past the donor's recorded steps "
              f"(donor hit EOS first); the last donor step was held.")
    print("\nScore each arm with:")
    for arm in arm_names:
        print(f"  python {os.path.join(args.vade_root, 'eval', 'score.py')} --entity {args.entity} "
              f"--attribute all --predictions {os.path.join(out_dir, arm + '.jsonl')}")


if __name__ == "__main__":
    main()
