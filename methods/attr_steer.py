"""Steer the requested attribute with a FIXED direction, and watch the image heads.

The sweep and attr_head_trace both patch the donor run's OWN activations into
the base run. That proves information is present and load-bearing, but it says
nothing about whether the attribute is carried by a single direction -- a full
swap moves every coordinate, including everything idiosyncratic to that row.

This asks the harder question. Estimate ONE vector per ordered attribute pair,

    d(a -> a') = mean over TRAIN items of ( resid[item, a'] - resid[item, a] )

then apply that same vector to HELD-OUT items and see whether the model answers
the other question. A direction estimated on 32 countries that flips the 33rd is
a fact about the model's representation; a full swap is not.

HELD-OUT ITEMS ARE NOT OPTIONAL. The direction is a mean over items, so
including the test item puts a fraction of its own activation into the vector
being applied to it, and the arm silently becomes a weak full swap. Train and
test item sets are disjoint and recorded.

TWO WAYS TO APPLY IT, because "substitute" is ambiguous:

    add       x' = x + alpha * d          -- the standard steering form. Leaves
                                             the base attribute's component in
                                             place and adds the donor's.
    project   x' = x - (x.u)u + (mu_donor.u)u, u = d/||d||
                                          -- genuinely SUBSTITUTES: removes
                                             whatever the row had along the axis
                                             and installs the donor's mean
                                             coordinate. Strictly weaker than
                                             `add` (it cannot overshoot), and it
                                             is the one that tests "this axis
                                             CARRIES the attribute" rather than
                                             "pushing along it changes the
                                             answer".

WHAT IS MEASURED, per layer:

  1. the generated answer, scored exactly as the sweep scores it (donor_first,
     donor_full, base_kept) via attr_head_trace.score_switch, so the numbers sit
     on the same axis as ATTRIBUTE_SWITCH_FINDINGS.md;
  2. the ATTENTION of the image heads -- for every (block, head) in an image
     head_trace's top-k, the attention from the readout position onto the IMAGE
     columns, clean vs steered. This is the question a logit-level result cannot
     answer: does steering the attribute change WHERE the model looks in the
     picture, or only what it does with what it already read?

Attention weights require attn_implementation="eager" -- sdpa and flash return
None for them even with output_attentions=True (see methods/attention_maps.py).
The attention phase therefore runs its own small-batch eager pass and is off
unless --image_trace is given.

A NULL CONTROL IS BUILT IN. --n_random random directions of the same norm are
applied at every layer. A steering result without it is uninterpretable: pushing
a residual stream along any large vector degrades the answer, and "base_kept
fell" is not evidence of anything on its own.
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

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.attr_capture import ENTITY_KEYS  # noqa: E402
from methods.attr_head_trace import build_switch_batch, score_switch  # noqa: E402
from methods.attribute_switch_sweep import ATTRIBUTES, PREFILL, question  # noqa: E402
from methods.common.hooks import extra_to_device, make_cache_aware_patch_hook  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.sites import RESIDUAL_SITE  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS  # noqa: E402
from methods.head_trace import generate_with_patches  # noqa: E402
from methods.ndm.verify_sites import generate_unhooked  # noqa: E402


def estimate_directions(adapter, model, processor, entity_dir, items, item_ids, attributes, layers,
                        batch_size, read_last):
    """-> {layer: {attribute: mean residual at the read position}} over `item_ids`.

    One forward per (item, attribute); the per-attribute MEAN over items is what
    survives, so whatever is idiosyncratic to an item averages out and what is
    left is the question. Deliberately computed here rather than read from an
    attr_capture directory: the steering layers are a free choice and need not
    be the ones some earlier capture happened to cover.
    """
    from PIL import Image
    totals = {L: {a: None for a in attributes} for L in layers}
    counts = {a: 0 for a in attributes}
    grid = [(i, a) for i in item_ids for a in attributes]
    for start in range(0, len(grid), batch_size):
        chunk = grid[start:start + batch_size]
        ids, pixels, grids = [], [], []
        for item_id, attribute in chunk:
            with Image.open(entity_dir / items[item_id]["image"]) as img:
                built = adapter.build_inputs(processor, img.convert("RGB"), question(attribute), PREFILL)
            ids.append(built["input_ids"].unsqueeze(0))
            pixels.append(built["extra"]["pixel_values"])
            grids.append(built["extra"]["image_grid_thw"])
        batch_ids = torch.cat(ids)
        mask = torch.ones_like(batch_ids)
        extra = {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.cat(grids)}
        pos = torch.full((len(chunk), 1), batch_ids.shape[1] - 1, dtype=torch.long)
        for L in layers:
            vals = RESIDUAL_SITE.capture(adapter, model, L, batch_ids, mask, extra, pos)  # [B,1,H]
            for offset, (_, attribute) in enumerate(chunk):
                v = vals[offset, 0].float()
                totals[L][attribute] = v.clone() if totals[L][attribute] is None else totals[L][attribute] + v
        for _, attribute in chunk:
            counts[attribute] += 1
        print(f"    {start + len(chunk)}/{len(grid)} training forwards", flush=True)
    return {L: {a: totals[L][a] / counts[a] for a in attributes} for L in layers}, counts


def build_steer_patch(means, batch, layer, attributes, mode, alpha, device,
                      random_direction=None):
    """-> [(site, layer, patch_fn)] applying ONE direction PER LANE.

    Each row of a batch has its own (base, donor) pair, so each lane needs its
    own vector; this is a single hook that indexes the stacked directions by
    lane rather than one hook per row.

      add       x' = x + alpha*d,                d = mu_donor - mu_base
      project   x' = x - (x.u)u + alpha*t*u,     u = d/||d||,
                                                 t = (mu_donor - grand mean).u

    `project` DISCARDS the row's own coordinate on the axis instead of pushing
    it, which is why it cannot overshoot and why it is the arm that tests
    whether the axis carries the attribute rather than whether shoving along it
    changes the answer.
    """
    grand = sum(means[layer].values()) / len(attributes)
    stack = [random_direction if random_direction is not None
             else means[layer][r["donor_attribute"]] - means[layer][r["base_attribute"]]
             for r in batch["rows"]]
    delta = torch.stack([d.to(device) for d in stack])                       # [B, H]
    if mode == "add":
        def fn(x, _d=delta):
            return (x.float() + alpha * _d[:, None, :].to(x.device)).to(x.dtype)
    else:
        u = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        tgt = torch.stack([(means[layer][r["donor_attribute"]] - grand).to(device) @ u[i]
                           for i, r in enumerate(batch["rows"])])            # [B]
        def fn(x, _u=u, _t=tgt):
            y = x.float()
            uu = _u[:, None, :].to(y.device)
            coord = (y * uu).sum(-1, keepdim=True)
            return (y - coord * uu + alpha * _t[:, None, None].to(y.device) * uu).to(x.dtype)
    return [(RESIDUAL_SITE, layer, make_cache_aware_patch_hook(batch["steer_positions"], fn))]


def image_columns(batch, image_token_id):
    """The prompt columns holding image tokens, asserted identical across the batch."""
    ids = batch["base_input_ids"]
    cols = (ids[0] == image_token_id).nonzero().flatten()
    assert cols.numel() > 0, "no image tokens in the prompt"
    for i in range(1, ids.shape[0]):
        assert torch.equal((ids[i] == image_token_id).nonzero().flatten(), cols), \
            "rows disagree on the image span; the attention comparison would average different columns"
    return cols


def head_attention(adapter, model, batch, heads, image_cols, patches=()):
    """-> {(block, head): [B, n_image_cols]} attention from the LAST prompt
    position onto the image columns, under `patches`.

    Needs eager attention: sdpa/flash return None for attn_weights even with
    output_attentions=True (methods/attention_maps.py documents this), and a
    silent None here would read as "the heads stopped attending".
    """
    layers = adapter.get_decoder_layers(model)
    handles = []
    for site, layer_idx, fn in patches:
        handles.extend(site.register(adapter, model, layers, layer_idx, fn))
    try:
        with torch.no_grad():
            out = model(input_ids=batch["base_input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                        **extra_to_device(batch["base_extra"], model.device, model.dtype),
                        output_attentions=True, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    assert out.attentions is not None and out.attentions[0] is not None, (
        "the model returned no attention weights -- load it with attn_implementation='eager'")
    cols = image_cols.to(model.device)
    return {(b, h): out.attentions[b][:, h, -1, :].index_select(-1, cols).float().cpu()
            for b, h in heads}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attributes", nargs="+", choices=ATTRIBUTES, default=ATTRIBUTES)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(14, 29)),
                    help="Residual layers to steer at (layer L = output of block L-1).")
    ap.add_argument("--mode", choices=["add", "project", "both"], default="both")
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0],
                    help="Scale on the direction. 1.0 is the natural scale (the measured mean "
                         "difference); sweep it to separate 'this axis carries it' from 'a big "
                         "enough push along anything breaks it'.")
    ap.add_argument("--positions", choices=["last_token", "question"], default="last_token",
                    help="Where to apply the direction. last_token is where the sweep says the "
                         "attribute lives from block 20 on; question is where it lives below 17.")
    ap.add_argument("--n_train_items", type=int, default=32, help="Items used to ESTIMATE directions.")
    ap.add_argument("--n_test_items", type=int, default=16, help="Disjoint items the steering is scored on.")
    ap.add_argument("--n_random", type=int, default=2,
                    help="Random directions of matched norm, per layer. 0 disables the null, which "
                         "makes every number below uninterpretable.")
    ap.add_argument("--image_trace", default=None,
                    help="head_trace JSON. Enables the attention phase on its top-k heads, and "
                         "forces eager attention for the whole run.")
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--attention_layers", type=int, nargs="+", default=None,
                    help="Layers to run the attention comparison at (default: every --layer whose "
                         "steering moved the answer most; costs one eager forward pair each).")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--attention_batch_size", type=int, default=2,
                    help="Eager + output_attentions materializes [B, heads, T, T] per layer; keep small.")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_ANSWER_TOKENS + 2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    from pathlib import Path
    entity_dir = Path(args.vade_root) / "data" / args.entity
    raw = (entity_dir / "ground_truth.json").read_bytes()
    gt = json.loads(raw)
    key = ENTITY_KEYS.get(args.entity)
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != "coverage")
    items = gt[key]
    attributes = list(dict.fromkeys(args.attributes))
    if len(attributes) < 2:
        ap.error("Need at least two attributes")
    pool = sorted(a for a in items if all(x in items[a] for x in attributes)
                  and (entity_dir / items[a]["image"]).is_file())
    need = args.n_train_items + args.n_test_items
    if len(pool) < need:
        ap.error(f"{len(pool)} usable items but {need} requested (train + test must be disjoint)")
    sample = random.Random(args.seed).sample(pool, need)
    train_items, test_items = sample[:args.n_train_items], sample[args.n_train_items:]
    assert not set(train_items) & set(test_items)
    rows = [{"row_index": i, "base": it, "source": it, "base_attribute": b, "donor_attribute": d,
             "base_label": items[it][b], "source_label": items[it][d],
             "template_id": "controlled_report_v1"}
            for i, (it, (b, d)) in enumerate(
                (it, p) for it in test_items for p in itertools.permutations(attributes, 2))]
    modes = ["add", "project"] if args.mode == "both" else [args.mode]

    out_dir = Path(REPO_ROOT) / "logs" / "attr_steer" / args.entity
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"attr_steer_{args.positions}_L{args.layers[0]}-{args.layers[-1]}_"
           f"{'-'.join(modes)}_train{len(train_items)}")
    out_path = args.out or str(out_dir / f"{tag}.json")
    print(f"[attr_steer] {args.entity}: directions from {len(train_items)} TRAIN items, scored on "
          f"{len(test_items)} disjoint TEST items = {len(rows)} directed rows; layers "
          f"{args.layers[0]}-{args.layers[-1]}; modes={modes}; alphas={args.alpha}; "
          f"positions={args.positions} -> {out_path}")
    if args.dry_run:
        n = len(args.layers) * len(modes) * len(args.alpha) + (1 if args.n_random else 0) * len(args.layers)
        print(f"Grid valid. {len(train_items) * len(attributes)} training forwards, then ~{n} "
              f"generation passes over {len(rows)} rows.")
        return

    need_attention = args.image_trace is not None
    with tee_to_log(str(out_dir / f"{tag}.log")):
        adapter = get_adapter(args.model_id)
        # eager only when the attention phase needs it -- it is markedly slower, and the generation
        # phase does not read attention weights.
        model, processor = adapter.load(device=args.device, dtype=torch.bfloat16,
                                        attn_implementation="eager" if need_attention else "sdpa")
        model.eval().requires_grad_(False)
        n_layers = len(adapter.get_decoder_layers(model))
        assert all(1 <= L <= n_layers for L in args.layers), f"layers must be in 1..{n_layers}"
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        image_id = adapter.image_token_id(model, processor)

        batches = []
        for i in range(0, len(rows), args.batch_size):
            b = build_switch_batch(rows[i:i + args.batch_size], entity_dir, items, adapter, model,
                                   processor, "all_text")
            if args.positions == "last_token":
                b["steer_positions"] = b["last_positions"]
            else:
                b["steer_positions"] = b["positions"]
            batches.append(b)

        print(f"\n{'=' * 78}\n=== phase 1: estimating directions on {len(train_items)} held-out-from-test "
              f"items\n{'=' * 78}")
        means, counts = estimate_directions(adapter, model, processor, entity_dir, items, train_items,
                                            attributes, args.layers, args.batch_size, True)
        print(f"  per-attribute means over {counts[attributes[0]]} items each")
        print(f"  {'layer':>5} {'block':>5} " + "".join(f"{'||d(' + a[:4] + ')||':>16}" for a in attributes))
        for L in args.layers:
            grand = sum(means[L].values()) / len(attributes)
            print(f"  {L:>5} {L - 1:>5} " + "".join(
                f"{float((means[L][a] - grand).norm()):>16.3f}" for a in attributes))

        report = {"experiment": "attr_steer", "entity": args.entity, "attributes": attributes,
                  "layers": args.layers, "modes": modes, "alphas": args.alpha,
                  "positions": args.positions, "train_items": train_items, "test_items": test_items,
                  "n_rows": len(rows), "seed": args.seed,
                  "direction_norms": {str(L): {a: float((means[L][a] - sum(means[L].values())
                                                         / len(attributes)).norm())
                                               for a in attributes} for L in args.layers},
                  "ground_truth_sha256": hashlib.sha256(raw).hexdigest(), "arms": []}

        def run(L, make_patch):
            acc = {"first": 0.0, "full": 0.0, "kept": 0.0, "n": 0}
            for batch in batches:
                patches = make_patch(batch, L)
                gen = (generate_with_patches(adapter, model, patches, batch["base_input_ids"],
                                             batch["attention_mask"], batch["base_extra"],
                                             pad_token_id, args.max_new_tokens) if patches else
                       generate_unhooked(model, batch["base_input_ids"], batch["attention_mask"],
                                         batch["base_extra"], pad_token_id, args.max_new_tokens))
                f, u, _, kept = score_switch(gen, batch)
                k = len(batch["rows"])
                acc["first"] += f * k
                acc["full"] += u * k
                acc["kept"] += kept * k
                acc["n"] += k
            return {m: acc[m] / acc["n"] for m in ("first", "full", "kept")} | {"n": acc["n"]}

        def per_row_patch(batch, L, mode, alpha, rand=None):
            return build_steer_patch(means, batch, L, attributes, mode, alpha, model.device,
                                     random_direction=rand)

        print(f"\n{'=' * 78}\n=== phase 2: steering the held-out items\n{'=' * 78}")
        clean = run(args.layers[0], lambda b, L: [])
        print(f"  {'unsteered':<34} first={clean['first']:6.1%} full={clean['full']:6.1%} "
              f"base_kept={clean['kept']:6.1%}   <-- floor; base_kept is the ceiling for a no-op")
        report["clean"] = clean
        rng = random.Random(args.seed + 1)
        best = []
        for L in args.layers:
            for mode in modes:
                for alpha in args.alpha:
                    r = run(L, lambda b, LL, m=mode, al=alpha: per_row_patch(b, LL, m, al))
                    label = f"L{L} (block {L - 1}) {mode} a={alpha}"
                    print(f"  {label:<34} first={r['first']:6.1%} full={r['full']:6.1%} "
                          f"base_kept={r['kept']:6.1%}", flush=True)
                    report["arms"].append({"layer": L, "block": L - 1, "mode": mode, "alpha": alpha,
                                           "kind": "attribute", **r})
                    best.append((r["first"], L))
            if args.n_random:
                grand = sum(means[L].values()) / len(attributes)
                scale = float(torch.stack([means[L][a] - grand for a in attributes]).norm(dim=-1).mean()) * 2
                for j in range(args.n_random):
                    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
                    v = torch.randn(means[L][attributes[0]].shape, generator=g)
                    v = (v / v.norm() * scale).to(model.device)
                    r = run(L, lambda b, LL, _v=v: per_row_patch(b, LL, "add", 1.0, rand=_v))
                    print(f"  {'L' + str(L) + ' RANDOM direction #' + str(j + 1):<34} "
                          f"first={r['first']:6.1%} full={r['full']:6.1%} base_kept={r['kept']:6.1%}"
                          f"   <-- null", flush=True)
                    report["arms"].append({"layer": L, "block": L - 1, "mode": "add", "alpha": 1.0,
                                           "kind": "random", "index": j, **r})
        print(f"\n  Read the attribute arms AGAINST the random arms at the same layer. A random "
              f"direction of matched norm should leave `first` at the unsteered floor "
              f"({clean['first']:.1%}); if it does not, this layer is simply fragile and its "
              f"attribute arm proves nothing.")

        # ---------------- phase 3: what the image heads look at ----------------
        if need_attention:
            trace = json.loads(Path(args.image_trace).read_text())
            field = trace.get("rank_by", "delta_resid")
            heads = [(int(r["block"]), int(r["head"]))
                     for r in sorted(trace["phase1"], key=lambda r: -abs(r[field]))[:args.top_k]]
            layers = args.attention_layers or [L for _, L in sorted(best, reverse=True)[:3]]
            print(f"\n{'=' * 78}\n=== phase 3: image-head attention, clean vs steered\n{'=' * 78}")
            print(f"  top-{args.top_k} heads of {Path(args.image_trace).name} by {field}; "
                  f"steering layers {layers}")
            print(f"  Attention is a distribution over the image columns at the readout position. "
                  f"`mass` is how much of it lands on the image at all; `cosine` compares the "
                  f"clean and steered patterns. Steering that changes the ANSWER without changing "
                  f"`cosine` means the model looked at the same pixels and reported something else "
                  f"-- selection happening downstream of the image read, not inside it.")
            report["attention"] = []
            small = [b for b in batches][:max(1, args.attention_batch_size)]
            for L in layers:
                for mode in modes:
                    agg = {}
                    for batch in small:
                        cols = image_columns(batch, image_id)
                        base = head_attention(adapter, model, batch, heads, cols)
                        steer = head_attention(adapter, model, batch, heads, cols,
                                               patches=per_row_patch(batch, L, mode, args.alpha[0]))
                        for hk in heads:
                            a, b_ = base[hk], steer[hk]
                            cos = torch.nn.functional.cosine_similarity(a, b_, dim=-1).mean().item()
                            agg.setdefault(hk, []).append(
                                (a.sum(-1).mean().item(), b_.sum(-1).mean().item(), cos,
                                 (b_ - a).abs().sum(-1).mean().item()))
                    print(f"\n  steering at L{L} (block {L - 1}), mode={mode}:")
                    print(f"    {'block.head':>12} {'mass clean':>11} {'mass steer':>11} "
                          f"{'cosine':>8} {'L1 shift':>9}")
                    for hk in heads:
                        v = torch.tensor(agg[hk]).mean(0)
                        print(f"    {hk[0]}.{hk[1]:<10} {v[0]:>11.4f} {v[1]:>11.4f} {v[2]:>8.4f} "
                              f"{v[3]:>9.4f}")
                        report["attention"].append({"layer": L, "block": L - 1, "mode": mode,
                                                    "head_block": hk[0], "head": hk[1],
                                                    "mass_clean": float(v[0]), "mass_steered": float(v[1]),
                                                    "cosine": float(v[2]), "l1_shift": float(v[3])})

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
