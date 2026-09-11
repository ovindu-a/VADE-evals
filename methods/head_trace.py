"""TRACE the image->text handoff: which attention heads write the object's
information into the text stream, and which of them are load-bearing.

WHY THIS EXISTS, AND WHY IT INTERVENES ON THE IMAGE SIDE. ceiling_sweep.py's
two position sets bracket the handoff without locating it. On flags/language
a full residual swap reads:

    layer      <=21    22      23      24+
    image      100%    100%    0%      0%
    last_token   0%    9.4%    68.8%   100%

Patching the image stops mattering at exactly the layer patching the last
token starts mattering, which is the read itself: before it, editing the
image still propagates; after it, editing the image is too late and editing
the destination is decisive. So the read is in blocks ~21-23.

The only operation in a transformer that moves information BETWEEN positions
is attention -- MLPs are position-wise and the residual stream never mixes
tokens -- so every bit of that transfer is

    delta resid[last] += sum_h sum_{j in image} a[h, last, j] * (v[j] . W_O^h)

for some (block, head). With the window above that is a few dozen candidates,
small enough to enumerate.

The naive probe does NOT work: swapping head h's dims AT THE LAST TOKEN is a
subset of swapping the whole attn_output there, which ceiling_sweep measured
at 0.0% everywhere except 9.4% at layer 24. Anything bounded by that is dead
on arrival. The leverage is on the SOURCE side, where a full image swap at
--patch_layer already produces a 100% effect. Both phases below therefore
patch the image and observe/ablate downstream.

PHASE 1 -- differential head trace (2 forward passes per batch, exact, no
generation). Capture every block's per-head attention output at the LAST
TOKEN, once on a clean base run and once with the image residual patched to
the source at --patch_layer. A head whose output changed is a head that READ
the patched image. Three numbers per (block, head), all exact:

  delta_z        ||dz_h||               how much the head's raw output moved
  delta_resid    ||dz_h W_O_h^T||       how much reached the RESIDUAL STREAM.
                                        The default ranking: W_O scales heads
                                        very differently, so a large dz can
                                        land as nothing and vice versa.
  delta_dla      direct logit effect    methods/dla.py's decomposition applied
                                        per head, differenced between the two
                                        runs (each with its OWN frozen scale,
                                        so each term is exact). Catches heads
                                        that move the ANSWER rather than just
                                        moving. Direct paths only -- a head
                                        acting through a later head shows up
                                        in delta_resid but not here.

PHASE 2 -- cumulative knockout (path patching; one generation per k). Keep the
image patched (cause ~100%) and RESTORE the top-k heads' outputs at the last
token to their clean-base values. If cause collapses, those heads carry the
signal. Cumulative -- k = 1, 2, 4, 8, ... -- rather than one head at a time,
because single-head knockout will read ~0 for exactly the reason single-
sublayer swaps do: a marginal measurement cannot see a conjunction. The
cumulative curve is the head-level analogue of ceiling_sweep's blocks:N.

A NULL CONTROL IS BUILT IN: --n_random knocks out the same number of RANDOMLY
chosen heads. If knocking out 8 random heads hurts cause as much as the top 8,
the ranking is not carrying information and the phase-2 curve means nothing.
Read the two curves together or not at all.

Usage:
    python methods/head_trace.py --entity flags --attribute language \
        --patch_layer 21 --positions flag_ring1 --blocks 18 19 20 21 22 23 24 25 26 27
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch  # noqa: E402

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import (  # noqa: E402
    BuildBatchCache, load_entity_assets, load_tuples, require_pruned_tuples,
)
from methods.common.hooks import extra_to_device, make_cache_aware_patch_hook  # noqa: E402
from methods.common.position_sets import build_batch_at  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.sites import RESIDUAL_SITE, InterventionSite  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS  # noqa: E402
from methods.dbm.intervention import SigmoidMaskIntervention  # noqa: E402
from methods.ndm.ceiling_sweep import score_generation  # noqa: E402
from methods.ndm.config import METHOD_NAME, ndm_logs_dir  # noqa: E402
from methods.ndm.verify_sites import HARD_TEMPERATURE, generate_unhooked  # noqa: E402

HEAD_SITE = InterventionSite("attn_head_output")
RANK_BY = ("delta_resid", "delta_z", "delta_dla")


def head_slice(head, head_dim):
    return slice(head * head_dim, (head + 1) * head_dim)


def head_mask(embed_dim, head_dim, heads, device):
    """A SigmoidMaskIntervention saturated to EXACTLY the indicator of `heads`
    -- 1.0 on those heads' contiguous dim ranges, 0.0 everywhere else. Uses
    the same module a trained mask would, so the numerics of a knockout match
    the numerics of a learned intervention rather than approximating them;
    the assert is what makes that safe, since a mask that failed to saturate
    would blend instead of replace and still produce plausible numbers."""
    iv = SigmoidMaskIntervention(embed_dim=embed_dim).to(device)
    with torch.no_grad():
        flat = iv.mask.view(-1)
        flat.fill_(-1.0)
        for h in heads:
            flat[head_slice(h, head_dim)] = 1.0
    iv.set_temperature(torch.tensor(HARD_TEMPERATURE))
    iv.eval()
    gate = torch.sigmoid(iv.mask.detach() / iv.temperature.detach()).view(-1)
    want = torch.zeros(embed_dim, device=gate.device, dtype=gate.dtype)
    for h in heads:
        want[head_slice(h, head_dim)] = 1.0
    assert torch.equal(gate, want), (
        f"head mask did not saturate to an exact indicator over heads {sorted(heads)} -- "
        f"{int((gate != want).sum())} of {embed_dim} dims are wrong")
    return iv


def _final_residual_hook(adapter, model, positions, sink):
    """Grabs the LAST decoder block's output -- the final PRE-norm residual
    stream -- at `positions`. Deliberately not read off out.hidden_states[-1]:
    in this project's transformers that entry is tied to last_hidden_state and
    is therefore POST-final-norm (the trap methods/logit_lens.py documents),
    which would make the frozen scale below wrong by exactly the factor it is
    supposed to be."""
    layers = adapter.get_decoder_layers(model)
    B = positions.shape[0]

    def post(mod, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        sink.append(torch.stack([t[i, positions[i]] for i in range(B)]).detach())

    return layers[-1].register_forward_hook(post)


def capture_head_outputs(adapter, model, blocks, positions, input_ids, attention_mask, extra, patches=()):
    """One forward pass -> ({block: [B, n_pos, hidden_size]} of attn_head_output
    at `positions`, [B, n_pos, hidden_size] final pre-norm residual at the same
    positions), optionally with `patches` (a list of (site, layer_idx,
    patch_fn)) active during it. Reuses common/sites.py's own capture hook, so
    this reads exactly the tensor the interventions patch.

    The final residual comes back because methods/dla.py's decomposition needs
    the norm's scalar FROZEN at what the real pass computed from the FULL
    residual -- and that scalar differs between the clean and patched runs, so
    each run must carry its own."""
    sinks = {b: [] for b in blocks}
    handles = [HEAD_SITE.register_capture(adapter, model, b + 1, positions, sinks[b]) for b in blocks]
    final_sink = []
    handles.append(_final_residual_hook(adapter, model, positions, final_sink))
    layers = adapter.get_decoder_layers(model)
    for site, layer_idx, fn in patches:
        handles.extend(site.register(adapter, model, layers, layer_idx, fn))
    try:
        extra_dev = extra_to_device(extra, model.device, model.dtype)
        with torch.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                  **extra_dev, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    for b in blocks:
        assert len(sinks[b]) == 1, (
            f"block {b}'s attn_head_output capture hook fired {len(sinks[b])} times in one forward pass, "
            f"expected 1")
    assert len(final_sink) == 1, (
        f"the final-residual hook fired {len(final_sink)} times in one forward pass, expected 1")
    return {b: sinks[b][0] for b in blocks}, final_sink[0]


def generate_with_patches(adapter, model, patches, input_ids, attention_mask, extra, pad_token_id,
                          max_new_tokens):
    """Greedy generation with SEVERAL patches live at once, each at its own
    site/layer/positions (positions are already baked into each patch_fn by
    make_cache_aware_patch_hook). sites.py's JointSite deliberately does not
    cover this case -- it requires one shared position set -- and phase 2
    needs an IMAGE-position patch and a LAST-TOKEN patch in the same pass."""
    layers = adapter.get_decoder_layers(model)
    handles = []
    for site, layer_idx, fn in patches:
        handles.extend(site.register(adapter, model, layers, layer_idx, fn))
    try:
        extra_dev = extra_to_device(extra, model.device, model.dtype)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                **extra_dev, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id,
            )
    finally:
        for h in handles:
            h.remove()
    return gen[:, input_ids.shape[1]:].cpu()


def image_patch(adapter, model, batch_img, patch_layer):
    """-> (site, layer, patch_fn) replacing the base's residual stream with the
    source's at the image positions -- the intervention ceiling_sweep already
    measured at ~100% cause below the handoff, and the thing both phases
    observe the downstream consequences of."""
    from methods.ndm.verify_sites import hard_mask
    src = RESIDUAL_SITE.capture(adapter, model, patch_layer, batch_img["source_input_ids"],
                                batch_img["attention_mask"], batch_img["source_extra"],
                                batch_img["positions"])
    one = hard_mask(RESIDUAL_SITE.width(adapter, model), 1.0, model.device)
    fn = make_cache_aware_patch_hook(batch_img["positions"], lambda base_vals: one(base_vals, src))
    return (RESIDUAL_SITE, patch_layer, fn)


def per_head_delta(adapter, model, base_z, patched_z, block, n_heads, head_dim, direction, scale_base,
                   scale_patched):
    """-> [(head, delta_z, delta_resid, delta_dla)] for one block, averaged over
    the batch.

      delta_z      ||dz_h||, the raw per-head output change.
      delta_resid  ||dz_h W_O_h^T||, the change that actually LANDS in the
                   residual stream. o_proj weights heads very differently, so
                   this and delta_z can disagree; this is the one to trust.
      delta_dla    methods/dla.py's exact per-component logit term for this
                   head, differenced between the two runs -- each computed
                   with ITS OWN run's frozen scale, so each is exact for that
                   run and so is the difference. DIRECT paths only.

    direction: [B, H] per row (each row has its own gold tokens -- averaging
    the direction across the batch would attribute row i's head movement along
    row j's answer axis). scale_*: [B, n_pos, 1] from each run's own final
    pre-norm residual."""
    w = adapter.get_attn_head_output_module(model, block).weight  # [H_out, H_in], bias-free
    d = direction.unsqueeze(1)                                    # [B, 1, H]
    out = []
    for h in range(n_heads):
        sl = head_slice(h, head_dim)
        wh = w[:, sl].float()                                                      # [H_out, head_dim]
        resid_base = base_z[:, :, sl].float() @ wh.T                               # [B, n_pos, H_out]
        resid_patched = patched_z[:, :, sl].float() @ wh.T
        dla_base = scale_base.squeeze(-1) * (resid_base * d).sum(-1)               # [B, n_pos]
        dla_patched = scale_patched.squeeze(-1) * (resid_patched * d).sum(-1)
        out.append((h,
                    (patched_z[:, :, sl] - base_z[:, :, sl]).float().norm(dim=-1).mean().item(),
                    (resid_patched - resid_base).norm(dim=-1).mean().item(),
                    (dla_patched - dla_base).mean().item()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--patch_layer", type=int, required=True,
                    help="Layer to patch the IMAGE residual at. Pick one where ceiling_sweep says an image "
                         "swap still produces a large cause (i.e. BELOW the handoff) -- patching where the "
                         "effect is already 0 leaves nothing downstream to trace.")
    ap.add_argument("--positions", default="flag_ring1", help="Image position set to patch.")
    ap.add_argument("--blocks", type=int, nargs="+", default=None,
                    help="Decoder blocks to trace heads in (default: --patch_layer-1 through the last). "
                         "A block before the patch cannot be affected by it, so those are excluded by "
                         "default rather than reported as a wall of zeros.")
    ap.add_argument("--rank_by", default="delta_resid", choices=list(RANK_BY),
                    help="Which phase-1 column orders the knockout. delta_resid (default) is the honest "
                         "'this head wrote something different' measure and catches heads acting through "
                         "later heads; delta_dla ranks by movement of the ANSWER but sees direct paths only.")
    ap.add_argument("--knockout_ks", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="Cumulative knockout sizes. Expect k=1 to read ~0 even for a real conduit -- see "
                         "the module docstring on why single-head ablation cannot see a conjunction.")
    ap.add_argument("--n_random", type=int, default=8,
                    help="Size of the RANDOM-head null control. 0 disables it, which you should not do: "
                         "without it a phase-2 curve is uninterpretable.")
    ap.add_argument("--skip_knockout", action="store_true", help="Phase 1 only (no generation at all).")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n_rows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_ANSWER_TOKENS + 2)
    ap.add_argument("--top_n", type=int, default=20, help="How many heads to print in the phase-1 table.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_dir = ndm_logs_dir(model_slug, args.entity, args.attribute, 0.0, 0.0, 0.0, 0.0,
                           args.positions, "attn_head_output", pruned)
    os.makedirs(log_dir, exist_ok=True)
    out_path = args.out or os.path.join(log_dir, f"head_trace_patch{args.patch_layer}.json")

    with tee_to_log(os.path.join(log_dir, f"head_trace_patch{args.patch_layer}.log")):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        entity_assets = load_entity_assets(args.vade_root, args.entity)
        n_layers = len(adapter.get_decoder_layers(model))
        n_heads = adapter.n_attention_heads(model)
        hidden = adapter.hidden_size(model)
        assert hidden % n_heads == 0, f"hidden_size {hidden} is not divisible by {n_heads} heads"
        head_dim = hidden // n_heads
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        blocks = args.blocks if args.blocks is not None else list(range(max(args.patch_layer - 1, 0), n_layers))
        assert all(0 <= b < n_layers for b in blocks), f"blocks must be in 0..{n_layers - 1}"

        rows = load_tuples(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir)
        for r in rows:
            r.setdefault("target_attribute", args.attribute)
        cause_all = [r for r in rows if r["rule"] == "match_source"]
        cause_rows = random.Random(args.seed).sample(cause_all, min(args.n_rows, len(cause_all)))
        print(f"[{METHOD_NAME}/head_trace] entity={args.entity} attribute={args.attribute} "
              f"patch_layer={args.patch_layer} positions={args.positions} blocks={blocks[0]}-{blocks[-1]} "
              f"heads={n_heads}x{head_dim} rows={len(cause_rows)} "
              f"({len({r['base'] for r in cause_rows})} distinct bases)")

        cache = BuildBatchCache()
        batches = []
        for i in range(0, len(cause_rows), args.batch_size):
            chunk = cause_rows[i:i + args.batch_size]
            b_img = build_batch_at(args.positions, chunk, entity_assets, adapter, model, processor,
                                   batch_cache=cache)
            b_last = build_batch_at("last_token", chunk, entity_assets, adapter, model, processor,
                                    batch_cache=cache)
            # Same rows and templates, so the two batches MUST be the same sequences -- only their
            # `positions` differ. Asserted rather than assumed: a silent mismatch here would patch one
            # prompt's image and read another prompt's last token, and every number downstream would
            # still look plausible.
            assert torch.equal(b_img["base_input_ids"], b_last["base_input_ids"]), \
                "image-position and last_token batches disagree on base_input_ids"
            assert torch.equal(b_img["attention_mask"], b_last["attention_mask"]), \
                "image-position and last_token batches disagree on attention_mask"
            batches.append((b_img, b_last))

        # ---------------- phase 1: differential head trace ----------------
        print(f"\n{'='*78}\n=== phase 1: differential head trace (2 forwards per batch, no generation)\n{'='*78}")
        agg = {(b, h): {"delta_z": 0.0, "delta_resid": 0.0, "delta_dla": 0.0, "n": 0} for b in blocks
               for h in range(n_heads)}
        for bi, (b_img, b_last) in enumerate(batches):
            last_pos = b_last["positions"]
            patch = image_patch(adapter, model, b_img, args.patch_layer)

            base_z, base_final = capture_head_outputs(
                adapter, model, blocks, last_pos, b_img["base_input_ids"], b_img["attention_mask"],
                b_img["base_extra"])
            patched_z, patched_final = capture_head_outputs(
                adapter, model, blocks, last_pos, b_img["base_input_ids"], b_img["attention_mask"],
                b_img["base_extra"], patches=[patch])

            # The DLA direction: base gold minus SOURCE gold at the first answer position -- the exact
            # axis `cause` moves along (see methods/dla.py's --direction base_minus_source). Kept
            # PER ROW, since each row's base/source pair has its own gold tokens.
            gold_b = b_img["base_gold_toks"][:, 0].to(model.device)
            gold_s = b_img["source_gold_toks"][:, 0].to(model.device)
            d = adapter.logit_direction(model, gold_b) - adapter.logit_direction(model, gold_s)  # [B, H]
            scale_base = adapter.final_norm_scale(model, base_final.float())                     # [B,n_pos,1]
            scale_patched = adapter.final_norm_scale(model, patched_final.float())

            for b in blocks:
                for h, dz, dr, dd in per_head_delta(adapter, model, base_z[b], patched_z[b], b, n_heads,
                                                    head_dim, d, scale_base, scale_patched):
                    a = agg[(b, h)]
                    a["delta_z"] += dz
                    a["delta_resid"] += dr
                    a["delta_dla"] += dd
                    a["n"] += 1
            print(f"  batch {bi + 1}/{len(batches)} traced", flush=True)

        table = [{"block": b, "head": h,
                  "delta_z": a["delta_z"] / a["n"], "delta_resid": a["delta_resid"] / a["n"],
                  "delta_dla": a["delta_dla"] / a["n"]}
                 for (b, h), a in agg.items() if a["n"]]
        ranked = sorted(table, key=lambda r: -abs(r[args.rank_by]))
        print(f"\ntop {args.top_n} heads by |{args.rank_by}| "
              f"(how much this head's write to the last token changed when the image was swapped):")
        print(f"{'block.head':>12} {'delta_resid':>12} {'delta_z':>10} {'delta_dla':>11}")
        for r in ranked[:args.top_n]:
            print(f"{r['block']}.{r['head']:<10} {r['delta_resid']:>12.4f} {r['delta_z']:>10.4f} "
                  f"{r['delta_dla']:>11.4f}")
        by_block = {}
        for r in table:
            by_block.setdefault(r["block"], 0.0)
            by_block[r["block"]] += abs(r[args.rank_by])
        print(f"\nper-block total |{args.rank_by}| (where the read happens):")
        for b in blocks:
            print(f"  block {b:>2}: {by_block[b]:>10.4f}")

        report = {"entity": args.entity, "attribute": args.attribute, "patch_layer": args.patch_layer,
                  "positions": args.positions, "blocks": blocks, "n_heads": n_heads, "head_dim": head_dim,
                  "n_rows": len(cause_rows), "rank_by": args.rank_by, "seed": args.seed,
                  "phase1": table, "phase1_ranked": [(r["block"], r["head"]) for r in ranked[:64]]}

        if args.skip_knockout:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nwrote {out_path}  (phase 2 skipped)")
            return

        # ---------------- phase 2: cumulative knockout ----------------
        print(f"\n{'='*78}\n=== phase 2: cumulative knockout under the image patch (path patching)\n{'='*78}")

        def run_cause(selected_heads, base_z_per_batch):
            """cause rate with the image patched AND `selected_heads` restored to
            their clean-base values at the last token. selected_heads: [(block, head)]."""
            ms, mb, n = 0.0, 0.0, 0
            per_block = {}
            for b, h in selected_heads:
                per_block.setdefault(b, []).append(h)
            for (b_img, b_last), base_z in zip(batches, base_z_per_batch):
                patches = [image_patch(adapter, model, b_img, args.patch_layer)]
                for b, heads in per_block.items():
                    mask = head_mask(hidden, head_dim, heads, model.device)
                    restore = base_z[b].to(model.device)
                    patches.append((HEAD_SITE, b + 1, make_cache_aware_patch_hook(
                        b_last["positions"], lambda bv, _m=mask, _s=restore: _m(bv, _s))))
                gen = generate_with_patches(adapter, model, patches, b_img["base_input_ids"],
                                            b_img["attention_mask"], b_img["base_extra"], pad_token_id,
                                            args.max_new_tokens)
                s, t = score_generation(gen, b_img)
                k = len(b_img["rows"])
                ms, mb, n = ms + s * k, mb + t * k, n + k
            return ms / n, mb / n

        # Re-capture the clean base head outputs once, to restore FROM. (Phase 1's were per batch and
        # not retained -- holding every batch's [B, 1, 3584] x n_blocks across phase 1 would be dead
        # weight for a run using --skip_knockout.)
        base_z_per_batch = [capture_head_outputs(adapter, model, blocks, b_last["positions"],
                                                 b_img["base_input_ids"], b_img["attention_mask"],
                                                 b_img["base_extra"])[0]
                            for b_img, b_last in batches]

        unhooked_ms, unhooked_mb, n = 0.0, 0.0, 0
        for b_img, _ in batches:
            gen = generate_unhooked(model, b_img["base_input_ids"], b_img["attention_mask"],
                                    b_img["base_extra"], pad_token_id, args.max_new_tokens)
            s, t = score_generation(gen, b_img)
            k = len(b_img["rows"])
            unhooked_ms, unhooked_mb, n = unhooked_ms + s * k, unhooked_mb + t * k, n + k
        print(f"  unhooked:                       cause={unhooked_ms / n:6.1%} base_kept={unhooked_mb / n:6.1%}")
        full_ms, full_mb = run_cause([], base_z_per_batch)
        print(f"  image patch only (k=0):         cause={full_ms:6.1%} base_kept={full_mb:6.1%}"
              f"   <-- the effect being traced")
        if full_ms < 0.10:
            print(f"  !! the image patch at layer {args.patch_layer} barely moves the answer, so there is "
                  f"nothing downstream to knock out. Pick a --patch_layer BELOW the handoff (check "
                  f"ceiling_sweep's image-position column) before reading anything below.")

        knock = []
        for k in args.knockout_ks:
            if k > len(ranked):
                continue
            heads = [(r["block"], r["head"]) for r in ranked[:k]]
            ms, mb = run_cause(heads, base_z_per_batch)
            knock.append({"k": k, "kind": "top", "heads": heads, "cause": ms, "base_kept": mb})
            print(f"  restore top-{k:<3} heads:            cause={ms:6.1%} base_kept={mb:6.1%}  "
                  f"(recovered {max(full_ms - ms, 0) / max(full_ms, 1e-9):5.1%} of the effect)", flush=True)

        if args.n_random:
            rng = random.Random(args.seed + 1)
            rand_heads = rng.sample([(r["block"], r["head"]) for r in table], min(args.n_random, len(table)))
            ms, mb = run_cause(rand_heads, base_z_per_batch)
            knock.append({"k": args.n_random, "kind": "random", "heads": rand_heads, "cause": ms,
                          "base_kept": mb})
            print(f"  restore {args.n_random} RANDOM heads (null):   cause={ms:6.1%} base_kept={mb:6.1%}")
            top_same = next((x for x in knock if x["kind"] == "top" and x["k"] == args.n_random), None)
            if top_same is not None:
                gap = ms - top_same["cause"]
                print(f"\n  top-{args.n_random} vs random-{args.n_random} gap: {gap:+.1%} of cause. A gap near "
                      f"zero means the phase-1 ranking is NOT carrying information and the curve above "
                      f"says nothing about which heads matter.")

        report["phase2"] = {"unhooked_cause": unhooked_ms / n, "image_patch_only_cause": full_ms,
                            "image_patch_only_base_kept": full_mb, "knockouts": knock}
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
