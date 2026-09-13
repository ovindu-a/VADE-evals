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
ATTN_OUT_SITE = InterventionSite("attn_output")
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


def _residual_hook(adapter, model, block, positions, sink):
    """Grabs decoder block `block`'s OUTPUT -- the PRE-norm residual stream
    entering block block+1 -- at `positions`. Deliberately not read off
    out.hidden_states: its last entry is tied to last_hidden_state and is
    therefore POST-final-norm in this project's transformers (the trap
    methods/logit_lens.py documents), which would make dla.py's frozen scale
    wrong by exactly the factor it is supposed to be. A hook is also the only
    way to read this under an active patch."""
    layers = adapter.get_decoder_layers(model)
    B = positions.shape[0]

    def post(mod, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        sink.append(torch.stack([t[i, positions[i]] for i in range(B)]).detach())

    return layers[block].register_forward_hook(post)


def _final_residual_hook(adapter, model, positions, sink):
    """The last block's output -- see _residual_hook."""
    return _residual_hook(adapter, model, len(adapter.get_decoder_layers(model)) - 1, positions, sink)


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
    # Patches register BEFORE the capture hooks, deliberately: PyTorch runs a module's forward
    # pre-hooks in registration order, so a capture registered first would read the very tensor the
    # patch is about to overwrite. That is invisible while the patch sits on a different module (the
    # image patch is a post-hook on an earlier block, so it has already run either way), but it is
    # the whole point the moment something captures at the SAME site it patches -- which is exactly
    # what the read-back identity below does.
    layers = adapter.get_decoder_layers(model)
    handles = []
    for site, layer_idx, fn in patches:
        handles.extend(site.register(adapter, model, layers, layer_idx, fn))
    sinks = {b: [] for b in blocks}
    handles += [HEAD_SITE.register_capture(adapter, model, b + 1, positions, sinks[b]) for b in blocks]
    final_sink = []
    handles.append(_final_residual_hook(adapter, model, positions, final_sink))
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


def capture_image_source(adapter, model, batch_img, patch_layer):
    """The source flag's residual stream at the image positions. Hoisted out of
    image_patch so a run that builds the same patch for a dozen knockout
    configurations pays for ONE source forward per batch instead of a dozen."""
    return RESIDUAL_SITE.capture(adapter, model, patch_layer, batch_img["source_input_ids"],
                                 batch_img["attention_mask"], batch_img["source_extra"],
                                 batch_img["positions"])


def image_patch(adapter, model, batch_img, patch_layer, src=None):
    """-> (site, layer, patch_fn) replacing the base's residual stream with the
    source's at the image positions -- the intervention ceiling_sweep already
    measured at ~100% cause below the handoff, and the thing every phase
    observes the downstream consequences of."""
    from methods.ndm.verify_sites import hard_mask
    if src is None:
        src = capture_image_source(adapter, model, batch_img, patch_layer)
    one = hard_mask(RESIDUAL_SITE.width(adapter, model), 1.0, model.device)
    fn = make_cache_aware_patch_hook(batch_img["positions"], lambda base_vals: one(base_vals, src))
    return (RESIDUAL_SITE, patch_layer, fn)


def head_patches(selected_heads, z_by_block, last_positions, hidden, head_dim, device, telemetry=None):
    """-> [(HEAD_SITE, block+1, patch_fn)] writing z_by_block[block]'s values
    into exactly `selected_heads`' 128-dim slices at the last-token column,
    leaving every other head of that block alone.

    Used in BOTH directions, which is the whole point of it being one function:

      KNOCKOUT (necessity)   image patched, z_by_block = the CLEAN base capture
                             -> "make these heads behave as if the image had
                             not been swapped". Does the effect survive?
      SUFFICIENCY            image NOT patched, z_by_block = the PATCHED
                             capture -> "make only these heads behave as if it
                             had". Does the effect appear?

    Necessity is the weaker question: redundancy makes genuinely important
    heads look unnecessary, which is why the knockout has to be cumulative.
    Sufficiency does not have that problem -- if a small set reproduces the
    effect on its own, that is localization, whatever the knockout says."""
    per_block = {}
    for b, h in selected_heads:
        per_block.setdefault(b, []).append(h)
    out = []
    for b, heads in per_block.items():
        mask = head_mask(hidden, head_dim, heads, device)
        vals = z_by_block[b].to(device)
        fn = make_cache_aware_patch_hook(last_positions, lambda bv, _m=mask, _s=vals: _m(bv, _s))
        if telemetry is not None:
            # Wraps the patch to record whether it is ever REACHED and whether it actually CHANGES
            # anything. Three outcomes, three different bugs:
            #   prefill_calls == 0  -> the hook is not firing on a multi-token tensor at all
            #   max_delta == 0      -> it fires and writes values identical to what is already there
            #   both nonzero        -> it fires and writes, and the output still does not move,
            #                          i.e. the patched tensor is not the one the block consumes
            def traced(hs, _f=fn, _b=b):
                rec = telemetry.setdefault(_b, {"prefill_calls": 0, "decode_skips": 0,
                                                 "max_delta": 0.0, "seq_len": None})
                out_hs = _f(hs)
                if hs.shape[1] == 1:
                    rec["decode_skips"] += 1
                else:
                    rec["prefill_calls"] += 1
                    rec["seq_len"] = int(hs.shape[1])
                    rec["max_delta"] = max(rec["max_delta"],
                                            float((out_hs - hs).abs().max().item()))
                return out_hs
            fn = traced
        out.append((HEAD_SITE, b + 1, fn))
    return out


def probe_identity(adapter, model, blocks, entry_block, positions, input_ids, attention_mask, extra,
                    patches=()):
    """One forward pass under `patches`, capturing at `positions`: o_proj's
    INPUT and o_proj's OUTPUT at every traced block, the residual entering
    `entry_block`, and the final pre-norm residual.

    Capturing BOTH sides of o_proj is the point. A forward PRE-hook that
    rewrites a module's input and a capture pre-hook registered after it will
    agree with each other whether or not the module ever consumes the rewrite
    -- PyTorch threads `args` through the hook chain, so the second hook reads
    the first hook's output, not the tensor the module is called with. Checking
    o_proj's OUTPUT against W_O @ (its captured input) is what actually closes
    that loop."""
    layers = adapter.get_decoder_layers(model)
    handles = []
    for site, layer_idx, fn in patches:
        handles.extend(site.register(adapter, model, layers, layer_idx, fn))
    z_sinks, o_sinks = {b: [] for b in blocks}, {b: [] for b in blocks}
    for b in blocks:
        handles.append(HEAD_SITE.register_capture(adapter, model, b + 1, positions, z_sinks[b]))
        handles.append(ATTN_OUT_SITE.register_capture(adapter, model, b + 1, positions, o_sinks[b]))
    entry_sink, final_sink = [], []
    handles.append(_residual_hook(adapter, model, entry_block - 1, positions, entry_sink))
    handles.append(_final_residual_hook(adapter, model, positions, final_sink))
    try:
        extra_dev = extra_to_device(extra, model.device, model.dtype)
        with torch.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                  **extra_dev, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return ({b: z_sinks[b][0] for b in blocks}, {b: o_sinks[b][0] for b in blocks},
            entry_sink[0], final_sink[0])


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
    ap.add_argument("--skip_knockout", action="store_true",
                    help="Phase 1 only -- skips BOTH generation phases (2 and 3).")
    ap.add_argument("--skip_sufficiency", action="store_true",
                    help="Run the knockout (phase 2) but not the sufficiency arm (phase 3). Phase 3 is "
                         "the stronger of the two -- necessity is what redundancy breaks -- so skip it "
                         "only for the generation budget.")
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
    # diagnostic=True: attn_head_output is not a trainable NDM site, and this is a read-only probe.
    # Without it this crashed on the NDM_SITES assert before loading the model.
    log_dir = ndm_logs_dir(model_slug, args.entity, args.attribute, 0.0, 0.0, 0.0, 0.0,
                           args.positions, "attn_head_output", pruned, diagnostic=True)
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
        # Retained per batch: phase 2 restores FROM base_z, phase 3 installs FROM patched_z, and
        # img_src saves re-running the source forward for every later configuration.
        base_z_per_batch, patched_z_per_batch, img_src = [], [], []
        patched_final_per_batch = []
        for bi, (b_img, b_last) in enumerate(batches):
            last_pos = b_last["positions"]
            img_src.append(capture_image_source(adapter, model, b_img, args.patch_layer))
            patch = image_patch(adapter, model, b_img, args.patch_layer, src=img_src[-1])

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

            base_z_per_batch.append(base_z)
            patched_z_per_batch.append(patched_z)
            patched_final_per_batch.append(patched_final)
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

        # ---------------- phase 1c: the read-back identity ----------------
        # The self-test that decides whether phases 2 and 3 mean anything, and the only one here that
        # is EXACT rather than thresholded. When --blocks covers every block from --patch_layer to the
        # last, installing the patched per-head values at the last token is not an approximation of the
        # image patch -- at that position it is algebraically THE SAME RUN:
        #
        #   * the image patch is at IMAGE positions, so the last token's residual entering the first
        #     traced block is bit-identical in both runs (earlier blocks read unpatched image K/V);
        #   * attention is the only cross-position operation, and each downstream block's attention
        #     write at the last token is exactly the value being installed;
        #   * MLPs are position-wise, so they recompute correctly from the updated residual.
        #
        # So the final residual at the last token MUST match, and phases 2/3's k=all arms must
        # reproduce the image patch's own cause. Two different faults can break that, and the
        # per-block read-back separates them cleanly: install v at block b, then read block b back
        # through the SAME capture hook. If it does not return v, the capture and the patch are not
        # addressing the same tensor/column -- plumbing, phases 2-3 void. If every block reads back
        # exactly and the final residual STILL diverges, the cause is --blocks coverage (or, with
        # complete coverage, a real bug). Costs one forward pass on one batch, and no generation.
        b_img0, b_last0 = batches[0]
        pos0 = b_last0["positions"]
        ids0, mask0, extra0 = b_img0["base_input_ids"], b_img0["attention_mask"], b_img0["base_extra"]
        every_head = [(b, h) for b in blocks for h in range(n_heads)]
        clean_z, clean_o, clean_entry, clean_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0)
        pat_z, pat_o, pat_entry, pat_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0,
            patches=[image_patch(adapter, model, b_img0, args.patch_layer, src=img_src[0])])
        inst_z, inst_o, inst_entry, inst_final = probe_identity(
            adapter, model, blocks, blocks[0], pos0, ids0, mask0, extra0,
            patches=head_patches(every_head, pat_z, pos0, hidden, head_dim, model.device))

        def _rel(got, want):
            want = want.float()
            return (got.float() - want).norm().item() / max(want.norm().item(), 1e-9)

        def _o_proj_rel(run_z, run_o, b):
            """o_proj's captured OUTPUT vs W_O @ its captured INPUT. ~0 means the module really was
            called with the tensor we captured; large means it was not."""
            mod = adapter.get_attn_head_output_module(model, b)
            expect = run_z[b].float() @ mod.weight.float().T
            if mod.bias is not None:
                expect = expect + mod.bias.float()
            return _rel(run_o[b], expect)

        print(f"\n{'='*78}\n=== phase 1c: read-back identity (batch 1, 3 forwards, no generation)\n{'='*78}")

        # (i) THE INDUCTION'S BASE CASE. The image patch is at image positions, so the read column's
        #     residual entering the first traced block must be untouched by it. If this is not ~0 the
        #     identity does not apply at all and nothing below it means anything.
        entry_rel = _rel(pat_entry, clean_entry)
        print(f"  residual entering block {blocks[0]} at the read column, image-patched vs clean: "
              f"{entry_rel:.3%}"
              f"{'  (as required)' if entry_rel < 0.01 else '   !! should be ~0 -- the patch reaches '
                'the read column BEFORE the traced blocks, so the identity does not apply'}")

        # (ii) READ-BACK, and (iii) the check that makes the read-back mean something. A forward
        #      PRE-hook that rewrites o_proj's input and a capture pre-hook registered after it agree
        #      with each other whether or not o_proj consumes the rewrite, because PyTorch threads
        #      `args` down the hook chain. Comparing o_proj's OUTPUT against W_O @ (captured input)
        #      is what closes that loop. The clean run is the control: it has no patch, so its
        #      residual MUST be ~0, and a large value there means this check itself is wrong.
        worst, worst_b = 0.0, blocks[0]
        for b in blocks:
            r = _rel(inst_z[b], pat_z[b])
            if r >= worst:
                worst, worst_b = r, b
        print(f"\n  o_proj INPUT reads back what was installed: worst {worst:.2e} (block {worst_b})")
        print(f"  {'block':>7} {'clean run':>12} {'installed run':>15}   (o_proj OUTPUT vs W_O @ its captured INPUT)")
        cons_clean = {b: _o_proj_rel(clean_z, clean_o, b) for b in blocks}
        cons_inst = {b: _o_proj_rel(inst_z, inst_o, b) for b in blocks}
        for b in blocks:
            print(f"  {b:>7} {cons_clean[b]:>11.2%} {cons_inst[b]:>14.2%}")
        control_ok = max(cons_clean.values()) < 0.05
        consumed = max(cons_inst.values()) < 0.05

        # (iv) The identity itself, stated numerically -- and the one comparison that separates
        #      "the patch did the wrong thing" from "the patch did nothing".
        to_patched, to_clean = _rel(inst_final, pat_final), _rel(inst_final, clean_final)
        downstream = set(range(args.patch_layer, n_layers))
        missing = sorted(downstream - set(blocks))
        print(f"\n  final pre-norm residual at the read column, installed-heads run vs:")
        print(f"    the image-patched run: {to_patched:.3%}   <-- the identity; must be ~0")
        print(f"    the CLEAN run:         {to_clean:.3%}   <-- ~0 means the patch changed NOTHING")

        if not control_ok:
            print(f"  -> the CLEAN control fails ({max(cons_clean.values()):.1%}), so this check is "
                  f"itself wrong: o_proj's output is not W_O @ the tensor register_capture reads. "
                  f"Fix the check before reading anything else.")
        elif not consumed:
            print(f"  -> FOUND IT: o_proj's input reads back perfectly but its OUTPUT does not match "
                  f"W_O @ that input, while the unpatched control is exact. The block is NOT "
                  f"consuming the rewritten input -- the forward PRE-hook's returned args reach the "
                  f"later capture hook and nothing else. The read-back was fooled by hook ordering, "
                  f"and every head-level number in phases 1b, 2 and 3 is void.")
        elif to_clean < 0.01:
            print(f"  -> the patch is CONSUMED by o_proj yet the final residual is unchanged from "
                  f"clean. That cannot happen through this site alone; look for the head patch being "
                  f"removed or overwritten before the blocks that matter run.")
        elif to_patched < 0.02:
            print(f"  -> MATCHES. The intervention is sound, so phases 2 and 3's k=all arms MUST "
                  f"reproduce the image patch's cause; if they do not the fault is in generation or "
                  f"scoring, not the patch.")
        elif missing:
            print(f"  -> diverges, and --blocks is missing downstream block(s) {missing}, which "
                  f"explains it: their attention at the read column still sees the CLEAN image. "
                  f"Re-run with --blocks {' '.join(str(b) for b in sorted(downstream))}.")
        else:
            print(f"  -> diverges with complete coverage ({args.patch_layer}..{n_layers - 1}) and a "
                  f"consumed patch, so the installed values are being applied but are not the whole "
                  f"of what the image patch changes at this column. Phases 2 and 3 are void.")
        report["phase1c"] = {"entry_residual_rel": entry_rel, "readback_worst_rel": worst,
                             "o_proj_consumption_clean": cons_clean, "o_proj_consumption_installed": cons_inst,
                             "final_vs_patched_rel": to_patched, "final_vs_clean_rel": to_clean,
                             "blocks_missing_downstream": missing}

        if args.skip_knockout:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nwrote {out_path}  (phase 2 skipped)")
            return

        def run_cause(selected_heads, z_per_batch, with_image_patch, telemetry=None):
            """cause / base_kept with `selected_heads` overwritten at the last
            token from z_per_batch, optionally under the image patch.

              with_image_patch=True,  z = base_z    -> KNOCKOUT (necessity)
              with_image_patch=False, z = patched_z -> SUFFICIENCY

            One function for both so the two curves cannot drift apart in the
            details -- same rows, same masks, same scoring, same generation."""
            ms, mb, n = 0.0, 0.0, 0
            for (b_img, b_last), z, src in zip(batches, z_per_batch, img_src):
                patches = ([image_patch(adapter, model, b_img, args.patch_layer, src=src)]
                           if with_image_patch else [])
                patches += head_patches(selected_heads, z, b_last["positions"], hidden, head_dim,
                                        model.device, telemetry=telemetry)
                gen = generate_with_patches(adapter, model, patches, b_img["base_input_ids"],
                                            b_img["attention_mask"], b_img["base_extra"], pad_token_id,
                                            args.max_new_tokens)
                s, t = score_generation(gen, b_img)
                k = len(b_img["rows"])
                ms, mb, n = ms + s * k, mb + t * k, n + k
            return ms / n, mb / n

        # ---------------- phase 1b: connectivity + the block-everything arm ----------------
        # TWO different questions, which an earlier version of this conflated into one bad test.
        #
        # (1) CONNECTIVITY is about the hook, and the only honest evidence is the telemetry below:
        #     did the patch fn get reached on a multi-token tensor, and did it change the tensor.
        #     Generated TEXT is a thresholded readout and absorbs large perturbations without
        #     moving -- verify_sites already showed a full attn_head_output swap shifting the
        #     logits by 0.25 while leaving the generation identical. Asserting on text here
        #     produced a confident "HEAD PATCH IS NOT CONNECTED" for a patch that was working.
        #
        # (2) The BLOCK-EVERYTHING arm is a finding, not a precondition. With the image patched and
        #     every traced head zeroed at the last token, no image information can reach that
        #     position after the patch layer -- attention is the only cross-position operation and
        #     MLPs are position-wise. So cause SHOULD collapse to the unhooked floor. If it does
        #     not, the read is happening somewhere --blocks does not cover (widen it), or it is
        #     not happening through the last token's attention at all, which would be the real
        #     result and would overturn the handoff picture.
        tel = {}
        zero_z = [{b: z[b] * 0.0 for b in blocks} for z in base_z_per_batch]
        all_heads = [(b, h) for b in blocks for h in range(n_heads)]
        print(f"\n{'='*78}\n=== phase 1b: connectivity + block-everything\n{'='*78}")
        blocked_ms, blocked_mb = run_cause(all_heads, zero_z, True, telemetry=tel)

        print(f"  patch telemetry (did the hook fire, and did it write anything?):")
        for b in blocks:
            r = tel.get(b)
            if r is None:
                print(f"    block {b:>2}: PATCH FN NEVER CALLED -- the hook was never reached")
            else:
                print(f"    block {b:>2}: prefill_calls={r['prefill_calls']} "
                      f"decode_skips={r['decode_skips']} seq_len={r['seq_len']} "
                      f"max|delta|={r['max_delta']:.4f}")
        reached = sum(r["prefill_calls"] for r in tel.values())
        wrote = sum(1 for r in tel.values() if r["max_delta"] > 0)
        assert reached > 0, (
            "HEAD PATCH NEVER REACHED: the patch fn was not called on a multi-token tensor, so the "
            "hook is not attached to a module that runs during prefill. Phases 2 and 3 are void.")
        assert wrote == len(tel), (
            f"HEAD PATCH WROTE NOTHING at {len(tel) - wrote} of {len(tel)} blocks -- it ran but "
            f"returned the tensor unchanged, so `positions` is pointing at columns that already "
            f"hold these values. Phases 2 and 3 are void.")
        print(f"  -> connected: reached on {reached} prefill call(s), changed the tensor at all "
              f"{wrote} blocks")

        print(f"\n  image patch + ALL {len(all_heads)} heads ZEROED: cause={blocked_ms:6.1%} "
              f"base_kept={blocked_mb:6.1%}")
        print(f"  (compare against the image-patch-only row in phase 2. Zeroing every traced head "
              f"at the last token severs every path by which the patched image can reach that "
              f"position after block {blocks[0]}, so cause here SHOULD fall to the unhooked floor. "
              f"If it does not, either --blocks is too narrow or the read does not go through the "
              f"last token's attention -- and that would be the finding.)")
        report["phase1b"] = {"telemetry": {str(b): tel.get(b) for b in blocks},
                             "blocked_all_cause": blocked_ms, "blocked_all_base_kept": blocked_mb}

        # ---------------- phase 2: cumulative knockout ----------------
        print(f"\n{'='*78}\n=== phase 2: cumulative knockout under the image patch (path patching)\n{'='*78}")

        unhooked_ms, unhooked_mb, n = 0.0, 0.0, 0
        for b_img, _ in batches:
            gen = generate_unhooked(model, b_img["base_input_ids"], b_img["attention_mask"],
                                    b_img["base_extra"], pad_token_id, args.max_new_tokens)
            s, t = score_generation(gen, b_img)
            k = len(b_img["rows"])
            unhooked_ms, unhooked_mb, n = unhooked_ms + s * k, unhooked_mb + t * k, n + k
        print(f"  unhooked:                       cause={unhooked_ms / n:6.1%} base_kept={unhooked_mb / n:6.1%}")
        full_ms, full_mb = run_cause([], base_z_per_batch, True)
        print(f"  image patch only (k=0):         cause={full_ms:6.1%} base_kept={full_mb:6.1%}"
              f"   <-- the effect being traced")
        if full_ms < 0.10:
            print(f"  !! the image patch at layer {args.patch_layer} barely moves the answer, so there is "
                  f"nothing downstream to knock out. Pick a --patch_layer BELOW the handoff (check "
                  f"ceiling_sweep's image-position column) before reading anything below.")

        # Drawn ONCE, before either phase, so phases 2 and 3 null against the identical head set.
        rand_heads = (random.Random(args.seed + 1)
                      .sample([(r["block"], r["head"]) for r in table], min(args.n_random, len(table)))
                      if args.n_random else [])

        knock = []
        for k in args.knockout_ks:
            if k > len(ranked):
                continue
            heads = [(r["block"], r["head"]) for r in ranked[:k]]
            ms, mb = run_cause(heads, base_z_per_batch, True)
            knock.append({"k": k, "kind": "top", "heads": heads, "cause": ms, "base_kept": mb})
            print(f"  restore top-{k:<3} heads:            cause={ms:6.1%} base_kept={mb:6.1%}  "
                  f"(recovered {max(full_ms - ms, 0) / max(full_ms, 1e-9):5.1%} of the effect)", flush=True)

        if args.n_random:
            ms, mb = run_cause(rand_heads, base_z_per_batch, True)
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

        # ---------------- phase 3: sufficiency ----------------
        if not args.skip_sufficiency:
            print(f"\n{'='*78}\n=== phase 3: sufficiency -- patch ONLY these heads, no image patch\n{'='*78}")
            print(f"  The mirror of phase 2. Instead of removing a head's contribution from a fully "
                  f"patched run, this INSTALLS it into an otherwise clean one: the image is NOT "
                  f"swapped, and the selected heads are forced to the values they took when it was. "
                  f"Necessity (phase 2) is the question redundancy breaks; sufficiency is not, so a "
                  f"small set reproducing the effect here is localization even if the knockout curve "
                  f"is flat.")
            suff = []
            z0_ms, z0_mb = run_cause([], patched_z_per_batch, False)
            print(f"\n  nothing patched (k=0):          cause={z0_ms:6.1%} base_kept={z0_mb:6.1%}"
                  f"   <-- must match the unhooked row above")
            if abs(z0_ms - unhooked_ms / n) > 1e-9:
                print(f"  !! k=0 differs from unhooked ({z0_ms:.1%} vs {unhooked_ms / n:.1%}) -- a patch "
                      f"is leaking when no heads are selected; every number below is suspect.")
            all_heads = [(r["block"], r["head"]) for r in table]
            ceil_ms, ceil_mb = run_cause(all_heads, patched_z_per_batch, False)
            # What this arm SHOULD read depends entirely on coverage, and conflating the two cases
            # is how a broken run reads as a finding. With every downstream block traced it is the
            # closed identity phase 1c checks -- it must reproduce the image patch's own cause, so
            # anything less is a fault, not a ceiling. With blocks missing it is a genuine partial
            # ceiling, because the untraced blocks' attention still reads the clean image.
            if missing:
                print(f"  ALL {len(all_heads)} traced heads:           cause={ceil_ms:6.1%} "
                      f"base_kept={ceil_mb:6.1%}   <-- a PARTIAL ceiling: downstream block(s) "
                      f"{missing} are not traced, so their attention still reads the clean image")
            else:
                print(f"  ALL {len(all_heads)} traced heads:           cause={ceil_ms:6.1%} "
                      f"base_kept={ceil_mb:6.1%}   <-- --blocks covers every downstream block, so "
                      f"this MUST equal the image-patch-only cause ({full_ms:.1%})")
                if abs(ceil_ms - full_ms) > 0.05:
                    print(f"  !! it does not ({ceil_ms:.1%} vs {full_ms:.1%}), and phase 1c says the "
                          f"final residual at the last token {'matches' if to_patched < 0.02 else 'does not match'}. "
                          f"Every top-k row below is therefore meaningless -- fix this first.")
            suff.append({"k": len(all_heads), "kind": "all", "cause": ceil_ms, "base_kept": ceil_mb})
            for k in args.knockout_ks:
                if k > len(ranked):
                    continue
                heads = [(r["block"], r["head"]) for r in ranked[:k]]
                ms, mb = run_cause(heads, patched_z_per_batch, False)
                suff.append({"k": k, "kind": "top", "heads": heads, "cause": ms, "base_kept": mb})
                print(f"  patch top-{k:<3} heads:              cause={ms:6.1%} base_kept={mb:6.1%}  "
                      f"({ms / max(ceil_ms, 1e-9):5.1%} of this arm's ceiling)", flush=True)
            if args.n_random:
                ms, mb = run_cause(rand_heads, patched_z_per_batch, False)
                suff.append({"k": args.n_random, "kind": "random", "heads": rand_heads, "cause": ms,
                             "base_kept": mb})
                print(f"  patch {args.n_random} RANDOM heads (null):     cause={ms:6.1%} "
                      f"base_kept={mb:6.1%}   <-- same heads as phase 2's null")
                top_same = next((x for x in suff if x["kind"] == "top" and x["k"] == args.n_random), None)
                if top_same is not None:
                    print(f"\n  top-{args.n_random} vs random-{args.n_random}: "
                          f"{top_same['cause']:.1%} vs {ms:.1%} of cause.")
            report["phase3_sufficiency"] = {"ceiling_all_traced_heads": ceil_ms, "k0_cause": z0_ms,
                                            "arms": suff}

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
