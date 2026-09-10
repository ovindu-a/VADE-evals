"""Correctness harness for common/sites.py's intervention sites. Run this
BEFORE spending GPU time on a real NDM training run -- it needs one model
load and a handful of forward passes (about a GPU-minute), and it catches
the failure mode that matters: a hook attached to the wrong tensor still
"works", trains, and produces plausible-looking numbers.

It also covers `residual`, so it doubles as a regression check that adding
the site abstraction did not change DBM's behavior.

The four checks, per site:

  1. WIDTH -- the captured source activation is [B, n_pos, expected_width]:
     hidden_size (3584) for residual/mlp_output, intermediate_size (18944)
     for mlp_hidden. Catches hooking `mlp` when you meant `mlp.down_proj`.

  2. NULL PATCH IS IDENTITY -- with the mask driven to sigma(m/T)=0 (pure
     base), free-running generation must match an unhooked run token for
     token. This is the single most informative check: it fails if the hook
     is on the wrong module, mutates the tensor in place, breaks the
     tuple/kwargs contract of a pre-hook, or corrupts dtype.

  3. FULL SWAP OF BASE-INTO-BASE IS IDENTITY -- with sigma(m/T)=1 (pure
     source) but the "source" activation captured from the BASE input,
     generation must again match the unhooked run. Check 2 only exercises
     the (1-s) branch; this one exercises the s branch, and it needs no
     independently-computed ground truth -- patching a value in over itself
     must be a no-op by construction. Together, 2 and 3 pin both branches.

  4. DECODE-STEP GUARD -- make_cache_aware_patch_hook must return a
     [B, 1, width] tensor untouched, since during generate()'s incremental
     decode steps the prefill patch is already baked into the KV cache and
     indexing absolute `positions` into a 1-column tensor would be wrong
     rather than merely redundant. Pure function test, no model needed.

A NOTE ON EXACTNESS. Checks 2 and 3 compare generated token ids, not raw
activations, because batched matmul kernel selection makes even a
mathematically exact identity drift in the last bits (see
common/source_cache.py's module docstring, which documents this same effect
for the source cache). Greedy decoding is robust to that drift; if token ids
still differ, the hook is wrong, not noisy. Max absolute activation
difference is printed either way so you can see the magnitude.

Usage:
    python methods/ndm/verify_sites.py --entity flags --attribute language --layer 16
    python methods/ndm/verify_sites.py --entity flags --attribute language --layer 16 \\
        --sites mlp_hidden            # just the one, if the others already passed
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch  # noqa: E402

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import BuildBatchCache, build_batch, load_entity_assets, load_tuples  # noqa: E402
from methods.common.hooks import extra_to_device, make_cache_aware_patch_hook  # noqa: E402
from methods.common.sites import SITES, InterventionSite  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS  # noqa: E402
from methods.dbm.intervention import SigmoidMaskIntervention  # noqa: E402

# sigma(m/T) is exactly 0.0 / 1.0 in float for |m|=1 at T=1e-7 -- no need for infinities, which would
# make the mask's own L1 term nan if this harness ever grew into something that also trained.
HARD_TEMPERATURE = 1e-7


def generate_unhooked(model, input_ids, attention_mask, extra, pad_token_id, max_new_tokens):
    """The no-intervention baseline that checks 2 and 3 compare against.
    Deliberately does NOT go through sites.generate_patched with a no-op
    patch -- that would be comparing the hook path to itself."""
    extra_dev = extra_to_device(extra, model.device, model.dtype)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
            **extra_dev, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id,
        )
    return gen[:, input_ids.shape[1]:].cpu()


def hard_mask(embed_dim, value, device):
    """A SigmoidMaskIntervention whose sigmoid is saturated to exactly 0.0
    (value<0, pure base) or exactly 1.0 (value>0, pure source)."""
    intervention = SigmoidMaskIntervention(embed_dim=embed_dim).to(device)
    with torch.no_grad():
        intervention.mask.fill_(value)
    intervention.set_temperature(torch.tensor(HARD_TEMPERATURE))
    intervention.eval()
    sat = torch.sigmoid(intervention.mask / intervention.temperature.detach()).unique().tolist()
    assert sat in ([0.0], [1.0]), f"mask did not saturate: sigmoid values {sat}"
    return intervention


def check_site(site_name, adapter, model, processor, batch, layer, pad_token_id, max_new_tokens, gen_clean):
    site = InterventionSite(site_name)
    layers = adapter.get_decoder_layers(model)
    positions = batch["positions"]
    expected_width = site.width(adapter, model)
    failures = []

    print(f"\n=== site={site_name} layer={layer} expected_width={expected_width} ===")

    # 1. WIDTH -----------------------------------------------------------------
    source_act = site.capture(adapter, model, layer, batch["source_input_ids"], batch["attention_mask"],
                               batch["source_extra"], positions)
    want_shape = (positions.shape[0], positions.shape[1], expected_width)
    if tuple(source_act.shape) == want_shape:
        print(f"[ok]   1. width: captured source activation {tuple(source_act.shape)}")
    else:
        failures.append(f"width: captured {tuple(source_act.shape)}, expected {want_shape}")
        print(f"[FAIL] 1. width: captured {tuple(source_act.shape)}, expected {want_shape}")
        return failures  # everything below assumes the right tensor; no point continuing

    # 2. NULL PATCH IS IDENTITY ------------------------------------------------
    zero = hard_mask(expected_width, -1.0, model.device)
    patch_fn = make_cache_aware_patch_hook(positions, lambda base_vals: zero(base_vals, source_act))
    gen_null = site.generate_patched(adapter, model, layers, layer, patch_fn, batch["base_input_ids"],
                                      batch["attention_mask"], batch["base_extra"], pad_token_id, max_new_tokens)
    if torch.equal(gen_null, gen_clean):
        print("[ok]   2. null patch (sigma=0) reproduces the unhooked generation exactly")
    else:
        failures.append("null patch (sigma=0) changed the generation -- hook is on the wrong tensor "
                        "or is corrupting it")
        print("[FAIL] 2. null patch (sigma=0) changed the generation")
        print(f"         unhooked: {processor.tokenizer.batch_decode(gen_clean, skip_special_tokens=True)}")
        print(f"         sigma=0 : {processor.tokenizer.batch_decode(gen_null, skip_special_tokens=True)}")

    # 3. FULL SWAP OF BASE-INTO-BASE IS IDENTITY -------------------------------
    base_act = site.capture(adapter, model, layer, batch["base_input_ids"], batch["attention_mask"],
                             batch["base_extra"], positions)
    drift = (base_act.float() - source_act.float()).abs().max().item()
    print(f"       (base vs source activation max|diff| = {drift:.4g} -- should be clearly nonzero, "
          f"else the two inputs aren't actually different)")
    one = hard_mask(expected_width, 1.0, model.device)
    patch_fn = make_cache_aware_patch_hook(positions, lambda base_vals: one(base_vals, base_act))
    gen_self = site.generate_patched(adapter, model, layers, layer, patch_fn, batch["base_input_ids"],
                                      batch["attention_mask"], batch["base_extra"], pad_token_id, max_new_tokens)
    if torch.equal(gen_self, gen_clean):
        print("[ok]   3. full swap (sigma=1) of base-into-base reproduces the unhooked generation exactly")
    else:
        failures.append("full swap (sigma=1) of base-into-base changed the generation -- capture and patch "
                        "disagree about which tensor/positions they address")
        print("[FAIL] 3. full swap (sigma=1) of base-into-base changed the generation")
        print(f"         unhooked: {processor.tokenizer.batch_decode(gen_clean, skip_special_tokens=True)}")
        print(f"         sigma=1 : {processor.tokenizer.batch_decode(gen_self, skip_special_tokens=True)}")

    # 4. DECODE-STEP GUARD -----------------------------------------------------
    single_col = torch.zeros(positions.shape[0], 1, expected_width, device=model.device, dtype=model.dtype)
    passed_through = patch_fn(single_col)
    if passed_through is single_col or torch.equal(passed_through, single_col):
        print("[ok]   4. decode-step guard: [B, 1, width] tensor passes through untouched")
    else:
        failures.append("decode-step guard did not pass a [B, 1, width] tensor through unchanged")
        print("[FAIL] 4. decode-step guard modified a [B, 1, width] tensor")

    # Informational: a REAL swap should actually change something. Not a pass/fail (a given layer/site
    # genuinely may not steer this attribute -- that's the research question, not a bug), but a site
    # where the full source swap changes NOTHING at all is worth knowing about before you train on it.
    #
    # Report the FIRST DIFFERING TOKEN INDEX per row, not just token inequality. An earlier version
    # printed "CHANGES the generation" for any difference at all, which was actively misleading: on
    # flags/language layer 14 both MLP sites' full swap differed from the unhooked run only at token
    # index 7 (" ...the flag in the image is" -> "...features"), i.e. a trailing filler word with the
    # ANSWER untouched -- while the residual site differed at index 1 (" Arabic." -> " English, ...")
    # which is a real answer flip. Those two cases are not the same finding, and index<MAX_ANSWER_TOKENS
    # separates them, since the answer occupies exactly the first MAX_ANSWER_TOKENS tokens (see
    # common/targets.py). For a proper headroom NUMBER across many layers, use ceiling_sweep.py --
    # this line is a smoke signal, not a measurement.
    patch_fn_real = make_cache_aware_patch_hook(positions, lambda base_vals: one(base_vals, source_act))
    gen_swapped = site.generate_patched(adapter, model, layers, layer, patch_fn_real, batch["base_input_ids"],
                                         batch["attention_mask"], batch["base_extra"], pad_token_id, max_new_tokens)
    clean_txt = processor.tokenizer.batch_decode(gen_clean, skip_special_tokens=True)
    swap_txt = processor.tokenizer.batch_decode(gen_swapped, skip_special_tokens=True)
    print("       (info) full swap of the REAL source, per row:")
    for i in range(gen_clean.shape[0]):
        diff = (gen_clean[i] != gen_swapped[i]).nonzero(as_tuple=True)[0]
        if len(diff) == 0:
            verdict = "IDENTICAL -- this site/layer has no causal effect on this row at all"
        elif int(diff[0]) < MAX_ANSWER_TOKENS:
            verdict = f"ANSWER CHANGED (first diff at token {int(diff[0])})"
        else:
            verdict = (f"answer UNCHANGED -- only trailing token {int(diff[0])} onward differs, so the "
                       f"swap landed but did not move the answer")
        print(f"         row {i}: {verdict}")
        print(f"           unhooked   : {clean_txt[i]!r}")
        print(f"           full source: {swap_txt[i]!r}")

    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layer", type=int, required=True, help="Must be >=1 for the MLP sites.")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n_rows", type=int, default=2, help="Rows in the single test batch (default 2).")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    entity_assets = load_entity_assets(args.vade_root, args.entity)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    rows = load_tuples(entity_assets, args.attribute, args.split)[:args.n_rows]
    for r in rows:
        r.setdefault("target_attribute", args.attribute)
    assert rows, f"no rows loaded for entity={args.entity} attribute={args.attribute} split={args.split}"
    batch = build_batch(rows, entity_assets, adapter, model, processor, args.positions,
                         batch_cache=BuildBatchCache())
    print(f"[verify_sites] entity={args.entity} attribute={args.attribute} layer={args.layer} "
          f"positions={args.positions} rows={len(rows)} n_pos={batch['positions'].shape[1]}")
    print(f"[verify_sites] hidden_size={adapter.hidden_size(model)} "
          f"intermediate_size={adapter.intermediate_size(model)}")

    gen_clean = generate_unhooked(model, batch["base_input_ids"], batch["attention_mask"], batch["base_extra"],
                                   pad_token_id, args.max_new_tokens)

    all_failures = {}
    for site_name in args.sites:
        if site_name != "residual" and args.layer < 1:
            print(f"\n=== site={site_name} SKIPPED: needs --layer >= 1 ===")
            continue
        all_failures[site_name] = check_site(site_name, adapter, model, processor, batch, args.layer,
                                              pad_token_id, args.max_new_tokens, gen_clean)

    print("\n================ SUMMARY ================")
    ok = True
    for site_name, failures in all_failures.items():
        if failures:
            ok = False
            print(f"{site_name}: {len(failures)} FAILURE(S)")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"{site_name}: all checks passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
