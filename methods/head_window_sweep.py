"""Stage 2 of the image->text head pipeline: FULL attention-block patches over
candidate block windows, all in one model load.

WHERE THIS SITS. methods/image_head_pipeline.py runs three stages per
(entity, attribute):

  1. ndm/ceiling_sweep.py  -- full residual swap at the image positions and at
                              the last token, every layer. The image curve
                              falling while the last-token curve rises brackets
                              the handoff and fixes --patch_layer.
  2. THIS SCRIPT           -- inside that bracket, which attention BLOCKS carry
                              the image into the text stream? Every head of a
                              window patched at once.
  3. head_trace.py         -- inside the best window, which HEADS.

WHAT ONE WINDOW ARM IS. Exactly head_trace.py's phase-3 "ALL traced heads"
arm for `--blocks <window>`, and its phase-2 restore-everything arm, computed
for many windows off ONE capture. That works because the per-head values
installed at the last token come from an image-patched run with NO head
patches active, so they do not depend on which blocks are traced -- capturing
the union of every window's blocks once and slicing is identical to running
head_trace once per window, minus a model load and a phase-1 pass per window.
The flags sweep (scripts/run_head_trace_sweep.sh) paid for 19 model loads and
19 full knockout curves to read one number per window; this reads the same
number, plus the arms that make it interpretable, for all of them at once.

  suff      image NOT patched; every head of the window installed at the last
            token with its image-patched value.  "Is this window enough?"
  nec       image patched; every head of the window restored to its clean
            value. "Does the effect survive without this window?" Weaker than
            suff (redundancy breaks necessity, see head_trace's docstring).
  shuffled  suff with the installed values ROLLED one row across the batch, so
            row i gets row i-1's source image's head outputs. `donor` rising
            while `own` collapses is transfer; `own` staying high means the
            effect never depended on which image was patched.

A FULL-COVERAGE REFERENCE WINDOW (--patch_layer .. last block) is added by
default, and it is a self-test rather than a result: with every downstream
block covered, `suff` is algebraically the image patch itself at the last
token (head_trace's phase-1c identity), so it must reproduce the
image-patch-only row. The script prints whether it does. If not, no other row
means anything.

SCORING, AND WHY FIRST-TOKEN IS REPORTED. Every arm here patches the LAST
PROMPT column only, which controls the first answer token and nothing after it
(see CLAUDE.md's swap_trace section: calling_code reads cause=3.4% but 99.7%
on the first token). So each arm reports

  cause / base_kept / other   the usual up-to-MAX_ANSWER_TOKENS exact match
  first_src / first_base      first answer token == source / base gold's,
                              over rows whose two golds' FIRST tokens differ
                              (a shared leading digit is not transfer)

and methods/image_head_pipeline.py picks windows by first_src by default,
because exact `cause` would rank every multi-token attribute's windows by an
artefact.

Usage:
    python methods/head_window_sweep.py --entity brands --attribute hq_country \\
        --patch_layer 21 --positions full_image --range 21 25 --max_len 3
    python methods/head_window_sweep.py --entity flags --attribute language \\
        --patch_layer 21 --windows 21-23 22 23 22-23
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
from methods.common.position_sets import build_batch_at  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS, exact_match  # noqa: E402
from methods.head_trace import (  # noqa: E402
    capture_head_outputs, capture_image_source, generate_with_patches, head_patches, image_patch,
    per_head_delta,
)
from methods.ndm.config import ndm_logs_dir  # noqa: E402
from methods.ndm.verify_sites import generate_unhooked  # noqa: E402


# ---------------------------------------------------------------- windows

def parse_window(spec):
    """'21-23' -> [21, 22, 23]; '22' -> [22]; '21.23' -> [21, 23] (non-contiguous)."""
    spec = str(spec)
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        assert lo <= hi, f"window {spec!r} is reversed"
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(".")]


def window_label(blocks):
    if blocks == list(range(blocks[0], blocks[-1] + 1)):
        return str(blocks[0]) if len(blocks) == 1 else f"{blocks[0]}-{blocks[-1]}"
    return ".".join(str(b) for b in blocks)


def enumerate_windows(lo, hi, max_len):
    """Every contiguous window of length 1..max_len inside [lo, hi], plus [lo, hi] itself."""
    out = [list(range(s, s + n)) for n in range(1, max_len + 1) for s in range(lo, hi - n + 2)]
    if list(range(lo, hi + 1)) not in out:
        out.append(list(range(lo, hi + 1)))
    return out


# ---------------------------------------------------------------- scoring

class Tally:
    """Row-weighted accumulator for one arm."""

    def __init__(self):
        self.n = self.src = self.base = 0
        self.n_first = self.first_src = self.first_base = 0
        self.donor = self.first_donor = self.n_first_donor = 0

    def add(self, gen, batch, donor_batch=None):
        pred = gen[:, :MAX_ANSWER_TOKENS]
        self.n += pred.shape[0]
        self.src += int(exact_match(pred, batch["source_gold_toks"], batch["source_gold_len"]).sum())
        self.base += int(exact_match(pred, batch["base_gold_toks"], batch["base_gold_len"]).sum())
        s0, b0 = batch["source_gold_toks"][:, 0], batch["base_gold_toks"][:, 0]
        differs = s0 != b0
        self.n_first += int(differs.sum())
        if pred.shape[1]:
            p0 = pred[:, 0]
            self.first_src += int(((p0 == s0) & differs).sum())
            self.first_base += int(((p0 == b0) & differs).sum())
        if donor_batch is not None:
            self.donor += int(exact_match(pred, donor_batch["source_gold_toks"],
                                          donor_batch["source_gold_len"]).sum())
            d0 = donor_batch["source_gold_toks"][:, 0]
            donor_differs = d0 != b0
            self.n_first_donor += int(donor_differs.sum())
            if pred.shape[1]:
                self.first_donor += int(((pred[:, 0] == d0) & donor_differs).sum())

    def rates(self, shuffled=False):
        n, nf = max(self.n, 1), max(self.n_first, 1)
        r = {"n": self.n, "n_first": self.n_first,
             "cause": self.src / n, "base_kept": self.base / n,
             "other": max(0.0, 1.0 - (self.src + self.base) / n),
             "first_src": self.first_src / nf, "first_base": self.first_base / nf}
        if shuffled:
            # `own` is the row's own source gold (what an artefact keeps), `donor` the gold of the
            # row whose values it actually received (what real transfer produces).
            r = {"n": self.n, "n_first": self.n_first, "own": r["cause"], "donor": self.donor / n,
                 "base_kept": r["base_kept"], "first_own": r["first_src"],
                 "first_donor": self.first_donor / max(self.n_first_donor, 1)}
        return r


# ---------------------------------------------------------------- the sweep

def capture(adapter, model, batches, patch_layer, blocks):
    """Per batch: the source image residual at patch_layer, and the per-head
    attn_head_output at the last token for `blocks`, clean and image-patched.
    Same calls, same order as head_trace's phase 1."""
    img_src, base_z, patched_z, finals = [], [], [], []
    for b_img, b_last in batches:
        src = capture_image_source(adapter, model, b_img, patch_layer)
        patch = image_patch(adapter, model, b_img, patch_layer, src=src)
        zb, fb = capture_head_outputs(adapter, model, blocks, b_last["positions"], b_img["base_input_ids"],
                                      b_img["attention_mask"], b_img["base_extra"])
        zp, fp = capture_head_outputs(adapter, model, blocks, b_last["positions"], b_img["base_input_ids"],
                                      b_img["attention_mask"], b_img["base_extra"], patches=[patch])
        img_src.append(src)
        base_z.append(zb)
        patched_z.append(zp)
        finals.append((fb, fp))
    return img_src, base_z, patched_z, finals


def per_block_mass(adapter, model, batches, blocks, base_z, patched_z, finals):
    """Sum over heads of head_trace's delta_resid / delta_dla, per block --
    where the image-patched run's writes to the last token changed. A cheap
    locator, not a causal result (marginal, and blind to conjunctions)."""
    n_heads = adapter.n_attention_heads(model)
    head_dim = adapter.hidden_size(model) // n_heads
    mass = {b: {"delta_resid": 0.0, "delta_dla": 0.0} for b in blocks}
    for (b_img, _), zb, zp, (fb, fp) in zip(batches, base_z, patched_z, finals):
        gold_b = b_img["base_gold_toks"][:, 0].to(model.device)
        gold_s = b_img["source_gold_toks"][:, 0].to(model.device)
        d = adapter.logit_direction(model, gold_b) - adapter.logit_direction(model, gold_s)
        sb = adapter.final_norm_scale(model, fb.float())
        sp = adapter.final_norm_scale(model, fp.float())
        for b in blocks:
            for _, _, dr, dd in per_head_delta(adapter, model, zb[b], zp[b], b, n_heads, head_dim, d, sb, sp):
                mass[b]["delta_resid"] += dr / len(batches)
                mass[b]["delta_dla"] += abs(dd) / len(batches)
    return mass


def run_arm(adapter, model, batches, img_src, z_per_batch, heads, with_image_patch, patch_layer,
            pad_token_id, max_new_tokens, shuffled=False):
    """One generation per batch with `heads` overwritten at the last token from
    z_per_batch, optionally under the image patch. Mirrors head_trace's
    run_cause / run_cause_shuffled (1-row batches are skipped when shuffled,
    since they roll onto themselves)."""
    hidden = adapter.hidden_size(model)
    head_dim = hidden // adapter.n_attention_heads(model)
    t = Tally()
    for (b_img, b_last), z, src in zip(batches, z_per_batch, img_src):
        if shuffled and len(b_img["rows"]) < 2:
            continue
        vals = {blk: torch.roll(v, shifts=1, dims=0) for blk, v in z.items()} if shuffled else z
        patches = [image_patch(adapter, model, b_img, patch_layer, src=src)] if with_image_patch else []
        patches += head_patches(heads, vals, b_last["positions"], hidden, head_dim, model.device)
        gen = generate_with_patches(adapter, model, patches, b_img["base_input_ids"], b_img["attention_mask"],
                                    b_img["base_extra"], pad_token_id, max_new_tokens)
        donor = None
        if shuffled:
            donor = {"source_gold_toks": torch.roll(b_img["source_gold_toks"], 1, 0),
                     "source_gold_len": torch.roll(b_img["source_gold_len"], 1, 0)}
        t.add(gen, b_img, donor)
    return t.rates(shuffled=shuffled)


def sweep_windows(adapter, model, batches, patch_layer, windows, pad_token_id, max_new_tokens,
                  skip_necessity=False, skip_shuffled=False, log=None):
    """The whole stage on prebuilt (image batch, last_token batch) pairs.
    Returns the report dict (minus run metadata)."""
    log = log or (lambda msg: print(msg, flush=True))
    n_heads = adapter.n_attention_heads(model)
    n_layers = len(adapter.get_decoder_layers(model))
    for w in windows:
        assert all(patch_layer <= b < n_layers for b in w), (
            f"window {window_label(w)} has a block outside [{patch_layer}, {n_layers - 1}]: a block below "
            f"--patch_layer runs BEFORE the image patch, so installing its captured values is a no-op")
    blocks = sorted({b for w in windows for b in w})
    log(f"capturing per-head outputs at blocks {window_label(blocks)} (clean + image-patched) ...")
    img_src, base_z, patched_z, finals = capture(adapter, model, batches, patch_layer, blocks)
    mass = per_block_mass(adapter, model, batches, blocks, base_z, patched_z, finals)

    t = Tally()
    for b_img, _ in batches:
        t.add(generate_unhooked(model, b_img["base_input_ids"], b_img["attention_mask"], b_img["base_extra"],
                                pad_token_id, max_new_tokens), b_img)
    unhooked = t.rates()
    image_only = run_arm(adapter, model, batches, img_src, base_z, [], True, patch_layer, pad_token_id,
                         max_new_tokens)
    log(f"  unhooked:          cause={unhooked['cause']:6.1%} first_src={unhooked['first_src']:6.1%}")
    log(f"  image patch only:  cause={image_only['cause']:6.1%} first_src={image_only['first_src']:6.1%} "
        f"base_kept={image_only['base_kept']:6.1%}   <-- the effect every window is read against")
    if image_only["first_src"] < 0.10 and image_only["cause"] < 0.10:
        log(f"  !! the image patch at layer {patch_layer} barely moves the answer -- pick a --patch_layer "
            f"below the handoff (stage 1's image curve) before reading anything below")

    rows = []
    for w in windows:
        heads = [(b, h) for b in w for h in range(n_heads)]
        rec = {"blocks": w, "label": window_label(w), "n_heads": len(heads),
               "suff": run_arm(adapter, model, batches, img_src, patched_z, heads, False, patch_layer,
                               pad_token_id, max_new_tokens)}
        if not skip_necessity:
            rec["nec"] = run_arm(adapter, model, batches, img_src, base_z, heads, True, patch_layer,
                                 pad_token_id, max_new_tokens)
        if not skip_shuffled:
            rec["shuffled"] = run_arm(adapter, model, batches, img_src, patched_z, heads, False, patch_layer,
                                      pad_token_id, max_new_tokens, shuffled=True)
        rows.append(rec)
        s = rec["suff"]
        line = (f"  window {rec['label']:>7} ({len(heads):>3} heads)  suff: cause={s['cause']:6.1%} "
                f"first_src={s['first_src']:6.1%} base_kept={s['base_kept']:6.1%}")
        if "nec" in rec:
            line += f" | nec: cause={rec['nec']['cause']:6.1%} first_src={rec['nec']['first_src']:6.1%}"
        if "shuffled" in rec:
            line += (f" | shuffled: own={rec['shuffled']['first_own']:6.1%} "
                     f"donor={rec['shuffled']['first_donor']:6.1%} (first tok)")
        log(line)
    return {"blocks_captured": blocks, "per_block_mass": {str(b): m for b, m in mass.items()},
            "unhooked": unhooked, "image_patch_only": image_only, "windows": rows}


# ---------------------------------------------------------------- CLI

def build_paired_batches(args, rows, entity_assets, adapter, model, processor):
    """(image-position batch, last_token batch) per chunk, with head_trace's
    own guards: same sequences, different -- and non-overlapping -- positions."""
    cache = BuildBatchCache()
    out = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        b_img = build_batch_at(args.positions, chunk, entity_assets, adapter, model, processor, batch_cache=cache)
        b_last = build_batch_at("last_token", chunk, entity_assets, adapter, model, processor, batch_cache=cache)
        assert torch.equal(b_img["base_input_ids"], b_last["base_input_ids"]), \
            "image-position and last_token batches disagree on base_input_ids"
        assert torch.equal(b_img["attention_mask"], b_last["attention_mask"]), \
            "image-position and last_token batches disagree on attention_mask"
        assert b_last["positions"].shape[1] == 1, \
            f"the last_token batch resolved to {b_last['positions'].shape[1]} columns, not 1"
        overlap = set(b_last["positions"][0].tolist()) & set(b_img["positions"][0].tolist())
        assert not overlap, f"last_token column(s) {sorted(overlap)} lie inside the {args.positions!r} span"
        out.append((b_img, b_last))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--patch_layer", type=int, required=True,
                    help="Residual layer to patch the IMAGE at -- below the handoff (stage 1).")
    ap.add_argument("--positions", default="full_image", help="Image position set to patch.")
    ap.add_argument("--windows", nargs="+", default=None,
                    help="Explicit windows: '21-23', '22', '21.23'. Overrides --range/--max_len.")
    ap.add_argument("--range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Block range to enumerate windows in (default: --patch_layer .. --patch_layer+4).")
    ap.add_argument("--max_len", type=int, default=3, help="Longest enumerated window (the full range is "
                                                           "always added as well).")
    ap.add_argument("--extra_windows", nargs="+", default=[],
                    help="Windows added on top of --range/--max_len's enumeration (same syntax as --windows).")
    ap.add_argument("--no_full_reference", action="store_true",
                    help="Drop the --patch_layer..last-block identity window (not recommended: it is the "
                         "check that the patch plumbing is right).")
    ap.add_argument("--skip_necessity", action="store_true")
    ap.add_argument("--skip_shuffled", action="store_true")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n_rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0,
                    help="Row-sampling seed; same default and sampling as head_trace, so seed 0 gives stage 3 "
                         "the same rows when n_rows matches.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_ANSWER_TOKENS + 2)
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_dir = ndm_logs_dir(model_slug, args.entity, args.attribute, 0.0, 0.0, 0.0, 0.0,
                           args.positions, "attn_head_output", pruned, diagnostic=True)
    os.makedirs(log_dir, exist_ok=True)
    tag = f"head_window_sweep_patch{args.patch_layer}"
    out_path = args.out or os.path.join(log_dir, f"{tag}.json")

    with tee_to_log(os.path.join(log_dir, f"{tag}.log")):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()
        n_layers = len(adapter.get_decoder_layers(model))
        if args.windows:
            windows = [parse_window(w) for w in args.windows]
        else:
            lo, hi = args.range or (args.patch_layer, args.patch_layer + 4)
            windows = enumerate_windows(max(lo, args.patch_layer), min(hi, n_layers - 1), args.max_len)
        for w in map(parse_window, args.extra_windows):
            if w not in windows:
                windows.append(w)
        full_ref = list(range(args.patch_layer, n_layers))
        if not args.no_full_reference and full_ref not in windows:
            windows.append(full_ref)

        entity_assets = load_entity_assets(args.vade_root, args.entity)
        rows = load_tuples(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir)
        for r in rows:
            r.setdefault("target_attribute", args.attribute)
        cause_all = [r for r in rows if r["rule"] == "match_source"]
        # Identical sampling to head_trace.py, so stage 2 and stage 3 see the same rows.
        cause_rows = random.Random(args.seed).sample(cause_all, min(args.n_rows, len(cause_all)))
        print(f"[head_window_sweep] entity={args.entity} attribute={args.attribute} "
              f"patch_layer={args.patch_layer} positions={args.positions} rows={len(cause_rows)} "
              f"({len({r['base'] for r in cause_rows})} distinct bases) windows={len(windows)}")
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        batches = build_paired_batches(args, cause_rows, entity_assets, adapter, model, processor)

        report = sweep_windows(adapter, model, batches, args.patch_layer, windows, pad_token_id,
                               args.max_new_tokens, args.skip_necessity, args.skip_shuffled)

        io = report["image_patch_only"]
        ref = next((w for w in report["windows"] if w["blocks"] == full_ref), None)
        if ref is not None:
            # First token only: that is where the identity is exact. Later decode steps also attend
            # to the image tokens' K/V, which the image patch changes and the head install does not,
            # so full-answer `cause` may legitimately differ on multi-token attributes.
            ok = abs(ref["suff"]["first_src"] - io["first_src"]) <= 0.05
            report["full_reference_identity_ok"] = ok
            print(f"\nfull-coverage identity (window {ref['label']} suff vs image patch only): "
                  f"cause {ref['suff']['cause']:.1%} vs {io['cause']:.1%}, first_src "
                  f"{ref['suff']['first_src']:.1%} vs {io['first_src']:.1%} -> "
                  + ("OK" if ok else "!! MISMATCH -- the head patch is not reproducing the image patch; "
                                     "no window row is trustworthy until this is fixed"))

        print("\nper-block |delta_resid| mass at the last token (where the image-patched writes changed):")
        for b, m in report["per_block_mass"].items():
            print(f"  block {b:>2}: delta_resid={m['delta_resid']:9.4f}  |delta_dla|={m['delta_dla']:9.4f}")

        report.update({"entity": args.entity, "attribute": args.attribute, "patch_layer": args.patch_layer,
                       "positions": args.positions, "n_rows": len(cause_rows), "seed": args.seed,
                       "split": args.split, "n_layers": n_layers,
                       "n_heads": adapter.n_attention_heads(model)})
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
