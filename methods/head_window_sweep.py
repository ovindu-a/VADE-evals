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
from methods.common.text_match import contains_label, decode_answers, labels_of  # noqa: E402
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

def divergence_hits(pred, ref_base, ref_src):
    """First-DIFFERING-token transfer, against the model's OWN answers.

    ref_base / ref_src are the model's unhooked answers on the base and the source
    image ([B, T] ids). j = the first position where they differ. A row counts for
    the source iff the prediction reproduces the shared prefix AND the source's
    token at j (likewise for the base). Rows whose two answers never differ are
    excluded. This is the honest single-token readout for answers that share a
    leading ' the', a century ('19..'), or a first digit -- where comparing the
    first token scores nothing -- and it sidesteps stored-label casing entirely.
    -> (valid [B] bool, hit_src [B] bool, hit_base [B] bool)"""
    L = min(ref_base.shape[1], ref_src.shape[1])
    diff = ref_base[:, :L] != ref_src[:, :L]
    valid = diff.any(1)
    j = diff.int().argmax(1)
    B = pred.shape[0]
    hit_src = torch.zeros(B, dtype=torch.bool)
    hit_base = torch.zeros(B, dtype=torch.bool)
    for i in range(B):
        ji = int(j[i])
        if not valid[i] or pred.shape[1] <= ji or not torch.equal(pred[i, :ji], ref_src[i, :ji]):
            continue
        hit_src[i] = bool(pred[i, ji] == ref_src[i, ji])
        hit_base[i] = bool(pred[i, ji] == ref_base[i, ji])
    return valid, hit_src, hit_base


class Tally:
    """Row-weighted accumulator for one arm.

    Two scoring modes. TOKEN (tokenizer=None, what flags used): cause/base_kept by
    token exact match against the stored gold labels, first_* on the gold labels'
    first tokens. TEXT (a tokenizer): cause/base_kept by VADE's text matcher; first_*
    at position 0 against the model's OWN unhooked base/source answers (the batch
    carries them as `ref_base_ans` / `ref_src_ans`), over rows where those differ at
    position 0; plus div_* via divergence_hits, at the first position they differ.

    Why both. first_* (position 0) is what a last-prompt-column patch controls, so it is
    what windows are ranked by. div_* reaches rows whose answers share a prefix
    (' the United States' / ' the Netherlands', years), but past position 0 every
    decode step also reads the image tokens' K/V, which a head install at the last
    prompt column leaves clean -- so for head arms div_* can sit far below the image
    patch's even for the right window. It is reported, not selected on."""

    def __init__(self, tokenizer=None):
        self.tok = tokenizer
        self.n = self.src = self.base = 0
        self.n_first = self.first_src = self.first_base = 0
        self.donor = self.first_donor = self.n_first_donor = 0
        self.n_div = self.div_src = self.div_base = 0

    def _full(self, pred, batch, role):
        if self.tok is not None:
            texts = decode_answers(self.tok, pred)
            return sum(contains_label(t, l) for t, l in zip(texts, labels_of(batch, role)))
        return int(exact_match(pred[:, :MAX_ANSWER_TOKENS], batch[f"{role}_gold_toks"],
                               batch[f"{role}_gold_len"]).sum())

    def _first(self, pred, batch):
        """-> (valid, hit_src, hit_base) at position 0."""
        if "ref_src_ans" in batch:
            s0, b0 = batch["ref_src_ans"][:, 0], batch["ref_base_ans"][:, 0]
        else:
            s0, b0 = batch["source_gold_toks"][:, 0], batch["base_gold_toks"][:, 0]
        valid = s0 != b0
        if not pred.shape[1]:
            z = torch.zeros_like(valid)
            return valid, z, z
        return valid, (pred[:, 0] == s0) & valid, (pred[:, 0] == b0) & valid

    def add(self, gen, batch, donor_batch=None):
        """donor_batch: for the shuffled control, the ROLLED golds -- `source_labels`,
        `source_gold_toks`/`source_gold_len`, and `ref_src_ans` in text mode."""
        self.n += gen.shape[0]
        self.src += self._full(gen, batch, "source")
        self.base += self._full(gen, batch, "base")
        valid, hs, hb = self._first(gen, batch)
        self.n_first += int(valid.sum())
        self.first_src += int(hs.sum())
        self.first_base += int(hb.sum())
        if "ref_src_ans" in batch:
            dv, ds, db = divergence_hits(gen, batch["ref_base_ans"], batch["ref_src_ans"])
            self.n_div += int(dv.sum())
            self.div_src += int(ds.sum())
            self.div_base += int(db.sum())
        if donor_batch is not None:
            merged = {**batch, **donor_batch}
            self.donor += self._full(gen, merged, "source")
            dv, dh, _ = self._first(gen, merged)
            self.n_first_donor += int(dv.sum())
            self.first_donor += int(dh.sum())

    def rates(self, shuffled=False):
        n, nf = max(self.n, 1), max(self.n_first, 1)
        r = {"n": self.n, "n_first": self.n_first,
             "cause": self.src / n, "base_kept": self.base / n,
             "other": max(0.0, 1.0 - (self.src + self.base) / n),
             "first_src": self.first_src / nf, "first_base": self.first_base / nf}
        if self.n_div:
            r.update({"n_div": self.n_div, "div_src": self.div_src / self.n_div,
                      "div_base": self.div_base / self.n_div})
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
            pad_token_id, max_new_tokens, shuffled=False, tokenizer=None):
    """One generation per batch with `heads` overwritten at the last token from
    z_per_batch, optionally under the image patch. Mirrors head_trace's
    run_cause / run_cause_shuffled (1-row batches are skipped when shuffled,
    since they roll onto themselves)."""
    hidden = adapter.hidden_size(model)
    head_dim = hidden // adapter.n_attention_heads(model)
    t = Tally(tokenizer)
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
            if tokenizer is not None:
                labels = labels_of(b_img, "source")
                donor["source_labels"] = labels[-1:] + labels[:-1]   # torch.roll(shifts=1): row i <- i-1
                donor["ref_src_ans"] = torch.roll(b_img["ref_src_ans"], 1, 0)
        t.add(gen, b_img, donor)
    return t.rates(shuffled=shuffled)


def sweep_windows(adapter, model, batches, patch_layer, windows, pad_token_id, max_new_tokens,
                  skip_necessity=False, skip_shuffled=False, log=None, tokenizer=None):
    """The whole stage on prebuilt (image batch, last_token batch) pairs.
    Returns the report dict (minus run metadata). tokenizer switches scoring to
    TEXT mode (see Tally); the model's own base/source answers are then generated
    once per batch and attached to it as the first-token references."""
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

    t = Tally(tokenizer)
    for b_img, _ in batches:
        gen = generate_unhooked(model, b_img["base_input_ids"], b_img["attention_mask"], b_img["base_extra"],
                                pad_token_id, max_new_tokens)
        if tokenizer is not None:
            b_img["ref_base_ans"] = gen
            b_img["ref_src_ans"] = generate_unhooked(model, b_img["source_input_ids"], b_img["attention_mask"],
                                                     b_img["source_extra"], pad_token_id, max_new_tokens)
        t.add(gen, b_img)
    unhooked = t.rates()
    if tokenizer is not None:
        log(f"  scoring: VADE text match. first_* = position 0 vs the model's own base/source answers, "
            f"over the {unhooked['n_first']}/{unhooked['n']} rows where those differ there; div_* = at the "
            f"first differing token ({unhooked.get('n_div', 0)}/{unhooked['n']} rows)")
        if unhooked["n_first"] < max(4, unhooked["n"] // 8):
            log(f"  !! only {unhooked['n_first']} rows differ at position 0: the model's base and source "
                f"answers share their first token almost everywhere (years, ' the ...'), so no "
                f"last-prompt-column patch can be measured here -- windows cannot be ranked for this "
                f"attribute; use continuous substitution (image_head_pipeline --stages ... decode)")
    image_only = run_arm(adapter, model, batches, img_src, base_z, [], True, patch_layer, pad_token_id,
                         max_new_tokens, tokenizer=tokenizer)
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
                               pad_token_id, max_new_tokens, tokenizer=tokenizer)}
        if not skip_necessity:
            rec["nec"] = run_arm(adapter, model, batches, img_src, base_z, heads, True, patch_layer,
                                 pad_token_id, max_new_tokens, tokenizer=tokenizer)
        if not skip_shuffled:
            rec["shuffled"] = run_arm(adapter, model, batches, img_src, patched_z, heads, False, patch_layer,
                                      pad_token_id, max_new_tokens, shuffled=True, tokenizer=tokenizer)
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
    ap.add_argument("--text_match", action="store_true",
                    help="TEXT scoring (see Tally): VADE's matcher for cause/base_kept, first-differing-token "
                         "against the model's own answers for first_*. Required for lowercase-label entities; "
                         "use --max_new_tokens 8 with it.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_dir = ndm_logs_dir(model_slug, args.entity, args.attribute, 0.0, 0.0, 0.0, 0.0,
                           args.positions, "attn_head_output", pruned, diagnostic=True)
    os.makedirs(log_dir, exist_ok=True)
    tag = f"head_window_sweep_patch{args.patch_layer}" + ("_text" if args.text_match else "")
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
                               args.max_new_tokens, args.skip_necessity, args.skip_shuffled,
                               tokenizer=processor.tokenizer if args.text_match else None)

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
                       "scoring": "text" if args.text_match else "token", "max_new_tokens": args.max_new_tokens,
                       "positions": args.positions, "n_rows": len(cause_rows), "seed": args.seed,
                       "split": args.split, "n_layers": n_layers,
                       "n_heads": adapter.n_attention_heads(model)})
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
