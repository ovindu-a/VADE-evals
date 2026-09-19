"""Entity or attribute? Cross the donor's FLAG and the donor's QUESTION.

THE QUESTION THIS SETTLES

R10 showed that swapping ~10 attention heads in blocks 21-23 transplants the
source country wholesale: cause 95.5%, iso leaking to the source 98.6%. But every
row there had donor question == base question, so two different hypotheses were
indistinguishable:

  ENTITY     the heads carry "this flag is Argentina", and the BASE PROMPT
             decides which attribute gets reported
  ATTRIBUTE  the heads carry "the answer to the question that was asked", i.e.
             already-extracted attribute content

Under a matched donor those predict the same string, so R10 cannot separate them.
Crossing the donor's question against the base's does.

THE 2x2

    base = flag A asking Q1        donor = flag B asking Q2

  self       A=B, Q1=Q2   a run patched with its OWN values. A provable no-op:
                          anything but ~100% base means the plumbing is wrong.
  flag       A!=B, Q1=Q2  R10's arm, reproduced here.
  question   A=B, Q1!=Q2  THE CLEAN ONE. The entity never changes, so any shift
                          in the answer has to come from question-conditioned
                          content in the heads. If the output stays A's answer to
                          Q1, the heads carry nothing question-specific.
  both       A!=B, Q1!=Q2 the full cross, classified four ways.

WHAT EACH OUTCOME MEANS (in the `both` cell, where all four golds differ)

  B's answer to Q1   the heads carried the ENTITY; the base prompt chose the
                     attribute. Entity-specific.
  B's answer to Q2   the heads carried the donor's EXTRACTED ANSWER and overrode
                     the base's question. Attribute-specific.
  A's answer to Q1   the patch did not take.
  A's answer to Q2   the heads carried WHICH QUESTION was asked but not the
                     entity. Live possibility: question identity decodes at ~100%
                     from any head in these blocks, including random ones.

READ THE CONTRAST, NOT JUST THE 4-WAY SPLIT. The base prefill constrains the
answer TYPE -- after "The capital city is" the model emits a city whatever is
patched -- so "B's answer to Q2" is suppressed by the prompt, not only by the
mechanism. Seeing it would be strong evidence for attribute content; NOT seeing
it proves little. The confound-free quantity holds the base prompt fixed and
varies only the donor's question:

    P(output = B's answer to Q1 | donor asked Q1)      <- `flag` cell
    P(output = B's answer to Q1 | donor asked Q2)      <- `both` cell

Same base prompt, same answer type, same gold; only the donor's question moves.
Equal rates => the heads' content is question-INDEPENDENT (entity). A drop =>
question-conditioned (attribute). That contrast is printed as CONTENT SHIFT.

Prior from the captures (ATTRIBUTE_HEAD_EXPERIMENTS.md R9/R11): language decodes
from these heads at 67.8% when the CAPITAL was asked vs 65.3% when the LANGUAGE
was asked, +0.4pp. If that correlational result is causal, the contrast is flat.

Matching is imported from VADE's own scorer, so a hit here means what it means
in eval/score.py -- whole-word, case- and diacritic-insensitive.

Usage
-----
    python methods/head_cross.py --dry_run
    python methods/head_cross.py --n_pairs 60 --batch_size 16
    python methods/head_cross.py --n_pairs 60 --subspace_dim 7      # the DAS-shaped patch
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.head_swap_vade import (ATTRIBUTES, DEFAULT_VADE_ROOT, MODEL_ID, build_prompt,   # noqa: E402
                                    build_value_subspace, capture_donor, decode, load_assets,
                                    parse_heads, patched_generate, plain_generate, verify_readback)

COMMON10 = "21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
CELLS = ("self", "flag", "question", "both")


def vade_matcher(vade_root):
    """VADE's own `contains_label`, imported rather than reimplemented so a hit
    here means exactly what it means in eval/score.py."""
    eval_dir = os.path.join(vade_root, "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("vade_score", os.path.join(eval_dir, "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.contains_label


def build_rows(items, attributes, n_pairs, seed):
    """Every (Q1, Q2) for each sampled flag pair, twice: once with the donor on
    the SOURCE flag (self/flag/both) and once on the BASE flag (self/question).

    The second is not a control bolted on -- it is the arm with no entity change
    at all, and therefore the only one whose result cannot be explained by
    entity transfer."""
    rng = random.Random(seed)
    names = sorted(items)
    pairs = [(a, b) for a in names for b in names if a != b]
    rng.shuffle(pairs)
    rows = []
    for base, source in pairs[:n_pairs]:
        for q1 in attributes:
            for q2 in attributes:
                for donor_flag in (source, base):
                    same_flag = donor_flag == base
                    cell = ("self" if same_flag and q1 == q2 else
                            "question" if same_flag else
                            "flag" if q1 == q2 else "both")
                    rows.append({"base": base, "source": source, "donor_flag": donor_flag,
                                 "q1": q1, "q2": q2, "cell": cell})
    return rows


def classify(text, row, truth, matcher):
    """-> which of the four golds the generation matched (or 'other'/'ambiguous').

    Order matters only for reporting; a generation matching two distinct golds is
    reported as ambiguous rather than silently credited to the first."""
    a, b, q1, q2 = row["base"], row["donor_flag"], row["q1"], row["q2"]
    golds = {"base_q1": (a, q1), "source_q1": (b, q1), "source_q2": (b, q2), "base_q2": (a, q2)}
    hits = []
    for name, (flag, attr) in golds.items():
        if matcher(text, str(truth[flag][attr])):
            hits.append(name)
    if not hits:
        return "other"
    # Collapse labels that are the SAME string in this cell (self/flag/question
    # cells make some of the four golds coincide) before calling it ambiguous.
    values = {str(truth[golds[h][0]][golds[h][1]]) for h in hits}
    return hits[0] if len(values) == 1 else "ambiguous"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--heads", default=COMMON10)
    ap.add_argument("--n_pairs", type=int, default=60, help="Distinct (base, source) flag pairs.")
    ap.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS),
                    help="Which of the 2x2 to generate. NOTE: dropping `flag` removes the CONTENT "
                         "SHIFT contrast (it is flag-vs-both), which is the only confound-free "
                         "number here; dropping `self` removes the no-op plumbing check.")
    ap.add_argument("--subspace_dim", type=int, default=0,
                    help="Patch only this many value-centroid dims per block (see head_swap_vade).")
    ap.add_argument("--subspace_attribute", default="language")
    ap.add_argument("--capture_dir", default=os.path.join(REPO_ROOT, "results", "attr_capture",
                                                          "flags", "blocks15-27_n84"))
    ap.add_argument("--template_index", type=int, default=0,
                    help="Which of each attribute's templates to use, by sorted template_id. One "
                         "fixed phrasing per question keeps the cross about the ATTRIBUTE asked.")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    blocks = sorted({b for b, _ in heads})
    entity_dir, items, lookup = load_assets(args.vade_root, args.entity)
    gt = json.load(open(os.path.join(args.vade_root, "data", args.entity, "ground_truth.json")))
    truth = gt["countries"] if "countries" in gt else gt["items"]
    rows = [r for r in build_rows(items, list(ATTRIBUTES), args.n_pairs, args.seed)
            if r["cell"] in args.cells]
    out_path = args.out or os.path.join(REPO_ROOT, "results", "head_cross", args.entity,
                                        f"cross_n{args.n_pairs}"
                                        + (f"_sub{args.subspace_dim}" if args.subspace_dim else "")
                                        + ".jsonl")

    from collections import Counter
    print(f"[head_cross] {args.entity}: {args.n_pairs} flag pairs x {len(ATTRIBUTES)}x{len(ATTRIBUTES)} "
          f"question pairs x 2 donors = {len(rows)} generations")
    print(f"  cells: {dict(Counter(r['cell'] for r in rows))}")
    print(f"  heads ({len(heads)}): " + ", ".join(f"{b}.{h}" for b, h in heads))
    print(f"  -> {out_path}")
    if args.dry_run:
        for r in rows[:4]:
            print(f"    [{r['cell']:8}] base={r['base']}/{r['q1']:12} donor={r['donor_flag']}/{r['q2']}")
        print("Grid valid.")
        return

    from methods.adapters.registry import get_adapter
    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    hidden, n_heads = adapter.hidden_size(model), adapter.n_attention_heads(model)
    head_dim = hidden // n_heads
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    matcher = vade_matcher(args.vade_root)

    subspace = None
    if args.subspace_dim:
        subspace, n_val, n_it = build_value_subspace(args.capture_dir, heads, args.subspace_attribute,
                                                     args.subspace_dim, args.vade_root, args.entity)
        print(f"  subspace: {args.subspace_attribute}, {n_val} values over {n_it} items, "
              f"dims/block {{{', '.join(f'{b}: {P.shape[1]}' for b, P in subspace.items())}}}")

    def batch_of(chunk):
        from PIL import Image
        import torch
        seqs = {"base": [], "donor": []}
        px = {"base": [], "donor": []}
        thw = {"base": [], "donor": []}
        for r in chunk:
            for side, flag, q in (("base", r["base"], r["q1"]), ("donor", r["donor_flag"], r["q2"])):
                # One fixed template per question: base and donor must differ in
                # WHICH attribute is asked, not in phrasing, or the cross confounds
                # question identity with wording.
                t = lookup[q][sorted(lookup[q])[args.template_index]]
                with Image.open(os.path.join(entity_dir, items[flag]["image"])) as im:
                    p = build_prompt(processor, im.convert("RGB"), t["question"], t["prefill"])
                seqs[side].append(p["input_ids"][0]); px[side].append(p["pixel_values"])
                thw[side].append(p["image_grid_thw"])

        def pack(ss):
            n = max(len(s) for s in ss)
            ids = torch.full((len(ss), n), pad_id, dtype=torch.long)
            msk = torch.zeros((len(ss), n), dtype=torch.long)
            for i, s in enumerate(ss):
                ids[i, n - len(s):] = s; msk[i, n - len(s):] = 1
            return ids, msk

        bi, bm = pack(seqs["base"]); di, dm = pack(seqs["donor"])
        return (bi, bm, {"pixel_values": torch.cat(px["base"]), "image_grid_thw": torch.cat(thw["base"])},
                di, dm, {"pixel_values": torch.cat(px["donor"]), "image_grid_thw": torch.cat(thw["donor"])})

    probe = batch_of(rows[:min(4, len(rows))])
    pz = capture_donor(adapter, model, blocks, probe[3], probe[4], probe[5], args.max_new_tokens, pad_id)
    worst = verify_readback(adapter, model, heads, pz, probe[0], probe[1], probe[2],
                            args.max_new_tokens, pad_id, head_dim, subspace=subspace)
    print(f"  read-back check: worst = {worst:.2e} over {args.max_new_tokens} steps")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tally = {c: Counter() for c in CELLS}
    with open(out_path, "w") as fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            bi, bm, bx, di, dm, dx = batch_of(chunk)
            z = capture_donor(adapter, model, blocks, di, dm, dx, args.max_new_tokens, pad_id)
            toks = patched_generate(adapter, model, heads, z, bi, bm, bx, args.max_new_tokens,
                                    pad_id, head_dim, subspace=subspace)
            clean = plain_generate(model, bi, bm, bx, args.max_new_tokens, pad_id)
            for r, text, ctext in zip(chunk, decode(processor, toks), decode(processor, clean)):
                label = classify(text, r, truth, matcher)
                tally[r["cell"]][label] += 1
                fh.write(json.dumps({**r, "generated": text, "clean": ctext, "label": label}) + "\n")
            print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    labels = ["base_q1", "source_q1", "source_q2", "base_q2", "other", "ambiguous"]
    print(f"\n{'cell':>10} {'n':>6} " + "".join(f"{l:>11}" for l in labels))
    for c in [c for c in CELLS if tally[c]]:
        n = sum(tally[c].values()) or 1
        print(f"{c:>10} {sum(tally[c].values()):>6} "
              + "".join(f"{tally[c][l] / n * 100:10.1f}%" for l in labels))

    if tally["self"]:
        print("\n  self must be ~100% base_q1 -- it patches a run with its own values.")
    if tally["flag"] and tally["both"]:
        matched = tally["flag"]["source_q1"] / sum(tally["flag"].values()) * 100
        crossed = tally["both"]["source_q1"] / sum(tally["both"].values()) * 100
        print(f"\n  CONTENT SHIFT (the confound-free number):")
        print(f"    donor asked the SAME question  -> B's answer to Q1 in {matched:.1f}% of rows")
        print(f"    donor asked a DIFFERENT one    -> B's answer to Q1 in {crossed:.1f}% of rows")
        print(f"    shift = {crossed - matched:+.1f}pp")
        print("    ~0 means the heads' content does not depend on which question produced it:\n"
              "    ENTITY data. A large drop means question-conditioned content: ATTRIBUTE data.")
    else:
        print("\n  CONTENT SHIFT not computed -- it needs both `flag` and `both`, and --cells "
              "excluded one. The 4-way split below is still subject to the prefill confound.")
    if tally["question"]:
        qn = sum(tally["question"].values())
        print(f"\n  QUESTION-ONLY cell (entity never changes, so nothing here is entity transfer):\n"
              f"    stayed at A's answer to Q1: {tally['question']['base_q1'] / qn * 100:.1f}%   "
              f"moved to A's answer to Q2: {tally['question']['base_q2'] / qn * 100:.1f}%")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
