"""PER-ROW trace of a full source swap: not "what fraction flipped", but
"what did the model actually say instead", row by row, token by token.

Same intervention as ceiling_sweep.py -- sigma(m/T)=1 on every dimension, the
whole source activation replacing the whole base activation -- and the same
sites and position sets. What differs is the reporting, and the reporting is
the point.

WHY THIS EXISTS. ceiling_sweep reports cause/base_kept/iso as rates over 32
rows. That is the right shape for choosing a layer to train at, and the wrong
shape for understanding a site that behaves strangely. flags/calling_code at
last_token, residual, layer 24 reads cause=3.4% base_kept=14.3% -- which reads
as a dead site, and the sweep's own verdict line said exactly that. It is not
dead. The missing 82.3% is `other`: the model emits a THIRD country's dialing
code, belonging to neither the base nor the source.

    row 383: base=GY (gold='592') -> source=ET (gold='251')
      unhooked: '592. This corresponds to the'
      L22:      '592. This corresponds to the'   [base]
      L23:      '246. This corresponds to the'   [other]

So the swap is violently causal -- it destroys the answer -- it simply does not
TRANSFER the source's. That is the opposite of an inert site, and no aggregate
of two rates can distinguish the two cases. Only the generated text can.

Compare flags/language at the same site/position/layer: cause=100%,
base_kept=0%, other=0%. Same instrument, same intervention, opposite outcome.
Whatever explains that split is not visible in a rate table.

THREE-WAY CLASSIFICATION per (row, layer), using the trainer's own token-level
gold matching (common/targets.py's exact_match, up to MAX_ANSWER_TOKENS):

    [HIT]    generation == the SOURCE's gold answer  -> the swap transferred
    [base]   generation == the BASE's gold answer    -> the swap did nothing
    [other]  neither                                 -> the swap DESTROYED

`other` is the column ceiling_sweep could only imply. Reading it: high `other`
with low `HIT` means the site carries something the answer is COMPUTED from
rather than the answer itself, so corrupting it yields a plausible wrong
answer instead of the source's right one.

TWO LOGS ARE WRITTEN. The main one shows decoded text. The _TOKENS one shows
every answer as its actual token sequence, because Qwen2.5-VL splits digits
one per token ('250' -> ['2','5','0']) and a digit-level answer can be half
right in a way decoded text hides -- worth having whenever the attribute's
answers are numeric.

Deliberately defaults to n_rows=384, not ceiling_sweep's 32: distinguishing 3%
from 0%, or reading an `other` fraction at all, needs more than a 32-row
sample where one row is 3.1%.

Usage:
    python methods/ndm/swap_trace.py --entity flags --attribute calling_code \
        --positions last_token --site residual --layers 19 20 21 22 23 24 25 26 27 28
"""
import argparse
import datetime
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import (  # noqa: E402
    BuildBatchCache, load_entity_assets, load_tuples, require_pruned_tuples,
)
from methods.common.position_sets import build_batch_at, path_safe  # noqa: E402
from methods.common.sites import resolve_site, site_name  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS, exact_match  # noqa: E402
from methods.ndm.ceiling_sweep import full_swap_generation  # noqa: E402
from methods.ndm.config import METHOD_NAME  # noqa: E402
from methods.ndm.verify_sites import generate_unhooked  # noqa: E402

HIT, BASE, OTHER = "HIT", "base", "other"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "logs", "adhoc_probes")


def classify_batch(gen_toks, batch):
    """-> [HIT|base|other] per row. HIT takes precedence over base, which can
    only collide if base_label == source_label -- pruned tuples exclude that,
    and the collision count is reported rather than silently resolved."""
    pred = gen_toks[:, :MAX_ANSWER_TOKENS]
    ms = exact_match(pred, batch["source_gold_toks"], batch["source_gold_len"])
    mb = exact_match(pred, batch["base_gold_toks"], batch["base_gold_len"])
    both = int((ms & mb).sum())
    return [HIT if ms[i] else (BASE if mb[i] else OTHER) for i in range(len(ms))], both


def token_view(tokenizer, ids):
    """'250' = ['2', '5', '0'] -- the decoded string plus its per-token split."""
    toks = [tokenizer.decode([int(t)]) for t in ids]
    return f"{tokenizer.decode([int(t) for t in ids], skip_special_tokens=False)!r} = {toks}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--site", default="residual", type=site_name,
                    help="Any single site, or a joint/blocks:N one -- the intervention is "
                         "ceiling_sweep's, so everything it accepts works here.")
    ap.add_argument("--positions", default="last_token",
                    help="Any spec common/position_sets.py accepts.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n_rows", type=int, default=384,
                    help="Cause rows to trace (seeded random sample). Default 384, NOT ceiling_sweep's "
                         "32 -- at 32 rows one row is 3.1%%, which is the whole dynamic range of the "
                         "effects this script exists to look at.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--no_tokens_log", action="store_true",
                    help="Skip the _TOKENS log. Keep it for numeric attributes -- Qwen splits digits "
                         "one per token, so a half-right answer is invisible in decoded text.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    site = resolve_site(args.site)
    bad = [l for l in args.layers if l < site.min_layer()]
    assert not bad, f"site {args.site!r} needs --layer >= {site.min_layer()}; got {bad}"
    layers_tag = f"{min(args.layers)}-{max(args.layers)}"
    stem = (f"{args.entity}_{args.attribute}_{path_safe(args.positions)}_{path_safe(args.site)}"
            f"_n{args.n_rows}_layers{layers_tag}")
    os.makedirs(args.out_dir, exist_ok=True)
    main_path = os.path.join(args.out_dir, stem + ".log")
    tok_path = os.path.join(args.out_dir, stem + "_TOKENS.log")

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()
    entity_assets = load_entity_assets(args.vade_root, args.entity)
    layers_stack = adapter.get_decoder_layers(model)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    cache = BuildBatchCache()

    rows = load_tuples(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir)
    for r in rows:
        r.setdefault("target_attribute", args.attribute)
    cause_all = [r for r in rows if r["rule"] == "match_source"]
    cause_rows = random.Random(args.seed).sample(cause_all, min(args.n_rows, len(cause_all)))
    n_rows = len(cause_rows)
    print(f"[{METHOD_NAME}/swap_trace] {args.entity}/{args.attribute} @ {args.positions}, {args.site}, "
          f"layers {layers_tag}: {n_rows} cause rows "
          f"({len({r['base'] for r in cause_rows})} distinct bases), seed={args.seed}")

    # rec[i] accumulates everything about one row across every layer, so both logs can be written
    # in row order at the end rather than interleaving generation with I/O.
    recs = [{"row": r, "per_layer": {}, "tokens": {}} for r in cause_rows]
    per_layer_counts = {l: {HIT: 0, BASE: 0, OTHER: 0} for l in args.layers}
    collisions = 0

    for lo in range(0, n_rows, args.batch_size):
        idx = list(range(lo, min(lo + args.batch_size, n_rows)))
        batch = build_batch_at(args.positions, [cause_rows[i] for i in idx], entity_assets, adapter,
                               model, processor, batch_cache=cache)
        gen = generate_unhooked(model, batch["base_input_ids"], batch["attention_mask"],
                                batch["base_extra"], pad_token_id, args.max_new_tokens)
        for k, i in enumerate(idx):
            recs[i]["unhooked"] = tokenizer.decode(gen[k].tolist(), skip_special_tokens=True)
            recs[i]["unhooked_toks"] = token_view(tokenizer, gen[k].tolist())
            gl = int(batch["base_gold_len"][k].item())
            sl = int(batch["source_gold_len"][k].item())
            recs[i]["gold_base_toks"] = token_view(tokenizer, batch["base_gold_toks"][k][:gl].tolist())
            recs[i]["gold_source_toks"] = token_view(tokenizer, batch["source_gold_toks"][k][:sl].tolist())

        for layer in args.layers:
            g = full_swap_generation(site, adapter, model, layers_stack, layer, batch, pad_token_id,
                                     args.max_new_tokens)
            labels, both = classify_batch(g, batch)
            collisions += both
            for k, i in enumerate(idx):
                recs[i]["per_layer"][layer] = (tokenizer.decode(g[k].tolist(), skip_special_tokens=True),
                                               labels[k])
                recs[i]["tokens"][layer] = token_view(tokenizer, g[k].tolist())
                per_layer_counts[layer][labels[k]] += 1
            print(f"  rows {lo}-{idx[-1]} layer {layer}: "
                  + " ".join(f"{k}={labels.count(k)}" for k in (HIT, BASE, OTHER)), flush=True)

    if collisions:
        print(f"\n!! {collisions} (row, layer) generations matched BOTH golds -- base_label == "
              f"source_label for those rows, which pruned tuples are supposed to exclude. They are "
              f"counted as {HIT}; treat that rate as an upper bound.")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (f"{args.entity}/{args.attribute} @ {args.positions}, {args.site} site, full source swap, "
              f"layers {layers_tag}\n"
              f"n_rows={n_rows} (seed={args.seed}), batch_size={args.batch_size}, "
              f"max_new_tokens={args.max_new_tokens}\n")

    def summary_lines():
        out = ["", "=" * 100,
               f"SUMMARY: per-layer cause_hit / base_kept / other, over {n_rows} rows", "=" * 100,
               f"{'layer':>6} {'cause_hit':>16} {'base_kept':>16} {'other':>12}"]
        for l in args.layers:
            c = per_layer_counts[l]
            out.append(f"{l:>6} {c[HIT]:>5}/{n_rows} ({c[HIT] / n_rows:6.1%}) "
                       f"{c[BASE]:>5}/{n_rows} ({c[BASE] / n_rows:6.1%}) "
                       f"{c[OTHER]:>5} ({c[OTHER] / n_rows:6.1%})")
        return out

    def template_of(r):
        t = entity_assets.template_lookup[r["queried"]][r["template_id"]]
        return t["question"], t["prefill"]

    with open(main_path, "w") as f:
        f.write(header + f"Generated {stamp}\n\n")
        f.write(f"FULL PER-ROW LOG: {args.entity}/{args.attribute} @ {args.positions}, {args.site}, "
                f"full source swap, n_rows={n_rows}, layers={args.layers}\n" + "=" * 100 + "\n\n")
        for i, rec in enumerate(recs):
            r = rec["row"]
            q, prefill = template_of(r)
            f.write(f"row {i}: base={r['base']} (gold={r['base_label']!r}) -> source={r['source']} "
                    f"(gold={r['source_label']!r})  template={r['template_id']}\n")
            f.write(f"  question: {q!r}\n  prefill : {prefill!r}\n  unhooked: {rec['unhooked']!r}\n")
            for l in args.layers:
                text, label = rec["per_layer"][l]
                f.write(f"  L{l}: {text!r}".ljust(48) + f" [{label}]\n")
            f.write("\n")
        f.write("\n".join(summary_lines()) + "\n")

    if not args.no_tokens_log:
        with open(tok_path, "w") as f:
            f.write(header + "Token breakdown: every answer shown as its actual tokens "
                             "(Qwen2.5-VL splits digits one-per-token)\n" + f"Generated {stamp}\n\n")
            f.write(f"FULL PER-ROW LOG (TOKEN BREAKDOWN): {args.entity}/{args.attribute} @ "
                    f"{args.positions}, {args.site}, full source swap, n_rows={n_rows}, "
                    f"layers={args.layers}\n" + "=" * 100 + "\n\n")
            for i, rec in enumerate(recs):
                r = rec["row"]
                q, prefill = template_of(r)
                f.write(f"row {i}: base={r['base']} -> source={r['source']}  "
                        f"template={r['template_id']}\n")
                f.write(f"  question: {q!r}\n  prefill : {prefill!r}\n")
                f.write(f"  gold base   (expected unhooked) : {rec['gold_base_toks']}\n")
                f.write(f"  gold source (expected if swap works): {rec['gold_source_toks']}\n")
                f.write(f"  unhooked: {rec['unhooked_toks']}\n")
                for l in args.layers:
                    f.write(f"  L{l}: {rec['tokens'][l]} [{rec['per_layer'][l][1]}]\n")
                f.write("\n")
            f.write("\n".join(summary_lines()) + "\n")

    print("\n".join(summary_lines()))
    json_path = os.path.join(args.out_dir, stem + ".json")
    with open(json_path, "w") as f:
        json.dump({"entity": args.entity, "attribute": args.attribute, "positions": args.positions,
                   "site": args.site, "layers": args.layers, "n_rows": n_rows, "seed": args.seed,
                   "split": args.split, "gold_collisions": collisions,
                   "per_layer": {str(l): per_layer_counts[l] for l in args.layers}}, f, indent=2)
    print(f"\nwrote {main_path}" + ("" if args.no_tokens_log else f"\n      {tok_path}")
          + f"\n      {json_path}")


if __name__ == "__main__":
    main()
