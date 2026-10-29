"""Recognition check + tuple pruning for ANY adapter, in this repo (X15).

VADE prunes with two scripts of its own (models/run_accuracy_sweep.py, then models/prune_tuples.py),
which methods/vade_accuracy_sweep.py runs unmodified. Those scripts are not in the molab data bucket,
and they build prompts with the processor's batched chat-template call rather than the adapter's
build_inputs -- the input every experiment here actually feeds the model. This script is the same two
steps through the adapter:

  1. accuracy  For every (attribute, template_id, item) that some tuple row's target depends on,
               ONE clean greedy generation from adapter.build_inputs (left-padded batches of one
               template), judged by VADE's own matcher (common/text_match.contains_label: casefold,
               diacritics stripped, whole word). Written in run_accuracy_sweep's record format to
               <vade_root>/models/<slug>/stats/<entity>_predictions.jsonl (resumable: done keys
               are skipped), plus <entity>_accuracy.json (per attribute and template).
  2. prune     prune_tuples.py's rule, unchanged: a row survives iff the model answered the item its
               target depends on correctly -- the SOURCE for match_source ("cause") rows, the BASE for
               match_base ("iso") rows. Written to <vade_root>/models/<slug>/<entity>/tuples/<attr>/
               {train,test}.jsonl, where every script here looks for pruned tuples, and
               <entity>_tuples_prune_summary.json. A PRUNED_BY.json sidecar records that this script,
               not VADE's, made them.

Usage:
    python methods/own_accuracy_prune.py --model_id google/gemma-3-12b-it --entities flags brands \\
        --attributes capital language currency calling_code
    python methods/own_accuracy_prune.py --model_id ... --entities flags --dry_run   # count work, no model
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))
SPLITS = ("train", "test")


def check_item_for_row(row):
    """VADE/models/prune_tuples.py's rule: cause rows depend on the source, iso rows on the base."""
    return row["source"] if row["rule"] == "match_source" else row["base"]


def gold_for_row(row):
    return row["source_label"] if row["rule"] == "match_source" else row["base_label"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def needed_keys(vade_root, entity, attributes):
    """-> {(attribute, template_id, item_id): gold_label} over every tuple row of every split, and the
    attributes actually found."""
    root = os.path.join(vade_root, "data", entity, "tuples")
    attrs = sorted(a for a in os.listdir(root) if os.path.isdir(os.path.join(root, a)))
    if attributes:
        missing = sorted(set(attributes) - set(attrs))
        assert not missing, f"{entity}: no tuples for {missing} (have {attrs})"
        attrs = [a for a in attrs if a in attributes]
    keys = {}
    for a in attrs:
        for split in SPLITS:
            path = os.path.join(root, a, f"{split}.jsonl")
            if not os.path.exists(path):
                continue
            for row in read_jsonl(path):
                if "rule" not in row:
                    # `identity` tuples (queried = recognition_freeform, no rule): not pruned, but both
                    # items' names are generated -- that IS the recognition check.
                    pairs = [(row["base"], row["base_label"]), (row["source"], row["source_label"])]
                else:
                    pairs = [(check_item_for_row(row), gold_for_row(row))]
                for item, gold in pairs:
                    k = (row["queried"], row["template_id"], item)
                    assert keys.setdefault(k, gold) == gold, f"two gold labels for {k}: {keys[k]!r} vs {gold!r}"
    return keys, attrs


def prune(vade_root, slug, entity, attrs, lookup):
    out_root = os.path.join(vade_root, "models", slug, entity, "tuples")
    summary = {}
    for a in attrs:
        summary[a] = {}
        for split in SPLITS:
            src = os.path.join(vade_root, "data", entity, "tuples", a, f"{split}.jsonl")
            if not os.path.exists(src):
                continue
            rows = read_jsonl(src)
            if rows and "rule" not in rows[0]:
                print(f"  [{entity}/{a}/{split}] no `rule` field (identity tuples): recognition only, not pruned")
                break
            kept, wrong, missing = [], 0, 0
            for row in rows:
                ok = lookup.get((row["queried"], row["template_id"], check_item_for_row(row)))
                if ok is None:
                    missing += 1
                elif ok:
                    kept.append(row)
                else:
                    wrong += 1
            path = os.path.join(out_root, a, f"{split}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                for row in kept:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cause = sum(r["rule"] == "match_source" for r in kept)
            summary[a][split] = {"original_n": len(rows), "kept_n": len(kept), "dropped_wrong": wrong,
                                 "dropped_missing_baseline": missing, "cause_kept": cause,
                                 "iso_kept": len(kept) - cause}
            print(f"  [{entity}/{a}/{split}] {len(rows)} -> {len(kept)} rows (cause {cause}, wrong {wrong}, "
                  f"missing {missing})")
    with open(os.path.join(vade_root, "models", slug, entity, "PRUNED_BY.json"), "w") as f:
        json.dump({"script": "VADE-evals/methods/own_accuracy_prune.py", "model_slug": slug,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rule": "VADE prune_tuples.py: source correct for match_source, base correct for match_base",
                   "matcher": "common/text_match.contains_label (copy of VADE eval/score.py)"}, f, indent=2)
    return summary


def generate_batch(adapter, model, processor, assets, attribute, template_id, item_ids, max_new_tokens):
    import torch
    from PIL import Image
    from methods.common.entities import concat_extra
    from methods.common.hooks import extra_to_device
    tmpl = assets.template_lookup[attribute][template_id]
    built = []
    for it in item_ids:
        img = Image.open(os.path.join(assets.entity_dir, assets.items[it]["image"])).convert("RGB")
        built.append(adapter.build_inputs(processor, img, tmpl["question"], tmpl["prefill"]))
    pad = processor.tokenizer.pad_token_id
    pad = pad if pad is not None else processor.tokenizer.eos_token_id
    T = max(b["input_ids"].shape[0] for b in built)
    ids = torch.full((len(built), T), pad, dtype=torch.long)
    mask = torch.zeros((len(built), T), dtype=torch.long)
    for i, b in enumerate(built):
        n = b["input_ids"].shape[0]
        ids[i, T - n:] = b["input_ids"]
        mask[i, T - n:] = 1
    extra = extra_to_device(concat_extra([b["extra"] for b in built]), model.device, model.dtype)
    with torch.no_grad():
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), **extra,
                             max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
    return processor.tokenizer.batch_decode(gen[:, T:], skip_special_tokens=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--entities", nargs="+", required=True)
    ap.add_argument("--attributes", nargs="+", default=None, help="Default: every attribute with tuples.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=16, help="VADE run_accuracy_sweep.py's default.")
    ap.add_argument("--attn_implementation", default=None)
    ap.add_argument("--dry_run", action="store_true", help="Count the generations needed; load no model.")
    args = ap.parse_args(argv)

    slug = args.model_id.split("/")[-1]
    stats_dir = os.path.join(args.vade_root, "models", slug, "stats")
    plan = {e: needed_keys(args.vade_root, e, args.attributes) for e in args.entities}
    for e, (keys, attrs) in plan.items():
        per = defaultdict(int)
        for a, _, _ in keys:
            per[a] += 1
        print(f"[{e}] {len(keys)} clean generations needed: {dict(per)}")
    if args.dry_run:
        print("Grid valid.")
        return

    from methods.adapters.registry import get_adapter
    from methods.common.entities import load_entity_assets
    from methods.common.text_match import contains_label
    adapter = get_adapter(args.model_id)
    model, processor = adapter.load(attn_implementation=args.attn_implementation)
    os.makedirs(stats_dir, exist_ok=True)
    for entity, (keys, attrs) in plan.items():
        assets = load_entity_assets(args.vade_root, entity)
        pred_path = os.path.join(stats_dir, f"{entity}_predictions.jsonl")
        done = {}
        if os.path.exists(pred_path):
            for r in read_jsonl(pred_path):
                done[(r["attribute"], r["template_id"], r["item_id"])] = bool(r["correct"])
        todo = defaultdict(list)
        for (a, t, it) in sorted(keys):
            if (a, t, it) not in done:
                todo[(a, t)].append(it)
        n_todo = sum(map(len, todo.values()))
        print(f"\n[{entity}] {len(keys) - n_todo} cached, {n_todo} to generate -> {pred_path}", flush=True)
        t0 = time.time()
        with open(pred_path, "a") as f:
            for (a, t), items in sorted(todo.items()):
                for i in range(0, len(items), args.batch_size):
                    chunk = items[i:i + args.batch_size]
                    texts = generate_batch(adapter, model, processor, assets, a, t, chunk, args.max_new_tokens)
                    for it, text in zip(chunk, texts):
                        gold = keys[(a, t, it)]
                        ok = contains_label(text, gold)
                        done[(a, t, it)] = ok
                        f.write(json.dumps({"entity": entity, "attribute": a, "template_id": t, "item_id": it,
                                            "generated_text": text, "gold_label": gold, "correct": ok},
                                           ensure_ascii=False) + "\n")
                    f.flush()
                print(f"  {a}/{t}: {sum(done[(a, t, it)] for it in items)}/{len(items)} correct "
                      f"({time.time() - t0:.0f}s)", flush=True)
        acc = defaultdict(lambda: [0, 0])
        for (a, t, it) in keys:
            acc[a][0] += done[(a, t, it)]
            acc[a][1] += 1
            acc[f"{a}/{t}"][0] += done[(a, t, it)]
            acc[f"{a}/{t}"][1] += 1
        report = {k: {"n": n, "correct": c, "accuracy": c / n} for k, (c, n) in sorted(acc.items())}
        with open(os.path.join(stats_dir, f"{entity}_accuracy.json"), "w") as f:
            json.dump({"model_id": args.model_id, "entity": entity, "by": report}, f, indent=2)
        print(f"[{entity}] accuracy: " + ", ".join(f"{k} {v['accuracy']:.1%} (n={v['n']})"
                                                    for k, v in report.items() if "/" not in k))
        summary = prune(args.vade_root, slug, entity, attrs, done)
        with open(os.path.join(stats_dir, f"{entity}_tuples_prune_summary.json"), "w") as f:
            json.dump({"model_id": slug, "entity": entity, "pruned_by": "own_accuracy_prune.py",
                       "attributes": summary}, f, indent=2)


if __name__ == "__main__":
    main()
