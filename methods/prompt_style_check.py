"""Which controlled prompt does a model actually ANSWER? (X4 prep for LLaVA-1.5, see X9.)

attribute_switch_sweep's controlled prompts ('<instruction>Report the <field> of this country.' +
prefill 'Answer:') were built on Qwen2.5-VL. LLaVA-1.5 answers the country NAME to all of them on
images (X9), so its routers / compose / crossmodal runs would score nothing. This script generates
under several candidate styles for every flag x attribute, image and text-named, and reports per
style: VADE-matched accuracy, how often the answer is the country name instead, and whether the
style keeps the controlled design (every attribute's prompt the same token length with the same
final token, i.e. attribute_switch_sweep.alignment holds for every base/donor pair).

    python methods/prompt_style_check.py --model_id llava-hf/llava-1.5-7b-hf --dry_run
    python methods/prompt_style_check.py --model_id llava-hf/llava-1.5-7b-hf
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.attribute_switch_sweep import (ATTRIBUTES, ENTITY_SPECS, PROMPT_STYLES, style_prefill,  # noqa: E402
                                            style_question)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def text_ids(m, q, prefill):
    """The text-entity prompt with a given question/prefill, tokenized as circuit_common.text_prompt."""
    import torch
    messages = [{"role": "user", "content": [{"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": prefill}]}]
    text = m.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                           continue_final_message=True)
    add = getattr(m.adapter, "_add_special", lambda p, t: False)(m.processor, text)
    return torch.tensor(m.tokenizer(text, add_special_tokens=add)["input_ids"], dtype=torch.long)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--vade_root", default=os.environ.get("VADE_ROOT", os.path.join(REPO_ROOT, "..", "VADE")))
    ap.add_argument("--styles", nargs="+", default=list(PROMPT_STYLES))
    ap.add_argument("--modes", nargs="+", default=["image", "text"], choices=["image", "text"])
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    from methods.circuit_common import Model, left_pack, load_entity
    from methods.common.text_match import contains_label
    entity_dir, items, _ = load_entity(args.vade_root, "flags")
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    noun = ENTITY_SPECS["flags"]["noun"]
    for s in args.styles:
        assert s in PROMPT_STYLES, f"unknown style {s}; known: {list(PROMPT_STYLES)}"
        for a in ATTRIBUTES:
            print(f"  [{s}] {a}: {style_question(a, s)!r} / {style_prefill(a, s)!r}")
    if args.dry_run:
        print("Grid valid.")
        return

    import torch
    from PIL import Image
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "prompt_style_check",
                                           args.model_id.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    m = Model(args.model_id, vade_root=args.vade_root)
    images = {}
    for k in keys:
        with Image.open(os.path.join(entity_dir, items[k]["image"])) as im:
            images[k] = im.convert("RGB")
    summary, rows = {}, []
    for style in args.styles:
        for mode in args.modes:
            lengths, finals = {}, {}
            for a in ATTRIBUTES:
                prefill = style_prefill(a, style)
                built = []
                for k in keys:
                    if mode == "image":
                        x = m.adapter.build_inputs(m.processor, images[k], style_question(a, style), prefill)
                        built.append((x["input_ids"], x["extra"]))
                    else:
                        built.append((text_ids(m, style_question(a, style, text_name=items[k]["name"]), prefill), {}))
                lengths[a] = sorted({int(b[0].shape[0]) for b in built})
                finals[a] = sorted({int(b[0][-1]) for b in built})
                gens = []
                for i in range(0, len(built), args.batch_size):
                    chunk = built[i:i + args.batch_size]
                    ids, mask = left_pack([b[0] for b in chunk], m.pad_id)
                    extra = {}
                    if mode == "image":
                        extra = {kk: torch.cat([b[1][kk] for b in chunk]) for kk in chunk[0][1]}
                    gens += m.decode(m.generate(ids, mask, extra, args.max_new_tokens))
                n_ok = n_name = 0
                for k, g in zip(keys, gens):
                    ok = contains_label(g, str(items[k][a]))
                    name = (not ok) and contains_label(g, items[k]["name"])
                    n_ok += ok
                    n_name += name
                    rows.append({"style": style, "mode": mode, "item": k, "attribute": a, "generated": g,
                                 "label": str(items[k][a]), "correct": ok, "said_name": name})
                summary.setdefault(style, {}).setdefault(mode, {})[a] = {
                    "n": len(keys), "accuracy": n_ok / len(keys), "said_name": n_name / len(keys),
                    "examples": gens[:4]}
                print(f"  [{style}/{mode}] {a:<13} acc {n_ok / len(keys):6.1%}  said name "
                      f"{n_name / len(keys):6.1%}  e.g. {gens[:3]}", flush=True)
            if mode == "image":   # the controlled design: one length and one final token across attributes
                aligned = len({tuple(v) for v in lengths.values()}) == 1 and all(len(v) == 1 for v in lengths.values()) \
                    and len({tuple(v) for v in finals.values()}) == 1
                summary[style][mode]["_aligned"] = aligned
                summary[style][mode]["_lengths"] = lengths
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"model_id": args.model_id, "n_items": len(keys), "styles": {
            s: {"question": {a: style_question(a, s) for a in ATTRIBUTES},
                "prefill": {a: style_prefill(a, s) for a in ATTRIBUTES}} for s in args.styles},
            "summary": summary}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nstyle            mode   " + " ".join(f"{a[:12]:>14}" for a in ATTRIBUTES) + "   aligned")
    for s, d in summary.items():
        for mode, per in d.items():
            print(f"{s:<16} {mode:<6} " + " ".join(f"{per[a]['accuracy']:6.0%}/{per[a]['said_name']:<6.0%}"
                                                   for a in ATTRIBUTES) + f"   {per.get('_aligned', '-')}")
    print("(accuracy / said the country name instead)")
    print(f"wrote {out_dir}/summary.json and rows.jsonl")


if __name__ == "__main__":
    main()
