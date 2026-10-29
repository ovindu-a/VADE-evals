"""X10: once the entity is in the answer position's residual, do the MLPs write the value ON THEIR OWN?

From block L on, the answer position (the last prompt token, and every generated token after it) is
denied any NEW information from other positions, and we check whether the model still answers.

  attn_self@L   blocks >= L: the position attends ONLY TO ITSELF. Its attention output becomes
                o_proj(repeat_kv(V[self])), exactly what a softmax with all weight on itself gives.
                The residual keeps everything it already holds, and the MLPs run as usual.
  attn_zero@L   blocks >= L: its attention output is zeroed (stricter: drops even the self term).
  mlp_zero@L    blocks >= L: its MLP output is zeroed instead. This is the control: if the value
                still appears, the MLPs past L are not where it is written.

The cut applies at column -1 on every forward call, so it covers the prefill's last prompt token and
each generated token (KV-cached decoding feeds one column at a time). The earlier prompt columns are
untouched, so what the cut position could have READ from them is exactly what is removed.

Readouts per (arm, L), against the clean run on the same rows:
  * the final answer: VADE-matched correct, and its first content token equal to clean's;
  * the logit lens at the last prompt token (residual after each block, through the model's own
    final norm and lm_head): the rank of the asked value's first content token and of the name's.
If attn_self@L keeps the clean answer and the value's lens curve for every L past the layer where the
name arrives, the value is computed from the residual by the MLPs alone. The first L where it breaks
says how late attention is still needed. Rows are restricted to those the clean model gets right.

    python methods/attn_cutoff.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --dry_run
    VADE_PROMPT_STYLE=what_is python methods/attn_cutoff.py --model_id llava-hf/llava-1.5-7b-hf
    python methods/attn_cutoff.py --entity brands --n_items 128          # X12: other entities

--entity brands|celebrities (X12): their labels are lowercase and the model adds articles, so
`correct` is VADE's case-insensitive matcher (as for flags), and the lens tracks the clean run's
OWN first answer token instead of the label's (a second clean pass, so the clean ranks use it too).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("attn_self", "attn_zero", "mlp_zero")
ATTRIBUTES = ["capital", "currency", "language", "calling_code"]


def n_kv_heads(model):
    cfg = getattr(model.config, "text_config", None) or model.config
    return getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads


def self_only(v_last, n_heads, n_kv, o_proj):
    """[B, n_kv*hd] value vectors of ONE position -> that position's attention output if every query
    head put all its weight on itself: each query head h reads kv head h // (n_heads // n_kv), as
    transformers' repeat_kv does."""
    B = v_last.shape[0]
    hd = v_last.shape[-1] // n_kv
    v = v_last.view(B, n_kv, hd).repeat_interleave(n_heads // n_kv, dim=1)
    return o_proj(v.reshape(B, n_heads * hd))


def register_cut(attn_modules, mlp_modules, arm, blocks, n_heads, n_kv):
    """Hooks implementing `arm` at column -1 in `blocks`. -> handles. attn_modules/mlp_modules map
    block -> module (self_attn returns a tuple whose [0] is the sublayer output; the MLP a tensor)."""
    handles = []
    for b in blocks:
        if arm == "mlp_zero":
            def mlp_hook(mod, inp, out):
                out = out.clone()
                out[:, -1] = 0
                return out
            handles.append(mlp_modules[b].register_forward_hook(mlp_hook))
            continue
        attn = attn_modules[b]
        cap = {}
        if arm == "attn_self":
            handles.append(attn.v_proj.register_forward_hook(
                lambda mod, inp, out, cap=cap: cap.__setitem__("v", out[:, -1])))

        def attn_hook(mod, inp, out, cap=cap):
            first = out[0] if isinstance(out, tuple) else out
            first = first.clone()
            first[:, -1] = 0 if arm == "attn_zero" else self_only(cap["v"], n_heads, n_kv, mod.o_proj).to(first.dtype)
            return (first,) + tuple(out[1:]) if isinstance(out, tuple) else first
        handles.append(attn.register_forward_hook(attn_hook))
    return handles


def register_lens(layers, sink):
    """Captures the residual after every block at column -1 during the PREFILL only (T > 1)."""
    handles = []
    for i, layer in enumerate(layers):
        def hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                sink[i + 1] = h[:, -1].detach()
        handles.append(layer.register_forward_hook(hook))
    return handles


def lens_ranks(adapter, model, sink, tracked):
    """-> {layer: [B, n_tracked] full-vocab ranks (1 = top)} and {layer: [B] top-1 ids}."""
    import torch
    ranks, top1 = {}, {}
    with torch.no_grad():
        for L, h in sink.items():
            lg = adapter.unembed(model, h).float()
            sel = lg.gather(1, tracked)
            ranks[L] = (1 + (lg[:, None, :] > sel[:, :, None]).sum(-1)).cpu()
            top1[L] = lg.argmax(-1).cpu()
    return ranks, top1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=os.environ.get("VADE_ROOT", os.path.join(REPO_ROOT, "..", "VADE")))
    ap.add_argument("--entity", default="flags", choices=["flags", "brands", "celebrities"])
    ap.add_argument("--attributes", nargs="+", default=None, help="default: the entity's controlled attributes")
    ap.add_argument("--n_items", type=int, default=16, help="items, evenly spaced over the sorted keys")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--layer_step", type=int, default=2, help="cut-from layers 0, step, 2*step, ...")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    from methods.circuit_common import (Model, cat_extra, first_content_token, gold_ids, load_entity,
                                        PREFILL, question)
    from methods.attribute_switch_sweep import ENTITY_SPECS, PROMPT_STYLE
    from methods.common.text_match import contains_label
    args.attributes = args.attributes or list(ENTITY_SPECS[args.entity]["attributes"])
    entity_dir, items, _ = load_entity(args.vade_root, args.entity)
    keys = sorted(items)
    keys = keys[::max(1, len(keys) // args.n_items)][:args.n_items]
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "attn_cutoff", args.model_id.split("/")[-1]
                                           + ("" if args.entity == "flags" else f"_{args.entity}"))
    print(f"[attn_cutoff] {args.model_id}: {len(keys)} {args.entity} x {len(args.attributes)} attributes; "
          f"prompt style {PROMPT_STYLE!r}: {question(args.attributes[0])[-45:]!r} + {PREFILL!r}; arms {args.arms} -> {out_dir}")
    if args.dry_run:
        print("items:", " ".join(keys))
        print("Grid valid.")
        return

    import torch
    m = Model(args.model_id, vade_root=args.vade_root)
    tok, model, adapter = m.tokenizer, m.model, m.adapter
    n_layers, n_heads, n_kv = m.n_layers, m.n_heads, n_kv_heads(model)
    attn_mods = {b: adapter.get_attn_block(model, b) for b in range(n_layers)}
    mlp_mods = {b: adapter.get_mlp_block(model, b) for b in range(n_layers)}
    cut_layers = list(range(0, n_layers, args.layer_step))
    os.makedirs(out_dir, exist_ok=True)
    fct = lambda s: first_content_token(tok, gold_ids(tok, str(s)))

    # Per attribute one batch shape: every flag's controlled prompt has the same length.
    batches = []
    for a in args.attributes:
        labelled = [k for k in keys if items[k].get(a) not in (None, "")]   # brands: AIRBUS has no industry
        for i in range(0, len(labelled), args.batch_size):
            ks = labelled[i:i + args.batch_size]
            built = [m.image_prompt(entity_dir, items[k], a) for k in ks]
            assert len({b[0].shape[0] for b in built}) == 1, "controlled prompts must share one length"
            ids = torch.stack([b[0] for b in built])
            tracked = torch.tensor([[fct(items[k][a]), fct(items[k]["name"])] for k in ks])
            batches.append(dict(attribute=a, keys=ks, ids=ids, mask=torch.ones_like(ids),
                                extra=cat_extra([b[1] for b in built]), tracked=tracked))

    def run(arm, L):
        """-> one record per row: generated text, first content token, lens ranks per layer."""
        recs = []
        for bt in batches:
            sink = {}
            hs = register_lens(m.layers, sink)
            if arm != "clean":
                hs += register_cut(attn_mods, mlp_mods, arm, range(L, n_layers), n_heads, n_kv)
            try:
                gen = m.generate(bt["ids"], bt["mask"], bt["extra"], args.max_new_tokens)
            finally:
                for h in hs:
                    h.remove()
            ranks, top1 = lens_ranks(adapter, model, sink, bt["tracked"].to(model.device))
            texts = m.decode(gen)
            for r, k in enumerate(bt["keys"]):
                first = next((int(t) for t in gen[r] if tok.decode([int(t)]).strip()), None)
                recs.append({"item": k, "attribute": bt["attribute"], "generated": texts[r], "first": first,
                             "correct": contains_label(texts[r], str(items[k][bt["attribute"]])),
                             "said_name": contains_label(texts[r], items[k]["name"]),
                             "value_rank": {L_: int(ranks[L_][r, 0]) for L_ in ranks},
                             "name_rank": {L_: int(ranks[L_][r, 1]) for L_ in ranks},
                             "lens_top1": tok.decode([int(top1[n_layers][r])])})
        return recs

    clean = run("clean", None)
    if args.entity != "flags":
        # Lowercase labels do not tokenize like the model's answer: track its own first answer token.
        it = iter(clean)
        for bt in batches:
            for r in range(len(bt["keys"])):
                first = next(it)["first"]
                if first is not None:
                    bt["tracked"][r, 0] = first
        clean = run("clean", None)
    # Sanity: the lens at the last layer IS the model's next-token prediction, so its top-1 must be
    # the first generated token (or whitespace before it) on clean rows.
    agree = sum(r["lens_top1"].strip() == tok.decode([r["first"]]).strip() for r in clean if r["first"] is not None)
    print(f"  clean: accuracy {sum(r['correct'] for r in clean) / len(clean):.1%}, said name "
          f"{sum(r['said_name'] for r in clean) / len(clean):.1%}; lens@last top-1 == first answer token on "
          f"{agree}/{len(clean)} rows", flush=True)
    ok = [i for i, r in enumerate(clean) if r["correct"]]
    assert ok, "the clean model answers nothing correctly under this prompt style -- pick another"
    med = lambda xs: float(torch.tensor(xs, dtype=torch.float).median()) if xs else float("nan")
    arrival = next((L for L in range(1, n_layers + 1) if med([clean[i]["name_rank"][L] for i in ok]) <= 10), None)
    value_top = next((L for L in range(1, n_layers + 1) if med([clean[i]["value_rank"][L] for i in ok]) <= 1), None)
    print(f"  clean lens (correct rows): name median rank <= 10 from L{arrival}; asked value median rank 1 "
          f"from L{value_top}", flush=True)

    summary = {"clean": {"n": len(clean), "n_correct": len(ok), "accuracy": len(ok) / len(clean),
                         "name_arrives": arrival, "value_top1": value_top,
                         "value_rank_median": {L: med([clean[i]["value_rank"][L] for i in ok]) for L in range(1, n_layers + 1)},
                         "name_rank_median": {L: med([clean[i]["name_rank"][L] for i in ok]) for L in range(1, n_layers + 1)}}}
    rows = [dict(r, arm="clean", cut_from=None) for r in clean]
    for arm in args.arms:
        for L in cut_layers:
            recs = run(arm, L)
            kept = [recs[i] for i in ok]
            s = {"n": len(kept),
                 "correct": sum(r["correct"] for r in kept) / len(kept),
                 "same_first_token": sum(r["first"] == clean[i]["first"] for r, i in zip(kept, ok)) / len(kept),
                 "said_name": sum(r["said_name"] for r in kept) / len(kept),
                 "value_rank_final_median": med([r["value_rank"][n_layers] for r in kept]),
                 "value_rank_median": {L_: med([r["value_rank"][L_] for r in kept]) for L_ in range(1, n_layers + 1)},
                 "by_attribute": {a: sum(r["correct"] for r in kept if r["attribute"] == a) /
                                  max(1, sum(r["attribute"] == a for r in kept)) for a in args.attributes}}
            summary.setdefault(arm, {})[L] = s
            rows += [dict(r, arm=arm, cut_from=L) for r in recs]
            print(f"  {arm:<9} from L{L:<3} correct {s['correct']:6.1%}  same 1st token {s['same_first_token']:6.1%}  "
                  f"said name {s['said_name']:5.1%}  value rank@final {s['value_rank_final_median']:7.0f}  "
                  + " ".join(f"{a[:4]} {v:.0%}" for a, v in s["by_attribute"].items()), flush=True)

    json.dump({"model_id": args.model_id, "entity": args.entity, "prompt_style": PROMPT_STYLE, "n_layers": n_layers, "items": keys,
               "attributes": args.attributes, "summary": summary},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir}/summary.json and rows.jsonl")


if __name__ == "__main__":
    main()
