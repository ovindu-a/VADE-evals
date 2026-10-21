"""X6: what do the ICLR 2027 'entity-recognition gate' heads (16.1 16.14 16.17 16.20 17.21) read,
and do they carry IDENTITY or FAMILIARITY?

X5 (gate_knockout.py) found on flags that knocking these five heads out leaves open
identification (100%) and fact recall (97%) intact and only turns `known,<name>` into `unknown`
(81.7% vs 0% for random heads) -- an abstention signal, not recognition. This script looks inside.

THREE PARTS, one eager model load, one entity (flags / brands / celebrities)

  A. ATTENTION. For each gate head (and, for comparison, common10's movers and random heads from
     blocks 16-17), the attention mass from the LAST prompt column onto key groups:
         system | object (entity_only image tokens) | background (other image tokens) |
         question | contract_words (question tokens containing 'known'/'unknown') | assistant
     on two prompts per item: the paper's `known`/`unknown` contract, and our controlled fact
     prompt. Mass is reported in total and per token (groups differ ~20x in size).

  B. THE DECISION. On the contract prompt the first answer token is 'known' or 'unknown'.
       ld          logit(known) - logit(unknown) at the last column, from the real logits,
                   clean vs KO(gate5) vs KO(random5) (gate_knockout's mean ablation)
       head_dla    each gate head's DIRECT write z_h W_O_h^T projected on the known-minus-unknown
                   unembedding direction, times the model's own final-norm scale (dla.py's exact
                   per-component term); positive = pushes toward 'known'

  C. SWAPS (interchange, not ablation). Replace the five heads' o_proj input at every prompt
     position with their values from a DONOR run of the same contract prompt (same length, so
     positions align) and generate:
       donor = another item         name follows donor -> heads carry IDENTITY
                                    status/name stay   -> they do not
       donor = blank canvas         status -> unknown  -> heads carry (lack of) FAMILIARITY
       donor = degraded same item   (32x32 down/up-sampled, the paper's perturbation)
       donor = self                 must reproduce clean exactly (self-check)
     Each row is classified: known+base name / known+donor name / known+other / unknown.

Usage
-----
    python methods/gate_heads_probe.py --dry_run
    python methods/gate_heads_probe.py --limit_items 8                    # smoke
    python methods/gate_heads_probe.py --entity flags                     # full
    python methods/gate_heads_probe.py --entity celebrities
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import DEFAULT_VADE_ROOT, MODEL_ID, MOVERS, PREFILL, load_entity  # noqa: E402
from methods.attribute_switch_sweep import ENTITY_SPECS, question  # noqa: E402
from methods.gate_knockout import (GATE5, block_matched_random, head_cols, name_hit, norm,  # noqa: E402
                                   parse_heads)

GROUPS = ["system", "object", "background", "question", "contract_words", "assistant"]

CONTRACT = {
    "flags": ("Do you know which country this flag belongs to? Answer \"known\" if you recognise it, "
              "and \"unknown\" otherwise.", "the country name in lowercase (e.g. \"france\")",
              "known,france / known,japan / known,brazil / unknown,4"),
    "brands": ("Do you know which real-world brand this logo belongs to? Answer \"known\" if you "
               "recognise it, and \"unknown\" otherwise.", "the brand name in lowercase (e.g. \"adidas\")",
               "known,adidas / known,apple / known,nike / unknown,5"),
    "celebrities": ("Do you know who the person is in the image? Answer with \"known\" if yes and "
                    "\"unknown\" otherwise.", "the person's full real-world name in lowercase "
                    "(e.g. \"barack obama\")", "known,barack obama / known,serena williams / unknown,4"),
}


def contract_prompt(entity):
    q, value, examples = CONTRACT[entity]
    return ('Return exactly: "<status>,<value>"\nRules:\n- <status> must be either "known" or "unknown".\n'
            f'- {q}\nOutput:\n- If status is "known", <value> must be {value}.\n'
            '- If status is "unknown", <value> must be the number of distinct visible elements in the '
            f'image as digits only.\nExamples: {examples}\nDo not explain your reasoning. Output only the answer.')


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------

def key_groups(ids, image_id, object_flat, assistant_col, contract_cols=()):
    """ids: 1-D token list. -> {group: [cols]}. object_flat indexes the image tokens in raster order."""
    img = [i for i, t in enumerate(ids) if t == image_id]
    assert img, "no image tokens in the prompt"
    obj = {img[j] for j in object_flat if j < len(img)}
    contract = set(contract_cols)
    g = {k: [] for k in GROUPS}
    for c in range(len(ids)):
        if c in obj:
            g["object"].append(c)
        elif c in img:
            g["background"].append(c)
        elif c < img[0]:
            g["system"].append(c)
        elif c >= assistant_col:
            g["assistant"].append(c)
        elif c in contract:
            g["contract_words"].append(c)
        else:
            g["question"].append(c)
    return g


def classify_swap(text, base_key, donor_key, items):
    """-> 'unknown' | 'known_base' | 'known_donor' | 'known_other' (the paper's contract parse)."""
    status, _, value = text.strip().strip('"').strip().partition(",")
    if not norm(status).startswith("known"):
        return "unknown"
    if name_hit(value, items[base_key]["name"]):
        return "known_base"
    if donor_key in items and name_hit(value, items[donor_key]["name"]):
        return "known_donor"
    return "known_other"


def degrade(img, size=32):
    """The paper's 32x32 downsample (then back up, so the prompt keeps its token count)."""
    from PIL import Image
    return img.resize((size, size), Image.BILINEAR).resize(img.size, Image.BILINEAR)


def blank_like(img):
    from PIL import Image
    return Image.new("RGB", img.size, (128, 128, 128))


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

def capture_ohp_input(adapter, model, blocks, sink):
    """Pre-hooks recording o_proj's input (all positions) of the FIRST forward per block."""
    handles = []
    for b in blocks:
        def grab(_m, args, _b=b):
            if _b not in sink:
                sink[_b] = args[0].detach().clone()
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(grab))
    return handles


def install_values(adapter, model, spec, head_dim, donor):
    """spec {block: [heads]}; donor {block: [B, T, W]} -> pre-hooks writing the donor's columns at every
    prompt position (prefill only; decode steps, T == 1, pass through)."""
    import torch
    handles = []
    for b, heads in spec.items():
        cols = torch.tensor(head_cols(heads, head_dim), dtype=torch.long)

        def pre(_m, args, _b=b, _cols=cols):
            x = args[0]
            if x.shape[1] == 1:
                return None
            d = donor[_b].to(x.device, x.dtype)
            assert d.shape == x.shape, f"donor {tuple(d.shape)} vs base {tuple(x.shape)}: prompts must align"
            out = x.clone()
            c = _cols.to(x.device)
            out[..., c] = d[..., c]
            return (out,) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(pre))
    return handles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags", choices=sorted(CONTRACT))
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--gate", default=GATE5)
    ap.add_argument("--movers", default=MOVERS)
    ap.add_argument("--n_random", type=int, default=2)
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    entity_dir, items, _ = load_entity(args.vade_root, args.entity)
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    gate = parse_heads(args.gate)
    rng = random.Random(args.seed)
    randoms = [block_matched_random(gate, 28, rng) for _ in range(args.n_random)]
    movers = parse_heads(args.movers)
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    object_flat = loc["object_token_indices"]["entity_only"]["flat"]
    fact_attr = ENTITY_SPECS[args.entity]["attributes"][0]
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "gate_heads_probe", args.entity)
    donors = {k: keys[(i + 1) % len(keys)] for i, k in enumerate(keys)}      # a fixed derangement
    print(f"[gate_heads_probe] {args.entity}: {len(keys)} items; gate {args.gate}; "
          f"{len(randoms)} random sets; fact probe '{fact_attr}'; object tokens {len(object_flat)} -> {out_dir}")
    if args.dry_run:
        print(contract_prompt(args.entity))
        print("Grid valid.")
        return

    import torch
    from PIL import Image
    from methods.circuit_common import Model
    from methods.gate_knockout import register_mean_ablation

    m = Model(args.model_id, device=args.device, attn_impl="eager", vade_root=args.vade_root)
    tok = m.tokenizer
    os.makedirs(out_dir, exist_ok=True)
    contract_q = contract_prompt(args.entity)

    def tid(s):
        return tok(s, add_special_tokens=False)["input_ids"][0]
    known_ids = sorted({tid(" known"), tid("known")})
    unknown_ids = sorted({tid(" unknown"), tid("unknown")})
    assert not set(known_ids) & set(unknown_ids), "known/unknown share a first token"
    with torch.no_grad():
        d_known = m.adapter.logit_direction(m.model, torch.tensor(known_ids, device=m.model.device)).float()
        d_unk = m.adapter.logit_direction(m.model, torch.tensor(unknown_ids, device=m.model.device)).float()
    direction = (d_known.mean(0) - d_unk.mean(0)).cpu()                                    # [H]

    images = {}

    def image(k, kind="clean"):
        if k not in images:
            with Image.open(os.path.join(entity_dir, items[k]["image"])) as im:
                images[k] = im.convert("RGB")
        im = images[k]
        return {"clean": im, "blank": blank_like(im), "degraded": degrade(im)}[kind]

    def batch(prompts):
        built = [m.adapter.build_inputs(m.processor, im, q, PREFILL) for im, q in prompts]
        lens = {len(b["input_ids"]) for b in built}
        assert len(lens) == 1, f"prompt lengths differ {lens}; swaps need aligned positions"
        ids = torch.stack([b["input_ids"] for b in built])
        mask = torch.ones_like(ids)
        extra = {k: torch.cat([b["extra"][k] for b in built]) for k in built[0]["extra"]}
        return ids, mask, extra

    def ld_of(logits):
        lg = logits[:, -1].float()
        return (lg[:, known_ids].max(-1).values - lg[:, unknown_ids].max(-1).values).cpu()

    heads_report = {"gate5": gate, "movers": movers, **{f"random{i}": r for i, r in enumerate(randoms)}}
    blocks_needed = sorted({b for hs in heads_report.values() for b, _ in hs})
    attn = {p: defaultdict(lambda: defaultdict(list)) for p in ("contract", "fact")}
    dla = defaultdict(list)
    ld = defaultdict(list)
    swaps = []

    for s in range(0, len(keys), args.batch_size):
        ks = keys[s:s + args.batch_size]
        # ---- A + B: clean eager forward on both prompts ----
        for pname, qtext in (("contract", contract_q), ("fact", question(fact_attr))):
            ids, mask, extra = batch([(image(k), qtext) for k in ks])
            sink = {}
            hs = capture_ohp_input(m.adapter, m.model, blocks_needed, sink)
            fin = {}

            def grab_final(_m, a):
                fin.setdefault("x", a[0][:, -1].detach().float())          # returns None: input unchanged
            hs.append(m.adapter._final_norm(m.model).register_forward_pre_hook(grab_final))
            try:
                out = m.forward(ids, mask, extra, output_attentions=True)
            finally:
                for h in hs:
                    h.remove()
            row_ids = ids[0].tolist()
            a_col = m.adapter.assistant_start_col(m.processor, ids[0])
            contract_cols = [c for c in range(len(row_ids)) if "known" in tok.decode([row_ids[c]]).lower()]
            g = key_groups(row_ids, m.image_id, object_flat, a_col, contract_cols)
            for name, hset in heads_report.items():
                for b, h in hset:
                    a = out.attentions[b][:, h, -1].float().cpu()                         # [B, T]
                    for grp, cols in g.items():
                        if cols:
                            mass = a[:, cols].sum(-1)
                            attn[pname][f"{b}.{h}"][grp] += mass.tolist()
                            attn[pname][f"{b}.{h}"][grp + "_per_tok"] += (mass / len(cols)).tolist()
                    attn[pname][f"{b}.{h}"]["set"] = name
            if pname == "contract":
                ld["clean"] += ld_of(out.logits).tolist()
                scale = m.adapter.final_norm_scale(m.model, fin["x"]).cpu()                        # [B, 1], pre-norm
                for b, h in gate + movers:
                    cols = torch.tensor(head_cols([h], m.head_dim))
                    w_o = m.adapter.get_attn_head_output_module(m.model, b).weight.detach().float().cpu()
                    z = sink[b][:, -1].float().cpu()
                    write = z[:, cols] @ w_o[:, cols].T                                            # [B, H]
                    dla[f"{b}.{h}"] += (scale[:, 0] * (write @ direction)).tolist()
        # ---- B: knockouts on the contract prompt ----
        ids, mask, extra = batch([(image(k), contract_q) for k in ks])
        for name, hset in [("ko:gate5", gate)] + [(f"ko:random{i}", r) for i, r in enumerate(randoms)]:
            spec = defaultdict(list)
            for b, h in hset:
                spec[b].append(h)
            hs = register_mean_ablation(m.adapter, m.model, dict(spec), m.head_dim, {"mask": mask})
            try:
                out = m.forward(ids, mask, extra)
            finally:
                for h in hs:
                    h.remove()
            ld[name] += ld_of(out.logits).tolist()
        # ---- C: swaps ----
        gate_spec = defaultdict(list)
        for b, h in gate:
            gate_spec[b].append(h)
        gate_spec = dict(gate_spec)
        base_ids, base_mask, base_extra = ids, mask, extra
        clean_text = m.decode(m.generate(base_ids, base_mask, base_extra, args.max_new_tokens))
        donor_sets = {"self": [(image(k), contract_q) for k in ks],
                      "other_item": [(image(donors[k]), contract_q) for k in ks],
                      "blank": [(image(k, "blank"), contract_q) for k in ks],
                      "degraded": [(image(k, "degraded"), contract_q) for k in ks]}
        donor_text = {}
        for dname, prompts in donor_sets.items():
            d_ids, d_mask, d_extra = batch(prompts)
            sink = {}
            hs = capture_ohp_input(m.adapter, m.model, sorted(gate_spec), sink)
            try:
                donor_text[dname] = m.decode(m.generate(d_ids, d_mask, d_extra, args.max_new_tokens))
            finally:
                for h in hs:
                    h.remove()
            hs = install_values(m.adapter, m.model, gate_spec, m.head_dim, sink)
            try:
                texts = m.decode(m.generate(base_ids, base_mask, base_extra, args.max_new_tokens))
            finally:
                for h in hs:
                    h.remove()
            for i, k in enumerate(ks):
                dk = donors[k] if dname == "other_item" else k
                swaps.append({"item": k, "donor": dname, "donor_item": dk, "clean": clean_text[i],
                              "donor_own": donor_text[dname][i], "patched": texts[i],
                              "outcome": classify_swap(texts[i], k, dk, items),
                              "clean_outcome": classify_swap(clean_text[i], k, dk, items)})
        print(f"  {s + len(ks)}/{len(keys)}", flush=True)

    def mean(v):
        return sum(v) / len(v) if v else None
    attn_summary = {p: {h: {"set": d["set"], **{g2: mean(v) for g2, v in d.items() if g2 != "set"}}
                        for h, d in hd.items()} for p, hd in attn.items()}
    swap_summary = defaultdict(lambda: defaultdict(int))
    for r in swaps:
        if r["clean_outcome"] == "known_base":                       # eligible: clean answer is known+right
            swap_summary[r["donor"]][r["outcome"]] += 1
            swap_summary[r["donor"]]["n"] += 1
            swap_summary[r["donor"]]["donor_own_known"] += int(norm(r["donor_own"]).startswith("known"))
    swap_summary = {d: {k: (v / c["n"] if k != "n" else v) for k, v in c.items()} for d, c in swap_summary.items()}
    report = {"config": vars(args), "attention": attn_summary,
              "head_dla_known_minus_unknown": {h: mean(v) for h, v in dla.items()},
              "ld_known_minus_unknown": {a: mean(v) for a, v in ld.items()},
              "swaps": swap_summary, "random_sets": [[f"{b}.{h}" for b, h in r] for r in randoms]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "swaps.jsonl"), "w") as f:
        for r in swaps:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nA. attention from the last prompt column (total mass; per-token in brackets), contract prompt")
    hdr = "  head    set     " + " ".join(f"{g2[:10]:>16}" for g2 in GROUPS)
    print(hdr)
    for h, d in attn_summary["contract"].items():
        print(f"  {h:<7} {d['set']:<7} " + " ".join(
            f"{(d.get(g2) or 0):6.3f} [{(d.get(g2 + '_per_tok') or 0):6.4f}]" for g2 in GROUPS))
    print("\n   same, fact prompt")
    for h, d in attn_summary["fact"].items():
        if d["set"] == "gate5":
            print(f"  {h:<7} {d['set']:<7} " + " ".join(
                f"{(d.get(g2) or 0):6.3f} [{(d.get(g2 + '_per_tok') or 0):6.4f}]" for g2 in GROUPS))
    print("\nB. logit(known) - logit(unknown) at the answer position, contract prompt (mean over items)")
    for a, v in report["ld_known_minus_unknown"].items():
        print(f"  {a:<12} {v:+.3f}")
    print("   direct write of each head on known-minus-unknown (exact DLA term; + = toward 'known')")
    for h, v in report["head_dla_known_minus_unknown"].items():
        print(f"  {h:<7} {'gate5' if parse_heads(h)[0] in gate else 'mover':<6} {v:+.4f}")
    print("\nC. gate-head value swaps on the contract prompt (rows whose clean answer is known + correct)")
    for d, c in swap_summary.items():
        print(f"  donor={d:<11} n={int(c['n']):>3}  known_base {c.get('known_base', 0):6.1%}  "
              f"known_donor {c.get('known_donor', 0):6.1%}  known_other {c.get('known_other', 0):6.1%}  "
              f"unknown {c.get('unknown', 0):6.1%}   (donor's own answer known: {c.get('donor_own_known', 0):6.1%})")
    print(f"\nwrote {out_dir}/summary.json and swaps.jsonl")


if __name__ == "__main__":
    main()
