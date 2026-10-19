"""E4: does the circuit COMPOSE? Build an answer that no single run produced.

The account so far has two independent routes into the answer position:

  routers  ~8 heads, blocks 19-21, carry WHICH ATTRIBUTE was asked   (R1, R4)
  movers   ~10 heads, blocks 21-23, carry WHICH ENTITY is shown       (R8-R11, R16.1)

followed by the last-token MLPs 23-26 doing the lookup (R16.2). If that is the whole
mechanism, the two routes are separable inputs to the lookup, and we can feed them
from two DIFFERENT runs:

    base           image C, asked X        -> clean answer C.X
    entity donor   image B, asked X        -> movers' values carry "B"
    question donor image C, asked A        -> routers' values carry "asked A"

    compose = base + movers<-entity donor + routers<-question donor   => B.A ?

B.A is an answer that neither donor and not the base ever produced. The movers come
from a run that was asked X, the routers from a run that saw C. Only the circuit's
composition can produce it.

ARMS (all continuous: the selected heads are installed at the readout column of
EVERY step, so multi-token answers can complete -- see head_swap_vade.py)

  clean               base, unpatched                                   -> C.X
  movers              movers <- entity donor                            -> B.X
  routers             routers <- question donor                         -> C.A
  compose             both, from the two donors                         -> B.A
  compose_random      size-matched random heads from the same blocks    -> C.X
  target_circuit      movers + routers <- the TARGET run (image B, asked A)
  target_window       EVERY head of the circuit's blocks <- target      (the ceiling)
  target_complement   every head of those blocks EXCEPT the circuit <- target

`compose` vs `target_circuit` separates "the circuit carries enough" from "values from
a single consistent run carry enough". target_window vs target_complement is the
necessity reading: if the complement transfers nearly as well as the window, the
named heads are not where the effect lives.

DONOR PREFIX POLICY. Generation is full-prefix recomputation (no KV cache), and at
step t every donor is run on its OWN prompt followed by the RECIPIENT's generated
tokens so far (head_followup_common's policy). That keeps donors on-manifold for the
answer being produced, which step-matched replay of a donor's own rollout is not when
the donor answered a different question (R11.2b).

SCORING. VADE's own matcher (whole word, casefold, diacritics) against all four golds;
each generation is labelled C.X / B.X / C.A / B.A / other / ambiguous. Rows where two
of the four labels coincide (two countries sharing a language or currency) are kept
but reported separately: B.A cannot be told apart from its twin there.

Usage
-----
    python methods/circuit_compose.py --dry_run
    python methods/circuit_compose.py --n_items 4 --out_dir results/circuit_compose/smoke
    python methods/circuit_compose.py                        # 32 countries x 12 attribute pairs
    python methods/circuit_compose.py --entity brands --movers <traced> --router_trace <brands trace>

OTHER ENTITIES (--entity brands|celebrities). Their own controlled prompts
(attribute_switch_sweep.ENTITY_SPECS). Scoring is already VADE's casefold matcher, so stored
lowercase labels need no own-answer golds here. --movers and --router_trace (or --routers)
have no default off flags: the flags heads are not a claim about another entity. Rows whose
four golds include an item with no label for that attribute (brands: AIRBUS/industry) are
skipped.
"""
import argparse
import itertools
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import (ALL_ATTRIBUTES, DEFAULT_VADE_ROOT, ENTITIES, MODEL_ID,  # noqa: E402
                                    MOVERS, ROUTER_TRACE, entity_attributes, load_entity, parse_heads,
                                    router_heads, write_json)

LABELS = ("C.X", "B.X", "C.A", "B.A")
ARMS = ("clean", "movers", "routers", "compose", "compose_random",
        "target_circuit", "target_window", "target_complement")


def build_rows(items, attributes, n_items, seed):
    rng = random.Random(seed)
    names = sorted(items)
    chosen = rng.sample(names, min(n_items, len(names)))
    rows = []
    for c in chosen:
        b = rng.choice([x for x in names if x != c])
        for x, a in itertools.permutations(attributes, 2):
            if any(k not in items[i] for i in (c, b) for k in (x, a)):
                continue                       # brands: AIRBUS has no industry label
            golds = {"C.X": items[c][x], "B.X": items[b][x], "C.A": items[c][a], "B.A": items[b][a]}
            rows.append({"row_index": len(rows), "C": c, "B": b, "X": x, "A": a,
                         "golds": {k: str(v) for k, v in golds.items()}})
    return rows


def distinct_golds(row):
    from methods.common.text_match import normalize
    return len({normalize(v) for v in row["golds"].values()}) == 4


def classify(text, row):
    from methods.common.text_match import contains_label, normalize
    hits = [k for k in LABELS if contains_label(text, row["golds"][k])]
    if not hits:
        return "other"
    return hits[0] if len({normalize(row["golds"][h]) for h in hits}) == 1 else "ambiguous"


def random_sets(movers, routers, n_heads, seed):
    """Size-matched nulls drawn from the SAME blocks as each real set, excluding both sets."""
    rng = random.Random(seed + 1)
    used = set(movers) | set(routers)

    def draw(real):
        pool = [(b, h) for b in sorted({b for b, _ in real}) for h in range(n_heads) if (b, h) not in used]
        return sorted(rng.sample(pool, len(real)))
    return draw(movers), draw(routers)


def arm_plan(arm, movers, routers, rnd_movers, rnd_routers, window, complement):
    """-> [(donor_name, heads)] installs for an arm. Donor names: E, Q, T."""
    return {
        "clean": [],
        "movers": [("E", movers)],
        "routers": [("Q", routers)],
        "compose": [("E", movers), ("Q", routers)],
        "compose_random": [("E", rnd_movers), ("Q", rnd_routers)],
        "target_circuit": [("T", sorted(movers + routers))],
        "target_window": [("T", window)],
        "target_complement": [("T", complement)],
    }[arm]


def capture_o_proj_inputs(m, ids, extra, blocks, start):
    """One full-prefix forward -> {block: [B, T-start, hidden]} of o_proj's input
    (attn_head_output) at columns start.."""
    import torch
    sinks, handles = {}, []
    for b in blocks:
        mod = m.adapter.get_attn_head_output_module(m.model, b)
        handles.append(mod.register_forward_pre_hook(
            lambda _m, inp, _b=b: sinks.__setitem__(_b, inp[0][:, start:].detach().clone())))
    try:
        m.forward(ids, torch.ones_like(ids), extra)
    finally:
        for h in handles:
            h.remove()
    assert set(sinks) == set(blocks)
    return sinks


def patched_logits(m, ids, extra, installs, start, observe=None):
    """Recipient forward with installs = [(heads, {block: donor z})], each donor z
    [B, T-start, hidden]. Heads are replaced at columns start.. (continuous mode).
    `observe` (optional dict) receives o_proj inputs AFTER the patch, for read-back."""
    import torch
    from methods.head_followup_common import grouped_heads, replace_heads
    per_block = {}
    for heads, z in installs:
        for b, hs in grouped_heads(heads).items():
            per_block.setdefault(b, []).append((hs, z[b]))
    handles = []
    for b, parts in per_block.items():
        def pre(_m, inp, _parts=parts):
            t = inp[0]
            for hs, donor in _parts:
                t = replace_heads(t, donor.to(t.device), hs, m.head_dim, start, "continuous")
            return (t,) + tuple(inp[1:])
        mod = m.adapter.get_attn_head_output_module(m.model, b)
        handles.append(mod.register_forward_pre_hook(pre))
        if observe is not None:
            handles.append(mod.register_forward_pre_hook(
                lambda _m, inp, _b=b: observe.__setitem__(_b, inp[0][:, start:].detach().clone())))
    try:
        return m.forward(ids, torch.ones_like(ids), extra).logits[:, -1]
    finally:
        for h in handles:
            h.remove()


def run_batch(m, args, rows, prompts, plans, check=False):
    """-> {arm: [texts]}, {ref: [texts]} for one batch."""
    import torch
    P = prompts["base"][0].shape[1]
    start = P - 1
    eos = m.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}

    refs = {}
    for name in ("base", "E", "Q", "T"):
        ids, extra = prompts[name]
        refs[name] = m.decode(m.generate(ids, torch.ones_like(ids), extra, args.max_new_tokens))

    out = {}
    for arm, plan in plans.items():
        gen = torch.zeros((len(rows), 0), dtype=torch.long)
        donor_blocks = {}
        for donor, heads in plan:
            donor_blocks.setdefault(donor, set()).update(b for b, _ in heads)
        for step in range(args.max_new_tokens):
            installs = []
            for donor, heads in plan:
                ids_d, x_d = prompts[donor]
                z = capture_o_proj_inputs(m, torch.cat([ids_d, gen], 1), x_d, sorted(donor_blocks[donor]), start)
                installs.append((heads, z))
            ids_r, x_r = prompts["base"]
            observe = {} if (check and step == 0 and installs) else None
            logits = patched_logits(m, torch.cat([ids_r, gen], 1), x_r, installs, start, observe)
            if observe is not None:
                worst = 0.0
                for heads, z in installs:
                    for b, h in heads:
                        sl = slice(h * m.head_dim, (h + 1) * m.head_dim)
                        worst = max(worst, float((observe[b][:, :, sl].float() - z[b][:, :, sl].float()).abs().max()))
                print(f"    [check] {arm}: read-back worst |installed - read| = {worst:.2e}")
                assert worst < 1e-2, f"{arm}: patch and capture address different tensors"
            gen = torch.cat([gen, logits.argmax(-1, keepdim=True).cpu()], 1)
        texts = []
        for row_toks in gen.tolist():
            cut = next((i for i, t in enumerate(row_toks) if t in eos), len(row_toks))
            texts.append(m.tokenizer.decode(row_toks[:cut], skip_special_tokens=True))
        out[arm] = texts
    if check:
        # Self-install: the base's OWN values into every head of the window must be a no-op.
        ids_r, x_r = prompts["base"]
        window = next(heads for arm, plan in plans.items() if arm == "target_window" for _, heads in plan)
        own = capture_o_proj_inputs(m, ids_r, x_r, sorted({b for b, _ in window}), start)
        a = patched_logits(m, ids_r, x_r, [(window, own)], start)
        b = m.forward(ids_r, torch.ones_like(ids_r), x_r).logits[:, -1]
        err = float((a.float() - b.float()).abs().max())
        print(f"    [check] self-install of {len(window)} heads vs clean: max |dlogit| = {err:.2e}")
        assert err < 1e-2, "installing a run's own head values changed its logits"
    return out, refs


def build_prompts(m, entity_dir, items, rows):
    import torch
    from methods.circuit_common import cat_extra
    spec = {"base": ("C", "X"), "E": ("B", "X"), "Q": ("C", "A"), "T": ("B", "A")}
    prompts = {}
    for name, (item_key, attr_key) in spec.items():
        ids, extras = [], []
        for r in rows:
            i, x = m.image_prompt(entity_dir, items[r[item_key]], r[attr_key])
            ids.append(i); extras.append(x)
        assert len({len(i) for i in ids}) == 1
        prompts[name] = (torch.stack(ids), cat_extra(extras))
    assert len({p[0].shape[1] for p in prompts.values()}) == 1, "controlled prompts must share one length"
    return prompts


def summarize(records):
    out = {}
    for subset, keep in (("all", lambda r: True), ("eligible", lambda r: r["eligible"]),
                         ("eligible_distinct", lambda r: r["eligible"] and r["distinct"])):
        rr = [r for r in records if keep(r)]
        out[subset] = {"n": len(rr)}
        for arm in ARMS:
            labs = [r["labels"][arm] for r in rr]
            if labs:
                out[subset][arm] = {k: labs.count(k) / len(labs) for k in LABELS + ("other", "ambiguous")}
    return out


def resolve_entity_args(ap, args):
    """Per-entity defaults. Flags resolves to exactly the old defaults (so its summary config is
    unchanged); other entities must name their own movers and routers."""
    if args.entity == "flags":
        args.movers = args.movers or MOVERS
        args.router_trace = args.router_trace or ROUTER_TRACE
    elif not args.movers or not (args.routers or args.router_trace):
        ap.error(f"--entity {args.entity} needs --movers and --router_trace (or --routers) from its own "
                 f"traces; the defaults are the FLAGS heads")
    args.attributes = args.attributes or entity_attributes(args.entity)
    if not set(args.attributes) <= set(entity_attributes(args.entity)):
        ap.error(f"{args.entity} attributes are {entity_attributes(args.entity)}")
    args.out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "circuit_compose", args.entity)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--entity", default="flags", choices=ENTITIES)
    ap.add_argument("--attributes", nargs="+", default=None, choices=ALL_ATTRIBUTES,
                    help="Default: every attribute of --entity.")
    ap.add_argument("--n_items", type=int, default=32, help="Base countries C; each gets one random partner B.")
    ap.add_argument("--movers", default=None, help="Default (flags only): common10.")
    ap.add_argument("--routers", default=None, help="Explicit BLOCK.HEAD list; default = top --router_k of "
                                                    "--router_trace, skipping mover heads.")
    ap.add_argument("--router_trace", default=None, help="Default (flags only): R1's attr_head_trace JSON.")
    ap.add_argument("--router_k", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None, help="Default: results/circuit_compose/<entity>.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    resolve_entity_args(ap, args)

    movers = parse_heads(args.movers)
    routers = parse_heads(args.routers) if args.routers else router_heads(args.router_trace, args.router_k, movers)
    assert not set(movers) & set(routers), "a head cannot be in both sets: it has one value per forward"
    entity_dir, items, gt_sha = load_entity(args.vade_root, args.entity)
    rows = build_rows(items, args.attributes, args.n_items, args.seed)
    n_heads = 28
    rnd_m, rnd_r = random_sets(movers, routers, n_heads, args.seed)
    blocks = range(min(b for b, _ in movers + routers), max(b for b, _ in movers + routers) + 1)
    window = [(b, h) for b in blocks for h in range(n_heads)]
    complement = [x for x in window if x not in set(movers) | set(routers)]
    plans = {a: arm_plan(a, movers, routers, rnd_m, rnd_r, window, complement) for a in args.arms}
    if "target_window" not in plans:
        plans["target_window"] = arm_plan("target_window", movers, routers, rnd_m, rnd_r, window, complement)

    fmt = lambda hs: ", ".join(f"{b}.{h}" for b, h in hs)  # noqa: E731
    print(f"[circuit_compose] {args.entity}: {len(rows)} rows ({args.n_items} items x "
          f"{len(args.attributes) * (len(args.attributes) - 1)} attribute pairs)")
    print(f"  movers  ({len(movers)}): {fmt(movers)}")
    print(f"  routers ({len(routers)}): {fmt(routers)}")
    print(f"  random movers / routers: {fmt(rnd_m)} / {fmt(rnd_r)}")
    print(f"  window: blocks {blocks[0]}-{blocks[-1]} ({len(window)} heads), complement {len(complement)}")
    n_fwd = sum(1 + len(p) for a, p in plans.items() if a != "clean") * args.max_new_tokens
    print(f"  ~{n_fwd} forwards + 4 generate calls per batch of {args.batch_size}")
    if args.dry_run:
        print("Grid valid.")
        return

    from methods.circuit_common import Model
    m = Model(args.model_id, args.device, attn_impl="sdpa", vade_root=args.vade_root)
    assert m.n_heads == n_heads, f"model has {m.n_heads} heads per block, script assumed {n_heads}"
    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as fh:
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            prompts = build_prompts(m, entity_dir, items, chunk)
            texts, refs = run_batch(m, args, chunk, prompts, plans, check=(i == 0))
            for j, r in enumerate(chunk):
                ref_ok = {"base": "C.X", "E": "B.X", "Q": "C.A", "T": "B.A"}
                ref_labels = {k: classify(refs[k][j], r) for k in ref_ok}
                rec = {**r, "distinct": distinct_golds(r),
                       "eligible": all(ref_labels[k] == v for k, v in ref_ok.items()),
                       "refs": {k: refs[k][j] for k in ref_ok}, "ref_labels": ref_labels,
                       "texts": {a: texts[a][j] for a in plans},
                       "labels": {a: classify(texts[a][j], r) for a in plans}}
                fh.write(json.dumps(rec) + "\n")
                records.append(rec)
            print(f"  {len(records)}/{len(rows)}", flush=True)

    summary = summarize(records)
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": {k: v for k, v in vars(args).items()
                           if k != "dry_run" and not (k == "entity" and v == "flags")},
                "movers": movers, "routers": routers, "random_movers": rnd_m, "random_routers": rnd_r,
                "ground_truth_sha256": gt_sha, **summary})
    s = summary["eligible_distinct"]
    print(f"\nELIGIBLE rows with four distinct golds: n={s['n']} "
          f"(all four reference runs give their own answer)")
    print(f"  {'arm':<20}" + "".join(f"{k:>9}" for k in LABELS + ("other",)))
    expect = {"clean": "C.X", "movers": "B.X", "routers": "C.A", "compose": "B.A", "compose_random": "C.X",
              "target_circuit": "B.A", "target_window": "B.A", "target_complement": "C.X"}
    for arm in ARMS:
        if arm in s:
            print(f"  {arm:<20}" + "".join(f"{s[arm][k]:>8.1%} " for k in LABELS + ("other",))
                  + f"   (circuit predicts {expect[arm]})")
    print(f"\nwrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
