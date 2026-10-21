"""X5: is the ICLR 2027 "entity-recognition gate" (Qwen2.5-VL-7B blocks 16-17) recognition,
or an answer-side signal (abstention / question routing)?

"Localizing and Disrupting Entity Identification in Vision-Language Models" (anonymous ICLR
2027 submission, OpenReview 2M35KkgY5g) mean-ablates attention -- every head's o_proj input,
at every input position, replaced by its mean over the sequence -- and finds a two-layer band,
0-based layers 16-17, whose knockout turns `known,<name>` identity answers into `unknown`
(~90% refusals, <=8% substitutions) while perceptual control answers survive. Five heads carry
most of it: 16.1, 16.14, 16.17, 16.20, 17.21. Their own limitation: the knockout "does not
distinguish disruption of identity-relevant visual processing, retrieval, confidence, or
answer production", and it is never applied with the entity NAMED in text.

On our flags data blocks 16-18 are exactly where the QUESTION starts reaching the answer
position (ATTRIBUTE_SWITCH_FINDINGS.md: question swap 98.8 / 78.6 / 28.6% at blocks 16/17/18),
while identity at the image tokens is not built there (P7) and the movers read it later
(21-23). This script applies their exact intervention to our setup and asks which reading fits.

THE INTERVENTION (theirs). A forward pre-hook on o_proj of each ablated block replaces the
selected heads' columns, at every position, by their mean over the (unpadded) sequence. It acts
on the prompt forward pass only: a decode step has sequence length 1, where the mean is the
value itself, so later tokens attend to the cached keys/values of the ablated prompt -- as in
the paper. Left padding is excluded from the mean.

PROBES (Qwen controlled prompts, prefill 'Answer:')
    image/fact:<attr>     our four flags attributes ("Report the capital of this country.")
    image/identify        their output contract adapted to flags: `known,<country>` / `unknown,<n>`
    image/identify_open   "Which country does this flag belong to?" -- no `unknown` option
    image/color           perceptual control: the flag's dominant colour (no gold; % changed vs clean)
    text/fact:<attr>      the same fact questions with the country NAMED and no image

ARMS
    clean
    ko:<a>-<b>            every head in blocks a..b (default windows 16-17, 12-13, 19-20, 24-25)
    ko:gate5              the paper's five heads
    ko:gate5:random<i>    --n_random five-head draws from blocks 16-17 (same per-block counts,
                          never a gate head)

OUTCOME of every probe row (eligible rows = clean answer correct; the color probe is scored
as changed / unchanged against the clean answer instead):
    correct        still right
    refusal        "unknown" / "not sure" / "cannot ..."     -> abstention
    other_entity   another country's value for the asked attribute, or another country's name
    wrong_attr     this country's value for a DIFFERENT attribute (or a wrong-format answer
                   such as a currency code under a language question)  -> routing
    other          none of the above

READING IT (image/fact and text/fact together)
    image drops, text survives, failures = other_entity     -> visual recognition
    image AND text drop, failures = refusal                 -> answer-side abstention signal
    failures = wrong_attr                                   -> the question is not routed
Plus the movers' direct write of this flag's name (rank among all names, as in vrh_ablation):
if it survives the knockout, the gate acts downstream of / beside the entity payload.

Usage
-----
    python methods/gate_knockout.py --dry_run
    python methods/gate_knockout.py --limit_items 8 --n_random 1 --windows 16-17   # smoke
    python methods/gate_knockout.py                                                # full
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import DEFAULT_VADE_ROOT, MODEL_ID, MOVERS, PREFILL, load_entity  # noqa: E402
from methods.attribute_switch_sweep import ATTRIBUTES, question  # noqa: E402
from methods.common.text_match import contains_label  # noqa: E402

GATE5 = "16.1,16.14,16.17,16.20,17.21"
DEFAULT_WINDOWS = ["16-17", "12-13", "19-20", "24-25"]

IDENTIFY_CONTRACT = (
    'Return exactly: "<status>,<value>"\n'
    'Rules:\n'
    '- <status> must be either "known" or "unknown".\n'
    '- Do you know which country this flag belongs to? Answer "known" if you recognise it, '
    'and "unknown" otherwise.\n'
    'Output:\n'
    '- If status is "known", <value> must be the country name in lowercase (e.g. "france").\n'
    '- If status is "unknown", <value> must be the number of distinct visible elements in the '
    'image as digits only.\n'
    'Examples: known,france / known,japan / known,brazil / unknown,4\n'
    'Do not explain your reasoning. Output only the answer.')
IDENTIFY_OPEN = "Which country does this flag belong to? Give only the country name."
COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
COLOR_PROBE = ("Analyze the image and identify the dominant color of the flag. Return exactly one "
               "lowercase color name from this list: " + ", ".join(COLORS) + ". Output only the color name.")

REFUSAL = re.compile(r"\bunknown\b|not sure|cannot|can't|can not|unable|don't know|do not know|"
                     r"not (?:able|possible)|no idea|unclear|unrecogni", re.I)


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------

def parse_heads(spec):
    out = []
    for tok in spec.replace(" ", "").split(","):
        if tok:
            b, h = tok.split(".")
            out.append((int(b), int(h)))
    return sorted(out)


def parse_window(spec):
    a, b = (int(x) for x in spec.split("-"))
    assert 0 <= a <= b, f"bad window {spec!r}"
    return list(range(a, b + 1))


def block_matched_random(heads, n_heads, rng):
    """Same number of heads per block as `heads`, never one of them."""
    banned = set(heads)
    per = Counter(b for b, _ in heads)
    out = []
    for b, c in sorted(per.items()):
        free = [h for h in range(n_heads) if (b, h) not in banned]
        out += [(b, h) for h in rng.sample(free, c)]
    return sorted(out)


def build_arms(windows, gate, n_random, n_heads, seed):
    """-> [(name, {block: [heads] or None})]; None = every head of the block."""
    arms = [("clean", {})]
    for w in windows:
        arms.append((f"ko:{w}", {b: None for b in parse_window(w)}))
    by_block = defaultdict(list)
    for b, h in gate:
        by_block[b].append(h)
    arms.append(("ko:gate5", dict(by_block)))
    rng = random.Random(seed)
    for i in range(n_random):
        r = defaultdict(list)
        for b, h in block_matched_random(gate, n_heads, rng):
            r[b].append(h)
        arms.append((f"ko:gate5:random{i}", dict(r)))
    return arms


def mean_ablate(t, cols, mask):
    """t [B, T, W]: columns `cols` replaced at every position by their mean over the valid
    (mask == 1) positions of each row. T == 1 (a cached decode step) is returned unchanged,
    since the mean of one position is itself. `cols` None = every column."""
    import torch
    if t.shape[1] == 1:
        return t
    out = t.clone()
    sel = out if cols is None else out[..., cols]
    if mask is None or mask.shape[1] != t.shape[1]:
        m = torch.ones(t.shape[:2], dtype=t.dtype, device=t.device)
    else:
        m = mask.to(device=t.device, dtype=t.dtype)
    mean = (sel * m[..., None]).sum(1, keepdim=True) / m.sum(1, keepdim=True)[..., None].clamp_min(1)
    filled = mean.expand_as(sel)
    if cols is None:
        return filled.clone()
    out[..., cols] = filled
    return out


def head_cols(heads, head_dim):
    return [c for h in heads for c in range(h * head_dim, (h + 1) * head_dim)]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def classify_fact(text, item_key, attribute, items, attributes):
    """-> outcome for a fact answer (see module docstring)."""
    gold = str(items[item_key][attribute])
    if contains_label(text, gold):
        return "correct"
    if REFUSAL.search(text):
        return "refusal"
    if any(contains_label(text, str(items[item_key][a])) for a in attributes if a != attribute
           and norm(str(items[item_key][a])) != norm(gold)):
        return "wrong_attr"
    for k, it in items.items():
        v = str(it.get(attribute, ""))
        if k != item_key and v and norm(v) != norm(gold) and contains_label(text, v):
            return "other_entity"
    # A value of another attribute from ANY country (e.g. a currency code under a language
    # question): the question was answered in the wrong format.
    for it in items.values():
        for a in attributes:
            v = str(it.get(a, ""))
            if a != attribute and len(norm(v)) >= 3 and contains_label(text, v):
                return "wrong_attr"
    return "other"


def name_hit(text, name):
    a, b = norm(text), norm(name)
    return bool(a) and bool(b) and (b in a or a in b)


def classify_identify(text, item_key, items, contract):
    """-> outcome for an identity answer. The contract parse follows the paper: lowercase,
    strip, split on the first comma; anything not starting with `known` is unknown."""
    t = text.strip().strip('"').strip()
    if contract:
        status, _, value = t.partition(",")
        if not norm(status).startswith("known"):
            return "refusal"
        t = value
    elif REFUSAL.search(t):
        return "refusal"
    if name_hit(t, items[item_key]["name"]):
        return "correct"
    if any(k != item_key and name_hit(t, it["name"]) for k, it in items.items()):
        return "other_entity"
    return "other"


def color_of(text):
    for w in norm(text).split():
        if w in COLORS:
            return w
    return norm(text)[:20]


def summarize(rows):
    """rows -> {probe: {arm: stats}}. Eligibility is per (item, probe, template) and comes
    from the CLEAN arm; the color probe reports % changed vs clean on every item."""
    clean = {(r["probe"], r["item"]): r for r in rows if r["arm"] == "clean"}
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe"], r["arm"])].append(r)
    for (probe, arm), rs in sorted(groups.items()):
        if probe == "image/color":
            pairs = [(r, clean.get((probe, r["item"]))) for r in rs]
            pairs = [(r, c) for r, c in pairs if c is not None]
            changed = sum(color_of(r["generated"]) != color_of(c["generated"]) for r, c in pairs)
            out[probe][arm] = {"n": len(pairs), "changed": changed / len(pairs) if pairs else None}
            continue
        elig = [r for r in rs if clean.get((probe, r["item"]), {}).get("outcome") == "correct"]
        cnt = Counter(r["outcome"] for r in elig)
        n = len(elig)
        ranks = [r["mover_name_rank"] for r in elig if r.get("mover_name_rank") is not None]
        out[probe][arm] = {"n": n, **{k: cnt.get(k, 0) / n if n else None for k in
                                      ("correct", "refusal", "other_entity", "wrong_attr", "other")},
                           "loss": (1 - cnt.get("correct", 0) / n) if n else None,
                           "mover_rank_mean": sum(ranks) / len(ranks) if ranks else None}
    return dict(out)


def pooled(summary, prefix):
    """Mean over the probes starting with `prefix` (e.g. 'image/fact'), weighted by n."""
    acc = defaultdict(lambda: defaultdict(float))
    ns = defaultdict(int)
    for probe, arms in summary.items():
        if not probe.startswith(prefix):
            continue
        for arm, s in arms.items():
            if not s.get("n"):
                continue
            ns[arm] += s["n"]
            for k in ("correct", "refusal", "other_entity", "wrong_attr", "other"):
                acc[arm][k] += s[k] * s["n"]
    return {arm: {"n": ns[arm], **{k: v / ns[arm] for k, v in acc[arm].items()}} for arm in ns}


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

def register_mean_ablation(adapter, model, spec, head_dim, mask_ref):
    """spec {block: [heads] or None}; mask_ref a dict whose 'mask' holds the batch's attention
    mask while generating. Pre-hooks on o_proj (registered FIRST, so captures read the result)."""
    import torch
    handles = []
    for b, heads in sorted(spec.items()):
        cols = None if heads is None else torch.tensor(head_cols(heads, head_dim), dtype=torch.long)

        def pre(_m, args, _cols=cols):
            c = None if _cols is None else _cols.to(args[0].device)
            return (mean_ablate(args[0], c, mask_ref.get("mask")),) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(pre))
    return handles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS)
    ap.add_argument("--gate", default=GATE5)
    ap.add_argument("--movers", default=MOVERS)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    ap.add_argument("--probes", nargs="+", default=["image/fact", "image/identify", "image/identify_open",
                                                    "image/color", "text/fact"])
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "gate_knockout", "flags"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    entity_dir, items, gt_sha = load_entity(args.vade_root, "flags")
    keys = sorted(items)[:args.limit_items] if args.limit_items else sorted(items)
    gate = parse_heads(args.gate)
    movers = parse_heads(args.movers)
    arms = build_arms(args.windows, gate, args.n_random, 28, args.seed)

    jobs = []                       # (probe, item, attribute or None)
    for p in args.probes:
        if p.endswith("/fact"):
            jobs += [(f"{p}:{a}", k, a) for a in args.attributes for k in keys]
        else:
            jobs += [(p, k, None) for k in keys]
    print(f"[gate_knockout] {len(keys)} flags, {len(jobs)} generations per arm, {len(arms)} arms -> {args.out_dir}")
    for name, spec in arms:
        desc = " ".join(f"{b}.*" if hs is None else " ".join(f"{b}.{h}" for h in hs) for b, hs in sorted(spec.items()))
        print(f"  {name:<22} {desc}")
    if args.dry_run:
        print("Grid valid.")
        return

    import torch
    from methods.circuit_common import Model, left_pack
    from methods.vrh_ablation import mover_write, register_mover_capture

    m = Model(args.model_id, device=args.device, attn_impl=args.attn_impl, vade_root=args.vade_root)
    assert m.n_heads == 28, f"random draws assumed 28 heads; model has {m.n_heads}"
    tok = m.tokenizer

    # Movers' direct write of each flag's name: first token of ' <name>' and '<name>'.
    name_toks = {k: sorted({tok(" " + items[k]["name"], add_special_tokens=False)["input_ids"][0],
                            tok(items[k]["name"], add_special_tokens=False)["input_ids"][0]}) for k in items}
    all_ids = sorted({i for v in name_toks.values() for i in v})
    with torch.no_grad():
        name_dirs = m.adapter.logit_direction(m.model, torch.tensor(all_ids, device=m.model.device)).float().cpu()
    col = {t: i for i, t in enumerate(all_ids)}

    def build(probe, k, a):
        if probe.startswith("text/"):
            return m.text_prompt(items[k]["name"], a)[0], {}
        qtext = {"image/identify": IDENTIFY_CONTRACT, "image/identify_open": IDENTIFY_OPEN,
                 "image/color": COLOR_PROBE}.get(probe) or question(a)
        from PIL import Image
        path = os.path.join(entity_dir, items[k]["image"])
        if path not in m._images:
            with Image.open(path) as im:
                m._images[path] = im.convert("RGB")
        x = m.adapter.build_inputs(m.processor, m._images[path], qtext, PREFILL)
        return x["input_ids"], x["extra"]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    by_probe = defaultdict(list)
    for j in jobs:
        by_probe[j[0]].append(j)
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as fh:
        for arm, spec in arms:
            for probe, pj in by_probe.items():
                for s in range(0, len(pj), args.batch_size):
                    chunk = pj[s:s + args.batch_size]
                    built = [build(p, k, a) for p, k, a in chunk]
                    ids, mask = left_pack([b[0] for b in built], m.pad_id)
                    extra = ({k2: torch.cat([b[1][k2] for b in built]) for k2 in built[0][1]}
                             if built[0][1] else {})
                    mask_ref = {"mask": mask}
                    sink = {}
                    handles = register_mean_ablation(m.adapter, m.model, spec, m.head_dim, mask_ref)
                    if probe.startswith("image/fact") or probe.startswith("text/fact"):
                        handles += register_mover_capture(m.adapter, m.model, movers, sink)
                    try:
                        gen = m.generate(ids, mask, extra, args.max_new_tokens)
                    finally:
                        for h in handles:
                            h.remove()
                    texts = m.decode(gen)
                    scores = (mover_write(m.adapter, m.model, sink, movers, m.head_dim) @ name_dirs.T
                              if sink else None)
                    for i, ((p, k, a), text) in enumerate(zip(chunk, texts)):
                        if p.startswith("image/identify"):
                            outcome = classify_identify(text, k, items, contract=(p == "image/identify"))
                        elif p == "image/color":
                            outcome = None
                        else:
                            outcome = classify_fact(text, k, a, items, args.attributes)
                        rank = None
                        if scores is not None:
                            own = max(scores[i, col[t]].item() for t in name_toks[k])
                            per_item = [max(scores[i, col[t]].item() for t in name_toks[c]) for c in items]
                            rank = 1 + sum(v > own for v in per_item)
                        r = {"arm": arm, "probe": p, "item": k, "attribute": a, "generated": text,
                             "outcome": outcome, "mover_name_rank": rank}
                        rows.append(r)
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {arm}: done", flush=True)

    summary = summarize(rows)
    report = {"config": vars(args), "ground_truth_sha256": gt_sha, "n_items": len(keys), "summary": summary,
              "pooled": {"image/fact": pooled(summary, "image/fact"), "text/fact": pooled(summary, "text/fact")}}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    def line(name, s):
        return (f"  {name:<22} n={s['n']:>4}  correct {s['correct']:6.1%}  refusal {s['refusal']:6.1%}  "
                f"other_entity {s['other_entity']:6.1%}  wrong_attr {s['wrong_attr']:6.1%}  other {s['other']:6.1%}")
    for title in ("image/fact", "text/fact"):
        print(f"\n{title} (pooled over attributes; eligible = clean correct)")
        for arm, s in report["pooled"][title].items():
            print(line(arm, s))
    for probe in ("image/identify", "image/identify_open"):
        if probe in summary:
            print(f"\n{probe}")
            for arm, s in summary[probe].items():
                if s["n"]:
                    print(line(arm, s))
    if "image/color" in summary:
        print("\nimage/color (perceptual control; % answers changed vs clean)")
        for arm, s in summary["image/color"].items():
            print(f"  {arm:<22} {s['changed']:6.1%}")
    print(f"\nwrote {args.out_dir}/rows.jsonl and summary.json")


if __name__ == "__main__":
    main()
