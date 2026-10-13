"""E1: are the entity's FACTS already present at the image tokens, or only its identity?

"Move-then-recall" claims the image tokens carry WHICH COUNTRY this is, the blocks
21-23 movers carry that identity to the answer position, and the facts are looked up
there by the last-token MLPs 23-26. Venhoff et al. (NeurIPS 2025) report the opposite
pattern for Qwen2.5-VL-7B: like its text backbone, it uses early MLPs at the ENTITY
positions -- the LLM "subject enrichment" site (Geva et al. 2023), where facts are
attached to the subject before anything reads them. If the image tokens are enriched,
the movers could be carrying facts along with the identity, and the headline changes.

Three measurements off ONE forward pass per (flag, question), no intervention:

  identity  (E0b, folded in) image-position activations must be identical across the
            four questions -- the image precedes the question under causal attention.
            Measured in the batched capture, and bit-for-bit at batch size 1 on
            --identity_items flags. Turns R13.3's argument into a measurement.

  vocab     logit lens at the image tokens, layers 1-27, per region (object, ring,
            background, all; pooled mean, and max-over-tokens): the rank of the
            country's NAME token among all 84 names, against the rank of each
            attribute VALUE's first content token among that attribute's values. The
            last-token rank of the queried value is printed alongside as the reference
            curve -- it is where the answer is known to appear.

  probe     leave-one-COUNTRY-out linear probes for attribute values that REPEAT across
            countries (language; EUR vs not; calling-code world zone = first digit).
            A held-out country's value can only be predicted from features SHARED with
            other countries of that value, so identity alone cannot do it. Read every
            curve against its LAYER-0 value (the vision features as they enter the LM):
            flags of the same language share colours and motifs, so visual similarity
            alone predicts some of it. ENRICHMENT shows up as image-token accuracy rising
            ABOVE layer 0 with depth. The last token is probed twice: under the question
            asking that attribute, and under the capital question (does the answer
            position hold the value even when it was not asked?).

READING IT
  name rank high, value ranks at chance, image probes flat at their layer-0 level
      -> identity, not facts: move-then-recall stands.
  value ranks / image probes rising with depth well above layer 0
      -> facts are attached at the image tokens (Venhoff's early site). The R16.2 freeze
         (and E6's) then decides whether that copy is USED or only present.

Usage
-----
    python methods/image_enrichment.py --dry_run
    python methods/image_enrichment.py                      # capture + vocab + probe
    python methods/image_enrichment.py --phases probe       # re-analyse a saved capture, CPU only
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import ATTRIBUTES, DEFAULT_VADE_ROOT, MODEL_ID, load_flags, write_json  # noqa: E402

REGIONS = ("object", "ring", "background", "all")
PHASES = ("capture", "probe")


def region_index(vade_root, n_image_tokens):
    """-> {region: [image-relative token indices]} from object_location.json."""
    loc = json.load(open(os.path.join(vade_root, "data", "flags", "object_location.json")))
    obj = sorted(loc["object_token_indices"]["flag_only"]["flat"])
    ring1 = sorted(loc["object_token_indices"]["flag_ring1"]["flat"])
    assert loc["vlm_token_grid"]["n_tokens"] == n_image_tokens, "geometry does not match the model's image grid"
    ring = sorted(set(ring1) - set(obj))
    bg = sorted(set(range(n_image_tokens)) - set(ring1))
    return {"object": obj, "ring": ring, "background": bg, "all": list(range(n_image_tokens))}


def calling_zone(code):
    return str(code).strip().lstrip("+")[:1]


PROBE_TARGETS = {
    "language": lambda it: it["language"],
    "currency_eur": lambda it: "EUR" if it["currency"] == "EUR" else "other",
    "calling_zone": lambda it: calling_zone(it["calling_code"]),
}
# Which question the last-token features come from, per target.
TARGET_QUESTION = {"language": "language", "currency_eur": "currency", "calling_zone": "calling_code"}


def candidates(tokenizer, items):
    """-> (names {country: [ids]}, values {attr: {country: id}}) -- first tokens of the
    name (with/without a leading space) and the first CONTENT token of each gold."""
    from methods.circuit_common import first_content_token, gold_ids
    names = {c: sorted({tokenizer(" " + it["name"], add_special_tokens=False)["input_ids"][0],
                        tokenizer(it["name"], add_special_tokens=False)["input_ids"][0]})
             for c, it in items.items()}
    values = {a: {c: first_content_token(tokenizer, gold_ids(tokenizer, it[a])) for c, it in items.items()}
              for a in ATTRIBUTES}
    return names, values


def rank_of(logit_by_tok, own, cands):
    """1 + #distinct candidate tokens scoring strictly above the best of `own`."""
    lo = max(logit_by_tok[t] for t in own)
    return 1 + sum(1 for t in cands if logit_by_tok[t] > lo and t not in own)


def name_rank(logit_by_tok, own, names):
    """Rank of country `own` among COUNTRIES, each scored by its best spelling
    (' France' / 'France') -- so two variants per name do not double the field."""
    score = {c: max(logit_by_tok[t] for t in ids) for c, ids in names.items()}
    return 1 + sum(1 for c, v in score.items() if c != own and v > score[own])


def collisions(names, values):
    """names {country: [ids]}, values {attr: {country: (first token, label)}} ->
    (countries whose name shares a first token with another country's name,
     {attr: countries whose value's first token also stands for a DIFFERENT value}).

    A shared token makes the rank measure "this family" (' United' for the US and the
    UK; ' San' for San Marino and San Jose), and ties count as rank 1, so these rows
    would inflate top-1. They are excluded from the rank statistics and counted."""
    from collections import defaultdict
    owner = defaultdict(set)
    for c, ids in names.items():
        for t in ids:
            owner[t].add(c)
    name_coll = {c for c, ids in names.items() if any(len(owner[t]) > 1 for t in ids)}
    value_coll = {}
    for a, by_c in values.items():                  # by_c: {country: (first token, label) or None}
        labels = defaultdict(set)
        for tl in by_c.values():
            if tl is not None:
                labels[tl[0]].add(tl[1])
        # Countries SHARING a value (two Spanish-speaking flags) legitimately share its token;
        # only a token standing for two DIFFERENT values is a collision.
        value_coll[a] = {c for c, tl in by_c.items() if tl is not None and len(labels[tl[0]]) > 1}
    return name_coll, value_coll


# ---------------------------------------------------------------------------
# capture (GPU)
# ---------------------------------------------------------------------------

def capture(args, items):
    import numpy as np
    import torch
    from methods.circuit_common import Model, cat_extra
    m = Model(args.model_id, args.device, attn_impl="sdpa", vade_root=args.vade_root)
    entity_dir = os.path.join(args.vade_root, "data", "flags")
    countries = sorted(items)
    names, values = candidates(m.tokenizer, items)
    name_coll, value_coll = collisions(names, {a: {c: (t, str(items[c][a])) if t is not None else None
                                                   for c, t in v.items()} for a, v in values.items()})
    print(f"  excluded from ranks (shared first token): names {len(name_coll)}, values "
          + ", ".join(f"{a} {len(v)}" for a, v in value_coll.items()))
    name_cands = sorted({t for ids in names.values() for t in ids})
    value_cands = {a: sorted({t for t in v.values() if t is not None}) for a, v in values.items()}
    all_cands = sorted(set(name_cands) | {t for v in value_cands.values() for t in v})
    D = m.adapter.logit_direction(m.model, torch.tensor(all_cands, device=m.model.device))   # [n_cand, H]
    col = {t: i for i, t in enumerate(all_cands)}
    n_layers, H = m.n_layers, m.hidden
    layers = list(range(n_layers))                  # hidden_states[n_layers] is post-norm: excluded

    pools = np.zeros((len(countries), n_layers, len(REGIONS), H), dtype=np.float16)
    last = np.zeros((len(countries), len(ATTRIBUTES), n_layers, H), dtype=np.float16)
    vocab = []                                      # one record per (country, layer, region, mode)
    last_rank = []                                  # (country, attribute, layer, rank of own value)
    identity_batched = 0.0
    regions = None

    def lens_logits(h):
        """h [..., H] residual (pre-final-norm) -> [..., n_cand] candidate logits."""
        h = h.float()
        return m.adapter.final_norm_scale(m.model, h) * (h @ D.T)

    per_batch = max(1, args.batch_size // len(ATTRIBUTES))
    for i in range(0, len(countries), per_batch):
        chunk = countries[i:i + per_batch]
        ids, extras = [], []
        for c in chunk:
            for a in ATTRIBUTES:
                x, e = m.image_prompt(entity_dir, items[c], a)
                ids.append(x); extras.append(e)
        ids = torch.stack(ids)
        out = m.forward(ids, torch.ones_like(ids), cat_extra(extras), output_hidden_states=True)
        hs = out.hidden_states
        img_cols = (ids[0] == m.image_id).nonzero().flatten().to(m.model.device)
        if regions is None:
            regions = region_index(args.vade_root, len(img_cols))
        for ci, c in enumerate(chunk):
            r0 = ci * len(ATTRIBUTES)
            ki = countries.index(c)
            for L in layers:
                img = hs[L][r0:r0 + len(ATTRIBUTES), img_cols]                         # [4, n_img, H]
                identity_batched = max(identity_batched, float((img - img[:1]).abs().max()))
                x = img[0]
                for ri, reg in enumerate(REGIONS):
                    sel = x[regions[reg]]
                    pooled = sel.mean(0)
                    pools[ki, L, ri] = pooled.float().cpu().numpy().astype(np.float16)
                    if L == 0:
                        continue                    # the lens is meaningless on raw embeddings
                    for mode, lg in (("pooled", lens_logits(pooled)), ("max_token", lens_logits(sel).max(0).values)):
                        lg = lg.cpu().tolist()
                        by_tok = {t: lg[col[t]] for t in all_cands}
                        rec = {"country": c, "layer": L, "region": reg, "mode": mode}
                        if c not in name_coll:
                            rec["name_rank"] = name_rank(by_tok, c, names)
                        for a in ATTRIBUTES:
                            if values[a][c] is not None and c not in value_coll[a]:
                                rec[f"{a}_rank"] = rank_of(by_tok, [values[a][c]], value_cands[a])
                        vocab.append(rec)
                for ai, a in enumerate(ATTRIBUTES):
                    v = hs[L][r0 + ai, -1]
                    last[ki, ai, L] = v.float().cpu().numpy().astype(np.float16)
                    if L > 0 and values[a][c] is not None and c not in value_coll[a]:
                        lg = lens_logits(v).cpu().tolist()
                        by_tok = {t: lg[col[t]] for t in all_cands}
                        last_rank.append({"country": c, "attribute": a, "layer": L,
                                          "rank": rank_of(by_tok, [values[a][c]], value_cands[a])})
        del out, hs
        print(f"  capture {min(i + per_batch, len(countries))}/{len(countries)}", flush=True)

    identity_solo = None
    if args.identity_items:
        identity_solo = 0.0
        for c in countries[:args.identity_items]:
            ref = None
            for a in ATTRIBUTES:
                x, e = m.image_prompt(entity_dir, items[c], a)
                x = x[None]
                hs = m.forward(x, torch.ones_like(x), e, output_hidden_states=True).hidden_states
                img = torch.stack([hs[L][0, x[0] == m.image_id] for L in layers])
                if ref is None:
                    ref = img
                else:
                    identity_solo = max(identity_solo, float((img - ref).abs().max()))
        print(f"  identity (batch 1, {args.identity_items} flags x 4 questions): max |dh| = {identity_solo:.3e}")
    print(f"  identity (batched capture): max |dh| at image tokens across questions = {identity_batched:.3e}")

    np.savez_compressed(os.path.join(args.out_dir, "capture.npz"), pools=pools, last=last,
                        countries=np.array(countries), attributes=np.array(ATTRIBUTES), regions=np.array(REGIONS))
    return {"identity_batched_max_abs": identity_batched, "identity_batch1_max_abs": identity_solo,
            "n_name_candidates": len(names), "n_value_candidates": {a: len(v) for a, v in value_cands.items()},
            "rank_exclusions": {"name": sorted(name_coll), **{a: sorted(v) for a, v in value_coll.items()}},
            "regions": {k: len(v) for k, v in regions.items()}}, vocab, last_rank


def summarize_vocab(vocab, last_rank, n_names, n_values):
    out = {}
    for rec in vocab:
        key = f"{rec['region']}/{rec['mode']}"
        slot = out.setdefault(key, {}).setdefault(rec["layer"], {})
        for k, v in rec.items():
            if k.endswith("_rank"):
                slot.setdefault(k, []).append(v)
    for key, by_layer in out.items():
        for L, slot in by_layer.items():
            by_layer[L] = {k: {"mean_rank": sum(v) / len(v), "top1": sum(x == 1 for x in v) / len(v)}
                           for k, v in slot.items()}
    ref = {}
    for r in last_rank:
        ref.setdefault(r["attribute"], {}).setdefault(r["layer"], []).append(r["rank"])
    ref = {a: {L: {"mean_rank": sum(v) / len(v), "top1": sum(x == 1 for x in v) / len(v)} for L, v in d.items()}
           for a, d in ref.items()}
    chance = {"name_rank": (n_names + 1) / 2, **{f"{a}_rank": (n + 1) / 2 for a, n in n_values.items()}}
    return {"image": out, "last_token_queried_value": ref, "chance_mean_rank": chance}


def print_vocab(s, layers=(4, 8, 12, 16, 18, 20, 22, 24, 27)):
    print("\nVOCAB (logit lens, top-1 rate): NAME vs attribute VALUE at the image tokens; last token for reference")
    print("  candidates (chance top-1 ~ 1/n): " + ", ".join(
        f"{k[:-5]}={2 * v - 1:.0f}" for k, v in s["chance_mean_rank"].items())
          + "\n  rows whose first token is shared with a different name/value are excluded (counts in "
            "summary.json rank_exclusions); that removes calling_code entirely -- every code starts with a "
            "digit another code shares -- so the probe is its only test here")
    for key in ("object/max_token", "all/max_token", "background/pooled"):
        tab = s["image"].get(key, {})
        print(f"  [{key}]  L: " + " ".join(f"{L:>6}" for L in layers))
        for k in ("name_rank",) + tuple(f"{a}_rank" for a in ATTRIBUTES):
            print(f"    {k:<18}" + " ".join(f"{tab.get(L, {}).get(k, {}).get('top1', float('nan')):>6.1%}"
                                             for L in layers))
    print("  [last token, queried value]")
    for a in ATTRIBUTES:
        tab = s["last_token_queried_value"].get(a, {})
        print(f"    {a + '_rank':<18}" + " ".join(f"{tab.get(L, {}).get('top1', float('nan')):>6.1%}"
                                                   for L in layers))


# ---------------------------------------------------------------------------
# probe (CPU)
# ---------------------------------------------------------------------------

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def loo_accuracy(X, y, balanced, Cs=C_GRID):
    """max over C of the leave-one-out accuracy of a standardized L2 logistic regression.

    Per-feature standardization makes the probe blind to raw scale (a huge but
    uninformative dimension cannot swamp it); C is chosen as the best of a small grid,
    and `probe`'s permutation null takes the SAME max, so the selection is paid for.
    Exact speed-up: an L2-regularized linear model's optimal weights lie in the span of
    its training rows, so each fold is rotated onto that span (<= n-1 dims instead of
    3584) before fitting -- identical predictions, ~40x cheaper. LOO accuracy is biased
    BELOW chance under the null on small balanced data; read it against the
    permutation null, never against 1/n_classes."""
    import warnings
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    preds = {C: np.empty_like(y) for C in Cs}
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Z, z = (X[tr] - mu) / sd, (X[i:i + 1] - mu) / sd
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        Zr, zr = Z @ Vt.T, z @ Vt.T
        for C in Cs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                preds[C][i] = LogisticRegression(C=C, max_iter=3000).fit(Zr, y[tr]).predict(zr)[0]
    score = (lambda p: balanced_accuracy_score(y, p)) if balanced else (lambda p: (p == y).mean())
    return float(max(score(p) for p in preds.values()))


def probe(args, items):
    import numpy as np
    from collections import Counter
    from joblib import Parallel, delayed
    d = np.load(os.path.join(args.out_dir, "capture.npz"))
    countries = [str(c) for c in d["countries"]]
    pools, last = d["pools"].astype(np.float32), d["last"].astype(np.float32)
    n_layers = pools.shape[1]
    layers = [L for L in args.probe_layers if L < n_layers]
    rng = np.random.default_rng(args.seed)
    out = {}
    for target, fn in PROBE_TARGETS.items():
        labels = np.array([fn(items[c]) for c in countries])
        counts = Counter(labels)
        keep = np.array([counts[v] >= 2 for v in labels])
        y = labels[keep]
        balanced = target == "currency_eur"
        maj = 0.5 if balanced else max(Counter(y).values()) / len(y)
        families = {f"image:{r}": pools[keep][:, :, ri] for ri, r in enumerate(REGIONS)}
        qa = ATTRIBUTES.index(TARGET_QUESTION[target])
        families["last_token:asked"] = last[keep][:, qa]
        families["last_token:capital_q"] = last[keep][:, ATTRIBUTES.index("capital")]
        res = {"n": int(keep.sum()), "classes": len(set(y)), "baseline": maj, "balanced": balanced, "curves": {}}
        for fam, F in families.items():
            vals = Parallel(n_jobs=args.n_jobs)(delayed(loo_accuracy)(F[:, L], y, balanced) for L in layers)
            res["curves"][fam] = dict(zip(layers, vals))
            curve = res["curves"][fam]
            best = max(curve, key=curve.get)
            nulls = {}
            for L in sorted({layers[0], best}):
                perm = Parallel(n_jobs=args.n_jobs)(delayed(loo_accuracy)(F[:, L], rng.permutation(y), balanced)
                                                    for _ in range(args.n_perm))
                nulls[L] = {"p": (1 + sum(p >= curve[L] for p in perm)) / (1 + len(perm)),
                            "null_mean": float(np.mean(perm))}
            res.setdefault("perm", {})[fam] = {"best_layer": best, **{str(k): v for k, v in nulls.items()}}
            print(f"  {target:<13} {fam:<22} L0={curve[layers[0]]:.2f} best=L{best}:{curve[best]:.2f} "
                  f"(base {maj:.2f}, p@best={nulls[best]['p']:.3f})", flush=True)
        out[target] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--phases", nargs="+", default=list(PHASES), choices=list(PHASES))
    ap.add_argument("--batch_size", type=int, default=16, help="Rows per forward (4 questions per flag).")
    ap.add_argument("--identity_items", type=int, default=4, help="Flags for the batch-1 bit-identity check.")
    ap.add_argument("--probe_layers", type=int, nargs="+", default=list(range(0, 28, 2)) + [27],
                    help="Layers to probe (CPU cost is linear in this). Default every other layer.")
    ap.add_argument("--n_perm", type=int, default=50,
                    help="Label permutations for the null, at layer 0 and at each curve's best layer.")
    ap.add_argument("--n_jobs", type=int, default=-1, help="CPU workers for the probes (joblib).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "image_enrichment", "flags"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    entity_dir, items, gt_sha = load_flags(args.vade_root)
    print(f"[image_enrichment] {len(items)} flags x {len(ATTRIBUTES)} questions; phases={args.phases}")
    for t, fn in PROBE_TARGETS.items():
        from collections import Counter
        c = Counter(fn(it) for it in items.values())
        print(f"  probe target {t:<13} {sum(v for v in c.values() if v >= 2)} countries in "
              f"{sum(1 for v in c.values() if v >= 2)} classes with >=2 members")
    if args.dry_run:
        print("Metadata valid.")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "summary.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    report["ground_truth_sha256"] = gt_sha
    if "capture" in args.phases:
        meta, vocab, last_rank = capture(args, items)
        report["capture"] = meta
        report["vocab"] = summarize_vocab(vocab, last_rank, meta["n_name_candidates"], meta["n_value_candidates"])
        with open(os.path.join(args.out_dir, "vocab_rows.jsonl"), "w") as fh:
            for r in vocab:
                fh.write(json.dumps(r) + "\n")
        print_vocab(report["vocab"])
        write_json(path, report)
    if "probe" in args.phases:
        report["probe"] = probe(args, items)
        write_json(path, report)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
