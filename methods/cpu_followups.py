"""CPU-only re-analyses of result files already on disk (no model, no torch, no numpy).

Each analysis answers one question from CLAIM_CANDIDATES.md using per-row outputs that
earlier GPU runs already wrote. Nothing here regenerates text or touches activations;
every number is recomputed from JSONL/JSON/log files plus VADE's ground truth and
tuples, with VADE score.py's own matcher (`methods/common/text_match.py`).

ANALYSES (`--analyses`, default all)
    entity_swap      G1. For every VADE-format predictions file: cause, and how often each
                     NON-target attribute flips to the SOURCE's value. index = mean
                     non-target source rate / cause. An attribute edit has index << 1; an
                     entity swap has index ~ 1 (everything moves together).
    error_geography  A8. For answers that match neither base nor source, map the emitted
                     value back to the country it belongs to, and ask whether that third
                     country sits in the source's (or base's) UN M49 subregion / continent
                     more often than a uniformly drawn other value would. Pooled over the
                     head installs, MLP readout edits, image-site interventions, E4
                     composition, R11 head_cross and E9's zero:common10.
    name_rank_null   A5. The entity-name rank in head_vocab_projection is the BEST of two
                     spellings among n candidates, so its null is (n+1)/3 and its z-null is
                     the expected z of a max-of-2, not (n+1)/2 and 0. Simulates that null and
                     reports every head set against it.
    per_token        A9. head_vocab_projection's key-group split, normalized by group size:
                     attention and direct entity write per OBJECT token vs per BACKGROUND
                     token, per head.
    compose_failures B9. E4's composed 'other' outputs classified: B's answer to the base
                     question X (right entity, base format), B's value of a third attribute,
                     C's value, another country's A value, unmapped -- split by whether X and
                     A share an answer format.
    currency_unique  E2. Is currency's extra dependence on the direct image route (E5) and
                     on Park's heads (E9) present on rows whose base AND source currencies
                     are unique? route_knockout row pairs are rebuilt from its seed.
    survivors        E3(b). Under zero:common10 (E9), which answers survive: are surviving
                     language answers concentrated on languages shared by many countries
                     (a prior / visual-similarity guess) rather than on rare ones?
    cross_bootstrap  A7. head_cross n64 rates with pair-level vs country-cluster bootstrap CIs
                     (only 44 bases / 45 sources back the 64 pairs).
    proxy_ceiling    G2. Per layer, Phase B's best proxy cause vs the full-swap residual
                     ceiling at flag_ring1 / full_image (Spearman), and each winner's layer.
    late_facts       D7. Logit lens at image tokens using POOLED region readouts (size-fair),
                     object vs background, at layers 18-27.
    lasttoken_probe  D8. The last-token leave-one-country-out probe curves at every layer,
                     not only the best one.

USAGE
    python methods/cpu_followups.py                       # all analyses
    python methods/cpu_followups.py --analyses error_geography entity_swap
Outputs: results/cpu_followups/<analysis>.json and results/cpu_followups/REPORT.md.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from methods.common.text_match import contains_label, normalize  # noqa: E402

DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(REPO_ROOT), "VADE"))
OUT_DIR = os.path.join(REPO_ROOT, "results", "cpu_followups")
ATTRIBUTES = ("capital", "currency", "language", "calling_code")

# UN M49 subregions for the 84 flags countries (ISO 3166-1 alpha-2). Continent = the M49
# region the subregion belongs to.
M49 = {
    "Africa": {
        "Northern Africa": "DZ EG LY MA SD TN",
        "Eastern Africa": "ET KE MU RW TZ UG ZM",
        "Middle Africa": "AO CM",
        "Southern Africa": "NA",
        "Western Africa": "CV GH LR SN",
    },
    "Americas": {
        "Caribbean": "CU HT",
        "Central America": "BZ GT HN MX NI PA",
        "South America": "AR BR CL CO GY PE PY UY",
    },
    "Asia": {
        "Central Asia": "UZ",
        "Eastern Asia": "JP KP KR",
        "South-eastern Asia": "KH LA SG TH VN",
        "Southern Asia": "AF BD IN NP PK",
        "Western Asia": "AM AZ BH CY IQ JO KW LB OM QA SA YE",
    },
    "Europe": {
        "Eastern Europe": "BY HU RO UA",
        "Northern Europe": "DK FI IE IS SE",
        "Southern Europe": "AL ES GR IT MK PT RS",
        "Western Europe": "BE DE LU",
    },
    "Oceania": {
        "Australia and New Zealand": "AU NZ",
        "Melanesia": "FJ",
    },
}
SUBREGION, CONTINENT = {}, {}
for _cont, _subs in M49.items():
    for _sub, _codes in _subs.items():
        for _c in _codes.split():
            SUBREGION[_c], CONTINENT[_c] = _sub, _cont

FORMAT = {"capital": "word", "language": "word", "currency": "code", "calling_code": "number"}


# ----------------------------------------------------------------------------- io

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    return path


def rel(path):
    return os.path.relpath(path, REPO_ROOT)


def load_ground_truth(vade_root, entity="flags"):
    return json.load(open(os.path.join(vade_root, "data", entity, "ground_truth.json")))["countries"]


def load_tuples(vade_root, entity="flags"):
    """{(target_attribute, row_index): row} over test and train (test wins)."""
    out = {}
    for split in ("train", "test"):
        for path in glob.glob(os.path.join(vade_root, "data", entity, "tuples", "*", f"{split}.jsonl")):
            attr = os.path.basename(os.path.dirname(path))
            if attr == "identity":
                continue
            for r in read_jsonl(path):
                out[(attr, r["row_index"])] = r
    return out


# ----------------------------------------------------------------------------- stats

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def poisson_binomial_z(hits, probs):
    """Observed hits vs a sum of independent Bernoulli(p_i): (expected, z)."""
    mu = sum(probs)
    var = sum(p * (1 - p) for p in probs)
    return mu, ((hits - mu) / math.sqrt(var) if var > 0 else None)


def same_label(a, b):
    return normalize(str(a)) == normalize(str(b))


# ----------------------------------------------------------------------------- prediction files

def prediction_files():
    """(family, name, path) for every VADE-format flags predictions file on disk."""
    out = []
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "methods", "interventions", "flags_*_predictions.jsonl"))):
        out.append(("image_site_dictionary", os.path.basename(p).replace("_predictions.jsonl", ""), p))
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "results", "Qwen2.5-VL-7B-Instruct", "flags", "dbm",
                                           "language", "*", "layer*_predictions_test.jsonl"))):
        cfg = os.path.basename(os.path.dirname(p))
        out.append(("image_site_dbm(Z1-suspect)", f"dbm/{cfg}/{os.path.basename(p).split('_')[0]}", p))
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "results", "head_swap_vade", "flags", "*.jsonl"))):
        arm = os.path.basename(p)[:-6]
        fam = "clean" if arm == "clean" else ("full_image_patch" if arm == "full_image" else
                                              ("random_heads" if arm.startswith("random") else "head_install"))
        out.append((fam, f"head_swap_vade/{arm}", p))
    for sub, site in (("flags", "heads"), ("flags__mlp_hidden_b23-26", "mlp23-26")):
        for p in sorted(glob.glob(os.path.join(REPO_ROOT, "results", "readout_jacobian", sub, "*.jsonl"))):
            arm = os.path.basename(p)[:-6]
            fam = {"clean": "clean", "random": "random_subspace", "full": f"full_swap@{site}"}.get(arm, f"readout_edit@{site}")
            out.append((fam, f"readout_jacobian[{site}]/{arm}", p))
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "results", "head_das", "flags", "*", "predictions.jsonl"))):
        cfg = os.path.basename(os.path.dirname(p))
        fam = "trained_mask@mlp23-26(dense,see E3)" if "_dbm_" in cfg else "trained_das@heads"
        out.append((fam, f"head_das/{cfg}", p))
    return out


def joined_rows(path, tuples):
    """Prediction rows joined to their tuples rows; (rows, n_missing)."""
    rows, missing = [], 0
    for p in read_jsonl(path):
        t = tuples.get((p["attribute"], p["row_index"]))
        if t is None:
            missing += 1
            continue
        rows.append((t, p.get("generated_text", "")))
    return rows, missing


# ----------------------------------------------------------------------------- G1 entity swap

def analysis_entity_swap(ctx):
    tuples = ctx["tuples"]
    results = []
    for fam, name, path in prediction_files():
        rows, missing = joined_rows(path, tuples)
        by_target = defaultdict(list)
        for t, g in rows:
            by_target[t["target_attribute"]].append((t, g))
        for target, rr in sorted(by_target.items()):
            cause_rows = [(t, g) for t, g in rr if t["queried"] == target]
            if not cause_rows:
                continue
            cause = sum(contains_label(g, t["source_label"]) for t, g in cause_rows) / len(cause_rows)
            per_q = {}
            for q in ATTRIBUTES:
                if q == target:
                    continue
                qr = [(t, g) for t, g in rr if t["queried"] == q and not same_label(t["base_label"], t["source_label"])]
                if not qr:
                    continue
                per_q[q] = {"n": len(qr),
                            "matches_source": sum(contains_label(g, t["source_label"]) for t, g in qr) / len(qr),
                            "matches_base": sum(contains_label(g, t["base_label"]) for t, g in qr) / len(qr)}
            leak = st.mean(v["matches_source"] for v in per_q.values()) if per_q else None
            iso = st.mean(v["matches_base"] for v in per_q.values()) if per_q else None
            neither = (1 - leak - iso) if per_q else None
            results.append({
                "family": fam, "name": name, "target": target, "n_cause": len(cause_rows),
                "cause": cause, "iso_mean": iso, "nontarget_source_mean": leak, "nontarget_neither": neither,
                "entity_swap_index": (leak / cause) if (leak is not None and cause >= 0.05) else None,
                "per_queried": per_q, "missing_rows": missing, "file": rel(path)})
    # Family-level summary: median index over files x targets with cause >= 5%.
    fam_summary = {}
    for fam in dict.fromkeys(r["family"] for r in results):
        idx = [r["entity_swap_index"] for r in results if r["family"] == fam and r["entity_swap_index"] is not None]
        cs = [r["cause"] for r in results if r["family"] == fam]
        fam_summary[fam] = {"n_cells": len(cs), "n_cells_cause_ge_5pct": len(idx),
                            "median_index": st.median(idx) if idx else None,
                            "min_index": min(idx) if idx else None, "max_index": max(idx) if idx else None,
                            "median_cause": st.median(cs) if cs else None}
    out = {"note": "index = mean over non-target attributes of P(answer matches SOURCE) / cause. "
                   "Non-target rows whose base and source labels are equal are excluded. Index "
                   "only reported where cause >= 5%. Rows are scored as VADE score.py does, on "
                   "every row the file contains (no clean-correct filter).",
           "families": fam_summary, "cells": results}
    write_json("entity_swap", out)
    return out


# ----------------------------------------------------------------------------- A8 error geography

class ValueIndex:
    """attribute -> {normalized value: set(countries)} over the flags ground truth."""

    def __init__(self, gt):
        self.gt = gt
        self.by_attr = {a: defaultdict(set) for a in ATTRIBUTES}
        for c, it in gt.items():
            for a in ATTRIBUTES:
                if it.get(a):
                    self.by_attr[a][normalize(str(it[a]))].add(c)
        self.raw = {a: {normalize(str(it[a])): str(it[a]) for c, it in gt.items() if it.get(a)} for a in ATTRIBUTES}

    def countries_in_text(self, attr, text, exclude_values=()):
        ex = {normalize(str(v)) for v in exclude_values}
        hit = set()
        for nv, cs in self.by_attr[attr].items():
            if nv in ex:
                continue
            if contains_label(text, self.raw[attr][nv]):
                hit |= cs
        return hit

    def null_prob(self, attr, ref, exclude_values, region_of):
        """P(a uniformly drawn OTHER value of `attr` belongs to a country in ref's region)."""
        ex = {normalize(str(v)) for v in exclude_values}
        vals = [cs for nv, cs in self.by_attr[attr].items() if nv not in ex]
        if not vals:
            return None
        return sum(any(region_of[c] == region_of[ref] for c in cs) for cs in vals) / len(vals)


def geo_record(vi, attr, text, ref, other_ref, exclude_values, source_tag, cluster):
    """One 'other' answer -> per-region hit flags and null probabilities (or unmapped).

    `placebo` is a country drawn (deterministically per cluster) from everything except
    ref/other_ref: if the null is calibrated, its observed rate matches its null."""
    cs = vi.countries_in_text(attr, text, exclude_values)
    rec = {"source": source_tag, "attribute": attr, "mapped": bool(cs), "cluster": cluster}
    if not cs:
        return rec
    pool = sorted(c for c in vi.gt if c not in (ref, other_ref))
    placebo = random.Random(str(cluster)).choice(pool)
    for level, region in (("subregion", SUBREGION), ("continent", CONTINENT)):
        rec[f"{level}_hit_placebo"] = any(region[c] == region[placebo] for c in cs)
        rec[f"{level}_null_placebo"] = vi.null_prob(attr, placebo, exclude_values, region)
        rec[f"{level}_hit_ref"] = any(region[c] == region[ref] for c in cs)
        rec[f"{level}_null_ref"] = vi.null_prob(attr, ref, exclude_values, region)
        if other_ref:
            rec[f"{level}_hit_other"] = any(region[c] == region[other_ref] for c in cs)
            rec[f"{level}_null_other"] = vi.null_prob(attr, other_ref, exclude_values, region)
    if attr == "calling_code":
        zone = str(vi.gt[ref]["calling_code"])[0]
        vals = [nv for nv in vi.by_attr["calling_code"] if nv not in {normalize(str(v)) for v in exclude_values}]
        rec["itu_zone_hit_ref"] = any(str(vi.gt[c]["calling_code"])[0] == zone for c in cs)
        rec["itu_zone_null_ref"] = sum(v[0] == zone for v in vals) / len(vals)
    return rec


def dedupe_clusters(recs):
    """One record per cluster (file x base x source x queried): templates and the per-target
    copies head_swap_vade writes are not independent observations. Keeps the first MAPPED
    record of a cluster if any, else its first record."""
    best = {}
    for r in recs:
        k = r["cluster"]
        if k not in best or (r["mapped"] and not best[k]["mapped"]):
            best[k] = r
    return list(best.values())


def summarize_geo(recs):
    recs = dedupe_clusters(recs)
    mapped = [r for r in recs if r["mapped"]]
    out = {"n_other": len(recs), "n_mapped": len(mapped),
           "mapped_frac": len(mapped) / len(recs) if recs else None}
    keys = [("subregion", "ref"), ("continent", "ref"), ("subregion", "other"), ("continent", "other"),
            ("subregion", "placebo"), ("continent", "placebo")]
    for level, who in keys:
        rr = [r for r in mapped if f"{level}_hit_{who}" in r and r[f"{level}_null_{who}"] is not None]
        if not rr:
            continue
        hits = sum(r[f"{level}_hit_{who}"] for r in rr)
        mu, z = poisson_binomial_z(hits, [r[f"{level}_null_{who}"] for r in rr])
        out[f"{level}_{who}"] = {"n": len(rr), "observed": hits / len(rr), "null": mu / len(rr),
                                 "ratio": (hits / mu) if mu else None, "z": z}
    zz = [r for r in mapped if "itu_zone_hit_ref" in r]
    if zz:
        hits = sum(r["itu_zone_hit_ref"] for r in zz)
        mu, z = poisson_binomial_z(hits, [r["itu_zone_null_ref"] for r in zz])
        out["itu_zone_ref"] = {"n": len(zz), "observed": hits / len(zz), "null": mu / len(zz), "z": z}
    return out


def analysis_error_geography(ctx):
    gt, tuples = ctx["gt"], ctx["tuples"]
    vi = ValueIndex(gt)
    groups = defaultdict(list)
    # VADE-format files: 'other' = matches neither label. ref = source (what was installed),
    # other_ref = base (what the rest of the prompt still carries).
    for fam, name, path in prediction_files():
        if fam in ("clean",) or fam.startswith("random"):
            continue
        rows, _ = joined_rows(path, tuples)
        for t, g in rows:
            if contains_label(g, t["source_label"]) or contains_label(g, t["base_label"]):
                continue
            q = t["queried"]
            ex = (gt[t["base"]][q], gt[t["source"]][q])
            groups[fam].append(geo_record(vi, q, g, t["source"], t["base"], ex, name,
                                          (name, t["base"], t["source"], q)))
    # E4 composition: composed answer should be B.A; 'other' answers checked against B (ref)
    # and C (the base image, other_ref).
    cc = os.path.join(REPO_ROOT, "results", "circuit_compose", "flags", "rows.jsonl")
    if os.path.isfile(cc):
        for r in read_jsonl(cc):
            if not r.get("eligible") or r["labels"].get("compose") != "other":
                continue
            A, B, C = r["A"], r["B"], r["C"]
            ex = (gt[B][A], gt[C][A])
            groups["E4_compose"].append(geo_record(vi, A, r["texts"]["compose"], B, C, ex, "circuit_compose",
                                                   ("E4", B, C, A, r["X"])))
    # R11 head_cross: label 'other' in cells where the donor flag differs from the base.
    hc = os.path.join(REPO_ROOT, "results", "head_cross", "flags", "cross_n64.jsonl")
    if os.path.isfile(hc):
        for r in read_jsonl(hc):
            if r["label"] != "other" or r["donor_flag"] == r["base"]:
                continue
            q = r["q1"]
            ex = (gt[r["base"]][q], gt[r["donor_flag"]][q])
            groups["R11_head_cross"].append(geo_record(vi, q, r["generated"], r["donor_flag"], r["base"], ex, "head_cross",
                                                       ("R11", r["base"], r["donor_flag"], q, r["q2"], r["cell"])))
    # E9 zero:common10: wrong answers about a single image. ref = the item itself.
    vr = os.path.join(REPO_ROOT, "results", "vrh_ablation", "flags", "rows.jsonl")
    if os.path.isfile(vr):
        for r in read_jsonl(vr):
            if r["arm"] != "zero:common10" or r["correct"]:
                continue
            q = r["attribute"]
            groups["E9_zero_common10(ref=item)"].append(
                geo_record(vi, q, r["generated_text"], r["item"], None, (gt[r["item"]][q],), "vrh_ablation",
                           ("E9", r["item"], q)))
    summary = {g: summarize_geo(rs) for g, rs in groups.items()}
    by_attr = {}
    for g, rs in groups.items():
        by_attr[g] = {a: summarize_geo([r for r in rs if r["attribute"] == a]) for a in ATTRIBUTES
                      if any(r["attribute"] == a for r in rs)}
    pooled_install = [r for g, rs in groups.items() if g not in ("E9_zero_common10(ref=item)",) for r in rs]
    out = {"note": "Deduplicated to one record per (file, base, source, queried) cluster. "
                   "Among answers matching NEITHER base nor source: does the emitted value belong to a "
                   "country in the reference (source/installed) country's UN M49 subregion/continent "
                   "more often than a uniformly drawn other value would? 'other' = the base country. "
                   "Values shared by several countries (EUR, Spanish) count as a hit if ANY owner is "
                   "in the region; the null applies the same rule. z is a Poisson-binomial z.",
           "pooled_all_installs": summarize_geo(pooled_install),
           "by_group": summary, "by_group_attribute": by_attr}
    write_json("error_geography", out)
    return out


# ----------------------------------------------------------------------------- A5 name rank null

def simulate_best_of_k(n, k, trials=20000, seed=0):
    rng = random.Random(seed)
    ranks, zs, top1 = [], [], 0
    for _ in range(trials):
        xs = [rng.gauss(0, 1) for _ in range(n)]
        own = max(xs[i] for i in rng.sample(range(n), k))
        rank = 1 + sum(x > own for x in xs)
        ranks.append(rank)
        top1 += rank == 1
        zs.append((own - st.mean(xs)) / st.stdev(xs))
    return {"mean_rank": st.mean(ranks), "top1": top1 / trials, "mean_z": st.mean(zs)}


def analysis_name_rank_null(ctx):
    out = {"note": "Entity rank = best of the ' Name'/'Name' first tokens among n distinct candidates. "
                   "Under exchangeable logits that null is (n+1)/(k+1) with k=2, not (n+1)/2. k=1 is "
                   "given as a bound (spellings that collapse to one token).", "entities": {}}
    for entity in ("flags", "brands", "celebrities"):
        path = os.path.join(REPO_ROOT, "results", "head_vocab_projection", entity, "rows.jsonl")
        if not os.path.isfile(path):
            continue
        rows = read_jsonl(path)
        n = Counter(r["rec"]["entity"]["n"] for r in rows).most_common(1)[0][0]
        null2, null1 = simulate_best_of_k(n, 2), simulate_best_of_k(n, 1, seed=1)
        heads = defaultdict(list)
        for r in rows:
            heads[(r["head_set"], r["head"])].append(r["rec"]["entity"])
        per_head = []
        for (hs, h), es in sorted(heads.items()):
            per_head.append({"head_set": hs, "head": h, "n_rows": len(es),
                             "mean_rank": st.mean(e["rank"] for e in es),
                             "top1": sum(e["rank"] == 1 for e in es) / len(es),
                             "mean_z": st.mean(e["z"] for e in es)})
        sets = {}
        for hs in dict.fromkeys(p["head_set"] for p in per_head):
            ph = [p for p in per_head if p["head_set"] == hs]
            sets[hs] = {"n_heads": len(ph), "mean_rank": st.mean(p["mean_rank"] for p in ph),
                        "median_head_rank": st.median(p["mean_rank"] for p in ph),
                        "heads_below_half_null": sum(p["mean_rank"] < null2["mean_rank"] / 2 for p in ph),
                        "mean_z": st.mean(p["mean_z"] for p in ph)}
        out["entities"][entity] = {"n_candidates": n, "stored_chance": rows[0]["rec"]["entity"]["chance"],
                                   "null_best_of_2": null2, "null_k1": null1,
                                   "head_sets": sets, "heads": per_head}
    write_json("name_rank_null", out)
    return out


# ----------------------------------------------------------------------------- A9 per-token

def analysis_per_token(ctx):
    out = {"note": "Group totals from head_vocab_projection divided by group size. object = the "
                   "entity_only footprint; background = the other image tokens (144 - object).",
           "entities": {}}
    for entity in ("flags", "brands", "celebrities"):
        path = os.path.join(REPO_ROOT, "results", "head_vocab_projection", entity, "rows.jsonl")
        loc = os.path.join(ctx["vade_root"], "data", entity, "object_location.json")
        if not (os.path.isfile(path) and os.path.isfile(loc)):
            continue
        L = json.load(open(loc))
        n_obj = min(len(v["flat"]) for v in L["object_token_indices"].values())
        n_img = L["vlm_token_grid"]["n_tokens"]
        n_bg = n_img - n_obj
        heads = defaultdict(list)
        for r in read_jsonl(path):
            heads[(r["head_set"], r["head"])].append(r["rec"]["groups"])
        rows = []
        for (hs, h), gs in sorted(heads.items()):
            ao = st.mean(g["object"]["attention"] for g in gs)
            ab = st.mean(g["image_background"]["attention"] for g in gs)
            do = st.mean(g["object"]["dla_entity"] for g in gs)
            db = st.mean(g["image_background"]["dla_entity"] for g in gs)
            rows.append({"head_set": hs, "head": h, "attn_object": ao, "attn_background": ab,
                         "attn_per_token_ratio_obj_over_bg": (ao / n_obj) / (ab / n_bg) if ab > 0 else None,
                         "dla_entity_object": do, "dla_entity_background": db,
                         "dla_per_token_ratio_obj_over_bg": (do / n_obj) / (db / n_bg) if abs(db) > 1e-9 else None})
        out["entities"][entity] = {"n_object_tokens": n_obj, "n_background_tokens": n_bg, "heads": rows}
    write_json("per_token", out)
    return out


# ----------------------------------------------------------------------------- B9 compose failures

def analysis_compose_failures(ctx):
    gt = ctx["gt"]
    vi = ValueIndex(gt)
    path = os.path.join(REPO_ROOT, "results", "circuit_compose", "flags", "rows.jsonl")
    rows = [r for r in read_jsonl(path) if r.get("eligible")]
    cats = defaultdict(Counter)
    totals = Counter()
    examples = defaultdict(list)
    for r in rows:
        X, A, B, C = r["X"], r["A"], r["B"], r["C"]
        key = "same_format" if FORMAT[X] == FORMAT[A] else "different_format"
        totals[key] += 1
        if r["labels"]["compose"] != "other":
            continue
        t = r["texts"]["compose"]
        if contains_label(t, gt[B][X]):
            c = "B.X (right entity, base question's answer)"
        elif any(contains_label(t, gt[B][a]) for a in ATTRIBUTES if a not in (X, A)):
            c = "B.third_attribute"
        elif any(contains_label(t, gt[C][a]) for a in ATTRIBUTES):
            c = "C.any (base image's value)"
        elif vi.countries_in_text(A, t, (gt[B][A], gt[C][A])):
            c = "another country's A value"
        else:
            c = "unmapped (e.g. code/abbrev for B)"
        cats[key][c] += 1
        if len(examples[c]) < 6:
            examples[c].append({"X": X, "A": A, "B": B, "C": C, "text": t, "gold_B.A": gt[B][A]})
    by_pair = defaultdict(lambda: [0, 0])
    for r in rows:
        by_pair[f"{r['X']}->{r['A']}"][0] += 1
        by_pair[f"{r['X']}->{r['A']}"][1] += r["labels"]["compose"] == "other"
    out = {"n_eligible": len(rows),
           "other_rate_by_format": {k: sum(cats[k].values()) / totals[k] for k in totals},
           "n_by_format": dict(totals),
           "other_categories_by_format": {k: dict(v) for k, v in cats.items()},
           "other_rate_by_X_to_A": {k: {"n": n, "other": o / n} for k, (n, o) in sorted(by_pair.items())},
           "examples": dict(examples)}
    write_json("compose_failures", out)
    return out


# ----------------------------------------------------------------------------- E2 currency unique

def rebuild_route_pairs(gt, n_pairs, seed, attributes):
    """circuit_common.sample_pairs + pair_rows without a tokenizer: a row is dropped iff the
    two golds tokenize identically, i.e. iff the labels are equal."""
    names = sorted(gt)
    pairs = [(a, b) for a in names for b in names if a != b]
    random.Random(seed).shuffle(pairs)
    rows = []
    for base, source in pairs[:n_pairs]:
        for attr in attributes:
            if str(gt[base][attr]) == str(gt[source][attr]):
                continue
            rows.append({"row_index": len(rows), "base": base, "source": source, "attribute": attr})
    return rows


def analysis_currency_unique(ctx):
    gt = ctx["gt"]
    counts = Counter(str(it["currency"]) for it in gt.values())
    unique = {c for c, it in gt.items() if counts[str(it["currency"])] == 1}
    out = {"shared_currencies": {k: v for k, v in counts.items() if v > 1},
           "n_unique_currency_countries": len(unique)}
    # E5 route_knockout.
    rk_dir = os.path.join(REPO_ROOT, "results", "route_knockout", "flags")
    summ = json.load(open(os.path.join(rk_dir, "summary.json")))
    cfg = summ["config"]
    recs = read_jsonl(os.path.join(rk_dir, "rows.jsonl"))
    rows = rebuild_route_pairs(gt, cfg.get("n_pairs", 64), cfg.get("seed", 0), cfg.get("attributes", list(ATTRIBUTES)))
    meta = {r["row_index"]: r for r in rows}
    check = all(meta.get(r["row_index"], {}).get("attribute") == r["attribute"] for r in recs)
    out["route_knockout_pair_rebuild_ok"] = check and len(rows) == len({r["row_index"] for r in recs})
    base_ok = {r["row_index"] for r in recs if r["arm"] == "clean" and r["base_hit"]}
    src_ok = {r["row_index"] for r in recs if r["arm"] == "source_clean" and r["src_hit"]}
    elig = base_ok & src_ok
    arms = ["img_patch", "ko:readout_all+img_patch", "ko:question_all+img_patch", "qpatch@22+ko:readout_all"]
    tab = {}
    for arm in arms:
        cell = {}
        for attr in ATTRIBUTES:
            for subset in ("all", "both_unique") if attr == "currency" else ("all",):
                rr = [r for r in recs if r["arm"] == arm and r["attribute"] == attr and r["row_index"] in elig
                      and (subset == "all" or (meta[r["row_index"]]["base"] in unique
                                               and meta[r["row_index"]]["source"] in unique))]
                k = sum(r["src_hit"] for r in rr)
                cell[f"{attr}/{subset}"] = {"n": len(rr), "transfer": k / len(rr) if rr else None, "ci95": wilson(k, len(rr))}
        tab[arm] = cell
    out["route_knockout"] = tab
    # E9 vrh_ablation: currency accuracy by item currency uniqueness.
    vr = read_jsonl(os.path.join(REPO_ROOT, "results", "vrh_ablation", "flags", "rows.jsonl"))
    vt = {}
    for arm in dict.fromkeys(r["arm"] for r in vr):
        cell = {}
        for subset in ("all", "unique", "shared"):
            rr = [r for r in vr if r["arm"] == arm and r["attribute"] == "currency"
                  and (subset == "all" or ((r["item"] in unique) == (subset == "unique")))]
            k = sum(r["correct"] for r in rr)
            cell[subset] = {"n": len(rr), "acc": k / len(rr) if rr else None}
        vt[arm] = cell
    out["vrh_ablation_currency"] = vt
    write_json("currency_unique", out)
    return out


# ----------------------------------------------------------------------------- E3(b) survivors

def analysis_survivors(ctx):
    gt = ctx["gt"]
    vr = read_jsonl(os.path.join(REPO_ROOT, "results", "vrh_ablation", "flags", "rows.jsonl"))
    vi = ValueIndex(gt)
    out = {}
    for attr in ATTRIBUTES:
        share = Counter(normalize(str(it[attr])) for it in gt.values())
        def k_of(item):
            return share[normalize(str(gt[item][attr]))]
        clean = {(r["item"], r["template_id"]): r["correct"] for r in vr if r["arm"] == "clean" and r["attribute"] == attr}
        z = [r for r in vr if r["arm"] == "zero:common10" and r["attribute"] == attr and clean.get((r["item"], r["template_id"]))]
        by_k = defaultdict(lambda: [0, 0])
        for r in z:
            b = "shared(>=2 countries)" if k_of(r["item"]) >= 2 else "unique"
            by_k[b][0] += 1
            by_k[b][1] += r["correct"]
        wrong_owners = Counter()
        wrong_mapped = 0
        for r in z:
            if r["correct"]:
                continue
            cs = vi.countries_in_text(attr, r["generated_text"], (gt[r["item"]][attr],))
            if cs:
                wrong_mapped += 1
                vals = {normalize(str(gt[c][attr])) for c in cs}
                for v in vals:
                    wrong_owners[v] += 1
        n_wrong = sum(not r["correct"] for r in z)
        out[attr] = {"n_clean_correct": len(z), "survival": sum(r["correct"] for r in z) / len(z) if z else None,
                     "survival_by_value_sharing": {b: {"n": n, "survival": c / n} for b, (n, c) in by_k.items()},
                     "n_wrong": n_wrong, "wrong_mapped_to_other_value": wrong_mapped,
                     "top_wrong_values": [(v, c, share[v]) for v, c in wrong_owners.most_common(8)]}
    out["note"] = ("Rows restricted to (item, template) answered correctly in clean. survival = still correct "
                   "under zero:common10. top_wrong_values = (value, times emitted, #countries owning it).")
    write_json("survivors", out)
    return out


# ----------------------------------------------------------------------------- A7 cross bootstrap

def analysis_cross_bootstrap(ctx, n_boot=2000):
    path = os.path.join(REPO_ROOT, "results", "head_cross", "flags", "cross_n64.jsonl")
    rows = read_jsonl(path)
    rng = random.Random(0)
    pairs = sorted({(r["base"], r["source"]) for r in rows})
    bases = sorted({p[0] for p in pairs})
    sources = sorted({p[1] for p in pairs})

    def cells():
        for cell in ("flag", "both", "question", "self"):
            for qmode in ("q1==q2", "q1!=q2"):
                yield cell, qmode

    def rate(rr, cell, qmode):
        sel = [r for r in rr if r["cell"] == cell and ((r["q1"] == r["q2"]) == (qmode == "q1==q2"))]
        if not sel:
            return None, 0
        target = "source_q1" if cell in ("flag", "both") else "base_q1"
        return sum(r["label"] == target for r in sel) / len(sel), len(sel)

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r["base"], r["source"])].append(r)
    out = {"n_pairs": len(pairs), "n_bases": len(bases), "n_sources": len(sources), "cells": {}}
    for cell, qmode in cells():
        point, n = rate(rows, cell, qmode)
        if point is None:
            continue
        res = {"n": n, "rate": point}
        for scheme in ("pair", "base_country", "source_country"):
            boots = []
            for _ in range(n_boot):
                if scheme == "pair":
                    samp = [rng.choice(pairs) for _ in pairs]
                    rr = [x for p in samp for x in by_pair[p]]
                else:
                    pool = bases if scheme == "base_country" else sources
                    pick = Counter(rng.choice(pool) for _ in pool)
                    idx = 0 if scheme == "base_country" else 1
                    rr = [x for p in pairs for _ in range(pick.get(p[idx], 0)) for x in by_pair[p]]
                v, _ = rate(rr, cell, qmode)
                if v is not None:
                    boots.append(v)
            boots.sort()
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
            res[f"ci95_{scheme}"] = (lo, hi)
            res[f"halfwidth_{scheme}"] = (hi - lo) / 2
        out["cells"][f"{cell}/{qmode}"] = res
    out["note"] = ("cell flag/both: rate = answer is the donor flag's answer to q1; question/self: rate = base "
                   "answer kept. Country-cluster bootstraps resample base (or source) countries with all "
                   "their pairs.")
    write_json("cross_bootstrap", out)
    return out


# ----------------------------------------------------------------------------- G2 proxy vs ceiling

CEIL_HDR = re.compile(r"\[ndm/ceiling_sweep\] entity=(\S+) attribute=(\S+) .*?positions=(\S+) cause_rows=(\d+)")
CEIL_LINE = re.compile(r"^\s*residual layer\s+(\d+): cause_ceiling=\s*([\d.]+)%")


def parse_residual_ceilings(entity="flags", attribute="language"):
    """{positions: {layer: weighted-mean cause ceiling}} over every residual run in the logs."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    root = os.path.join(REPO_ROOT, "logs", "Qwen2.5-VL-7B-Instruct", entity, "ndm", attribute)
    for path in glob.glob(os.path.join(root, "*", "ceiling_sweep*.log")):
        pos, n, seen = None, 0, set()
        for line in open(path, errors="replace"):
            m = CEIL_HDR.search(line)
            if m:
                pos, n, seen = m.group(3), int(m.group(4)), set()
                if pos.startswith("["):
                    pos = None  # multi-position runs label rows differently; skip
                continue
            m = CEIL_LINE.match(line)
            if m and pos and n:
                L = int(m.group(1))
                if L in seen:  # the end-of-run summary repeats rows
                    continue
                seen.add(L)
                acc[pos][L][0] += float(m.group(2)) * n
                acc[pos][L][1] += n
    return {p: {L: s / n for L, (s, n) in sorted(d.items())} for p, d in acc.items()}


def analysis_proxy_ceiling(ctx):
    ceil = parse_residual_ceilings()
    out = {"ceilings_language_residual": {p: d for p, d in ceil.items() if p in ("flag_ring1", "full_image")},
           "selections": {}}
    for sel in sorted(glob.glob(os.path.join(REPO_ROOT, "methods", "selections", "flags", "*", "sweep.jsonl"))):
        name = os.path.basename(os.path.dirname(sel))
        best = defaultdict(float)
        for r in read_jsonl(sel):
            if r.get("attribute") == "language" and r.get("cause_score") is not None:
                best[r["layer"]] = max(best[r["layer"]], r["cause_score"])
        win_path = os.path.join(os.path.dirname(sel), "winners.json")
        winner = json.load(open(win_path)) if os.path.isfile(win_path) else None
        res = {"best_proxy_cause_by_layer": dict(sorted(best.items()))}
        for pos in ("flag_ring1", "full_image"):
            c = ceil.get(pos, {})
            layers = [L for L in sorted(best) if L in c]
            res[f"spearman_vs_{pos}"] = spearman([best[L] for L in layers], [c[L] for L in layers])
            res[f"n_layers_{pos}"] = len(layers)
        if winner:
            w = winner.get("winners", {}).get("language") if isinstance(winner, dict) else None
            if isinstance(w, dict) and "layer" in w:
                L = w["layer"]
                res["winner"] = {"layer": L, "proxy_cause": w.get("cause_score"),
                                 "ceiling_flag_ring1": ceil.get("flag_ring1", {}).get(L),
                                 "ceiling_full_image": ceil.get("full_image", {}).get(L)}
        out["selections"][name] = res
    out["note"] = ("Ceilings: weighted mean over every residual ceiling_sweep run for flags/language at that "
                   "position set (n=32/64 rows each). The current *_sae selections are the Kaggle-pool run "
                   "(RESULTS.md). Proxy = best cause_score at the layer over direction x C.")
    write_json("proxy_ceiling", out)
    return out


# ----------------------------------------------------------------------------- D7 / D8

def analysis_late_facts(ctx):
    s = json.load(open(os.path.join(REPO_ROOT, "results", "image_enrichment", "flags", "summary.json")))
    img = s["vocab"]["image"]
    out = {"regions": s["capture"]["regions"], "chance_mean_rank": s["vocab"]["chance_mean_rank"], "layers": {}}
    for L in [str(x) for x in range(18, 28)]:
        row = {}
        for region in ("object", "ring", "background", "all"):
            for mode in ("pooled", "max_token"):
                d = img.get(f"{region}/{mode}", {}).get(L)
                if d:
                    row[f"{region}/{mode}"] = {k: round(v["top1"], 3) for k, v in d.items()}
        if row:
            out["layers"][L] = row
    out["note"] = "top1 = fraction of countries whose own value ranks first among candidates (E1 exclusions apply)."
    write_json("late_facts", out)
    return out


def analysis_lasttoken_probe(ctx):
    s = json.load(open(os.path.join(REPO_ROOT, "results", "image_enrichment", "flags", "summary.json")))
    out = {}
    for probe, d in s["probe"].items():
        out[probe] = {"baseline": d["baseline"], "n": d["n"],
                      "curves": {k: v for k, v in d["curves"].items()},
                      "perm_best_layer": {k: v.get("best_layer") for k, v in d.get("perm", {}).items()}}
    write_json("lasttoken_probe", out)
    return out


# ----------------------------------------------------------------------------- report

def pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def write_report(res):
    L = ["# CPU-only follow-ups (generated by methods/cpu_followups.py)", "",
         "Numbers only; interpretation lives in CLAIM_CANDIDATES.md. Re-run the script to regenerate.", ""]
    if "entity_swap" in res:
        r = res["entity_swap"]
        L += ["## G1 entity-swap index", "",
              "index = mean P(non-target answer = SOURCE's value) / cause. ~1 = entity swap, << 1 = attribute edit.", "",
              "| family | cells | cells cause>=5% | median cause | median index | range |", "|---|---|---|---|---|---|"]
        for fam, v in r["families"].items():
            L.append(f"| {fam} | {v['n_cells']} | {v['n_cells_cause_ge_5pct']} | {pct(v['median_cause'])} | "
                     f"{fmt(v['median_index'])} | {fmt(v['min_index'])}-{fmt(v['max_index'])} |")
        L += ["", "| file | target | cause | non-target -> source | iso | non-target neither | index |", "|---|---|---|---|---|---|---|"]
        for c in r["cells"]:
            if c["family"] in ("clean",) or c["cause"] < 0.05:
                continue
            L.append(f"| {c['name']} | {c['target']} | {pct(c['cause'])} | {pct(c['nontarget_source_mean'])} | "
                     f"{pct(c['iso_mean'])} | {pct(c['nontarget_neither'])} | {fmt(c['entity_swap_index'])} |")
        L.append("")
    if "error_geography" in res:
        r = res["error_geography"]
        L += ["## A8 error geography", "",
              "Answers matching neither base nor source, mapped back to the country owning the emitted value.", "",
              "Deduplicated to one record per (file, base, source, queried). placebo = a random third country as "
              "reference; it should sit at its null if the null is calibrated.", "",
              "| group | clusters | mapped | same subregion as source (null) | same continent as source (null) | same subregion as base (null) | placebo subregion (null) | ITU zone (null) |",
              "|---|---|---|---|---|---|---|---|"]
        rows = [("POOLED (all installs)", r["pooled_all_installs"])] + list(r["by_group"].items())
        for g, v in rows:
            def geo_cell(key, v=v):
                d = v.get(key)
                return "-" if not d else f"{pct(d['observed'])} ({pct(d['null'])}, z={fmt(d['z'], 1)})"
            L.append(f"| {g} | {v['n_other']} | {pct(v['mapped_frac'])} | {geo_cell('subregion_ref')} | "
                     f"{geo_cell('continent_ref')} | {geo_cell('subregion_other')} | {geo_cell('subregion_placebo')} | "
                     f"{geo_cell('itu_zone_ref')} |")
        L.append("")
    if "name_rank_null" in res:
        r = res["name_rank_null"]
        L += ["## A5 name-rank null", "",
              "| entity | n cand. | stored chance | null mean rank (best of 2) | null top-1 | null z | head set | set mean rank | set mean z |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e, v in r["entities"].items():
            for hs, s in v["head_sets"].items():
                L.append(f"| {e} | {v['n_candidates']} | {v['stored_chance']} | {fmt(v['null_best_of_2']['mean_rank'], 1)} | "
                         f"{pct(v['null_best_of_2']['top1'])} | {fmt(v['null_best_of_2']['mean_z'])} | {hs} | "
                         f"{fmt(s['mean_rank'], 1)} | {fmt(s['mean_z'])} |")
        L.append("")
    if "per_token" in res:
        r = res["per_token"]
        L += ["## A9 per-token attention and entity write (object / background)", "",
              "| entity | head | attn object | attn background | per-token ratio | DLA object | DLA background | per-token DLA ratio |",
              "|---|---|---|---|---|---|---|---|"]
        for e, v in r["entities"].items():
            for h in v["heads"]:
                if h["head_set"] not in ("common10",) and e == "flags":
                    continue
                L.append(f"| {e} | {h['head_set']}:{h['head']} | {pct(h['attn_object'])} | {pct(h['attn_background'])} | "
                         f"{fmt(h['attn_per_token_ratio_obj_over_bg'], 1)} | {fmt(h['dla_entity_object'], 3)} | "
                         f"{fmt(h['dla_entity_background'], 3)} | {fmt(h['dla_per_token_ratio_obj_over_bg'], 1)} |")
        L.append("")
    if "compose_failures" in res:
        r = res["compose_failures"]
        L += ["## B9 E4 compose failures", "", f"eligible rows: {r['n_eligible']}", "",
              "| X vs A format | n | other rate | categories |", "|---|---|---|---|"]
        for k, n in r["n_by_format"].items():
            cats = ", ".join(f"{c}: {m}" for c, m in r["other_categories_by_format"].get(k, {}).items())
            L.append(f"| {k} | {n} | {pct(r['other_rate_by_format'][k])} | {cats} |")
        L += ["", "| X -> A | n | other |", "|---|---|---|"]
        for k, v in r["other_rate_by_X_to_A"].items():
            L.append(f"| {k} | {v['n']} | {pct(v['other'])} |")
        L.append("")
    if "currency_unique" in res:
        r = res["currency_unique"]
        L += ["## E2 currency on unique-currency rows", "",
              f"pair rebuild ok: {r['route_knockout_pair_rebuild_ok']}; shared currencies: {r['shared_currencies']}", "",
              "| route_knockout arm | " + " | ".join(r["route_knockout"]["img_patch"].keys()) + " |",
              "|---" * (1 + len(r["route_knockout"]["img_patch"])) + "|"]
        for arm, cell in r["route_knockout"].items():
            L.append(f"| {arm} | " + " | ".join(f"{pct(v['transfer'])} (n={v['n']})" for v in cell.values()) + " |")
        L += ["", "| vrh_ablation arm | currency all | unique | shared |", "|---|---|---|---|"]
        for arm, cell in r["vrh_ablation_currency"].items():
            L.append(f"| {arm} | " + " | ".join(f"{pct(cell[s]['acc'])} (n={cell[s]['n']})" for s in ("all", "unique", "shared")) + " |")
        L.append("")
    if "survivors" in res:
        r = res["survivors"]
        L += ["## E3(b) what survives zero:common10", "",
              "| attribute | clean-correct n | survival | unique-value survival | shared-value survival | top wrong values (value, times, owners) |",
              "|---|---|---|---|---|---|"]
        for a in ATTRIBUTES:
            v = r[a]
            sv = v["survival_by_value_sharing"]
            L.append(f"| {a} | {v['n_clean_correct']} | {pct(v['survival'])} | "
                     f"{pct(sv.get('unique', {}).get('survival'))} (n={sv.get('unique', {}).get('n', 0)}) | "
                     f"{pct(sv.get('shared(>=2 countries)', {}).get('survival'))} (n={sv.get('shared(>=2 countries)', {}).get('n', 0)}) | "
                     f"{v['top_wrong_values'][:5]} |")
        L.append("")
    if "cross_bootstrap" in res:
        r = res["cross_bootstrap"]
        L += ["## A7 head_cross CIs", "", f"{r['n_pairs']} pairs, {r['n_bases']} bases, {r['n_sources']} sources", "",
              "| cell | n | rate | ±pair | ±base-cluster | ±source-cluster |", "|---|---|---|---|---|---|"]
        for k, v in r["cells"].items():
            L.append(f"| {k} | {v['n']} | {pct(v['rate'])} | {pct(v['halfwidth_pair'])} | "
                     f"{pct(v['halfwidth_base_country'])} | {pct(v['halfwidth_source_country'])} |")
        L.append("")
    if "proxy_ceiling" in res:
        r = res["proxy_ceiling"]
        L += ["## G2 proxy vs ceiling (flags/language)", "",
              "| selection | Spearman vs flag_ring1 ceiling (layers) | vs full_image (layers) | winner layer | winner proxy | ceiling@winner ring1 / full |",
              "|---|---|---|---|---|---|"]
        for name, v in r["selections"].items():
            w = v.get("winner", {})
            L.append(f"| {name} | {fmt(v['spearman_vs_flag_ring1'])} ({v['n_layers_flag_ring1']}) | "
                     f"{fmt(v['spearman_vs_full_image'])} ({v['n_layers_full_image']}) | {w.get('layer', '-')} | "
                     f"{fmt(w.get('proxy_cause'))} | {fmt(w.get('ceiling_flag_ring1'), 1)} / {fmt(w.get('ceiling_full_image'), 1)} |")
        L.append("")
    if "late_facts" in res:
        r = res["late_facts"]
        L += ["## D7 late image-token facts (top-1 rate)", "",
              "| layer | object pooled name/capital/language | background pooled | object max_token | background max_token |",
              "|---|---|---|---|---|"]
        for Ly, row in r["layers"].items():
            def t(key):
                d = row.get(key, {})
                return "/".join(fmt(d.get(k), 2) for k in ("name_rank", "capital_rank", "language_rank"))
            L.append(f"| {Ly} | {t('object/pooled')} | {t('background/pooled')} | {t('object/max_token')} | {t('background/max_token')} |")
        L.append("")
    if "lasttoken_probe" in res:
        r = res["lasttoken_probe"]
        L += ["## D8 probe curves at every layer", ""]
        for probe, v in r.items():
            L.append(f"**{probe}** (n={v['n']}, baseline {fmt(v['baseline'])}; best layers {v['perm_best_layer']})")
            L.append("")
            curves = v["curves"]
            layers = sorted({int(k) for c in curves.values() for k in c})
            L.append("| curve | " + " | ".join(str(x) for x in layers) + " |")
            L.append("|---" * (1 + len(layers)) + "|")
            for name, c in curves.items():
                L.append(f"| {name} | " + " | ".join(fmt(c.get(str(x))) for x in layers) + " |")
            L.append("")
    path = os.path.join(OUT_DIR, "REPORT.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


ANALYSES = {
    "entity_swap": analysis_entity_swap,
    "error_geography": analysis_error_geography,
    "name_rank_null": analysis_name_rank_null,
    "per_token": analysis_per_token,
    "compose_failures": analysis_compose_failures,
    "currency_unique": analysis_currency_unique,
    "survivors": analysis_survivors,
    "cross_bootstrap": analysis_cross_bootstrap,
    "proxy_ceiling": analysis_proxy_ceiling,
    "late_facts": analysis_late_facts,
    "lasttoken_probe": analysis_lasttoken_probe,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--analyses", nargs="+", choices=list(ANALYSES), default=list(ANALYSES))
    args = ap.parse_args()
    assert set(SUBREGION) and len(SUBREGION) == 84, f"M49 table has {len(SUBREGION)} countries, expected 84"
    gt = load_ground_truth(args.vade_root)
    assert set(gt) == set(SUBREGION), f"M49 table / ground truth mismatch: {set(gt) ^ set(SUBREGION)}"
    ctx = {"vade_root": args.vade_root, "gt": gt, "tuples": load_tuples(args.vade_root)}
    res = {}
    for name in args.analyses:
        print(f"[cpu_followups] {name} ...", flush=True)
        res[name] = ANALYSES[name](ctx)
    print(f"[cpu_followups] report -> {rel(write_report(res))}")


if __name__ == "__main__":
    main()
