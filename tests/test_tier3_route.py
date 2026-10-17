"""Offline checks for route_knockout's Tier-3 additions (FOLLOWUP_PLAN C10):
key groups + stacked knockouts (C10a), head install + knockout (C10b), and the
non-flags entity rows (C10c).

Tiny random Qwen2.5-VL from test_head_followups, given a 4-token image here so the
object/background split has something to split.
"""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import circuit_common as cc
from methods import route_knockout as rk
from methods.head_token_ablation import AttentionEdges


def wide(tiny, rows=1, prefix=()):
    """-> (m, runner, ids_b, ids_s, mask, xb, xs): a 4-image-token prompt (1x4x4 patches,
    2x2 merge) with `prefix` appended, `rows` copies, base/source differing only in pixels."""
    runner, _ = tiny
    m = cc.Model.wrap(runner.adapter, runner.model, None)
    g = torch.Generator().manual_seed(7)
    ids = torch.tensor([[1, 2, 2, 2, 2, 3, 12, 13, 14, *prefix]] * rows)

    def extra():
        return {"pixel_values": torch.randn(16 * rows, 24, generator=g),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * rows)}
    return m, runner, ids, ids.clone(), torch.ones_like(ids), extra(), extra()


def logits(m, ids, mask, extra, attn=False):
    return m.forward(ids, mask, extra, output_attentions=attn).logits[:, -1]


# ---------------------------------------------------------------------------
# C10a: spec grammar
# ---------------------------------------------------------------------------

def legacy_parse_ko(spec, n_layers=None):
    """route_knockout.parse_ko as it was before C10, verbatim."""
    name, body = spec.split("=", 1)
    queries, span = body.split(":", 1)
    assert queries in rk.QUERY_GROUPS
    a, _, b = span.partition("-")
    blocks = list(range(int(a), int(b or a) + 1))
    if n_layers is not None:
        assert all(0 <= x < n_layers for x in blocks)
    return name, queries, blocks


def test_legacy_specs_parse_exactly_as_before():
    for spec in rk.DEFAULT_KO + ["one=readout:7", "x=text:3-3"]:
        old = legacy_parse_ko(spec, 28)
        assert rk.parse_ko(spec, 28) == old
        assert rk.parse_ko(spec + "@image", 28) == old       # explicit default group
        assert rk.parse_ko_parts(spec, 28) == (old[0], [(old[1], old[2], "image")])
        assert rk.describe_ko(rk.parse_ko_parts(spec)[1]) == f"{old[1]:>8} -> image, blocks {old[2][0]}-{old[2][-1]}"


def test_stacked_specs_round_trip_and_reject_typos():
    spec = "obj_noq=readout:21-23@object+question:0-27@image+text:5@background"
    name, parts = rk.parse_ko_parts(spec, 28)
    assert name == "obj_noq" and parts == [("readout", [21, 22, 23], "object"),
                                           ("question", list(range(28)), "image"), ("text", [5], "background")]
    again = name + "=" + "+".join(f"{q}:{bl[0]}-{bl[-1]}@{g}" for q, bl, g in parts)
    assert rk.parse_ko_parts(again, 28) == (name, parts)
    assert rk.parse_ko_parts("m=readout:21-23+question:0-27")[1][1] == ("question", list(range(28)), "image")
    for bad in ("x=readout:1-2@ring", "x=image:1-2", "x=readout:1-2@object+image:0-3"):
        with pytest.raises(AssertionError):
            rk.parse_ko_parts(bad)
    with pytest.raises(AssertionError):
        rk.parse_ko_parts("x=readout:20-28", 28)
    with pytest.raises(AssertionError, match="parse_ko_parts"):
        rk.parse_ko("x=readout:1-2@object")
    with pytest.raises(AssertionError, match="parse_ko_parts"):
        rk.parse_ko("x=readout:1-2+question:1")


def test_object_and_background_partition_the_image_columns():
    cols = list(range(10, 154))                        # 144 image columns at an arbitrary offset
    for entity, n_obj in (("flags", 8), ("brands", 8), ("celebrities", 16)):
        d = os.path.join(cc.DEFAULT_VADE_ROOT, "data", entity)
        if not os.path.exists(os.path.join(d, "object_location.json")):
            pytest.skip("VADE checkout not present")
        n, obj = rk.object_index(d)
        assert n == 144 and len(obj) == n_obj, entity
        if entity == "flags":                          # flags' entity_only (= flag_only), image-relative
            loc = json.load(open(os.path.join(d, "object_location.json")))
            assert obj == sorted(loc["object_token_indices"]["entity_only"]["flat"])
        k = rk.key_columns(cols, (n, obj))
        assert k["object"] == [cols[i] for i in obj] and len(k["background"]) == 144 - n_obj
        assert sorted(k["object"] + k["background"]) == k["image"] == cols
        assert not set(k["object"]) & set(k["background"])
    with pytest.raises(AssertionError, match="image tokens"):
        rk.key_columns(cols[:100], (n, obj))
    assert rk.key_columns([3, 4]) == {"image": [3, 4]}


# ---------------------------------------------------------------------------
# C10a: stacking = one knockout over the union
# ---------------------------------------------------------------------------

def test_stacked_object_plus_background_equals_whole_image(tiny):
    m, _, ids, _, mask, xb, _ = wide(tiny, rows=2, prefix=(20,))
    img, q, r = rk.layout(ids, m.image_id, 1)
    k = rk.key_columns(img, (4, [1, 2]))
    for renorm in (True, False):
        with rk.EdgeKnockout(m, [1, 2], r, img, renormalize=renorm, check=True) as one:
            ref = logits(m, ids, mask, xb, attn=True)
        parts = [([1, 2], r, k["object"]), ([1, 2], r, k["background"])]
        with rk.EdgeKnockout.union(m, parts, renormalize=renorm, check=True) as both:
            got = logits(m, ids, mask, xb, attn=True)
        assert one.calls == both.calls == 2 and both.max_recon_rel < 1e-4
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
        assert (got - logits(m, ids, mask, xb)).abs().max() > 1e-4         # and it is live


def test_union_with_mixed_blocks_matches_attention_edges_reference(tiny):
    """Per-block key sets differ (block 1: {1}; block 2: {1,2,3}; block 3: {2,3}),
    checked against head_token_ablation's independent per-head implementation."""
    m, runner, ids, _, mask, xb, _ = wide(tiny)
    T = ids.shape[1]
    want = {1: [1], 2: [1, 2, 3], 3: [2, 3]}
    for renorm in (True, False):
        removals = {(b, h): ks for b, ks in want.items() for h in range(m.n_heads)}
        with AttentionEdges(runner, list(removals), T, removals=removals, renormalize=renorm):
            ref = logits(m, ids, mask, xb, attn=True)
        with rk.EdgeKnockout.union(m, [([1, 2], [T - 1], [1]), ([2, 3], [T - 1], [2, 3])],
                                   renormalize=renorm, check=True) as ko:
            got = logits(m, ids, mask, xb, attn=True)
        assert ko.blocks == [1, 2, 3] and ko.calls == 3
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_naive_nesting_composes_only_on_disjoint_query_rows(tiny):
    """Why the union exists: two separate corrections on a SHARED query row both start
    from the original A, so overlapping edges are removed twice and each renormalizes
    with only its own keys; on disjoint rows they are independent and agree."""
    m, _, ids, _, mask, xb, _ = wide(tiny)
    T = ids.shape[1]
    with rk.EdgeKnockout.union(m, [([1], [T - 1], [1, 2]), ([1], [T - 1], [2, 3])]):
        union = logits(m, ids, mask, xb, attn=True)
    with rk.EdgeKnockout(m, [1], [T - 1], [1, 2]), rk.EdgeKnockout(m, [1], [T - 1], [2, 3]):
        nested = logits(m, ids, mask, xb, attn=True)
    assert (nested - union).abs().max() > 1e-4
    with rk.EdgeKnockout.union(m, [([1], [T - 1], [1, 2]), ([1], [6, 7], [3, 4])]):
        union = logits(m, ids, mask, xb, attn=True)
    with rk.EdgeKnockout(m, [1], [T - 1], [1, 2]), rk.EdgeKnockout(m, [1], [6, 7], [3, 4]):
        nested = logits(m, ids, mask, xb, attn=True)
    torch.testing.assert_close(nested, union, atol=1e-5, rtol=1e-5)


def test_union_with_no_keys_is_an_exact_noop(tiny):
    m, _, ids, _, mask, xb, _ = wide(tiny, rows=2)
    clean = logits(m, ids, mask, xb)
    with rk.EdgeKnockout.union(m, [(range(4), [5, 6, 7, 8], []), ([1, 2], [8], [])], check=True):
        same = logits(m, ids, mask, xb, attn=True)
    assert torch.equal(same, clean)


# ---------------------------------------------------------------------------
# C10b: head install (+ knockout)
# ---------------------------------------------------------------------------

def test_install_own_values_is_exact_and_source_values_are_live(tiny):
    m, _, ids_b, ids_s, mask, xb, xs = wide(tiny, rows=2, prefix=(20,))
    _, _, r = rk.layout(ids_b, m.image_id, 1)
    heads = [(1, 0), (1, 3), (2, 1)]
    clean = logits(m, ids_b, mask, xb)
    with rk.HeadCapture(m, [1, 2], r) as own:
        logits(m, ids_b, mask, xb)
    with rk.HeadInstall(m, heads, r, own.z) as inst:
        same = logits(m, ids_b, mask, xb)
    assert inst.calls == 2 and torch.equal(same, clean)
    with rk.HeadCapture(m, [1, 2], r) as src:
        logits(m, ids_s, mask, xs)
    with rk.HeadInstall(m, heads, r, src.z):
        installed = logits(m, ids_b, mask, xb)
    assert (installed - clean).abs().max() > 1e-4
    # the install writes exactly the installed slices and leaves the other heads alone
    with rk.HeadInstall(m, heads, r, src.z), rk.HeadCapture(m, [1], r) as seen:
        logits(m, ids_b, mask, xb)
    d = m.head_dim
    torch.testing.assert_close(seen.z[1][..., :d], src.z[1][..., :d])
    torch.testing.assert_close(seen.z[1][..., 3 * d:], src.z[1][..., 3 * d:])
    # an empty knockout inside the install changes nothing, and a live one in a later
    # block passes the A@V check on the installed run
    with rk.HeadInstall(m, heads, r, src.z), rk.EdgeKnockout(m, [3], r + [5, 6], [], check=True):
        assert torch.equal(logits(m, ids_b, mask, xb, attn=True), installed)
    with rk.HeadInstall(m, heads, r, src.z), rk.EdgeKnockout(m, [3], r, [1, 2, 3, 4], check=True) as ko:
        both = logits(m, ids_b, mask, xb, attn=True)
    assert ko.max_recon_rel < 1e-4 and (both - installed).abs().max() > 1e-5


def test_install_clashes_are_refused():
    kos = [rk.parse_ko_parts(s) for s in ("late=readout:24-27", "q=question:21-23",
                                          "mix=question:0-27+readout:20-21@object", "t=text:23")]
    assert rk.install_clashes(kos, cc.parse_heads(cc.MOVERS)) == [("mix", [21]), ("t", [23])]
    assert rk.install_clashes(kos, None) == []


def run_args(**kw):
    return SimpleNamespace(**{"img_layer": 1, "q_layers": [1, 2], "no_renormalize": False, **kw})


def batch_rows(n, k):
    return [{"row_index": i, "attribute": "capital", "prefix": [20] * k, "base_tok": 7 + i, "src_tok": 30 + i}
            for i in range(n)]


def test_run_batch_new_arms_and_self_checks(tiny, capsys):
    m, _, ids_b, ids_s, mask, xb, xs = wide(tiny, rows=2, prefix=(20,))
    rows = batch_rows(2, 1)
    kos = [rk.parse_ko_parts(s, m.n_layers) for s in
           ("late=readout:2-3", "obj=readout:2-3@object+question:0-1@background", "q=question:1", "none=readout:3@object")]
    heads = [(1, 0), (1, 2)]
    recs = rk.run_batch(m, run_args(), kos, rows, ids_b, ids_s, mask, xb, xs, check=True, obj=(4, [1, 2]),
                        install=heads)
    arms = list(dict.fromkeys(r["arm"] for r in recs))
    assert arms == (["clean", "source_clean", "img_patch"]
                    + [f"ko:{n}{s}" for n, _ in kos for s in ("", "+img_patch")]
                    + [f"qpatch@{L}{s}" for L in (1, 2) for s in ("", "+ko:readout_all")]
                    + ["install"] + [f"install+ko:{n}" for n, _ in kos])
    out = capsys.readouterr().out
    assert "stacked knockout with NO keys" in out and "install of the base's own values" in out
    # the arms equal the same interventions run by hand
    img, q, r = rk.layout(ids_b, m.image_id, 1)
    with rk.HeadCapture(m, [1], r) as src:
        logits(m, ids_s, mask, xs)
    with rk.HeadInstall(m, heads, r, src.z), rk.EdgeKnockout.union(
            m, [([2, 3], r, [img[1], img[2]]), ([0, 1], q, [img[0], img[3]])]):
        want = cc.readout_metrics(logits(m, ids_b, mask, xb, attn=True), rows)
    got = [x for x in recs if x["arm"] == "install+ko:obj"]
    assert [g["ld"] for g in got] == pytest.approx([w["ld"] for w in want], abs=1e-5)
    with pytest.raises(ValueError, match="installed heads"):
        rk.run_batch(m, run_args(), [rk.parse_ko_parts("bad=readout:1")], rows, ids_b, ids_s, mask, xb, xs,
                     install=heads)


def test_run_batch_default_path_matches_a_hand_built_knockout(tiny):
    """No obj, no install: the original arms only, and a legacy ko arm is the plain
    single-part EdgeKnockout over every image column."""
    m, _, ids_b, ids_s, mask, xb, xs = wide(tiny, rows=2)
    rows = batch_rows(2, 0)
    kos = [rk.parse_ko_parts(s, m.n_layers) for s in ("ra=readout:0-3", "qa=question:0-3")]
    recs = rk.run_batch(m, run_args(), kos, rows, ids_b, ids_s, mask, xb, xs, check=True)
    assert not any(r["arm"].startswith("install") for r in recs)
    img, q, r = rk.layout(ids_b, m.image_id, 0)
    with rk.EdgeKnockout(m, range(4), q, img):
        want = cc.readout_metrics(logits(m, ids_b, mask, xb, attn=True), rows)
    assert [x["ld"] for x in recs if x["arm"] == "ko:qa"] == [w["ld"] for w in want]


# ---------------------------------------------------------------------------
# C10c: non-flags entity rows
# ---------------------------------------------------------------------------

def test_own_rows_use_the_models_answers_and_drop_what_cannot_be_read():
    items = {"A": {"hq": "united states"}, "B": {"hq": "germany"}, "C": {"hq": "united states"},
             "D": {"hq": "japan"}}
    words = {1: " the", 2: " United", 3: " States", 4: " Germany", 5: ".", 6: " Japan"}
    answers = {("A", "hq"): [1, 2, 3, 5], ("B", "hq"): [4, 5], ("C", "hq"): [1, 2, 3],
               ("D", "hq"): [1, 4, 5]}                     # D's own answer is wrong (Germany)

    def decode(t):
        return "".join(words[x] for x in t)
    pairs = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "A")]
    rows, dropped = rk.own_rows(items, pairs, ["hq"], answers, decode)
    assert dropped == {"no_label": 0, "same_label": 1, "model_wrong": 1, "no_divergence": 0}
    assert [(r["base"], r["source"], r["prefix"], r["base_tok"], r["src_tok"]) for r in rows] == [
        ("A", "B", [], 1, 4), ("B", "A", [], 4, 1)]
    assert [r["row_index"] for r in rows] == [0, 1] and rows[0]["base_label"] == "united states"
    # answers that agree up to where one stops have no divergence point
    items["E"] = {"hq": "united"}
    answers[("E", "hq")] = [1, 2]
    assert rk.own_rows(items, [("A", "E")], ["hq"], answers, decode)[1]["no_divergence"] == 1
    items["F"] = {}                                    # brands: AIRBUS has no industry label
    assert rk.own_rows(items, [("A", "F"), ("F", "A")], ["hq"], answers, decode)[1]["no_label"] == 2


def test_chunks_by_attribute_never_mix_attributes_or_prefix_lengths():
    rows = [{"attribute": a, "prefix": [0] * k} for a in ("x", "y") for k in (0, 1, 0, 2, 1)]
    chunks = list(rk.chunks_by_attribute(rows, 2))
    assert all(len({(r["attribute"], len(r["prefix"])) for r in c}) == 1 for c in chunks)
    assert sum(len(c) for c in chunks) == len(rows)


def test_load_entity_picks_vade_templates_by_id():
    for entity, n_attrs in (("brands", 3), ("celebrities", 4)):
        if not os.path.exists(os.path.join(cc.DEFAULT_VADE_ROOT, "data", entity, "ground_truth.json")):
            pytest.skip("VADE checkout not present")
        entity_dir, items, sha, prompts = rk.load_entity(cc.DEFAULT_VADE_ROOT, entity)
        assert len(prompts) == n_attrs and len(sha) == 64 and items
        tpl = json.load(open(os.path.join(entity_dir, "prompt_templates.json")))
        for a, (question, prefill) in prompts.items():
            first = min(tpl[a], key=lambda t: t["template_id"])
            assert (question, prefill) == (first["question"], first["prefill"])
        missing = [(k, a) for k, it in items.items() for a in prompts if a not in it]
        assert missing == ([("AIRBUS", "industry")] if entity == "brands" else [])
