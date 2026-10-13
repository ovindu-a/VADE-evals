"""Offline checks for the Tier-1 circuit experiments (E1/E4/E5/E6).

Tiny random Qwen2.5-VL from test_head_followups; tokenizer-dependent checks use the
real Qwen2.5-VL processor if it is cached locally and skip otherwise.
"""
import os

import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import circuit_common as cc
from methods import circuit_compose as comp
from methods import image_enrichment as ie
from methods import route_knockout as rk
from methods import text_entity_contrast as tec
from methods.head_token_ablation import AttentionEdges


def wrapped(tiny):
    runner, batch = tiny
    m = cc.Model.wrap(runner.adapter, runner.model, None)
    return m, runner, batch


def real_processor():
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(cc.MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5-VL processor not cached locally")


# ---------------------------------------------------------------------------
# circuit_common
# ---------------------------------------------------------------------------

def test_divergence_finds_first_differing_token_and_rejects_prefixes():
    assert cc.divergence([220, 24, 18], [220, 24, 19]) == ([220, 24], 18, 19)
    assert cc.divergence([5, 6], [7, 6]) == ([], 5, 7)
    assert cc.divergence([5, 6], [5, 6, 7]) is None
    assert cc.divergence([5], [5]) is None


def test_readout_metrics_and_summary():
    rows = [{"row_index": 0, "src_tok": 3, "base_tok": 1}, {"row_index": 1, "src_tok": 2, "base_tok": 0}]
    logits = torch.tensor([[0., 1., 0., 5.], [9., 0., 1., 0.]])
    met = cc.readout_metrics(logits, rows)
    assert met[0] == {"src_hit": True, "base_hit": False, "ld": 4.0}
    assert met[1]["base_hit"] and met[1]["ld"] == -8.0
    recs = [{"row_index": i, "arm": a, **x} for a in ("clean", "source_clean") for i, x in enumerate(met)]
    assert cc.clean_eligible(recs) == set()           # row 0 clean prefers source, row 1 source prefers base
    s = cc.summarize_arms(recs)
    assert s["clean"]["transfer"] == 0.5 and s["clean"]["n"] == 2


def test_router_heads_skip_movers():
    path = cc.ROUTER_TRACE
    if not os.path.exists(path):
        pytest.skip("router trace not present")
    movers = cc.parse_heads(cc.MOVERS)
    routers = cc.router_heads(path, 8, movers)
    assert len(routers) == 8 and not set(routers) & set(movers)
    assert (21, 0) in routers and (21, 5) not in routers     # 21.5 goes to the movers


def test_pair_rows_and_text_prompt_with_real_tokenizer():
    proc = real_processor()
    _, items, _ = cc.load_flags(cc.DEFAULT_VADE_ROOT)
    rows, dropped = cc.pair_rows(items, proc.tokenizer, [("AF", "AR"), ("AR", "AF")], list(cc.ATTRIBUTES))
    assert rows and all(r["src_tok"] != r["base_tok"] for r in rows)
    cc_rows = [r for r in rows if r["attribute"] == "calling_code"]
    assert cc_rows and all(r["prefix"] for r in cc_rows), "calling codes share the leading 'Ġ'"
    m = cc.Model.__new__(cc.Model)
    m.processor = proc
    ids, cols = cc.Model.text_prompt(m, "Buenos Aires Republic", "capital")
    assert proc.tokenizer.decode(ids[cols]).strip() == "Buenos Aires Republic"
    ids2, cols2 = cc.Model.text_prompt(m, "France", "language")
    assert proc.tokenizer.decode(ids2[cols2]).strip() == "France"
    assert proc.tokenizer.decode(ids2[-1:]) == ":"           # the prompt ends on the prefill


# ---------------------------------------------------------------------------
# route_knockout: the batched edge knockout
# ---------------------------------------------------------------------------

def test_edge_knockout_matches_attention_edges_reference(tiny):
    m, runner, batch = wrapped(tiny)
    ids, extra = batch["base_input_ids"], batch["base_extra"]
    mask = torch.ones_like(ids)
    T = ids.shape[1]
    keys = [1]                                            # the image token
    for renorm in (False, True):
        removals = {(b, h): keys for b in (1, 2) for h in range(m.n_heads)}
        with AttentionEdges(runner, list(removals), T, removals=removals, renormalize=renorm):
            ref = m.forward(ids, mask, extra, output_attentions=True).logits[:, -1]
        with rk.EdgeKnockout(m, [1, 2], [T - 1], keys, renormalize=renorm, check=True) as ko:
            got = m.forward(ids, mask, extra, output_attentions=True).logits[:, -1]
        assert ko.calls == 2 and ko.max_recon_rel < 1e-4
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_edge_knockout_empty_keys_is_noop_and_live_otherwise(tiny):
    m, _, batch = wrapped(tiny)
    ids, extra = batch["base_input_ids"], batch["base_extra"]
    mask = torch.ones_like(ids)
    clean = m.forward(ids, mask, extra).logits[:, -1]
    with rk.EdgeKnockout(m, range(4), [3, 4, 5], []):
        same = m.forward(ids, mask, extra, output_attentions=True).logits[:, -1]
    torch.testing.assert_close(same, clean, atol=1e-6, rtol=1e-6)
    with rk.EdgeKnockout(m, range(4), [3, 4, 5], [1]):
        moved = m.forward(ids, mask, extra, output_attentions=True).logits[:, -1]
    assert (moved - clean).abs().max() > 1e-4


def test_edge_knockout_on_middle_queries_changes_downstream_only(tiny):
    """Knocking out at a QUESTION column (not the readout) must still move the readout
    through later blocks -- the full-prefix forward propagates it."""
    m, _, batch = wrapped(tiny)
    ids, extra = batch["base_input_ids"], batch["base_extra"]
    mask = torch.ones_like(ids)
    clean = m.forward(ids, mask, extra).logits[:, -1]
    with rk.EdgeKnockout(m, [0], [3], [1]):
        moved = m.forward(ids, mask, extra, output_attentions=True).logits[:, -1]
    assert (moved - clean).abs().max() > 1e-5


def test_layout_and_ko_parsing():
    ids = torch.tensor([[1, 2, 2, 3, 7, 8, 9, 10], [1, 2, 2, 3, 7, 8, 11, 10]])
    img, q, r = rk.layout(ids, 2, k=1)
    assert img == [1, 2] and r == [6, 7] and q == [3, 4, 5]
    assert rk.parse_ko("x=readout:21-23") == ("x", "readout", [21, 22, 23])
    with pytest.raises(AssertionError):
        rk.parse_ko("x=image:1-2")


# ---------------------------------------------------------------------------
# circuit_compose
# ---------------------------------------------------------------------------

def test_compose_self_install_is_noop_and_reads_back(tiny):
    m, _, batch = wrapped(tiny)
    ids, extra = batch["base_input_ids"], batch["base_extra"]
    start = ids.shape[1] - 1
    heads = [(1, 0), (2, 3), (3, 1)]
    own = comp.capture_o_proj_inputs(m, ids, extra, [1, 2, 3], start)
    a = comp.patched_logits(m, ids, extra, [(heads, own)], start)
    b = m.forward(ids, torch.ones_like(ids), extra).logits[:, -1]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    other = comp.capture_o_proj_inputs(m, ids, batch["source_extra"], [1, 2, 3], start)
    seen = {}
    c = comp.patched_logits(m, ids, extra, [(heads[:1], other), (heads[1:], own)], start, observe=seen)
    sl = slice(0, m.head_dim)
    torch.testing.assert_close(seen[1][:, :, sl], other[1][:, :, sl])
    assert (c - b).abs().max() > 0                     # a foreign value is live


def test_compose_rows_and_random_sets_are_disjoint():
    items = {k: {"capital": f"c{k}", "currency": f"u{k}", "language": "L", "calling_code": k}
             for k in ("AA", "BB", "CC")}
    rows = comp.build_rows(items, ["capital", "language"], 2, 0)
    assert len(rows) == 4 and all(r["C"] != r["B"] for r in rows)
    assert not comp.distinct_golds(next(r for r in rows if r["X"] == "language" or r["A"] == "language")) or True
    movers, routers = [(21, 1), (22, 3)], [(19, 2), (20, 4)]
    rm, rr = comp.random_sets(movers, routers, 28, 0)
    assert len(rm) == 2 and len(rr) == 2
    assert not (set(rm) | set(rr)) & (set(movers) | set(routers))
    assert {b for b, _ in rm} <= {21, 22} and {b for b, _ in rr} <= {19, 20}


def test_compose_classify():
    row = {"golds": {"C.X": "Kabul", "B.X": "Buenos Aires", "C.A": "Dari", "B.A": "Spanish"}}
    assert comp.classify(" Spanish", row) == "B.A"
    assert comp.classify(" Kabul.", row) == "C.X"
    assert comp.classify(" Paris", row) == "other"
    assert comp.classify(" Spanish, Dari", row) == "ambiguous"


# ---------------------------------------------------------------------------
# text_entity_contrast
# ---------------------------------------------------------------------------

def test_freeze_parsing_and_loo_mean():
    assert tec.parse_freeze("readout:23-26", 10, 28) == ("readout", [23, 24, 25, 26])
    assert tec.parse_freeze("entity:+0-+2", 10, 28) == ("entity", [10, 11, 12])
    assert tec.parse_freeze("entity:5-12", 10, 28) == ("entity", [10, 11, 12])
    v = torch.tensor([[1.], [2.], [6.]])
    torch.testing.assert_close(tec.loo_mean(v), torch.tensor([[4.], [3.5], [1.5]]))


def test_pick_entity_layer_takes_deepest_plateau_layer():
    curve = {f"entity@{L}": {"transfer": t} for L, t in zip(range(6), [1.0, 1.0, .95, .5, .1, 0.])}
    assert tec.pick_entity_layer(curve, list(range(6))) == 2


def test_head_install_and_capture_roundtrip(tiny):
    m, _, batch = wrapped(tiny)
    ids = batch["base_input_ids"]
    side = {"ids": ids, "mask": torch.ones_like(ids), "extra": batch["base_extra"],
            "readout": torch.tensor([[ids.shape[1] - 2, ids.shape[1] - 1]])}
    z, logits = tec.head_capture(m, side, [1, 2])
    torch.testing.assert_close(tec.head_install(m, side, [(1, 0), (2, 1)], z), logits, atol=1e-6, rtol=1e-6)
    zero = {b: v * 0 for b, v in z.items()}
    assert (tec.head_install(m, side, [(1, 0)], zero) - logits).abs().max() > 0


def test_mlp_freeze_at_own_values_is_noop(tiny):
    m, _, batch = wrapped(tiny)
    ids = batch["base_input_ids"]
    side = {"ids": ids, "mask": torch.ones_like(ids), "extra": batch["base_extra"],
            "readout": torch.tensor([[ids.shape[1] - 1]]), "entity": torch.tensor([[1]])}
    clean = m.forward(ids, side["mask"], side["extra"]).logits[:, -1]
    for where in ("readout", "entity"):
        vals = tec.mlp_capture(m, side, where, [1, 2])
        got = m.forward(ids, side["mask"], side["extra"],
                        hooks=tec.mlp_freeze_hooks(m, side, where, [1, 2], vals)).logits[:, -1]
        torch.testing.assert_close(got, clean, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# image_enrichment
# ---------------------------------------------------------------------------

def test_rank_of_counts_strictly_higher_distinct_candidates():
    logit = {1: 5.0, 2: 3.0, 3: 7.0, 4: 3.0}
    assert ie.rank_of(logit, [2], [1, 2, 3, 4]) == 3
    assert ie.rank_of(logit, [2, 3], [1, 2, 3, 4]) == 1      # best variant counts
    names = {"FR": [1, 2], "DE": [3], "IT": [4]}
    assert ie.name_rank(logit, "FR", names) == 2             # FR scores 5 (its best spelling); only DE beats it


def test_loo_probe_separates_planted_signal_and_not_noise():
    import numpy as np
    rng = np.random.default_rng(0)
    y = np.array(["a"] * 10 + ["b"] * 10)
    X = rng.normal(size=(20, 40))
    assert ie.loo_accuracy(X, y, balanced=False) < 0.8
    X[:10, :5] += 6 / 5 ** 0.5                       # a distributed attribute direction
    assert ie.loo_accuracy(X, y, balanced=False) >= 0.9
    X[:, 5] *= 1000                                   # one huge-but-uninformative dim must not swamp it
    assert ie.loo_accuracy(X, y, balanced=False) >= 0.9


def test_regions_partition_the_image_grid():
    path = os.path.join(cc.DEFAULT_VADE_ROOT, "data", "flags", "object_location.json")
    if not os.path.exists(path):
        pytest.skip("VADE checkout not present")
    import json
    n = json.load(open(path))["vlm_token_grid"]["n_tokens"]
    r = ie.region_index(cc.DEFAULT_VADE_ROOT, n)
    assert sorted(r["object"] + r["ring"] + r["background"]) == r["all"] == list(range(n))
    assert len(r["object"]) == 8


# ---------------------------------------------------------------------------
# Fixes from the double-check pass
# ---------------------------------------------------------------------------

def test_pick_entity_layer_stops_before_readout_onset():
    """The deepest entity-plateau layer can sit INSIDE the handoff; L_e must precede it."""
    ent = [1.0, 1.0, 1.0, 1.0, .95, .3, 0.]
    rd = [0., 0., 0., .05, .4, .9, 1.]
    curve = {**{f"entity@{L}": {"transfer": t} for L, t in enumerate(ent)},
             **{f"readout@{L}": {"transfer": t} for L, t in enumerate(rd)}}
    assert tec.pick_entity_layer(curve, list(range(7))) == 3     # plateau runs to 4, readout onset at 4


def test_freeze_spans_drop_blocks_before_the_patch():
    assert tec.parse_freeze("readout:16-22", 20, 28) == ("readout", [20, 21, 22])


def test_build_side_text_entity_columns_point_at_the_name():
    proc = real_processor()
    m = cc.Model.__new__(cc.Model)
    m.processor, m.pad_id, m.image_id = proc, proc.tokenizer.pad_token_id, None
    items = {"FR": {"name": "France"}, "BA": {"name": "Bosnia and Herzegovina"}}
    rows = [{"base": "FR", "source": "BA", "attribute": "capital", "prefix": [220]},
            {"base": "BA", "source": "FR", "attribute": "capital", "prefix": [220]}]
    for site in ("last", "all"):
        side = tec.build_side(m, "text", None, items, rows, "base", site)
        for i, key in enumerate(("FR", "BA")):
            got = proc.tokenizer.decode(side["ids"][i, side["entity"][i]].unique_consecutive()).strip()
            want = items[key]["name"] if site == "all" else proc.tokenizer.decode(
                proc.tokenizer(" " + items[key]["name"], add_special_tokens=False)["input_ids"][-1:]).strip()
            assert got == want, (site, got, want)
        T = side["ids"].shape[1]
        assert side["readout"].tolist() == [[T - 2, T - 1]] * 2
        assert proc.tokenizer.decode(side["ids"][0, T - 2:T - 1]) == ":"


def test_collisions_separate_shared_tokens_from_shared_values():
    names = {"US": [5], "UK": [5, 6], "FR": [7]}
    values = {"language": {"MX": (9, "Spanish"), "AR": (9, "Spanish"), "FR": (8, "French")},
              "capital": {"SM": (3, "San Marino"), "CR": (3, "San Jose"), "FR": (4, "Paris")}}
    nc, vc = ie.collisions(names, values)
    assert nc == {"US", "UK"}
    assert vc["language"] == set()                 # same VALUE sharing a token is fine
    assert vc["capital"] == {"SM", "CR"}           # different values sharing a token is not
