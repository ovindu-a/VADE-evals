"""Offline checks for the three head-localization experiments:

    head_vocab_projection.py   per-head vocab projection + source-position split
    head_severed.py            ROME-style freeze of downstream components
    readout_jacobian.py        readout-Jacobian geometry + VADE-scored edits

Each failure mode these guard against -- a group split that does not sum to the
head's write, a freeze that lands on the wrong column or step, a Jacobian that is
not the derivative it claims to be -- produces plausible numbers rather than an
error. Tiny random Qwen2.5-VL from test_head_followups; no weights, no data.
"""
import numpy as np
import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import head_severed as hs
from methods import head_vocab_projection as hvp
from methods import readout_jacobian as rj
from methods.head_swap_vade import capture_donor, head_columns, patched_generate, plain_generate

HEAD_DIM, MAX_NEW = 8, 4
HEADS = [(1, 0), (2, 3)]


def pieces(tiny):
    runner, batch = tiny
    return runner.adapter, runner.model, batch["base_input_ids"], batch["attention_mask"], batch["base_extra"]


def two_rows(tiny):
    """A 2-row batch with DIFFERENT texts and images, same length (no padding)."""
    runner, batch = tiny
    ids = torch.tensor([[1, 2, 3, 12, 13, 14], [1, 2, 3, 15, 16, 17]])
    extra = {"pixel_values": torch.cat([batch["base_extra"]["pixel_values"], batch["source_extra"]["pixel_values"]]),
             "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
    return runner.adapter, runner.model, ids, torch.ones_like(ids), extra


# ---------------------------------------------------------------------------
# head_vocab_projection
# ---------------------------------------------------------------------------

def test_token_groups_partition_the_prompt():
    #        sys sys img img img txt txt im_start asst prefill
    ids = [9, 9, 2, 2, 2, 7, 7, 50, 51, 52]
    g = hvp.token_groups(ids, image_token_id=2, im_start_id=50, footprint={1})
    assert g == ["system", "system", "image_background", "object", "image_background",
                 "question", "question", "assistant_prefix", "assistant_prefix", "assistant_prefix"]


def test_candidate_rank_is_over_distinct_tokens_and_takes_the_best_variant():
    logits = torch.tensor([0., 5., 3., 3., 1.])
    r = hvp.candidate_rank(logits, own_ids=[2, 4], cand_ids=[1, 2, 2, 3, 4])
    assert r["n"] == 4 and r["rank"] == 2          # only token 1 is strictly above 3.0
    assert hvp.candidate_rank(logits, [1], [1, 2, 3])["rank"] == 1


def test_classify_top_priority():
    own_vals = {"capital": {30}, "language": {40}}
    all_vals = {"capital": {30, 31}, "language": {40, 41}}
    c = lambda t: hvp.classify_top(t, "capital", {10}, own_vals, {10, 11}, all_vals)
    assert (c(30), c(10), c(40), c(11), c(41), c(99)) == (
        "own_value_queried", "own_entity", "own_value_other_attr", "other_entity", "other_value", "other")


def test_group_split_sums_to_the_head_write():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(7), 0)
    v, w_o = torch.randn(7, 8), torch.randn(16, 8)
    gi = torch.tensor([0, 1, 1, 2, 3, 3, 4])
    z, per = hvp.split_head_by_groups(w, v, w_o, gi, 5)
    assert torch.allclose(z, w @ v)
    assert torch.allclose(per.sum(0), z @ w_o.T, atol=1e-5)


def test_analyze_heads_on_a_real_forward(tiny):
    """The asserted self-checks (A@V reconstruction, frozen-scale logit) must pass
    on a real eager forward, and the per-group DLA must sum to the head's."""
    adapter, model, ids, _, extra = pieces(tiny)
    store = hvp.capture_forward(adapter, model, ids, extra, blocks=[1, 2])
    groups = ["system", "object", "question", "question", "assistant_prefix", "assistant_prefix"]
    cands = {"entity": ([10], [10, 11, 20]), "queried": "capital",
             "values": {"capital": ([30], [30, 31, 32]), "language": ([40], [40, 41])}}
    recs = hvp.analyze_heads(adapter, model, store, HEADS, HEAD_DIM, groups, cands, top_k=5)
    assert set(recs) == {"1.0", "2.3"}
    for rec in recs.values():
        assert rec["reconstruction_rel_err"] < 1e-4
        assert sum(g["dla_entity"] for g in rec["groups"].values()) == pytest.approx(rec["dla_entity"], abs=1e-4)
        assert sum(g["attention"] for g in rec["groups"].values()) == pytest.approx(1.0, abs=1e-5)
        assert len(rec["top"]) == 5 and rec["entity"]["n"] == 3


def test_analyze_heads_catches_a_wrong_value_head(tiny):
    adapter, model, ids, _, extra = pieces(tiny)
    store = hvp.capture_forward(adapter, model, ids, extra, blocks=[1, 2])
    # The GQA failure mode: values read from the NEIGHBOURING kv head.
    store["v"][1] = store["v"][1].roll(HEAD_DIM, dims=-1)
    groups = ["system", "object", "question", "question", "assistant_prefix", "assistant_prefix"]
    cands = {"entity": ([10], [10, 11]), "queried": "capital", "values": {"capital": ([30], [30, 31])}}
    with pytest.raises(AssertionError, match="A@V"):
        hvp.analyze_heads(adapter, model, store, [(1, 0)], HEAD_DIM, groups, cands, top_k=3)


# ---------------------------------------------------------------------------
# head_severed
# ---------------------------------------------------------------------------

NEED = [("mlp", 1), ("mlp", 2), ("attn", 3)]


def test_parse_span():
    assert hs.parse_span("22-27") == [22, 23, 24, 25, 26, 27] and hs.parse_span("24") == [24]


def test_freezing_at_own_base_values_is_bit_exact(tiny):
    """The identity gate the real run opens with."""
    adapter, model, ids, mask, extra = pieces(tiny)
    clean, vals = hs.capture_components(adapter, model, NEED, ids, mask, extra, MAX_NEW, 0)
    assert torch.equal(clean, plain_generate(model, ids, mask, extra, MAX_NEW, 0))
    assert all(v.shape[1] == MAX_NEW for v in vals.values()), "one capture per forward expected"
    frozen = patched_generate(adapter, model, [], {}, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                              extra_patches=hs.freeze_patches(NEED, vals))
    assert torch.equal(frozen, clean)


def test_freezing_to_other_values_moves_the_logits(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    _, vals = hs.capture_components(adapter, model, NEED, ids, mask, extra, MAX_NEW, 0)
    ref = hs.last_logits(adapter, model, ids, mask, extra)
    rolled = {k: v.roll(1, dims=0) for k, v in vals.items()}
    moved = hs.last_logits(adapter, model, ids, mask, extra, hs.freeze_patches(NEED, rolled))
    assert float((moved - ref).abs().max()) > 1e-3


def test_self_checks_pass_end_to_end(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    batch = {"base_ids": ids, "base_mask": mask, "base_extra": extra,
             "donor_ids": ids.flip(0), "donor_mask": mask, "donor_extra": extra}
    arms = [("clean", False, False, []), ("heads", True, False, []),
            ("heads+mlp[3]", True, False, [("mlp", 3)]), ("heads+attn[3]", True, False, [("attn", 3)])]
    rep = hs.self_checks(adapter, model, batch, HEADS, HEAD_DIM, arms, MAX_NEW, 0, "all")
    assert rep["identity"] == "bit-exact" and rep["liveness_max_logit_delta"] > 1e-3
    assert rep["readback_worst"] <= 1e-3


def test_freeze_writes_the_last_column_at_the_matching_step():
    vals = {("mlp", 0): torch.arange(3.).view(1, 3, 1).expand(1, 3, 2).clone()}      # step t -> value t
    (_, _, fn), = hs.freeze_patches([("mlp", 0)], vals)
    prefill = fn(torch.full((1, 5, 2), -1.))
    assert torch.equal(prefill[0, :-1], torch.full((4, 2), -1.)) and torch.equal(prefill[0, -1], torch.zeros(2))
    assert float(fn(torch.full((1, 1, 2), -1.))[0, -1, 0]) == 1.0
    assert float(fn(torch.full((1, 1, 2), -1.))[0, -1, 0]) == 2.0
    assert float(fn(torch.full((1, 1, 2), -1.))[0, -1, 0]) == 2.0, "the last base step should be held"
    (_, _, first), = hs.freeze_patches([("mlp", 0)], vals, freeze_steps="first")
    first(torch.zeros(1, 5, 2))
    assert torch.equal(first(torch.full((1, 1, 2), -1.)), torch.full((1, 1, 2), -1.))


def test_attention_freeze_inside_the_head_blocks_is_refused():
    from types import SimpleNamespace
    args = SimpleNamespace(freeze_mlp_span="", freeze_mlp_singles=[], freeze_attn_span="22-24",
                           freeze_attn_singles=[], full_image=False)
    with pytest.raises(AssertionError, match="overwrite the install"):
        hs.build_arms(args, head_blocks=[21, 22, 23])


def test_aggregate_restricts_first_token_rates_to_distinct_golds():
    rows = [dict(distinct_first=True, src_first=True, base_first=False, src_text=True, base_text=False,
                 other_text=False),
            dict(distinct_first=False, src_first=True, base_first=True, src_text=False, base_text=True,
                 other_text=False)]
    a = hs.aggregate(rows)
    assert a["n_distinct_first"] == 1 and a["src_first"] == 1.0 and a["base_first"] == 0.0
    assert a["src_text"] == 0.5


# ---------------------------------------------------------------------------
# readout_jacobian
# ---------------------------------------------------------------------------

def _readouts(dims_by_attr, d=12, n=40, seed=0):
    rng = np.random.default_rng(seed)
    G = {}
    for a, dims in dims_by_attr.items():
        M = np.zeros((n, d))
        M[:, dims] = rng.standard_normal((n, len(dims)))
        G[a] = M
    return G


def test_disjoint_readouts_give_a_selective_isolated_edit():
    G = _readouts({"a": [0, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]})
    U, rep = rj.block_geometry(G, energy=0.999, max_rank=6)
    assert rep["rank"] == {"a": 3, "b": 3, "c": 3}
    assert max(rep["cross"].values()) < 1e-10
    row = rep["reach"]["a"]["isolated"]
    assert row["a"] == pytest.approx(1.0) and row["b"] < 1e-10 and row["c"] < 1e-10
    assert rj.selectivity(row, "a") == float("inf")


def test_identical_readouts_leave_nothing_to_isolate():
    G = _readouts({"a": [0, 1, 2], "b": [0, 1, 2]})
    G["b"] = G["a"][::-1].copy()
    _, rep = rj.block_geometry(G, energy=0.999, max_rank=6)
    assert rep["cross"]["a->b"] == pytest.approx(1.0)
    assert rep["reach"]["a"]["isolated"]["a"] < 1e-10, "isolation must remove all of a shared readout"
    assert rep["reach"]["a"]["readout"]["b"] == pytest.approx(1.0), "the unisolated edit leaks fully"


def test_partial_overlap_is_removed_only_where_shared():
    G = _readouts({"a": [0, 1, 2, 3], "b": [2, 3, 4, 5]})
    _, rep = rj.block_geometry(G, energy=0.999, max_rank=6)
    assert 0.2 < rep["cross"]["a->b"] < 0.8
    iso = rep["reach"]["a"]["isolated"]
    assert iso["b"] < 1e-10 and 0.2 < iso["a"] < 0.8


def test_das_control_reports_readout_energy_in_the_core():
    G = _readouts({"a": [0, 1], "b": [2, 3]})
    U, _ = rj.block_geometry(G, energy=0.999, max_rank=4)
    V = np.eye(12)[[0, 1]]                                        # the core IS a's readout
    dc = rj.das_control(G, U, V)
    assert dc["readout_energy_in_core"]["a"] == pytest.approx(1.0)
    assert dc["readout_energy_in_core"]["b"] < 1e-10
    assert dc["core_in_union_readout"] == pytest.approx(1.0)


def test_fit_units_are_reproducible_across_processes():
    lookup = {"capital": {"capital_prefill_v1": {}, "capital_prefill_v5": {}}}
    u1 = rj.fit_units(["AR", "BR"], ["capital"], lookup, ["v1"], probes=2, seed=0)
    assert u1 == rj.fit_units(["AR", "BR"], ["capital"], lookup, ["v1"], probes=2, seed=0)
    assert len(u1) == 4 and {t for _, _, t, _ in u1} == {"capital_prefill_v1"}
    # crc32 is a fixed function of the string, unlike the salted builtin hash().
    import zlib
    assert u1[0][3] == zlib.crc32(b"0|capital|AR|capital_prefill_v1|0")


def test_probe_weights_are_zero_mean_unit_norm():
    w = rj.probe_weights(9, seed=3)
    assert abs(w.sum()) < 1e-12 and np.linalg.norm(w) == pytest.approx(1.0)


def _objective(adapter, model, ids, mask, extra, cands, weights, eps):
    """sum_i w_i . logits_i[cand_i] with a FIXED perturbation eps[b] added to the
    selected heads' last column -- the finite-difference reference."""
    handles = []
    for b in sorted(eps):
        cols = head_columns(HEADS, b, HEAD_DIM)

        def pre(_m, args, _cols=cols, _e=eps[b]):
            t = args[0].clone()
            t[:, -1, _cols] += _e.to(t.dtype)
            return (t,) + tuple(args[1:])
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, **extra, use_cache=False, logits_to_keep=1).logits[:, -1]
    finally:
        for h in handles:
            h.remove()
    return sum(float((logits[i, torch.tensor(c)] * torch.tensor(w, dtype=logits.dtype)).sum())
               for i, (c, w) in enumerate(zip(cands, weights)))


def test_jacobian_rows_match_finite_differences(tiny):
    """The sketch row for batch row i must be d(w_i . logits_i[cand_i]) / d z_b at
    row i's last column -- checked per row and per block against central
    differences in float32."""
    adapter, model, ids, mask, extra = two_rows(tiny)
    cands = [[10, 11, 12], [20, 21]]
    weights = [rj.probe_weights(3, 0), rj.probe_weights(2, 1)]
    rows, _ = rj.jacobian_rows(adapter, model, ids, mask, extra, HEADS, HEAD_DIM, cands, weights)
    assert set(rows) == {1, 2} and rows[1].shape == (2, 8) and rows[2].shape == (2, 8)
    g = torch.Generator().manual_seed(0)
    h = 1e-3
    for b in (1, 2):
        v = torch.randn(2, 8, generator=g)
        plus = {bb: (h * v if bb == b else torch.zeros(2, 8)) for bb in (1, 2)}
        minus = {bb: -x for bb, x in plus.items()}
        fd = (_objective(adapter, model, ids, mask, extra, cands, weights, plus)
              - _objective(adapter, model, ids, mask, extra, cands, weights, minus)) / (2 * h)
        assert float((rows[b] * v).sum()) == pytest.approx(fd, rel=2e-2, abs=1e-4)


def test_jacobian_rows_are_independent_per_batch_row(tiny):
    """Row i's gradient must not pick up row j's objective: zero row 1's weights
    and its sketch row must be exactly zero."""
    adapter, model, ids, mask, extra = two_rows(tiny)
    rows, _ = rj.jacobian_rows(adapter, model, ids, mask, extra, HEADS, HEAD_DIM,
                               [[10, 11], [10, 11]], [rj.probe_weights(2, 0), np.zeros(2)])
    for b in rows:
        assert float(rows[b][1].abs().max()) == 0.0 and float(rows[b][0].abs().max()) > 0.0


def test_transform_identity_and_zero_reproduce_full_patch_and_clean(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids.flip(0), mask, extra, MAX_NEW, 0)
    full = patched_generate(adapter, model, HEADS, z, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    width = {b: len(head_columns(HEADS, b, HEAD_DIM)) for b in blocks}
    ident = patched_generate(adapter, model, HEADS, z, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                             transform=rj.transform_for({b: np.eye(width[b]) for b in blocks}, model.device))
    zero = patched_generate(adapter, model, HEADS, z, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                            transform=rj.transform_for({b: np.zeros((width[b], width[b])) for b in blocks},
                                                       model.device))
    assert torch.equal(ident, full)
    assert torch.equal(zero, plain_generate(model, ids, mask, extra, MAX_NEW, 0))


def test_quick_scores_follow_vade_shape():
    rows = {("language", 0): {"rule": "match_source", "queried": "language", "source_label": "Spanish",
                              "base_label": "French"},
            ("language", 1): {"rule": "match_base", "queried": "capital", "source_label": "Lima",
                              "base_label": "Paris"},
            ("language", 2): {"rule": "match_base", "queried": "currency", "source_label": "PEN",
                              "base_label": "EUR"}}
    preds = {("language", 0): "Spanish.", ("language", 1): "Paris.", ("language", 2): "PEN."}
    m = lambda text, label: label.lower() in text.lower()
    s = rj.quick_scores(preds, rows, m)
    assert s["cause"] == 1.0 and s["iso_by_queried"] == {"capital": 1.0, "currency": 0.0}
    assert s["final"] == pytest.approx(0.75)


def test_held_out_measurement_deflates_a_noise_fitted_subspace():
    """Pure-noise readouts: in-sample, each attribute's top-k subspace captures
    far more than k/d of its own energy (it fits the sketch's noise); held out,
    that collapses to ~chance. The held-out number is the one reported."""
    rng = np.random.default_rng(0)
    G = {a: rng.standard_normal((20, 64)) for a in ("a", "b")}
    H = {a: rng.standard_normal((20, 64)) for a in ("a", "b")}
    _, ins = rj.block_geometry(G, energy=0.5, max_rank=8)
    _, out = rj.block_geometry(G, energy=0.5, max_rank=8, G_eval=H)
    assert ins["self_energy"]["a"] > 2 * out["self_energy"]["a"]
    assert out["self_energy"]["a"] == pytest.approx(8 / 64, abs=0.06)
    assert out["measured_on"] == "held-out items"


def test_item_folds_never_split_an_item():
    row_items = {"a": ["AR", "AR", "BR", "CL", "CL", "DE"], "b": ["AR", "BR", "BR", "CL", "DE", "DE"]}
    f = rj.item_folds(row_items, ["a", "b"])
    assert f["items"][0].isdisjoint(f["items"][1])
    for a, (i0, i1) in f["rows"].items():
        assert {row_items[a][i] for i in i0} <= f["items"][0] and {row_items[a][i] for i in i1} <= f["items"][1]
        assert sorted(list(i0) + list(i1)) == list(range(len(row_items[a])))


def test_average_reports_means_floats_and_keeps_ranks_per_fold():
    r1 = {"d": 4, "rank": {"a": 2}, "cross": {"a->b": 0.2}, "spectrum": {"a": [1.0]}}
    r2 = {"d": 4, "rank": {"a": 3}, "cross": {"a->b": 0.4}, "spectrum": {"a": [0.5]}}
    out = rj.average_reports([r1, r2])
    assert out["cross"]["a->b"] == pytest.approx(0.3) and out["rank_by_fold"] == [{"a": 2}, {"a": 3}]
