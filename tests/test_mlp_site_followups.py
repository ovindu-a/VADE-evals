"""Offline checks for the follow-ups to the head-localization results:

    head_swap_vade  site="mlp_hidden"      the continuous patch on down_proj's input
    readout_jacobian --site mlp_hidden     factored edit operators, Jacobian at MLP neurons
    head_das        --site mlp_hidden      the trained mask on MLP neurons
    head_severed    --freeze_values        mean / third-country freezes
    mask_overlap                            cross-attribute overlap of selected units

Tiny random Qwen2.5-VL from test_head_followups (4 blocks, intermediate_size 48).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import head_das as hd
from methods import head_severed as hs
from methods import mask_overlap as mo
from methods import readout_jacobian as rj
from methods.common.targets import MAX_ANSWER_TOKENS
from methods.head_swap_vade import (capture_donor, patched_generate, plain_generate, site_module, site_units,
                                    verify_readback)

MAX_NEW, D_MLP = 4, 48
UNITS = [(1, 0), (2, 0)]          # every neuron of blocks 1 and 2


def two_rows(tiny):
    runner, batch = tiny
    ids = torch.tensor([[1, 2, 3, 12, 13, 14], [1, 2, 3, 15, 16, 17]])
    extra = {"pixel_values": torch.cat([batch["base_extra"]["pixel_values"], batch["source_extra"]["pixel_values"]]),
             "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
    return runner.adapter, runner.model, ids, torch.ones_like(ids), extra


# ---------------------------------------------------------------------------
# head_swap_vade: the mlp_hidden site
# ---------------------------------------------------------------------------

def test_site_units_widen_a_block_to_all_its_neurons(tiny):
    runner, _ = tiny
    units, dim = site_units(runner.adapter, runner.model, "mlp_hidden", blocks=[2, 1])
    assert units == [(1, 0), (2, 0)] and dim == D_MLP
    assert site_module(runner.adapter, runner.model, 1, "mlp_hidden") is runner.model.model.language_model.layers[1].mlp.down_proj
    with pytest.raises(ValueError):
        site_module(runner.adapter, runner.model, 1, "residual")


def test_mlp_hidden_patch_reads_back_and_is_live(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    blocks = [b for b, _ in UNITS]
    z = capture_donor(adapter, model, blocks, ids.flip(0), mask, extra, MAX_NEW, 0, site="mlp_hidden")
    assert z[1].shape == (2, MAX_NEW, D_MLP)
    assert verify_readback(adapter, model, UNITS, z, ids, mask, extra, MAX_NEW, 0, D_MLP, site="mlp_hidden") <= 1e-3
    # Installing each row's OWN values must reproduce the clean generation exactly.
    own = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, 0, site="mlp_hidden")
    clean = plain_generate(model, ids, mask, extra, MAX_NEW, 0)
    assert torch.equal(patched_generate(adapter, model, UNITS, own, ids, mask, extra, MAX_NEW, 0, D_MLP,
                                        site="mlp_hidden"), clean)


def test_site_kwarg_survives_extra_patches(tiny):
    """patched_generate used to loop `for site, ... in extra_patches`; the unit
    site must not be clobbered by a freeze patch's InterventionSite."""
    adapter, model, ids, mask, extra = two_rows(tiny)
    need = [("mlp", 3)]
    _, vals = hs.capture_components(adapter, model, need, ids, mask, extra, MAX_NEW, 0)
    z = capture_donor(adapter, model, [1, 2], ids, mask, extra, MAX_NEW, 0, site="mlp_hidden")
    out = patched_generate(adapter, model, UNITS, z, ids, mask, extra, MAX_NEW, 0, D_MLP,
                           extra_patches=hs.freeze_patches(need, vals), site="mlp_hidden")
    assert torch.equal(out, plain_generate(model, ids, mask, extra, MAX_NEW, 0))


# ---------------------------------------------------------------------------
# readout_jacobian: factored operators + mlp_hidden Jacobian
# ---------------------------------------------------------------------------

def _U(d, k, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return q


def test_editop_matches_the_dense_operators():
    d = 20
    U = {"a": _U(d, 3, 0), "b": _U(d, 4, 1), "c": _U(d, 2, 2)}
    ops = rj.edit_ops(U, "a", seed=5)
    P = U["a"] @ U["a"].T
    Pperp = rj.complement_projector([U["b"], U["c"]], d)
    X = np.random.default_rng(9).standard_normal((7, d))
    np.testing.assert_allclose(ops["readout"].apply(X), X @ P, atol=1e-10)
    np.testing.assert_allclose(ops["isolated"].apply(X), X @ P @ Pperp, atol=1e-10)
    np.testing.assert_allclose(ops["isolated"].apply_T(X), X @ (P @ Pperp).T, atol=1e-10)
    for arm, op in ops.items():
        dense = rj.edit_operators(U, "a", seed=5)[arm]
        assert rj.reach(X, op) == pytest.approx(rj.reach(X, dense), rel=1e-10)


def test_transform_for_accepts_editops_and_scalars():
    d = 16
    U = {"a": _U(d, 3, 0), "b": _U(d, 5, 1)}
    op = rj.edit_ops(U, "a")["isolated"]
    have, want = torch.randn(2, 1, d), torch.randn(2, 1, d)
    f_op = rj.transform_for({0: op}, "cpu")[0]
    f_dense = rj.transform_for({0: op.dense()}, "cpu")[0]
    torch.testing.assert_close(f_op(have, want), f_dense(have, want), atol=1e-5, rtol=1e-5)
    assert torch.equal(rj.transform_for({0: 0.0}, "cpu")[0](have, want), have)
    torch.testing.assert_close(rj.transform_for({0: 1.0}, "cpu")[0](have, want), want)


def test_mlp_transform_gate_on_a_real_model(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    z = capture_donor(adapter, model, [1, 2], ids.flip(0), mask, extra, MAX_NEW, 0, site="mlp_hidden")
    gen = lambda tr=None: patched_generate(adapter, model, UNITS, z, ids, mask, extra, MAX_NEW, 0, D_MLP,
                                           transform=tr, site="mlp_hidden")
    assert torch.equal(gen(rj.transform_for({1: 0.0, 2: 0.0}, model.device)),
                       plain_generate(model, ids, mask, extra, MAX_NEW, 0))
    assert torch.equal(gen(rj.transform_for({1: 1.0, 2: 1.0}, model.device)), gen())


def test_mlp_jacobian_rows_match_finite_differences(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    cands = [[10, 11, 12], [20, 21]]
    weights = [rj.probe_weights(3, 0), rj.probe_weights(2, 1)]
    rows, _ = rj.jacobian_rows(adapter, model, ids, mask, extra, UNITS, D_MLP, cands, weights, site="mlp_hidden")
    assert rows[1].shape == (2, D_MLP)

    def objective(eps):
        handles = []
        for b, e in eps.items():
            def pre(_m, args, _e=e):
                t = args[0].clone()
                t[:, -1, :] += _e.to(t.dtype)
                return (t,) + tuple(args[1:])
            handles.append(site_module(adapter, model, b, "mlp_hidden").register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids, attention_mask=mask, **extra, use_cache=False, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()
        return sum(float((lg[i, torch.tensor(c)] * torch.tensor(w, dtype=lg.dtype)).sum())
                   for i, (c, w) in enumerate(zip(cands, weights)))

    g, h = torch.Generator().manual_seed(0), 1e-3
    for b in (1, 2):
        v = torch.randn(2, D_MLP, generator=g)
        plus = {bb: (h * v if bb == b else torch.zeros(2, D_MLP)) for bb in (1, 2)}
        fd = (objective(plus) - objective({bb: -x for bb, x in plus.items()})) / (2 * h)
        assert float((rows[b] * v).sum()) == pytest.approx(fd, rel=2e-2, abs=1e-4)


# ---------------------------------------------------------------------------
# head_das: the mask trained on MLP neurons
# ---------------------------------------------------------------------------

def _das_batch(tiny):
    adapter, model, ids, mask, extra = two_rows(tiny)
    toks = torch.tensor([[30, 31, 32], [33, 34, 0]])
    lens = torch.tensor([3, 2])
    return adapter, model, {"base_ids": ids, "base_mask": mask, "base_extra": extra,
                            "donor_ids": ids.flip(0), "donor_mask": mask, "donor_extra": extra,
                            "target_toks": toks, "target_len": lens,
                            "source_gold_toks": toks.flip(0), "source_gold_len": lens.flip(0)}


def test_head_das_donor_columns_at_mlp_hidden(tiny):
    adapter, model, batch = _das_batch(tiny)
    z = hd.capture_donor_columns(adapter, model, [1, 2], batch, 0, site="mlp_hidden")
    assert z[1].shape == (2, MAX_ANSWER_TOKENS, D_MLP)
    # column 0 of the teacher-forced donor = the donor's own prefill last column
    gen = capture_donor(adapter, model, [1], batch["donor_ids"], batch["donor_mask"], batch["donor_extra"],
                        1, 0, site="mlp_hidden")
    torch.testing.assert_close(z[1][:, 0], gen[1][:, 0], atol=1e-5, rtol=1e-5)


def test_head_das_mask_at_mlp_hidden_trains_and_starts_from_the_right_place(tiny):
    from methods.dbm.intervention import SigmoidMaskIntervention
    adapter, model, batch = _das_batch(tiny)
    colmap = {b: torch.arange(D_MLP) for b in (1, 2)}
    donor = hd.capture_donor_columns(adapter, model, [1, 2], batch, 0, site="mlp_hidden")
    # A mask pinned closed must give exactly the unpatched teacher-forced logits.
    shut = {b: SigmoidMaskIntervention(embed_dim=D_MLP) for b in (1, 2)}
    for iv in shut.values():
        with torch.no_grad():
            iv.mask.fill_(-50.0)
        iv.set_temperature(torch.tensor(1.0))
    closed = hd.intervened_logits(adapter, model, shut, colmap, donor, batch, site="mlp_hidden")
    from methods.common.targets import build_teacher_forced_extension
    ext_ids, ext_mask = build_teacher_forced_extension(batch["base_ids"], batch["base_mask"],
                                                       batch["target_toks"], batch["target_len"])
    with torch.no_grad():
        ref = model(input_ids=ext_ids, attention_mask=ext_mask, **batch["base_extra"],
                    logits_to_keep=MAX_ANSWER_TOKENS).logits
    torch.testing.assert_close(closed.detach(), ref, atol=1e-5, rtol=1e-5)
    # An open mask is differentiable back to its logits.
    ivs = {b: SigmoidMaskIntervention(embed_dim=D_MLP) for b in (1, 2)}
    for iv in ivs.values():
        iv.set_temperature(torch.tensor(1.0))
    logits = hd.intervened_logits(adapter, model, ivs, colmap, donor, batch, site="mlp_hidden")
    logits[:, :, 30].sum().backward()
    assert all(float(iv.mask.grad.abs().max()) > 0 for iv in ivs.values())


def test_dbm_temperature_overrides_are_honoured():
    flat = hd.temperature_schedule_for("dbm", 5, None, 1e-2, 1e-2)
    assert torch.allclose(flat, torch.full((5,), 1e-2))
    default = hd.temperature_schedule_for("dbm", 5, None)
    assert float(default[0]) == pytest.approx(1e-2) and float(default[-1]) == pytest.approx(1e-7)


# ---------------------------------------------------------------------------
# head_severed: mean / third-country freezes
# ---------------------------------------------------------------------------

def _rows():
    out, i = [], 0
    for tid in ("t1", "t2"):
        for base, lab in [("AR", "ars"), ("BR", "brl"), ("CL", "clp"), ("DE", "eur"), ("ES", "eur"), ("FR", "eur")]:
            out.append({"row_index": i, "base": base, "source": "ZA", "base_label": lab, "source_label": "zar",
                        "template_id": tid, "queried": "currency", "target_attribute": "currency"})
            i += 1
    return out


def test_choose_thirds_excludes_base_source_and_their_labels():
    rows = _rows()
    th = hs.choose_thirds(rows, seed=0)
    for r in rows:
        t, same = th[r["row_index"]]
        assert t is not None and same and t["template_id"] == r["template_id"]
        assert t["base"] not in (r["base"], r["source"])
        assert t["base_label"] not in (r["base_label"], r["source_label"])
    assert th == hs.choose_thirds(rows, seed=0), "must be deterministic"


def test_mean_groups_are_leave_one_out_and_fall_back_when_small():
    rows = _rows()
    g = hs.mean_groups(rows, min_size=4)
    r0 = rows[0]
    assert all(rows[i]["base"] != r0["base"] and rows[i]["template_id"] == "t1" for i in g[0]) and len(g[0]) == 5
    g_small = hs.mean_groups(rows, min_size=6)          # 5 same-template others < 6 -> all templates
    assert len(g_small[0]) == 10 and all(rows[i]["base"] != "AR" for i in g_small[0])


def test_freeze_values_for_each_mode():
    rows = _rows()[:3]
    key = ("mlp", 5)
    store = {r["row_index"]: {key: torch.full((2 + r["row_index"], 3), float(r["row_index"]))} for r in rows}
    thirds = {0: (rows[2], True), 1: (rows[0], True), 2: (None, False)}
    means = hs.row_means(rows, store, {0: [1, 2], 1: [0, 2], 2: [0, 1]})
    base = hs.freeze_values_for("base", rows, store, thirds, means)[key]
    assert base.shape == (3, 4, 3) and float(base[0, -1, 0]) == 0.0 and float(base[2, -1, 0]) == 2.0
    other = hs.freeze_values_for("other", rows, store, thirds, means)[key]
    assert float(other[0, 0, 0]) == 2.0 and float(other[1, 0, 0]) == 0.0 and float(other[2, 0, 0]) == 2.0
    mean = hs.freeze_values_for("mean", rows, store, thirds, means)[key]
    assert float(mean[0, 0, 0]) == pytest.approx(1.5) and float(mean[2, 0, 0]) == pytest.approx(0.5)


def test_build_arms_repeats_freezing_arms_per_value():
    args = SimpleNamespace(freeze_mlp_span="", freeze_mlp_singles=[25], freeze_attn_span="",
                           freeze_attn_singles=[], full_image=False, freeze_values=["base", "mean", "other"])
    names = [(a[0], a[4]) for a in hs.build_arms(args, head_blocks=[21, 22, 23])]
    assert names == [("clean", None), ("heads", None), ("heads+mlp[25]", "base"),
                     ("heads+mlp[25]@mean", "mean"), ("heads+mlp[25]@other", "other")]


def test_score_and_aggregate_track_the_third_country():
    tok = SimpleNamespace()
    import methods.common.targets as targets
    orig = targets.derive_gold_token_ids
    targets.derive_gold_token_ids = lambda _t, _p, label: {"eur": [1], "zar": [2], "ars": [3]}[label]
    try:
        r = {"queried": "currency", "template_id": "t1", "source_label": "zar", "base_label": "ars"}
        lookup = {"currency": {"t1": {"prefill": ""}}}
        m = lambda text, label: label in text
        sc = hs.score(["eur."], torch.tensor([[1]]), [r], lookup, tok, m, thirds=["eur"])[0]
    finally:
        targets.derive_gold_token_ids = orig
    assert sc["third_first"] and sc["third_text"] and sc["distinct3_first"] and not sc["other_text"]
    agg = hs.aggregate([sc])
    assert agg["third_first"] == 1.0 and agg["src_first"] == 0.0


# ---------------------------------------------------------------------------
# mask_overlap
# ---------------------------------------------------------------------------

def test_overlap_table_against_chance():
    runs = {"language": {"selected": {"25": list(range(0, 10))}, "width": {"25": 100}},
            "currency": {"selected": {"25": list(range(5, 15))}, "width": {"25": 100}}}
    t = mo.overlap_table(runs)[25]
    p = t["pairs"]["currency|language"]
    assert p["intersection"] == 5 and p["chance_intersection"] == pytest.approx(1.0)
    assert p["ratio_to_chance"] == pytest.approx(5.0) and p["jaccard"] == pytest.approx(5 / 15)
