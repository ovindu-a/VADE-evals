"""Offline tests for methods/mlp_lookup.py (pure pieces) and methods/toy_lookup.py (CPU toy)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.mlp_lookup import (EXPECT, additive_loo, arm_vector, attribute_values, col_mean_excl,  # noqa: E402
                                eta_sq, row_mean_excl, sample_compose_rows, shares, summarize_compose,
                                two_way, value_groups, value_test)


def additive_grid(I=6, A=4, D=5, seed=0):
    rng = np.random.default_rng(seed)
    mu, e, q = rng.normal(size=D), rng.normal(size=(I, D)), rng.normal(size=(A, D))
    return mu + e[:, None] + q[None], e, q


def test_two_way_exact_and_additive():
    X, _, _ = additive_grid()
    ss = two_way(X)
    np.testing.assert_allclose(ss["item"] + ss["attr"] + ss["inter"], ss["total"], rtol=1e-10)
    assert shares(ss)["inter"] < 1e-12
    Y = X + np.random.default_rng(1).normal(size=X.shape)              # add a pure interaction term
    s = shares(two_way(Y))
    assert s["inter"] > 0.1 and abs(sum(s.values()) - 1) < 1e-9


def test_additive_loo_recovers_cell_without_seeing_it():
    X, _, _ = additive_grid()
    for i, a in [(0, 0), (3, 2), (5, 3)]:
        Y = X.copy()
        Y[i, a] = 1e6                                    # poison the target cell: must not be used
        np.testing.assert_allclose(additive_loo(Y, i, a), X[i, a], atol=1e-9)


def test_arm_vectors_on_additive_grid():
    X, _, _ = additive_grid()
    c, x, b, a = 0, 1, 2, 3
    np.testing.assert_allclose(arm_vector("actual", X, b, a, c, x), X[b, a])
    np.testing.assert_allclose(arm_vector("additive", X, b, a, c, x), X[b, a], atol=1e-9)
    np.testing.assert_allclose(arm_vector("swap_entity", X, b, a, c, x), X[b, x], atol=1e-9)
    np.testing.assert_allclose(arm_vector("swap_question", X, b, a, c, x), X[c, a], atol=1e-9)
    r = arm_vector("random_offset", X, b, a, c, x, np.random.default_rng(0))
    assert np.isclose(np.linalg.norm(r - X[c, x]), np.linalg.norm(X[b, a] - X[c, x]))
    # The swaps leave out the target cell too.
    Y = X.copy()
    Y[b, x] = 1e6
    np.testing.assert_allclose(row_mean_excl(Y, b, x), X[b, [0, 2, 3]].mean(0))
    Y = X.copy()
    Y[c, a] = 1e6
    assert np.all(np.abs(col_mean_excl(Y, c, a)) < 1e5)
    assert set(EXPECT) == {"clean", "actual", "additive", "swap_entity", "swap_question", "random_offset"}


def test_sample_compose_rows_requires_four_distinct_correct_cells():
    I, A = 4, 3
    first = {(i, a): 10 * i + a for i in range(I) for a in range(A)}
    correct = {cell: True for cell in first}
    correct[(1, 1)] = False
    rows = sample_compose_rows(list(range(I)), list(range(A)), first, correct, 1000, 0)
    assert rows and all(c != b and x != a for c, x, b, a in rows)
    for c, x, b, a in rows:
        assert (1, 1) not in [(b, a), (c, x), (b, x), (c, a)]
    first[(2, 0)] = first[(3, 0)]                         # collide two first tokens
    rows2 = sample_compose_rows(list(range(I)), list(range(A)), first, correct, 1000, 0)
    assert len(rows2) < len(rows)


def test_value_groups_and_eta():
    kept, labels = value_groups(["a", "b", "a", "c", "b"])
    assert kept == [0, 1, 2, 4] and labels == ["a", "b", "a", "b"]
    Y = np.array([[1.0], [5.0], [1.0], [5.0]])
    assert np.isclose(eta_sq(Y, labels)[0], 1.0)


def test_value_test_finds_answer_keyed_neurons():
    rng = np.random.default_rng(0)
    I, A, D = 40, 3, 20
    values = [str(i % 4) for i in range(I)]
    Y = rng.normal(size=(I, A, D))
    Y[:, 1, :10] += 5 * np.array([int(v) for v in values])[:, None]   # keyed by value ONLY when attr 1 is asked
    res = value_test(Y, values, 1, list(range(10)), n_perm=50, seed=0)
    assert res["mean_diff"] > 0.5 and res["p_value"] < 0.05
    null = value_test(Y, values, 0, list(range(10, 20)), n_perm=50, seed=0)
    assert abs(null["mean_diff"]) < 0.2


def test_attribute_values():
    items = {"FR": {"currency": "EUR", "calling_code": "33"}, "JP": {"currency": "JPY", "calling_code": "81"}}
    assert attribute_values(items, ["FR", "JP"], "calling_code") == ["3", "8"]
    assert attribute_values(items, ["FR", "JP"], "currency") == ["EUR", "other"]


def test_summarize_compose():
    rows = [{"arm": "additive", "layer": 24, "first_hit": "B.A", "text_hit": "B.A"},
            {"arm": "additive", "layer": 24, "first_hit": "C.X", "text_hit": "other"},
            {"arm": "clean", "layer": 24, "first_hit": "C.X", "text_hit": "C.X"}]
    s = summarize_compose(rows)
    assert s["24"]["additive"]["first_B.A"] == 0.5 and s["24"]["additive"]["text_other"] == 0.5
    assert s["24"]["clean"]["expect"] == "C.X" and list(s["24"]) == ["clean", "additive"]


def test_toy_reproduces_move_then_lookup():
    pytest.importorskip("torch")
    from methods.toy_lookup import analyze, make_task, train
    task = make_task(n_entities=24, n_relations=3, n_values=6, seed=0)
    model, _ = train(task, steps=1500, seed=0)
    r = analyze(task, model, n_pairs=128, seed=0)
    assert r["accuracy"] > 0.95
    roles = sorted(h["role"] for h in r["heads"])
    assert roles == ["mover", "router"]
    assert r["resid_shares"]["inter"] < 0.05                     # additive into the MLP
    assert r["answer_paths"]["mlp_write_only"] > r["answer_paths"]["residual_only (no MLP)"]
    assert r["compose_additive_acc"] > 0.8
    assert r["cross_question"].get("B's asked attribute (B.A)", 0) > 0.8


def test_capture_compose_neurons_end_to_end_on_tiny_qwen(tmp_path, monkeypatch):
    """The model side on the tiny random Qwen2.5-VL: hooks fire once, arrays fill, the patch is
    live (actual != clean somewhere), and the CPU analysis runs on what capture wrote."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from types import SimpleNamespace
    from test_head_followups import tiny as tiny_fixture
    import methods.circuit_common as cc
    import methods.common.text_match as tm
    from methods import mlp_lookup as ml

    runner, batch = tiny_fixture.__wrapped__()
    items = {f"K{i}": {"name": f"n{i}", "capital": f"c{i}", "currency": "EUR" if i % 2 else f"u{i}",
                       "language": f"l{i % 2}", "calling_code": str(10 + i)} for i in range(4)}
    keys, attrs = sorted(items), ["capital", "currency", "language", "calling_code"]
    label_id = {}

    class Tok:
        pad_token_id, eos_token_id = 0, None

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"t{int(t)}" for t in ids)

    m = cc.Model.wrap(runner.adapter, runner.model, SimpleNamespace(tokenizer=Tok()))

    def image_prompt(entity_dir, item, attribute):
        g = torch.Generator().manual_seed(sum(map(ord, item["name"] + attribute)))
        q = 12 + attrs.index(attribute)                        # the question token differs per attribute
        return (torch.tensor([1, 2, 3, q, 13, 14]),
                {"pixel_values": torch.randn(4, 24, generator=g), "image_grid_thw": torch.tensor([[1, 2, 2]])})

    monkeypatch.setattr(m, "image_prompt", image_prompt)
    monkeypatch.setattr(cc, "gold_ids", lambda tok, label: [label_id.setdefault(label, 5 + len(label_id))])
    monkeypatch.setattr(tm, "contains_label", lambda text, label: True)

    meta = {"resid_layers": [2, 3, 4], "mlp_blocks": [1, 2, 3]}
    index = ml.run_capture(m, None, items, keys, attrs, meta["resid_layers"], meta["mlp_blocks"], 5, 3, str(tmp_path))
    assert len(index) == 16
    for name, width in (("resid", 32), ("mlp_hidden", 48), ("mlp_output", 32)):
        arr = np.load(tmp_path / f"{name}.npy")
        assert arr.shape == (16, 3, width) and np.isfinite(arr).all() and np.abs(arr).sum() > 0
    # Captured residual at layer L must equal the model's own hidden_states[L] at the last column.
    ids, extra = image_prompt(None, items[keys[0]], attrs[0])
    out = m.forward(ids[None], torch.ones(1, 6, dtype=torch.long), extra, output_hidden_states=True)
    np.testing.assert_allclose(np.load(tmp_path / "resid.npy")[0, 0], out.hidden_states[2][0, -1].numpy(),
                               rtol=1e-4, atol=1e-5)

    summary = ml.run_compose(m, None, items, keys, attrs, meta, [2, 4], 12, 4, 3, 0, str(tmp_path))
    assert set(summary) == {"2", "4"} and set(summary["2"]) == set(ml.ARMS)
    rows = [json.loads(l) for l in open(tmp_path / "compose_rows.jsonl")]
    by = {(r["layer"], r["arm"], r["C"], r["X"], r["B"], r["A"]): r["generated"] for r in rows}
    assert any(by[(L, "actual") + k[2:]] != g for k, g in by.items() if k[1] == "clean" for L in (k[0],)), \
        "installing another cell's residual never changed a generation -- is the patch connected?"

    rep = ml.analyze_neurons(str(tmp_path), items, keys, attrs, meta, top_k=10, heatmap_block=2, n_perm=10, seed=0)
    assert set(rep["mlp_hidden_shares"]) == {"1", "2", "3"}
    assert abs(sum(rep["resid_shares"]["3"].values()) - 1) < 1e-6
    assert (tmp_path / "heatmap.npz").exists()
