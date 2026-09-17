"""The direction/variance analysis, checked against grids with a PLANTED answer.

No model and no GPU: every statistic here is linear algebra over a synthetic
[items x attributes] grid built so the correct output is known in advance. That
is the only way to catch a decomposition that is self-consistent but wrong.
"""
import json

import numpy as np
import pytest

from methods.attr_directions import (geometry, load, separability, variance_decomposition,
                                     head_slices)

N_ITEMS, N_ATTR, WIDTH, N_HEADS, N_BLOCKS = 12, 4, 64, 4, 2
HEAD_DIM = WIDTH // N_HEADS


def grid(rng, item=1.0, attribute=0.0, interaction=0.0, noise=0.0):
    """X[c,a] = item_c*item + attr_a*attribute + (item_c ⊙ attr_a)*interaction + noise."""
    items = rng.normal(size=(N_ITEMS, WIDTH))
    attrs = rng.normal(size=(N_ATTR, WIDTH))
    X = item * items[:, None, :] + attribute * attrs[None]
    if interaction:
        X = X + interaction * (items[:, None, :] * attrs[None])
    return X + noise * rng.normal(size=X.shape)


def write_capture(tmp_path, arrays, items=None, attributes=None, shuffle=False):
    """arrays: {site: [n_items, n_attr, n_blocks, width]} -> a capture directory.

    Rows are optionally written in a SHUFFLED order so the loader has to use
    index.jsonl rather than assume the grid layout.
    """
    items = items or [f"I{i}" for i in range(N_ITEMS)]
    attributes = attributes or [f"a{j}" for j in range(N_ATTR)]
    order = [(i, j) for i in range(len(items)) for j in range(len(attributes))]
    if shuffle:
        np.random.default_rng(0).shuffle(order)
    meta = {"experiment": "attr_capture", "entity": "test", "model_id": "m", "attributes": attributes,
            "items": items, "blocks": list(range(N_BLOCKS)), "sites": sorted(arrays),
            "positions": "last_token", "grid_order": "item-major, attribute-minor",
            "shape": [len(order), N_BLOCKS, 1, WIDTH], "dtype": "float32",
            "n_heads": N_HEADS, "head_dim": HEAD_DIM, "prefill": "Answer:", "prompts": {},
            "head_proj_norms": {str(b): [1.0] * N_HEADS for b in range(N_BLOCKS)},
            "ground_truth_sha256": "x", "implementation_sha256": "y"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "index.jsonl").write_text("".join(
        json.dumps({"row": r, "item": items[i], "attribute": attributes[j], "condition": "clean",
                    "prompt_length": 9, "gold_ids": [1]}) + "\n" for r, (i, j) in enumerate(order)))
    for site, X in arrays.items():
        out = np.zeros((len(order), N_BLOCKS, 1, WIDTH), dtype=np.float32)
        for r, (i, j) in enumerate(order):
            out[r, :, 0, :] = X[i, j]
        np.save(tmp_path / f"acts_{site}.npy", out)
    return tmp_path


def test_a_planted_attribute_direction_is_recovered():
    rng = np.random.default_rng(0)
    X = grid(rng, item=1.0, attribute=1.0, noise=0.05)
    s = separability(X)
    assert s["loo_accuracy"] == 1.0
    assert s["attr_var_explained"] > 0.95
    g = geometry(X)
    # Four planted-random directions, so after centering they should be near-orthogonal and use the
    # full A-1 dimensions the centering leaves available.
    off = [abs(g["cosines"][i][j]) for i in range(N_ATTR) for j in range(N_ATTR) if i != j]
    assert max(off) < 0.5
    assert g["effective_rank"] > N_ATTR - 1.3


def test_no_attribute_effect_reads_as_chance():
    rng = np.random.default_rng(1)
    X = grid(rng, item=1.0, attribute=0.0, noise=0.3)
    s = separability(X)
    assert s["attr_var_explained"] < 0.2
    assert abs(s["loo_accuracy"] - s["chance"]) < 0.2


def test_item_dominance_does_not_leak_into_the_attribute_statistic():
    """A huge item effect must NOT inflate attribute separability -- the centering
    is the whole method, so this is the test that it is actually applied."""
    rng = np.random.default_rng(2)
    small = separability(grid(rng, item=1.0, attribute=0.5, noise=0.05))
    huge = separability(grid(np.random.default_rng(2), item=50.0, attribute=0.5, noise=0.05))
    assert abs(small["attr_var_explained"] - huge["attr_var_explained"]) < 1e-6
    assert small["loo_accuracy"] == huge["loo_accuracy"]


def test_variance_decomposition_matches_the_planted_split():
    rng = np.random.default_rng(3)
    pure_item = variance_decomposition(grid(rng, item=1.0, attribute=0.0))
    assert pure_item["item"] > 0.99 and pure_item["interaction"] < 0.01
    pure_attr = variance_decomposition(grid(np.random.default_rng(3), item=0.0, attribute=1.0))
    assert pure_attr["attribute"] > 0.99
    both = variance_decomposition(grid(np.random.default_rng(3), item=1.0, attribute=1.0))
    assert 0.4 < both["item"] < 0.6 and 0.4 < both["attribute"] < 0.6
    assert both["interaction"] < 0.02
    # The item-4 signature: the head reads the SAME image differently per question.
    # An EXACT pure interaction has zero row and column means by construction; the
    # elementwise item*attribute product below does not, which is why it splits.
    rng2 = np.random.default_rng(7)
    g = rng2.normal(size=(N_ITEMS, N_ATTR, WIDTH))
    g = g - g.mean(axis=0, keepdims=True) - g.mean(axis=1, keepdims=True) + g.mean(axis=(0, 1))
    inter = variance_decomposition(g)
    assert inter["interaction"] > 0.999
    # The realistic multiplicative case still lands the MAJORITY in interaction, which is what the
    # image-head reading relies on -- it does not need a pure term, only a dominant one.
    mult = variance_decomposition(grid(np.random.default_rng(3), item=0.0, attribute=0.0,
                                       interaction=1.0))
    assert mult["interaction"] > 0.7 and mult["interaction"] > mult["item"] + mult["attribute"]
    for d in (pure_item, pure_attr, both, inter, mult):
        assert abs(d["item"] + d["attribute"] + d["interaction"] - 1.0) < 1e-9


def test_the_one_head_carrying_the_attribute_ranks_first():
    rng = np.random.default_rng(4)
    X = grid(rng, item=1.0, attribute=0.0, noise=0.05)
    planted = 2
    attrs = rng.normal(size=(N_ATTR, HEAD_DIM)) * 3.0
    X[:, :, head_slices(WIDTH, N_HEADS)[planted]] += attrs[None]
    scores = [(separability(X[:, :, sl])["attr_var_explained"], h)
              for h, sl in enumerate(head_slices(WIDTH, N_HEADS))]
    assert max(scores)[1] == planted
    assert max(scores)[0] > 0.9


def test_loader_uses_the_index_and_rejects_a_broken_grid(tmp_path):
    rng = np.random.default_rng(5)
    X = grid(rng, item=1.0, attribute=1.0, noise=0.01)
    d = write_capture(tmp_path, {"residual": X, "attn_head_output": X}, shuffle=True)
    meta, data = load(d)
    assert set(data) == {"residual", "attn_head_output"}
    assert data["residual"].shape == (N_ITEMS, N_ATTR, N_BLOCKS, WIDTH)
    # Shuffled on disk, so equality here proves the index (not the file order) placed the cells.
    np.testing.assert_allclose(data["residual"][:, :, 0, :], X, rtol=0, atol=1e-5)
    assert separability(data["residual"][:, :, 0, :])["loo_accuracy"] == 1.0

    lines = (d / "index.jsonl").read_text().splitlines()
    (d / "index.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        load(d)


def test_loader_rejects_a_duplicated_cell(tmp_path):
    rng = np.random.default_rng(6)
    d = write_capture(tmp_path, {"residual": grid(rng)})
    lines = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines()]
    lines[1] = {**lines[1], "item": lines[0]["item"], "attribute": lines[0]["attribute"]}
    (d / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(ValueError, match="duplicate"):
        load(d)
