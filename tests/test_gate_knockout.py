"""Offline tests for methods/gate_knockout.py (no model)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.gate_knockout import (GATE5, build_arms, classify_fact, classify_identify, color_of,  # noqa: E402
                                   mean_ablate, parse_heads, parse_window, summarize)

ITEMS = {
    "FR": {"name": "France", "capital": "Paris", "currency": "EUR", "language": "French", "calling_code": "33"},
    "JP": {"name": "Japan", "capital": "Tokyo", "currency": "JPY", "language": "Japanese", "calling_code": "81"},
    "DE": {"name": "Germany", "capital": "Berlin", "currency": "EUR", "language": "German", "calling_code": "49"},
}
ATTRS = ["capital", "currency", "language", "calling_code"]


def test_parse():
    assert parse_heads(GATE5) == [(16, 1), (16, 14), (16, 17), (16, 20), (17, 21)]
    assert parse_window("16-17") == [16, 17]


def test_arms():
    arms = build_arms(["16-17", "24-25"], parse_heads(GATE5), 2, 28, 0)
    names = [a for a, _ in arms]
    assert names == ["clean", "ko:16-17", "ko:24-25", "ko:gate5", "ko:gate5:random0", "ko:gate5:random1"]
    d = dict(arms)
    assert d["ko:16-17"] == {16: None, 17: None}
    gate = set(parse_heads(GATE5))
    for i in range(2):
        spec = d[f"ko:gate5:random{i}"]
        heads = {(b, h) for b, hs in spec.items() for h in hs}
        assert len(heads) == 5 and not heads & gate
        assert sorted(len(hs) for hs in spec.values()) == [1, 4]


def test_mean_ablate():
    torch = pytest.importorskip("torch")
    t = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    out = mean_ablate(t, torch.tensor([1]), mask)
    # untouched columns
    assert torch.equal(out[..., [0, 2]], t[..., [0, 2]])
    # row 0 averages only the three valid positions
    assert torch.allclose(out[0, :, 1], t[0, 1:, 1].mean().expand(4))
    assert torch.allclose(out[1, :, 1], t[1, :, 1].mean().expand(4))
    # all columns
    full = mean_ablate(t, None, mask)
    assert torch.allclose(full[1], t[1].mean(0, keepdim=True).expand(4, 3))
    # a cached decode step is a no-op
    step = t[:, :1]
    assert mean_ablate(step, None, mask) is step
    # input not modified in place
    assert torch.equal(t, torch.arange(24, dtype=torch.float32).reshape(2, 4, 3))


def test_classify_fact():
    assert classify_fact(" Paris.", "FR", "capital", ITEMS, ATTRS) == "correct"
    assert classify_fact(" unknown", "FR", "capital", ITEMS, ATTRS) == "refusal"
    assert classify_fact(" Tokyo", "FR", "capital", ITEMS, ATTRS) == "other_entity"
    assert classify_fact(" French", "FR", "capital", ITEMS, ATTRS) == "wrong_attr"     # own other attribute
    assert classify_fact(" JPY", "FR", "language", ITEMS, ATTRS) == "wrong_attr"       # another country's code
    assert classify_fact(" Lyon", "FR", "capital", ITEMS, ATTRS) == "other"
    # a shared value (EUR) is correct, never "other_entity"
    assert classify_fact(" EUR", "DE", "currency", ITEMS, ATTRS) == "correct"


def test_classify_identify():
    assert classify_identify("known,france", "FR", ITEMS, contract=True) == "correct"
    assert classify_identify("unknown,4", "FR", ITEMS, contract=True) == "refusal"
    assert classify_identify("known,japan", "FR", ITEMS, contract=True) == "other_entity"
    assert classify_identify(" France.", "FR", ITEMS, contract=False) == "correct"
    assert classify_identify(" I don't know", "FR", ITEMS, contract=False) == "refusal"
    assert classify_identify(" Italy", "FR", ITEMS, contract=False) == "other"


def test_color_and_summary():
    assert color_of(" Red.") == "red"
    rows = [
        {"arm": "clean", "probe": "image/fact:capital", "item": "FR", "outcome": "correct", "mover_name_rank": 1},
        {"arm": "clean", "probe": "image/fact:capital", "item": "JP", "outcome": "other", "mover_name_rank": 3},
        {"arm": "ko:16-17", "probe": "image/fact:capital", "item": "FR", "outcome": "refusal", "mover_name_rank": 2},
        {"arm": "ko:16-17", "probe": "image/fact:capital", "item": "JP", "outcome": "correct", "mover_name_rank": 1},
        {"arm": "clean", "probe": "image/color", "item": "FR", "outcome": None, "generated": "blue"},
        {"arm": "ko:16-17", "probe": "image/color", "item": "FR", "outcome": None, "generated": "red"},
    ]
    s = summarize(rows)
    ko = s["image/fact:capital"]["ko:16-17"]
    assert ko["n"] == 1 and ko["refusal"] == 1.0 and ko["loss"] == 1.0 and ko["mover_rank_mean"] == 2
    assert s["image/color"]["ko:16-17"]["changed"] == 1.0
    assert s["image/color"]["clean"]["changed"] == 0.0
