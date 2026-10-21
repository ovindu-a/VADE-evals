"""Offline tests for methods/gate_heads_probe.py (no model)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.gate_heads_probe import CONTRACT, classify_swap, contract_prompt, key_groups  # noqa: E402

ITEMS = {"FR": {"name": "France"}, "JP": {"name": "Japan"}}


def test_key_groups():
    IMG = 9
    #       sys sys img img img img q   kw  q   asst asst
    ids = [1, 2, IMG, IMG, IMG, IMG, 5, 6, 7, 8, 8]
    g = key_groups(ids, IMG, object_flat=[1, 2], assistant_col=9, contract_cols=[7])
    assert g["system"] == [0, 1]
    assert g["object"] == [3, 4] and g["background"] == [2, 5]
    assert g["question"] == [6, 8] and g["contract_words"] == [7]
    assert g["assistant"] == [9, 10]
    assert sorted(c for v in g.values() for c in v) == list(range(len(ids)))


def test_classify_swap():
    assert classify_swap("known,france", "FR", "JP", ITEMS) == "known_base"
    assert classify_swap(" known,japan", "FR", "JP", ITEMS) == "known_donor"
    assert classify_swap("known,italy", "FR", "JP", ITEMS) == "known_other"
    assert classify_swap("unknown,4", "FR", "JP", ITEMS) == "unknown"


def test_contract_prompts():
    for e in CONTRACT:
        p = contract_prompt(e)
        assert '"known"' in p and '"unknown"' in p and p.endswith("Output only the answer.")
