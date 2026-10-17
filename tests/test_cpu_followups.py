"""Offline checks for methods/cpu_followups.py (stdlib only; VADE checkout optional)."""
import os

import pytest

from methods import cpu_followups as cf

HAVE_VADE = os.path.isfile(os.path.join(cf.DEFAULT_VADE_ROOT, "data", "flags", "ground_truth.json"))


def test_m49_table_is_a_partition_of_84_countries():
    codes = [c for subs in cf.M49.values() for s in subs.values() for c in s.split()]
    assert len(codes) == 84 and len(set(codes)) == 84
    assert set(cf.SUBREGION) == set(cf.CONTINENT) == set(codes)


@pytest.mark.skipif(not HAVE_VADE, reason="needs the sibling VADE checkout")
def test_m49_table_matches_flags_ground_truth():
    assert set(cf.load_ground_truth(cf.DEFAULT_VADE_ROOT)) == set(cf.SUBREGION)


def test_spearman_and_wilson():
    assert cf.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cf.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert cf.spearman([1, 2], [1, 2]) is None
    lo, hi = cf.wilson(50, 100)
    assert lo < 0.5 < hi and cf.wilson(0, 0) == (0.0, 0.0)


def test_route_pair_rebuild_drops_equal_labels_only():
    gt = {"A": {"x": "1", "y": "same"}, "B": {"x": "2", "y": "same"}, "C": {"x": "3", "y": "other"}}
    rows = cf.rebuild_route_pairs(gt, n_pairs=6, seed=0, attributes=["x", "y"])
    # 6 ordered pairs x 2 attributes, minus the two (A,B)/(B,A) 'y' rows whose labels are equal.
    assert len(rows) == 10
    assert [r["row_index"] for r in rows] == list(range(10))
    assert not any(r["attribute"] == "y" and {r["base"], r["source"]} == {"A", "B"} for r in rows)


def test_dedupe_keeps_one_mapped_record_per_cluster():
    recs = [{"cluster": 1, "mapped": False}, {"cluster": 1, "mapped": True, "tag": "m"},
            {"cluster": 2, "mapped": False}]
    out = cf.dedupe_clusters(recs)
    assert len(out) == 2 and any(r.get("tag") == "m" for r in out)


def test_best_of_two_null_is_a_third_not_a_half():
    sim = cf.simulate_best_of_k(n=60, k=2, trials=4000)
    assert sim["mean_rank"] == pytest.approx(61 / 3, rel=0.08)
    assert sim["top1"] == pytest.approx(2 / 60, abs=0.015)
    assert 0.4 < sim["mean_z"] < 0.75
