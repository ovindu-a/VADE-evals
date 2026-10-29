"""CPU tests for methods/own_accuracy_prune.py (X15): which clean generations it asks for, and that its
pruning is VADE prune_tuples.py's rule (source for cause rows, base for iso rows)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import own_accuracy_prune as oap                           # noqa: E402


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _vade(tmp_path):
    t = tmp_path / "data" / "flags" / "tuples"
    cause = {"queried": "capital", "template_id": "c1", "rule": "match_source", "base": "A", "source": "B",
             "base_label": "Aa", "source_label": "Bb"}
    iso = {"queried": "language", "template_id": "l1", "rule": "match_base", "base": "C", "source": "A",
           "base_label": "Cc-lang", "source_label": "Aa-lang"}
    _write(str(t / "capital" / "test.jsonl"), [dict(cause, row_index=0), dict(iso, row_index=1)])
    _write(str(t / "capital" / "train.jsonl"), [dict(cause, row_index=0, source="C", source_label="Cc")])
    _write(str(t / "identity" / "test.jsonl"), [{"queried": "recognition_freeform", "template_id": "r3",
                                                 "base": "A", "source": "B", "base_label": "Angola",
                                                 "source_label": "Belarus"}])
    return str(tmp_path)


def test_needed_keys_follow_the_prune_rule_and_add_recognition(tmp_path):
    root = _vade(tmp_path)
    keys, attrs = oap.needed_keys(root, "flags", None)
    assert attrs == ["capital", "identity"]
    assert keys == {("capital", "c1", "B"): "Bb",          # cause row: the SOURCE
                    ("language", "l1", "C"): "Cc-lang",    # iso row: the BASE
                    ("capital", "c1", "C"): "Cc",
                    ("recognition_freeform", "r3", "A"): "Angola",
                    ("recognition_freeform", "r3", "B"): "Belarus"}
    keys, attrs = oap.needed_keys(root, "flags", ["capital"])
    assert attrs == ["capital"] and all(k[0] != "recognition_freeform" for k in keys)


def test_prune_keeps_rows_whose_check_item_was_answered(tmp_path):
    root = _vade(tmp_path)
    lookup = {("capital", "c1", "B"): True, ("language", "l1", "C"): False, ("capital", "c1", "C"): True}
    summary = oap.prune(root, "M", "flags", ["capital", "identity"], lookup)
    kept = oap.read_jsonl(os.path.join(root, "models", "M", "flags", "tuples", "capital", "test.jsonl"))
    assert [r["row_index"] for r in kept] == [0]
    assert summary["capital"]["test"] == {"original_n": 2, "kept_n": 1, "dropped_wrong": 1,
                                          "dropped_missing_baseline": 0, "cause_kept": 1, "iso_kept": 0}
    assert summary["capital"]["train"]["kept_n"] == 1
    assert summary["identity"] == {}                     # recognition only, never pruned
    assert not os.path.exists(os.path.join(root, "models", "M", "flags", "tuples", "identity"))
    assert json.load(open(os.path.join(root, "models", "M", "flags", "PRUNED_BY.json")))["model_slug"] == "M"


def test_dry_run_needs_no_model(tmp_path, capsys):
    root = _vade(tmp_path)
    oap.main(["--model_id", "google/gemma-3-12b-it", "--vade_root", root, "--entities", "flags", "--dry_run"])
    assert "Grid valid." in capsys.readouterr().out
