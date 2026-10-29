"""CPU tests for methods/x15_summary.py: k90 and its block-matched null off a head_trace JSON."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import x15_summary as xs                                   # noqa: E402


def _trace():
    arms = [{"k": 48, "kind": "all", "cause": 0.95}]
    for k, c in ((1, .10), (2, .30), (4, .70), (6, .86), (8, .91), (16, .94)):
        arms.append({"k": k, "kind": "top", "cause": c, "heads": []})
    for k, cs in ((6, (.02, .04)), (8, (.03, .05, .01))):
        arms += [{"k": k, "kind": "random_block", "draw": i, "cause": c, "heads": []} for i, c in enumerate(cs)]
    arms.append({"k": 8, "kind": "random", "cause": .02, "heads": []})
    return {"phase2": {"image_patch_only_cause": 0.96}, "n_rows": 64, "blocks": [20, 21, 22],
            "phase3_sufficiency": {"ceiling_all_traced_heads": 0.95, "arms": arms},
            "phase1_ranked": [[22, 3], [21, 4], [22, 1], [20, 0], [21, 7], [22, 9], [21, 2], [20, 5], [21, 1]]}


def test_k90_is_the_first_k_reaching_90pct_of_the_image_swap():
    c = xs.concentration(_trace())
    assert c["k90"] == 8                                # .86 < .9*.96 = .864 <= .91
    assert c["k90_heads"] == ["22.3", "21.4", "22.1", "20.0", "21.7", "22.9", "21.2", "20.5"]
    assert c["null_at_k90"] == {"mean": 0.03, "max": 0.05, "draws": 3}
    assert c["n_traced_heads"] == 48 and c["image_swap_cause"] == 0.96
    assert [p["k"] for p in c["curve"]] == [1, 2, 4, 6, 8, 16]
    assert xs.concentration(_trace(), frac=0.99)["k90"] is None


def test_summarize_model_reads_a_copied_pipeline_folder(tmp_path):
    d = tmp_path / "gemma-3-12b-it" / "pipeline" / "flags" / "language"
    d.mkdir(parents=True)
    (d / "head_trace_patch20_blocks20-22_text.json").write_text(json.dumps(_trace()))
    (d / "windows_patch20_text.json").write_text(json.dumps({
        "n_heads": 16, "image_patch_only": {"first_src": .97, "n_first": 60},
        "windows": [{"blocks": [20, 21, 22], "suff": {"first_src": .93}}]}))
    state = {"entity": "flags", "attribute": "language", "scoring": "text",
             "handoff": {"patch_layer": 20, "read_end": 23, "handoff_blocks": [20, 21, 22],
                         "image_curve": {str(L): 1.0 for L in range(49)}, "warnings": []},
             "scoring_check": {"source_read": .98, "warnings": []},
             "window": {"blocks": [20, 21, 22], "label": "20-22"},
             # paths as the molab clone wrote them: resolved by file name next to pipeline.json
             "windows_json": "results/x15_cross_model_heads/gemma-3-12b-it/pipeline/flags/language/windows_patch20_text.json",
             "head_trace_json": "/elsewhere/head_trace_patch20_blocks20-22_text.json"}
    (d / "pipeline.json").write_text(json.dumps(state))
    s = xs.main(["--root", str(tmp_path)])[0]
    r = s["attributes"]["flags/language"]
    assert r["handoff_depth"] == "0.42-0.48" and r["window_first_src"] == .93 and r["heads_per_block"] == 16
    assert r["heads"]["k90"] == 8 and abs(r["heads"]["k90_frac_of_model_heads"] - 8 / (16 * 48)) < 1e-12
    assert r["role_checks"] == {"name_rank": None, "crossmodal": None, "mean_ablation": None}
    assert "| gemma-3-12b-it | flags/language | 20-22 |" in (tmp_path / "SUMMARY.md").read_text()
