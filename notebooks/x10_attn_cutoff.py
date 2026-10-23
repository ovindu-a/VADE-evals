# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "transformers==5.16.*",
#     "accelerate",
#     "pillow",
#     "numpy",
#     "pandas",
#     "altair",
#     "scikit-learn",
#     "pytest",
#     "matplotlib",
#     "huggingface_hub>=1.24",
# ]
# ///
"""X10 on molab: after the entity reaches the answer position, can the MLPs write the value with attention cut off?

Open on molab from GitHub, attach a GPU (notebook specs -> GPU), then press the buttons in order.
Code comes from this repo's `molab` branch; VADE data from the public HF bucket
ovindu-a/counterfact-data.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # X10: once the entity is in the residual, do the MLPs write the value alone?

    X9's logit lens shows the order at the answer position (the last prompt token): the country
    **name** arrives first (LLaVA L19-21), and the asked **value** appears later (L22-30). Is the value
    then computed by the MLPs **from the residual alone**, or does the answer position still need to
    read other positions (image, question) after the name has arrived?

    `methods/attn_cutoff.py` cuts attention at the answer position from block **L** on, at the last
    prompt token and every generated token, and sweeps L:

    | arm | blocks >= L, at the answer position |
    |---|---|
    | `attn_self` | attends **only to itself**: no new information from any other position (main arm) |
    | `attn_zero` | attention output zeroed (stricter) |
    | `mlp_zero` | MLP output zeroed instead (control: are the MLPs past L needed?) |

    Each (arm, L) is compared with the clean run on the rows the clean model answers correctly: the
    final answer, and the logit-lens rank of the asked value per layer. The prediction: `attn_self`
    keeps the answer once L is past the name's arrival, and `mlp_zero` breaks it until L is past the
    lookup MLPs. The tests check `attn_self` against a real attention mask on tiny Llama / Qwen2 stacks.

    Pilot: 16 flags x 4 attributes, every 2nd layer, on **Qwen2.5-VL-7B, InternVL3-8B and
    LLaVA-1.5-7B**, one after another (each ~15 GB; the previous model's weights are deleted to save
    disk).
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x10", label="work dir")
    mo.vstack([repo_url, mo.hstack([repo_branch, bucket, workdir])])
    return bucket, repo_branch, repo_url, workdir


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys

    def stream(cmd, cwd=None, env=None, check=True):
        """Run `cmd`, echoing its output live; raise on failure unless check=False."""
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
        if check and rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")
        return rc

    return json, os, stream, sys


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up (clone repo + pull data + adapter tests)")
    setup_btn
    return (setup_btn,)


@app.cell
def _(bucket, mo, os, repo_branch, repo_url, setup_btn, stream, sys, workdir):
    mo.stop(not setup_btn.value, mo.md("Press **1. Set up**."))
    import shutil

    import torch

    root = workdir.value
    repo = os.path.join(root, "VADE-evals")
    vade = os.path.join(root, "VADE")
    os.makedirs(root, exist_ok=True)
    if os.path.isdir(os.path.join(repo, ".git")):
        stream(["git", "fetch", "origin", repo_branch.value], cwd=repo)
        stream(["git", "checkout", "-B", repo_branch.value, f"origin/{repo_branch.value}"], cwd=repo)
    else:
        stream(["git", "clone", "--depth", "1", "--branch", repo_branch.value, repo_url.value, repo])
    hf_cli = os.path.join(os.path.dirname(sys.executable), "hf")
    hf_cli = hf_cli if os.path.exists(hf_cli) else (shutil.which("hf") or "hf")
    stream([hf_cli, "buckets", "sync", f"hf://buckets/{bucket.value}", vade])
    # pyvene is imported by methods/dbm (and so by ndm/ceiling_sweep). --no-deps, as on the pod
    # (GPU_RUNBOOK.md section 1), so it cannot move transformers off the pinned 5.16.
    if stream([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pyvene"], check=False) != 0:
        stream([shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "--no-deps", "pyvene"])
    stream([sys.executable, "-c", "import pyvene, transformers; print('pyvene ok, transformers', transformers.__version__)"])
    # the processor files only (no weights), so the adapter's real-processor tests RUN, not skip
    from huggingface_hub import list_repo_files, hf_hub_download
    for _f in list_repo_files("llava-hf/llava-1.5-7b-hf"):
        if not _f.endswith((".safetensors", ".bin")) and "index.json" not in _f:
            hf_hub_download("llava-hf/llava-1.5-7b-hf", _f)
    _env = dict(os.environ, VADE_ROOT=vade)
    tests_rc = stream([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_llava_adapter.py",
                       "tests/test_attn_cutoff.py", "tests/test_prompt_styles.py"],
                      cwd=repo, env=_env, check=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, "
          f"torch {torch.__version__}. adapter + cutoff tests: "
          f"**{'passed' if tests_rc == 0 else 'FAILED -- read the output above before running anything'}**.")
    return gpu, repo, tests_rc, vade


@app.cell
def _(mo):
    models = mo.ui.multiselect(options=["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B-hf",
                                        "llava-hf/llava-1.5-7b-hf"],
                               value=["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B-hf",
                                      "llava-hf/llava-1.5-7b-hf"], label="models")
    styles = mo.ui.text(value="Qwen2.5-VL-7B-Instruct=default InternVL3-8B-hf=default llava-1.5-7b-hf=what_is",
                        label="prompt style per model (VADE_PROMPT_STYLE; LLaVA needs what_is, X9)", full_width=True)
    n_items = mo.ui.number(start=4, stop=84, value=16, label="flags")
    step = mo.ui.number(start=1, stop=8, value=2, label="layer step")
    cleanup = mo.ui.checkbox(value=True, label="delete each model's weights after its run (disk)")
    run_btn = mo.ui.run_button(label="2. Run the cutoff pilot")
    mo.vstack([models, styles, mo.hstack([n_items, step, cleanup]), run_btn])
    return cleanup, models, n_items, run_btn, step, styles


@app.cell
def _(cleanup, gpu, mo, models, n_items, os, repo, run_btn, step, stream, styles, sys, tests_rc, vade):
    mo.stop(not run_btn.value, mo.md("Press **2. Run**. Per model: one download, then 1 clean + 3 arms x "
                                     "~15 cut layers over 64 prompts (a few minutes each)."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The tests failed.** Fix that first: every number would be suspect."))
    import shutil as _sh

    _style_of = dict(kv.split("=", 1) for kv in styles.value.split())
    out_dirs = {}
    for _m in models.value:
        _slug = _m.split("/")[-1]
        _out = os.path.join(repo, "results", "attn_cutoff", _slug)
        _env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade,
                    VADE_PROMPT_STYLE=_style_of.get(_slug, "default"))
        _rc = stream([sys.executable, "methods/attn_cutoff.py", "--model_id", _m, "--vade_root", vade,
                      "--n_items", str(int(n_items.value)), "--layer_step", str(int(step.value)),
                      "--out_dir", _out], cwd=repo, env=_env, check=False)
        if _rc == 0:
            out_dirs[_m] = _out
        if cleanup.value:
            from huggingface_hub.constants import HF_HUB_CACHE as _cache
            _d = os.path.join(_cache, "models--" + _m.replace("/", "--"))
            if os.path.isdir(_d):
                _sh.rmtree(_d, ignore_errors=True)
                print(f"freed {_d}", flush=True)
    mo.md("Done: " + ", ".join(f"`{k}`" for k in out_dirs) + (" (some runs FAILED; read the log)" if len(out_dirs) < len(models.value) else ""))
    return (out_dirs,)


@app.cell
def _(json, mo, os, out_dirs):
    import pandas as pd

    res = {m: json.load(open(os.path.join(d, "summary.json"), encoding="utf-8")) for m, d in out_dirs.items()}
    _head = []
    for _m, _r in res.items():
        _c = _r["summary"]["clean"]
        _head.append({"model": _m.split("/")[-1], "prompt style": _r["prompt_style"], "layers": _r["n_layers"],
                      "clean accuracy %": round(100 * _c["accuracy"], 1), "rows kept": _c["n_correct"],
                      "name reaches rank<=10 at L": _c["name_arrives"], "value reaches rank 1 at L": _c["value_top1"]})
    _rows = []
    for _m, _r in res.items():
        for _arm in ("attn_self", "attn_zero", "mlp_zero"):
            for _L, _s in (_r["summary"].get(_arm) or {}).items():
                _rows.append({"model": _m.split("/")[-1], "arm": _arm, "cut from L": int(_L),
                              "depth": round(int(_L) / _r["n_layers"], 2),
                              "correct %": round(100 * _s["correct"], 1),
                              "same 1st token %": round(100 * _s["same_first_token"], 1),
                              "said name %": round(100 * _s["said_name"], 1),
                              "value rank @final": _s["value_rank_final_median"],
                              **{f"{a} %": round(100 * v) for a, v in _s["by_attribute"].items()}})
    cut_df = pd.DataFrame(_rows)
    mo.vstack([mo.md("## Clean runs"), mo.ui.table(pd.DataFrame(_head), selection=None),
               mo.md("## Cut runs (rows the clean model gets right)"),
               mo.ui.table(cut_df, selection=None, page_size=60)])
    return cut_df, pd, res


@app.cell
def _(cut_df, mo):
    _p = cut_df.pivot_table(index="cut from L", columns=["model", "arm"], values="correct %").round(0)
    mo.vstack([mo.md("### Answer kept (% correct, rows the clean model gets right) by where the cut starts"),
               mo.plain_text(_p.to_string())])
    return


@app.cell
def _(mo, res):
    _lines = []
    for _m, _r in res.items():
        _cuts = [c for c in ("22", "24", "28") if c in (_r["summary"].get("attn_zero") or {})][:2]
        _lines.append(f"## {_m.split('/')[-1]}  (median rank of the asked value; columns = arm@cut-from)")
        for _L, _v in _r["summary"]["clean"]["value_rank_median"].items():
            if int(_L) >= 16:
                _lines.append(f"L{int(_L):>2}  clean {_v:>7.0f} | " + " ".join(
                    f"{_a}@{_c} {_r['summary'][_a][_c]['value_rank_median'][_L]:>7.0f}"
                    for _a in ("attn_self", "attn_zero", "mlp_zero") for _c in _cuts if _a in _r["summary"]))
    mo.vstack([mo.md("### Logit lens at the answer position: clean vs cut runs"), mo.plain_text("\n".join(_lines))])
    return


@app.cell
def _(mo, os, out_dirs):
    import io
    import zipfile

    _buf = io.BytesIO()
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for _m, _d in out_dirs.items():
            for _n in ("summary.json", "rows.jsonl"):
                if os.path.exists(os.path.join(_d, _n)):
                    _z.write(os.path.join(_d, _n), f"{_m.split('/')[-1]}/{_n}")
    mo.download(data=_buf.getvalue(), filename="x10_attn_cutoff.zip", label="download results (zip)")
    return


if __name__ == "__main__":
    app.run()
