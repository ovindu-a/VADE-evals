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
"""X8 on molab: a first look at LLaVA-1.5-7B's image->text handoff window, before the full X4 pod run.

Open on molab from GitHub, attach a GPU (notebook specs -> GPU), then press the buttons in order.
Code comes from this repo's `molab` branch; VADE data from the public HF bucket
ovindu-a/counterfact-data. No pruning here (VADE's scripts are not in the bucket), so the curves
run on VADE's unpruned flags tuples and are read against the model's own source accuracy.
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
    # X8: can we find LLaVA-1.5's head window?

    Stage 1 of `methods/image_head_pipeline.py` (the rule every model so far was localized
    with), on **LLaVA-1.5-7B** (Vicuna / Llama-2 language model, 32 blocks x 32 heads, 576 image
    tokens):

    - **image swap**: replace the residual stream at *all 576 image positions* with the source
      image's, at layer L, for L = 0..32. While the entity is still being read off the image,
      this transfers the source's answer; once the answer position has read it, it stops working.
    - **last-token swap**: the same at the *last prompt token* only. This starts working once the
      entity has arrived there.

    The crossover brackets the image->text handoff: `patch_layer` = the deepest layer of the
    image curve's top plateau, `read_end` = the first layer where the last-token curve reaches
    90% of its range. Optional **stage 2** then installs every attention head of each window of
    <=3 blocks inside that range at the last token, to narrow the handoff to a few blocks.

    For reference: Qwen2.5-VL-7B hands off in blocks 21-23 of 28 (0.75-0.82 of depth); Idefics3
    (Llama-3.1) needed 16-24 of 32 and no 3-block window carried it. Venhoff et al. report that
    LLaVA-style models resolve the entity late, so a late or diffuse window here is a finding.

    **Caveat:** no pruning on molab, so rows the model gets wrong cannot transfer. Each curve is
    shown raw and divided by `source_read` (the layer-0 image swap = the model reading the source
    image, i.e. its accuracy on these rows). Scoring is VADE's text match, since LLaVA's casing
    need not match the stored labels.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x8", label="work dir")
    model_id = mo.ui.text(value="llava-hf/llava-1.5-7b-hf", label="model", full_width=True)
    mo.vstack([repo_url, mo.hstack([repo_branch, bucket, workdir]), model_id])
    return bucket, model_id, repo_branch, repo_url, workdir


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
def _(bucket, mo, model_id, os, repo_branch, repo_url, setup_btn, stream, sys, workdir):
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
    for _f in list_repo_files(model_id.value):
        if not _f.endswith((".safetensors", ".bin")) and "index.json" not in _f:
            hf_hub_download(model_id.value, _f)
    _env = dict(os.environ, VADE_ROOT=vade)
    tests_rc = stream([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_llava_adapter.py"],
                      cwd=repo, env=_env, check=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, "
          f"torch {torch.__version__}. LLaVA adapter tests: "
          f"**{'passed' if tests_rc == 0 else 'FAILED -- read the output above before running anything'}**.")
    return gpu, repo, tests_rc, vade


@app.cell
def _(mo):
    attrs = mo.ui.multiselect(options=["capital", "language", "currency", "calling_code"],
                              value=["capital", "language"], label="flags attributes")
    rows = mo.ui.dropdown(options={"smoke (8)": 8, "small (32)": 32, "medium (64)": 64, "full (128)": 128},
                          value="medium (64)", label="stage-1 rows per curve")
    stage2 = mo.ui.checkbox(value=False, label="also stage 2 (block-window sweep, 64 rows)")
    batch = mo.ui.number(start=1, stop=128, value=32, label="batch size")
    run_btn = mo.ui.run_button(label="2. Run stage 1 (+ stage 2)")
    mo.vstack([mo.hstack([attrs, rows]), mo.hstack([stage2, batch, run_btn])])
    return attrs, batch, rows, run_btn, stage2


@app.cell
def _(attrs, batch, gpu, mo, model_id, os, repo, rows, run_btn, stage2, stream, sys, tests_rc, vade):
    mo.stop(not run_btn.value, mo.md("Choose attributes and size, then press **2. Run**. Medium is about "
                                     "33 layers x 2 position sets x 64 rows per attribute."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The adapter tests failed.** Fix that first: every number would be suspect."))
    stages = ["ceiling"] + (["windows"] if stage2.value else [])
    cmd = [sys.executable, "methods/image_head_pipeline.py", "--model_id", model_id.value, "--vade_root", vade,
           "--entity", "flags", "--attributes", *attrs.value, "--n_layers", "32", "--n_heads", "32",
           "--n_rows_ceiling", str(rows.value), "--n_rows", "64", "--scoring", "text", "--allow_unpruned",
           "--skip_scoring_check", "--stages", *stages, "--batch_size", str(int(batch.value))]
    _env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade)
    stream(cmd, cwd=repo, env=_env)
    out_root = os.path.join(repo, "results", "image_head_pipeline_" + model_id.value.split("/")[-1], "flags")
    mo.md(f"Done: `{out_root}`")
    return (out_root,)


@app.cell
def _(attrs, json, mo, os, out_root):
    import pandas as pd

    states = {}
    for a in attrs.value:
        p = os.path.join(out_root, a, "pipeline.json")
        if os.path.exists(p):
            states[a] = json.load(open(p))
    summary = []
    for _a, _s in states.items():
        _h, _c = _s.get("handoff", {}), _s.get("scoring_check", {})
        _w = _s.get("window") or {}
        summary.append({
            "attribute": _a,
            "source_read (accuracy)": round(100 * _c.get("source_read", float("nan")), 1),
            "clean base match": round(100 * _c.get("clean_base", float("nan")), 1),
            "patch_layer": _h.get("patch_layer"), "image_end": _h.get("image_end"), "read_end": _h.get("read_end"),
            "handoff blocks": f"{_h['handoff_blocks'][0]}-{_h['handoff_blocks'][-1]}" if _h.get("handoff_blocks") else "",
            "relative depth": (f"{_h['handoff_blocks'][0] / 32:.2f}-{(_h['handoff_blocks'][-1] + 1) / 32:.2f}"
                               if _h.get("handoff_blocks") else ""),
            "stage-2 window": _w.get("label", ""),
            "warnings": " | ".join(_h.get("warnings", []) + _c.get("warnings", [])),
        })
    mo.vstack([mo.md("## Stage 1: the rule's choices per attribute"),
               mo.ui.table(pd.DataFrame(summary), selection=None)])
    return pd, states


@app.cell
def _(mo, pd, states):
    import altair as alt

    long = []
    for _a, _s in states.items():
        _h = _s.get("handoff", {})
        _acc = _s.get("scoring_check", {}).get("source_read") or 1.0
        for _curve, _key in (("image swap: cause", "image_curve"), ("last-token swap: moved off base", "last_token_curve")):
            for _L, _v in (_h.get(_key) or {}).items():
                long.append({"attribute": _a, "curve": _curve, "layer": int(_L), "raw %": 100 * _v,
                             "÷ source_read %": 100 * _v / _acc})
    curves_df = pd.DataFrame(long)
    if curves_df.empty:
        out = mo.md("No curves yet.")
    else:
        def chart(metric):
            base = alt.Chart(curves_df).mark_line(point=True).encode(
                x=alt.X("layer:Q", scale=alt.Scale(domain=[0, 32])),
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 105])),
                color="curve:N", tooltip=["attribute", "curve", "layer", "raw %", "÷ source_read %"])
            return base.properties(width=260, height=200).facet(column="attribute:N", title=metric)
        out = mo.vstack([mo.md("## The two curves, by layer (layer L = the residual after block L-1)"),
                         chart("raw %"), chart("÷ source_read %")])
    out
    return


@app.cell
def _(json, mo, os, out_root, pd, states):
    tables = []
    for _a, _s in states.items():
        _wj_path = _s.get("windows_json")
        if not _wj_path:
            continue
        _wj = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(out_root))), _wj_path)))
        _io = _wj["image_patch_only"]
        _rows = [{"window": w["label"], "heads": w["n_heads"],
                  "suff. first %": round(100 * w["suff"]["first_src"], 1),
                  "suff. cause %": round(100 * w["suff"]["cause"], 1),
                  "nec. first %": round(100 * w.get("nec", {}).get("first_src", float("nan")), 1),
                  "shuffled donor %": round(100 * w.get("shuffled", {}).get("first_donor", float("nan")), 1)}
                 for w in _wj["windows"]]
        _df = pd.DataFrame(_rows).sort_values("suff. first %", ascending=False)
        tables.append(mo.vstack([
            mo.md(f"### {_a}: image patch only = {100 * _io['first_src']:.1f}% first-token transfer "
                  f"(n={_io.get('n')}); chosen window **{(_s.get('window') or {}).get('label', '-')}**; "
                  f"identity check {'ok' if _wj.get('full_reference_identity_ok') else 'FAILED'}"),
            mo.ui.table(_df, selection=None)]))
    (mo.vstack([mo.md("## Stage 2: every <=3-block window, all heads installed at the last token")] + tables)
     if tables else mo.md("Stage 2 not run (tick the box and press **2. Run** again; stage 1 is reused)."))
    return


@app.cell
def _(mo, os, out_root):
    import io as _io
    import zipfile

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for _d, _, _fs in os.walk(out_root):
            for _f in _fs:
                z.write(os.path.join(_d, _f), os.path.relpath(os.path.join(_d, _f), os.path.dirname(out_root)))
    mo.download(data=buf.getvalue(), filename="x8_llava_window_flags.zip", label="download results (zip)")
    return


if __name__ == "__main__":
    app.run()
