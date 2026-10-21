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
#     "huggingface_hub>=1.24",
# ]
# ///
"""X5 on molab: the ICLR 2027 'entity-recognition gate' (Qwen2.5-VL-7B blocks 16-17) in our setup.

Open on molab from GitHub, attach a GPU (notebook specs -> GPU), then press the run buttons in
order. Everything the run needs comes from public sources: this repo (cloned at REPO_BRANCH)
and the VADE data in the public HF bucket ovindu-a/counterfact-data.
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
    # X5: is the ICLR "entity-recognition gate" recognition, or an answer-side signal?

    The submission mean-ablates attention in Qwen2.5-VL-7B **blocks 16-17** (every head's
    `o_proj` input replaced by its mean over the sequence) and finds that `known,<name>`
    identity answers turn into `unknown`. On our data those blocks are where the *question*
    starts reaching the answer position, while the entity is moved later (blocks 21-23).

    This notebook runs `methods/gate_knockout.py`: their exact intervention on flags, with
    our fact questions, their identity contract, an open identity question, a perceptual
    control, and the **same fact questions with the country named in text**. Every failure
    is typed as *refusal*, *other entity*, *wrong attribute* or *other*.

    | pattern | reading |
    |---|---|
    | image drops, text survives, failures = other entity | visual recognition |
    | image **and** text drop, failures = refusal | answer-side abstention |
    | failures = wrong attribute | the question is not routed |
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x5", label="work dir")
    mo.vstack([repo_url, mo.hstack([repo_branch, bucket, workdir])])
    return bucket, repo_branch, repo_url, workdir


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys

    def stream(cmd, cwd=None, env=None):
        """Run `cmd`, echoing its output live; raise on failure."""
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")

    return json, os, stream, sys


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up (clone repo + pull data)")
    setup_btn
    return (setup_btn,)


@app.cell
def _(bucket, mo, os, repo_branch, repo_url, setup_btn, stream, sys, workdir):
    mo.stop(not setup_btn.value, mo.md("Press **1. Set up** to clone the repo and pull the VADE data."))
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
    import shutil
    hf_cli = os.path.join(os.path.dirname(sys.executable), "hf")
    hf_cli = hf_cli if os.path.exists(hf_cli) else (shutil.which("hf") or "hf")
    stream([hf_cli, "buckets", "sync", f"hf://buckets/{bucket.value}", vade])
    stream([sys.executable, "methods/gate_knockout.py", "--vade_root", vade, "--dry_run"], cwd=repo)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, "
          f"torch {torch.__version__}.")
    return gpu, repo, vade


@app.cell
def _(mo):
    mode = mo.ui.dropdown(options=["smoke", "full"], value="smoke", label="run size")
    batch = mo.ui.number(start=1, stop=256, value=32, label="batch size")
    run_btn = mo.ui.run_button(label="2. Run gate_knockout")
    mo.hstack([mode, batch, run_btn])
    return batch, mode, run_btn


@app.cell
def _(batch, gpu, mo, mode, os, repo, run_btn, stream, sys, vade):
    mo.stop(not run_btn.value, mo.md("Choose a size and press **2. Run**. Smoke is 8 flags, one random draw, "
                                     "window 16-17 only (a few minutes); full is all 84 flags (~10-20 min)."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    out_dir = os.path.join(repo, "results", "gate_knockout", f"flags_{mode.value}")
    cmd = [sys.executable, "methods/gate_knockout.py", "--vade_root", vade, "--out_dir", out_dir,
           "--batch_size", str(int(batch.value))]
    if mode.value == "smoke":
        cmd += ["--limit_items", "8", "--n_random", "1", "--windows", "16-17"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    stream(cmd, cwd=repo, env=env)
    mo.md(f"Done: `{out_dir}`")
    return (out_dir,)


@app.cell
def _(json, mo, os, out_dir):
    import pandas as pd

    report = json.load(open(os.path.join(out_dir, "summary.json")))
    cols = ["n", "correct", "refusal", "other_entity", "wrong_attr", "other"]

    def frame(d):
        df = pd.DataFrame(d).T[cols]
        for c in cols[1:]:
            df[c] = (100 * df[c].astype(float)).round(1)
        return df

    tables = {"image / fact (pooled)": frame(report["pooled"]["image/fact"]),
              "text / fact (pooled)": frame(report["pooled"]["text/fact"])}
    for p in ("image/identify", "image/identify_open"):
        if p in report["summary"]:
            tables[p] = frame({a: s for a, s in report["summary"][p].items() if s.get("n")})
    if "image/color" in report["summary"]:
        tables["image/color: % changed vs clean"] = pd.DataFrame(
            {a: {"n": s["n"], "changed %": round(100 * s["changed"], 1)}
             for a, s in report["summary"]["image/color"].items()}).T
    mo.vstack([mo.md("## Results (% of clean-correct rows)")] +
              [mo.vstack([mo.md(f"### {k}"), mo.ui.table(v, selection=None)]) for k, v in tables.items()])
    return pd, report


@app.cell
def _(mo, pd, report):
    import altair as alt

    def stacked(name):
        d = report["pooled"][name]
        long = [{"arm": a, "outcome": k, "pct": 100 * s[k]} for a, s in d.items()
                for k in ("correct", "refusal", "other_entity", "wrong_attr", "other")]
        return (alt.Chart(pd.DataFrame(long), title=name)
                .mark_bar().encode(x=alt.X("pct:Q", title="% of clean-correct rows", scale=alt.Scale(domain=[0, 100])),
                                   y=alt.Y("arm:N", sort=None, title=None),
                                   color=alt.Color("outcome:N", sort=["correct", "refusal", "other_entity",
                                                                      "wrong_attr", "other"]))
                .properties(width=520, height=220))

    mo.hstack([stacked("image/fact"), stacked("text/fact")], wrap=True)
    return


@app.cell
def _(mo, os, out_dir):
    files = {n: open(os.path.join(out_dir, n), "rb").read() for n in ("summary.json", "rows.jsonl")}
    mo.hstack([mo.download(data=b, filename=f"x5_{n}", label=f"download {n}") for n, b in files.items()])
    return


if __name__ == "__main__":
    app.run()
