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
"""X6 on molab: what the ICLR gate heads read, and whether they carry identity or familiarity.

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
    # X6: what do the gate heads (16.1 16.14 16.17 16.20 17.21) read?

    X5 showed their knockout leaves identification and fact recall intact and only turns
    `known,<name>` into `unknown`. `methods/gate_heads_probe.py` looks inside:

    - **A. attention** from the answer position onto system / object / background / question /
      the words `known`/`unknown` / assistant prefix, for the gate heads vs movers vs random heads;
    - **B. the decision**: logit(known) - logit(unknown), clean vs knocked out, and each head's exact
      direct write on that axis;
    - **C. swaps**: the five heads' values from another item, a blank canvas, or a 32x32-degraded copy.
      Name follows the donor = identity; status flips to `unknown` = familiarity.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x6", label="work dir")
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

    return os, stream, sys


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
    stream([sys.executable, "methods/gate_heads_probe.py", "--vade_root", vade, "--dry_run"], cwd=repo)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, "
          f"torch {torch.__version__}.")
    return gpu, repo, vade


@app.cell
def _(mo):
    entity = mo.ui.dropdown(options=["flags", "celebrities", "brands"], value="flags", label="entity")
    mode = mo.ui.dropdown(options=["smoke", "full"], value="smoke", label="run size")
    batch = mo.ui.number(start=1, stop=64, value=8, label="batch size")
    run_btn = mo.ui.run_button(label="2. Run gate_heads_probe")
    mo.hstack([entity, mode, batch, run_btn])
    return batch, entity, mode, run_btn


@app.cell
def _(batch, entity, gpu, mo, mode, os, repo, run_btn, stream, sys, vade):
    mo.stop(not run_btn.value, mo.md("Choose entity and size, then press **2. Run**. Smoke = 8 items."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    out_dir = os.path.join(repo, "results", "gate_heads_probe", f"{entity.value}_{mode.value}")
    cmd = [sys.executable, "methods/gate_heads_probe.py", "--entity", entity.value, "--vade_root", vade,
           "--out_dir", out_dir, "--batch_size", str(int(batch.value))]
    if mode.value == "smoke":
        cmd += ["--limit_items", "8"]
    stream(cmd, cwd=repo, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    mo.md(f"Done: `{out_dir}`")
    return (out_dir,)


@app.cell
def _(mo, os, out_dir):
    files = {n: open(os.path.join(out_dir, n), "rb").read() for n in ("summary.json", "swaps.jsonl")}
    mo.hstack([mo.download(data=b, filename=f"x6_{n}", label=f"download {n}") for n, b in files.items()])
    return


if __name__ == "__main__":
    app.run()
