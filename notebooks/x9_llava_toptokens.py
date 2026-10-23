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
"""X9 on molab: logit-lens top-20 tokens at LLaVA-1.5-7B's last prompt column (X7's toptokens phase).

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
    # X9: what is in LLaVA-1.5's last prompt column, layer by layer?

    The same readout as X7's `toptokens` on Qwen2.5-VL-7B (`methods/mlp_lookup.py --phase toptokens`),
    on **LLaVA-1.5-7B**. For 8 flags x 4 questions, shown as the image and with the country named in
    text, at the last prompt column:

    - **logit lens**: the residual after every block (1..32) through the model's own final norm and
      `lm_head`, giving the **top-20 tokens** with probabilities and the rank of the country's four
      value tokens and its name;
    - **writes**: what each block's attention and MLP *add* (blocks 14-31), read through the
      unembedding: top-20 promoted and top-10 suppressed tokens.

    Every token is typed as asked value / another attribute's value of the same country / the
    country's name / the same attribute of another country / another attribute of another
    country / other. The share per layer says when the entity, and then the asked value, appear.

    X8 put LLaVA's handoff at blocks ~17-21 and found it spread over 17-31. If the name appears in
    the lens around layers 18-22, and the asked value only later, the image->text read happens there.
    The MLPs that turn the name into the value should then be the blocks whose writes promote the value.

    **Caveat:** LLaVA's tokenizer splits many values into single letters ('C'+'airo', 'E'+'UR',
    'H'+'indi'). Those first tokens are shared by many countries, so the class shares for capital,
    currency and calling_code are noisier than on Qwen. Read the per-prompt tables for those.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x9", label="work dir")
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
    items = mo.ui.text(value="JP BR EG DE MX KE IN ES", label="flags (ISO codes)")
    wblocks = mo.ui.range_slider(start=0, stop=31, value=[14, 31], label="write blocks", show_value=True)
    run_btn = mo.ui.run_button(label="2. Run logit lens (toptokens)")
    mo.vstack([mo.hstack([items, wblocks]), run_btn])
    return items, run_btn, wblocks


@app.cell
def _(gpu, items, mo, model_id, os, repo, run_btn, stream, sys, tests_rc, vade, wblocks):
    mo.stop(not run_btn.value, mo.md("Press **2. Run**. About 64 forwards plus generations, a few minutes "
                                     "after the ~14 GB model download."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The adapter tests failed.** Fix that first: every number would be suspect."))
    out_dir = os.path.join(repo, "results", "mlp_lookup_" + model_id.value.split("/")[-1], "flags")
    _lo, _hi = wblocks.value
    cmd = [sys.executable, "methods/mlp_lookup.py", "--phase", "toptokens", "--model_id", model_id.value,
           "--vade_root", vade, "--out_dir", out_dir, "--top_items", *items.value.split(),
           "--write_blocks", *[str(b) for b in range(_lo, _hi + 1)]]
    _env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade)
    stream(cmd, cwd=repo, env=_env)
    mo.md(f"Done: `{out_dir}`")
    return (out_dir,)


@app.cell
def _(json, mo, os, out_dir):
    import pandas as pd

    tt = json.load(open(os.path.join(out_dir, "toptokens.json"), encoding="utf-8"))
    classes = ["asked_value", "other_attr_value", "country_name", "same_attr_other_country",
               "other_attr_other_country", "other"]

    def share_table(mode):
        d = tt["class_counts"][mode]
        df = pd.DataFrame({k: {c: round(100 * v[c], 1) for c in classes} for k, v in d.items()}).T
        df.index.name = "layer" if "writes" not in mode else "write"
        return df

    gen = pd.DataFrame([{"mode": p["mode"], "flag": p["name"], "asked": p["asked"], "generated": p["generated"],
                         **{f"gold {a}": v for a, v in p["values"].items()}} for p in tt["prompts"]])
    mo.vstack([mo.md("## What the model answered"), mo.ui.table(gen, selection=None, page_size=16),
               mo.md("## Share of the top-20 lens tokens by type, % (image prompts)"),
               mo.ui.table(share_table("image"), selection=None, page_size=40),
               mo.md("## Same, country named in text"),
               mo.ui.table(share_table("text"), selection=None, page_size=40)])
    return classes, pd, share_table, tt


@app.cell
def _(classes, mo, pd, tt):
    import altair as alt

    _long = []
    for _mode in ("image", "text"):
        for _L, _v in tt["class_counts"][_mode].items():
            for _c in ("asked_value", "other_attr_value", "country_name"):
                _long.append({"mode": _mode, "layer": int(_L), "type": _c, "share %": 100 * _v[_c]})
    _ranks = []
    for _p in tt["prompts"]:
        for _L, _d in _p["lens"].items():
            for _k, _r in _d["ranks"].items():
                _what = "name" if _k.startswith("name") else ("asked value" if _k.split("=")[0] == _p["asked"]
                                                              else "other value")
                _ranks.append({"mode": _p["mode"], "layer": int(_L), "token": _what, "log10 rank": __import__("math").log10(_r)})
    _rk = pd.DataFrame(_ranks).groupby(["mode", "layer", "token"], as_index=False)["log10 rank"].median()
    _c1 = (alt.Chart(pd.DataFrame(_long), title="share of the top-20 lens tokens")
           .mark_line(point=True).encode(x="layer:Q", y="share %:Q", color="type:N", strokeDash="mode:N")
           .properties(width=460, height=260))
    _c2 = (alt.Chart(_rk, title="median rank in the full vocabulary (log10; 0 = top-1)")
           .mark_line(point=True).encode(x="layer:Q", y=alt.Y("log10 rank:Q", scale=alt.Scale(reverse=True)),
                                         color="token:N", strokeDash="mode:N")
           .properties(width=460, height=260))
    mo.hstack([_c1, _c2], wrap=True)
    return


@app.cell
def _(mo, share_table):
    mo.vstack([mo.md("## Block writes: share of each block's top-20 PROMOTED tokens by type, % (image)"),
               mo.ui.table(share_table("image_writes"), selection=None, page_size=40),
               mo.md("## Same, text"),
               mo.ui.table(share_table("text_writes"), selection=None, page_size=40)])
    return


@app.cell
def _(mo, tt):
    pick = mo.ui.dropdown(options={f"{p['mode']} / {p['name']} / {p['asked']}": i for i, p in enumerate(tt["prompts"])},
                          value=f"{tt['prompts'][0]['mode']} / {tt['prompts'][0]['name']} / {tt['prompts'][0]['asked']}",
                          label="prompt")
    pick
    return (pick,)


@app.cell
def _(mo, pd, pick, tt):
    _p = tt["prompts"][pick.value]
    _mark = {"asked_value": "**{}**", "other_attr_value": "[{}]", "country_name": "<{}>"}
    _fmt = lambda t, c: _mark.get(c, "{}").format(repr(t)[1:-1])
    _lens = pd.DataFrame([{"layer": int(L), **d["ranks"],
                           "top-20 (p)": " ".join(f"{_fmt(t, c)}({p:.2f})" if p >= 0.01 else _fmt(t, c)
                                                  for t, p, c in d["top"])}
                          for L, d in _p["lens"].items()])
    _wr = pd.DataFrame([{"block": int(k[3:]), "MLP promotes": " ".join(_fmt(t, c) for t, c in _p["writes"][f"mlp{k[3:]}"]["promote"]),
                         "MLP suppresses": " ".join(_fmt(t, c) for t, c in _p["writes"][f"mlp{k[3:]}"]["suppress"]),
                         "attn promotes": " ".join(_fmt(t, c) for t, c in _p["writes"][k]["promote"])}
                        for k in _p["writes"] if k.startswith("attn")])
    mo.vstack([mo.md(f"### {_p['mode']} / {_p['name']} / asked **{_p['asked']}** -> `{_p['generated']!r}`  \n"
                     "values: " + ", ".join(f"{a}={v}" for a, v in _p["values"].items()) +
                     "  \nLegend: **asked value**, [other attribute's value], <country name>."),
               mo.ui.table(_lens, selection=None, page_size=40), mo.ui.table(_wr, selection=None, page_size=40)])
    return


@app.cell
def _(mo, os, out_dir):
    import io
    import zipfile

    _buf = io.BytesIO()
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for _n in ("toptokens.json", "TOPTOKENS.md", "summary.json"):
            _f = os.path.join(out_dir, _n)
            if os.path.exists(_f):
                _z.write(_f, _n)
    mo.download(data=_buf.getvalue(), filename="x9_llava_toptokens.zip", label="download results (zip)")
    return


if __name__ == "__main__":
    app.run()
