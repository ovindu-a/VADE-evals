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
#     "networkx",
#     "huggingface_hub>=1.24",
# ]
# ///
"""X15 on molab: does the rule-based head-discovery procedure find an entity-mover bottleneck in
Gemma-3, Pixtral-12B and Idefics3?

Open on molab from GitHub, attach a GPU (notebook specs -> GPU), then press the buttons in order.
Code comes from this repo's `molab` branch; VADE data from the public HF bucket
ovindu-a/counterfact-data. Gemma-3 is gated: accept its licence on huggingface.co with the account
whose token you paste below.
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
    # X15: same role, model-specific sparsity?

    On Qwen2.5-VL-7B, ten "mover" heads in blocks 21-23 carry the entity from the image tokens to the
    answer position. The cross-model claim is about the **procedure**, not the heads: run the same
    rule-based discovery on three more language-model families and report, per model, where the
    handoff is and **how concentrated** it is.

    Per model, in order (each step is an existing script; nothing here is notebook-only logic):

    1. **self-checks** -- `ndm/verify_sites.py` on every site (null patch = identity, base-into-base
       swap = identity, the patch moves the logits). head_trace's own read-back identity (phase 1c)
       runs again inside step 4.
    2. **recognition** -- `own_accuracy_prune.py`: one clean generation per (item, question) the tuples
       need, VADE's text matcher, then VADE's pruning rule. Only rows the model answers correctly are
       used downstream.
    3. **handoff** -- `image_head_pipeline.py` stage 1: residual swap at every image position vs at the
       last token, every layer. `patch_layer` = the end of the image curve's top plateau, `read_end` =
       where the last-token curve reaches 90% of its range. Stage 2 picks the smallest <=3-block window.
    4. **heads** -- stage 3, `head_trace.py` in that window: rank heads, then install the top-k heads'
       image-patched outputs at the last token with the image NOT patched, with block-matched random
       heads at the same k (`--null_ks`).

    **The number per model:** `k90`, the smallest k whose top-k install reaches 90% of the full image
    swap, with the random-head null at that k and the top-k list (`methods/x15_summary.py`).

    Models: **Gemma-3-12B** (Gemma LM; 4B for a smoke run), **Pixtral-12B** (Mistral-Nemo LM),
    **Idefics3-8B** (Llama-3.1 LM; rerun by rule at n=64 for a like-for-like comparison with its earlier
    hand-picked 16-24 window). Reference: Qwen2.5-VL-7B hands off in blocks 21-23 of 28 (depth
    0.75-0.82); common10 = 10 heads.

    Caveats: a last-prompt-column install cannot steer past the first answer token, so read
    `language` (mostly one-token answers) as the primary attribute; for `capital` k90 may be undefined
    while the heads still carry the entity (see the k-curve). Gemma-3's image tokens attend to each
    other bidirectionally, which the adapter reproduces.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x15", label="work dir")
    hf_token = mo.ui.text(value="", kind="password", label="HF token (Gemma-3 is gated; blank = use env HF_TOKEN)",
                          full_width=True)
    mo.vstack([repo_url, mo.hstack([repo_branch, bucket, workdir]), hf_token])
    return bucket, hf_token, repo_branch, repo_url, workdir


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys

    MODELS = {
        "google/gemma-3-4b-it": "Gemma-3 4B (smoke)",
        "google/gemma-3-12b-it": "Gemma-3 12B",
        "mistral-community/pixtral-12b": "Pixtral-12B",
        "HuggingFaceM4/Idefics3-8B-Llama3": "Idefics3-8B",
    }
    TESTS = ["tests/test_gemma3_adapter.py", "tests/test_pixtral_adapter.py", "tests/test_idefics3_adapter.py",
             "tests/test_own_accuracy_prune.py", "tests/test_x15_summary.py"]

    def stream(cmd, cwd=None, env=None, check=True, log=None):
        """Run `cmd`, echoing its output live (and into `log` if given); raise on failure unless check=False."""
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        fh = open(log, "a") if log else None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if fh:
                fh.write(line)
        rc = proc.wait()
        if fh:
            fh.close()
        if check and rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")
        return rc

    return MODELS, TESTS, json, os, stream, sys


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up (clone repo + pull data + adapter tests)")
    setup_btn
    return (setup_btn,)


@app.cell
def _(MODELS, TESTS, bucket, hf_token, mo, os, repo_branch, repo_url, setup_btn, stream, sys, workdir):
    mo.stop(not setup_btn.value, mo.md("Press **1. Set up**."))
    import shutil

    import torch

    if hf_token.value.strip():
        os.environ["HF_TOKEN"] = hf_token.value.strip()
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
    # pyvene is imported by methods/dbm (head_trace's mask module). --no-deps, as on the pod, so it
    # cannot move transformers off the pinned 5.16.
    if stream([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pyvene"], check=False) != 0:
        stream([shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "--no-deps", "pyvene"])
    stream([sys.executable, "-c", "import pyvene, transformers; print('pyvene ok, transformers', transformers.__version__)"])
    # processor / tokenizer / config files only (no weights), so the real-processor tests RUN, not skip
    from huggingface_hub import hf_hub_download, list_repo_files
    access = {}
    for _m in MODELS:
        try:
            for _f in list_repo_files(_m):
                if "/" not in _f and not _f.endswith((".safetensors", ".bin", ".pth")) and "index.json" not in _f:
                    hf_hub_download(_m, _f)
            access[_m] = "ok"
        except Exception as _e:  # gated without a token, typically
            access[_m] = f"NO ACCESS ({type(_e).__name__})"
            print(f"!! {_m}: {_e}", flush=True)
    _env = dict(os.environ, VADE_ROOT=vade)
    tests_rc = stream([sys.executable, "-m", "pytest", "-q", "-rs", *TESTS], cwd=repo, env=_env, check=False)
    stream([sys.executable, "methods/own_accuracy_prune.py", "--model_id", "google/gemma-3-12b-it",
            "--vade_root", vade, "--entities", "flags", "brands", "--dry_run"], cwd=repo, env=_env, check=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, torch "
          f"{torch.__version__}. Adapter tests: **{'passed' if tests_rc == 0 else 'FAILED -- read the output above'}**."
          "\n\n" + "\n".join(f"- `{m}`: {a}" for m, a in access.items())
          + "\n\nThe tests must show **0 skipped** for gemma3/pixtral/idefics3 (a skip means a processor "
            "download failed -- for Gemma, a missing token or an unaccepted licence).")
    return access, gpu, repo, tests_rc, vade


@app.cell
def _(MODELS, mo):
    models = mo.ui.multiselect(options={label: mid for mid, label in MODELS.items()}, value=["Gemma-3 4B (smoke)"],
                               label="models (run in order)")
    entities = mo.ui.multiselect(options=["flags", "brands"], value=["flags"], label="entities")
    flag_attrs = mo.ui.multiselect(options=["language", "capital", "currency", "calling_code"],
                                   value=["language", "capital"], label="flags attributes")
    brand_attrs = mo.ui.multiselect(options=["hq_country", "industry", "founded_year"], value=["hq_country"],
                                    label="brands attributes")
    size = mo.ui.dropdown(options={"smoke (16 rows, coarse k)": "smoke", "full (64 rows, fine k)": "full"},
                          value="smoke (16 rows, coarse k)", label="run size")
    batch = mo.ui.number(start=1, stop=64, value=16, label="batch size")
    cleanup = mo.ui.checkbox(value=True, label="delete each model's weights after its run (disk)")
    run_btn = mo.ui.run_button(label="2. Run X15 (self-checks -> recognition -> handoff -> heads)")
    mo.vstack([models, mo.hstack([entities, flag_attrs, brand_attrs]), mo.hstack([size, batch, cleanup]), run_btn])
    return batch, brand_attrs, cleanup, entities, flag_attrs, models, run_btn, size


@app.cell
def _(access, batch, brand_attrs, cleanup, entities, flag_attrs, gpu, mo, models, os, repo, run_btn, size,
      stream, sys, tests_rc, vade):
    mo.stop(not run_btn.value, mo.md("Choose models and size, then press **2. Run**. Full size on a 12B "
                                     "model is roughly 1-2 h per attribute; every step is resumable, so "
                                     "pressing **2** again after a disconnect continues where it stopped."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The adapter tests failed.** Fix that first: every number would be suspect."))
    import shutil as _sh

    from transformers import AutoConfig

    _full = size.value == "full"
    _grid = ["1", "2", "3", "4", "6", "8", "12", "16", "24", "32", "48", "64", "96"] if _full else ["1", "2", "4", "8", "16"]
    _attrs = {"flags": flag_attrs.value, "brands": brand_attrs.value}
    x15_root = os.path.join(repo, "results", "x15_cross_model_heads")
    status = {}
    for _m in models.value:
        _slug = _m.split("/")[-1]
        if not access.get(_m, "").startswith("ok"):
            status[_m] = "skipped: no hub access (see set up)"
            continue
        _tc = AutoConfig.from_pretrained(_m).text_config
        _L, _H = _tc.num_hidden_layers, _tc.num_attention_heads
        _out = os.path.join(x15_root, _slug)
        os.makedirs(os.path.join(_out, "accuracy"), exist_ok=True)
        _env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade)
        print(f"\n{'#' * 78}\n# {_m}: {_L} blocks x {_H} heads\n{'#' * 78}", flush=True)
        # 1. self-checks: every site at the middle block (full_image: no object set needed)
        _vlog = os.path.join(_out, "verify_sites.log")
        _rc = stream([sys.executable, "methods/ndm/verify_sites.py", "--model_id", _m, "--vade_root", vade,
                      "--entity", "flags", "--attribute", "language", "--layer", str(_L // 2),
                      "--positions", "full_image"], cwd=repo, env=_env, check=False, log=_vlog)
        if _rc != 0:                                   # exits 1 if ANY site fails any check
            status[_m] = "STOPPED: verify_sites failed (verify_sites.log)"
            continue
        # 2. recognition + pruning
        _rc = stream([sys.executable, "methods/own_accuracy_prune.py", "--model_id", _m, "--vade_root", vade,
                      "--entities", *entities.value,
                      "--attributes", *[a for e in entities.value for a in _attrs[e]],
                      "--batch_size", str(int(batch.value))], cwd=repo, env=_env, check=False,
                     log=os.path.join(_out, "accuracy", "own_accuracy_prune.log"))
        for _e in entities.value:
            for _f in (f"{_e}_accuracy.json", f"{_e}_tuples_prune_summary.json"):
                _p = os.path.join(vade, "models", _slug, "stats", _f)
                if os.path.exists(_p):
                    _sh.copy(_p, os.path.join(_out, "accuracy", _f))
        if _rc != 0:
            status[_m] = "STOPPED: own_accuracy_prune failed"
            continue
        # 3 + 4. the pipeline, one entity at a time (attributes differ)
        _fails = []
        for _e in entities.value:
            _rc = stream([sys.executable, "methods/image_head_pipeline.py", "--model_id", _m, "--vade_root", vade,
                          "--entity", _e, "--attributes", *_attrs[_e], "--n_layers", str(_L), "--n_heads", str(_H),
                          "--n_rows_ceiling", "64" if _full else "16", "--n_rows", "64" if _full else "16",
                          "--scoring", "text", "--batch_size", str(int(batch.value)),
                          "--knockout_ks", *_grid, "--null_ks", *_grid, "--null_draws", "3" if _full else "2",
                          "--control_k", "1", "2", "4", "8", "16",
                          "--out_root", os.path.join(_out, "pipeline")], cwd=repo, env=_env, check=False)
            if _rc != 0:
                _fails.append(_e)
        status[_m] = "ok" if not _fails else f"pipeline reported failures for {_fails} (see log)"
        if cleanup.value:
            from huggingface_hub.constants import HF_HUB_CACHE as _cache
            _d = os.path.join(_cache, "models--" + _m.replace("/", "--"))
            if os.path.isdir(_d):
                for _sub in ("blobs", "snapshots"):
                    _sh.rmtree(os.path.join(_d, _sub), ignore_errors=True)
                print(f"freed {_d}", flush=True)
    stream([sys.executable, "methods/x15_summary.py"], cwd=repo, check=False)
    mo.md("## Run status\n\n" + "\n".join(f"- `{m}`: **{s}**" for m, s in status.items()))
    return status, x15_root


@app.cell
def _(json, mo, os, status, x15_root):
    import pandas as pd

    summaries = {}
    for _m in status:
        _p = os.path.join(x15_root, _m.split("/")[-1], "summary.json")
        if os.path.exists(_p):
            summaries[_m] = json.load(open(_p))
    _rows, _acc = [], []
    for _m, _s in summaries.items():
        for _e, _by in _s.get("accuracy", {}).items():
            for _a, _v in _by.items():
                _acc.append({"model": _s["model"], "entity": _e, "question": _a, "n": _v["n"],
                             "clean accuracy %": round(100 * _v["accuracy"], 1)})
        for _k, _r in _s["attributes"].items():
            _hd = _r.get("heads") or {}
            _nl = _hd.get("null_at_k90") or {}
            _rows.append({
                "model": _s["model"], "entity/attribute": _k,
                "source_read %": None if _r["source_read"] is None else round(100 * _r["source_read"], 1),
                "handoff": _r["handoff_blocks"], "depth": _r["handoff_depth"], "window": _r["window"],
                "image swap %": None if _hd.get("image_swap_cause") is None else round(100 * _hd["image_swap_cause"], 1),
                "k90": _hd.get("k90"), "k90 / all heads %": (None if _hd.get("k90_frac_of_model_heads") is None
                                                             else round(100 * _hd["k90_frac_of_model_heads"], 2)),
                "null@k90 mean %": None if _nl.get("mean") is None else round(100 * _nl["mean"], 1),
                "null@k90 max %": None if _nl.get("max") is None else round(100 * _nl["max"], 1),
                "k90 heads": " ".join(_hd.get("k90_heads") or []), "top-8": " ".join(_hd.get("top8_heads") or []),
                "skipped / warnings": " | ".join(filter(None, [_r.get("skipped")] + _r.get("warnings", []))),
            })
    mo.vstack([mo.md("## Recognition (clean accuracy on the questions the tuples need)"),
               mo.ui.table(pd.DataFrame(_acc), selection=None) if _acc else mo.md("none yet"),
               mo.md("## Per model: handoff, window and concentration (k90 = smallest k reaching 90% of the image swap)"),
               mo.ui.table(pd.DataFrame(_rows), selection=None) if _rows else mo.md("none yet")])
    return pd, summaries


@app.cell
def _(mo, pd, summaries):
    import altair as alt

    _long = []
    for _m, _s in summaries.items():
        for _k, _r in _s["attributes"].items():
            for _c in (_r.get("heads") or {}).get("curve", []):
                for _arm, _v in (("top-k", _c["frac_of_image"]),
                                 ("block-matched random", None if _c["null_mean"] is None or not _r["heads"]["image_swap_cause"]
                                  else _c["null_mean"] / _r["heads"]["image_swap_cause"])):
                    if _v is not None:
                        _long.append({"model": _s["model"], "attribute": _k, "arm": _arm, "k": _c["k"],
                                      "% of image swap": 100 * _v})
    _df = pd.DataFrame(_long)
    (mo.vstack([mo.md("## k-curves: top-k install vs block-matched random heads, as % of the image swap"),
                alt.Chart(_df).mark_line(point=True).encode(
                    x=alt.X("k:Q", scale=alt.Scale(type="log")), y=alt.Y("% of image swap:Q"),
                    color="arm:N", strokeDash="arm:N", tooltip=list(_df.columns))
                .properties(width=240, height=180).facet(row="model:N", column="attribute:N")])
     if not _df.empty else mo.md("No k-curves yet."))
    return


@app.cell
def _(mo, os, repo, status, x15_root):
    import io as _io
    import zipfile

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        _roots = [x15_root] + [os.path.join(repo, "logs", "image_head_pipeline_" + m.split("/")[-1]) for m in status]
        for _r in _roots:
            for _d, _, _fs in os.walk(_r):
                for _f in _fs:
                    _p = os.path.join(_d, _f)
                    z.write(_p, os.path.relpath(_p, repo))
    mo.download(data=buf.getvalue(), filename="x15_cross_model_heads.zip",
                label="download results (zip; unzip at the repo root)")
    return


if __name__ == "__main__":
    app.run()
