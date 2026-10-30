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
    model = mo.ui.dropdown(options={label: mid for mid, label in MODELS.items()}, value="Gemma-3 12B",
                           label="model (one per run)")
    entities = mo.ui.multiselect(options=["flags", "brands"], value=["flags"], label="entities")
    flag_attrs = mo.ui.multiselect(options=["language", "capital", "currency", "calling_code"],
                                   value=["language"], label="flags attributes")
    brand_attrs = mo.ui.multiselect(options=["hq_country", "industry", "founded_year"], value=["hq_country"],
                                    label="brands attributes")
    size = mo.ui.dropdown(options={"smoke (16 rows, coarse k)": "smoke", "full (64 rows, fine k)": "full"},
                          value="full (64 rows, fine k)", label="run size")
    batch = mo.ui.number(start=1, stop=64, value=16, label="batch size")
    cleanup = mo.ui.checkbox(value=False, label="delete the model's weights after stage 5 (disk)")
    mo.vstack([mo.md("## 2. Run one model, stage by stage"),
               mo.md("Each stage below has its own button and ends with a **download of everything written so far**. "
                     "Download after every stage: if the sandbox is reclaimed, relaunch, set up, **restore** the last zip "
                     "(cell below) and press the next stage's button. Every stage skips work whose output exists."),
               model, mo.hstack([entities, flag_attrs, brand_attrs]), mo.hstack([size, batch, cleanup])])
    return batch, brand_attrs, cleanup, entities, flag_attrs, model, size


@app.cell
def _(batch, brand_attrs, entities, flag_attrs, json, mo, model, os, repo, size, stream, sys, vade):
    import io as _io
    import zipfile
    import subprocess as _sp

    from transformers import AutoConfig

    slug = model.value.split("/")[-1]
    _tc = AutoConfig.from_pretrained(model.value).text_config
    n_layers, n_heads = _tc.num_hidden_layers, _tc.num_attention_heads
    x15_root = os.path.join(repo, "results", "x15_cross_model_heads")
    out = os.path.join(x15_root, slug)
    os.makedirs(os.path.join(out, "accuracy"), exist_ok=True)
    full = size.value == "full"
    attrs = {"flags": flag_attrs.value, "brands": brand_attrs.value}
    # MALLOC_ARENA_MAX: fewer glibc malloc arenas, so a stage's host RSS does not balloon while it loads
    # 24 GB of weights (the two reclaimed runs both died on Gemma-3-12B with 32 GiB of host RAM).
    env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade, MALLOC_ARENA_MAX="2")
    grid = ["1", "2", "3", "4", "6", "8", "12", "16", "24", "32", "48", "64", "96"] if full else ["1", "2", "4", "8", "16"]

    def resources(tag):
        """free / df / nvidia-smi into <out>/resources.log, so a later reclaim leaves evidence of why."""
        _cmds = (["free", "-m"], ["df", "-h", "/tmp", os.path.expanduser("~/.cache")],
                 ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])
        with open(os.path.join(out, "resources.log"), "a") as _f:
            _f.write(f"\n===== {tag}\n")
            for _c in _cmds:
                try:
                    _f.write(_sp.run(_c, capture_output=True, text=True, timeout=30).stdout)
                except Exception as _e:
                    _f.write(f"{_c[0]}: {_e}\n")

    def pipeline(stages):
        """image_head_pipeline.py with identical arguments at every stage (it resumes from existing outputs)."""
        fails = []
        for _e in entities.value:
            _rc = stream([sys.executable, "methods/image_head_pipeline.py", "--model_id", model.value,
                          "--vade_root", vade, "--entity", _e, "--attributes", *attrs[_e],
                          "--n_layers", str(n_layers), "--n_heads", str(n_heads),
                          "--n_rows_ceiling", "64" if full else "16", "--n_rows", "64" if full else "16",
                          "--scoring", "text", "--batch_size", str(int(batch.value)),
                          "--knockout_ks", *grid, "--null_ks", *grid, "--null_draws", "3" if full else "2",
                          "--control_k", "1", "2", "4", "8", "16",
                          "--stages", *stages, "--out_root", os.path.join(out, "pipeline")],
                         cwd=repo, env=env, check=False)
            if _rc:
                fails.append(_e)
        return fails

    def bundle(stage):
        """-> a download of EVERYTHING this model has written so far: results, pipeline/stage logs, and its
        pruned tuples + accuracy stats under VADE/models/<slug> (stored as _VADE/... so restore puts them back)."""
        stream([sys.executable, "methods/x15_summary.py", "--models", slug], cwd=repo, check=False)
        _buf = _io.BytesIO()
        _n = 0
        with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
            for _r, _base, _pre in ((out, repo, ""),
                                    (os.path.join(repo, "logs", "image_head_pipeline_" + slug), repo, ""),
                                    (os.path.join(repo, "logs", slug), repo, ""),
                                    (os.path.join(vade, "models", slug), vade, "_VADE/")):
                for _d, _, _fs in os.walk(_r):
                    for _f in _fs:
                        _p = os.path.join(_d, _f)
                        _z.write(_p, _pre + os.path.relpath(_p, _base))
                        _n += 1
        _z_name = f"x15_{slug}_after_{stage}.zip"
        return mo.download(data=_buf.getvalue(), filename=_z_name,
                           label=f"download {_z_name} ({_n} files, {len(_buf.getvalue()) / 1e6:.1f} MB)")

    mo.md(f"Model **{model.value}**: {n_layers} blocks x {n_heads} heads; outputs in `{out}`.")
    return bundle, env, n_layers, out, pipeline, resources, slug, x15_root


@app.cell
def _(mo):
    restore_file = mo.ui.file(filetypes=[".zip"], kind="area", label="restore a stage zip from an earlier sandbox")
    restore_btn = mo.ui.run_button(label="Restore uploaded zip")
    mo.vstack([mo.md("### Restore (only after a reclaim)"), restore_file, restore_btn])
    return restore_btn, restore_file


@app.cell
def _(mo, os, repo, restore_btn, restore_file, vade):
    mo.stop(not restore_btn.value, mo.md("Nothing restored."))
    import io as _io2
    import zipfile as _zf

    mo.stop(not restore_file.value, mo.md("**Upload a zip first.**"))
    _n = 0
    with _zf.ZipFile(_io2.BytesIO(restore_file.contents())) as _z:
        for _name in _z.namelist():
            _dst_root, _rel = (vade, _name[len("_VADE/"):]) if _name.startswith("_VADE/") else (repo, _name)
            _dst = os.path.normpath(os.path.join(_dst_root, _rel))
            assert _dst.startswith(os.path.normpath(_dst_root) + os.sep), f"refusing {_name}"
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            with open(_dst, "wb") as _f:
                _f.write(_z.read(_name))
            _n += 1
    mo.md(f"Restored **{_n} files** from `{restore_file.name()}`. Press the next stage's button.")
    return


@app.cell
def _(mo):
    s1_btn = mo.ui.run_button(label="Stage 1/5: verify_sites self-checks")
    s1_btn
    return (s1_btn,)


@app.cell
def _(bundle, env, gpu, mo, model, n_layers, os, out, repo, resources, s1_btn, stream, sys, tests_rc, vade):
    mo.stop(not s1_btn.value, mo.md("Press **Stage 1/5**."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The adapter tests failed.** Fix that first."))
    _log = os.path.join(out, "verify_sites.log")
    if os.path.exists(_log) and open(_log).read().count("all checks passed") >= 5:
        _rc, _note = 0, "already passed (log restored); skipped"
    else:
        resources("before verify_sites")
        if os.path.exists(_log):
            os.remove(_log)
        _rc = stream([sys.executable, "methods/ndm/verify_sites.py", "--model_id", model.value, "--vade_root", vade,
                      "--entity", "flags", "--attribute", "language", "--layer", str(n_layers // 2),
                      "--positions", "full_image"], cwd=repo, env=env, check=False, log=_log)
        _note = "all sites passed" if _rc == 0 else "**FAILED -- do not continue; read verify_sites.log**"
    mo.vstack([mo.md(f"Stage 1 (verify_sites): {_note}"), bundle("stage1_verify")])
    return


@app.cell
def _(mo):
    s2_btn = mo.ui.run_button(label="Stage 2/5: recognition + pruning")
    s2_btn
    return (s2_btn,)


@app.cell
def _(batch, brand_attrs, bundle, entities, env, flag_attrs, json, mo, model, os, out, repo, resources, s2_btn,
      slug, stream, sys, vade):
    mo.stop(not s2_btn.value, mo.md("Press **Stage 2/5**."))
    import shutil as _sh

    resources("before own_accuracy_prune")
    _attrs = {"flags": flag_attrs.value, "brands": brand_attrs.value}
    # Resumable by itself: generations already in <vade>/models/<slug>/stats/*_predictions.jsonl are skipped.
    _rc = stream([sys.executable, "methods/own_accuracy_prune.py", "--model_id", model.value, "--vade_root", vade,
                  "--entities", *entities.value, "--attributes", *[a for e in entities.value for a in _attrs[e]],
                  "--batch_size", str(int(batch.value))], cwd=repo, env=env, check=False,
                 log=os.path.join(out, "accuracy", "own_accuracy_prune.log"))
    _acc = []
    for _e in entities.value:
        for _f in (f"{_e}_accuracy.json", f"{_e}_tuples_prune_summary.json"):
            _p = os.path.join(vade, "models", slug, "stats", _f)
            if os.path.exists(_p):
                _sh.copy(_p, os.path.join(out, "accuracy", _f))
        _p = os.path.join(out, "accuracy", f"{_e}_accuracy.json")
        if os.path.exists(_p):
            _by = json.load(open(_p))["by"]
            _acc.append(f"{_e}: " + ", ".join(f"{k} {100 * v['accuracy']:.1f}%" for k, v in _by.items() if "/" not in k))
    mo.vstack([mo.md(f"Stage 2 (recognition + pruning): {'ok' if _rc == 0 else '**FAILED**'}\n\n"
                     + "\n".join(f"- {a}" for a in _acc)), bundle("stage2_prune")])
    return


@app.cell
def _(mo):
    s3_btn = mo.ui.run_button(label="Stage 3/5: handoff curves (ceiling sweep)")
    s3_btn
    return (s3_btn,)


@app.cell
def _(bundle, mo, pipeline, resources, s3_btn):
    mo.stop(not s3_btn.value, mo.md("Press **Stage 3/5**."))
    resources("before pipeline stage ceiling")
    _fails = pipeline(["ceiling"])
    mo.vstack([mo.md(f"Stage 3 (handoff curves): {'ok' if not _fails else f'**failures for {_fails}** (see log)'}"),
               bundle("stage3_ceiling")])
    return


@app.cell
def _(mo):
    s4_btn = mo.ui.run_button(label="Stage 4/5: block-window sweep")
    s4_btn
    return (s4_btn,)


@app.cell
def _(bundle, mo, pipeline, resources, s4_btn):
    mo.stop(not s4_btn.value, mo.md("Press **Stage 4/5**."))
    resources("before pipeline stage windows")
    _fails = pipeline(["windows"])
    mo.vstack([mo.md(f"Stage 4 (window sweep): {'ok' if not _fails else f'**failures for {_fails}** (see log)'}"),
               bundle("stage4_windows")])
    return


@app.cell
def _(mo):
    s5_btn = mo.ui.run_button(label="Stage 5/5: head_trace + block-matched random heads")
    s5_btn
    return (s5_btn,)


@app.cell
def _(bundle, cleanup, mo, model, os, pipeline, resources, s5_btn):
    mo.stop(not s5_btn.value, mo.md("Press **Stage 5/5**."))
    import shutil as _sh5

    resources("before pipeline stage heads")
    _fails = pipeline(["heads"])
    s5_download = bundle("stage5_heads")
    if cleanup.value:
        from huggingface_hub.constants import HF_HUB_CACHE as _cache
        _d = os.path.join(_cache, "models--" + model.value.replace("/", "--"))
        for _sub in ("blobs", "snapshots"):
            _sh5.rmtree(os.path.join(_d, _sub), ignore_errors=True)
    s5_status = "ok" if not _fails else f"failures for {_fails}"
    mo.vstack([mo.md(f"Stage 5 (heads): **{s5_status}**"), s5_download])
    return (s5_status,)


@app.cell
def _(json, mo, os, s5_status, slug, x15_root):
    import pandas as pd

    _ = s5_status
    summaries = {}
    _p = os.path.join(x15_root, slug, "summary.json")
    if os.path.exists(_p):
        summaries[slug] = json.load(open(_p))
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
               mo.md("## Handoff, window and concentration (k90 = smallest k reaching 90% of the image swap)"),
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
    (mo.vstack([mo.md("## k-curve: top-k install vs block-matched random heads, as % of the image swap"),
                alt.Chart(_df).mark_line(point=True).encode(
                    x=alt.X("k:Q", scale=alt.Scale(type="log")), y=alt.Y("% of image swap:Q"),
                    color="arm:N", strokeDash="arm:N", tooltip=list(_df.columns))
                .properties(width=320, height=200).facet(column="attribute:N")])
     if not _df.empty else mo.md("No k-curves yet."))
    return


if __name__ == "__main__":
    app.run()
