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
"""X7 on molab: how the asked attribute and the entity combine into the answer in the late MLPs.

Part A is a CPU toy model (no GPU). Part B runs methods/mlp_lookup.py on Qwen2.5-VL-7B (needs a
GPU). Open on molab from GitHub, then press the buttons in order. Everything comes from public
sources: this repo (cloned at the branch below) and the VADE data in the public HF bucket
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
    # X7: how does *which attribute was asked* combine with *which country* to give the answer?

    The see → move → recall account: **movers** (blocks 21–23) write *which country* into the answer
    column, **routers** (19–21) write *which attribute was asked*, and the **last-token MLPs 23–26**
    turn the two into the fact. If that is right, three things should hold:

    | prediction | test here |
    |---|---|
    | the residual entering the MLPs is a **sum**: entity vector + question vector | two-way variance decomposition over the crossed grid (countries × questions): interaction ≈ 0 before the MLPs |
    | the MLPs make it **conjunctive** (this country's this attribute) | interaction share rises in MLP neurons / MLP output at blocks 23–26 |
    | the lookup works on the sum itself | **compose**: build `entity(B) + question(A)` from runs that never showed B asked A, install it at the answer column of an unrelated run (C asked X) → does the model say **B's A**? |
    | lookup neurons are keyed by the **answer value** | **value test**: do countries sharing a language group together in a neuron *only when language is asked*? |

    **Part A** builds the same mechanism in a toy transformer (one attention layer, one MLP) on CPU, so
    the shape of the result is visible before any GPU time. **Part B** runs the real thing on Qwen.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x7", label="work dir")
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
    stream([sys.executable, "methods/mlp_lookup.py", "--vade_root", vade, "--dry_run"], cwd=repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- Part A still runs; Part B needs one'}**, "
          f"torch {torch.__version__}.")
    return gpu, repo, vade


@app.cell
def _(mo):
    mo.md(r"""
    ## Part A: the toy model (CPU, ~1 minute)

    Sequence `[entity] [pad] [pad] [relation] [query]`, target = an arbitrary table `f[entity, relation]`
    whose values are shared across entities (like languages across countries). One causal attention
    layer, one ReLU MLP, no LayerNorm, so the residual entering the MLP is exactly
    `x_query + Σ heads`. Nothing at the query knows the entity or the relation, so attention has to
    **move** both there, and since `f` is arbitrary, a linear readout of the sum cannot produce it:
    the MLP has to **look it up**.
    """)
    return


@app.cell
def _(mo):
    n_ent = mo.ui.slider(16, 128, value=64, step=8, label="entities")
    n_rel = mo.ui.slider(2, 8, value=4, label="relations")
    n_val = mo.ui.slider(2, 16, value=8, label="values per relation")
    n_steps = mo.ui.slider(500, 6000, value=3000, step=500, label="training steps")
    toy_seed = mo.ui.number(start=0, stop=99, value=0, label="seed")
    toy_btn = mo.ui.run_button(label="A. Train + analyse the toy")
    mo.vstack([mo.hstack([n_ent, n_rel, n_val]), mo.hstack([n_steps, toy_seed, toy_btn])])
    return n_ent, n_rel, n_steps, n_val, toy_btn, toy_seed


@app.cell
def _(mo, n_ent, n_rel, n_steps, n_val, repo, toy_btn, toy_seed):
    mo.stop(not toy_btn.value, mo.md("Set up first, then press **A. Train + analyse the toy**."))
    assert repo
    from methods.toy_lookup import analyze, make_task, train

    toy_task = make_task(n_ent.value, n_rel.value, n_val.value, seed=int(toy_seed.value))
    toy_model, toy_hist = train(toy_task, steps=n_steps.value, seed=int(toy_seed.value), log_every=100)
    toy = analyze(toy_task, toy_model, seed=int(toy_seed.value))
    mo.md(f"Trained: accuracy **{toy['accuracy']:.1%}** on all {n_ent.value * n_rel.value} facts.")
    return toy, toy_hist


@app.cell
def _(mo, toy, toy_hist):
    import altair as alt
    import pandas as pd

    _curve = (alt.Chart(pd.DataFrame(toy_hist), title="training").mark_line()
              .encode(x="step:Q", y=alt.Y("acc:Q", title="accuracy", scale=alt.Scale(domain=[0, 1])))
              .properties(width=260, height=180))
    _heads = pd.DataFrame([{"head": f"head {h['head']} ({h['role']})", "attends to": k, "mass": h[c]}
                           for h in toy["heads"] for k, c in (("entity", "to_entity"), ("relation", "to_relation"))])
    _hchart = (alt.Chart(_heads, title="attention at the query: movers vs routers").mark_bar()
               .encode(x=alt.X("mass:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("head:N", title=None),
                       color="attends to:N", yOffset="attends to:N")
               .properties(width=300, height=140))
    _sh = pd.DataFrame([{"site": site, "component": k, "share": v}
                        for site, key in (("1. residual entering the MLP", "resid_shares"),
                                          ("2. MLP neurons", "hidden_shares"),
                                          ("3. MLP output", "mlp_out_shares"))
                        for k, v in toy[key].items()])
    _shchart = (alt.Chart(_sh, title="variance shares over the entity x relation grid").mark_bar()
                .encode(x=alt.X("share:Q", stack="normalize"), y=alt.Y("site:N", title=None),
                        color=alt.Color("component:N", sort=["item", "attr", "inter"],
                                        legend=alt.Legend(title="item=entity, attr=relation")))
                .properties(width=360, height=120))
    _paths = pd.DataFrame([{"path": k, "accuracy": v} for k, v in toy["answer_paths"].items()])
    _pchart = (alt.Chart(_paths, title="which path holds the fact").mark_bar()
               .encode(x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("path:N", sort=None, title=None))
               .properties(width=300, height=120))
    _cross = toy["cross_question"] or {}
    mo.vstack([
        mo.hstack([_curve, _hchart], wrap=True),
        mo.hstack([_shchart, _pchart], wrap=True),
        mo.md(f"""
    **Compose:** installing the leave-cell-out additive estimate `rowmean(B) + colmean(A) − mean` as the
    MLP's input gives `f[B, A]` **{toy['compose_additive_acc']:.1%}** of the time.

    **Cross-question:** mover-head output from a donor (B, asked X) installed into (C, asked A) →
    {", ".join(f"{k}: **{v:.1%}**" for k, v in _cross.items())}.

    **Neuron classes** (dominant component per live hidden neuron): {toy['neuron_classes']}.

    How to read it: the residual entering the MLP has ~0 interaction (it is a sum), the unembedding of
    that sum alone gets only part of the table, and the MLP's write gets all of it: the MLP is where
    the conjunction is made. Note the interaction's share of *variance* stays modest even in the MLP's
    output: most of what an MLP writes is main effects, and the lookup is the small conjunctive part
    that decides the answer. Expect the same in Qwen.
    """)])
    return alt, pd


@app.cell
def _(mo):
    mo.md(r"""
    ## Part B: Qwen2.5-VL-7B (GPU)

    `methods/mlp_lookup.py --phase all`, flags, controlled prompts (prefill `Answer:`), last prompt column.
    **capture** stores the residual at layers 18–28 and the MLP neurons + output at blocks 21–27 for the
    full 84 × 4 grid. **compose** installs constructed residuals at layers 22–27 (arms below).
    **neurons** is CPU analysis of the captured arrays.

    | arm | installed at the base (C asked X) column | expect |
    |---|---|---|
    | `clean` | nothing | C.X |
    | `actual` | the real residual of the (B, A) run: the ceiling for that layer | B.A |
    | `additive` | `rowmean₋A(B) + colmean₋B(A) − mean₋B,₋A`: **no (B, A) run used** | B.A |
    | `swap_entity` | `R[C,X] + rowmean₋X(B) − rowmean₋X(C)` | B.X |
    | `swap_question` | `R[C,X] + colmean₋C(A) − colmean₋C(X)` | C.A |
    | `random_offset` | `R[C,X]` + a random vector as long as `additive − R[C,X]` | C.X |

    Readout: the first answer token (a last-column prefill patch only controls the first token), on rows
    where all four golds start with different tokens and the clean model gets all four cells right.
    Late attention still reads image C and question X, so compare `additive` against `actual` at the
    same layer, not against 100%.
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.dropdown(options=["smoke", "full"], value="smoke", label="run size")
    batch = mo.ui.number(start=1, stop=256, value=32, label="batch size")
    run_btn = mo.ui.run_button(label="2. Run mlp_lookup (capture + compose + neurons)")
    mo.hstack([mode, batch, run_btn])
    return batch, mode, run_btn


@app.cell
def _(batch, gpu, mo, mode, os, repo, run_btn, stream, sys, vade):
    mo.stop(not run_btn.value, mo.md("Choose a size and press **2. Run**. Smoke: 12 flags, 48 compose rows, "
                                     "layers 23-25 (a few minutes plus the ~16 GB model download). Full: 84 flags, "
                                     "384 rows, layers 22-27 (~20-40 min)."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    out_dir = os.path.join(repo, "results", "mlp_lookup", f"flags_{mode.value}")
    _cmd = [sys.executable, "methods/mlp_lookup.py", "--phase", "all", "--vade_root", vade, "--out_dir", out_dir,
            "--batch_size", str(int(batch.value))]
    if mode.value == "smoke":
        _cmd += ["--limit_items", "12", "--n_rows", "48", "--layers", "23", "24", "25", "--n_perm", "50"]
    stream(_cmd, cwd=repo, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    mo.md(f"Done: `{out_dir}`")
    return (out_dir,)


@app.cell
def _(alt, json, mo, os, out_dir, pd):
    report = json.load(open(os.path.join(out_dir, "summary.json")))
    _rows = []
    for _L, _arms in report["compose"].items():
        for _arm, _s in _arms.items():
            _rows.append({"layer": int(_L), "arm": _arm, "expect": _s["expect"], "n": _s["n"],
                          "hit (expected target)": 100 * _s["first_" + _s["expect"]],
                          **{f"{t} %": round(100 * _s["first_" + t], 1) for t in ("B.A", "C.X", "B.X", "C.A")},
                          "other %": round(100 * _s["first_other"], 1)})
    compose_df = pd.DataFrame(_rows)
    _chart = (alt.Chart(compose_df, title="compose: first answer token = the arm's expected target")
              .mark_line(point=True)
              .encode(x=alt.X("layer:O", title="layer installed at (residual after block L-1)"),
                      y=alt.Y("hit (expected target):Q", title="% hit", scale=alt.Scale(domain=[0, 100])),
                      color=alt.Color("arm:N", sort=["clean", "actual", "additive", "swap_entity", "swap_question",
                                                     "random_offset"]),
                      tooltip=list(compose_df.columns))
              .properties(width=560, height=280))
    mo.vstack([mo.md("## Results\n### Compose"), _chart,
               mo.ui.table(compose_df.drop(columns=["hit (expected target)"]), selection=None),
               mo.md(f"Clean accuracy over the grid: "
                     f"{', '.join(f'{a} {v:.0%}' for a, v in report.get('clean_accuracy', {}).items())}")])
    return (report,)


@app.cell
def _(alt, mo, pd, report):
    _nr = report["neurons"]
    _rows = []
    for _site, _key in (("residual (layer L)", "resid_shares"), ("MLP neurons (block b)", "mlp_hidden_shares"),
                        ("MLP output (block b)", "mlp_output_shares")):
        for _x, _s in _nr[_key].items():
            _rows.append({"site": _site, "depth": int(_x), "interaction share": _s["inter"],
                          "item share": _s["item"], "attribute share": _s["attr"]})
    _df = pd.DataFrame(_rows)
    _chart = (alt.Chart(_df, title="item x attribute interaction share by depth").mark_line(point=True)
              .encode(x=alt.X("depth:Q", title="layer L (residual) / block b (MLP)"),
                      y=alt.Y("interaction share:Q"), color="site:N", tooltip=list(_df.columns))
              .properties(width=560, height=260))
    _classes = pd.DataFrame(_nr["neuron_classes"]).T.rename_axis("block").reset_index()
    _vt = pd.DataFrame([{"block": b, "attribute": a, "eta2 diff (asked - other)": round(v["mean_diff"], 3),
                         "null 95%": round(v["null_p95"], 3), "p": round(v["p_value"], 3),
                         "frac neurons asked>other": round(v["frac_asked_gt_other"], 2),
                         "items / values": f"{v['n_items']} / {v['n_values']}"}
                        for b, d in _nr["value_test"].items() for a, v in d.items() if v])
    mo.vstack([mo.md("### Additive in, conjunctive out?"), _chart,
               mo.md("Expect: the residual's interaction stays near 0 through layer ~22 (R13.1: ≤3.4%), "
                     "then rises; MLP output at blocks 24-26 carries most of it."),
               mo.md("### Neuron classes among each block's top-k highest-variance neurons"),
               mo.ui.table(_classes, selection=None),
               mo.md("### Value test: are the top interaction neurons keyed by the answer value?\n"
                     "eta² of the value (countries grouped by shared language / EUR / calling zone) when that "
                     "attribute is asked, minus when another is asked. Above the permutation 95% = keyed by "
                     "the answer, not by the country."),
               mo.ui.table(_vt, selection=None)])
    return


@app.cell
def _(alt, mo, np_load, os, out_dir, pd):
    _p = os.path.join(out_dir, "heatmap.npz")
    mo.stop(not os.path.exists(_p), mo.md("No heatmap (the --heatmap_block was not captured)."))
    _h = np_load(_p)
    _z, _items, _langs, _attrs = _h["z"], list(_h["items"]), list(_h["languages"]), list(_h["attributes"])
    _rows = [{"neuron": f"n{int(n)}", "attribute": _attrs[a], "item": f"{_langs[i]} | {_items[i]}",
              "z": float(_z[i, a, j])}
             for i in range(len(_items)) for a in range(len(_attrs)) for j, n in enumerate(_h["neurons"])]
    _chart = (alt.Chart(pd.DataFrame(_rows), title=f"top interaction neurons, block {int(_h['block'])} "
                                                   f"(countries sorted by language)")
              .mark_rect()
              .encode(x=alt.X("item:N", sort=None, axis=alt.Axis(labels=False, ticks=False)),
                      y=alt.Y("neuron:N", sort=None),
                      color=alt.Color("z:Q", scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True)),
                      column=alt.Column("attribute:N", title="question asked"),
                      tooltip=["item", "neuron", "attribute", "z"])
              .properties(width=230, height=420))
    mo.vstack([mo.md("### Heatmap"), _chart,
               mo.md("A lookup neuron lights up for one *group* of countries under one question only, e.g. "
                     "the Spanish-speaking block under `language` but not under `capital`.")])
    return


@app.cell
def _():
    import numpy as np

    def np_load(path):
        return np.load(path, allow_pickle=False)

    return (np_load,)


@app.cell
def _(mo, os, out_dir):
    _names = [n for n in ("summary.json", "compose_rows.jsonl", "index.jsonl", "heatmap.npz")
              if os.path.exists(os.path.join(out_dir, n))]
    _files = {n: open(os.path.join(out_dir, n), "rb").read() for n in _names}
    mo.vstack([mo.md("### Downloads (the .npy captures stay on the machine: ~265 MB)"),
               mo.hstack([mo.download(data=b, filename=f"x7_{n}", label=f"download {n}") for n, b in _files.items()])])
    return


if __name__ == "__main__":
    app.run()
