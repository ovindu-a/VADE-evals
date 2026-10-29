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
#     "scikit-learn",
#     "pytest",
#     "matplotlib",
#     "huggingface_hub>=1.24",
# ]
# ///
"""X11-X14 on molab: do the paper's headline results hold under the checks reviewers will ask for?

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
    # X11-X14: do the headline results hold?

    Four checks, one per claim the paper leans on (Qwen2.5-VL-7B; PAPER_FLOWS.md). Each is a thin
    wrapper over an existing script; only X11 and X12 needed new options.

    | # | Claim under test | Check | It HOLDS if |
    |---|---|---|---|
    | **X11** | Zeroing the 10 movers costs ~51 pp; random heads <= 1.3 pp (P2) | `vrh_ablation --ablation zero mean`, **10** block-matched random draws. Mean = each head's clean mean over items at the same (template, column, decode step) | `mean:common10` still drops far more than every `mean:...random` draw |
    | **X12** | From block ~24 the answer position builds the value without attending elsewhere (X10 pilot, 16 flags) | `attn_cutoff` on **all 84 flags, 128 brands, 84 celebrities** | `attn_self` keeps most clean-correct answers from L24 while `mlp_zero` breaks them |
    | **X13** | Cross-modal mover interchange (91.7%) and mover+router composition (78.8%) are not flags-only | X2b rerun on brands / celebrities with **common10** instead of each entity's weak top-8 (routers unchanged: top-8 of the same traces) | crossmodal transfer >> 30-47% and cross ~= same; composition >> 0% / 15.6% |
    | **X14** | The same edit is attribute-selective at MLPs 23-26 but swaps the whole entity at the movers (E3) | `readout_jacobian` heads vs `mlp_hidden 23-26` on **celebrities** | iso(MLP) >> iso(heads) at comparable cause |

    Run **smoke** first (minutes), check the logs, then **full**.
    """)
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/ovindu-a/VADE-evals.git", label="repo", full_width=True)
    repo_branch = mo.ui.text(value="molab", label="branch")
    bucket = mo.ui.text(value="ovindu-a/counterfact-data", label="HF bucket (VADE data)")
    workdir = mo.ui.text(value="/tmp/x11", label="work dir")
    mo.vstack([repo_url, mo.hstack([repo_branch, bucket, workdir])])
    return bucket, repo_branch, repo_url, workdir


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time

    def stream(cmd, cwd=None, env=None, check=True, log=None):
        """Run `cmd`, echoing its output live (and into `log`); raise on failure unless check=False."""
        print("$", " ".join(cmd), flush=True)
        fh = open(log, "w") if log else None
        if fh:
            fh.write("$ " + " ".join(cmd) + "\n")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            if fh:
                fh.write(line)
                fh.flush()
        rc = proc.wait()
        msg = f"[exit {rc} after {(time.time() - t0) / 60:.1f} min]\n"
        print(msg, flush=True)
        if fh:
            fh.write(msg)
            fh.close()
        if check and rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")
        return rc

    COMMON10 = "21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
    return COMMON10, json, os, stream, sys


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up (clone repo + pull data + tests + dry runs)")
    setup_btn
    return (setup_btn,)


@app.cell
def _(COMMON10, bucket, mo, os, repo_branch, repo_url, setup_btn, stream, sys, workdir):
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
    stream(["git", "log", "--oneline", "-1"], cwd=repo)
    hf_cli = os.path.join(os.path.dirname(sys.executable), "hf")
    hf_cli = hf_cli if os.path.exists(hf_cli) else (shutil.which("hf") or "hf")
    stream([hf_cli, "buckets", "sync", f"hf://buckets/{bucket.value}", vade])
    # pyvene: imported by methods/dbm. --no-deps keeps transformers on the pinned 5.16.
    if stream([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pyvene"], check=False) != 0:
        stream([shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "--no-deps", "pyvene"])
    _env = dict(os.environ, VADE_ROOT=vade)
    tests_rc = stream([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_x11_mean_ablation.py",
                       "tests/test_attn_cutoff.py", "tests/test_tier3_heads.py"], cwd=repo, env=_env, check=False)
    _dry = [["methods/vrh_ablation.py", "--lists", "park_vrh", "--n_random", "0", "--n_random_movers", "10",
             "--ablation", "zero", "mean"]]
    _dry += [["methods/attn_cutoff.py", "--entity", e, "--n_items", "200"] for e in ("flags", "brands", "celebrities")]
    _dry += [["methods/circuit_compose.py", "--entity", e, "--movers", COMMON10,
              "--router_trace", f"results/paper/x2b_attr_trace_{e}.json"] for e in ("brands", "celebrities")]
    _dry += [["methods/text_entity_contrast.py", "--entity", e, "--movers", COMMON10, "--phases", "crossmodal"]
             for e in ("brands", "celebrities")]
    _dry += [["methods/readout_jacobian.py", "--entity", "celebrities", "--phase", "all"],
             ["methods/readout_jacobian.py", "--entity", "celebrities", "--phase", "all", "--site", "mlp_hidden",
              "--blocks", "23-26", "--max_rank", "128"]]
    dry_fail = [c[0] + " " + " ".join(c[1:4]) for c in _dry
                if stream([sys.executable, *c, "--vade_root", vade, "--dry_run"], cwd=repo, env=_env, check=False)]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.md(f"Set up in `{root}`. GPU: **{gpu or 'none -- attach one in the notebook specs'}**, "
          f"torch {torch.__version__}. Tests: **{'passed' if tests_rc == 0 else 'FAILED'}**. Dry runs: "
          + ("**all valid**" if not dry_fail else "**FAILED:** " + "; ".join(f"`{d}`" for d in dry_fail)))
    return dry_fail, gpu, repo, tests_rc, vade


@app.cell
def _(mo):
    exps = mo.ui.multiselect(options=["X11 mean ablation", "X12 attention cut-off", "X13 common10 compose",
                                      "X13 common10 crossmodal", "X14 head vs MLP edit"],
                             value=["X11 mean ablation", "X12 attention cut-off", "X13 common10 crossmodal",
                                    "X13 common10 compose", "X14 head vs MLP edit"], label="experiments")
    size = mo.ui.dropdown(options=["smoke", "full"], value="smoke", label="run size")
    bs = mo.ui.number(start=4, stop=128, value=32, label="batch size")
    run_btn = mo.ui.run_button(label="2. Run")
    mo.vstack([exps, mo.hstack([size, bs]), run_btn])
    return bs, exps, run_btn, size


@app.cell
def _(COMMON10, bs, gpu, mo, os, repo, run_btn, size, stream, sys, tests_rc, exps, vade):
    mo.stop(not run_btn.value, mo.md("Press **2. Run**. smoke = minutes; full = a few hours in total."))
    mo.stop(gpu is None, mo.md("**No GPU.** Attach one in the notebook specs, then rerun set up."))
    mo.stop(tests_rc != 0, mo.md("**The tests failed.** Fix that first: every number would be suspect."))
    full = size.value == "full"
    B = str(int(bs.value))
    R = os.path.join("results", "x11_hold_checks", size.value)
    jobs = []
    if "X11 mean ablation" in exps.value:
        jobs.append(("x11_flags", ["methods/vrh_ablation.py", "--lists", "park_vrh", "--n_random", "0",
                                   "--n_random_movers", "10" if full else "2", "--ablation", "zero", "mean",
                                   "--index_items", "0", "--batch_size", B]
                     + ([] if full else ["--limit_items", "8", "--n_templates", "1"])))
    if "X12 attention cut-off" in exps.value:
        for _e, _n in (("flags", 84), ("brands", 128), ("celebrities", 84)):
            jobs.append((f"x12_{_e}", ["methods/attn_cutoff.py", "--entity", _e, "--batch_size", B,
                                       "--n_items", str(_n if full else 8), "--layer_step", "2" if full else "6"]))
    if "X13 common10 crossmodal" in exps.value:
        for _e in ("brands", "celebrities"):
            jobs.append((f"x13_crossmodal_{_e}", ["methods/text_entity_contrast.py", "--entity", _e, "--movers", COMMON10,
                                                  "--phases", "crossmodal", "--n_pairs", "128" if full else "8",
                                                  "--batch_size", B]))
    if "X13 common10 compose" in exps.value:
        for _e in ("brands", "celebrities"):
            jobs.append((f"x13_compose_{_e}", ["methods/circuit_compose.py", "--entity", _e, "--movers", COMMON10,
                                               "--router_trace", f"results/paper/x2b_attr_trace_{_e}.json",
                                               "--router_k", "8", "--n_items", "32" if full else "2",
                                               "--arms", "clean", "movers", "routers", "compose", "compose_random",
                                               "--batch_size", B]))
    if "X14 head vs MLP edit" in exps.value:
        _lim = [] if full else ["--items_limit", "8", "--limit_pairs", "8"]
        _att = ["--attributes", "nationality", "occupation", "native_language"]   # years: first token is a bare space
        jobs.append(("x14_celebrities_heads", ["methods/readout_jacobian.py", "--entity", "celebrities", *_att,
                                               "--phase", "all", "--batch_size", B] + _lim))
        jobs.append(("x14_celebrities_mlp", ["methods/readout_jacobian.py", "--entity", "celebrities", *_att, "--phase", "all",
                                             "--site", "mlp_hidden", "--blocks", "23-26", "--max_rank", "128",
                                             "--batch_size", B] + _lim))
    status = {}
    for _name, _cmd in jobs:
        _out = os.path.join(repo, R, _name)
        os.makedirs(_out, exist_ok=True)
        _env = dict(os.environ, PYTHONUNBUFFERED="1", VADE_ROOT=vade)
        status[_name] = stream([sys.executable, *_cmd, "--vade_root", vade, "--out_dir", _out], cwd=repo, env=_env,
                               check=False, log=os.path.join(_out, "run.log"))
    out_root = os.path.join(repo, R)
    mo.md("Finished: " + ", ".join(f"`{k}` {'ok' if v == 0 else f'**FAILED ({v})**'}" for k, v in status.items()))
    return out_root, status


@app.cell
def _(mo, os, out_root, status):
    _blocks = []
    for _name in status:
        _log = os.path.join(out_root, _name, "run.log")
        _lines = open(_log).read().splitlines() if os.path.exists(_log) else ["(no log)"]
        _blocks += [mo.md(f"### `{_name}` (exit {status[_name]})"), mo.plain_text("\n".join(_lines[-70:]))]
    mo.vstack([mo.md("## Log tails (the scripts print their own summary tables)"), *_blocks])
    return


@app.cell
def _(json, mo, os, out_root, status):
    # X11: the comparison the claim needs, computed from summary.json.
    _p = os.path.join(out_root, "x11_flags", "summary.json")
    mo.stop(not os.path.exists(_p), mo.md(""))
    _s = json.load(open(_p))["summary"]
    _clean = _s["clean"]["accuracy"]
    _lines = [f"clean accuracy {_clean:.1%}  (n={_s['clean']['n']})"]
    for _kind in ("zero", "mean"):
        if f"{_kind}:common10" not in _s:
            continue
        _m = _s[f"{_kind}:common10"]["accuracy"]
        _r = [v["accuracy"] for k, v in _s.items() if k.startswith(f"{_kind}:common10:random")]
        _lines.append(f"{_kind:>4}: movers {_m:.1%} (drop {100 * (_clean - _m):.1f} pp) | {len(_r)} random draws "
                      f"{min(_r):.1%}-{max(_r):.1%} (max drop {100 * (_clean - min(_r)):.1f} pp) | by attribute: "
                      + " ".join(f"{a[:8]} {v:.0%}" for a, v in _s[f'{_kind}:common10']['by_attribute'].items()))
    mo.vstack([mo.md("## X11 verdict table"), mo.plain_text("\n".join(_lines))])
    return


@app.cell
def _(mo, os, out_root, status):
    import io
    import zipfile

    _buf = io.BytesIO()
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for _name in status:
            _d = os.path.join(out_root, _name)
            for _dp, _, _fs in os.walk(_d):
                for _f in _fs:
                    if _f.endswith((".json", ".jsonl", ".log", ".md")):
                        _z.write(os.path.join(_dp, _f), os.path.relpath(os.path.join(_dp, _f), out_root))
    mo.download(data=_buf.getvalue(), filename=f"x11_hold_checks_{os.path.basename(out_root)}.zip",
                label="download results (zip)")
    return


if __name__ == "__main__":
    app.run()
