# GPU runbook: Tier 1 + Tier 2 on an A100 80GB

Instructions for the Claude session running on the hosted GPU. Read all of this before
running anything. `CLAUDE.md` in this repo explains every script; this file only says
what to run, in what order, what must hold, and what to report back.

## Your job

1. Set the pod up and verify it (section 1).
2. Run the two smoke tests and check them (section 2).
3. Run Tier 1 and Tier 2 in full, unattended, inside tmux (section 3).
4. Check every self-check listed in section 4 as each step finishes.
5. Write `RUN_REPORT.md` and package the outputs (section 5).

**Ground rules**

- **Do not edit anything under `methods/`, `tests/` or `scripts/`, and do not change any
  hyperparameter, layer, head list or row count.** These scripts were designed and
  reviewed off-pod. A failed self-check means something is wrong, and patching the code
  to make it pass destroys the evidence. Stop that step, record what happened, and report.
- **The one allowed change:** if a step runs out of GPU memory, rerun just that step
  without `GPU=a100` (section 3).
- Never delete or overwrite anything under `results/` or `logs/` that existed before you
  started. Every step writes to its own directory, so nothing should need deleting.
- Do not `git commit` or `git push` unless the user tells you to.
- If something is ambiguous, stop and ask the user. Do not guess.

## 0. Continuing on an existing pod (second session, 2026-10-14)

If the pod was already set up and E1, E2 and E6 have finished (commit `607b957`), do NOT
redo sections 1-2 or rerun those three. Instead:

```bash
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home
git pull                                  # brings the E5 fix and the larger A100 profile
python -m pytest -q tests/                # expect 0 failures
tmux new -s runs2
GPU=a100 bash scripts/run_tier1.sh e5 e4 e3 && GPU=a100 bash scripts/run_tier2.sh all
```

What changed since the first session:

- **E5 was failing its own no-op check** (`an empty knockout changed the logits`, max
  |dlogit| 1.31). The cause was a bf16 renormalization in `EdgeKnockout`, now fixed and
  covered by a test. The check must now print `max |dlogit| = 0.00e+00`. Anything else:
  stop E5 and report it.
- **The `GPU=a100` profile is larger.** Batch sizes go up to 64-128, and sample sizes
  double for E5 (128 pairs), E4 (64 base countries), E7 (128 pairs) and E11 (96 rows).
  E2 and E6 keep their finished 64-pair results. If a step runs out of memory, rerun it
  without `GPU=a100`.
- Expected time for this command: about 5.5-7.5 h (Tier 1 remainder ~2-2.5 h, Tier 2
  ~3.5-5 h).

Then do sections 4 and 5 as written, for the steps run in this session.

## 1. Setup

Work in the pod's large data volume (usually `/workspace`), not the root disk. The three
models need about 40 GB (Qwen2.5-VL-7B 16.6 GB, Qwen2.5-VL-3B 7.5 GB, InternVL3-8B ~16 GB).

```bash
cd /workspace
git clone https://github.com/Shaveen12/VADE.git
git clone https://github.com/ovindu-a/VADE-evals.git
mkdir -p /workspace/hf_home
export HF_HOME=/workspace/hf_home          # add this to ~/.bashrc too: every new shell needs it
```

The two repos MUST sit next to each other (`/workspace/VADE`, `/workspace/VADE-evals`);
every script finds VADE as `../VADE`.

Pin both repos to the commits everything was validated against, then confirm:

```bash
cd /workspace/VADE && git checkout b8dfaf9
cd /workspace/VADE-evals && git log --oneline -1   # must be ac2a7f1 or later
```

If VADE-evals is OLDER than `ac2a7f1`, stop: the run would fail against the new VADE.

Python environment. If the pod image already ships a CUDA build of torch, keep it and drop
`torch torchvision` from the first line.

```bash
cd /workspace/VADE-evals
pip install torch torchvision "transformers==5.16.*" accelerate scikit-learn pillow numpy tqdm huggingface_hub joblib pytest
pip install --no-deps pyvene
```

`pyvene` is installed with `--no-deps` so it cannot downgrade transformers. Check it imports:
`python -c "import pyvene"`. If that fails, install only the module the error names.

Verify:

```bash
nvidia-smi                                           # an A100 80GB, nothing else using it
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
python -m pytest -q tests/                           # expect 0 failures (no GPU used); a few tokenizer tests SKIP until the model is downloaded
```

If `python` is not the interpreter you installed into, prefix every launcher command
below with `PY=/path/to/python`.

## 2. Smoke tests (~20-35 min including the 7B download)

```bash
tmux new -s runs
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home
GPU=a100 bash scripts/run_tier1.sh smoke && GPU=a100 bash scripts/run_tier2.sh smoke
```

Smoke runs use tiny sizes, so their NUMBERS mean nothing. Check only that:

- every step prints `-> [smoke_...] done in N min`, with no traceback;
- `smoke_e9` (vrh_ablation) prints an INDEX CHECK block and a results table;
- `smoke_e7` (head_severed) prints `self-checks: {... 'identity': 'bit-exact' ...}`;
- `smoke_e8` is a dry run: it should print the stage-1 `ceiling_sweep` command for
  InternVL and stop. That is expected.

Smoke outputs go to `results/*/smoke` and `logs/tier{1,2}/smoke_*`. Leave them in place.

## 3. Full runs (~6-7.7 h total)

In the same tmux session:

```bash
GPU=a100 bash scripts/run_tier1.sh all && GPU=a100 bash scripts/run_tier2.sh all
```

- `GPU=a100` raises batch sizes for the 80 GB card. It changes only how many rows share a
  forward pass, never what is computed. `head_das` keeps its effective batch of 64 rows
  per update. Without `GPU=a100`, each script runs at its own 24 GB default.
- Each step loads the model, runs, and prints `-> [step] done in N min (log path)`.
  Logs: `logs/tier1/<step>_<stamp>.log` and `logs/tier2/...`.
- **The launchers stop at the first failing step** (`set -euo pipefail`). If that happens:
  1. Read the end of that step's log.
  2. Out of memory (`CUDA out of memory`): rerun that one step without `GPU=a100`, e.g.
     `bash scripts/run_tier1.sh e5`, then resume the remaining steps by name.
  3. Anything else, including a failed self-check assertion: do NOT fix code. Record the
     step, the error and the last ~40 log lines. Then continue with the remaining
     steps that do not depend on it (every step is independent except E3's
     `e3_overlap`, which needs the four `e3_das_*` runs). Report it in `RUN_REPORT.md`.
- Step names, to run or resume individually:
  - Tier 1, order used by `all`: `e1 e5 e2 e6 e4 e3`.
  - Tier 2, order used by `all`: `e10 e7 e9 e11 e8_3b e8_internvl`.

Expected times on this card (estimates, not measured; use the logged `done in N min` to
correct them):

| Step | What it is | Est. |
|---|---|---|
| T1 e1 | are attributes present at the image tokens? (+ E0b identity check) | 7-13 min |
| T1 e5 | direct vs question-token route (attention-edge knockouts) | 5-8 min |
| T1 e2 | severed MLP freezing at base/mean/other values + 4x DLA | 10-13 min |
| T1 e6 | same pairs with the country NAMED in text instead of shown | 12-18 min |
| T1 e4 | compose router + mover heads into an answer no single run gave | 15-20 min |
| T1 e3 | readout Jacobian at MLP 23-26 + 4 trained neuron masks + mask overlap | 70-90 min |
| T2 e10 | head-set overlap statistics (CPU) | 1 min |
| T2 e7 | brands/celebrities: vocab projection + 3 severed runs | 20-25 min |
| T2 e9 | zero Park et al.'s Visual Retrieval Heads on flags | 6-9 min |
| T2 e11 | object vs background token patching | 12-18 min |
| T2 e8_3b | Qwen2.5-VL-3B: VADE accuracy sweep, pruning, head pipeline | 65-85 min |
| T2 e8_internvl | InternVL3-8B: same | 95-130 min |

`e8_*` write that model's baseline predictions and pruned tuples into
`../VADE/models/<model>/` -- that is VADE's own convention and expected.

## 4. Self-checks that must hold

Check each one in the step's log as the step finishes. Several are asserted, so a failure
already stops the step; the rest are printed and you must read them. Any check that fails
makes that step's numbers VOID: report it, do not interpret its results.

| Step | Must hold |
|---|---|
| e1 | The E0b identity line reports **0** difference: image-token activations are identical whichever question is asked. |
| e2, e7 | `self-checks:` shows `readback_worst` small, `identity: bit-exact`, `liveness_max_logit_delta` > 1e-3. |
| e4 | Read-back is exact, and installing a run's own values is a no-op (both asserted). |
| e5 | Knockout self-checks pass (asserted). If it WARNS that the injected image does not transfer, report that prominently: every image-patch arm is read against it. |
| e6 | If it warns that L_e excludes a mover block, report the warning verbatim. |
| e3 | `readout_jacobian`'s eval `full` arm is the ceiling for the masks. If `full` is near 0, the four masks cannot work: say so before reporting mask results. |
| e11 | The `seq:before` row must read **0%** at every layer (those tokens are bit-identical in base and source). Anything else voids the whole e11 run. |
| e9 | Read the INDEX CHECK: a correctly indexed `park_vrh` sits well ABOVE 50 (random heads average ~50; Park chose these as the top box-attending heads). If `park_vrh@1based` is clearly higher than `park_vrh`, the list is 1-based and only the `@1based` arms count. Report both percentiles either way. |
| e8_* | `image_head_pipeline` may stop an attribute after stage 1 with a scoring guard (source match below 0.85). That is a designed stop, not a crash: list which attributes stopped. |

## 5. Report and package

Write `RUN_REPORT.md` at the repo root containing:

1. **Environment:** GPU, torch/transformers versions, both repos' commit hashes.
2. **Per step:** status (ok / failed / OOM-rerun-at-default / skipped), wall time from
   the log, and the self-check outcome from section 4.
3. **Headline numbers**, copied from each step's summary output, not paraphrased:
   - **e1:** identity result; for each attribute, image-token probe accuracy vs its
     layer-0 value and the permutation null; name rank vs value rank at the image tokens.
   - **e5:** compare `ko:readout_all+img_patch` with `ko:question_all+img_patch`.
   - **e2:** for `heads+mlp[23-26]` and each single block, the `src 1st` / `kept` columns
     at base vs `@mean` vs `@other` (plus the `3rd` columns for `@other`); the MLP 23-26
     share in each DLA summary.
   - **e6:** the four handoff columns, and whether the `readout:23-26` freeze affects
     text the way it affects images.
   - **e4:** the transfer rate to A-of-B against each single-route control.
   - **e3:** the jacobian `full` / `readout` / `isolated` / `random` arms, each mask's
     `n_selected` and cause/iso, and `mask_overlap`'s overlap vs chance.
   - **e7:** the same severed columns as e2, using the `div` columns (these runs use
     `--gold own`); vocab projection name rank vs value rank per head set.
   - **e9:** INDEX CHECK percentiles; recall per arm; mover name rank per arm.
   - **e11:** per position set, cause at each layer.
   - **e8:** each model's chosen handoff window and top-8 heads per attribute; the
     overlap table from the final `head_overlap` step.
   - **e10:** the overlap table, from `results/head_overlap/OVERLAP.md`.
4. **Anything unexpected**, verbatim: warnings, assertion messages, tracebacks.

Then package everything new for download (the user pulls this file off the pod):

```bash
cd /workspace/VADE-evals
tar czf /workspace/tier12_outputs.tgz RUN_REPORT.md logs/tier1 logs/tier2 results methods/dla \
    logs/image_head_pipeline_* ../VADE/models/Qwen2.5-VL-3B-Instruct ../VADE/models/InternVL3-8B-hf
ls -lh /workspace/tier12_outputs.tgz
```

If any of those paths does not exist because its step failed, drop it from the command,
and say so in the report. Finally, tell the user the tarball's path and size, and give a
5-line summary: what ran, what failed, and the single most important number from each tier.
