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

## Tier 3 (FOLLOWUP_PLAN.md), third session

Same ground rules as above: no edits under `methods/`, `tests/` or `scripts/`; never overwrite
existing `results/` or `logs/` (every Tier-3 step writes under `results/tier3/` or a new
`*_n128` / `_tier3`-suffixed name).

```bash
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home
git pull && python -m pytest -q tests/              # expect 0 failures (401+ tests)
tmux new -s tier3
GPU=a100 bash scripts/run_tier3.sh smoke            # ~15 min; numbers meaningless, every step must finish
GPU=a100 bash scripts/run_tier3.sh session1         # P1-P12, ~4-6 h; P1, P2, P5 first
```

Then `session2` (needs `results/readout_jacobian/flags__mlp_hidden_b23-26/readout_fit.pt` from the
E3 pod for P13, else P13 prints SKIPPED; P18 needs the 3B pruned tuples from E8), then `session3`
if asked.

### Alternative: run Session 1 as three parallel lanes

Every step loads its own copy of Qwen2.5-VL-7B (~16.6 GB bf16) plus activations that grow with
batch size.
- **`GPU=a100` raises batch sizes to 64-128** for the scripts listed in `run_tier3.sh`'s profile.
  For Session 1 it was kept off every lane, because no a100-profile memory figure existed yet.
- **Without it,** every script runs at its own default batch, sized for a 24 GB card. Measured in
  Session 1 (`logs/tier3/GPU_USAGE_session1.md`): 16.5-18 GB per lane, 52 GB peak for three lanes.

```bash
bash scripts/run_tier3.sh smoke                  # ALONE first, no GPU=a100: downloads + checks every changed path
# then one tmux window per lane (no GPU=a100 anywhere):
bash scripts/run_tier3.sh p1 p4 p10              # lane 1
bash scripts/run_tier3.sh p2 p5 p3 p11           # lane 2
bash scripts/run_tier3.sh p6 p7 p8 p9 p12        # lane 3
```

1. Start lanes 1 and 2, watch `nvidia-smi` for ~10 minutes, and start lane 3 only if peak usage
   stays below ~55 GB.
2. Check `free -g` first. Each process passes through system RAM while loading the model, so
   three lanes want roughly 64 GB or more.
3. Optionally `export PYTORCH_ALLOC_CONF=expandable_segments:True` in each window, which reduces
   memory fragmentation. (`PYTORCH_CUDA_ALLOC_CONF` is the deprecated name; torch warns about it.)
4. If a step runs out of memory, only its lane stops. Rerun that step by name after the other
   lanes finish.

Things to know:
- **Keep P1 in one lane.** It is the only Tier-3 step that writes a shared on-disk cache
  (`../VADE/results/<model>/flags/source_activation_cache/full_image.pt`); its cached and
  uncached arms already run one after another inside P1. No other step shares an output file:
  each writes its own directory, and each lane gets its own log names.
- **Speed:** lanes share the GPU's compute. Jobs that already saturate it gain little;
  small-batch jobs gain most (P11 at batch 1, the P1/P4 evaluations). Expect Session 1 in
  roughly 3-4 h instead of 4-6 h, not a third of the time.
- **Batch size can change a few answers** (CLAIM_CANDIDATES G7: batch shape and attention
  backend move clean answers on a few rows). P2's baseline arms are checked against the
  committed E9 run, which used batch 128; at the default 16 a few of those rows may differ.
  P2's real comparison (movers vs random heads) is within the same run, so it is unaffected.

**Self-checks that must hold:**

| Step | Must hold |
|---|---|
| p1 | `L16_ones` and `L16_ones_nocache` agree with each other. Report both against the ceiling's 31.2% (flag_ring1 L16) and DBM's 60-62%. |
| p2 | The `clean`, `zero:park_vrh` and `zero:common10` rows reproduce `results/vrh_ablation/flags/summary.json`: exactly with `GPU=a100` (batch 128, as committed); within a few rows at the default batch (parallel lanes). |
| p3 | Print `heldout_consensus.txt` and its overlap with common10. |
| p6 | In `patch15_blocks15-27_n84.json`, phase 3's ALL arm equals the question patch's `first`. The identity is closed at full coverage. |
| p9 | The `entity_patch` and `image_movers` arms reproduce E6 (87-91% transfer). |
| p10 | `self-checks:` shows `identity: bit-exact`. |
| p11 | `all_downstream_continuous` equals `image_patch` (asserted). |
| p15, p27 | The empty-knockout no-op check prints `max |dlogit| = 0`. p15 also checks install-of-own-values = clean. |
| p26 | `--verify_cached` raises on the first mismatch. If it raises, record the message, then continue with the other steps. |

### What Session 1's GPU log showed (2026-10-18)

Source: `logs/tier3/GPU_USAGE_session1.md`.
- **The lanes are CPU-dispatch bound, not memory or compute bound.**
  - Memory: about 17 GB per lane; three lanes peaked at 52 GB.
  - "Util" read 97%, but power averaged 213 of 400 W.
  - Each lane's Python process held ONE core at 100% while 93% of the 96 cores idled: token-by-token
    generation at small batches, with a hook on every layer.
- **Three lanes gave about 1.5x, not 3x.** One P1 eval took 7 minutes alone and 14 with three lanes:
  without MPS the processes take turns on the GPU instead of overlapping.

Two consequences:
- **Bigger batches are the cheapest speed-up.** A dispatch-bound batch of 64 costs about the same wall
  time as a batch of 16.
- **One lane alone is not enough.** About half of Session 2's work is in scripts `GPU=a100` does not
  raise (head_trace, attr_head_trace, attr_steer, the serial switch checks). A single lane would run
  those one after another at CPU-bound speed.

`run_tier3.sh`'s a100 profile therefore also raises `readout_jacobian` (P13) and `head_swap_vade`
(P14, P18, P28) to batch 64 (`BS_JAC`, `BS_SWAP`). Batch size can move a few clean answers
(CLAIM_CANDIDATES G7). Session 2's comparisons are within each run or about curve shape, so that is
acceptable there.

### Session 2 (trimmed, 2026-10-18): eight steps, three lanes

**Deferred, do not run:**
- P13 (readout rank sweep)
- P15 (brands late attention)
- P16 (n=128 stage-1 curves)
- P17 (animals text positions)
- P18 (3B windows and install)
- P21 (currency with 10 random draws)
- P24 (steering robustness)

`bash scripts/run_tier3.sh session2` now runs only the eight below; the deferred ones are in
`session2_deferred`, for later.

| step | claim it tests | lane |
|---|---|---|
| p14 | the movers carry entity identity: animals' entity facts transfer, category traits do not (A10) | 1 |
| p20 | the answer format comes from the base question, not more routers (B9) | 1 |
| p27 | the movers read identity from object vs background keys, causally (A9) | 1 |
| p23 | identity spreads outward ring by ring; face identity is object-local (D2, D3) | 1 |
| p19 | one shared router serves all four attributes (B8) | 2 |
| p25 | the question is relayed via "Answer"/":"; block 19 is no keystone (B6, B7) | 2 |
| p26 | the cached attribute-switch engine reproduces serial decoding (G7) | 3 |
| p22 | whether text has an early subject-enrichment stage that images skip (D9) | 3 |

**0. If you started Session 2's old lane 1** (`p16 p17 p23`, from the earlier instructions), stop
it with Ctrl-C in its window. P16 and P17 are deferred. Partial files under
`results/image_head_pipeline_n128/` are harmless: leave them and don't package them.

**1. Wait for Session 1 to finish,** then refresh the checkout. Run nothing while any lane is still
going; see "Moving from a running Session 1" below if you want to overlap.

```bash
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home PYTORCH_ALLOC_CONF=expandable_segments:True
git add results/tier3 logs/tier3 && git commit -m "Tier-3 Session 1 results"   # anything not yet committed
rm -f scripts/run_tier3_next.sh && git pull
python -m pytest -q tests/                               # expect 0 failures
```

**2. Optional: start MPS before any lane** (see the note below; skip it if the daemon refuses):

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps
mkdir -p /tmp/mps && nvidia-cuda-mps-control -d
```

Export the same two variables in every lane window if MPS is on.

**3. Start the lanes, one at a time, each in its own tmux window:**

```bash
GPU=a100 bash scripts/run_tier3.sh p14 p20 p27 p23      # lane 1: steps whose batch GPU=a100 raises
bash scripts/run_tier3.sh p19 p25                       # lane 2: default batches
bash scripts/run_tier3.sh p26 p22                       # lane 3: default batches
```

- Start lane 1 first and read its `nvidia-smi` peak after ~10 minutes. No a100-profile lane has
  been measured yet, and default lanes were ~17 GB each in Session 1.
- Then start lanes 2 and 3, keeping the total under ~70 GB. If lane 1 alone goes past ~40 GB,
  restart it without `GPU=a100`.
- Expected wall time is roughly 2-3 h, with lane 1 the longest because of P14's ~8.2k head-install
  generations. This is an estimate from Session 1's rates, not a measurement.

**4. Self-checks to read as each step finishes:**

| step | must hold |
|---|---|
| p14 | The `clean` arm's accuracy on animals is high (pruned tuples). The `full_image` arm (patch at layer 10) transfers far more than `random_heads` (~0). |
| p20 | Read-back is exact, and installing a run's own values is a no-op (both asserted, as in E4). `movers` and `clean` reproduce `results/circuit_compose/flags/summary.json`. The router-dependent arms are new and are not compared. |
| p27 | The empty-knockout no-op check prints `max |dlogit| = 0`. `img_patch` transfers ~98% as in E5. |
| p23 | Nothing is asserted. `ring:0` is the same token set as `entity_only`, so at L10/L18 it should sit near E11's capital `entity_only` (34% / 16%, n=96; this run uses 32 rows). Both face halves run. |
| p19 | Each donor attribute's phase-1 top-8 is printed. Report how many heads each shares with router8. |
| p25 | No assertion fires for `field` (it requires exactly one differing token per prompt pair). Each gapped `attention_set` arm (16-23 minus block 19, 20 or 21) reads at or below the full `attention_blocks:8` arm (blocks 16-23) in the same run. |
| p26 | `--verify_cached` raises on the first mismatch. If it does, record the message and continue with the paired runs. |
| p22 | The freeze specs print, and no warning says L_e excludes a mover block. |

**5. After the lanes:**
1. `echo quit | nvidia-cuda-mps-control`, if MPS was started.
2. Commit `results/tier3 logs/tier3` and the `_tier3`-suffixed ceiling logs under
   `logs/Qwen2.5-VL-7B-Instruct/` (from P23).
3. Write `RUN_REPORT_TIER3.md` as in section 5.

**About MPS.** It lets kernels from the three processes overlap instead of time-slicing, which is
what held Session 1's three lanes to 1.5x. MPS changes no numbers. It is untested on this pod and
may be blocked inside the container. Start it only while **no** CUDA process is running.

### Moving from a running Session 1 into Session 2

**Never `git pull` or edit `scripts/run_tier3.sh` while a lane is running.** bash reads its script
incrementally, so a running lane can execute a mix of the old and new file. To start a Session 2
lane while Session 1 lanes are still going, run it from an untracked copy of the new launcher. The
copy must sit in `scripts/`, because the launcher `cd`s relative to its own path:

```bash
cd /workspace/VADE-evals && git fetch origin
git show origin/main:scripts/run_tier3.sh > scripts/run_tier3_next.sh     # untracked copy
GPU=a100 bash scripts/run_tier3_next.sh p14 p20 p27 p23                  # e.g. lane 1, in a new window
```

Only the launcher has changed since Session 1 started; `methods/` is identical, so the copy runs the
same code. Once every Session 1 lane has finished:
1. commit their results;
2. `rm scripts/run_tier3_next.sh`;
3. `git pull`;
4. start the remaining Session 2 lanes with `scripts/run_tier3.sh`.

Report as in section 5, writing `RUN_REPORT_TIER3.md`, and package `logs/tier3 results/tier3` plus
the `_tier3`-suffixed ceiling logs under `logs/Qwen2.5-VL-7B-Instruct/`. (`results/image_head_pipeline_n128`
belongs to the deferred P16; don't package a partial one.)
