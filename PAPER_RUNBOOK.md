# Paper runbook: experiments for the final paper

Instructions for the session running on the GPU pod. This file is a queue: experiments are
added over time (section 3) and run together in one batch. `PAPER_TODO.md` says *why* each
one is needed; this file only says what to run, what must hold, and what to report.
`CLAUDE.md` explains every script.

## Ground rules

- **Do not edit anything under `methods/`, `tests/` or `scripts/`, and do not change any
  hyperparameter, head list, layer or row count.** A failed self-check is evidence; patching
  the code to make it pass destroys it. Stop that step, record what happened, and report.
- **The one allowed change:** if a step runs out of GPU memory, rerun just that step by name
  without `GPU=a100` (section 2), and record that you did.
- Never delete or overwrite anything under `results/` or `logs/` that existed before you
  started. Every step below writes to its own directory.
- Do not `git commit` or `git push` unless the user says to.
- If something is ambiguous, stop and ask the user. Do not guess.

## 1. Setup and checks

If the pod is new, do `GPU_RUNBOOK.md` section 1 (clone both repos side by side under
`/workspace`, `HF_HOME`, pip install, `pyvene --no-deps`). On an existing pod:

```bash
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home
git pull
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
python -m pytest -q tests/                # expect 0 failures; tokenizer tests may SKIP
```

`tests/test_crossmodal.py` has never been run (it was written off-pod without torch). If
any test in it fails, stop and report the failure verbatim; do not run X1.


## 2. Run order: smoke, then three lanes on one A100

Everything runs through `scripts/run_paper.sh`, which logs each step to
`logs/paper/<step>_<stamp>.log` and writes results under `results/paper/`. Section 3's
per-experiment commands are what each named step runs; you do not need to type them.

| ID | Experiment | PAPER_TODO item | Steps |
|---|---|---|---|
| X1 | Mover-head content, image vs text | §2 image-vs-text convergence | `x1` |
| X2 | Brands + celebrities: mover install, cross-question, MLP lookup | §0 full story on brands/celebrities | `x2_swap_*`, `x2_cross_*`, `x2_severed_*`, `x2_score` |

Not runnable yet (section 5 lists the code each one needs; do NOT attempt them on the pod):
X2b (brands/celebrities routers, compose, text contrast) and X3 (Idefics3, a non-Qwen LM).

### Why three lanes, and how they are balanced

The last two pod sessions (`logs/tier3/GPU_USAGE_session1.md`, `RUN_REPORT_TIER3.md` §5)
measured how these jobs use the card:
- They are **CPU-dispatch bound.** One Python core runs at 100% per job; mean power is about
  205 of 400 W.
- That makes **bigger batches the cheapest speed-up**, so every lane runs `GPU=a100` (batch 64).
- **Parallel lanes help, but not linearly.** Three lanes gave about 1.5x throughput, not 3x.
- **Memory limits the lane count.** a100-profile jobs peaked at 22-32 GB each, and three lanes
  reached 78.8 GB once. So three lanes is the ceiling; a fourth would run out of memory.

The lanes are balanced by generation count (dry runs, 2026-10-19). Celebrities' install is
2,308 generations per arm against brands' 1,394, so it is split across two lanes by head set.

| lane | steps | ~generations |
|---|---|---|
| 1 | `x2_swap_celeb_a`: celebrities clean, full_image, nat_top8, lang_top8 (+ random nulls) | ~14k |
| 2 | `x2_swap_brands` (8 arms), `x2_cross_brands` (×2 alignments), `x2_severed_brands` | ~16k |
| 3 | `x2_swap_celeb_b` (cons6, common10 + nulls), `x2_cross_celeb` (×2), `x2_severed_celeb`, `x1` | ~16k |

Expected wall time is about 2.5-4 h per lane, extrapolated from Tier-3 rates (P3: 14k
generations in 217 min at batch 16 alongside two other lanes). This is not measured. X1 sits
last in lane 3 because it is the one step with no earlier timing; if it runs long, nothing
waits on it.

### Steps

**1. Smoke (~15-25 min, one window).** It also runs the head-list check:

```bash
tmux new -s paper
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home PYTORCH_ALLOC_CONF=expandable_segments:True
bash scripts/run_paper.sh smoke
```

Smoke numbers mean nothing. Check only that every step prints `-> [smoke_...] done`, with no
traceback, and that the first line reads `head check ok: ...`.

**2. Optional: MPS.** It lets the three lanes' kernels overlap instead of time-slicing. It is
untested on this pod. Start it only while no CUDA process is running, and skip it if the
daemon refuses:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps
mkdir -p /tmp/mps && nvidia-cuda-mps-control -d
```

If MPS is on, export the same two variables in every lane window.

**3. Start the lanes one at a time,** each in its own tmux window (`Ctrl-b c`), with the same
`cd` / `export` line as step 1:

```bash
GPU=a100 bash scripts/run_paper.sh lane1
GPU=a100 bash scripts/run_paper.sh lane2
GPU=a100 bash scripts/run_paper.sh lane3
```

- Start lane 1, wait ~10 minutes, and read the `nvidia-smi` peak. Then start lane 2, wait
  again, then lane 3.
- Keep the total under ~72 GB. If adding a lane would pass that, wait for one to finish.
- In a spare window, log `nvidia-smi` every 30 s for the report:

  ```bash
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
      --format=csv -l 30 > logs/paper/gpu_usage.csv
  ```

- **If a step runs out of memory:**
  1. Only its lane stops (`set -e`).
  2. Let the other lanes finish.
  3. Rerun that step by name, without `GPU=a100` (e.g. `bash scripts/run_paper.sh x2_cross_celeb`).
  4. Record that you did.
- **No step resumes.** A rerun rewrites that step's own outputs from scratch. That is safe:
  every step writes only its own directory.

**4. After all three lanes finish:**

```bash
bash scripts/run_paper.sh x2_score               # CPU only, a few minutes
echo quit | nvidia-cuda-mps-control              # only if MPS was started
```

Then check every self-check in section 3 and write the report (section 4).

## 3. Experiments

### X1. Mover-head content: image vs text-named entity

**Question.** E6 showed the same heads move the entity in both conditions. Is what they
carry the same? Two parts in one run: representational (does image-run France sit nearest
text-run France, per layer and per mover head?) and causal (do text-run mover values install
into an image run, and the reverse?).

**Rerun after the fix (2026-10-19).** The first pod run crashed at the layer-0 ridge solve.
Layer 0's last-column residual is the same token embedding in every row, so the centered matrix
was all zeros and the solve was singular. The regularizer is now floored, so that layer reads
chance (self-check 5), and a test covers the case. After `git pull` and
`python -m pytest -q tests/test_crossmodal.py`, run X1 on its own:
`GPU=a100 bash scripts/run_paper.sh x1`. It overwrites only its own output directory.

**Runs as:** step `x1` (lane 3), with smoke in `smoke`. It executes
`text_entity_contrast.py --phases crossmodal --batch_size 64 --out_dir
results/text_entity_contrast/flags_crossmodal`.

Defaults: flags, 64 pairs × 4 attributes, `common10` movers, batch 16, seed 0. The
representation part always uses all 84 countries × 4 attributes.

**Self-checks** (on the full run; report each as pass/fail with the number):

1. In both install tables, `clean` transfer = 0% and `source_clean` = 100% (eligibility
   filter working). `n_eligible` should be roughly 220-240 per condition (E6 had 230/232).
2. `install:random10<-same` and `install:random10<-cross`: transfer ≤ 5%.
3. `install:movers<-cross:self` (the base country's own values from the other condition):
   base kept ≥ 85%. If it is far lower, foreign-modality vectors break the run on their own,
   and every cross arm is uninterpretable. Report it prominently.
4. `install:movers<-same` is in the same range as E6's `install:image_movers` (87-91%). E6
   installed entity-patched values, this run installs clean source values, so a few points'
   difference is expected; more than ~15 points is not.
5. In the representation table, `resid@0` top-1 ≈ chance (≈1.2% for 84 countries). The
   last column is the same token in every prompt there, so anything above chance is a bug.

**Outputs:** `summary.json` (`crossmodal.representation`, `crossmodal.install`),
`crossmodal_rows.jsonl` (per-row install records), `crossmodal_reps.pt` (all captured
vectors, for CPU re-analysis; gitignored, so copy it off the pod).

**Report back:**
- Both install tables as printed.
- The representation table's resid curve (layers 0-27), the entity curve, and the mover
  rows.
- The first layer where `resid` ridge top-1 exceeds 50%: "where the image run becomes the
  text run".

**How to read it:**

| result | reading |
|---|---|
| cross ≈ same | the moved payload is modality-invariant |
| cross low, normmatch recovers it | same direction, different scale |
| cross low, self also breaks | the two conditions' mover spaces are incompatible |

### X2. Brands and celebrities: the parts of the story that run today

**Question.** Does the flags story hold on two more entities? This covers the three pieces
whose scripts already support them: the continuous mover install (VADE-scored),
cross-question patching (do the movers carry the entity rather than the asked attribute?),
and MLP freezing on the attributes not yet covered. Routers, composition and the text
contrast need code first (X2b, section 5).

Already done, so do not rerun: `head_severed` on brands/hq_country and celebrities/
nationality + native_language (`results/head_severed/*__gold-own`), and
`head_vocab_projection` on both entities.

**Runs as** (section 2 assigns them to lanes):

| step | runs | output under `results/paper/` |
|---|---|---|
| `x2_swap_brands` | `head_swap_vade` on brands: hq_top8, cons6, common10 (heads + random nulls), clean, full_image | `x2_swap_brands/` |
| `x2_swap_celeb_a` | the same on celebrities: nat_top8, lang_top8, clean, full_image | `x2_swap_celebrities_a/` |
| `x2_swap_celeb_b` | celebrities: cons6, common10 (heads + random nulls) | `x2_swap_celebrities_b/` |
| `x2_cross_brands` / `x2_cross_celeb` | `head_cross`, 64 pairs, seed 0, `--align matched` and `step0` | `x2_cross_*/cross_n64_<align>.jsonl` |
| `x2_severed_brands` | `head_severed --gold own`: founded_year, industry | `x2_severed_brands_year_industry/` |
| `x2_severed_celeb` | `head_severed --gold own`: occupation, birth_year | `x2_severed_celebrities_occ_year/` |
| `x2_score` | VADE `score.py` for every install arm × attribute, on the pruned tuples (CPU) | `x2_swap_*/scores/<arm>/<attribute>/` |

Settings shared by these steps:
- `--limit_pairs 128` for the installs.
- `--patch_layer 19` for `full_image`: both entities' image plateau ends there, and the
  default 21 under-reads them.
- Severed flags as in the committed E7 runs: n=128, freeze values base / mean / other, MLP span
  22-27, singles 23-26, attention 24-27.
- Brands cross-question stays at seed 0 with n ≤ 64. Larger samples draw AIRBUS, which has no
  `industry` label and crashes `head_cross` (fix G, section 5).
- **Random nulls differ from a single run.** Each install step draws its random-head nulls from
  its own seed-0 RNG, so the celebrities nulls are not the ones a single combined run would draw.

Head sets come from `results/image_head_pipeline/<entity>/<attr>/head_trace_*.json`
(`phase1_ranked[:8]`). The launcher refuses to run if they drift from:

| set | heads |
|---|---|
| brands hq_country (BH) | `23.3,21.4,22.17,23.4,23.6,22.15,23.17,21.1` |
| celebrities nationality (CN) | `23.3,22.17,23.4,23.6,21.4,22.15,21.5,21.1` |
| celebrities native_language (CL) | `22.17,23.4,21.1,23.6,23.3,22.19,22.15,19.6` |
| cons6 | `21.1,22.15,22.17,23.3,23.4,23.6` (shared by all three) |

**Self-checks:**

0. Every step prints `head check ok: ...` (the launcher exits otherwise).
1. Install: `random_heads_*` cause ≤ 5% on every attribute; `clean` cause ≈ 0.
2. Install: `full_image` cause ≥ 70% on hq_country, nationality and native_language (the
   pipeline's own image-swap ceiling there was 90-100%).
3. Cross: each run prints its cells and a self-patch control. Self-patch must keep ≥ 85% of
   the base answer (flags read 91.8-93.4%).
4. Severed: every run prints `self-checks: {... 'identity': 'bit-exact' ...}`. Anything
   else: stop that run and report.
5. If any step raises `KeyError` on an item or attribute, report it verbatim; do not
   change the seed or sample size to dodge it.

**Outputs:** `results/paper/x2_*` (table above).

**Report back:**
- Per entity × attribute × arm: cause / iso / final from the score summaries.
- Each cross run's cell table, matched and step0 side by side, next to flags' R11 (81.2% same
  question, 65.1% donor asked another question).
- The severed tables for the four new attributes.
- One line on whether the category attributes (industry, occupation) behave like entity
  facts or like the animals category traits (P14).

<!-- Template for the next experiment. Copy it, give it the next X-number, add its steps to
scripts/run_paper.sh (a function per step, plus the case line), add it to section 2's table,
and put each step in the lane with the fewest generations (dry-run it for the count).

### XN. Title

**Question.** One or two sentences, and the PAPER_TODO item it closes.

**Smoke:**
```bash
...  2>&1 | tee logs/paper/xN_smoke.log
```

**Full:**
```bash
...  2>&1 | tee logs/paper/xN_<name>.log
```

**Self-checks:** numbered, each with a threshold.

**Outputs:** paths.

**Report back:** what to paste into the report.
-->

## 4. Report

When the batch is done, write `RUN_REPORT_PAPER.md`, with:
- the environment (GPU, torch and transformers versions, commit, MPS on/off);
- per-step status and wall time (each step's `done in N min` line), and which lane ran it;
- GPU usage from `logs/paper/gpu_usage.csv`: per-lane memory peaks, mean power, and whether
  three lanes beat Tier-3's ~1.5x;
- every self-check as pass/fail with its number;
- each experiment's "Report back" items;
- anything unexpected, quoted verbatim from the log.

Then package `results/paper/`, `results/text_entity_contrast/flags_crossmodal/` and `logs/paper/` for the user, including the gitignored `.pt` files.

## 5. Not runnable yet: needs code written off-pod first

These are on the lineup but must NOT be attempted on the pod: the scripts do not support
them yet, and the ground rules forbid editing code there. Each moves into section 3 with an
X-number once its code lands and its offline tests pass.

### X2b. Brands and celebrities: routers, composition, text contrast

The router pipeline (`attribute_switch_sweep` → `attr_head_trace`), `circuit_compose` and
`text_entity_contrast` only know flags. Code needed (file:line from the 2026-10-19 audit):

- **A. Per-entity controlled prompts** (`attribute_switch_sweep.py:15-33`, imported by
  `attr_head_trace`, `attr_capture`, `circuit_common`, `text_entity_contrast`):
  - an attribute list, field words and an instruction per entity ("For the brand shown, …",
    "For the person shown, …");
  - field words that tokenize to equal length (`alignment()` enforces it);
  - the `Answer:` prefill.
- **B. Own-answer golds** in place of the stored lowercase labels:
  - files: `attribute_switch_sweep.py:343`, `head_followup_common.py:96`,
    `attr_head_trace.py:118-119/149-160/402-406`, `circuit_common.py:101-137`;
  - reuse `route_knockout.own_answers` / `own_rows`.
- **C. Entity loading and `--entity`:**
  - replace `circuit_common.load_flags`;
  - widen the argparse choices in `attr_head_trace`, `attr_capture`, `circuit_compose`,
    `text_entity_contrast`;
  - give each script per-entity output dirs.
- **D.** `attr_capture.py:234` must not call `question()` on non-flags attributes.
- **E.** Entity-aware `text_question` / `text_prompt` (`circuit_common.py:52-57, 226-232`).
- **F.** Pass per-entity `--movers` / `--router_trace` rather than the flags defaults.
- **G.** `head_cross.py:136` crashes on items missing an attribute (AIRBUS has no
  `industry`); skip those rows. This also lifts X2's brands cap of seed 0 / n ≤ 64.

Run order once the code exists: residual attribute-switch sweep → `attr_head_trace` (window
from the sweep) → `circuit_compose` → `text_entity_contrast` (+ `--phases crossmodal`, as X1).

### X3. A non-Qwen language model: Idefics3-8B-Llama3

**Why this model.** Its language model is Llama-3.1-8B, and every architecture assumption
here holds:
- RMSNorm with `variance_epsilon`;
- a bias-free `lm_head`;
- o_proj input = 32 heads × 128 = hidden;
- SwiGLU `down_proj`;
- 1-D RoPE and causal attention;
- with `do_image_splitting=False`, one contiguous run of 169 image tokens (a 13 × 13 grid).

**Ruled out:**
- **Llama-3.2-Vision:** images enter through cross-attention layers, not tokens in the
  sequence, so there are no image positions to patch.
- **Gemma-3:** scientifically stronger, but 2-3× the work, so it comes second:
  - head_dim ≠ hidden / n_heads in ~14 places;
  - post-sublayer norms break DLA, `delta_resid` and vocab projection;
  - bidirectional image attention needs `token_type_ids` injected.

Code needed:
1. **Generic non-Qwen fixes** (~100 lines; they also unblock InternVL's router and MLP stages):
   - Route these prompt builders through `adapter.build_inputs`:
     `head_swap_vade.build_prompt/build_batch` (395-446), `attr_head_trace.py:117,141`.
   - `head_vocab_projection.py:507-528`: replace the hard-coded `<|im_start|>`.
   - Make `--execution cached` raise for non-Qwen models.
2. **`methods/adapters/idefics3.py`** (~170 lines, cloned from `internvl.py`), plus a
   registry entry. Details:
   - module paths `model.model.text_model.{layers,norm}`;
   - `do_image_splitting=False`;
   - extra = `pixel_values` + `pixel_attention_mask`;
   - a `vision_patch_grid` for 13 × 13;
   - `shares_object_grid = False`;
   - a BOS check.
3. **VADE side:**
   - a `--processor_kwargs` option on `run_accuracy_sweep.py`, so pruning sees the same
     single-tile input;
   - optionally, an `object_location.json` `models.Idefics3-8B-Llama3` override builder (only
     object-set experiments need it; the mover pipeline uses `full_image` / `last_token`).
4. **Launcher.** Same path as E8, but with the router and MLP stages added. Pass explicit
   `--heads` / `--blocks` / `--freeze_mlp_*` from the new traces, never the Qwen defaults:
   - accuracy sweep
   - `prune_tuples`
   - `image_head_pipeline --n_layers 32 --n_heads 32` on flags + brands
   - head_overlap vs Qwen-7B
   - routers
   - `head_severed`

Check flags/brands accuracy with the cheap accuracy sweep before investing in the rest. If
recognition is poor, there are too few eligible rows to trace.
