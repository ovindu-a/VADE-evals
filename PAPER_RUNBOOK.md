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
- **Never edit, commit or pull-modify code in `../VADE`.** It is a separate benchmark repo and is
  used as-is. The only writes there are new output files under `../VADE/models/<model slug>/`
  (accuracy predictions, pruned tuples), as for every earlier model.
- Do not `git commit` or `git push` unless the user says to.
- If something is ambiguous, stop and ask the user. Do not guess.

## 1. Setup and checks

If the pod is new, do `GPU_RUNBOOK.md` section 1 (clone both repos side by side under
`/workspace`, `HF_HOME`, pip install, `pyvene --no-deps`). On an existing pod:

```bash
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home PYTORCH_ALLOC_CONF=expandable_segments:True
git pull
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceM4/Idefics3-8B-Llama3')"   # ~17 GB, X3
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
python -m pytest -q -rs tests/            # expect 0 failures and 0 skips (Idefics3 tokenizer now cached)
```

Off-pod (2026-10-19, CPU, torch 2.14.1, transformers 5.16.1): 465 passed, 8 skipped (the 8 Idefics3
real-tokenizer tests, which need the download above). If any test fails on the pod, stop and report it
verbatim. A failure in `test_idefics3_adapter.py` blocks X3, one in `test_x2b_entities.py` blocks
X2b, and one in `test_head_cross.py` blocks the X2 rerun. The other lanes may still run.


## 2. Batch 2 (next): X2b, X3 and the X2 cross-question rerun

| ID | Experiment | PAPER_TODO item | Steps | Lane |
|---|---|---|---|---|
| X3 | Idefics3-8B-Llama3: movers, routers, MLP lookup on a Llama LM | §0 non-Qwen model | `x3` (= `scripts/run_x3_idefics3.sh all`) | 1 |
| X2b | Brands, celebrities: routers, composition, text contrast + crossmodal | §0 full story | `x2b_brands`, `x2b_celeb` | 2, 3 |
| X2r | Cross-question rerun: common10, own-answer golds, entity facts only | §0 cross-question | `x2r_cross_brands`, `x2r_cross_celeb` | 2, 3 |

**Lanes.** Each lane is one dependent chain, so nothing inside a lane can run in parallel:
- X3: gate → prep → pipeline → overlap → depth → routers → severed.
- X2b, per entity: sweep → window → trace → compose → text contrast → crossmodal.

The two entities and the model are independent of each other.

| lane | runs | the chain stops early if |
|---|---|---|
| `b2_lane1` | X3 | `GATE FAILED` (accuracy < 50%); the lane exits, report the table |
| `b2_lane2` | X2b brands, then X2r brands | a step raises (set -e). A missing trace window is NOT a stop: trace and compose are skipped, the rest runs |
| `b2_lane3` | X2b celebrities, then X2r celebrities | same as lane 2 |

**Memory: this batch is heavier than batch 1, so stagger the lanes and watch.** Known peaks:
- `circuit_compose` at batch 64: 31.6 GB (Tier-3 P20);
- the cached attribute-switch engine at batch 64: 24.5 GB (P26);
- `head_swap_vade`: ~22 GB (batch 1);
- Idefics3-8B has never been run here. Its weights are ~17 GB, so expect lane 1 at 25-35 GB.

Three lanes could therefore reach ~90 GB if their peaks coincide.
- Start lane 1, wait ~15 minutes (past the gate), and read `nvidia-smi`.
- Start lane 2 and wait ~15 minutes.
- Start lane 3 only if the total is under ~45 GB. Otherwise start it when lane 1 or 2 finishes.
- If a step runs out of memory: let the others finish, then rerun that step by name without
  `GPU=a100`, and record it. For X3 the steps are `bash scripts/run_x3_idefics3.sh <step>`.

**Time is only roughly estimated.** Batch 1 ran in 70 min against an estimate of 2.5-4 h per lane, so
treat any estimate here as an upper bound. Known sizes:
- X2b's attribute-switch sweep is its largest stage: 71k generation runs for brands and 95k for
  celebrities (dry run). Tier-3 P25 did ~69k cached runs in ~45 min, so expect ~45-60 min per entity.
- The X2 cross rerun is ~1.1-2k generations per run, three runs per entity.
- X3 has no timing at all: it downloads a new model and runs the full pipeline on it.

### Steps

**1. Smoke (one window).** It covers every batch-2 step, including X3 on the real model:

```bash
tmux new -s paper2
cd /workspace/VADE-evals && export HF_HOME=/workspace/hf_home PYTORCH_ALLOC_CONF=expandable_segments:True
bash scripts/run_paper.sh smoke2
```

Check that:
- every step prints `-> [smoke_...] done` (X3's print `-> [x3_smoke_...] done`), with no traceback;
- the first line reads `head check ok: ...`;
- `x3_smoke_tests` reports 26 passed, 0 skipped.

If one entity's or X3's smoke fails, hold that lane back and run the others.

**2. Optional: MPS**, exactly as in batch 1 (section 2b, step 2).

**3. Lanes**, each in its own tmux window with the same `cd` / `export` line, staggered as above:

```bash
GPU=a100 bash scripts/run_paper.sh b2_lane1
GPU=a100 bash scripts/run_paper.sh b2_lane2
GPU=a100 bash scripts/run_paper.sh b2_lane3
```

Log the GPU in a spare window, as in batch 1:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv -l 30 > logs/paper/gpu_usage_batch2.csv
```

**4. After the lanes:**
1. `echo quit | nvidia-cuda-mps-control`, if MPS was started.
2. Check every self-check in X2r, X2b and X3 (section 3).
3. Write the report (section 4) as `RUN_REPORT_PAPER_BATCH2.md`.

## 2b. Batch 1 (X1, X2): done 2026-10-19, RUN_REPORT_PAPER.md

Kept for reference. The lanes are now named `b1_lane1..3` (`lane1..3` still work).

### Run order: smoke, then three lanes on one A100

Everything runs through `scripts/run_paper.sh`, which logs each step to
`logs/paper/<step>_<stamp>.log` and writes results under `results/paper/`. Section 3's
per-experiment commands are what each named step runs; you do not need to type them.

| ID | Experiment | PAPER_TODO item | Steps |
|---|---|---|---|
| X1 | Mover-head content, image vs text | §2 image-vs-text convergence | `x1` |
| X2 | Brands + celebrities: mover install, cross-question, MLP lookup | §0 full story on brands/celebrities | `x2_swap_*`, `x2_cross_*`, `x2_severed_*`, `x2_score` |

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

### X2r. Cross-question rerun on brands and celebrities (own-answer golds, common10)

**Question.** Is the moved payload the ENTITY rather than the asked attribute on brands and celebrities,
as on flags? Base A is asked Q1 and donor B is asked Q2; is the answer B's answer to Q1? Batch 1's X2
run could not say, for three reasons:
- it scored against stored lowercase labels, so the self cell read 81%, not 100%;
- it used each entity's top-8, which installs far worse than common10 (62% vs 96% on brands);
- it averaged over attributes the movers do not carry (years, industry, occupation).

A per-attribute rescore of those rows, by hand, gave same question 39-61% and another question
6-38%, against flags' 81% / 65%. That is too confounded to quote.

**What is new in the code:** `head_cross --gold own` and `--q1_attributes`.
- `--gold own` first generates the model's own unhooked answer for every (item, question) a row
  needs. It keeps an answer only if VADE's matcher accepts it against the stored label, then
  classifies by normalized exact match.
  - Rows whose four golds are not all valid are `excluded`.
  - CONTENT SHIFT is reported per base question, on INFORMATIVE rows only: base and donor answer
    Q1 differently.
  - Each row also stores `label_stored`, the old scoring, for comparison.
- `--q1_attributes` restricts the base question to the entity facts. The donor still asks every
  question, so the `both` cell survives.

**Runs as** (`results/paper/x2r_cross_*/`):

| step | runs |
|---|---|
| `x2r_cross_brands` | common10, base question hq_country, n=256 pairs, `--align matched` and `step0`; plus brands' top-8 (matched) for comparison with batch 1 |
| `x2r_cross_celeb` | the same with nationality + native_language, n=128 pairs |

**Self-checks:**
1. `self` cell: base_q1 ≥ 99%. With own golds the self-patch is an exact identity (batch 1 showed
   640/640 rows reproducing the clean answer).
2. The `own answers matching the stored label` line prints for every attribute. Report it
   verbatim; it sets how many rows survive.
3. Informative rows in the per-question table: n ≥ 60 in the `flag` column for each base question.
   Fewer: report it and say the CI is wide.
4. `question` cell: moved to A's answer to Q2 at ≤ 5% (flags 0.0%).

**Report back:**
- each run's 4-way cell table;
- the per-base-question table (same / another / shift), matched and step0 side by side, next to
  flags R11 (81.2 / 65.1, shift −16.1; step0 removes most of it);
- common10 vs top-8 at matched;
- the `excluded` count.

**How to read it:**

| result | reading |
|---|---|
| same ≈ another (shift ≈ 0 under step0) | entity, not attribute, as on flags |
| large shift that survives step0 | question-conditioned content on these entities |
| low "same" even with common10 | the movers carry less of these entities at the answer column |

### X2b. Brands and celebrities: routers, composition, text contrast

**Question.** The rest of the flags story, on two more entities:
- **Routers:** do router heads in blocks ~19-21 carry *which attribute was asked*? (Attribute-switch
  sweep → `attr_head_trace`.)
- **Composition:** do movers from one run and routers from another compose into an answer neither
  run produced? (`circuit_compose`.)
- **Text contrast:** does the entity arrive later when shown than when named, and do the same mover
  heads carry it in both conditions? (`text_entity_contrast`, plus `crossmodal` as in X1.)

**What is new in the code (read before judging any number):**
- **Controlled prompts.** Each entity has its own controlled prompt (`attribute_switch_sweep.ENTITY_SPECS`):
  - one instruction defining every field word;
  - `Report the <field> of this brand|person.`;
  - the `Answer:` prefill.

  Field words are one token each (brands: country / year / industry; celebrities: country /
  profession / year / language), so prompts differ in exactly one column.
- **Own-answer golds.** Labels are stored lowercase and the model adds articles, so golds are the
  model's own clean answers under these prompts, kept only where VADE's matcher accepts them
  against the stored label. Consequences:
  - the sweep runs only the baselines on rows the model gets wrong;
  - `attr_head_trace` drops those rows before phase 0, so phase 0 reads ~100%;
  - the text contrast keeps a pair only if the image-run and text-run answers agree up to the
    divergence token.
- **Year answers** tokenize as `['Ġ', '1', '9', '6', '1']`, like flags' calling codes. A year donor's
  `donor_first` therefore only says "a number was started":
  - in the sweep, read `donor_full` for year donors;
  - in `attr_head_trace`, whose scorer reads at most 3 answer tokens, even `full` covers only the
    century.
- **The trace window is not chosen by hand.** `attribute_switch_report.py --window` applies a fixed rule
  to the sweep (it reproduces flags' patch_layer 15, blocks 15-22). The launcher evals the
  `PATCH=... BLOCKS="..."` line it prints. If there is no window, trace and compose are skipped and
  the lane continues.

**Runs as** `x2b_brands` / `x2b_celeb` (`scripts/run_paper.sh`'s `x2b_entity`), each with the entity's
own traced mover top-8 (BH / CN). Output goes to `results/paper/x2b_*_<entity>*`:

| stage | brands | celebrities |
|---|---|---|
| attribute-switch sweep (`--sites residual attention --scopes earlier_text last_token --block_spans --attention_spans 3 --execution cached`) | 24 items | 16 items |
| report + window (`--window`) | | |
| `attr_head_trace` at that window | 128 items | 84 items |
| `circuit_compose` (own movers + own routers) | 64 items | 64 items |
| `text_entity_contrast` (all phases), then `--phases crossmodal` | 128 pairs | 128 pairs |

`--n_pairs 128` (flags used 64), because own-answer rows lose more pairs to a shared label (61 of 128
brands are US-headquartered), a wrong answer, or image/text disagreement.

**Self-checks** (per entity; report each as pass/fail with the number):

1. Sweep report: the `self` control's base_kept ≥ 0.99 and `paraphrase` ≥ 0.97 (flags: 0.9998 /
   0.9977). Eligible rows should be ≥ 50% of the rows with baselines. Report the per-attribute
   clean-accuracy line verbatim either way.
2. Sweep report: `--window` printed a `PATCH=... BLOCKS="..."` line. If it raised ("no handoff" or
   "degenerate window"), the launcher prints `no trace window`; report the handoff table.
3. Trace phase 0: base and donor `first` ≥ 95% (own-answer golds; anything lower means generation
   differs between batchings). Report the `own-answer golds: kept ... dropped ...` line.
4. Trace phase 1b asserts the head patch was reached (it raises otherwise). Phase 1c's verdict line
   must be either `MATCHES` or `diverges, and --blocks omits downstream block(s)` (as on flags). Any
   other verdict: report it and do not read phases 2-4.
5. Trace phase 2: `question patch only (k=0)` first ≥ 80%; it prints a warning below 10%. Phase 3:
   `nothing patched (k=0)` equals the `unhooked` row (it prints `!!` otherwise).
6. Compose: the first batch prints `[check]` read-back and self-install lines (they assert).
   On eligible-distinct rows, `clean` ≥ 95% C.X and `compose_random` ≥ 85% C.X (flags E4: 99.1% /
   99.1%).
7. Text contrast: `curves` has `n_eligible` ≥ 100 per condition. If fewer, say so; the `rows
   (dropped: {...})` line shows why. Crossmodal: X1's self-checks 1-3 and 5 apply unchanged
   (`n_eligible` has no fixed expectation here).
8. Any `KeyError`, `AssertionError` or `ValueError`: the lane stops. Quote it verbatim.

**Report back** (per entity):
- **Sweep:** the handoff table (layer, block, question resid, last resid, last attn), the
  attention-span table and the chosen window, with flags' for comparison (crossing in blocks 17-21;
  patch 15, blocks 15-22).
- **Trace:**
  - the top-20 heads table, per-block totals, the phase 2/3 curves and phase 4's verdict line;
  - do the top-8 routers sit in blocks 19-21, as on flags?
  - how many of flags' router top-8 recur
    (`logs/attr_head_trace/flags/attr_head_trace_patch15_blocks15-22_all_text_n84.json`)?
- **Compose:** the eligible-distinct table as printed, next to flags' E4 (n=548): movers 91.8% B.X,
  routers 85.6% C.A, compose 78.8% B.A, target_window 92.2% B.A, target_complement 91.8% C.X.
- **Text contrast:** the accuracy table, the curves table, L_e per condition, the head-set transfer
  table and the freeze table; for crossmodal, X1's report-back items.

### X3. A non-Qwen language model: Idefics3-8B-Llama3

**Question.** Do movers, routers and the late-MLP lookup exist on a model whose language model is not
Qwen2.5? Idefics3-8B-Llama3's is Llama-3.1-8B (32 blocks × 32 heads). InternVL3-8B shares Qwen2.5-7B's
text stack, so it does not count. Closes PAPER_TODO §0's first item.

Every head list, trace window and freeze span is derived from this model's own runs by
`methods/circuit_windows.py`. Its rules are unit-tested to reproduce Qwen's hand-picked arguments from
Qwen's own numbers. Never pass Qwen's blocks or heads (`common10`, 15-22, 22-27) to an X3 command.

**Needs, beyond section 1:** the Idefics3 download. VADE's code is not changed. Its accuracy sweep runs
unmodified through `methods/vade_accuracy_sweep.py`, which loads it by path and injects
`--processor_kwargs` (tests: `tests/test_vade_accuracy_sweep.py`). Like E8's InternVL run, the sweep and
`prune_tuples` still WRITE their outputs (predictions, pruned tuples) under
`../VADE/models/Idefics3-8B-Llama3/`, the path every reader here expects. Those are new files only;
nothing existing in VADE changes.

**Model facts the code relies on** (checked against transformers 5.16; the offline tests assert them):
- one 364 px tile (`do_image_splitting=False`), so 169 image tokens in a 13 × 13 grid;
- `shares_object_grid = False`, so only `full_image` and `last_token` position sets are valid;
- exactly one BOS (the default processor call doubles it; the adapter asserts it does not);
- `--execution cached` is refused for non-Qwen models; every step here is serial.

**Runs as** step `x3` (lane 1), which is `GPU=a100 bash scripts/run_x3_idefics3.sh all`. Its steps run in
this order and log to `logs/paper/x3_<step>_<stamp>.log`:
- **`gate`** (~15 min): sweeps flags capital / language and brands hq_country, and needs each ≥ 50%
  accurate. On `GATE FAILED` the lane stops; report the table. The rows land in the predictions file
  `prep` resumes from.
- **`prep`:** the full accuracy sweep on flags and brands (`methods/vade_accuracy_sweep.py
  --processor_kwargs '{"do_image_splitting": false}'`), then `prune_tuples`. Output goes to `../VADE/models/Idefics3-8B-Llama3/`.
- **`pipeline`:** `image_head_pipeline --n_layers 32 --n_heads 32` on the four flags attributes and on
  brands hq_country.
- **`overlap`:** `head_overlap` *within* Idefics3. Head IDs do not transfer across models.
- **`depth`:** the handoff and mover window as a fraction of depth, next to Qwen-7B, Qwen-3B and
  InternVL.
- **`routers`:** flags only. A `--baselines_only` gate, then the residual crossover sweep (layers 1-32),
  then `attr_head_trace` on the window `circuit_windows router` picks.
- **`severed`:** `head_severed --gold own --freeze_values base mean other`, once per flags attribute
  and once for brands hq_country. Each run uses that attribute's own top-8 movers, with freeze spans
  from their deepest block.

To rerun one step after a failure: `GPU=a100 bash scripts/run_x3_idefics3.sh <step>`. To leave out a
flags attribute whose trace was voided by self-check 4:
`X3_SEVERED_ATTRS="capital language" GPU=a100 bash scripts/run_x3_idefics3.sh severed`.

**Self-checks** (report each as pass/fail with the number):

0. Smoke: `x3_smoke_tests` reports 26 passed, 0 skipped.
1. Gate: `GATE PASSED`, with the three accuracies.
2. Pruning: `stats/*_tuples_prune_summary.md` keeps ≥ 64 test cause rows for each traced attribute.
3. Pipeline: every attribute passes its own scoring guard (layer-0 `full_image` swap ≥ 85% source
   match). If one stops after stage 1, report which, with its printed `swap_trace` command. Do not
   run that command.
4. Pipeline: each `head_trace` log's phase 1c ends in `-> MATCHES.`, and there is no
   `!! k=0 differs from unhooked` line. Any other verdict voids that attribute's trace: report it,
   and rerun `severed` without it (above).
5. Routers: `ROUTER GATE PASSED` (≥ 24 of 96 rows eligible). Every `/self` arm has
   donor_first ≤ 5%. `circuit_windows router` prints no `WARNING:` lines; if it does, quote them.
6. Severed: every run prints `self-checks: {... 'identity': 'bit-exact' ...}`.
7. Only new paths under `results/x3_idefics3/`, `results/image_head_pipeline_Idefics3-8B-Llama3/`,
   `logs/image_head_pipeline_Idefics3-8B-Llama3/`, `logs/paper/` and
   `../VADE/models/Idefics3-8B-Llama3/`.

**Outputs:**
- `results/x3_idefics3/`;
- `results/image_head_pipeline_Idefics3-8B-Llama3/` (`SUMMARY.md` per entity);
- `../VADE/models/Idefics3-8B-Llama3/`. This is a separate repo; package it too.

**Report back:**
- the accuracy table;
- both pipeline `SUMMARY.md` files;
- the `depth` table (Qwen-7B's flags movers are blocks 21-23 of 28, relative depth 0.79-0.86);
- the overlap consensus table;
- the router crossover curves, `router_window.json` and `attr_head_trace`'s phase-3 curve;
- the severed tables next to Qwen's E7 / P10;
- every derived-argument line the launcher echoes (`router window ...`, `... severed args ...`).

**How to read it:**

| result | reading |
|---|---|
| movers in one window, top-8 transfer high, freezing late MLPs kills it | same circuit on a Llama LM |
| movers found but the MLP freeze leaves transfer intact | the lookup sits elsewhere in this model |
| no router crossover (warnings) | the controlled prompts do not elicit a routed read here |

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

When the batch is done, write its report: `RUN_REPORT_PAPER.md` (batch 1, written) or
`RUN_REPORT_PAPER_BATCH2.md` (batch 2), with:
- the environment (GPU, torch and transformers versions, commit, MPS on/off);
- per-step status and wall time (each step's `done in N min` line), and which lane ran it;
- GPU usage from `logs/paper/gpu_usage.csv`: per-lane memory peaks, mean power, and whether
  three lanes beat Tier-3's ~1.5x;
- every self-check as pass/fail with its number;
- each experiment's "Report back" items;
- anything unexpected, quoted verbatim from the log.

Then package for the user, including the gitignored `.pt` files:
- batch 1: `results/paper/`, `results/text_entity_contrast/flags_crossmodal/`, `logs/paper/`;
- batch 2: `results/paper/x2b_*`, `results/paper/x2r_*`, `results/paper/smoke2/`, `results/x3_idefics3/`,
  `results/image_head_pipeline_Idefics3-8B-Llama3/`, `logs/image_head_pipeline_Idefics3-8B-Llama3/`,
  `logs/attr_head_trace/{brands,celebrities}/`, `logs/paper/`, and `../VADE/models/Idefics3-8B-Llama3/`.

## 5. Not runnable yet

Nothing. X2b and X3 moved into section 3 on 2026-10-19, once their code and offline tests landed.
