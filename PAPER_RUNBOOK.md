# Paper runbook: experiments for the final paper

Instructions for the session running on the GPU pod. This file is a queue: experiments are
added over time (section 3) and run together in one batch. `PAPER_TODO.md` says *why* each
one is needed; this file only says what to run, what must hold, and what to report.
`CLAUDE.md` explains every script.

## Ground rules

- **Do not edit anything under `methods/`, `tests/` or `scripts/`, and do not change any
  hyperparameter, head list, layer or row count.** A failed self-check is evidence; patching
  the code to make it pass destroys it. Stop that step, record what happened, and report.
- **The one allowed change:** if a step runs out of GPU memory, rerun just that step with the
  batch size halved (`--batch_size`), and record that you did.
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
mkdir -p logs/paper
```

`tests/test_crossmodal.py` has never been run (it was written off-pod without torch). If
any test in it fails, stop and report the failure verbatim; do not run X1.

Run everything inside tmux (`tmux new -s paper`), each step logged with `tee`.

## 2. Run order

Run the batch in the order of section 3. Smoke each new experiment first: smoke numbers mean
nothing, they only prove the step runs end to end. Then run the full version.

| ID | Experiment | PAPER_TODO item | Smoke | Full | Est. time (unverified) |
|---|---|---|---|---|---|
| X1 | Mover-head content, image vs text | §2 image-vs-text convergence | yes | yes | ~45 min |
| X2 | Brands + celebrities: mover install, cross-question, MLP lookup | §0 full story on brands/celebrities | yes | yes | ~3-4 h |

Not runnable yet (section 5 lists the code each one needs; do NOT attempt them on the pod):
X2b (brands/celebrities routers, compose, text contrast) and X3 (Idefics3, a non-Qwen LM).

## 3. Experiments

### X1. Mover-head content: image vs text-named entity

**Question.** E6 showed the same heads move the entity in both conditions. Is what they
carry the same? Two parts in one run: representational (does image-run France sit nearest
text-run France, per layer and per mover head?) and causal (do text-run mover values install
into an image run, and the reverse?).

**Smoke:**

```bash
python methods/text_entity_contrast.py --phases crossmodal --n_pairs 8 \
    --out_dir results/text_entity_contrast/smoke_crossmodal 2>&1 | tee logs/paper/x1_smoke.log
```

**Full:**

```bash
python methods/text_entity_contrast.py --phases crossmodal \
    --out_dir results/text_entity_contrast/flags_crossmodal 2>&1 | tee logs/paper/x1_crossmodal.log
```

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

**Setup** (once per shell; same helper as `scripts/run_tier3.sh`):

```bash
P=results/image_head_pipeline; T=../VADE/models/Qwen2.5-VL-7B-Instruct; O=results/paper
COMMON10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17
top8() { python - "$@" <<'EOF'
import glob, json, sys
root, entity, attribute = sys.argv[1:4]
paths = sorted(glob.glob(f"{root}/{entity}/{attribute}/head_trace_*.json"))
assert paths, f"no head_trace JSON under {root}/{entity}/{attribute}"
print(",".join(f"{b}.{h}" for b, h in json.load(open(paths[0]))["phase1_ranked"][:8]))
EOF
}
BH=$(top8 $P brands hq_country); CN=$(top8 $P celebrities nationality); CL=$(top8 $P celebrities native_language)
C6=21.1,22.15,22.17,23.3,23.4,23.6        # shared by all three entity top-8s
echo "BH=$BH"; echo "CN=$CN"; echo "CL=$CL"
```

Self-check 0: the three lists must print exactly
`BH=23.3,21.4,22.17,23.4,23.6,22.15,23.17,21.1`,
`CN=23.3,22.17,23.4,23.6,21.4,22.15,21.5,21.1`,
`CL=22.17,23.4,21.1,23.6,23.3,22.19,22.15,19.6`. Anything else: stop and report.

**Smoke** (only proves each script runs; numbers mean nothing):

```bash
python methods/head_swap_vade.py --entity brands --tuples_dir $T/brands/tuples --head_sets "hq_top8=$BH" \
    --arms clean heads --limit_pairs 4 --out_dir $O/smoke_x2_swap 2>&1 | tee logs/paper/x2_smoke_swap.log
python methods/head_cross.py --entity celebrities --heads "$CN" --n_pairs 4 --seed 0 \
    --out $O/smoke_x2_cross/cross_n4.jsonl 2>&1 | tee logs/paper/x2_smoke_cross.log
python methods/head_severed.py --entity brands --attributes founded_year --heads "$BH" --n_pairs 8 --gold own \
    --freeze_values base --freeze_mlp_span 23-26 --freeze_attn_span '' \
    --out_dir $O/smoke_x2_severed 2>&1 | tee logs/paper/x2_smoke_severed.log
```

**Full.** (a) Continuous mover install, every attribute, VADE-scored. `--patch_layer 19` is
where both entities' image plateau ends (the default 21 under-reads them); it only affects
the `full_image` arm.

```bash
python methods/head_swap_vade.py --entity brands --tuples_dir $T/brands/tuples \
    --head_sets "hq_top8=$BH" "cons6=$C6" "common10=$COMMON10" \
    --arms clean heads random_heads full_image --patch_layer 19 --limit_pairs 128 --batch_size 64 \
    --out_dir $O/x2_swap_brands 2>&1 | tee logs/paper/x2_swap_brands.log
python methods/head_swap_vade.py --entity celebrities --tuples_dir $T/celebrities/tuples \
    --head_sets "nat_top8=$CN" "lang_top8=$CL" "cons6=$C6" "common10=$COMMON10" \
    --arms clean heads random_heads full_image --patch_layer 19 --limit_pairs 128 --batch_size 64 \
    --out_dir $O/x2_swap_celebrities 2>&1 | tee logs/paper/x2_swap_celebrities.log
# score every arm, per attribute, against the same pruned tuples
for e in brands celebrities; do
  for f in $O/x2_swap_$e/*.jsonl; do
    for a in $(ls $T/$e/tuples | grep -v identity); do
      python ../VADE/eval/score.py --predictions "$f" --entity $e --attribute $a --tuples_dir $T/$e/tuples \
          --out_dir $O/x2_swap_$e/scores/$(basename "$f" .jsonl)/$a
    done
  done
done 2>&1 | tee logs/paper/x2_swap_scores.log
```

(b) Cross-question patching. Brands stays at `--seed 0 --n_pairs 64`: larger samples draw
AIRBUS, which has no `industry` label and crashes `head_cross` (a known bug, fix G in
section 5). `step0` is the alignment control R11 used.

```bash
for al in matched step0; do
  python methods/head_cross.py --entity brands --heads "$BH" --n_pairs 64 --seed 0 --align $al --batch_size 64 \
      --out $O/x2_cross_brands/cross_n64_$al.jsonl 2>&1 | tee logs/paper/x2_cross_brands_$al.log
  python methods/head_cross.py --entity celebrities --heads "$CN" --n_pairs 64 --seed 0 --align $al --batch_size 64 \
      --out $O/x2_cross_celebrities/cross_n64_$al.jsonl 2>&1 | tee logs/paper/x2_cross_celebrities_$al.log
done
```

(c) MLP freezing on the year and category attributes (Z2's test for `founded_year`).
Same settings as the committed E7 runs.

```bash
SEV=(--n_pairs 128 --gold own --freeze_values base mean other --freeze_mlp_span 22-27
     --freeze_mlp_singles 23 24 25 26 --freeze_attn_span 24-27 --batch_size 64)
python methods/head_severed.py --entity brands --attributes founded_year industry --heads "$BH" "${SEV[@]}" \
    --out_dir $O/x2_severed_brands_year_industry 2>&1 | tee logs/paper/x2_severed_brands.log
python methods/head_severed.py --entity celebrities --attributes occupation birth_year --heads "$CN" "${SEV[@]}" \
    --out_dir $O/x2_severed_celebrities_occ_year 2>&1 | tee logs/paper/x2_severed_celebrities.log
```

**Self-checks:**

0. The head lists match (above).
1. Install: `random_heads_*` cause ≤ 5% on every attribute; `clean` cause ≈ 0.
2. Install: `full_image` cause ≥ 70% on hq_country, nationality and native_language (the
   pipeline's own image-swap ceiling there was 90-100%).
3. Cross: each run prints its cells and a self-patch control. Self-patch must keep ≥ 85% of
   the base answer (flags read 91.8-93.4%).
4. Severed: every run prints `self-checks: {... 'identity': 'bit-exact' ...}`. Anything
   else: stop that run and report.
5. If any step raises `KeyError` on an item or attribute, report it verbatim; do not
   change the seed or sample size to dodge it.

**Outputs:** `results/paper/x2_*`.

**Report back:**
- Per entity × attribute × arm: cause / iso / final from the score summaries.
- Each cross run's cell table, matched and step0 side by side, next to flags' R11 (81.2% same
  question, 65.1% donor asked another question).
- The severed tables for the four new attributes.
- One line on whether the category attributes (industry, occupation) behave like entity
  facts or like the animals category traits (P14).

<!-- Template for the next experiment. Copy it, give it the next X-number, and add it to
the table in section 2.

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
- the environment (GPU, torch and transformers versions, commit);
- per-step status and wall time;
- every self-check as pass/fail with its number;
- each experiment's "Report back" items;
- anything unexpected, quoted verbatim from the log.

Then package `results/` and `logs/paper/` for the user, including the gitignored `.pt` files.

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
