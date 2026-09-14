# Attribute editing through the selected image heads

These are the two stages following the head sufficiency and token-knockout runs.
They test whether the selected heads support selective attribute transfer. They
do not assume that head necessity establishes attribute-specific information.

## Commands

Run in the same GPU environment as the head follow-ups. No new dependencies.

```bash
# Validate both phases' shared dataset without loading the model.
PY=python3 bash scripts/run_head_attribute_edit.sh matrix --dry_run

# Phase 1: each attribute's top 8/16 heads × every queried attribute.
bash scripts/run_head_attribute_edit.sh matrix

# Phase 2: fit each target attribute's masks, then evaluate on held-out pairs/prompts.
bash scripts/run_head_attribute_edit.sh train

# Or fit only calling_code; isolation still tests language, capital and currency.
bash scripts/run_head_attribute_edit.sh train --target calling_code
```

For a quick smoke test, use a separate directory and the **same data flags in
both phases**:

```bash
bash scripts/run_head_attribute_edit.sh matrix --train_pairs 2 --test_pairs 2 --out_dir results/head_attribute_smoke
bash scripts/run_head_attribute_edit.sh train --train_pairs 2 --test_pairs 2 --out_dir results/head_attribute_smoke --epochs 1 --target calling_code
```

Default: 32 train pairs, 16 test pairs, train prompt versions v1–v4, test v5–v6,
three epochs, Adam LR 0.05, isolation weight 1, sparsity weight 0.01, sigmoid
temperature 1. Set `--head_ks 8` in **both** phases to run just top 8. Use a fresh
output directory when changing data, head sets, code or training hyperparameters.
The direct CLI accepts arbitrary compatible entity traces with `--traces`;
the launcher supplies the four existing flag traces.

## Phase 1: fixed-head cross-attribute matrix

Each head set is ranked once from the complete saved `phase1` scores and then
frozen. Every set gets exactly the same ordered country pairs and questions.
There are also clean, source-image-patched and fixed random-head controls.
Only pruned cause tuples are used to construct the paired questions; every
included pair must have every requested prompt for every attribute, and distinct
base/source labels for all attributes. This excludes trivial isolation cases,
but means results describe this restricted pair population.

Output: `results/head_attribute_edit/flags/matrix/cross_attribute_matrix.json`.
Each cell records source full-answer transfer, base full-answer accuracy, and
base preservation conditioned on the clean answer being correct. `rows.jsonl`
contains generated text, token scores and pair IDs for paired analysis;
`summary.json` provides additional per-token metrics grouped by arm.

## Phase 2: a fixed coordinate mask within each selected head set

For each target and head-set size, learn one sigmoid gate per head coordinate:

`z_edited = z_recipient + sigmoid(mask_logits / temperature) * (z_donor - z_recipient)`

The model stays frozen. The **same mask stays active for every queried attribute**.
The training objective is full-answer source CE on the target question plus
weighted mean base-answer CE on the other attributes, plus mean-gate sparsity.
Each optimizer step includes one complete country-pair bundle; prompt counts
and the number of isolation attributes do not change the cause/isolation balance.
The mask starts at 0.5. Temperature stays fixed; hard evaluation thresholds at
0.5. This is coordinate selection, not a learned rotation or low-rank subspace.

Teacher forcing is used only for optimization. Evaluation freely generates with
soft and hard masks, plus clean, image-patch and unmasked full-head controls.
All interventions continue from the final prompt token through every generated
position. Every donor uses the recipient's actual prefix. Full-prefix recomputation
keeps earlier substitutions in force without stale cached states.

Outputs: `masks/<target>/top<k>/mask.pt` (mask + Adam state, resumed after each
completed pair), `mask.json` (per-head gates and active coordinates),
`training_history.json`, and `eval/cross_attribute_matrix.json`. Evaluate **cause
transfer and isolation together**; an empty mask can preserve attributes while
failing to transfer the target. Soft success with hard failure also needs reporting.

The train/test pairs are disjoint even under reversing base/source, and prompt
versions are disjoint for mask fitting/evaluation. Entities themselves are not
held out. Existing head rankings are reused: the prompt holdout applies to mask
fitting and is not a claim that the original head discovery held out these prompts.
No test results select checkpoints: the final fixed-epoch checkpoint is evaluated.
All gold suffix tokens are scored; insufficient generation budgets raise an error
instead of silently truncating answers. A full match accepts the full gold prefix
even if the model continues with explanatory text, consistent with the follow-ups.

The shared `manifest.json` freezes exact rows, head rankings, metadata and code
hashes. Per-run configs and runtime metadata reject incompatible resumes. These
experiments recompute the multimodal model for each donor/recipient pass and are
intentionally expensive; start with the smoke commands.
