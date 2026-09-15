# Attribute subspaces and contrastive directions

Two experiments operate on the selected attention outputs before W_O:

- `mdas`: learn a small subspace that transfers a source item's target attribute
  while preserving the recipient's other attributes.
- `contrastive`: estimate same-item, different-question mean differences and
  add a fixed direction to switch which attribute an unseen item is asked about.

These are different tasks, with separate output directories and metrics. Neither
changes model weights. Run from the repo in the existing GPU environment.

## Commands

```bash
# Validate the selected traces, item split, images and paired MDAS data without loading a model.
python3 methods/head_directions.py mdas --entity flags --dry_run
python3 methods/head_directions.py contrastive --entity flags --dry_run

# Small MDAS smoke run: fit only calling code, still preserve all other flag attributes.
python methods/head_directions.py mdas --entity flags --targets calling_code \
  --ranks 1 2 --head_ks 8 --train_pairs 2 --test_pairs 2 --epochs 1 \
  --out_dir results/head_directions_smoke

# Full rank sweep for all four flag attributes.
python methods/head_directions.py mdas --entity flags \
  --ranks 1 2 4 8 16 32 --head_ks 8 16

# Inexpensive contrastive smoke run: capital → currency, held-out countries/prompts.
python methods/head_directions.py contrastive --entity flags \
  --from_attributes capital --targets currency --contrast_train_limit 4 \
  --contrast_test_limit 2 --strengths 0.5 1 --mode prefill \
  --out_dir results/head_directions_smoke

# All directed flag-attribute switches with a strength sweep.
python methods/head_directions.py contrastive --entity flags \
  --head_ks 8 16 --strengths 0.25 0.5 1 2 --mode continuous
```

The full MDAS command fits 48 models (four targets × two head counts × six
ranks), so use the smoke run first. The default has only top 8 heads, reducing
that to 24 models. Repeat an identical command to resume; use a new `--out_dir`
when changing settings. Completed MDAS pair steps and completed evaluation rows
are reused. Contrastive mean fitting saves after the complete fitting pass;
an interruption during that pass restarts mean fitting, while evaluation resumes
per row. No experiment chooses its best rank/strength/checkpoint on test results.

## Restrict entities and individual items

`--entity flags` and `--entities flags` are aliases. Multiple categories are
supported, e.g. `--entities flags brands animals`. Each needs its **own saved,
compatible head traces**; flag head rankings are never silently reused for another
category. Default discovery searches:

`<trace_root>/<entity>/ndm/<attribute>/*/<trace_name>`

Defaults are `logs/Qwen2.5-VL-7B-Instruct` and
`head_trace_patch21_blocks21-23.json`. Use `--trace_root`, `--trace_name`, or supply
`--traces path/to/attribute1.json path/to/attribute2.json ...` explicitly. There
must be exactly one compatible trace per selected attribute. Missing/ambiguous
rankings fail with guidance before the model loads.

`--items FR DE IT ES ...` restricts actual country/item IDs. `--item_limit 40`
selects at most 40 items deterministically. `--train_items ... --test_items ...`
sets explicit non-overlapping identity pools. Explicit IDs require one entity
category per command, because categories have different ID namespaces.

By default, all available IDs are deterministically partitioned with
`--train_fraction 0.7` and `--seed 0`. On the current flag data that is 58 training
and 26 held-out country identities. No country can occur on both sides. The
original head-discovery traces are reused: identity holdout applies to fitting
these new edits, not retrospectively to the head-discovery procedure.

MDAS then samples official pruned train/test tuples **inside** those pools,
requiring both base/source IDs to belong to the corresponding pool. Each pair
must have every selected attribute/prompt and differing attribute values, as in
the head-mask experiments. Small explicit pools may have insufficient eligible
pairs; reduce `--train_pairs`/`--test_pairs` or enlarge the pools. Counts are never
silently reduced. Contrastive fitting uses individual images and labels, so it
does not require country-pair tuples.

## Parameters

| Option | Default | Purpose |
|---|---|---|
| `--attributes` | All entity attributes | Questions included; at least two |
| `--targets` | All selected attributes | MDAS masks to fit / contrastive destination attributes |
| `--from_attributes` | All selected attributes | Contrastive source questions only |
| `--head_ks` | `8` | Saved ranking cutoffs |
| `--head_selection` | `target` | Target's top-k, or union/intersection across selected attributes |
| `--ranks` | `1 2 4 8 16 32` | MDAS rank **per selected layer** |
| `--train_pairs`, `--test_pairs` | `32`, `16` | MDAS country-pair counts |
| `--train_templates` | `v1 v2 v3 v4` | Training prompt versions |
| `--test_templates` | `v5 v6` | Disjoint evaluation prompt versions |
| `--epochs`, `--lr` | `3`, `0.01` | MDAS Adam optimization |
| `--isolation_weight` | `1` | Mean other-attribute CE relative to target CE |
| `--grad_clip` | `1` | MDAS gradient norm clipping |
| `--contrast_train_limit` | `32` | Maximum fitting item count |
| `--contrast_test_limit` | `16` | Maximum evaluation item count |
| `--strengths` | `0.25 0.5 1 2` | Contrastive vector scales; zero/negative scales also supported |
| `--mode` | `continuous` | Prefill-only or all answer positions |
| `--max_new_tokens` | `12` | Full-answer generation budget |
| `--prefill` | `Answer:` | Shared contrastive answer prefill |
| `--model_id`, `--device`, `--vade_root` | Existing Qwen defaults | Model and data location |

Use `--targets calling_code` to train just that edit **without removing its
isolation questions**. Using `--attributes` actually narrows the preservation
claim. Union/intersection sets can contain more/fewer than k heads; exact head
lists and layer widths are saved. Invalid ranks exceeding any layer's selected
width fail before loading the model. Contrastive prompt versions refer to its
six controlled templates; MDAS versions refer to existing VADE templates.

## Multi-task DAS details

For each layer, concatenate only the selected heads' coordinates into z. Learn
a D×r matrix whose reduced QR factorization produces an orthonormal basis U:

`z_edited = z_recipient + U Uᵀ (z_donor − z_recipient)`

Layers have separate bases, optimized jointly. Cross-layer activations are never
concatenated into a single vector, which would mix different computation times.
A rank of 4 across three layers therefore supplies four directions per layer,
not four directions total. There is no sigmoid gate or soft/hard threshold.

The donor is the source-image-residual-patched run from the head experiments.
One optimizer step includes a complete item-pair bundle. The objective is mean
full-answer source CE for the target plus weighted mean base CE for the other
attributes. The same subspace stays active on every question. Teacher forcing
is used only for fitting. Evaluation freely generates; each donor forward uses
the recipient's current prefix and captures aligned activations for every edited
position. Full-prefix recomputation keeps earlier edits active without stale KV
cache entries. The model is frozen; only QR-parameterized bases are optimized.

Evaluation includes clean, image-patch, full-head, learned subspace, random
subspace of the same rank, and matched-strength uniform blending. The last
control scales the full donor difference to the learned projection's L2 norm
at each current layer/position; it tests local edit magnitude rather than a
selective direction. It does not claim to match downstream trajectories globally.

Outputs under `<out>/<entity>/mdas/<target>/<selection><k>/rank<r>/`:

- `subspace.pt`: trainable basis parameters, Adam state, step and configuration.
- `subspace.json`, `training_history.json`: ranks/widths/heads and per-attribute CE.
- `eval/cross_attribute_matrix.json`: target transfer and other-attribute preservation,
  including preservation conditioned on a clean-correct answer.
- `eval/joint_success.json`: target transfer **and every other attribute preserved**
  within the same pair/template-version bundle.
- `eval/rows.jsonl`: free generations, full-answer and per-token scores.

## Contrastive baseline details

For every training identity and prompt version, run all selected attribute
questions on the identical image. Collect final-prompt pre-W_O activations.
Estimate paired mean differences:

`direction[A→B] = mean_item,template(z(item,B) − z(item,A))`

Equal sampling makes the difference of saved attribute means exactly the paired
mean difference. Restrict each vector to the selected heads, then add
`strength × direction` to the recipient's head outputs. Test items and prompt
versions are unseen during estimation. No target-country answer or donor-question
activation is supplied to the edited evaluation forward. The target-question
clean baseline is a separate generation used only for evaluation.

Prompts use common output-format instructions and a shared prefill. Unlike MDAS,
this baseline uses six new controlled prompt variants with VADE images/labels,
not the original pruned tuples. Prompts can differ in length because vectors are
collected at each prompt's own final position; no token-by-token transplant occurs.
Clean accuracy and both-clean-correct metrics reveal whether the controlled
questions work. Matching items reduces country confounding but does not prove
that all wording or answer-format information has been removed.

Controls are clean, target-question clean, reversed direction, and random vectors
with the same per-layer norm and the same selected-head support. In continuous
mode the **same fitted vector** is added at every readout position; it is not a
fresh donor activation. Compare prefill/continuous modes in separate output roots.

Outputs under `<out>/<entity>/contrastive/`: `directions.pt` (attribute means and
configuration), `direction_metadata.json` (heads and direction norms),
`switch_summary.json` (target/original/neither rates by directed attribute pair,
both unconditional and conditioned on both clean answers being correct), and
`rows.jsonl`. These are attribute-selection scores, not selective-value-edit scores.

All full-answer metrics use every canonical gold suffix token, accepting trailing
text as in the head follow-ups. They are not semantic/alias-aware. Insufficient
generation budgets fail rather than silently truncating labels. Both experiments
save configuration/code/image provenance and runtime versions and reject
incompatible resumes. Rank and strength sweeps are exploratory: use a separate
validation split before claiming a selected configuration's test performance.
