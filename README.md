# VADE-evals

Execution repo for the [VADE](https://github.com/Shaveen12/VADE) benchmark
(Visual Attribute DisEntanglement). VADE itself stays a pure benchmark --
data, ground truth, prompt templates, object-token geometry, and a
method-agnostic scorer (`eval/score.py`) that "never touches model
internals." This repo holds everything that *does* touch model internals:
activation extraction, feature selection, and the actual interpretability
methods (DAS/MDAS, PCA, SAE, RLAP, DBM, ...) run against VADE's entities.

## Layout

```
VADE-evals/
  methods/
    sae.py                Phase-A: activation extraction
    activations/          extracted activation tensors (gitignored -- large binaries)
    features.py           shared dictionary classes (PCA/SAE) + pooling/label utilities
    fit_dictionaries.py   Phase-B step 2: fit PCA/SAE dictionaries per (entity, token_set, layer)
    dictionaries/         fitted dictionaries (gitignored -- large binaries) + fit_log.jsonl
    select_features.py    Phase-B steps 3-4: pool positions, two-step L1-SVC feature selection
    selections/           per-attribute sweep results + winners.json (small, not gitignored)

    common/, adapters/    copied verbatim from the sibling VADE repo's methods/common,methods/
                          adapters -- model/entity-agnostic infra (activation-patching hooks,
                          gold-token/teacher-forcing utilities, entity asset + batch loading,
                          per-entity source-activation caching, one ModelAdapter per model family).
                          Shared by any gradient-trained intervention method (currently just dbm/);
                          entirely separate code path from features.py/fit_dictionaries.py/
                          select_features.py above -- the PCA/SAE stack shares nothing with it.
    dbm/                  Differential Binary Masking (Cao et al. 2020/2022, evaluated in RAVEL --
                          see "methods/dbm/" section below) -- train.py/eval.py/run_layer.py/
                          layer_sweep.py, mirroring the sibling VADE repo's own methods/das/.
```

## Expected sibling layout

Scripts here default to reading VADE's data from a sibling checkout:

```
<some root>/
  VADE/            benchmark data + scorer
  VADE-evals/      this repo
```

Override with `--vade_root /path/to/VADE` or the `VADE_ROOT` environment
variable if your checkout isn't laid out this way.

## methods/sae.py

Loads Qwen2.5-VL-7B-Instruct, runs one forward pass per entity image, and
records the residual-stream hidden state at every decoder layer, restricted
to the object's image-token positions (e.g. flags' 8-token `flag_only` and
24-token `flag_ring1` sets from `<entity>/object_location.json`). This is
the raw activation corpus an SAE or PCA baseline is trained on downstream --
see the script's module docstring for the full design rationale (why no
question text is in the prompt, the row-major token-ordering assumption,
etc.).

Requires a real GPU -- 7B params, all-layer hidden states. Sanity-check
without one via `--dry_run`, which only validates that VADE's images and
metadata resolve correctly:

```
pip install "transformers>=4.49" torch torchvision pillow tqdm accelerate

python methods/sae.py --entity flags --dry_run
python methods/sae.py --entity flags --limit 3          # smoke test
python methods/sae.py --entity flags                     # full run
```

## methods/fit_dictionaries.py

Fits one PCA and/or SAE dictionary per (entity, token_set, layer) on
sae.py's extracted activations -- unsupervised, no attribute/entity labels
involved. Dictionaries are fit on *flattened per-position rows* (every real
token position across every image is a training row), not pooled entity
vectors, so the same dictionary can later encode a single token position at
intervention time.

PCA sweeps k (default 32/128/256, auto-capped to the sample count) at every
layer by default -- cheap. SAE is expensive and data-starved (each entity
has only a few hundred to a couple thousand flattened rows, nowhere near
typical SAE training-set sizes), so it trains only on that entity's own
activations with a deliberately modest dict_size (default 2x hidden_dim)
and only at a handful of representative layers by default (4/14/24 -- early,
the layer used for the existing DAS comparison, and late). See the script's
module docstring for the full rationale and how to fine-sweep around a
promising layer/dict_size region.

```
pip install scikit-learn   # in addition to methods/sae.py's requirements

python methods/fit_dictionaries.py --entity flags --method pca
python methods/fit_dictionaries.py --entity flags --method sae
python methods/fit_dictionaries.py --entity flags --method both \
    --token_sets flag_only --sae_layers 10,14,18 --sae_dict_size 14336
```

## methods/select_features.py

Pools a token-set's positions into one vector per image (mean -- attributes
are whole-entity facts, not local visual features, and the positions are
correlated, not independent samples), encodes with a fitted dictionary, then
runs two-step L1-SVC + SelectFromModel feature selection swept over
(direction, layer, C), independently per attribute:

  - **forward**: broad filter on the entity-ID label (1 example/class,
    deliberately weak -- fit on all data, never cross-validated) -> narrow
    to the attribute label within that subset (real stratified CV).
  - **inverse**: same function, labels swapped -- broad filter on the
    attribute label (CV'd) -> narrow to entity-ID (fit on all data).

Each candidate's finally-selected feature set is scored by a cause/iso proxy
(held-out accuracy predicting the target attribute, vs. chance-adjusted
leakage of every *other* attribute from those same dims) that mirrors
`eval/score.py`'s real `final_score = 1/2(cause + mean(iso))` shape at the
feature level. The (direction, layer, C) combination maximizing that score
is each attribute's winner, written to `selections/<entity>/<token_set>_<dict_method>/winners.json`
-- the `(layer, feature_indices)` a later intervention script consumes.

Note: on flags' 84-country sample, `capital` and `calling_code` are unique
per country (no class has 2+ members), so they behave exactly like the
entity-ID label -- no held-out split is possible and they're reported as
unscored rather than given a misleading result. `currency` has only one
repeated class (EUR x10). Only `language` (8 classes with 2+ members) has
enough within-class repetition on this sample to support real CV. This is a
property of the 84-flag dataset, not a bug -- worth knowing before reading
too much into "no winner" for an attribute.

```
python methods/select_features.py --entity flags --token_set flag_only --dict_method pca
python methods/select_features.py --entity flags --token_set flag_only --dict_method sae
```

## methods/intervene.py

Phase-B step 5, the actual causal intervention. For every attribute with a
winner in `select_features.py`'s `winners.json`: runs the source image
through the model once, caches its hidden state at the winning layer, then
on the base image's forward pass patches in the winning dictionary-feature
subset (`encode(base)`/`encode(source)`, copy over only the winning dims,
`decode`) at the object's token positions, and lets the model generate
freely. Writes predictions in `VADE/eval/score.py`'s format. No training
involved -- the "intervention" is just an encode/swap/decode through the
already-fitted dictionary.

```
python methods/intervene.py --entity flags --token_set flag_only \
    --dict_method sae --out methods/interventions/flags_flag_only_sae_predictions.jsonl
```

Predictions files are append-and-resume: re-running the same command skips
`(attribute, row_index)` pairs already present in `--out`, so an interrupted
run just picks up where it left off.

## eval/score.py (in the sibling VADE repo)

Method-agnostic scorer -- takes any predictions JSONL (the format above) and
reports `cause` (did the target attribute flip to the source's value) and
`iso` (did every *other* attribute stay at the base's value) accuracy per
attribute, plus `final_score = 1/2(cause + mean(iso))`.

```
python ../VADE/eval/score.py --predictions methods/interventions/flags_flag_only_sae_predictions.jsonl \
    --entity flags --attribute all
```

Writes `<predictions-file-stem>_summary.json` and `.md` next to the
predictions file (or under `--out_dir`).

## methods/dbm/ -- Differential Binary Masking

DBM (Cao et al. 2020/2022, evaluated as a RAVEL baseline in Huang et al.
2024, https://aclanthology.org/2024.acl-long.470/) learns a sigmoid-gated
binary mask `m` directly over the RAW residual-stream dimensions at one
layer -- no dictionary, no rotation (`F_A(n) = n`, unlike PCA/SAE/DAS):

```
n = (1 - sigma(m/T)) . GetVals(M(x), N) + sigma(m/T) . GetVals(M(x'), N)
L_Cause = CE(tau(M_{N<-n}(x)), A_E') + lambda * ||m||_1
```

`T` (temperature) is annealed continuously through training, sharpening the
sigmoid into a near-binary gate; RAVEL's own Appendix B.4 says this exact
mechanism was implemented via the [pyvene](https://github.com/stanfordnlp/pyvene)
library (`pip install pyvene`) -- `methods/dbm/intervention.py` imports
pyvene's own `SigmoidMaskIntervention` directly (verified byte-for-byte
against the paper's formula) rather than reimplementing it, but does NOT
use pyvene's `IntervenableModel` wrapper (built/tested against text-only HF
models, no documented VLM/pixel_values support) -- the actual hook
mechanism is this project's own `methods/common/hooks.py`, copied verbatim
from the sibling VADE repo's `methods/das/` and already proven against
Qwen2.5-VL there.

Unlike PCA/SAE (fit once per layer offline, then a cheap CPU-only
classifier-based feature selection), DBM's mask is trained end-to-end
against the real generation objective, with the model itself in the
training loop -- there is no separate "selection" step, and each
(entity, attribute, layer) combination needs its own training run:

```
pip install pyvene   # in addition to methods/sae.py's requirements

python methods/dbm/train.py --entity flags --attribute language --layer 14 --positions flag_ring1
python methods/dbm/eval.py  --entity flags --attribute language --layer 14 --positions flag_ring1 --split test

# train+eval+score one layer, or sweep several layers in one model load:
python methods/dbm/run_layer.py --entity flags --attribute language --layer 14 --positions flag_ring1
python methods/dbm/layer_sweep.py --entity flags --attribute language \
    --layers 4 10 14 18 24 --positions flag_ring1
```

`--l1_coef` (default `0.001`) and `--temperature_start`/`--temperature_end`
(default `1e-2`/`1e-7`) match RAVEL's own reported optimum/schedule for
DBM (Appendix B.4) -- not re-derived here. `--positions` takes the same
named sets as everything else in this project (`flag_only`/`flag_ring1`
for flags, etc.) plus `last_token` -- the closest analogue of RAVEL's own
*text-only* intervention site (the entity mention's single last token);
`flag_only`/`flag_ring1` are this project's own extension into the
image-object-span setting that PCA/SAE/DAS already use.

Training requires VADE's own baseline-pruned tuples (`VADE/models/
prune_tuples.py`, via `VADE/models/run_accuracy_sweep.py` first) by
default, same as DAS -- pass `--allow_unpruned` to opt out. Trained
checkpoints/predictions/logs land under THIS repo's own `results/`/`logs/`
trees (never under `--vade_root`); the shared per-entity source-activation
cache (reused across every method/config/layer for that entity, including
DAS if you've also run that) still lives under `--vade_root/results/`,
matching its own existing convention.

## Reproducing results on a fresh machine from the Kaggle-pool SAE checkpoints

This section is a self-contained runbook for picking up the pipeline on a
**different machine** starting from the SAE dictionaries fit against the
Kaggle "country-flags-in-the-wild" augmentation pool (6,500 real photos,
full 144-token canvas, mixed into training alongside VADE's own 84 flag
images -- see `fit_dictionaries.py`'s docstring). Those checkpoints
reconstruct meaningfully better than the real-only-84-image SAEs (see the
sweep numbers in `methods/dictionaries_kaggle_pool/flags/fit_log.jsonl`),
so this is the config worth running interventions against first. Every
command below is meant to be run as-is, in order, from a clean checkout.

### 0. Prerequisites

- A CUDA GPU (Qwen2.5-VL-7B-Instruct + generation; ~20GB+ VRAM recommended).
- `rclone` installed (`wget -qO- cli.runpod.net | sudo bash` works, or your
  distro's package manager) if pulling the checkpoints from Google Drive.
- Python 3.10+.

### 1. Clone both repos as siblings

```
git clone https://github.com/Shaveen12/VADE.git
git clone https://github.com/ovindu-a/VADE-evals.git
```

They must sit next to each other (`VADE/` and `VADE-evals/` as siblings) --
every script here defaults `--vade_root` to `../VADE` relative to this repo,
overridable via the `VADE_ROOT` env var if you need a different layout.
`VADE/data/` (images, ground_truth.json, object_location.json,
prompt_templates.json, tuples/) is tracked directly in the VADE repo, so
cloning it is all that's needed to get the benchmark data -- no separate
download step.

### 2. Python environment

```
cd VADE-evals
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision "transformers>=4.49" accelerate \
    scikit-learn pillow numpy tqdm huggingface_hub
```

(This repo pins no `requirements.txt`; the versions this pipeline was
built/verified against are torch 2.13, transformers 5.16, scikit-learn 1.9 --
install those explicitly if you hit an incompatibility with `latest`.)

### 3. Download the SAE checkpoints

The fitted dictionaries (`dictionaries_kaggle_pool/`, ~11GB, 58 `.pt` files
+ `fit_log.jsonl`) are too large to live in git (each checkpoint is
individually ~200MB, over GitHub's 100MB/file limit) and were pushed to
Google Drive instead:

```
rclone config create gdrive-transfer drive scope=drive.file \
    token='<paste a token from `rclone authorize "drive"`, run in a browser-capable shell>'

rclone copy --progress gdrive-transfer:VADE-evals-sae-checkpoints/dictionaries_kaggle_pool \
    methods/dictionaries_kaggle_pool
```

If a bulk copy stalls with climbing ETAs (minutes turning into "weeks"),
that's Google Drive's shared per-minute API quota for rclone's default
OAuth client throttling concurrent/many-small-request transfers -- not a
network problem. Fixes, cheapest first: (a) use `--drive-chunk-size 256M`
(each of these checkpoints is ~196MB, so this makes every upload/download a
single request instead of ~25 chunked ones), (b) drop concurrency
(`--transfers 2 --checkers 1`), (c) wrap the command in a restart loop since
`rclone copy` skips files already present:
```
while ! timeout 90 rclone copy --drive-chunk-size 256M --transfers 2 --checkers 1 \
    gdrive-transfer:VADE-evals-sae-checkpoints/dictionaries_kaggle_pool methods/dictionaries_kaggle_pool; do
  sleep 3
done
```
Verify the transfer matches before moving on: `rclone size
gdrive-transfer:VADE-evals-sae-checkpoints/dictionaries_kaggle_pool` should
report 59 objects / ~11.1 GiB, matching `find methods/dictionaries_kaggle_pool -type f | wc -l`.

### 4. Pre-fetch the model weights (optional, but avoids a silent 16GB first-run download)

```
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"
```

### 5. Re-extract the real (non-augmented) entity activations

`select_features.py` always scores against the *real* 84 flag images and
their real ground-truth labels (never the Kaggle pool, which was only ever
a training-data supplement for `fit_dictionaries.py`) -- so it needs
`methods/activations/flags_Qwen2.5-VL-7B-Instruct_all_layers.pt` on disk.
This is cheap and fast to regenerate locally rather than transfer:

```
python methods/sae.py --entity flags --dry_run     # sanity-check paths/images resolve
python methods/sae.py --entity flags               # full extraction, ~1GB, well under a minute on a GPU
```

### 6. Feature selection against the Kaggle-pool SAE dictionaries

Run once per token set. Because this SAE was fit at **every** layer
(0-28), not just the 4/14/24 subset from the original real-only run, expect
a wider sweep (more `sweep.jsonl` rows, longer runtime) than the earlier
examples in this README:

```
python methods/select_features.py --entity flags --token_set flag_only \
    --dict_method sae --dictionaries_dir methods/dictionaries_kaggle_pool

python methods/select_features.py --entity flags --token_set flag_ring1 \
    --dict_method sae --dictionaries_dir methods/dictionaries_kaggle_pool
```

Each writes `methods/selections/flags/<token_set>_sae/{sweep.jsonl,winners.json}`.
Check the printed per-attribute winner lines (or `winners.json` directly) --
an attribute with "no winner" means no (layer, direction, C) combination
scored (see the note above about `capital`/`calling_code`/`currency` having
too little within-class repetition on flags' 84-image sample for a
held-out CV split; this is a property of the dataset, not a bug).

### 7. Run the intervention

```
python methods/intervene.py --entity flags --token_set flag_only --dict_method sae \
    --dictionaries_dir methods/dictionaries_kaggle_pool \
    --out methods/interventions/flags_flag_only_sae_kagglepool_predictions.jsonl

python methods/intervene.py --entity flags --token_set flag_ring1 --dict_method sae \
    --dictionaries_dir methods/dictionaries_kaggle_pool \
    --out methods/interventions/flags_flag_ring1_sae_kagglepool_predictions.jsonl
```

Add `--limit 5` first if you want a fast smoke test before committing to a
full run (flags' test split is a few thousand rows across 4 attributes).

### 8. Score it

```
python ../VADE/eval/score.py \
    --predictions methods/interventions/flags_flag_only_sae_kagglepool_predictions.jsonl \
    --entity flags --attribute all --method_name sae_kagglepool_flag_only

python ../VADE/eval/score.py \
    --predictions methods/interventions/flags_flag_ring1_sae_kagglepool_predictions.jsonl \
    --entity flags --attribute all --method_name sae_kagglepool_flag_ring1
```

This writes `sae_kagglepool_flag_only_summary.{json,md}` (and the
`flag_ring1` counterpart) next to the predictions files, with `cause`/`iso`
per attribute and the overall `final_score = 1/2(cause + mean(iso))`. That
final number is the end of the pipeline -- it's what should be compared
against the existing DAS baseline (see VADE's own `results/` for those
numbers) and against the real-only (non-Kaggle-pool) SAE run, i.e. the same
steps 6-8 but with `--dictionaries_dir methods/dictionaries` (default) and
no `_kagglepool` suffix, to see whether the extra training pool actually
translated into better causal intervention accuracy, not just better
unsupervised reconstruction.
