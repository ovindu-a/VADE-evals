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
                          Shared by any gradient-trained intervention method (dbm/) AND by the
                          read-only probes (probe_common.py/logit_lens.py/attention_maps.py) --
                          entirely separate code path from features.py/fit_dictionaries.py/
                          select_features.py above -- the PCA/SAE stack shares nothing with it.
    dbm/                  Differential Binary Masking (Cao et al. 2020/2022, evaluated in RAVEL --
                          see "methods/dbm/" section below) -- train.py/eval.py/run_layer.py/
                          layer_sweep.py, mirroring the sibling VADE repo's own methods/das/.

    probe_common.py        shared row-selection + teacher-forced-forward helper for the three
                          read-only probes below (no training, no intervention -- just "how does
                          the model's own forward pass arrive at its answer").
    logit_lens.py          per-layer "does the correct answer already dominate the vocabulary-
                          space prediction" probe (top-k tokens, other-attributes' rank) -- see
                          "methods/logit_lens.py, methods/attention_maps.py, methods/mech_probe.py"
                          section below.
    attention_maps.py      per-(layer, head) attention-flow probe (image->text conduits ranked +
                          thresholded, answer/attribute-mention breakdowns, raw quantized attention
                          dumps) -- same section below.
    mech_probe.py          runs both probes above off ONE forward pass per row instead of two --
                          same section below.
    logit_lens/, attention_maps/   small JSON/JSONL reports from the probes above (not gitignored
                          -- human-inspectable, like selections/); attention_maps/raw/ (optional,
                          gitignored via *.pt) holds --dump_raw_rows' quantized attention tensors.
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

`train.py` prints a `[progress] epoch=... opt_step=X/N loss=... ce=... l1=...
temp=... lr=... elapsed=...min eta=...min` line every completed optimizer
step (in addition to the full per-step record already written to
`layer{L}_train_log.jsonl`) -- unlike VADE's own `das/train.py`, which only
prints once per epoch, silent for however long a full epoch over a real
tuples file takes otherwise. `tee_to_log` mirrors all of this to a file
under this repo's own `logs/` tree too, so `tail -f` works on a detached run.

Training requires VADE's own baseline-pruned tuples (`VADE/models/
prune_tuples.py`, via `VADE/models/run_accuracy_sweep.py` first) by
default, same as DAS -- pass `--allow_unpruned` to opt out. Trained
checkpoints/predictions/logs land under THIS repo's own `results/`/`logs/`
trees (never under `--vade_root`); the shared per-entity source-activation
cache (reused across every method/config/layer for that entity, including
DAS if you've also run that) still lives under `--vade_root/results/`,
matching its own existing convention.

## methods/logit_lens.py, methods/attention_maps.py, methods/mech_probe.py -- read-only interpretability probes

Two small, ad hoc probes -- NOT a sweep, NOT a new pipeline stage, and they
train/patch nothing. Both ask "given the model's own unpatched forward
pass on a real VADE prompt, how does it arrive at its answer" over a
handful of example rows (`--limit` distinct images x `--questions_per_image`
phrasings each, default 12x6 = up to 72 rows), which is a different
question from everything above: PCA/SAE/DAS/DBM all ask "can we causally
steer this attribute," these two just look at what the frozen model is
already doing. Both are built on `methods/probe_common.py`, which reuses
`common/entities.py`'s REAL question+prefill machinery (unlike `sae.py`,
whose activations are provably independent of the question text -- see
its own docstring -- so they can't answer this). `methods/mech_probe.py`
runs BOTH probes off a single forward pass per row instead of the two
separate passes running each script alone would cost -- see "Running both
at once" below.

This section is written as a self-contained runbook -- e.g. for a fresh
Claude Code session picking up this repo with no prior context on these
scripts.

### 0. Prerequisites

- Same Python env as the rest of this repo (`pip install torch torchvision
  "transformers>=4.49" accelerate scikit-learn pillow numpy tqdm
  huggingface_hub`) -- no extra dependency beyond what `sae.py` already needs.
- A sibling `../VADE` checkout with `data/<entity>/` built (images,
  `ground_truth.json`, `object_location.json`, `prompt_templates.json`).
- **Every command below has a `--dry_run` mode that needs NO GPU and does
  NOT download the 7B model weights** -- it only loads `AutoConfig`
  (a few KB) to validate that rows/positions/images resolve correctly.
  Always run `--dry_run` first, especially after editing
  `--entity`/`--attribute`/`--positions`, before spending GPU time.
- The actual (non-dry-run) probe needs a CUDA GPU with the full
  Qwen2.5-VL-7B-Instruct weights (~20GB+ VRAM recommended, same as `sae.py`).
- Pruned tuples (`VADE/models/prune_tuples.py`, via
  `VADE/models/run_accuracy_sweep.py` first) are required by default, same
  as `dbm/train.py` -- pass `--allow_unpruned` to opt out and read
  `data/<entity>/tuples/` directly instead.

### 1. Validate first, no GPU needed

```bash
python methods/logit_lens.py     --entity flags --attribute language --dry_run
python methods/attention_maps.py --entity flags --attribute language --dry_run
python methods/mech_probe.py     --entity flags --attribute language --dry_run
```

Expected output: the resolved example rows (DISTINCT flag images x
question phrasings -- see `probe_common.load_probe_rows`'s
`one_per_image`/`templates_per_image`, which every script here passes, so
a small `--limit` doesn't silently return N phrasings of the SAME image),
how many of the entity's image tokens `--positions` resolves to, and
confirmation every row's image file exists on disk. No model, no GPU, no
download beyond `AutoConfig`.

### 2. Run for real, on a GPU box

```bash
python methods/logit_lens.py     --entity flags --attribute language
python methods/attention_maps.py --entity flags --attribute language
```

`logit_lens.py` prints a per-layer curve (`top1_match_rate`,
`mean_gold_rank`, averaged over every valid row/answer-position) across
**every** decoder layer (0 = embedding output .. 28 = the model's real
output for Qwen2.5-VL-7B -- cheap enough not to subsample; override with
`--layers` to restrict). Two extra layers of detail beyond the aggregate
curve, both in the per-row `.jsonl` (not the `_summary.json`, since
neither collapses cleanly into one number per layer):

- **`--top_k` (default 5)**: each layer/position's top-K predicted
  tokens (id, decoded text, probability), not just the top-1 -- see what
  the model is "considering" at a layer, not just whether it's already
  right.
- **other attributes' rank** (on by default; `--skip_other_attributes` to
  turn off): at each layer, where would THIS entity's OTHER scored
  attributes' own ground-truth values rank, evaluated as a continuation of
  THIS row's own prefill (`probe_common.other_attribute_gold_toks` --
  BPE-tokenized against the queried attribute's own prefill text, not in
  isolation, so it's comparable to the primary gold-rank at the exact same
  sequence position). Answers "does the model already carry
  capital/currency/calling_code information at this position even though
  only language was asked, or does only the queried attribute ever
  surface" -- also aggregated into the `_summary.json`'s
  `other_attributes` field per layer.

Writes `methods/logit_lens/<entity>_<attribute>_<positions>_report.jsonl`
(full per-row/per-layer/per-position detail, including top-k and other-
attribute ranks) and `..._summary.json` (the aggregated curves).

`attention_maps.py` requires `attn_implementation="eager"` internally
(handled automatically -- sdpa/flash-attention silently return `None` for
attention weights even with `output_attentions=True`) and only scores a
representative layer spread by default (`--layers 4 10 14 18 24`, matching
`dbm/layer_sweep.py`'s own example spread, for comparability) since eager
attention's memory cost scales with sequence length squared -- pass every
index `0..26` explicitly for an all-layer sweep. It prints these and
writes them to `methods/attention_maps/<entity>_<attribute>_<positions>_report.json`:

1. **text->image flow heads**, reported two ways: `flagged_text_to_image_heads`
   (a fixed `--threshold`, `"layer.head"` list, can come back sparse or
   even empty) AND `ranked_image_attending_heads` (**the top `--top_n_heads`,
   default 20, by this same score, ungated by any threshold** -- "the
   heads that most attend to image tokens," always populated). Both
   measure mean attention mass question-text query positions place on
   image-token keys -- candidate image-to-text information conduits, and
   the only causally valid direction to check: Qwen2.5-VL's image tokens
   sit *before* the question text, so under causal masking they can never
   attend forward into it.
2. **answer-position attention breakdown**: at the model's first
   answer-prediction position, how attention splits across the entity's
   own tokens / other image tokens / question text / prior answer tokens
   -- "what the model looks at when it answers." Reports two different
   "most object-focused head" picks per layer: `top_object_attending_head`
   (highest RAW object-group share) and `top_object_selective_head`
   (highest object share / (object share + image_other share) --
   normalizes away image_other's ~5x token-count advantage for
   `flag_ring1`, so this can legitimately be a DIFFERENT head).
3. **attribute-mention-token attention breakdown**: same 4-way split (+
   the same two top-head stats), but queried FROM the specific token that
   names the attribute in the question itself (e.g. the `" language"`
   token in "What is the official language..."), located via
   `probe_common.attribute_mention_col` +
   `adapters.qwen2_5_vl.Qwen25VLAdapter.find_last_phrase_token_col` --
   "when the question first mentions the attribute, where does that token
   look." Registered phrasings live in `probe_common.ATTRIBUTE_KEYWORDS`
   (currently flags' 4 scored attributes, verified against all 6 template
   variants each); an unregistered attribute or an unmatched new template
   wording is reported as `0/N rows matched`, not a crash -- extend that
   dict rather than assuming the number silently means something else.
4. **`--dump_raw_rows N` (default 0 = off)**: the FULL (not group-
   summarized) attention weights for the first N rows, quantized to uint8
   (`attn_weight * 255`, rounded -- reconstruct via `value / 255.0`;
   ~1/4 the size of bf16) purely so you can hand-plot a real heatmap
   instead of reading only the aggregate summaries above. Restricted to
   `--dump_raw_layers` (defaults to `--layers`) to keep file size sane
   (roughly `n_layers * 28 heads * seq^2` bytes per row -- ~5.6MB/row for
   5 layers at a ~200-token sequence). Written under
   `methods/attention_maps/raw/<entity>_<attribute>_<positions>/row<N>.pt`
   -- already covered by the project's `*.pt` gitignore rule, same as
   activations/dictionaries.

### 3. Running both at once (recommended if you want both anyway)

```bash
python methods/mech_probe.py --entity flags --attribute language
```

Runs the exact same two scorers (`logit_lens.score_one_row`/
`attention_maps.score_one_row`) off **one** forward pass per row
(`output_hidden_states=True` AND `output_attentions=True` together)
instead of two, and writes the *identical* two output files the standalone
scripts would (tagged `"single_forward_pass": true`). Defaults to every
decoder layer for both probes (matching the "all-layer" runs this project
has already done by hand) -- pass a smaller `--layers` to cut attention's
memory cost back down if a full sweep isn't needed; logit lens always also
scores the final "real output" pseudo-layer on top of whatever `--layers`
you pass, since it's free. Same `--dump_raw_rows`/`--top_k`/`--threshold`/
etc. flags as the two standalone scripts (see `--help`).

### 4. Reading the results

- `logit_lens.py`: rising `top1_match_rate` / falling `mean_gold_rank`
  across layers shows WHERE in the decoder stack the correct answer
  becomes dominant -- compare that layer against whichever layer
  `select_features.py`'s sweep picked as its winner for the same
  attribute, and against wherever `intervene.py`/`dbm/train.py` actually
  patches. If an OTHER attribute's rank also drops sharply around the same
  layers as the queried one, that's evidence of a shared "which entity is
  this" representation feeding every attribute, not an attribute-specific
  computation.
- `attention_maps.py`: a high `object` share in the attribute-mention
  breakdown at an early/mid layer would mean the model resolves "which
  image tokens are relevant" as soon as it reads the attribute's name in
  the question, well before it starts generating; a high `text_to_image`
  score at a DIFFERENT layer than intervene.py's/dbm's patch layer is a
  candidate explanation for the proxy-vs-real intervention gap already
  documented in RESULTS.md -- the causal information may be flowing
  through a layer/position this project isn't currently patching. The same
  head showing up in `ranked_image_attending_heads` across DIFFERENT
  entities/attributes is a stronger claim than any single run: it suggests
  a general-purpose image-to-text routing head, not something specific to
  one attribute -- worth an ablation follow-up.
- None of these scripts are resumable or append-only (unlike
  `intervene.py`'s predictions) -- a forward-pass-only probe over a
  handful of rows is cheap enough to just rerun with different
  `--entity`/`--attribute`/`--positions`/`--limit` rather than needing to
  resume a partial run.

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
