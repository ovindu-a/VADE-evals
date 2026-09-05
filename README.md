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
