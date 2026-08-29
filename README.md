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
    sae.py            Phase-A activation extraction for the SAE baseline
    activations/      extracted activation tensors (gitignored -- large binaries)
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
pip install "transformers>=4.49" torch pillow tqdm accelerate

python methods/sae.py --entity flags --dry_run
python methods/sae.py --entity flags --limit 3          # smoke test
python methods/sae.py --entity flags                     # full run
```
