# VADE-evals

The latest head-trace follow-ups—continuous answer-token substitution and
two-phase token identification/knockout—are documented in
[HEAD_EXPERIMENTS.md](HEAD_EXPERIMENTS.md).

The Tier-1/Tier-2 circuit experiments (E1-E11: what the image->answer circuit
is, and whether it generalizes across entities and models) are summarized in
[Circuit experiment results](#circuit-experiment-results-tier-1--tier-2).

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
                          Shared by any gradient-trained intervention method (dbm/, ndm/) AND by the
                          read-only probes (probe_common.py/logit_lens.py/attention_maps.py) --
                          entirely separate code path from features.py/fit_dictionaries.py/
                          select_features.py above -- the PCA/SAE stack shares nothing with it.
                          TWO FILES HERE ARE NOT VERBATIM COPIES, both added for ndm/ and both new
                          FILES rather than edits, so the copied ones stay byte-identical to VADE's
                          and keep taking upstream fixes cleanly:
                            common/sites.py             the InterventionSite abstraction -- WHICH
                                                        tensor inside a decoder block an intervention
                                                        reads/patches (residual / attn_output /
                                                        attn_head_output / mlp_output / mlp_hidden),
                                                        plus the JOINT site attn_output+mlp_output,
                                                        which patches both of a block's sublayer
                                                        contributions in one pass (diagnostic only).
                                                        The residual site DELEGATES to
                                                        common/hooks.py's own functions, so DBM/DAS
                                                        behavior is unchanged by its existence.
                            common/position_sets.py     WHICH token columns a site is read/patched
                                                        at, beyond the four entities.py knows --
                                                        image-grid bands (ring:/side:/seq:/~),
                                                        text windows (tok:/pre_image/vision_end),
                                                        semantic tokens (phrase:), and A+B unions.
                                                        A wrapper that re-selects columns from a
                                                        batch build_batch already produced, so
                                                        entities.py is never edited.
                            common/site_source_cache.py the MLP sites' source-activation cache (one
                                                        file per site/positions/layer). MLP internals
                                                        are absent from output_hidden_states, so
                                                        common/source_cache.py cannot serve them.
                          adapters/{base,qwen2_5_vl}.py also gained three ADDITIVE methods absent
                          from VADE's copies (intermediate_size/get_mlp_block/
                          get_mlp_hidden_module), so common/sites.py can address MLP internals while
                          adapters/ stays the only module that knows a model family's attribute names.
    dbm/                  Differential Binary Masking (Cao et al. 2020/2022, evaluated in RAVEL --
                          see "methods/dbm/" section below) -- train.py/eval.py/run_layer.py/
                          layer_sweep.py, mirroring the sibling VADE repo's own methods/das/.
                          Also the shared ENGINE behind ndm/ below: its train_layer/eval_layer/
                          run_one_layer/run_sweep take an optional `site=`, defaulting to the
                          residual stream (i.e. plain DBM, unchanged).
    ndm/                  Native Dictionary Masking -- DBM's trainer, with the mask learned over a
                          decoder block's MLP HIDDEN state (the post-SwiGLU neuron vector, width
                          intermediate_size) instead of the residual stream. That makes the model's
                          own MLP the featurizer: a nonlinear, zero-parameter dictionary that needs
                          no fitting and has no reconstruction error. Thin CLIs (config.py/train.py/
                          eval.py/run_layer.py/layer_sweep.py) over dbm/'s engine, plus two
                          diagnostics to run BEFORE any real training run: verify_sites.py
                          (correctness -- is the hook on the right tensor) and ceiling_sweep.py
                          (training-free causal headroom -- CAN any mask at this site/layer work).
                          See "methods/ndm/" section below.

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
    dla.py                 DIRECT LOGIT ATTRIBUTION: exactly how many logits each component
                          (embedding + every block's attention and MLP output) writes toward
                          the answer, from ONE forward pass, with no patching and no gradient --
                          same section below.
    mech_probe.py          runs all three probes above off ONE forward pass per row instead of
                          three -- same section below.
    ndm/swap_trace.py      PER-ROW trace of a full swap -- what the model says INSTEAD, with a
                          three-way HIT/base/other classification and a token-level log. The
                          companion to ceiling_sweep's rates; see its section below.
    head_trace.py          TRACES the image->text handoff: which attention heads write the
                          object's information into the text stream (differential head trace
                          under an image patch), and which of them are load-bearing (cumulative
                          knockout / path patching) -- same section below.
    logit_lens/, attention_maps/, dla/   small JSON/JSONL reports from the probes above (not gitignored
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
DBM (Appendix B.4) -- not re-derived here. `--lr` (default `1e-3`), by
contrast, is copied from DAS's `train.py` and is **not** validated for
DBM -- DAS learns a `D x D`/`D x K` orthogonal rotation, a very different
parametrization from DBM's unconstrained length-`H` mask vector, so
there's no reason to assume the same learning rate suits both. Worth
sweeping if training loss plateaus early relative to `--num_epochs` (see
"first real training run" notes below) -- all three of `--l1_coef`/
`--temperature_start`/`--temperature_end`/`--lr` are encoded in the
results/logs directory's `config_tag`, so different values never collide.
`--positions` takes the same named sets as everything else in this
project (`flag_only`/`flag_ring1` for flags, etc.) plus `last_token` --
the closest analogue of RAVEL's own *text-only* intervention site (the
entity mention's single last token); `flag_only`/`flag_ring1` are this
project's own extension into the image-object-span setting that
PCA/SAE/DAS already use.

**First real training run (flags/language/flag_ring1, layers 8/10/14/16/
20/22) -- a cautionary note on trusting the mask itself.** Real
`final_score`s ranged 47.1%-58.6% (best: layer 16), meaningfully beating
the existing SAE `flag_ring1` result (41.9%, see RESULTS.md) -- the
generation-time numbers are real and trustworthy. But every layer's
`mask_stats.json` showed the SAME suspicious pattern: mask magnitudes
maxing out around +/-0.02-0.03 (tiny), `sigmoid_mean` sitting near
0.5-0.58 (barely different from a coin flip) at every layer, ~54-58% of
dimensions "selected" everywhere -- looking less like a confident sparse
selection and more like near-arbitrary sign noise getting locked in once
temperature got small. The `layer{L}_train_log.jsonl`'s (properly
averaged) `loss` column confirms why: it drops fast for the first ~60-90
optimizer steps, then plateaus completely (bouncing flat, no further
improvement) for the remaining ~400 steps of a 480-step, 1-epoch run --
meaning temperature kept annealing all the way to `1e-7` long after the
mask had stopped learning anything new, likely locking in noise for
whatever dimensions hadn't been clearly decided by that early point. If
you hit the same pattern: try `--num_epochs 3` (or more) on a **fresh**
run (not a `--keep_checkpoint` resume) -- the temperature schedule is
anchored to that run's own total optimizer-step count from the start
(see `train.py`'s `temp_schedule_total_steps`), so more epochs means the
*same* 1e-2->1e-7 range gets spread over proportionally more steps,
giving the mask a longer smooth-gradient window before commitment is
forced. Also worth sweeping `--lr` (e.g. `5e-3`, `1e-2`) for the reason
above. Based on the actual loss composition observed (`l1_coef * l1_term`
contributing roughly 0.02-0.05 to a total loss of ~1-2, i.e. a small
fraction of it), lowering `--l1_coef` further looks like the *less*
promising lever to try first -- L1 doesn't appear to be the dominant
force keeping the mask near zero here.

**Results always land in a file, not just stdout.** Per layer, `eval.py`
(and `run_layer.py`/`layer_sweep.py`, which call the same code) write
`layer{L}_predictions_{split}.jsonl` plus, via VADE's own `eval/score.py`
(`score_file()`), a comprehensive `layer{L}_predictions_{split}_summary.
{json,md}` -- per-attribute `cause`/`iso`/`final_score` breakdown, identical
shape to PCA/SAE/DAS's own summary files. `layer_sweep.py` additionally
writes a sweep-level `sweep_layers<tag>_summary.{json,md}` aggregating
every swept layer's scores side by side plus the winning layer
(`best_layer`/`best_final_score`) -- this is the file to read/parse for
"which layer won", rather than re-reading `print_summary`'s console/log
text. All of these land under the same `results/<model_slug>/<entity>/
dbm/<attribute>/<config_tag>/` directory as the checkpoints/train logs.

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

## methods/ndm/ -- Native Dictionary Masking

NDM keeps DBM's trainer exactly (sigmoid-gated mask, temperature-annealed
toward binary, teacher-forced generation CE + `lambda * ||m||_1`, used for a
source->base interchange intervention) and moves the mask to a decoder
block's **MLP hidden state** -- the post-SwiGLU neuron vector
`act_fn(gate_proj(x)) * up_proj(x)`, width `intermediate_size` (18944 on
Qwen2.5-VL-7B-Instruct) -- instead of the residual stream (width
`hidden_size`, 3584).

### Why this is a different method, not a DBM hyperparameter

In RAVEL's own framing each method in this family is characterized by its
featurizer `F_A`:

| method | `F_A` | learned? |
|---|---|---|
| DBM | `F_A(n) = n` (identity) | -- |
| DAS / MDAS | orthogonal rotation `R` | yes |
| PCA / SAE (Phase B above) | fitted dictionary | yes, offline |
| **NDM** | **the model's own MLP encoder** | **no -- architecturally given** |

Two consequences:

1. **There is a decoder between the masked vector and the causal variable.**
   DBM's blend lands directly in the residual stream; NDM's passes through
   `down_proj` first. Since `down_proj` is linear,

   ```
   resid = base_resid + down_proj((1-s).h_base + s.h_source)
         = base_resid + (1-s).down_proj(h_base) + s.down_proj(h_source)
   ```

   so an axis-aligned binary mask in neuron space induces a
   **non**-axis-aligned intervention in residual space -- a linear image of
   a binary mask. That is not DBM's hypothesis class; it is closer to DAS,
   with the "rotation" fixed by the architecture rather than learned (and a
   18944->3584 projection rather than a square rotation).

2. **The encoder is nonlinear**, the only nonlinear featurizer in the table
   -- and the reason the space has a *privileged basis*: an elementwise
   nonlinearity plus an elementwise gating product mean the only
   function-preserving transformations of that space are permutations, so
   its coordinates ("neurons") are real objects rather than a choice of
   axes. The residual stream has no such property: rotate it, fix up every
   read/write matrix, and the model computes an identical function -- so
   DBM's selected dimensions are a fact about one checkpoint's arbitrary
   coordinate system, not about the model. See `methods/common/sites.py`'s
   module docstring, and Elhage et al.'s *Toy Models of Superposition* /
   *Privileged Bases in the Transformer Residual Stream*.

So NDM is "SAE-style encode -> mask -> decode patching, where the dictionary
is the model's own MLP."

### The two axes: `--site` and `--positions`

Every intervention run picks one of each. They are independent:

- **`--site`** — *which tensor* inside a decoder block is read and patched.
- **`--positions`** — *which token columns* that tensor is read and patched at.

A dead result means one of the two is wrong, and only varying them separately
tells you which.

#### Sites (`--site`, `--sites`)

Five single sites plus one joint site, all addressing the **same block**:
`--layer L` means decoder block `L-1`, so the same `--layer` is comparable
across every site.

```
  resid[L-1] ──────────────────────────────────────────────────┐
      │                                                        │
      ▼                                                        │
  self_attn ──> [attn_head_output] ──o_proj──> [attn_output]    │
                 o_proj's INPUT                     │          │
                 28 heads x 128 = 3584              ▼          │
                                                   (+) <───────┘   residual add
                                                    │
                                                    ▼
                                                    h ─────────┐
      ┌─────────────────────────────────────────────┘          │
      ▼                                                        │
     mlp ──────> [mlp_hidden] ────down_proj────> [mlp_output]   │
                 down_proj's INPUT                  │          │
                 intermediate_size = 18944          ▼          │
                                                   (+) <───────┘   residual add
                                                    │
                                                    ▼
                                        [residual]  =  resid[L]
```

The two sublayers run in **sequence**, not in parallel: `h = x + attn(x)`,
then `out = h + mlp(h)`. That is why `residual` at layer L is exactly the
embedding plus every `attn_output` and `mlp_output` before it -- an identity
worth remembering, because it means a nonzero `residual` ceiling has to be
accounted for by *something*, and if every sublayer reads zero then the
content is in the embedding.

| `--site` | hooked module | pre/post | width | what it swaps |
|---|---|---|---|---|
| `residual` | decoder block `L-1` | post | 3584 | everything accumulated through the block. `--layer 0` is the embedding output |
| `attn_output` | `self_attn` | post | 3584 | attention's contribution only |
| `attn_head_output` | `o_proj` | **pre** | 3584 | per-head `z`: 28 heads x 128, so masks can select whole heads |
| `mlp_output` | `mlp` | post | 3584 | that MLP's contribution only |
| `mlp_hidden` | `down_proj` | **pre** | **18944** | post-SwiGLU neurons -- NDM's own site |
| `attn_output+mlp_output` | both sublayers | post | 7168 (sum) | **joint**: the block's whole contribution. Diagnostic only |
| `blocks:N` | both sublayers × N blocks | post | 7168·N | **joint span**: N consecutive blocks ending at `L`. `blocks:1` is an alias for the row above |

The default `--sites` for `ceiling_sweep.py` is the **four independent**
ones, `residual attn_output mlp_output attn_output+mlp_output`, so the three
comparisons fall out of one run: global-vs-local (`residual` vs the two
single sublayers, all at width 3584), which-sublayer (`attn_output` vs
`mlp_output`), and sublayers-vs-prefix (the joint site -- next subsection).

`attn_head_output` and `mlp_hidden` are **omitted from the default on
purpose**: under a full swap they are mathematically identical to
`attn_output` and `mlp_output` (see the third bullet below), so probing all
six spent a third of the sweep re-deriving two columns you can copy.
Crucially this does not leave NDM's own training site unmeasured -- **the
`mlp_output` row *is* the `mlp_hidden` ceiling**, at every layer, and the run
prints that mapping under the summary table so an absent row is never
mistaken for an unprobed one. Pass them explicitly when you want the identity
spot-checked: agreement to the row is a real canary for a mis-hooked module
or nondeterministic generation, and it is the only check `attn_head_output`
has ever had.

Three things that bite:

- `--layer 0` is `residual`-only. There is no sublayer before the first block.
- `self_attn` returns a **tuple**, `mlp` returns a bare tensor. The hooks in
  `common/sites.py` patch element 0 and pass the rest through, so both work,
  but a hand-rolled hook on `self_attn` that assumes a tensor will break.
- **`ceiling_sweep.py` cannot distinguish a pre/post pair.**
  `down_proj(h_source)` is exactly `mlp_out_source`, so under a FULL swap
  `mlp_hidden` == `mlp_output` and `attn_head_output` == `attn_output` --
  measured, identical in every cell of four full sweeps. They share one
  ceiling, which is why the default omits the pre-projection half of each
  pair (`common/sites.py`'s `FULL_SWAP_EQUIVALENT` holds the mapping). A
  privileged basis only buys anything for a SPARSE mask -- a subset of
  neurons or heads decodes to a residual update no axis-aligned residual mask
  can express -- so that question can only be settled by training one, or by
  a partial (top-k / per-head) swap.

Only `mlp_hidden` and `mlp_output` are NDM **training** sites (`NDM_SITES`,
default `mlp_hidden`); the other three single sites exist for the
diagnostics. `methods/dbm/train.py`'s engine accepts any of them, so enabling
training at another single site is a config change, not engine work. The
joint site is **not** trainable at all -- a mask there would need one mask
and one L1 term per part, which is a different method, so `JointSite`
raises on the training entry points and `ndm/config.py`'s `--site` choices
never offer it.

#### The joint site: `attn_output+mlp_output`

The three same-block sites are **not additive**, and reading them as if they
were manufactures a fake paradox. The identity is real:

```
residual@L  =  residual@L-1  +  attn_output@L  +  mlp_output@L
```

so a `residual` curve that climbs `0% -> 9.4% -> 68.8% -> 100%` across layers
21-24 looks like it has to be *caused* by those blocks' sublayers -- which
measure `0.0%` and `0.0%`. Both numbers are correct, because the two
interventions do different things:

- a **sublayer** swap only *inserts* source evidence. The entire accumulated
  base prefix `resid_base@L-1` survives untouched.
- a **residual** swap also *deletes* the base prefix.

For an attribute that is encoded redundantly across depth, the deletion is
the operative half, and no single-sublayer swap deletes anything.

The joint site closes that gap. Patching both sublayers of one block at once
yields `resid_base@L-1 + attn_src@L + mlp_src@L`, which differs from a
`residual@L` swap in **exactly one term**, so the comparison is a clean
subtraction:

| result | reading |
|---|---|
| joint ≈ `residual@L` | the block's own sublayers do the work; the accumulated prefix is irrelevant |
| joint ≈ 0 | neither half suffices alone -- the prefix is *necessary*. The attribute is encoded conjunctively/redundantly across depth, and the `residual` curve is measuring how much depth remains to **repair** the edit, not when information arrived |

Nothing else in the site list separates those two.

#### Block spans: `blocks:N`

`blocks:N` widens that arm from one block to **N consecutive blocks ending at
`--layer L`**, which is exactly what `residual@L` has and `residual@(L-N)`
does not:

```
residual@L  =  residual@(L-N)  +  Σ (attn_output@i + mlp_output@i)   i = L-N+1 … L
                    ^ kept as base          ^ all swapped to source
```

So the retained prefix moves earlier as `N` grows, and `blocks:N` converges on
`residual@L`. That turns the joint site's yes/no into a dial: sweeping
`blocks:1 … blocks:5` at a layer where `residual` is live measures **how many
consecutive blocks must be swapped before the prefix stops mattering** -- how
deep the redundancy goes.

```bash
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers 24 --positions last_token \
    --sites residual blocks:1 blocks:2 blocks:3 blocks:4 blocks:5
```

Reading it: a curve that climbs from `blocks:1 ≈ 0` to `blocks:5 ≈ residual@L`
localizes the computation to a ~5-block window *without ever deleting the
prefix*. A curve that stays flat at 0 all the way to `blocks:5` says the
prefix is necessary no matter how wide the window -- the attribute is not
recomputed within any local span.

Mechanics and constraints:

- `N` is **parsed, not enumerated**, so any `N >= 1` works (the same way
  `position_sets.py` parses `ring:K`). `--sites` therefore validates with a
  `type=` function rather than a `choices=` list.
- Needs `--layer >= N`; the earliest block has to exist. Shallower layers are
  skipped with a message, and `JointSite` asserts rather than wrapping to a
  negative block.
- Each part gets its own source capture, its own hard mask at its own width,
  and its own patch hook, but **all `2N` of them are registered before a
  single source forward and a single generate** -- a five-block span costs the
  same number of model calls as a one-block one.
- Hook order is forward order: within a block the attention post-hook fires
  before the MLP post-hook, and across a span block `L-2`'s patched output is
  what block `L-1` reads. Irrelevant under a *full* swap (each output is
  overwritten wholesale regardless of what it computed), but it would matter
  for a partial one.

#### Position sets (`--positions`, `--positions_list`)

VADE's own `entities.py` knows four; `methods/common/position_sets.py` adds
the rest as a **wrapper** (it post-processes a batch `build_batch` already
produced, so `entities.py` stays a byte-identical copy of VADE's).

**Built in to VADE** — counts shown for `flags` (12x12 = 144 image tokens):

| spec | n | what |
|---|---|---|
| `flag_only` | 8 | the flag's own tokens (per `object_location.json`) |
| `flag_ring1` | 24 | `flag_only` dilated by one cell |
| `full_image` | 144 | every image token |
| `last_token` | 1 | the final prompt token, RAVEL's own site |

**Image grid** — all derived from the object's token bbox, so they work for
any entity (`logo_only` for brands, etc.), not just flags:

| spec | n (flags) | what |
|---|---|---|
| `~<name>` | 120 for `~flag_ring1` | complement within the image span |
| `ring:K[@base]` | 0:8, 1:16, 2:24, 3:32, 4:40, 5:24 | K-th Chebyshev band around the bbox. `ring:0` == `flag_only`; `ring:0`+`ring:1` == `flag_ring1`; rings 0-5 sum to 144 |
| `side:left` / `side:right` | 8 / 8 | flanking the object on its **own rows** |
| `side:beside` | 16 | `left` + `right` |
| `side:above` / `side:below` | 60 / 60 | full-width bands |
| `seq:before` / `seq:after` | 64 / 64 | background split by **raster = sequence = causal** order |
| `seq:between` | 8 | same rows as the object, outside its columns |

**Text**:

| spec | n | what |
|---|---|---|
| `tok:-K[:N]` | N (default 1) | N tokens ending K back from the prompt end. `tok:-1` == `last_token` |
| `pre_image[:N]` | N | tokens immediately **before** the image span |
| `vision_end[:N]` | N | tokens immediately **after** the image span |
| `phrase:attribute` | 1 | the token naming the **queried** attribute in the question |
| `phrase:<lit>[,<alt>...]` | 1 | the token ending an arbitrary literal, alternatives tried in order |

**Composition**:

- `A+B+C` — union of any specs, including mixing image and text. Columns are
  concatenated, deduped and sorted. `flag_ring1+~flag_ring1` reconstructs
  `full_image` exactly, which is a free self-check of the machinery.
- `--positions_list a b c` — probe several specs in **one model load**. The
  model load (~2 min) dominates a ceiling run (~10s of forwards per spec), so
  a dozen separate invocations spend most of their wall clock loading. Each
  spec still gets its own JSON in its own directory, and the row sample
  (`--seed`) is shared, so specs are always compared on identical rows.

#### `pre_image` is a guaranteed NULL; `seq:before` is NOT

Every image in a VADE entity is **one fixed canvas render** with only the
object's pixels varying (`object_location.json` states this explicitly).
`pre_image` tokens precede the image entirely, so under the language model's
causal attention they are bit-identical in base and source at every layer:
patching them is provably a no-op and MUST score 0%.

**`seq:before` was designed as a second null and measured as one that fails
(E11, 2026-10-16: 15-24% cause on flags/brands/celebrities).** The argument
("background tokens before the object see identical pixels and identical
causal context") ignores the vision encoder: Qwen2.5-VL's ViT has
full-attention blocks (`fullatt_block_indexes` in its config) that are
bidirectional over the whole image, so every background token already carries
some object information when it enters the language model. Read `seq:before`
and `seq:after` as "how much of the entity sits in that background region",
not as a null and an LLM-only leakage measure respectively. A real null for
image-token sets needs a layer-0 identity check (base vs source activations at
those positions), not this spec.

#### `phrase:` mechanics

`phrase:` reuses `methods/probe_common.py`'s `ATTRIBUTE_KEYWORDS` and the
adapter's `find_last_phrase_token_col` -- the same machinery the read-only
probes use, with its already-hardened synonym lists (validated 24/24 across
every flags template, including `capital_prefill_v6`, which says "seat of
government" and contains no "capital" at all).

- It matches the **question only, never the prefill**. Prefill words are
  reachable with `tok:-K` instead: flags' prefills are 4-5 tokens, so `tok:-2`
  lands on "language" in "One official language is".
- Literal alternatives matter because wording varies across templates:
  `phrase:image` matches only 4 of 6 language templates, `phrase:image,flag`
  matches 6/6.
- A phrase that matches no template asserts loudly with the offending
  question, rather than silently selecting the wrong token.

#### Gotchas that apply to every position set

1. **Fixed count per row.** `batch["positions"]` is a rectangular `[B, n_pos]`
   tensor, so a set must yield the same *number* of positions for every row.
   Fixed windows and grid subsets are fine; "all question tokens" is not,
   because the six templates have different question lengths.
2. **Columns may still vary per row.** Rows built from a shorter template get
   more left padding, so the image span sits at a different absolute column.
   `position_sets.py` resolves image- and phrase-anchored specs per row.
   Only `tok:` is row-invariant (left-padding puts every row's last real
   token in the same column).
3. **Template variation blurs deep `tok:` offsets.** `tok:-1..-5` is the
   prefill in every template and is comparable as-is; beyond that the offsets
   land on different words. Use `--template_id` to pin one template.
4. **Text sets disable the source cache.** They inherit `is_last_token=True`,
   which is correct: their columns depend on the prompt, so the per-entity
   cache (keyed by image alone) is invalid for them. `ceiling_sweep.py` never
   caches, so this only matters for training runs.

#### Worked examples

```bash
# Where does the attribute live in the image, by distance from the flag?
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers $(seq 0 28) --sites residual --n_rows 32 --seed 0 \
    --positions_list ring:1 ring:2 ring:3 ring:4 ring:5

# Does it leak into the background -- and are the controls really null?
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers $(seq 0 28) --sites residual --n_rows 32 --seed 0 \
    --positions_list seq:before pre_image seq:after ~flag_ring1

# Trace the handoff into the text stream, per token
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers $(seq 18 28) --sites residual --n_rows 32 --seed 0 \
    --template_id language_prefill_v1 \
    --positions_list phrase:attribute vision_end tok:-4 tok:-3 tok:-2 tok:-1

# Union: is the flag plus its background more than the flag alone?
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers $(seq 0 28) --sites residual --n_rows 32 --seed 0 \
    --positions_list flag_ring1 ~flag_ring1 flag_ring1+~flag_ring1

# Train at a position set once a ceiling shows headroom there
python methods/dbm/run_layer.py --entity flags --attribute language \
    --layer 23 --positions last_token --temperature_start 1e-2 --temperature_end 1e-2
```

### Usage

```bash
# STEP 1 -- correctness. One model load, ~a GPU-minute, and it catches a
# hook attached to the wrong tensor (which otherwise trains fine and
# produces plausible numbers). Covers `residual` too, so it doubles as a
# regression check that DBM still behaves identically. Exits nonzero on
# failure, so you can gate a run on it.
python methods/ndm/verify_sites.py --entity flags --attribute language --layer 16

# STEP 2 -- CAUSAL HEADROOM, before spending GPU on training. Swaps in 100%
# of the source at each (site, layer) and measures how far the answer moves.
# A mask selects a SUBSET of what a full swap uses, so this is a hard upper
# bound on `cause` for any amount of training there. Costs a couple of
# forward passes per layer instead of a training run.
# the four independent sites (the default -- no --sites needed)
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers 2 6 10 14 18 22 26

# just the sublayers-vs-prefix arm, where a residual curve turns over
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers 21 22 23 24 --positions last_token \
    --sites residual attn_output mlp_output attn_output+mlp_output

python methods/ndm/train.py --entity flags --attribute language --layer 16 \
    --positions flag_ring1 --site mlp_hidden
python methods/ndm/eval.py  --entity flags --attribute language --layer 16 \
    --positions flag_ring1 --site mlp_hidden --split test

# train+eval+score one layer, or sweep several in one model load:
python methods/ndm/run_layer.py --entity flags --attribute language --layer 16 \
    --positions flag_ring1 --site mlp_hidden
python methods/ndm/layer_sweep.py --entity flags --attribute language \
    --layers 8 10 14 16 20 22 --positions flag_ring1 --site mlp_hidden
```

`verify_sites.py` runs four checks per site: the captured width is right;
a mask driven to `sigma=0` (pure base) reproduces an unhooked generation
token-for-token; a mask driven to `sigma=1` with the **base's own** activation
as the "source" also reproduces it (this exercises the other branch and
needs no independently computed ground truth -- patching a value in over
itself must be a no-op); and `make_cache_aware_patch_hook` passes a
`[B, 1, width]` decode-step tensor through untouched.

### Hyperparameters -- `--l1_coef` does NOT transfer from DBM

`l1_penalty` is a plain `mask.abs().sum()`, so at `intermediate_size=18944`
the sparsity term is ~5.3x larger than at 3584 for the same per-dimension
mask magnitude -- and RAVEL's reported optimum (`0.001`) was tuned on a
~4096-wide residual stream. The default is deliberately left at `0.001`
rather than silently renormalized (that would deviate from the paper's
formula), but it is **not calibrated for `mlp_hidden`**. Sweep it, e.g.
`0.0002 / 0.001 / 0.005 / 0.02`, and read `layer{L}_mask_stats.json`'s
`n_selected` alongside `final_score`: too low and the mask selects nearly
every neuron (no sparsity, and `iso` suffers); too high and it selects
almost none (`cause` collapses). The config tag encodes `l1_coef`, so
parallel sweeps land in separate directories and cannot clobber each other.

### Where artifacts land

`results/<model_slug>/<entity>/ndm/<attribute>/<config_tag>/` and
`logs/<...>/ndm/<...>/` -- a separate tree from `dbm/`'s, so NDM gets its own
summary files and its own row in comparison tables, and the existing
committed DBM results stay exactly where they are. The config tag reads
`L1_<l1>_T<start>-<end>_LR<lr>_<positions>_<site>[_pruned]`; the **site is
encoded**, because `mlp_hidden` and `mlp_output` runs are different artifacts
that would otherwise collide in one directory (this repo has been bitten by
that class of bug twice already -- `select_features.py`'s
`--dictionaries_dir`, then `dbm_config_tag` omitting the temperature
schedule).

Per-layer and sweep-level summaries are written exactly as for DBM (via
VADE's own `eval/score.py`), with `site` recorded in the sweep summary.

### Source-activation cache

NDM has its **own** cache (`methods/common/site_source_cache.py`), on by
default, disabled with `--no_source_cache`. It is a different cache from the
one DBM/DAS share: `common/source_cache.py` stores
`torch.stack(out.hidden_states)`, i.e. the residual stream at every layer,
and MLP internals appear nowhere in `output_hidden_states`. Both live under
`--vade_root/results/` since both are keyed by entity+model rather than by
method.

It is sound for the same reason the residual cache is: under causal
attention a block's MLP at position `p` reads only the residual stream at
`p`, which depends only on tokens `<= p`, and the image precedes the
question in the chat template -- so an image token's MLP hidden state is a
function of the source image alone. It is keyed by
**(site, positions, layer)**, one file each (~76MB for flags/flag_ring1/
mlp_hidden per layer), because an MLP site needs a capture hook at one
specific module and so cannot get every layer from a single pass the way the
residual cache does. Changing `--positions` rebuilds it; changing `--layer`
builds another file. `positions=last_token` is not cacheable at any site
(that position depends on each row's own prompt length) and falls back to a
live forward pass. A cache built for the wrong site/layer/positions raises
rather than mis-patching -- necessary because an `mlp_output` cache and a
residual cache have identical shapes.

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

### 2b. `dla.py` -- direct logit attribution

```bash
python methods/dla.py --entity flags --attribute language
python methods/dla.py --entity flags --attribute language --direction base_minus_source
```

"Which components help make the identification right", answered **exactly**,
from one forward pass. The residual stream is a sum

```
resid_final = embed + Σ_b (attn_output[b] + mlp_output[b])
```

and the head is a norm plus a bias-free linear map. The norm's only
nonlinearity is a per-position **scalar**; freeze it at the value the real
forward pass computed (from the *full* final residual) and the rest is
linear, so the answer's logit splits into one exact term per component:

```
logit(t) = Σ_c  scale · ( component_c · (W_U[t] ⊙ final_norm.weight) )
```

No patching, no counterfactual, no gradient, no approximation -- this is the
model's own logit, decomposed. Two adapter primitives carry the model-specific
half (`final_norm_scale`, `logit_direction`), and the sublayer outputs come
from `common/sites.py`'s own capture hook, i.e. the identical tensors the
interventions read and patch.

**`--direction` decides what "the answer's logit" means**, and it changes what
the ranking means:

| `--direction` | decomposes | use when |
|---|---|---|
| `gold_minus_mean` (default) | gold logit − mean logit over the vocabulary | general "what writes the answer"; a component that lifts every logit equally gets no credit |
| `gold` | the raw gold logit | simplest; rewards components that just push the whole distribution up |
| `base_minus_source` | base gold − **source** gold | the VADE-specific one, usually the most on-point: it is the exact axis `cause` moves along |

**The self-checks are the point.** Every failure mode here (a hook on the
wrong tensor, a double-applied final norm, an off-by-one over blocks, a head
with a bias) yields plausible numbers rather than an error, so two things are
verified per row and asserted, not assumed:

- **reconstruction** -- `embed` plus every captured sublayer, against the
  model's own `hidden_states[n_layers-1]`. Isolates hook correctness from the
  algebra. (`hidden_states[-1]` is deliberately unused: in this project's
  transformers it is tied to `last_hidden_state` and is therefore
  *post*-final-norm -- the same trap `logit_lens.py` documents.)
- **additivity** -- the per-component terms summed, against the real logit
  read off `out.logits`. End to end.

Both are reported as relative errors; `--tolerance` (default 2%) sets the bar.
bf16 accumulation over ~57 components puts the honest floor near 1%, so a
failure well above that is a wrong tensor, not rounding.

**What it does and does not tell you.** DLA measures **direct** paths to the
logit only: a component that matters by changing a later head's attention
pattern is credited to that head, not to itself. A zero here means "writes
nothing along the answer direction", *not* "causally irrelevant" -- that is
what `ndm/ceiling_sweep.py` answers, and the two disagreeing is informative.
It is also, like every per-component attribution, a **marginal** measure: one
number per component, so it cannot represent a conjunction. Where an attribute
is encoded redundantly across depth (the signature being a live `residual`
ceiling at a layer whose individual sublayers all read 0), expect the DLA mass
spread thin across many blocks rather than concentrated -- and read the
spreading as the finding, not as a null.

### 3. Running all three at once (recommended if you want more than one)

```bash
python methods/mech_probe.py --entity flags --attribute language
python methods/mech_probe.py --entity flags --attribute language --dla_direction base_minus_source
```

Runs the exact same three scorers (`logit_lens.score_one_row`/
`attention_maps.score_one_row`/`dla.score_one_row`) off **one** forward pass
per row (`output_hidden_states=True` AND `output_attentions=True` together,
with DLA's sublayer capture hooks live during it) instead of three, and writes
the *identical* output files the standalone scripts would (tagged
`"single_forward_pass": true`). `--skip_dla` drops the third; its only cost is
one hook per sublayer and no extra forward. Defaults to every
decoder layer for both probes (matching the "all-layer" runs this project
has already done by hand) -- pass a smaller `--layers` to cut attention's
memory cost back down if a full sweep isn't needed; logit lens always also
scores the final "real output" pseudo-layer on top of whatever `--layers`
you pass, since it's free. Same `--dump_raw_rows`/`--top_k`/`--threshold`/
etc. flags as the two standalone scripts (see `--help`).

### 3b. `head_trace.py` -- which heads carry image -> text

```bash
python methods/head_trace.py --entity flags --attribute language \
    --patch_layer 21 --positions flag_ring1
```

`ceiling_sweep.py`'s two position sets *bracket* the image->text handoff
without locating it. On flags/language a full residual swap reads:

| `--layer` | ≤21 | 22 | 23 | 24+ |
|---|---|---|---|---|
| image positions | **100%** | 100% | 0% | 0% |
| `last_token` | 0% | 9.4% | 68.8% | **100%** |

Patching the image stops mattering at exactly the layer patching the last
token starts mattering. That crossover **is** the read: before it, editing the
image still propagates; after it, editing the image is too late and editing
the destination is decisive. So the read is in blocks ~21-23 -- which follows
from the table alone, with no new experiment.

From there the search space is tiny, because **the only operation in a
transformer that moves information between positions is attention** (MLPs are
position-wise; the residual stream never mixes tokens). Every bit of transfer
is `Δresid[last] += Σ_h Σ_{j∈image} a[h,last,j]·(v[j]·W_O^h)` for some
(block, head) -- a few dozen candidates.

**Why this intervenes on the image side.** Swapping head *h*'s dims *at the
last token* is a subset of swapping the whole `attn_output` there, which
`ceiling_sweep` measured at 0.0% everywhere except 9.4% at layer 24. Anything
bounded by that is dead on arrival. The leverage is on the source side, where
an image swap at `--patch_layer` already produces a ~100% effect.

**Phase 1 -- differential head trace** (2 forwards per batch, exact, no
generation). Capture every block's per-head attention output at the last
token, once clean and once with the image residual patched to the source. A
head whose output changed is a head that *read* the patched image:

| column | what it is |
|---|---|
| `delta_z` | `‖Δz_h‖` -- the raw per-head output change |
| `delta_resid` | `‖Δz_h W_O_h^T‖` -- the change that actually lands in the residual stream. **The default ranking**: `o_proj` weights heads very differently, so a large `Δz` can land as nothing |
| `delta_dla` | `dla.py`'s exact per-head logit term, differenced between the runs (each with its own frozen scale). Ranks by movement of the *answer*, but sees direct paths only |

**Phase 2 -- cumulative knockout / NECESSITY** (path patching; one generation
per `k`). Keep the image patched (cause ~100%) and **restore** the top-*k*
heads' outputs at the last token to their clean-base values. If cause
collapses, those heads carry the signal. Cumulative (`k = 1, 2, 4, 8, 16`)
rather than one head at a time, because single-head knockout will read ~0 for
exactly the reason single-sublayer swaps do -- a marginal measurement cannot
see a conjunction. This is the head-level analogue of `ceiling_sweep`'s
`blocks:N`.

**Phase 3 -- SUFFICIENCY**, the mirror, and the stronger of the two. Do *not*
patch the image; instead **install** the patched values into only the top-*k*
heads, and ask whether that alone reproduces the effect. Necessity is precisely
the question redundancy breaks -- a genuinely load-bearing head looks
unnecessary when three others can cover for it -- whereas a small set that
reproduces the effect on its own is localization no matter how flat the
knockout curve is.

Phase 3 reports **its own ceiling** (all traced heads patched), which is *not*
100%: the image patch also moves blocks outside `--blocks`, the MLPs, and the
residual stream directly, so the top-*k* arms are normalized against what this
intervention can actually reach. Its `k=0` arm must reproduce the unhooked row
exactly, and the run says so if it doesn't. Both phases go through one
`head_patches` builder and one `run_cause(..., with_image_patch=)`, so the
curves cannot drift apart in rows, masks, scoring or generation, and
`--n_random`'s null set is drawn once and shared. `--skip_sufficiency` drops
phase 3 for the generation budget; `--skip_knockout` drops both.

**A null control is built in.** `--n_random` knocks out the same number of
*randomly chosen* heads. If random-8 hurts cause as much as top-8, the phase-1
ranking is not carrying information and the phase-2 curve says nothing. Read
the two together or not at all.

Gotchas: `--patch_layer` must be **below** the handoff (a layer where the
image-position column of `ceiling_sweep` is still large) -- patching where the
effect is already 0 leaves nothing downstream to trace, and the script says so
rather than reporting a flat curve. Blocks before `--patch_layer` are excluded
by default since they cannot be affected by it.

### 3c. `ndm/swap_trace.py` -- what the model says *instead*

```bash
python methods/ndm/swap_trace.py --entity flags --attribute calling_code \
    --positions last_token --site residual --layers 19 20 21 22 23 24 25 26 27 28
```

Same intervention as `ceiling_sweep.py`, different reporting -- and the
reporting is the point. Every (row, layer) is classified three ways using the
trainer's own `exact_match`:

| label | meaning |
|---|---|
| `[HIT]` | generation == the **source's** gold -- the swap **transferred** |
| `[base]` | generation == the **base's** gold -- the swap did **nothing** |
| `[other]` | neither -- the swap **destroyed** the answer |

**Why `other` matters.** flags/`calling_code` at `last_token`, residual, layer
24 reads `cause=3.4% base_kept=14.3%`, and the sweep's own verdict line called
it dead. It is the opposite of dead -- the missing **82.3%** is a *third*
country's dialing code:

```
row 383: base=GY (gold='592') -> source=ET (gold='251')
  unhooked: '592. This corresponds to the'
  L22:      '592. This corresponds to the'   [base]
  L23:      '246. This corresponds to the'   [other]
```

Compare flags/`language` at the **same** site, position and layer:
`cause=100%, base_kept=0%, other=0%`. Same instrument, same intervention,
opposite outcome. Two rates cannot tell "inert" from "destructive"; only the
generated text can. `ceiling_sweep` now derives and prints `other` as well,
and its no-headroom verdict points here when `other` exceeds 25%.

That split is **unexplained**, and it is the reason not to treat
flags/`language` as representative: the last-token residual at L23+ is not
simply "the answer". For a retrieved one-word answer a swap copies it; for a
composed multi-digit one a swap corrupts it.

Two logs are written. The main one shows decoded text; the `_TOKENS` one shows
each answer's actual token sequence, because Qwen2.5-VL splits digits one per
token (`'250'` -> `['2','5','0']`) and a half-right numeric answer is invisible
in decoded text. `--n_rows` defaults to **384**, not `ceiling_sweep`'s 32 --
at 32 rows a single row is 3.1%, which is the entire dynamic range of the
effects this script exists to look at.

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

## Circuit experiment results (Tier 1 + Tier 2)

Experiments E1-E11 test one account of how Qwen2.5-VL-7B answers "what is the
capital/currency/... of the country whose flag is shown":

1. **See**: the entity is encoded at (and around) the object's image tokens.
2. **Move**: about 10 question-blind *mover* heads in blocks 21-23 (`common10` =
   21.1 21.5 22.13 22.15 22.17 22.19 23.3 23.4 23.6 23.17) copy it to the last
   token, while *router* heads in blocks 19-21 (`router8`) carry which attribute
   was asked.
3. **Recall**: last-token MLPs in blocks 23-26 (22-27 for brands/celebrities)
   look up the attribute value and write the answer.

Launchers: `scripts/run_tier1.sh` (E1-E6) and `scripts/run_tier2.sh` (E7-E11).
Pod instructions: [GPU_RUNBOOK.md](GPU_RUNBOOK.md). Results were committed in
607b957 (E1, E2, E6), be66844 (E3, E4, E5) and c1f773b (E7-E11). Logs are in
`logs/tier1/` and `logs/tier2/`; outputs are under `results/`.

**Bottom line** (reviewed 2026-10-16 against the result files; every number below
was re-checked, and the DBM scores were recomputed with VADE's `score.py`):

- **Strongly supported:**
  - The `common10` movers carry the entity to the answer position (E4, E6, E9).
  - They are shared across flags, brands and celebrities (E10).
  - InternVL3-8B, which is the Qwen2.5-7B language model behind a different vision
    encoder, independently recovers exactly `common10` (E8).
  - Movers and routers compose into an answer neither donor run gave (E4).
- **Supported for flags, partly elsewhere:**
  - MLPs 23-26 do the lookup on flags (E2, E3, E6).
  - On brands/celebrities they write the answer, but are only partly necessary (E7).
  - The MLP stage was not tested on Qwen2.5-VL-3B or InternVL3-8B. "Holds across
    models" applies to the handoff and the movers only.
- **Ruled out:** "the image route is different from (or later than) the text route."
  With the country NAMED in text, the handoff layers, the mover heads and the MLP
  lookup are the same (E6), and text accuracy is only 0-8 points higher. The VLM
  reuses its language model's entity->fact circuit; what is image-specific is the
  encoding stage and where the movers read from.
- **Qualifications:**
  - Identity is never confined to the object's tokens (E11).
  - The movers are a bottleneck as HEADS, but they do not need the image tokens
    themselves: with the image cut off, they read the entity from the question
    tokens (E5).
  - The lookup is concentrated in MLP 25 for flags/calling_code but spread across
    blocks elsewhere.

Rates below are on the rows where both clean runs are correct. 95% Wilson
intervals are given for headline numbers; they treat rows as independent, so they
are a lower bound on the real uncertainty.

### Tier 1: the circuit on flags (Qwen2.5-VL-7B)

**E1, are attributes already present at the image tokens?**
(`methods/image_enrichment.py`: logit lens and leave-one-country-out probes at
the object's tokens, read against layer 0 and a permutation null.)

- Image-token activations are identical whichever question is asked (E0b
  difference = 0, batched and at batch size 1), so the image encoding is
  question-independent.
- **Probes** (leave-one-country-out accuracy, compared with the same probe at layer 0):
  - **Language:** no enrichment above the layer-0 (visual similarity) level
    through the handoff. Image-token probes stay at 0.59-0.73 against 0.63-0.71
    at layer 0, and only rise at L26-27 (to 0.76).
  - **Region-level enrichment, mid-network:** EUR vs not on background tokens
    goes 0.67 → 0.77-0.83 over L8-14. Calling zone over all image tokens goes
    0.67 → 0.75-0.79 over L8-10. The often-quoted "+0.23" (ring, L10) is a
    single-layer spike; neighbouring layers sit near 0.69.
  - The p-values (≤0.02, 50 permutations) are taken at each curve's best layer,
    without correcting for choosing that layer.
- **Vocabulary projection** (logit lens):
  - **Before L22 it reads nothing at any position**, not even the queried value
    at the last token (capital top-1 1% through L20). It therefore cannot test
    early enrichment.
  - **At L24-27 the image tokens DO decode the facts:** name top-1 97%, capital
    53%, language 50% at L26 (all tokens, best token). This is after the
    handoff (image-position patches are dead from L23, see E6/E11), so this copy
    does not reach the answer.
  - MLPs are position-wise: blocks 23-26 apparently look facts up wherever the
    entity is represented, and only the answer position's copy is used.
- **Reading:** through the handoff, the image tokens carry the entity plus some
  region-level semantics, not its specific facts. Facts are attached in blocks
  23-26, at every position holding the entity, and only the answer position's
  copy is used. This is not "the image tokens never hold facts".

**E2, severed MLP freezing.** (`methods/head_severed.py`: install the source
flag through `common10`, then freeze later last-token components at base, @mean
or @other values.)

| | capital | currency | language | calling_code |
|---|---|---|---|---|
| movers alone install the source answer | 97% | 78% | 100% | 98% |
| + MLP 23-26 frozen at base | 0% | 0% | 0% | 0% |
| + MLP 23-26 frozen at a third country (third country's answer) | 95% | 62% | 95% | 100% |

- Freezing at the mean of other countries collapses the answer (0-22% source,
  mostly neither answer), so the blocks are needed, not just re-asserting the
  base.
- MLP 25 alone writes calling_code (third country's code on 100% of rows).
- For language, every single-block freeze at @mean recovers to 89-98%. Its
  lookup is spread over 23-26 rather than sitting in one block.
- **DLA** (`dla.py --direction gold_minus_mean`, 144 rows per attribute):
  - MLPs 23-27 write most of the answer logit directly; attention contributes
    little.
  - **MLP 27 is the largest single direct writer for every attribute** (about
    40%: capital 9.7 of 24.9).
  - Yet freezing MLP 27 at base leaves transfer largely intact (R16.2). The
    accurate statement is "23-26 decide, 27 amplifies".
- **Reading:** the movers deliver an entity, and MLPs 23-26 turn it into the
  answer. Freezing them at another country's values makes the model answer for
  that country.

**E3, is the lookup attribute-specific at the neuron level?**
(`methods/readout_jacobian.py --site mlp_hidden --blocks 23-26`, then
`head_das.py --method dbm` masks.)

- **How the edit works:** for each attribute T, take the top singular directions
  of the Jacobian of T's first-answer-token logits with respect to the 18,944
  MLP neurons, at ≤128 directions per block. Fit on 59 train countries and
  evaluate on 25 disjoint test countries (~1,434 rows per target).
- **What is copied:** at blocks 23-26, every answer step, only the
  source-minus-base difference inside that subspace
  (`h_new = h_base + P_T (h_source - h_base)`).
- **Isolated variant:** `isolated` also removes the directions the other three
  attributes read.

VADE scores (cause / mean iso / final):

| target | clean | full swap | readout | isolated | random (same rank) |
|---|---|---|---|---|---|
| capital | 0.0 / 93.8 / 46.9 | 96.9 / 0.8 / 48.8 | **91.9 / 76.0 / 84.0** | 62.5 / 91.3 / 76.9 | 0.0 / 94.0 / 47.0 |
| currency | 0.0 / 98.9 / 49.4 | 70.6 / 0.7 / 35.6 | 59.4 / 78.5 / 69.0 | **43.9 / 94.9 / 69.4** | 0.0 / 98.9 / 49.4 |
| language | 1.7 / 94.9 / 48.3 | 87.3 / 0.1 / 43.7 | **74.0 / 82.6 / 78.3** | 48.0 / 94.1 / 71.0 | 1.7 / 95.3 / 48.5 |
| calling_code | 0.0 / 93.8 / 46.9 | 100.0 / 0.8 / 50.4 | **60.6 / 92.3 / 76.4** | 53.9 / 94.0 / 73.9 | 0.0 / 93.8 / 46.9 |

- A question-blind edit of ≤0.7% of the neuron space reaches 69-84% final.
  A random subspace of the same size does nothing.
- **Where the edit happens:** at the answer position, downstream of the
  question. VADE's intended site is the image positions, so these finals are not
  an image-site disentanglement result.
- **Leakage:** the capital edit drags along calling_code and currency (iso 68%
  and 63%), and `isolated` removes most of it. calling_code's readout is small
  (rank 6-8 in blocks 25-26) and barely overlaps the others.
- **Attributes separate at the MLPs, not at the heads.** The evidence is
  behavioural:
  - The same `isolated` edit at the mover heads (R16.3) reproduced the clean
    run exactly (0% cause).
  - Here it keeps 44-62% cause with 91-95% iso.
  - Held-out cross-energy is lower at the MLPs (0.2-29%) than at the heads
    (50-66%). But the two spaces differ ~37× in width (18,944 vs 512 columns per
    block) and so in chance level (0.7% vs ~6-12%). Do not compare the raw
    numbers across sites.
- **DBM masks (VADE final, recomputed with `score.py` on the 4,000 eval rows):**
  calling_code 88.2 (cause 94.0 / iso 82.5), capital 60.3, currency 59.4,
  language 59.3.
- **These are NOT sparse masks:**
  - `n_selected` = 0, and the mask logits barely moved from their 0 init (within
    ±0.04 after 63 updates at lr 1e-3).
  - At temperature 0.01, though, that spreads the gates over 0.02-0.98 (mean
    0.49). The evaluated intervention is a dense partial swap of all 4 × 18,944
    neurons.
  - Without a fixed gate=0.5 control, it is unknown whether the 63 updates did
    anything. calling_code's 88.2 may come from almost any partial swap, given
    that MLP 25 alone decides it (E2).
  - Rerun at `--mask_lr 1e-2 --epochs 2`, and add the gate=0.5 control, before
    reading masks or mask overlap.

**E4, compose the routes.** (`methods/circuit_compose.py`: movers from (image B,
asked X) plus routers from (image C, asked A), installed into (C, X).)

| arm (n = 548 eligible rows, 64 countries) | gives B's answer to A | 95% CI |
|---|---|---|
| composed movers + routers (two donors) | 78.8% | 75-82 |
| random heads, same blocks | 0% | |
| full 18-head circuit (one donor, image B asked A) | 80.3% | 77-83 |
| whole block window 19-23 (140 heads) | 92.2% | 90-94 |
| complement of the circuit (the other 122 heads) | 0% (C.X kept 91.8%) | |

- **Reading:** entity and attribute travel on separate head sets and compose into
  an answer neither donor run gave. Composing two donors costs almost nothing
  against one consistent donor (78.8 vs 80.3), and the circuit carries 87% of
  what the whole window does.
- **Failures:**
  - **17.7% of composed outputs are neither answer, mostly the right country in
    the wrong FORMAT.** Asked a language with a base question about currency, it
    answers ' ITA' / ' SEK'; asked a calling code, it answers ' 392' for Italy.
    The routers carry *which* attribute, but part of the answer format still
    comes from the base question's tokens.
  - **By target attribute A:** capital 100%, calling_code 76.8%, currency 74.0%,
    language 61.8%.

**E5, direct vs question-mediated route.** (`methods/route_knockout.py`,
attention-edge knockouts after Salazar et al.)

Source image patched at L16 (98.5% transfer on its own); n = 472 eligible rows.

| knockout (+ source image patch) | transfer | 95% CI |
|---|---|---|
| none | 98.5% | |
| question → image, all blocks | 95.3% | 93-97 |
| answer position → image, all blocks | 86.4% | 83-89 |
| answer position → image, **blocks 21-23 only** (the mover blocks) | 93.2% | 91-95 |
| answer position → image, blocks 0-20 | 97.9% | |
| all text → image | 1.3% | |

- **Either route suffices; cutting both removes the entity (1%).**
- **The direct route is the one used in the intact model:**
  - Patching the question tokens alone does nothing at any layer (≤0.6% at L4-24).
  - Once the direct route is cut, the same patch transfers 84% at L22 (CI 80-87),
    against 18% at L20 and ≤4% earlier. The question tokens hold the entity by
    L22, but only as a fallback.
- **The movers do not need the image itself:**
  - Cutting the answer position's attention to the image in blocks 21-23 costs
    only 5 points.
  - Zeroing the mover heads drops accuracy to 38% (E9).
  - So the movers are a bottleneck as heads, but when the image is cut off they
    read the entity from the question tokens. "Movers read the entity from the
    image" describes the intact model, not a necessity.
- The empty-knockout no-op check is exact (`max |dlogit| = 0`) after the bf16
  renormalization fix in bdcd2c4.

**E6, the same pairs with the country named in text.**
(`methods/text_entity_contrast.py`)

- **Accuracy (clean, all 84 countries):** image vs text.

  | | capital | currency | language | calling_code |
  |---|---|---|---|---|
  | image | 97.6 | 89.3 | 90.5 | 91.7 |
  | text | 100 | 92.9 | 90.5 | 100 |

  The image-vs-text recall gap on this model is 0-8 points.
- **Same handoff:**
  - Image: entity-site patching transfers ≥97% through L21, then 67% at L22, 19%
    at L23 and 0% at L24.
  - Text (the name's last token): 96% at L21, 87% at L22, 20% at L23, 0% at L24.
  - The answer position takes over at L23-24 in both (48% at L23, ~95% at L24).
  - n = 230 image / 232 text eligible rows.
- **Same heads:**
  - Installing the 10 image movers in the TEXT condition transfers 87.1%
    (CI 82-91), against 89.7% for the text condition's own top-10.
  - Image and text top-10 rankings share 9 heads, but only 4 of those are
    movers (21.1 22.19 23.3 23.4). The other 5 are downstream heads in blocks
    24-26 whose output changes with the patch.
  - Each condition's top-10 contains 5 of the 10 movers. Random heads give 0%.
- **Same lookup:** freezing MLPs 23-26 at base removes the answer for both
  (1.7% image, 1.3% text).
- **Limits of this experiment:**
  - The entity-position freeze (blocks 21-26 at the name/image) is
    uninformative because L_e = 21 for both conditions (post-handoff).
  - A Geva-style early subject-enrichment test for text needs a run at an early
    `--entity_layer`.
- **Reading:** the circuit belongs to the language model's entity→fact
  machinery, not to vision. This rules out a "distinct, later image route"
  framing for Qwen2.5-VL-7B, consistent with Venhoff et al.'s finding that this
  model barely degrades relative to its text backbone.

### Tier 2: generality (E7-E11)

**Errors and caveats in the Tier-2 run (c1f773b)**

1. `e7_vocab_brands` first crashed with `KeyError: 'industry'` because AIRBUS
   has no industry label. `head_vocab_projection.py` now skips rows without a
   ground-truth value; only that one row was skipped.
2. The first `e8_internvl_prune` crashed because VADE's `identity/` tuples have
   no train split. `run_tier2.sh` now prunes every tuple directory except
   `identity`, which E8 does not use.
3. **E11's `seq:before` control failed (15-24%, not 0%).** This is a design
   error, not a code bug: the vision encoder's full-attention blocks spread
   object information into every background token. See the
   [`seq:before` section](#pre_image-is-a-guaranteed-null-seqbefore-is-not).
   This voids `seq:before` as a null, not E11's other rows.
4. Harmless warnings: InternVL's "will NOT tie" embeddings (its output layer
   really is untied), and transformers 5.x processor-kwargs deprecations.
5. The inert `brands/founded_year` trace (image patch 0%) was not filtered out of
   the E10 overlap. It only adds zero-overlap rows.
6. The InternVL-vs-Qwen-7B comparison has no flags rows, because the 7B flags
   traces predate `image_head_pipeline`. Flags is compared against `common10`
   by hand below.

**E10, do the head lists overlap beyond chance?**
(`methods/head_overlap.py`, 20,000 permutations; the `block` null keeps each
list's per-block counts.)

- Each of brands/hq_country, celebrities/nationality and
  celebrities/native_language shares **7 of 8** top heads with `common10`
  (p < 1e-4, both nulls).
- Animals share part of the set (native_continent 5/8); body_covering shares
  almost none.
- Heads in ≥4 of the 13 traces: 21.1 21.4 21.5 22.15 22.17 22.19 23.3 23.4 23.6
  23.11 (block-null p = 0.0005).
- Park et al.'s Visual Retrieval Heads (blocks 14-19) share **0** heads with any
  trace.
- Lietzow et al.'s promoting heads share 3 with `common10` (21.5 22.13 23.6).
  Significant under the pool null (p = 0.002) but not the block null (p = 0.022).
  Their single "writing" head, 23.6, is one of ours.
- **Lietzow caveat:** their paper labels the model "Qwen-VL-2.5" but cites the
  Qwen2-VL report, so the checkpoint behind their list is uncertain.
- **Multiple comparisons:** the table has 100+ pairs and none of the p-values
  are corrected. The headline rows sit at the permutation floor (5e-5) and
  survive any correction; p ≈ 0.02 rows do not.

**E9, zero Park's Visual Retrieval Heads.** (`methods/vrh_ablation.py`, flags,
672 generations per arm.)

- **Index check:** `park_vrh` sits at the 98.1st percentile of box-attention
  mass, against 74.2 for the 1-based reading. The list is 0-based and correctly
  indexed.

| arm | accuracy | currency | mover name rank | name rank = 1 |
|---|---|---|---|---|
| clean | 88.8% | 82.7% | 1.61 | 93.8% |
| zero `park_vrh` (20 heads) | 81.7% | 61.9% | 2.49 | 91.4% |
| zero random, same blocks (×3) | 86.3-87.8% | 73-80% | 1.58-1.70 | 93.3-93.8% |
| zero `common10` (10 heads) | 37.9% | 27.4% | - | - |

- **Reading:** Park's heads do attend to the flag, but they are not the
  bottleneck: −7 points against −1 to −2.5 for random heads, mostly on currency.
- The movers still rank the right name first in 91% of rows. Zeroing just the 10
  movers drops accuracy to 38% (CI 34-42; calling_code 11%).
- **Missing control:** this run has no random set of 10 heads zeroed in blocks
  21-23 (same size, same blocks). Until it runs, "the movers are the bottleneck"
  rests on the install results (E4, E6), not on this ablation.
- The currency-specific drop under `park_vrh` (−21 points, against −3 to −10 for
  random sets) is unexplained.

**E7, brands and celebrities: severed test and vocabulary projection.**
(`head_severed.py --gold own`, 128 pairs each, each entity's own top-8 heads.
All self-checks pass: read-back 0.0, identity bit-exact, liveness 13-15
logits.)

| source answer transferred (divergence-point scoring) | brands hq_country | celeb nationality | celeb native_language |
|---|---|---|---|
| heads only | 64.6% | 76.2% | 89.8% |
| + MLP 22-27 frozen at base | 1.6% | 1.6% | 1.6% |
| ... @mean | 38.6% | 23.8% | 25.8% |
| ... @other (third entity's answer) | 91.9% | 82.1% | 85.9% |
| MLP 25 alone, frozen at base → @mean | 11.0 → 52.8% | 22.2 → 61.1% | 60.2 → 89.1% |
| attention 24-27 frozen at base | 73.2% | 81.0% | 89.1% |

- **Reading:** MLPs 22-27 write the answer on two new entity types (@other lands
  on the third entity 82-92%), and late attention does not matter.
- **This only half-reproduces flags' E2:**
  - The heads alone transfer less here (brands 64.6%).
  - **Mean-freezing MLPs 22-27 keeps 60% of the brands transfer** (38.6 of
    64.6) and 29-31% for celebrities, against ~0-22% on flags.
  - Many brands share an HQ country, so the mean value may itself point at a
    common answer. Necessity is therefore weaker here than on flags.
- **Difference from flags:** no single block is the lookup. A single-block freeze
  at base mostly recovers at @mean (kept 0.8-1.0), so the base freeze was
  re-asserting the base answer. Block 25 contributes most, writing the third
  entity's answer in about 24% of rows. On flags, MLP 25 alone wrote
  calling_code.
- **Vocabulary projection:** name ranks are weak (brands 18-48 of 112,
  celebrities 20-43 of 73; flags was about 1-10 of 78). Two caveats: many
  names share a first token (43 brands, 29 celebrities), and the entity may be
  encoded non-lexically.
- The heads stay question-blind: the asked value scores about the same as other
  values, e.g. 22.15 1.51 vs 1.45.
- For almost every head, the direct entity write comes more from
  **background** image tokens than from the object (see E11).

**E11, object vs background tokens.** (`ndm/ceiling_sweep.py`, full residual
swap per position set, 96 rows.) flags/capital, cause by layer:

| layer | `entity_only` | `~entity_only` | `~entity_ring1` | `full_image` | `seq:before` |
|---|---|---|---|---|---|
| 10 | 34% | 34% | 14% | 100% | 16% |
| 18 | 16% | 68% | 24% | 100% | 19% |
| 21 | 15% | 59% | 20% | 98% | 18% |
| 23 | 2% | 1% | 1% | 19% | 1% |

- Brands and celebrities show the same shape:
  - **brands:** `entity_only` falls from 50% to 9-13%; `~entity_only` peaks at
    64% at L19.
  - **celebrities:** `entity_only` falls from 83% to 42% by L20; `~entity_only`
    peaks at 44%.
- **Not confined to the object:** even at layer 10, the object's tokens alone
  transfer only 34-83%, and part of this spread comes from the vision encoder
  (`seq:before` ≠ 0).
- **It migrates outward inside the language model** between layers 10 and 18:
  object-only falls while everything-except-object rises 34 → 68%. Most of
  that gain sits in the one-token ring around the object.
- So the movers (blocks 21-23) mostly read the entity from around the object,
  which matches E7's background-dominated head writes.
- **Partial patches are not shares:** a partial patch pits the source region
  against the rest of the image, which still carries the BASE entity. "Object
  alone transfers 34%" is the outcome of that conflict, not the fraction of
  identity the object holds.

**E8, other models.** (`image_head_pipeline.py` on Qwen2.5-VL-3B and
InternVL3-8B, after VADE accuracy sweeps and pruning per model.)

| model | handoff (depth) | consensus heads (in ≥3 traces) | top-8 sufficiency, flags |
|---|---|---|---|
| Qwen2.5-VL-7B (28×28) | blocks 21-23 (75-82%) | `common10` | see E2/E4 |
| Qwen2.5-VL-3B (36×16) | blocks 27-33 (75-92%) | 27.10 27.11 27.12 31.1 31.3 31.7 32.13 33.3 | 66-92% (calling_code 1.6%) |
| InternVL3-8B (28×28) | blocks 21-23 | **21.1 21.5 22.13 22.15 22.17 22.19 23.3 23.4 23.6 23.17 = `common10`** | 84-92% (calling_code 1.6%) |

- **InternVL3-8B inherits the movers:** it uses a different vision encoder and
  256-token tiling, but its language model comes from Qwen2.5-7B, and it
  independently recovers exactly `common10`.
  - This holds at the ≥3-of-7-traces threshold. At ≥4 it is 7 of the 10, both
    p = 0.0005.
  - Its flags traces share 8/8 (capital, language) and 7/8 (currency,
    calling_code) heads with `common10`.
  - Celebrities share 7/8 and brands 6/8 with Qwen-7B's traces (p < 1e-4).
  - The movers survive a new vision encoder and multimodal training, so they
    are a property of the language model.
- **Qwen2.5-VL-3B** puts its handoff at the same relative depth, with 27.11 in
  every flags top-8.
  - Lietzow's 3B promoting heads share 31.3 and 31.7 (pool p ≈ 0.005, not
    significant under the block null).
- Park's InternVL VRH list again shares 0 heads.
- **calling_code** is the exception in both models: no 8 heads suffice (1.6%),
  but the whole block window does (72-98%). It is spread across a whole block
  and needs several answer tokens.
- InternVL brands/hq_country is similar: top-8 gives 7.8%, the window 88%.
- The 3B stage-1 curves are coarse (32 rows per layer).

### LLaVA-1.5-7B: the same order, read off the logit lens (X8, X9)

Molab pilots on a Vicuna (Llama-2) language model. Notebooks: `notebooks/x8_llava_window.py` and
`notebooks/x9_llava_toptokens.py`.

- **Handoff (X8):** residual swaps put LLaVA's image→text handoff at blocks ~18-21 (0.56-0.69 of
  depth). No window of ≤3 blocks carries it, so the read is spread out as on Idefics3, not compact as
  on Qwen's 21-23.
- **Logit lens (X9):** 8 flags × 4 questions, read at the last prompt token, through the model's own
  final norm and unembedding.
  - **Entity arrives first:** the country name enters the top-20 at layers 19-21 and reaches rank 1
    by about L20-26. This dates the read independently of patching and matches X8's handoff.
  - **Question next:** the question word ('capital', 'currency') appears at L22-24, or at L18 when
    the country is named in text.
  - **Value last:** the asked value climbs over L22-30 and is top-1 from L29 to 32. Values that were
    not asked stay around rank 30-60.
- **Prompt wording matters for LLaVA.** With the Qwen-designed controlled prompt
  (`Report the capital of this country.` + `Answer:`), LLaVA answers the country NAME for every image
  question: the name is still rank 1 at the last layer, with the value at rank 2-6. The pipeline is
  intact; only the output step picks the wrong token.
  - `What is the capital of this country?` (`VADE_PROMPT_STYLE=what_is`) fixes it: 37-52% accuracy
    on 84 flags, ≤1% name answers.
  - `methods/prompt_style_check.py` compares the candidate wordings.
- **Reading:** see → move → recall holds on LLaVA in the same order and at a similar relative depth
  as on Qwen:
  - the entity reaches the answer position at about 0.6 of depth;
  - the value is looked up afterwards, at about 0.7-0.95.

### Open items

- **E9:** zero a random 10-head set from blocks 21-23 (the matched null for "zero
  `common10` → 38%").
- **E3:** add a fixed gate=0.5 half-swap control, then rerun the DBM masks at a
  higher mask learning rate (`n_selected` = 0, gates dense).
- **E6:** rerun the freeze phase at an early `--entity_layer` for the text
  condition, to test Geva-style subject enrichment.
- **E2/E3/E7 on other models:** the MLP-lookup stage was only tested on
  Qwen2.5-VL-7B.
- Add a layer-0 identity check as the real image-position null, and correct
  `common/position_sets.py`'s docstring and CLAUDE.md, which still call
  `seq:before` a guaranteed null.
- Explain why zeroing Park's heads hurts currency specifically.

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
- **Point `HF_HOME` at a sibling directory before downloading anything.**
  Qwen2.5-VL-7B-Instruct is ~16GB, and many pod images put the default
  `~/.cache/huggingface` on a small root-disk overlay (not the large data
  mount) -- a first-run download there fills the root disk and crashes
  mid-transfer. Use the same sibling convention as `VADE_ROOT`:
  ```
  export HF_HOME=$(realpath ../hf_home)
  ```
  run once per shell (add it to your shell rc to make it permanent), before
  step 4 below or any script that loads the model. Don't rely on a pod's
  preset `HF_HOME` even if `env | grep HF_HOME` already shows one -- verify
  it resolves to spacious storage, since a stale or partial cache at a
  *different* path than the one your shell/scripts actually use is how you
  end up with several redundant multi-GB copies of the same model.

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

Make sure `HF_HOME` is set per step 0 first, so this lands in one place:

```
echo "HF_HOME=$HF_HOME"   # should print the sibling ../hf_home path, not a default
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
