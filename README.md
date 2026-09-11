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

#### `seq:before` and `pre_image` are guaranteed NULLs -- run them

Every image in a VADE entity is **one fixed canvas render** with only the
object's pixels varying (`object_location.json` states this explicitly). So:

- `pre_image` tokens precede the image entirely.
- `seq:before` tokens are background tokens that precede the object in raster
  order, and under causal attention can only attend to tokens at or before
  themselves -- identical pixels, identical context.

Both therefore have **bit-identical activations in base and source at every
layer**, so patching them is provably a no-op and they MUST score 0%. Nothing
else in the design is a guaranteed null, which makes these two the controls
that license reading every other 0% as a real null rather than a measurement
failure. `seq:after` is the informative twin: those tokens differ from base to
source *only* via attention to the object, so their ceiling measures how far
the object's information has leaked into the background by a given layer.

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
