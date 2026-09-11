# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Execution repo for the [VADE](https://github.com/Shaveen12/VADE) benchmark (Visual
Attribute DisEntanglement). VADE itself stays a pure benchmark: data, ground truth,
prompt templates, object-token geometry, and a method-agnostic scorer (`eval/score.py`)
that never touches model internals. **This repo holds everything that does touch model
internals**: activation extraction, dictionary learning, feature selection, and the
actual causal intervention, run against VADE's entities (flags/brands/animals; a fourth,
`compounds`, is a known VADE entity name with no data built yet).

Expected checkout layout — every script here defaults `--vade_root` to a sibling
directory, and default output paths assume this repo root:

```
<some root>/
  VADE/            benchmark data (VADE/data/<entity>/...) + scorer (VADE/eval/score.py)
  VADE-evals/      this repo
```

Override with `--vade_root /path/to/VADE` or the `VADE_ROOT` environment variable if a
checkout isn't laid out this way. This repo pins no `requirements.txt`; see the README's
"Reproducing results on a fresh machine" runbook for exact versions the pipeline was
built/verified against (torch 2.13, transformers 5.16, scikit-learn 1.9) if `pip install`
of latest breaks.

## The pipeline (Phase A → C), in order

Each stage's script consumes the previous stage's output file; none re-runs the model
behind an earlier stage.

1. **`methods/sae.py`** (Phase A) — loads Qwen2.5-VL-7B-Instruct, one forward pass per
   entity image, records the residual-stream hidden state at every decoder layer,
   restricted to the object's image-token positions (per-entity token sets defined in
   `<entity>/object_location.json`, e.g. flags' 8-token `flag_only`/24-token
   `flag_ring1`). Writes `methods/activations/<entity>_<model>_all_layers.pt`.
   `--dry_run` validates images/metadata resolve without loading the model (no GPU
   needed) — the fastest sanity check after touching any entity-loading code.
   `--augment_variants N` additionally builds a pixel-perturbation-only augmented pool
   (color jitter/noise/blur, never geometric, so `object_location.json` token positions
   stay valid) for `fit_dictionaries.py` to optionally train on.

2. **`methods/fit_dictionaries.py`** (Phase B step 2) — fits one PCA and/or SAE
   dictionary per (entity, token_set, layer) on Phase A's activations, unsupervised (no
   attribute/entity labels). Fit on *flattened per-position rows* (`features.py`'s
   `flatten_positions`), not pooled per-image vectors, because the intervention script
   later needs to encode a single token position. PCA sweeps cheaply at every layer; SAE
   is data-starved (each entity has only ~84-130 images) so it's deliberately modest
   (dict_size ~2x hidden_dim, not the usual 8x-32x overcomplete) and by default only fit
   at a few representative layers. `--augmented_pool_variants N` folds in Phase A's
   augmented pool to widen the SAE's training set — `select_features.py` is unaffected
   either way, since it always scores against the real (non-augmented) activations.
   Writes to `methods/dictionaries/` (or `--dictionaries_dir`, e.g.
   `methods/dictionaries_kaggle_pool/` for the externally-augmented checkpoints
   described in the README).

3. **`methods/select_features.py`** (Phase B steps 3-4) — mean-pools each image's
   token-set positions into one vector (attributes are whole-entity facts, not local
   features, and positions are correlated — see `features.py`'s `pool_positions`),
   encodes through a fitted dictionary, then runs two-step L1-SVC + `SelectFromModel`
   feature selection swept over (direction, layer, C), independently per attribute:
   - **forward**: broad filter on entity-ID label (1 example/class, never CV'd) → narrow
     to the attribute label (real stratified CV).
   - **inverse**: same, swapped — broad filter on attribute (CV'd) → narrow to entity-ID.

   Each candidate is scored by a cause/iso proxy mirroring `eval/score.py`'s
   `final_score = 1/2(cause + mean(iso))` shape at the feature level; the
   (direction, layer, C) maximizing that score is the attribute's winner, written to
   `methods/selections/<entity>/<token_set>_<dict_method>/winners.json` (plus a full
   `sweep.jsonl`). **Important gotcha**: this default output path does not encode which
   `--dictionaries_dir` was used to fit the dictionary being scored — running against a
   different dictionary set (e.g. `dictionaries_kaggle_pool`) silently overwrites a
   previous run's `winners.json`/`sweep.jsonl` unless you pass a distinct
   `--output_dir`. An attribute can come back "no winner" legitimately: it needs ≥2
   classes with ≥2 members to support a held-out CV split at all (on flags' 84-country
   sample, only `language` qualifies — `capital`/`calling_code` are unique per country
   and `currency` has just one repeated class), which is a property of the dataset, not
   a bug.

4. **`methods/intervene.py`** (Phase B step 5, the actual causal intervention) — for
   every attribute with a winner in `winners.json`: runs the source image through the
   model once, caches its hidden state at the winning layer, then on the base image's
   forward pass patches in the winning dictionary-feature subset
   (`encode(base)`/`encode(source)`, copy over only the winning dims, `decode`) at the
   object's token positions via a forward hook on that layer (a no-op after the initial
   multi-token prefill, since the patch is already baked into the KV cache by then), and
   lets the model generate freely. No training involved. Writes predictions in
   `VADE/eval/score.py`'s format; runs are **append-and-resume** — re-running the same
   command skips `(attribute, row_index)` pairs already present in `--out`, so an
   interrupted run just continues.

5. **`VADE/eval/score.py`** (in the sibling VADE repo, not this one) — method-agnostic
   scorer: takes a predictions JSONL and reports `cause` (did the target attribute flip
   to the source's value) and `iso` (did every *other* attribute stay at the base's
   value) per attribute, plus `final_score = 1/2(cause + mean(iso))`. Writes
   `<predictions-file-stem>_summary.{json,md}` next to the predictions file (or under
   `--out_dir`). This final number is the thing to compare across methods/runs.

`methods/features.py` is the shared module behind steps 2-4: activation
loading/reshaping (`flatten_positions`, `pool_positions`), the interchangeable
`PCADictionary`/`SAEDictionary` classes (same `encode(X)`/`decode(F)`/`save()`/`load()`
contract regardless of which kind downstream code is holding), and the `fit_pca()`/
`fit_sae()` routines. `methods/build_external_flag_pool.py` builds the Kaggle
"country-flags-in-the-wild" augmentation pool referenced in the README's reproduction
runbook.

## methods/dbm/ -- a second, separate pipeline (Differential Binary Masking)

DBM is architecturally unlike PCA/SAE: it has no dictionary/featurizer at
all (`F_A(n) = n`, the raw residual stream) and its "feature selection" is
a gradient-trained sigmoid mask learned end-to-end against the real
generation objective (teacher-forced cross-entropy + an L1 sparsity term
on the mask, temperature-annealed), not an offline classifier probe over
precomputed activations. Because of that, it shares **no code** with
`features.py`/`fit_dictionaries.py`/`select_features.py`/`intervene.py` --
it lives entirely under `methods/dbm/` (`intervention.py`/`train.py`/
`eval.py`/`run_layer.py`/`layer_sweep.py`), built on top of
`methods/common/` + `methods/adapters/`, which were copied **verbatim**
from the sibling VADE repo's own `methods/common`/`methods/adapters` (the
same model/entity-agnostic infra VADE's own DAS implementation uses --
activation-patching hooks, gold-token/teacher-forcing utilities, entity
asset + batch loading, per-entity source-activation caching, one
`ModelAdapter` per model family). `methods/dbm/intervention.py` imports
`pyvene`'s own `SigmoidMaskIntervention` directly (RAVEL's own Appendix B.4
says pyvene was the reference implementation) rather than reimplementing
it, but deliberately does **not** use pyvene's `IntervenableModel` wrapper
-- that machinery is built/tested against text-only HF models with no
documented multimodal support, whereas `methods/common/hooks.py`'s hook
mechanism is already proven against Qwen2.5-VL by VADE's DAS. See the
README's "methods/dbm/" section for usage and exact hyperparameters
(`--l1_coef`/`--temperature_start`/`--temperature_end`, matching RAVEL's
own reported optimum/schedule).

Where artifacts land: entity data/tuples/pruned-tuples and the shared
per-entity source-activation cache are read from (and, for the cache,
written to) `--vade_root` (the sibling VADE repo) -- same as everywhere
else in this project. DBM's own trained checkpoints/predictions/train logs
are NOT written into VADE, though -- they land under this repo's own
`results/`/`logs/` trees instead (mirroring VADE's own `results/`/`logs/`
layout, just rooted here). Nothing under `methods/dbm/`, `methods/common/`,
or `methods/adapters/` has ever been added to `.gitignore` -- worth
revisiting once real runs exist, since VADE gitignores its own
`results/`/`logs/` entirely (pod-specific run artifacts) while this repo's
existing convention is closer to committing predictions/selections
directly (see `methods/interventions/`/`methods/selections/` above).

## methods/ndm/ -- Native Dictionary Masking, DBM's engine at a different site

NDM reuses DBM's trainer verbatim and moves the mask from the residual stream
to a decoder block's **MLP hidden state** (the post-SwiGLU neuron vector
`act_fn(gate_proj(x)) * up_proj(x)`, width `intermediate_size` = 18944 on
Qwen2.5-VL-7B, vs `hidden_size` = 3584). In RAVEL's `F_A` framing: DBM's
featurizer is the identity, DAS's is a learned rotation, PCA/SAE's is a fitted
dictionary, and NDM's is **the model's own MLP** -- nonlinear and
zero-parameter, so there is nothing to fit and no reconstruction error.
Structurally it is `encode -> mask -> decode` like `intervene.py`'s SAE
patching, with `down_proj` as the decoder; because `down_proj` is linear, an
axis-aligned mask in neuron space induces a *non*-axis-aligned intervention in
residual space, which is a different hypothesis class from DBM's. The
motivation is the privileged basis: elementwise nonlinearity + elementwise
gating mean only *permutations* preserve that space's function, so its
coordinates are real objects, whereas any rotation of the residual stream
yields an identical model -- making DBM's chosen dimensions a fact about one
checkpoint's arbitrary axes.

**Code layout -- read this before editing either method.** `methods/dbm/`
is the shared ENGINE; `methods/ndm/` owns only the name, the CLIs and the
results namespace. `dbm/`'s `train_layer`/`eval_layer`/`run_one_layer`/
`run_sweep` all take an optional `site=` that defaults to the residual
stream, so DBM's behavior is unchanged. What actually differs between the
methods is ~40 lines (hook attachment, mask width, source-side capture)
against ~800 lines of already-debugged infrastructure -- the
gradient-accumulation tail flush and the temperature-resume anchor (both
commit 77470b4), checkpoint/resume, progress logging, the predictions format
`score.py` expects. **Do not fork the loop**; add a site or a parameter.

Two NEW files in `methods/common/` (new files, not edits, so the
verbatim-from-VADE ones stay byte-identical): `sites.py` (the
`InterventionSite` abstraction -- `residual` delegates to `hooks.py`'s own
functions) and `site_source_cache.py` (the MLP sites' source cache, keyed by
(site, positions, layer), since MLP internals are absent from
`output_hidden_states`). `adapters/{base,qwen2_5_vl}.py` gained three
additive methods (`intermediate_size`/`get_mlp_block`/
`get_mlp_hidden_module`) so `sites.py` never needs to know Qwen attribute
names.

**Five single sites + one joint site, all addressing the same block**
(`--layer L` = block L-1, so `L` means the same block for every site):
`residual` (3584, DBM's site), `attn_output` and `mlp_output` (3584, one
sublayer's contribution each), `attn_head_output` (3584 = 28 heads x 128,
`o_proj`'s input, privileged per head), `mlp_hidden` (18944, NDM's own site,
default) and the JOINT `attn_output+mlp_output` (both sublayer contributions
patched in ONE pass; `sites.py`'s `JointSite`, width reported as the 7168
sum). Only the two MLP sites are NDM *training* sites (`NDM_SITES`); the
other single sites exist for the diagnostics, though `dbm/train.py`'s engine
accepts any of them. The joint site is diagnostic-ONLY -- training there
would need one mask and one L1 term per part, so `JointSite.forward_patched`
/`lookup_source` raise and `ndm/config.py`'s `--site` choices never offer it.
`ceiling_sweep.py`'s `--sites` DEFAULTS to the FOUR independent ones
(`residual attn_output mlp_output attn_output+mlp_output`) and deliberately
OMITS `attn_head_output`/`mlp_hidden`, which a full swap cannot distinguish
from their post-projection partners -- `common/sites.py`'s
`FULL_SWAP_EQUIVALENT` holds that mapping, and the run prints it under the
summary table so the absent rows are not mistaken for unprobed ones. **The
`mlp_output` row IS `mlp_hidden`'s ceiling**, so the "run ceiling_sweep
before training" rule below is still satisfied by a default run even though
NDM trains on `mlp_hidden`. Pass the omitted sites explicitly to spot-check
the identity (a hook/determinism canary, and the only check
`attn_head_output` has ever had). `resolve_site()` accepts `ALL_SITES`;
`InterventionSite()` still rejects joint names, so training paths fail
loudly. The extras are control arms: `residual` vs
`attn_output`/`mlp_output` isolates locality at matched width, `attn_output`
vs `mlp_output` isolates which sublayer, and the joint site isolates the
accumulated PREFIX. `blocks:N` (parsed, not enumerated, so any N>=1; needs
--layer>=N; `blocks:1` aliases `attn_output+mlp_output`) widens that to N
CONSECUTIVE blocks ending at --layer L, swapping exactly what residual@L has
that residual@(L-N) does not -- so sweeping blocks:1..5 measures HOW DEEP the
redundancy goes. All 2N parts register before ONE source forward and ONE
generate, so a five-block span costs the same model calls as a one-block one. That last one matters because the three are **not
additive**: `residual@L = residual@L-1 + attn_output@L + mlp_output@L`, but a
sublayer swap only INSERTS source evidence while a residual swap also DELETES
the base prefix -- so `residual` legitimately reads 68.8% at a layer where
both its sublayers read 0.0% each. joint ~= residual means the block does the
work; joint ~= 0 means the prefix is necessary and the residual curve is
measuring remaining depth to REPAIR the edit, not information arrival.
**A pre-projection site and its post-projection site are NOT separable by
`ceiling_sweep.py`** -- `down_proj(h_source)` is exactly `mlp_out_source`, so
a FULL swap of either produces the identical residual update; measured, and
`mlp_output == mlp_hidden` / `attn_output == attn_head_output` in every cell
of all three position sweeps. The privileged basis only buys anything for a
SPARSE mask, so that question can only be settled by a trained mask. Note `self_attn` returns a TUPLE, unlike `mlp` -- sites.py's hooks
patch element 0 and pass the rest through.

**Position sets beyond VADE's own** (`methods/common/position_sets.py`, a
WRAPPER around `build_batch` -- `entities.py` stays a verbatim copy): `~<name>`
(complement within the image span), `ring:K[@base]` (K-th Chebyshev band around
the object bbox; `ring:0`==`flag_only`, `ring:0|ring:1`==`flag_ring1`),
`side:left|right|beside|above|below`, `seq:before|after|between` (raster =
sequence = causal order), `tok:-K[:N]` (text window K back from the prompt end;
`tok:-1`==`last_token`), `pre_image[:N]`, `vision_end[:N]`, and `A+B` unions
(may mix image and text specs). `ceiling_sweep.py --positions_list` runs many
specs in one model load, sharing the row sample.

**`seq:before` is the only guaranteed NULL in the whole design, and is worth
running before trusting any other zero.** Every image in an entity is one fixed
canvas render with only the object's pixels varying, so background tokens that
precede the object in raster order have bit-identical activations in base and
source under causal attention -- patching them is provably a no-op and MUST
score 0%. `seq:after` is its informative twin: those tokens can differ only via
attention to the object, so their ceiling measures how far the object's
information has leaked into the background by a given layer.

**Gotchas.** (1) `--l1_coef`'s default `0.001` is RAVEL's optimum for a
~4096-wide residual stream and is NOT calibrated for 18944 dims -- since
`l1_penalty` is a plain `mask.abs().sum()`, the term is ~5.3x larger at equal
per-dim magnitude. Sweep it and read `mask_stats.json`'s `n_selected`
alongside `final_score`. (2) `--layer 0` is invalid for MLP sites (the
embedding output has no MLP). (3) The site IS encoded in the config tag, so
`mlp_hidden`/`mlp_output` runs don't collide. (4) Run
`methods/ndm/verify_sites.py` before any real run -- a hook on the wrong
tensor trains fine and produces plausible numbers; it also covers `residual`,
so it doubles as a DBM regression check. (5) **Run
`methods/ndm/ceiling_sweep.py` before training a new (site, layer) at all.**
A mask selects a subset of what a full source swap uses, so the full swap is
a hard upper bound on `cause` -- and it costs forwards, not a training run.
Learned the hard way: NDM's first run (flags/language, layer 14, mlp_hidden)
scored cause=2.4%, which looked like a mask-training failure (it was one --
ce_loss flat across all 480 steps), but the full-swap ceiling there was also
~0: swapping ALL 18944 neurons at all 24 object positions did not move the
answer. No temperature/l1/lr tuning could have helped. Notably `mlp_output`
(same width as residual, same locality as mlp_hidden) had the same ~0 ceiling
while `residual` flipped the answer outright -- so at layer 14 the limiting
variable is LOCALITY, not the basis: one block's additive down_proj update
does not carry a whole-entity attribute.

## methods/dla.py -- direct logit attribution (read-only, exact, one forward pass)

"Which components help make the identification right", answered without
patching, gradients, or approximation. The residual stream is a sum
(`resid_final = embed + sum_b (attn_output[b] + mlp_output[b])`) and the head
is an RMSNorm plus a bias-free linear map whose only nonlinearity is a
per-position SCALAR -- freeze that at the value the real forward pass computed
(from the FULL final residual, never per component) and the answer's logit
splits exactly into one term per component:
`logit(t) = sum_c scale * (component_c . (W_U[t] * final_norm.weight))`.
Two additive adapter primitives carry the model-specific half
(`final_norm_scale`, `logit_direction`, both asserting their assumptions --
an `lm_head` bias or a renamed `variance_epsilon` fails loudly rather than
silently changing the numbers). Sublayer outputs come from
`common/sites.py`'s `register_capture` (made public for this), so DLA reads
the IDENTICAL tensors the interventions patch.

`--direction` chooses what is decomposed: `gold_minus_mean` (default),
`gold`, or `base_minus_source` -- the VADE-specific one, and usually the most
on-point since it is the exact axis `cause` moves along.

**Two self-checks run per row and are asserted, not assumed** (every failure
mode here -- wrong tensor, double-applied final norm, off-by-one over blocks,
head bias -- produces plausible numbers rather than an error): RECONSTRUCTION
(`embed` + every captured sublayer vs the model's own
`hidden_states[n_layers-1]`; note `hidden_states[-1]` is POST-final-norm in
this project's transformers, the same trap `logit_lens.py` documents) and
ADDITIVITY (the terms summed vs the real logit off `out.logits`). Both are
relative errors against `--tolerance` (default 2%); bf16 accumulation over ~57
components puts the honest floor near 1%.

**Read it alongside `ndm/ceiling_sweep.py`, not instead of it.** DLA measures
DIRECT paths to the logit only (a component acting through a later head's
attention pattern is credited to that head), and like every per-component
attribution it is MARGINAL -- one number per component, so it cannot represent
a conjunction. Where an attribute is encoded redundantly across depth (live
`residual` ceiling at a layer whose sublayers all read 0), expect DLA mass
spread thin rather than concentrated; that spreading IS the finding. The two
probes disagreeing is informative, not a bug in either.

`methods/mech_probe.py` runs `dla.score_one_row` off the SAME forward pass as
`logit_lens` and `attention_maps` (its capture hooks are live during that one
pass; `--skip_dla` opts out, `--dla_direction`/`--dla_tolerance` configure it).

## methods/head_trace.py -- tracing the image->text handoff

`ceiling_sweep`'s two position sets BRACKET the handoff without locating it:
on flags/language an image-position residual swap reads 100% through layer 22
then 0% from 23, while `last_token` reads 0% through 21 then 68.8%/100% at
23/24. The crossover IS the read (before it, editing the image propagates;
after it, editing the image is too late and editing the destination is
decisive), so the read sits in blocks ~21-23 -- from the table alone.

Attention is the ONLY cross-position operation in a transformer (MLPs are
position-wise; the residual stream never mixes tokens), so the whole transfer
is some set of (block, head) pairs in that window. head_trace finds them in
two phases, both patching the IMAGE side -- a per-head swap AT THE LAST TOKEN
is a subset of `attn_output` there, which measured ~0, so it is dead on
arrival. PHASE 1 (2 forwards/batch, no generation): capture per-head
`attn_head_output` at the last token clean vs with the image residual patched
to source at `--patch_layer`; rank by `delta_resid` = ||dz_h W_O_h^T|| (what
actually lands in the residual -- `o_proj` weights heads very differently, so
`delta_z` can disagree), with `delta_dla` (dla.py's exact per-head logit term,
differenced, each run with its own frozen scale) as the answer-axis view.
PHASE 2: keep the image patched and RESTORE the top-k heads at the last token
to base; cumulative k because single-head knockout cannot see a conjunction,
same reason `blocks:N` exists. `--n_random` is a built-in null control -- if
random-k hurts as much as top-k the ranking is uninformative and the curve
means nothing.

`--patch_layer` MUST be below the handoff (where ceiling_sweep's image column
is still large); the script warns rather than reporting a flat curve. Needs
two batches per chunk (image positions to patch, `last_token` to read) and
ASSERTS they agree on `base_input_ids`/`attention_mask` -- a silent mismatch
would patch one prompt's image and read another's last token with every
downstream number still plausible.

## A key finding worth knowing before trusting proxy scores (see RESULTS.md)

The Phase B feature-selection proxy (a classifier reading the target attribute off
dictionary-encoded features) can score much higher `cause` than the real Phase C
generation-time intervention actually achieves (e.g. flags: proxy ~0.74-0.76 vs. real
2.2%-19.2% flip rate). Linear separability of a feature for a classifier does not mean
patching it in reliably steers what the model generates — treat `select_features.py`'s
`sweep.jsonl` scores as a candidate filter, not a prediction of real intervention
accuracy; only `intervene.py` + `eval/score.py`'s numbers are the real result.

## Commands

```bash
pip install torch torchvision "transformers>=4.49" accelerate scikit-learn pillow numpy tqdm huggingface_hub

# Phase A
python methods/sae.py --entity flags --dry_run          # validate paths/images, no GPU/model
python methods/sae.py --entity flags --limit 3          # smoke test
python methods/sae.py --entity flags                    # full extraction
python methods/sae.py --entity brands --dry_run
python methods/sae.py --entity animals --dry_run

# Phase B
python methods/fit_dictionaries.py --entity flags --method pca
python methods/fit_dictionaries.py --entity flags --method sae
python methods/select_features.py --entity flags --token_set flag_only --dict_method pca
python methods/select_features.py --entity flags --token_set flag_only --dict_method sae

# Phase C
python methods/intervene.py --entity flags --token_set flag_only --dict_method sae \
    --out methods/interventions/flags_flag_only_sae_predictions.jsonl

# Scoring (in the sibling VADE repo)
python ../VADE/eval/score.py --predictions methods/interventions/flags_flag_only_sae_predictions.jsonl \
    --entity flags --attribute all

# Read-only probes (no training, no intervention) -- all three off ONE forward pass
python methods/mech_probe.py --entity flags --attribute language --dry_run
python methods/mech_probe.py --entity flags --attribute language
python methods/dla.py --entity flags --attribute language --direction base_minus_source
python methods/head_trace.py --entity flags --attribute language --patch_layer 21 --positions flag_ring1

# NDM (Native Dictionary Masking) -- MLP-hidden site, shares dbm/'s engine
python methods/ndm/verify_sites.py --entity flags --attribute language --layer 16   # 1. is the hook right
python methods/ndm/ceiling_sweep.py --entity flags --attribute language \
    --layers 2 6 10 14 18 22 26 --sites mlp_hidden mlp_output residual              # 2. is there headroom
python methods/ndm/run_layer.py --entity flags --attribute language --layer 16 \
    --positions flag_ring1 --site mlp_hidden
python methods/ndm/layer_sweep.py --entity flags --attribute language \
    --layers 8 10 14 16 20 22 --positions flag_ring1 --site mlp_hidden
```

There is no build step, lint config, or test suite in this repo — it's a pipeline of
standalone scripts run directly with `python`, each stage feeding the next via files on
disk. `--dry_run` (Phase A) and `--limit N` (Phase A/C smoke tests) are the fast,
GPU-cheap way to validate a change before spending real GPU time on a full run.

`torchvision` is a required dependency of `sae.py` even though only images are used —
Qwen2.5-VL's `AutoProcessor` eagerly builds a video processor too, and that needs
torchvision present regardless. `methods/activations/`, `methods/dictionaries*/`
(large `.pt` checkpoints, one set fetched separately via `rclone` from Google Drive per
the README — GitHub's 100MB/file limit), `methods/external_data/`, `*.pt`, `*.npz` are
all gitignored.
