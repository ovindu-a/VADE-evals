# Same-image text-attribute swap experiment

## Objective

Locate the positions, sublayers, and depth windows whose donor activations can
switch the requested attribute while holding the image fixed. This is a full-swap
localization experiment, following the existing ceiling-swap methodology. It
does not fit masks or change model weights.

Primary question: which internal substitutions make a recipient asking attribute
A answer attribute B for the same country? Secondary question: do effective
upstream substitutions change the later heads' image-only contributions?

This document specifies the experiment. The core scope/site/window sweep is now
implemented in `methods/attribute_switch_sweep.py`; no pretrained-model results
are claimed. See `ATTRIBUTE_SWITCH_SWEEP.md` for runnable commands.

## 1. Paired inputs

- Recipient: image I + question A + common assistant prefill `Answer:`.
- Donor: exactly the same image I + question B + the same prefill.
- Use capital, currency, language, and calling code: all 12 directed A-to-B pairs
  for every sampled country, including both directions.
- Start from the controlled questions in `methods/attribute_switch_sweep.py`.
  All questions share field definitions and output-format instructions; only the
  requested field changes. The paraphrase control uses a different verb with the
  recipient's original attribute.
- Validate identical vision tensors, image positions, positional inputs, and
  context through the image. Require equal prompt token lengths and identical
  final tokens. Save decoded tokens and exact position lists. Reject misaligned
  pairs rather than silently shifting or padding.
- Run both clean questions before intervention. Report all rows and separately
  rows where both clean answers are correct. Flag equal answer strings/shared
  leading tokens so they cannot masquerade as an attribute switch.

"Text swap" means replacing internal activations at corresponding positions
with those from the donor prompt. Recipient input IDs stay fixed. Actually
replacing the complete input question is the donor-clean baseline.

The donor last-token residual goes into the recipient last-token residual at
the same layer. We do not pool earlier donor tokens into the last position.

## 2. Factor positions separately from the swapped tensor

### Position scopes

| Scope | Exact positions | Purpose |
|---|---|---|
| Earlier text | All ordinary prompt text positions except the final prompt token; exclude image and tokenizer special tokens | Can the changed question be communicated to the readout later? |
| Last token | Final prompt position only | Where can the accumulated state controlling the answer be replaced? |
| All text | Union of earlier text and the final prompt position | Broad text-state substitution, including the readout |

These scopes exclude generated tokens in the primary prefill experiment. Common
text before the image is included in the earlier-text scope for consistency with
the existing code, but is identical across donors and recipients before any
intervention. Save scope definitions explicitly; do not silently include role or
vision delimiter tokens under the name "all text."

### Single-layer sites

Run the following at each position scope:

| Site | Replacement at layer L |
|---|---|
| Residual | Full residual after decoder block L-1 |
| Attention | Full attention output after output projection, before residual addition |
| MLP | Full MLP output before residual addition |
| Attention + MLP | Both outputs from the same clean donor forward in that block |

MLP-only is a comparison arm: it helps distinguish an attention-specific effect
from general sublayer substitution. Full attention output substitutes all heads;
individual-head selection is a subsequent refinement. There is no need to run
full pre-projection head replacement separately, since its full-swap effect is
equivalent to full attention-output replacement.

## 3. Two distinct depth sweeps

### A. Independent single-layer sweep

Sweep L=1 through 28 for Qwen2.5-VL-7B, deriving/checking the actual layer count
from the model. Every (scope, site, L) is a separate recipient run. No replacement
from an earlier sweep cell remains active in a later cell.

Residual layer 0, the embedding output, can be added as a diagnostic control.
It is not an attention/MLP site. With this template, swapping the entire earlier
text at layer 0 should supply the changed question embeddings; swapping only the
identical last-token embedding should be a no-op. This requires runner support.

### B. Consecutive-window sweep

For endpoint L and width N, simultaneously patch layers L-N+1 through L.
Test widths 1, 2, 3, 4, 5, and 8 where N <= L, at each endpoint. Width 1 aliases
the corresponding single-layer condition and must not be counted twice.

Run two window families at each scope:

1. Attention-only window: replace each block's attention output; recipient MLPs
   compute naturally on the changed states.
2. Joint window: replace attention and MLP outputs at every block in the window.

All window tensors come from one unpatched donor forward on that generation
step. Never capture later donor tensors from a partially patched recipient.

For example, endpoint L=24, width N=4 patches one-based layers 21-24, i.e.
zero-based decoder blocks 20-23. Log both conventions. Existing saved head traces
may name zero-based blocks directly; convert them before comparing results.

Repeated residual replacement across a window is a separate optional experiment,
not the meaning of "block span." At the last-token scope, earlier residual
replacements are overwritten by the final one at the patched readout; at broader
scopes, effects can escape through unpatched positions. Keep it out of the primary
matrix to avoid conflating it with consecutive sublayer replacement.

## 4. Decoding and donor alignment

Primary sweep: prefill-only substitution. Patches target original prompt
positions. Recompute full prefixes with `use_cache=False`, retaining the same
prompt interventions on every forward so later generation sees the patched
prompt history.

Follow-up: compare prefill with continuous substitution at effective windows.
For last-token scope, continuous adds every generated position. For all-text
scope, it adds generated positions to all original selected text positions.
Earlier-text scope remains fixed and is evaluated only once.

Before each next-token prediction, the donor sees its own question followed by
the recipient's actually generated suffix. Capture donor values at corresponding
positions; do not repeat a first-token vector or use independently generated
donor suffixes. Gold answers are used only for evaluation, never to supply the
free-generation donor prefix.

## 5. Controls and implementation checks

- Recipient clean and donor clean for every image/question pair.
- Self donor at each intervention configuration: logits must match recipient clean.
- Same-attribute paraphrase donor with aligned tokenization: checks generic
  wording sensitivity rather than an attribute change.
- Final-layer earlier-text residual replacement must leave the next-token
  logits unchanged, since no later attention can read the changed positions.
- Final-layer residual replacement including the current readout must reproduce
  donor logits under the same generated suffix.
- Width-1 attention/joint windows must match their single-layer counterparts.
- Image-token states should remain unchanged across these runs when all changed
  question tokens are after the image and attention is causal. Check numerically.
- Keep image preprocessing, precision, attention backend, generation budget,
  country sample, and prompts fixed across intervention configurations.

## 6. Measurements

Primary output per (scope, site/window, endpoint, width, direction, mode):

- Donor-attribute full-answer match, original-attribute full-answer match, and
  neither-answer rate; retain generated text for inspection.
- First-token scores with shared-first-token rows distinguished; full-answer
  scores use every gold suffix token and a sufficient generation budget.
- Both unconditional and both-clean-correct results, with row/country counts.
- Paired first-step donor and recipient candidate log probabilities; these are
  directly comparable across arms at the unchanged prefix. Later free-rollout
  scores have different conditioning contexts and must be labeled accordingly.
- Aggregate uncertainty by resampling countries, retaining all directions and
  prompt variants together within each sampled country.

Plots: layer curves for single sites, and endpoint-by-width heatmaps for the
two window families, separated by position scope and attribute direction.
Show paraphrase controls alongside switch effects.

Secondary instrumentation on the previously identified image-reading heads:
capture queries, image attention mass, image-normalized attention, and the
image-only vector `sum_image A[t,k] V[k]`. Compare clean recipient, clean donor,
and patched recipient at the final prompt query. Save vectors, not just norms.
Measure naturally computed downstream heads; if their own outputs were directly
patched by a window, label that overlap and do not call it upstream mediation.

An answer switch establishes sufficiency of the substitution in this setup.
It does not by itself establish attribute-specific image content. Text-position
states may already contain image information or an answer. Late full-residual
success can copy an already-computed answer. Window improvements do not uniquely
prove redundancy, and these full swaps are not mathematical upper bounds on
every possible selective intervention.

If an upstream window changes both image-only readouts and answers, a later
mediation experiment can restore the identified heads' image contributions to
recipient-clean values while retaining the upstream swap. That is a follow-up,
not a prerequisite for this localization sweep.

## 7. Execution stages

1. Validate metadata and token alignment, then run clean baselines for all four
   questions on a fixed 8-country exploratory sample.
2. Smoke-test two countries, capital/currency in both directions, layers
   8/16/20/24/28 and widths 1/2/4. Include all scopes, sites, and controls.
3. Run the independent full-depth sweep and attention-only/joint window sweep
   on the fixed exploratory sample. Estimate and print run counts before launch;
   deduplicate width-1 arms and reuse clean baselines across configurations.
4. Freeze selected windows based on exploratory results. Confirm them on at
   least 16 disjoint country identities (subject to dataset availability) and a
   new aligned prompt formulation. Compare prefill and continuous decoding.
   Report prior head-discovery overlap separately; this is not automatically a
   holdout from the historical head-ranking experiments.

Save prompt/image/code provenance, exact position and block lists, rows,
configuration, runtime, and summaries. Resume only identical configurations.
Use separate output directories for smoke, exploration, and confirmation.

## 8. Existing implementation versus required extensions

Reuse `methods/attribute_switch_sweep.py` and shared site/hook machinery rather
than the image-donor construction in `methods/ndm/ceiling_sweep.py`.

Already implemented in the attribute-switch runner:

- Same-image controlled prompt pairing and alignment checks.
- Earlier-text residual replacement.
- Last-token residual, attention, MLP, and joint replacement.
- Last-token joint windows through `blocks:N`.
- Prefill/continuous donor-prefix alignment, controls, and answer summaries.

Now implemented:

- Separate position scope from site, adding all-text scope and supporting every
  sublayer/window at earlier-text and all-text positions.
- Add attention-only windows alongside the existing joint windows.

Remaining optional diagnostics and analysis extensions:

- Optionally enable residual layer 0 with explicit validity checks.
- Add downstream image-readout instrumentation and country-level uncertainty.
- Add explicit discovery/confirmation manifests and held-out prompt selection.

`--block_spans` means attention PLUS MLP; `--attention_spans` means attention
only. Both support explicit `--scopes`; without scopes they retain last-token
semantics.
