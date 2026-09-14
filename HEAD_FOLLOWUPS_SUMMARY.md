# Head Follow-Up Results Summary

## Scope

These results come from the saved decode summaries under
`results/head_followups/flags/<attribute>/decode/summary.json`, plus the two
latest commits:

- `8833259` (`token identification`)
- `1b2064e` (`token knockout results`)

The current outputs are smoke tests with **4 rows per attribute**, not the
planned full 64-row evaluation. The identify and knockout commits extend the
decode results below but do not yet provide a full-sample evaluation.

## Results by Attribute

### Language

- The full image patch reached 100% first-token and full-answer accuracy.
- Top-8 and top-16 selected-head sufficiency reached 100% in both prefill and
  continuous modes.
- Restoring the selected heads in the necessity conditions returned behavior
  to the clean/base result.
- The selected heads therefore appear sufficient and necessary on this sample.
  Most language answers are one token, so continuous substitution has little
  additional opportunity to help.

### Capital

- The full image patch reached 100% first-token and 75% full-answer accuracy.
- Top-8 and top-16 sufficiency reached 50% full-answer accuracy in prefill
  mode and 75% in continuous mode.
- The continuous selected-head condition matched the full image-patch result.
- Necessity conditions returned to clean/base behavior.
- This suggests that the selected heads carry the relevant information and that
  continued substitution helps preserve longer answers.

### Currency

- The full image patch reached 100% first-token and 75% full-answer accuracy.
- Top-8 and top-16 sufficiency reached 75% full-answer accuracy in both
  prefill and continuous modes.
- Necessity conditions dropped to clean/base behavior.
- Continuous substitution provided no additional improvement on this sample.

### Calling Code

- The full image patch reached 100% first-token and full-answer accuracy.
- Top-8 and top-16 sufficiency reached 100% first-token accuracy but 0% full
  answer accuracy in prefill mode.
- The same selected heads reached 100% first-token and full-answer accuracy in
  continuous mode.
- Necessity conditions disrupted the transferred answer.
- This is the clearest evidence for the continuation hypothesis: calling codes
  are multi-token answers, and patching only the prompt position transfers the
  first token. Reapplying the selected head values at every generated position
  preserves the complete answer.

## Token Identification and Knockout Results

The identification phase freezes token rankings from the image-patched last
prompt query. Across attributes, the selected heads mostly read image tokens,
with a mixture of object and image-background positions. Some heads also read
newline, prompt, or other text/special tokens. The identified tokens are not
uniformly object-only, so the result supports an image-mediated path but does
not by itself localize the attribute to the object pixels alone.

The knockout phase removes selected attention edges while retaining the image
residual patch. Single-edge knockouts did not change the aggregate results for
any attribute in this sample. The joint cumulative results were:

| Attribute | Selected-token knockout | Matched random knockout | Interpretation |
|---|---:|---:|---|
| Language | 100% full match through top-8 | 100% | Strong redundancy or non-essential individual edges |
| Capital | 75% at top-1/2; 50% at top-4/8 | 75% throughout | Selected edges have a measurable cumulative effect |
| Currency | 75% through top-4; 50% at top-8 | 75% throughout | A weak cumulative effect appears only at top-8 |
| Calling code | 100% through top-8 | 100% | No measurable effect from these token-edge removals |

Here, the knockout percentages are full-answer accuracy on four rows. Because
the selected-token and random controls are very close for language, currency,
and calling code, these results do not establish that the identified edges are
uniquely causal. Capital is the only attribute with a clear separation from
the matched random control, although the sample is too small for a strong
claim.

## Overall Interpretation

The selected heads reproduce most or all of the full image-patch behavior in
this small sample. The strongest result is calling code, where continuous
substitution changes full-answer accuracy from 0% to 100% relative to prefill
substitution while preserving 100% first-token accuracy.

The results also support a causal role for the selected heads: necessity
conditions generally restore clean/base behavior or disrupt the transferred
answer, while matched random-head conditions are substantially weaker or
inconsistent.

## Limitations and Next Steps

- Each attribute currently has only 4 evaluated rows, so percentages can move
  substantially on the full sample.
- The identify and knockout outputs are committed, but they still contain only
  4 evaluated rows per attribute.
- Run the full decode evaluation before making quantitative claims:

```bash
OUT_ROOT=results/head_followups/flags_full bash scripts/run_head_followups.sh decode
```

After inspecting the full decode output, run token identification and then
knockout as described in [HEAD_EXPERIMENTS.md](HEAD_EXPERIMENTS.md).
