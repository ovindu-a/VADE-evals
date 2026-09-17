# Where the requested attribute is read: findings from the attribute-switch sweep

Analysis of the completed `methods/attribute_switch_sweep.py` runs under
`results/attribute_switch/`. Regenerate every table here with

```bash
python3 methods/attribute_switch_report.py results/attribute_switch/text_full_cached_b64 \
    results/attribute_switch/attention_18_24_64_cached_b96 --json_out /tmp/switch_tables.json
```

The design and run commands live in [ATTRIBUTE_SWITCH_SWEEP.md](ATTRIBUTE_SWITCH_SWEEP.md);
this file is only what the numbers say.

**Indexing.** `--layer L` is the residual stream AFTER decoder block `L-1`, and a
sublayer site at `L` is block `L-1`'s sublayer (`methods/common/sites.py`). Every
table below prints both. Head indices are 0-based BLOCK numbers.

## Runs analysed

| run | status | eligible rows | what it covers |
|---|---|---|---|
| `text_full_cached_b64` | complete, 96 rows | 84/96 | 28 layers x {residual, attention, mlp, joint} x 3 scopes x spans 1-8 |
| `attention_18_24_64_cached_b96` | **partial, 147/768 rows** | 118/147 | 64 countries, attention only, L18-24, spans 1-6 |
| `text_continuous_cached_b2` | **partial, 71/96 rows** | 58/71 | `prefill` vs `continuous`, L20-25 |
| `text_baselines` | complete | — | clean controlled-prompt accuracy |

The two partial runs resume on an identical command.

## Before anything else: the metric

**Use `donor_first`, not `donor_full`.** A `last_token` intervention is
structurally unable to steer an answer past its first token — everything after it
is generated against a KV cache that still belongs to the base, the effect
`methods/ndm/swap_trace.py` documents for `calling_code`. `text_continuous_cached_b2`
measures exactly this by re-running the same arms with the patch live on every
generated position:

| site @ layer | prefill first / full | continuous first / full |
|---|---|---|
| residual L21 | 100.0 / 69.0 | 100.0 / **100.0** |
| residual L22-25 | 100.0 / 63.8-65.5 | 100.0 / **100.0** |
| attention_blocks:4 L23 | 100.0 / 72.4 | 100.0 / **100.0** |
| blocks:4 L24 | 100.0 / 69.0 | 100.0 / **100.0** |

The `full` deficit is answer length, not a failed intervention. `first` is the
number that compares across sites, and every table below reports it.

## The sweep is sound

Clean accuracy on the controlled prompts (8 countries): `capital` and `language`
1.000 full, `currency` and `calling_code` 0.875; **first-token accuracy is 1.000
for all four**. 84 of 96 rows are eligible (both clean answers correct).

Controls, pooled over all 3,222 arms:

| control | donor_first | donor_full | base_kept | expected |
|---|---|---|---|---|
| `self` | 0.0238 | 0.0119 | **0.9998** | exact no-op |
| `paraphrase` | 0.0238 | 0.0119 | **0.9977** | near no-op |

The 2.4% donor floor is 2/84 — chance. Nothing leaks.

## 1. The handoff is in blocks 17-21

Patching the question tokens (`earlier_text`) stops working at exactly the depth
where patching the readout position (`last_token`) starts working. The crossing
IS the read.

| block | question residual | last-token residual | last-token attention | last-token MLP |
|---:|---:|---:|---:|---:|
| ≤15 | 100.0 | 2.4 | 2.4 | 2.4 |
| 16 | 98.8 | 2.4 | 2.4 | 2.4 |
| 17 | 78.6 | 13.1 | 2.4 | 2.4 |
| 18 | 28.6 | 50.0 | 9.5 | 2.4 |
| **19** | 13.1 | 65.5 | **19.0** | 2.4 |
| **20** | 2.4 | 97.6 | **14.3** | 3.6 |
| **21** | 2.4 | 100.0 | 9.5 | 8.3 |
| 22+ | 2.4 | 100.0 | 2.4 | ≤9.5 |

Three things follow:

1. **The MLP column is at floor throughout the crossover.** The transfer is
   entirely attention-mediated, as it must be — attention is the only
   cross-position operation in a transformer.
2. **Single-block attention peaks at block 19** (19.0%), with 20 and 21 next.
   19.0% is small because the mechanism is redundant, not because it is weak —
   see the span table.
3. **Below block 16 there is nothing to find at the readout position.** Editing
   the question still works there because the question tokens have not yet been
   read, not because those blocks route anything.

## 2. Blocks 19-21 are the operative set; block 19 is the keystone

Attention spans at `last_token`, read by which blocks they cover rather than by
(layer, width). Both runs agree.

| covered blocks | width | donor_first |
|---|---:|---:|
| 19..21 | 3 | **96.6 - 97.6** |
| 18..21 | 4 | **97.5 - 100.0** |
| 19..22 | 4 | **97.5 - 100.0** |
| 19..26 | 8 | 100.0 |
| 18..20 (no 21) | 3 | 81.4 - 86.9 |
| 16..20 (no 21) | 5 | 91.5 - 94.0 |
| 13..20 (no 21) | 8 | 97.6 |
| 20..22 (no 19) | 3 | 68.6 - 75.0 |
| 20..23 (no 19) | 4 | 66.9 - 75.0 |
| **20..27 (no 19)** | **8** | **79.8** |
| 21..23 (no 19, 20) | 3 | 11.9 - 18.6 |
| 16..18 | 3 | 43.2 - 44.0 |

The asymmetry is the finding. Dropping block 21 costs little as long as enough
earlier blocks are present (13..20 still reads 97.6%). **Dropping block 19 caps
the result at ~80% no matter how many later blocks are added** — eight blocks,
20..27, reach only 79.8%. Blocks 21 and later are substitutable; block 19 is not.

Blocks 16-18 carry a real ~45% on their own: the representation is being built
there before it is read.

## 3. No single block works for all four attributes

Single-block attention at `last_token`, by donor attribute (64-country run,
n≈28-32 per cell — indicative, not tight):

| donor attribute | block 18 | block 19 | block 20 | block 21 |
|---|---:|---:|---:|---:|
| `calling_code` | 32.1 | **42.9** | 25.0 | 0.0 |
| `currency` | 0.0 | **36.7** | 3.3 | **36.7** |
| `capital` | 0.0 | 3.6 | **28.6** | 3.6 |
| `language` | 3.1 | 3.1 | 3.1 | 3.1 |

`language` sits at chance for **every** single block yet reaches 87.5% across
blocks 19-21 — fully distributed. Any per-block head hunt on `language` will read
as dead when it is not. This is the same redundancy that makes single-head
knockout uninformative, one level up.

At the 3-block span (blocks 19-21): `capital` 100.0 first / 96.4 full,
`currency` 100.0 / 76.7, `language` 87.5 / 78.1, `calling_code` 100.0 / 50.0 —
the `full` spread being answer length again.

## 4. Consequences for head tracing

- **Trace blocks 15-22, read at the last token.** 19-21 is the core, 16-18 is
  where the representation is assembled, and the span table says both halves
  carry signal.
- **Patch the question tokens at `--patch_layer` ≤ 16**, where the question-side
  residual still reads 98.8-100%. At 18 it is already down to 28.6% and there is
  little left downstream to trace.
- **Sufficiency is the load-bearing phase.** Given how substitutable blocks 21+
  are, a cumulative-knockout (necessity) curve will look flat for genuinely
  load-bearing heads. `head_trace.py`'s phase 3, and its phase-4 shuffled-donor
  control, are what can localize here.
- **Do not trust generated text for connectivity checks** — same reason as
  `verify_sites`: a full swap can move logits by 0.25 and leave the generation
  identical.
- **This is a different circuit from the one `head_trace.py` traces.** That
  script patches an *image* position set and swaps the entity. This sweep
  localizes the **question -> readout** path, which selects WHICH attribute is
  reported for a fixed entity. The window here (19-21) and the image->text
  handoff window are not the same claim and should not be quoted as one.

## 5. Attention moves it; the late MLPs write it out

The `mlp` column above is at floor through the crossover, and the earlier version
of this document stopped there. Comparing the two SPAN families at matched width
says something much stronger. `attention_blocks:N` patches N blocks' attention
contributions; `blocks:N` patches the same N blocks' attention AND MLP
contributions, so the difference is what the MLP outputs add:

| covered blocks | attention only | attention + MLP | delta |
|---|---:|---:|---:|
| 16..18 | 44.0 | 40.5 | −3.6 |
| 18..20 | 86.9 | 83.3 | −3.6 |
| **19..21** | **97.6** | **100.0** | +2.4 |
| 20..22 | 75.0 | 97.6 | +22.6 |
| 21..23 | 11.9 | 97.6 | **+85.7** |
| 22..24 | 2.4 | 67.9 | +65.5 |
| 22..26 | 2.4 | 98.8 | +96.4 |
| 25..27 | 1.2 | 76.2 | +75.0 |

Below block 21 the MLPs add nothing — the delta is zero or slightly negative.
From block 21 on it is everything: attention-only spans are dead at 2.4% while
the same blocks with their MLPs read 68-99%.

This is the standard division of labour, and it sharpens the head advice rather
than changing it. **Attention moves the attribute selection to the readout
position in blocks 19-21; the MLPs of blocks 21-27 turn it into the emitted
token.** The late blocks are a write-out path reachable only through MLPs, so
there are no attribute-routing heads to find there — exactly as the
attention-only column says.

Two caveats. The sweep has no `mlp_blocks:N` family, so this is a subtraction of
two arms rather than an isolation; a one-command follow-up patching MLP outputs
alone would settle it. And single-block `joint` at block 21 reads 45.2% where its
attention (9.5%) and MLP (8.3%) each read under 10% — the two sublayers are
strongly super-additive there, which a subtraction cannot represent.

## 6. The scopes are super-additive exactly in the handoff

`all_text` patches the question columns and the readout column together. Outside
the handoff it is simply whichever of the two works; inside it, it is much more
than both.

| block | question | last token | all_text | max of the two | excess |
|---:|---:|---:|---:|---:|---:|
| 16 | 98.8 | 2.4 | 100.0 | 98.8 | +1.2 |
| **17** | 78.6 | 13.1 | 100.0 | 78.6 | **+21.4** |
| **18** | 28.6 | 50.0 | 100.0 | 50.0 | **+50.0** |
| **19** | 13.1 | 65.5 | 100.0 | 65.5 | **+34.5** |
| 20 | 2.4 | 97.6 | 100.0 | 97.6 | +2.4 |
| 21+ | 2.4 | 100.0 | 100.0 | 100.0 | +0.0 |

The excess is zero everywhere except blocks 17-19, where it reaches +50 points.
That is what "in transit" looks like: the attribute identity is split across the
question columns and the readout column, and neither half alone is sufficient.
It localizes the handoff from a completely different measurement than the span
table, and to the same blocks.

## 7. `language` is hard to switch TO; the rest of the asymmetry is weaker

At a saturated site every cell reads ~100%, so the directed matrix is only
readable where there is headroom. At the 2-block attention span over blocks
19-20 (72.6% pooled, 84 observations over 8 countries, **6-8 per cell**):

| from \\ to | capital | currency | language | calling_code |
|---|---:|---:|---:|---:|
| capital | — | 5/7 | 6/8 | 7/7 |
| currency | 6/7 | — | **0/7** | 3/6 |
| language | 8/8 | 6/7 | — | 7/7 |
| calling_code | 7/7 | 4/6 | **2/7** | — |

That operating point was chosen AFTER seeing which sites had spread, so the
matrix alone proves nothing. Three tests decide what survives.

**1. Is there an asymmetry at all?** Within-country permutation (outcomes
shuffled inside each country, so each country's own hit rate is preserved and
the clustering is respected), statistic = spread of the four margins:

| margin | observed | null 95th pct | p |
|---|---:|---:|---:|
| FROM | 0.505 | 0.355 | **0.0007** |
| TO | 0.591 | 0.355 | **<0.0001** |

**2. Does it replicate?** The same arm in `attention_18_24_64_cached_b96`
(118 observations, 13 different countries):

| | capital | currency | language | calling_code | |
|---|---:|---:|---:|---:|---|
| FROM (8-country / 64-country) | 82 / 77 | 45 / 31 | **95 / 94** | 65 / 57 | rank order identical, r = 1.000 |
| TO (8-country / 64-country) | 95 / 89 | 75 / 80 | **36 / 19** | 85 / 79 | r = 0.974, top two swap |

**3. Does it hold away from the chosen site?** Over all 43 `last_token/switch`
arms with headroom (pooled rate 25-90%):

| claim | holds at | mean r vs reference |
|---|---:|---:|
| FROM ordering | 40/43 above +0.5, 1 negative | **+0.846** |
| TO ordering | 31/43 above +0.5, 2 negative | +0.657 |
| `language` is the hardest to switch TO | **37/43 (86%)** | mean gap **+44.2 pp** |
| `language` is the easiest to switch AWAY from | 32/43 (74%) | |
| `currency` is the hardest to switch away from | 27/43 (63%) | |

### What that licenses

- **Established**: `language` is the hardest attribute to switch TO, by about 44
  points. Replicated on different countries, stable at 86% of operating points,
  and far larger than the sample can manufacture.
- **Established**: an overall FROM/TO asymmetry exists (p <= 0.0007, clustered).
- **Supported**: the FROM ordering `language > capital > calling_code > currency`.
  It replicates with identical rank order, but the gaps between the middle two
  are within noise at many operating points.
- **NOT established**: the ordering of capital/currency/calling_code as switch-TO
  targets. Across the 43 operating points it comes out four different ways
  (17/12/8/4), and the two country samples disagree. Do not quote those three
  against each other.

The `language` result is the same fact as section 3's per-attribute table seen
from another angle: `language` has no single-block peak and needs blocks 19-21
together, so its selection is the most distributed of the four and a partial
intervention reaches it last.

**Sample limit.** 84 observations, 8 countries, 6-8 per cell and ~21 per margin.
That supports a 44-point effect and does not support 10-point differences between
neighbours.

## 8. Reproducibility of individual rows, and two data-hygiene notes

`text_full` (serial) and `text_full_cached_b64` (cached) overlap on 16 rows =
48,896 arm-rows, which bounds how far an individual number can be trusted:

- arms both runs execute on the **same** serial path (`clean`, `donor_clean`):
  **32/32 identical**. The environment is deterministic; there is no run-to-run
  noise to explain anything else.
- `self` control arms, which are provable no-ops: **98.9% identical**. That is
  the pure bf16 batch-geometry floor.
- `switch` arms: **88.2%** identical token sequences, **96.9%** identical scores.

So the engines agree on pooled rates (r = 0.99 across 3,222 arms even at n=14 vs
n=84) but an individual arm-row is reproducible only to ~3%. Interventions push
the model near a decision boundary, where a different batch geometry flips the
greedy token; 42.6% of the disagreements begin at the very first token, which is
the prefill, not decode drift. Quote pooled cells, never single rows.

Two things to know about the runs themselves:

- **`text_full` covered only 16 of 96 rows** and carries a different
  `implementation_sha256`, so it is not an independent replication of
  `text_full_cached_b64` and should not be quoted alongside it.
- **`text_continuous_cached_b2` ran on transformers 5.17.0**, every other run on
  5.14.1. Its internal prefill-vs-continuous comparison is sound (both arms share
  the version); comparing its absolute rates against the other runs is not.

## 9. A structural property of the prompts, worth knowing

Across all 96 rows the base and donor prompts are token-identical except for
**one token at index 207** (paraphrase differs at 205), out of 219 tokens, with
the image occupying tokens 15-158. Under causal attention that makes every patch
position below 207 a provable no-op.

So the `earlier_text` scope's 67 positions are effectively 9, and `all_text`'s are
10. That does not change any result above — it is what the numbers already
measure — but it means the scopes differ less than their names suggest, and it is
the fact the cached engine's prefix reuse exploits.
