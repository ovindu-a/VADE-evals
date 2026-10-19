# Tier-3 results (FOLLOWUP_PLAN.md), Session 1, first batch

Results committed in `688bab8` (pod run of 2026-10-18; three parallel lanes at default batch
sizes, see `logs/tier3/GPU_USAGE_session1.md`):
- **In this batch:** P1, P2, P4, P5 and P10.
- **Still running when this was written:** P3, P6-P9, P11 and P12.

Every self-check that applies passed:
- **P1:** the cached and uncached evaluations agree.
- **P2:** the baseline arms reproduce the committed E9 run.
- **P10:** `readback_worst 0.0, identity bit-exact, liveness 2.625`.

Claim IDs (Z1, C2, ...) refer to CLAIM_CANDIDATES.md.

## Summary

| # | Question | Answer | Effect on written claims |
|---|---|---|---|
| P1 | Is DBM's "cause above the full-swap ceiling" (Z1) a bug? | **No.** The eval path is sound, and a learned blend genuinely beats the full swap. | Z1 becomes a finding. The "full swap = hard upper bound" rule is wrong at image positions. |
| P4 | Are E3's DBM masks just a 0.5 blend? | **No.** They beat gate=0.5 by 18-43 final points and are attribute-selective, though dense. | Overturns the E3 caveat. Supports claim 6 / C5 with trained masks. |
| P10 | Is the MLP lookup distributed? Is MLP 25 a bottleneck? Does MLP 27 decide? | Distributed and redundant over 23-26; MLP 25 is needed for calling_code (and currency); MLP 27 formats but does not decide. | Confirms C2, C3 and C4 on a new seed at n=128. |
| P2 | Is zeroing the movers worse than zeroing random heads in the same blocks? | **Yes:** −51 points vs ≤1.3. | Closes E9's missing control. Claim 1's necessity is now controlled. |
| P5 | Does head_cross replicate on new pairs? | **Yes:** every cell is within the CI. | Resolves Z4 (the first real replication). Confirms A6/A7. |

---

## P1. DBM identity-mask test (Z1)

`dbm/eval.py`, flags/language, layer 16, `flag_ring1` (24 tokens), all 3,036 test cause rows.
- An all-ones mask (`ones`) equals the full swap.
- An untrained mask (`half`) is gate 0.5 everywhere.

| mask | cause (matches source) | base kept |
|---|---|---|
| ones (full swap), cached source | **37.5%** | 57.5% |
| ones, `--no_source_cache` | 37.4% | 57.5% |
| half (0.5 everywhere), cached | 4.6% | 90.3% |
| half, `--no_source_cache` | 4.5% | 90.4% |
| trained DBM masks, same rows (committed runs) | **53.0-62.1%** | 34-43% |

Files: `results/tier3/dbm_identity/L16_{ones,half}{,_nocache}/`; trained runs under
`results/Qwen2.5-VL-7B-Instruct/flags/dbm/language/*/layer16_predictions_test_summary.md`.

**Reading:**
- **The eval path is sound.** Cached and uncached sources agree to 0.1pp, and the full swap reads
  close to the ceiling sweep's 31.2% (that figure was n=32; this is the full test set).
- **A trained per-dimension blend transfers ~24 points more than copying every dimension.** So
  "a mask selects a subset of what a full swap uses, hence the full swap is a hard upper bound on
  cause" (CLAUDE.md, README) does **not** hold at image positions.
  - The likely mechanism is E11's: a partial-image patch is a contest between the patched source
    region and the rest of the image, which still carries the base entity. A blend can win that
    contest more often than a full copy.
  - The upper-bound rule still holds where nothing competes. At `last_token` and in the late layers,
    every measurement so far agrees with it.
- **DBM numbers at image positions are usable again,** with this caveat.
- A uniform half blend does almost nothing (4.6%). DBM's gain comes from *which* dimensions are
  blended, not from blending as such.

## P4. Gate=0.5 control for E3's DBM masks at MLPs 23-26 (claim 6, C5)

`head_das.py --method dbm --site mlp_hidden --blocks 23-26 --skip_train`. An untrained pyvene mask
has gate sigmoid(0)=0.5 on every neuron at any temperature. Same 4,000 eval rows (seed 0) as E3.
E3's own masks were scored here for the first time
(`results/tier3/head_das_e3_scores/`).

VADE final score (cause / mean iso):

| target | gate fixed at 0.5 | trained E3 mask |
|---|---|---|
| language | **16.4** (12.3 / 20.4) | **59.3** (69.3 / 49.3) |
| capital | **37.0** (41.2 / 32.8) | **60.3** (68.0 / 52.6) |
| currency | **36.9** (33.5 / 40.4) | **59.4** (49.4 / 69.4) |
| calling_code | **70.0** (93.0 / 46.9) | **88.2** (94.0 / 82.5) |

How often a NON-target calling_code question gets the SOURCE's code:

| target | gate 0.5 | trained |
|---|---|---|
| language | 93.6% | 11.8% |
| capital | 93.4% | 13.8% |
| currency | 94.1% | 10.4% |

Files: `results/tier3/head_das_gate0.5/<attr>_mlp23-26/`, `results/tier3/head_das_e3_scores/<attr>/`.

**Reading:**
- **The trained masks learned something.** They beat the half blend by 18-43 final points.
- **The masks are attribute-selective.**
  - A half swap at MLPs 23-26 moves calling_code almost regardless of what was asked: 93-94% of
    non-target calling_code answers flip. That fits MLP 25 being calling_code's lookup (P10).
  - The trained masks cut that leak to 10-14% while still moving the target.
  - This is the trained-mask counterpart of E3's readout edit. Attributes separate at the MLP
    neurons.
- **The masks are still not sparse.** `n_selected` is 0. The selectivity lives in graded gates:
  logits within ±0.04 at temperature 0.01, i.e. gates spread over 0.02-0.98. Describe them as
  dense, graded, attribute-selective masks. Mask-overlap statistics on `n_selected` remain
  meaningless.
- The E3 finals (59.3 / 60.3 / 59.4 / 88.2) match the README figures, which were recomputed off-pod.

## P10. MLPs 23-26: leave one block out, singles, third-country values (C2, C3, C4)

`head_severed.py --n_pairs 128 --seed 1`: new pairs, twice E2's n, `common10` installed
continuously.
- **base** freezes a component at the base run's value.
- **@mean** uses the leave-one-out mean of other countries ("needed?").
- **@other** uses a third country's value ("does it write the answer?").

Columns are text-matched unless marked 1st (first answer token). File:
`results/tier3/severed_loo_n128_s1/summary.json`; log `logs/tier3/p10_severed_20261018_055521.log`.

**Reproduction of E2:**

| | capital | currency | language | calling_code |
|---|---|---|---|---|
| movers alone, source 1st | 96.9 | 79.5 | 100 | 99.0 |
| MLP 23-26 @base, source | 0.0 | 0.0 | 3.1 | 0.0 |
| MLP 23-26 @other, third country's answer | 100 | 66.4 | 89.1 | 100 |

(E2 at n=64: 97/78/100/98, 0/0/0/0, 95/62/95/100.)

**C2. Leave one block free (freeze the other three of 23-26 at @mean), source answer:**

| free block | capital | currency | language | calling_code |
|---|---|---|---|---|
| 23 (24-26 frozen) | 9.4 | 8.6 | 43.0 | 0.8 |
| 24 (23,25-26 frozen) | 10.2 | 7.8 | 46.1 | 0.0 |
| 25 (23-24,26 frozen) | 3.9 | 6.2 | 50.8 | 7.8 |
| 26 (23-25 frozen) | 8.6 | 1.6 | 43.0 | 0.8 |
| none (23-26 frozen) | 0.0 | 3.9 | 25.8 | 1.6 |

At @other, every 3-block span still writes the third country: capital 78-94, currency 57-60,
language 73-85, calling_code 99-100.

**C3 and C4. Single blocks (source answer, text):**

| arm | capital | currency | language | calling_code |
|---|---|---|---|---|
| MLP 25 @mean | 82.0 | **21.9** | 86.7 | **0.8** |
| MLP 25 @other, third country's answer | 6.2 | 19.5 | 2.3 | **93.0** |
| MLP 27 @base, source 1st / text | 86.7 / 43.8 | 50.4 / 14.1 | 86.3 / 87.5 | 95.0 / 57.8 |
| MLP 27 @mean, source text | 73.4 | 48.4 | 93.8 | 93.8 |
| MLP 27 @other, third country 1st / text | 0.0 / 0.0 | 1.6 / 1.6 | 0.0 / 0.0 | 0.0 / 7.0 |

**Reading:**
- **The lookup is distributed and redundant over 23-26 (C2 confirmed).**
  - One free block cannot rebuild the answer: capital, currency and calling_code stay at 0-10%.
  - Freezing one block (R16.5, and MLP 25 here) mostly does not hurt either, except as below.
  - So no single block does the lookup, and no single block is required. Language is the most
    robust: one free block still gives 43-51%.
- **MLP 25 is a single-block bottleneck for calling_code, and largely for currency (C3 confirmed
  at n=128).** Frozen alone at the mean, calling_code drops to 0.8% and currency to 22%, while
  capital (82%) and language (87%) survive. Frozen alone at a third country's values, it writes
  that country's calling code 93% of the time.
- **MLP 27 renders the answer but does not choose it (C4 confirmed).**
  - Frozen at a third country's values, it produces that country's answer at most 7% of the time;
    the source's first token still wins 86-95%.
  - Frozen at base, the first token transfers but the continuation breaks (capital 87% → 44% in
    full text, currency 50% → 14%).
  - This matches DLA (MLP 27 is the largest direct writer) without being where the decision is made.

## P2. Matched null for zeroing the movers (E9, claim 1)

`vrh_ablation.py --n_random_movers 5`: five block-matched random 10-head sets in blocks 21-23
(2/4/4 heads, never a mover), on the same 672 generations as E9. File:
`results/tier3/vrh_mover_random/summary.json`.

| arm | accuracy | capital | currency | language | calling_code | mover name rank |
|---|---|---|---|---|---|---|
| clean | 88.7% | 87.5 | 82.7 | 93.5 | 91.1 | 1.62 |
| zero `common10` | **38.1%** | 37.5 | 26.8 | 75.6 | 12.5 | – |
| zero random-10 (×5) | **87.4-89.6%** | 86.9-87.5 | 79.8-85.1 | 92.3-94.0 | 89.9-92.9 | 1.60-1.69 |
| zero `park_vrh` (check) | 81.5% | 82.1 | 61.3 | 90.5 | 92.3 | 2.49 |

The committed E9 run gives 88.8 / 37.9 / 81.7 for clean / `common10` / `park_vrh`, so the
reproduction holds to within a row or two.

**Reading:**
- Zeroing 10 random heads from the same blocks costs at most 1.3 points, and the movers' name rank
  is unchanged. Zeroing `common10` costs 51.
- "The movers are the bottleneck" now has its matched control.
- Language survives zeroing the movers (75.6%) and calling_code does not (12.5%), as in E9 and E3(b).

## P5. head_cross on 64 new pairs (A6, A7, Z4)

`head_cross.py --seed 1 --n_pairs 64`: no pair overlaps seed 0's first 64 (or 256); 46 bases, 49
sources. Files: `results/tier3/head_cross_seed1/cross_n64_seed1.jsonl`,
`results/tier3/cpu_followups_seed1/`.

| cell | seed 1 | country-cluster CI | seed 0 |
|---|---|---|---|
| flag, same question (donor flag's answer) | 81.2% | ±4.1-4.6 | 81.2% |
| donor asked a different question | 65.2% | ±4.8-5.2 | 65.1% |
| question only (base answer kept) | 78.8% | ±3.8-4.2 | 83.2% |
| self-patch (base answer kept) | 91.8% | ±3.0-3.8 | 93.0% |

Per attribute, flag cell: capital 98, currency 44, language 88, calling_code 95. In the
different-question cell, calling_code is 33%.

**Reading:**
- **This is the first independent replication** of R11. The n=16 and n=256 runs were subsets of the
  seed-0 pairs (Z4). Every cell is within its interval.
- The question-only cell drops 4.4pp, at the edge of the interval.
- Two weak spots repeat on new pairs:
  - currency transfers poorly through the heads (44%);
  - calling_code fails when the donor was asked a different question (33%), the matched-step replay
    artefact (M3).

---

## Docs to update when Session 1 finishes

- **CLAIM_CANDIDATES.md:**
  - Z1 becomes a finding, not a bug.
  - E3 / claim 6: the DBM masks are dense but selective.
  - C2, C3, C4: confirmed (P10).
  - E3 / claim 1: the null exists (P2).
  - Z4: resolved (P5).
- **README** (E3, E9) and **CLAUDE.md:** remove "the full swap is a hard upper bound on cause" as a
  general rule. It does not hold at image positions (P1).
