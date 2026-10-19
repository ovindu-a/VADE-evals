# Reconciling router8 (8 heads) with P6 (top-32 needed)

CPU-only re-reading of existing artefacts. No model was run.

## Verdict

**It is a ranking shift caused by adding blocks 23-27. It is not a blocks-15-22 artefact, and it is not a scoring or normalization difference.** P6 never tested router8. Its "top-8" is a different set of heads: 7 of its 8 are late heads in blocks 24-27, and only one (21.0) is from router8.

**Safe to write:** "Eight heads in blocks 19-21 (router8) suffice to transfer the queried attribute at the first answer token: 87.0% vs a 95.4% ceiling, and 87.3% on 42 countries held out from head selection. Restoring them removes 96% of the effect. Eight random heads do nothing either way." Report the first-token column, and pair it with the caveats below.

**Do not write:** "32 heads are needed." That number comes from a ranking metric inflated by late-block norms (see 2). P6's top-32 works only because it contains all of router8.

## 1. The two runs are the same experiment apart from `--blocks`

| | original R1 | P6 |
|---|---|---|
| file | `logs/attr_head_trace/flags/attr_head_trace_patch15_blocks15-22_all_text_n84.json` | `results/tier3/attr_heads/patch15_blocks15-27_n84.json` (log `logs/tier3/p6_blocks15-27_20261018_060611.log`) |
| blocks | 15-22 (224 heads) | 15-27 (364 heads) |
| items / rows | 84 countries, 1008 rows | the same 84 countries in the same order, 1008 rows |
| patch_layer / positions / rank_by / seed | 15 / all_text / delta_resid / 0 | identical |
| clean first / full | 95.5 / 92.9% | 95.5 / 92.6% |
| question patch only (first) | 95.4% | 95.3% |
| phase 3 all-heads ceiling (first / full) | 95.4 / 76.6% | 95.3 / 76.3% |
| normalization | "% of this arm's ceiling" in phase 3; "removed % of effect" in phase 2 | same code, same definitions |

The scoring columns match: both runs headline `first`, and both runs' `full` columns are similarly lower (76.6% vs 76.3% at all heads). The ceilings match, so the "% of ceiling" normalization is not what differs.

A notable side result: installing all 224 heads from blocks 15-22 already reaches the full first-token ceiling (95.4% vs P6's 95.3% with 364 heads). Blocks 23-27 add nothing to first-token transfer.

## 2. Phase-1 values are identical; only the ranking changes

- Across the 224 shared heads, per-head `delta_resid` agrees to a median relative difference of 0.26% (max 1.7%). The Spearman correlation is 0.99997, and the top-16 overlap is 16/16. Per-block totals for blocks 15-22 agree to within 0.2: 19.8 / 38.2 / 46.4 / 50.3 / 75.4 / 101.4 / 133.7 / 57.2.
- **P6's ranking, restricted to blocks ≤22, has exactly router8 as its top 8.**
- What changes is that blocks 23-27 enter the ranking with larger `delta_resid`. Their mean is 5.18 per head, against 2.33 in blocks 15-22, and block totals are 194 for block 26 and 230 for block 27.
- P6 top-8 = 26.24, 26.22, **21.0**, 27.13, 27.10, 24.18, 27.12, 27.11. Only one is a router8 head.
- Router8's ranks in P6: 21.0 → 3, 20.17 → 10, 20.18 → 14, 19.21 → 18, 21.25 → 19, 21.4 → 20, 21.5 → 22, 19.27 → 25.
- `delta_resid` measures how much a head's write changes when the question is swapped. A late head reading an already-changed residual changes a lot without causing anything. In a downstream window the ranking therefore mixes consequences with causes.

## 3. P6's curves are router8's curves with inert late heads mixed in

The heads P6 ranks from blocks ≤22 are always a prefix of the original ranking. That holds for the top-8, top-16 and top-32. Each P6 arm can therefore be predicted from the original curve, using only its early heads:

| P6 arm | early heads it contains | P6 sufficiency (first) | original curve at that many early heads |
|---|---|---|---|
| top-8 | 1 (21.0) | 0.5% | k=1: 0.4% |
| top-16 | 3 (21.0, 20.17, 20.18) | 39.5% | between k=2 (4.6%) and k=4 (59.3%) |
| top-32 | 12 (all of router8 + 19.23, 21.19, 21.21, 20.22) | 94.4% | k=8: 87.0%; k=16: 94.2% |

Necessity behaves the same way:

| arm | first token | effect removed |
|---|---|---|
| P6 restore top-8 | 93.6% | 1.9% |
| P6 restore top-16 | 53.6% | 43.8% |
| P6 restore top-32 | 0.6% | 99.4% |
| original restore top-4 | 31.9% | — |
| original restore top-8 | 3.8% | — |

So 20 late heads in P6's top-32 contribute essentially nothing. P6's result is "router8 is needed and suffices," reached at k=32 only because of dilution.

P6's own top-8 shuffled-donor control agrees: own = 0.3%, donor = 9.0%. The heads P6 ranks top-8 carry no row-specific content.

Phase 3 installs patched values only into the selected heads and leaves every other head free to recompute (`methods/head_trace.py:241`, `head_patches`). Installing router8 is therefore the same intervention whether `--blocks` ends at 22 or at 27. Widening the window cannot change router8's number, only which heads get called "top-8".

## 4. Held-out replication already exists, inside P6

`logs/tier3/p6_sel_A_20261018_060611.log` and `logs/tier3/p6_eval_B_20261018_060611.log` (`results/tier3/attr_heads/sel_A.json`, `eval_B_rankA.json`) use disjoint halves of 42 + 42 countries that together make up the 84.

- Selection on A gives a top-8 identical to router8.
- Evaluated on B with A's ranking:

| arm on B | first token |
|---|---|
| sufficiency, k=8 | **87.3%** (92.6% of ceiling) |
| sufficiency, k=16 | 94.2% |
| random-8 | 0.4% |
| restore top-8 | 94.4 → 3.8% (96.0% removed) |
| shuffled donor, top-8 | own 0.4%, donor 90.3% |

This closes B1's "selected and evaluated on the same items" gap.

P19 (`logs/tier3/p19_donor_*`, blocks 15-22, per donor attribute) finds top-8 sets sharing 7-8 of 8 heads with router8. Restoring them removes 90.0-99.1% of the effect.

## 5. Caveats to keep in the paper

1. **Report first-token numbers.**

   | | first token | full answer |
   |---|---|---|
   | router8 sufficiency | 87.0% | 58.3% |
   | all-heads ceiling | 95.4% | 76.6% |

   This gap is the known `last_token` limit: a read-column install cannot steer past the first answer token. It applies equally to all heads, so it is not specific to router8.
2. **Necessity at the full 15-27 window was not run with router8 itself.** The only 15-27 evidence for router8 is indirect: P6 top-32 ⊃ router8, and restoring it removes 99.4%. P6 top-16, which holds 3/8 of router8, removes 43.8%. If a reviewer may ask, the clean confirmation is one GPU run of `attr_head_trace --blocks 15-27` with the ranking taken from `sel_A.json` via `--rank_from` (or with a block-restricted ranking).
3. **Donor-specific weakness.** In P19, router8-like top-8 sufficiency is 98.8 / 99.6 / 87.3% for donor capital / calling_code / currency, but only **40.5%** (44% of ceiling) for donor = language. "Eight heads suffice" is weakest when language is the asked attribute.
4. **Do not use `delta_resid` rankings over windows that include blocks after the handoff (≥22).** Late-block norms dominate them. Any future "top-k" claim should rank within the router window or use a sufficiency-based ranking.
