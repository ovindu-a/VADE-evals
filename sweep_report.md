# Attribute-switch sweeps: aggregate results and analysis
Latest local commit: `cf37dcb` (continuous attribute switching sweep). Raw records recomputed; source files unchanged. All percentages below are percentage points on a 0–100 scale.

## Method and interpretation
- Model: Qwen/Qwen2.5-VL-7B-Instruct. Same country image, different requested attribute; attributes: capital, currency, language, calling code. One controlled prompt template, seed 0.
- An arm is one scope/site/layer/mode/control combination. Only question pairs present in every arm of an experiment are included. No duplicate row/arm records were found.
- Main success metric: donor gold-token prefix match, restricted to pairs where clean base and donor outputs both pass that scorer. This is the existing “full_match” metric, not strict exact-answer accuracy.
- Strict diagnostic: remove one terminal EOS token, require exact generated-token equality to gold, and restrict to pairs where both baselines satisfy that stricter metric. No aliases, whitespace normalization, or semantic equivalence are added. Its cohort therefore differs from the main metric.
- Means average all tested endpoint layers within the stated family; because each arm uses the same pairs, this is also the pooled arm-by-pair success rate. Broad-family means are descriptive, not independent samples or intrinsic model accuracies.
- Best means best observed endpoint, with the earliest endpoint displayed for ties. It is selected on the same evaluation sample.
- A span of width W ending at layer L includes L−W+1 through L, using one-based labels. Attention+MLP spans replace both outputs across those layers, not the entire residual stream. These experiments do not isolate individual heads.
- Earlier text excludes the final prompt token; all text includes it. Continuous mode also patches generated positions; the earlier-text-only scope has no continuous arms.
- Original and donor prefix matches can overlap, so their percentages plus neither may exceed 100%.

## Experiment coverage
| Experiment | Raw records | Arms incl. controls | Common/planned pairs | Countries | Prefix-eligible | Strict-eligible | Clean prefix % | Clean exact % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text_full_cached_b64 | 309504 | 3224 | 96/96 | 8 complete; 8 represented | 84 | 72 | 93.8 | 87.5 |
| attention_18_24_64_cached_b96 | 37182 | 254 | 146/768 | 12 complete; 13 represented | 118 | 100 | 89.7 | 83.6 |
| text_continuous_cached_b2 | 51246 | 722 | 70/96 | 5 complete; 6 represented | 58 | 52 | 91.4 | 87.1 |

The attention sweep has 146–147 records per arm; the continuous sweep has 70–71. Their final partial question pair is excluded. The full sweep is complete. The continuous directory says b2, but its config specifies batch_size=96. All configs have verify_cached=false.

## Experiment 1 — full prefill sweep
Layers 1–28; three position scopes; residual, attention, MLP, attention+MLP, and multi-layer spans. 84 eligible pairs from eight countries.

| Site | Earlier mean | Earlier best | Final-token mean | Final-token best | All-text mean | All-text best |
| --- | --- | --- | --- | --- | --- | --- |
| Residual | 64.8 | 100.0 @L1 | 23.7 | 72.6 @L22 | 90.4 | 100.0 @L1 |
| Attention | 5.0 | 98.8 @L1 | 2.0 | 9.5 @L20 | 6.5 | 98.8 @L1 |
| MLP | 5.8 | 50.0 @L1 | 1.6 | 4.8 @L23 | 6.3 | 50.0 @L1 |
| Attention+MLP ×1 | 8.6 | 100.0 @L1 | 3.3 | 26.2 @L22 | 11.6 | 100.0 @L1 |
| Attention+MLP ×2 | 15.5 | 98.8 @L2 | 11.9 | 63.1 @L22 | 29.1 | 100.0 @L2 |
| Attention+MLP ×3 | 22.9 | 100.0 @L3 | 19.7 | 75.0 @L22 | 44.6 | 100.0 @L3 |
| Attention+MLP ×4 | 33.0 | 100.0 @L4 | 24.7 | 77.4 @L22 | 58.6 | 100.0 @L4 |
| Attention+MLP ×5 | 43.7 | 100.0 @L5 | 28.0 | 76.2 @L22 | 70.2 | 100.0 @L5 |
| Attention+MLP ×8 | 71.1 | 100.0 @L8 | 34.4 | 84.5 @L22 | 94.2 | 100.0 @L8 |
| Attention ×2 | 7.6 | 100.0 @L2 | 5.6 | 50.0 @L21 | 14.2 | 98.8 @L2 |
| Attention ×3 | 12.1 | 100.0 @L3 | 10.1 | 75.0 @L22 | 24.1 | 100.0 @L3 |
| Attention ×4 | 16.7 | 100.0 @L4 | 14.7 | 77.4 @L22 | 32.0 | 100.0 @L4 |
| Attention ×5 | 21.7 | 100.0 @L5 | 19.3 | 77.4 @L23 | 40.1 | 100.0 @L5 |
| Attention ×8 | 38.0 | 100.0 @L8 | 34.1 | 84.5 @L22 | 61.9 | 100.0 @L8 |

### Residual layer profile
| Layer | Earlier text | Final token | All text |
| --- | --- | --- | --- |
| 1 | 100.0 | 1.2 | 100.0 |
| 2 | 100.0 | 1.2 | 100.0 |
| 3 | 98.8 | 1.2 | 100.0 |
| 4 | 100.0 | 1.2 | 100.0 |
| 5 | 100.0 | 1.2 | 100.0 |
| 6 | 100.0 | 1.2 | 100.0 |
| 7 | 100.0 | 1.2 | 100.0 |
| 8 | 100.0 | 1.2 | 100.0 |
| 9 | 100.0 | 1.2 | 100.0 |
| 10 | 98.8 | 1.2 | 98.8 |
| 11 | 98.8 | 1.2 | 98.8 |
| 12 | 100.0 | 1.2 | 98.8 |
| 13 | 100.0 | 1.2 | 98.8 |
| 14 | 100.0 | 1.2 | 98.8 |
| 15 | 100.0 | 1.2 | 98.8 |
| 16 | 98.8 | 1.2 | 97.6 |
| 17 | 97.6 | 1.2 | 97.6 |
| 18 | 75.0 | 4.8 | 97.6 |
| 19 | 26.2 | 31.0 | 91.7 |
| 20 | 11.9 | 48.8 | 90.5 |
| 21 | 1.2 | 70.2 | 78.6 |
| 22 | 1.2 | 72.6 | 71.4 |
| 23 | 1.2 | 72.6 | 71.4 |
| 24 | 1.2 | 72.6 | 71.4 |
| 25 | 1.2 | 70.2 | 70.2 |
| 26 | 1.2 | 69.0 | 69.0 |
| 27 | 1.2 | 65.5 | 65.5 |
| 28 | 1.2 | 65.5 | 65.5 |

### Analysis
Earlier-text residual swapping remains near ceiling through L17, then falls from 75.0% at L18 to 1.2% at L21. Final-token residual swapping rises over the same region, reaching 72.6% at L22. This supports a transition in causal accessibility around L18–21, not a unique localized attribute circuit. Early broad swaps can effectively substitute the donor prompt representation, so ceiling success there is not evidence of sparse localization.

Single final-token attention or MLP outputs are weak (best 9.5% and 4.8%); eight-layer attention swaps ending at L22 reach 84.5%. Broad attention swapping can therefore succeed where individual layers do not. Prefill interventions often start a donor answer but fail to maintain its multi-token continuation. Final-token residual L22 has 100% donor-first-token success but only 72.6% donor-prefix success. The strict diagnostic for this arm is 65.3% on 72 strict-eligible pairs.

## Experiment 2 — focused attention sweep
Endpoints L18–24; attention spans 1–6; earlier-text and final-token scopes. 118 eligible pairs, with 12 complete countries plus two pairs for a 13th country in the common cohort. Configured 64-country run is incomplete.

### earlier_text
| Width / endpoint | L18 | L19 | L20 | L21 | L22 | L23 | L24 | Mean | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 @L18 |
| 2 | 28.8 | 0.0 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 4.7 | 28.8 @L18 |
| 3 | 24.6 | 27.1 | 4.2 | 0.8 | 0.8 | 0.8 | 0.8 | 8.5 | 27.1 @L19 |
| 4 | 49.2 | 23.7 | 32.2 | 5.1 | 0.8 | 0.8 | 0.8 | 16.1 | 49.2 @L18 |
| 5 | 55.9 | 51.7 | 32.2 | 33.1 | 5.9 | 0.8 | 0.8 | 25.8 | 55.9 @L18 |
| 6 | 58.5 | 56.8 | 58.5 | 32.2 | 33.1 | 4.2 | 0.8 | 34.9 | 58.5 @L18 |

### last_token
| Width / endpoint | L18 | L19 | L20 | L21 | L22 | L23 | L24 | Mean | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8 | 3.4 | 10.2 | 8.5 | 5.9 | 0.8 | 0.8 | 4.4 | 10.2 @L20 |
| 2 | 2.5 | 4.2 | 15.3 | 45.8 | 47.5 | 9.3 | 0.8 | 17.9 | 47.5 @L22 |
| 3 | 4.2 | 25.4 | 18.6 | 60.2 | 75.4 | 46.6 | 12.7 | 34.7 | 75.4 @L22 |
| 4 | 6.8 | 25.4 | 39.8 | 56.8 | 75.4 | 75.4 | 45.8 | 46.5 | 75.4 @L22 |
| 5 | 7.6 | 28.8 | 40.7 | 73.7 | 72.9 | 76.3 | 75.4 | 53.6 | 76.3 @L23 |
| 6 | 8.5 | 30.5 | 43.2 | 72.9 | 82.2 | 72.0 | 77.1 | 55.2 | 82.2 @L22 |

### Attribute directions at the strongest final-token window (L17–22)
| Original → donor attribute | Eligible n | Donor prefix % | Original prefix % | Neither % |
| --- | --- | --- | --- | --- |
| calling_code → capital | 9 | 100.0 | 0.0 | 0.0 |
| calling_code → currency | 9 | 66.7 | 0.0 | 33.3 |
| calling_code → language | 10 | 90.0 | 0.0 | 10.0 |
| capital → calling_code | 9 | 66.7 | 0.0 | 33.3 |
| capital → currency | 10 | 80.0 | 0.0 | 20.0 |
| capital → language | 11 | 90.9 | 9.1 | 0.0 |
| currency → calling_code | 9 | 22.2 | 0.0 | 77.8 |
| currency → capital | 9 | 88.9 | 0.0 | 11.1 |
| currency → language | 11 | 100.0 | 0.0 | 0.0 |
| language → calling_code | 10 | 80.0 | 0.0 | 20.0 |
| language → capital | 10 | 100.0 | 0.0 | 0.0 |
| language → currency | 11 | 90.9 | 0.0 | 9.1 |

### Analysis
Final-token attention across L20–22 reaches 75.4%, versus 10.2%, 8.5%, and 5.9% for the three layers separately. Three-layer windows shifted later fall to 46.6% (L21–23) and 12.7% (L22–24), giving a focused region for individual-head follow-up. L17–22 achieves the best observed result, 82.2% (97/118); its strict diagnostic is 76/100. Increasing window width is not monotonically beneficial.

Earlier-text windows have a different optimum: L13–18 and L15–20 reach 58.5%, while L19–24 gives only 0.8%. This is consistent with earlier contextual processing and later answer-position processing contributing at different stages. The design tests sufficiency of swaps, not necessity.

The best final-token window remains direction-dependent: currency → calling code is 2/9, versus calling code → capital 9/9, language → capital 10/10, and currency → language 11/11. Tokenization and answer length are plausible contributors, so this is not yet evidence of an attribute-specific head.

## Experiment 3 — prefill versus continuous
Endpoints L20–25; residual, attention, MLP, joint, and spans 2/4. 58 prefix-eligible pairs from five complete countries plus ten pairs for a sixth country. Each prefill/continuous comparison below uses exactly the same pairs and layer endpoints.

| Scope | Site | Prefill mean | Continuous mean | Δ pp | Best prefill | Best continuous |
| --- | --- | --- | --- | --- | --- | --- |
| last_token | Residual | 62.9 | 94.5 | +31.6 | 69.0 @L21 | 100.0 @L21 |
| last_token | Attention | 5.2 | 5.7 | +0.6 | 12.1 @L20 | 10.3 @L20 |
| last_token | MLP | 3.4 | 5.2 | +1.7 | 6.9 @L23 | 10.3 @L22 |
| last_token | Attention+MLP ×1 | 10.3 | 14.7 | +4.3 | 25.9 @L22 | 46.6 @L22 |
| last_token | Attention+MLP ×2 | 39.1 | 58.9 | +19.8 | 58.6 @L22 | 93.1 @L22 |
| last_token | Attention+MLP ×4 | 63.8 | 89.9 | +26.1 | 72.4 @L23 | 100.0 @L24 |
| last_token | Attention ×2 | 20.7 | 30.5 | +9.8 | 48.3 @L22 | 74.1 @L22 |
| last_token | Attention ×4 | 52.3 | 71.6 | +19.3 | 75.9 @L22 | 100.0 @L23 |
| all_text | Residual | 70.7 | 100.0 | +29.3 | 87.9 @L20 | 100.0 @L20 |
| all_text | Attention | 8.6 | 10.9 | +2.3 | 29.3 @L20 | 37.9 @L20 |
| all_text | MLP | 3.7 | 5.5 | +1.7 | 6.9 @L23 | 10.3 @L22 |
| all_text | Attention+MLP ×1 | 13.8 | 20.4 | +6.6 | 29.3 @L20 | 48.3 @L22 |
| all_text | Attention+MLP ×2 | 48.9 | 66.1 | +17.2 | 63.8 @L21 | 93.1 @L22 |
| all_text | Attention+MLP ×4 | 76.4 | 95.4 | +19.0 | 87.9 @L22 | 100.0 @L22 |
| all_text | Attention ×2 | 28.2 | 36.2 | +8.0 | 58.6 @L21 | 75.9 @L22 |
| all_text | Attention ×4 | 64.7 | 77.9 | +13.2 | 84.5 @L20 | 100.0 @L22 |

### Earlier-text prefill reference (no continuous counterpart)
| Site | Mean % | Best % |
| --- | --- | --- |
| Residual | 3.7 | 13.8 @L20 |
| Attention | 1.7 | 1.7 @L20 |
| MLP | 1.7 | 1.7 @L20 |
| Attention+MLP ×1 | 1.7 | 1.7 @L20 |
| Attention+MLP ×2 | 1.7 | 1.7 @L20 |
| Attention+MLP ×4 | 7.8 | 34.5 @L20 |
| Attention ×2 | 1.7 | 1.7 @L20 |
| Attention ×4 | 7.5 | 34.5 @L20 |

### Selected paired endpoint profiles
| Scope | Site | Endpoint | Prefill success | Continuous success | Prefill neither | Continuous neither |
| --- | --- | --- | --- | --- | --- | --- |
| last_token | Residual | 20 | 48.3 | 67.2 | 32.8 | 15.5 |
| last_token | Residual | 21 | 69.0 | 100.0 | 29.3 | 0.0 |
| last_token | Residual | 22 | 65.5 | 100.0 | 32.8 | 0.0 |
| last_token | Residual | 23 | 65.5 | 100.0 | 32.8 | 0.0 |
| last_token | Residual | 24 | 65.5 | 100.0 | 32.8 | 0.0 |
| last_token | Residual | 25 | 63.8 | 100.0 | 34.5 | 0.0 |
| last_token | Attention ×4 | 20 | 46.6 | 58.6 | 27.6 | 17.2 |
| last_token | Attention ×4 | 21 | 60.3 | 84.5 | 32.8 | 10.3 |
| last_token | Attention ×4 | 22 | 75.9 | 96.6 | 22.4 | 3.4 |
| last_token | Attention ×4 | 23 | 72.4 | 100.0 | 25.9 | 0.0 |
| last_token | Attention ×4 | 24 | 48.3 | 75.9 | 36.2 | 15.5 |
| last_token | Attention ×4 | 25 | 10.3 | 13.8 | 24.1 | 31.0 |
| all_text | Attention ×4 | 20 | 84.5 | 86.2 | 12.1 | 10.3 |
| all_text | Attention ×4 | 21 | 72.4 | 84.5 | 20.7 | 8.6 |
| all_text | Attention ×4 | 22 | 82.8 | 100.0 | 15.5 | 0.0 |
| all_text | Attention ×4 | 23 | 84.5 | 100.0 | 13.8 | 0.0 |
| all_text | Attention ×4 | 24 | 53.4 | 82.8 | 34.5 | 12.1 |
| all_text | Attention ×4 | 25 | 10.3 | 13.8 | 25.9 | 31.0 |

### Analysis
Repeated interventions substantially improve multi-token continuation. Final-token residual L22 rises from 65.5% to 100%; attention L20–23 rises from 72.4% to 100%. Both also achieve 100% exact-token success on all 52 strict-eligible pairs. The result is therefore not solely an artifact of prefix scoring. All-text residual continuous swaps reach 100% at every tested endpoint L20–25.

Across matched L20–25 endpoints, final-token residual improves by 31.6 percentage points; four-layer attention improves by 19.3 points; four-layer attention+MLP by 26.1 points. Single-layer attention and MLP remain weak, so repeating a weakly effective site is insufficient. At attention L22–25, continuous final-token swapping reaches only 13.8%, versus 100% for L20–23. Location still matters.

Continuous mode patches the final prompt token and generated positions at each decoding step. It demonstrates externally maintained steering, not a permanent edit caused by one intervention. It also runs serially while prefill runs use cached/batched execution; therefore mode and execution path are confounded. Paired serial-prefill checks are needed to isolate timing effects precisely.

## Controls and scoring audit
| Experiment | Mode | Control | Arms | Mean original preservation % | Minimum % | Output changed vs clean % |
| --- | --- | --- | --- | --- | --- | --- |
| text_full_cached_b64 | prefill | self | 1074 | 100.0 | 97.6 | 12.5 |
| text_full_cached_b64 | prefill | paraphrase | 1074 | 99.8 | 90.5 | 9.9 |
| attention_18_24_64_cached_b96 | prefill | self | 84 | 100.0 | 100.0 | 8.2 |
| attention_18_24_64_cached_b96 | prefill | paraphrase | 84 | 99.1 | 90.7 | 10.4 |
| text_continuous_cached_b2 | prefill | self | 144 | 100.0 | 100.0 | 3.8 |
| text_continuous_cached_b2 | prefill | paraphrase | 144 | 99.9 | 94.8 | 5.3 |
| text_continuous_cached_b2 | continuous | self | 96 | 100.0 | 100.0 | 0.0 |
| text_continuous_cached_b2 | continuous | paraphrase | 96 | 99.9 | 94.8 | 0.2 |

Control output-change rates use all common pairs and compare full token sequences, so harmless suffix changes count. Main preservation uses the prefix-eligible cohort. Cached self outputs do not always exactly equal clean outputs; continuous self outputs do. Paraphrase controls change only Report to Return, a narrow robustness test.

First-token success can be a whitespace match, notably for calling codes. Prefix matching can accept 366 against gold 36 or Luxembourgish against Luxembourg, and permits trailing generated text. Eligible cohorts depend on this scorer. Strict exact-token diagnostics address those specific issues but do not recognize alternate valid answers.

## Overall interpretation and next experiments
1. Full sweep identifies a broad earlier-text → final-token transition around L18–21.
2. Focused attention sweep points to a jointly effective region around L20–22, with L17–22 stronger in the sampled data.
3. Continuous sweep suggests a major prefill limitation is maintaining the donor answer during generation; appropriately positioned repeated swaps can remove this limitation.
4. Complete both partial runs, rescore complete answers, verify cached versus serial prefill, and evaluate held-out countries/templates. Then test individual heads and combinations in L20–22, with L17–22 as a broad comparison. Add ablations if claiming necessity.
5. Countries, not thousands of repeated interventions, are the independent sampling units for generalization; use country-clustered uncertainty estimates. The two partial cohorts are ordered prefixes rather than complete planned samples.

## Files
- `arm_aggregates.csv`: every arm, both baselines, controls, all/common eligible/strict cohorts, rates as fractions.
- `aggregates.json`: machine-readable per-arm and attribute-pair aggregates.
- Source data: `/Users/ovindu/Desktop/Repos/fyp/VADE-evals/results/attribute_switch/`.
