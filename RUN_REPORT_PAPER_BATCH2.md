# Run report: paper batch 2 (X2b, X2r, X3), 2026-10-19

Batch 2 of PAPER_RUNBOOK.md: X2b (routers, composition, text contrast on brands and celebrities), X2r (the
cross-question rerun with own-answer golds) and X3 (the circuit on Idefics3-8B-Llama3). Every experiment ran.
Four things departed from the runbook, all decided by the user during the run (section 2):
1. X3's mover window was chosen by hand.
2. X3's router window was chosen by hand.
3. A `--own_golds` code change was made for the X3 routers.
4. Celebrities composition was stopped at 256 of 768 rows.

Full verbatim tables for every step are in the logs named below.

## 1. Environment and timeline

| item | value |
|---|---|
| GPU | 1x NVIDIA A100-SXM4-80GB |
| torch / transformers | 2.9.1+cu130 (pod image) / 5.16.1 |
| commit | VADE-evals `097f595` (plus the uncommitted `--own_golds` change, section 2.3); VADE `b8dfaf9` |
| MPS | off |
| tests | before: 473 passed, 0 skipped (after the Idefics3 download); after the code change: 476 passed |

The pod was missing `transformers`, so GPU_RUNBOOK.md section 1 was applied as in batch 1:
- kept the image's torch;
- installed `transformers==5.16.*`, accelerate, scikit-learn, pillow, numpy, tqdm, huggingface_hub, joblib and pytest;
- installed `pyvene --no-deps`, then `matplotlib`.

Wall times are each step's `done in` line; lanes ran concurrently from about 09:34 to 16:21 UTC.

| lane | steps (min) |
|---|---|
| smoke2 | all X2b/X2r smoke steps 0-4 min each. X3 smoke stopped at its ceiling step (section 2.1) |
| 2: brands | switch 22, report 0, attr_trace 39, compose 128, tec 57, crossmodal 10, X2r c10 matched 13 / step0 13 / top-8 20 |
| 3: celebrities | switch 81, report 0, attr_trace 56, compose stopped at 256/768 rows (~85 min), tec 69, crossmodal 8, X2r c10 matched 13 / step0 13 / top-8 14 |
| 1: X3 | gate 3, accuracy 4, prune 0, stage 1 (ceilings) flags 25 + brands 14, pipeline stages 2-3 flags 59 + brands 23, overlap 0, depth 0, severed 7/7/9/5/4, router baselines 2 (gate failed), own-golds baselines 1, router sweep 83, router trace about 8 (16:13-16:21) |

**GPU** (`logs/paper/gpu_usage_batch2.csv`, 09:34-16:22; statistics over the 807 samples to 16:17):
- mean utilization 92.0%, mean power 249.5 W;
- peak memory 79,866 MiB at 10:39, when the X3 pipeline ran next to brands composition and the celebrities sweep.

Memory steps taken to avoid an out-of-memory failure:
- After 10:39, X3 ran at batch 8 rather than the A100 profile's 32. Batch size only changes rows per forward.
- X3 was paused once (12:20-12:45) so that brands and celebrities composition (about 33 GB each) did not overlap with it.

No step ran out of memory.

## 2. Departures from the runbook (all user-approved)

### 2.1 X3 smoke: launcher bug
- **What happened:** `scripts/run_x3_idefics3.sh smoke` passes `--out` together with `--positions_list` to `ceiling_sweep.py`, which refuses that combination (`AssertionError: --out names a single file; drop it when using --positions_list`).
- **What was done:** with user approval, the remaining five smoke commands were run by hand, verbatim except for dropping `--out` (`logs/paper/x3_smoke_*_manual_*.log`). All passed, and severed read `identity: bit-exact`.
- **Status:** the launcher is unfixed. Only its smoke function has the bug; the real pipeline does not pass `--out`.

### 2.2 X3 mover window (pipeline stage 1): user-chosen patch 16, blocks 16-24
The automatic plateau rule (`--patch_frac 0.95` on 32-row ceilings, about 3.1% per row) stopped at noisy one-row dips. It picked patch layer 4 (capital), 9 (language, brands) and 14 (calling_code), giving 15-21-block windows.
- **Verdict on the rule:** capital's patch-4 trace (4-24) was voided by its phase 1c and is kept in `results/image_head_pipeline_Idefics3-8B-Llama3_patchfrac095/`.
- **Why 16-24:** every curve's image swap holds through layer 16 and falls from 17. The last token rises in two steps: to about 50-59% by layer 20 (blocks 17-19), then to 88-100% at 25 (block 24); calling_code completes at 20.
- **The window:** the user chose patch 16, blocks 16-24, for all four flags attributes and brands hq_country. They first chose 16-20, then widened it before any stage 3 ran.
- **How it was run:** by hand, as the launcher's `pipeline` command plus `--patch_layer 16 --blocks 16 ... 24 --batch_size 8`.
- **Effect on phase 1c:** it necessarily reads `diverges, --blocks omits 25-31` (only blocks through 31 can MATCH), so self-check 4 is recorded against this deliberate window, not as a failure.

### 2.3 X3 routers: the gate failed on casing; user-approved `--own_golds`
- **The failure:** the first router gate read 8/96 eligible, against a minimum of 24 (`logs/paper/x3_switch_gate_20261019_140915.log`). Idefics3 answers flags' controlled prompts in lowercase (`' luxembourg.'`, `' eur.'`), and flags scored by token against the stored, capitalised labels (`uses_own_golds()` was `entity != 'flags'`).
- **Checked by hand:** case-insensitively the clean answers were right 21/24 (capital), 24/24 (currency), 18/24 (language) and 15/24 (calling_code).
- **Code change (uncommitted), approved by the user:**
  - `attribute_switch_sweep.py` and `attr_head_trace.py` gain an opt-in `--own_golds` flag. With it, flags uses the own-answer path that brands and celebrities already use.
  - With the flag off, a flags config is unchanged (the key is left out of the run identity).
  - `attr_head_trace` adds `_owngolds` to its default tag.
  - Three tests were added to `tests/test_x2b_entities.py`; the full suite passed (476).
  - Side effect: the sweep hashes its own source into `implementation_sha256`, so sweep directories made before the edit cannot be *resumed*. Their results are unaffected.
- **Rerun:** with `--own_golds` the gate passed at 68/96. The router window was chosen by the user from the crossover (section 4.3). `circuit_windows router` had picked 5 / 5-20 off a dip at layer 7.

### 2.4 Celebrities composition stopped at 256/768 rows
- **Decision:** to save about 1.5 h, the user decided to stop it at 256 rows; the stop point was fixed before any celebrities composition result had been seen.
- **How:** SIGINT was sent right after the `256/768` line. Rows are written item by item, so this is the first 21 of 64 seeded-random celebrities, complete, plus 4 of 12 pairs of the 22nd.
- **Summary:** rebuilt on CPU with the script's own `summarize()` into `results/paper/x2b_compose_celebrities/summary_partial256.json`, which is marked partial.
- **Remaining steps:** lane 3's text contrast, crossmodal and X2r were then run with the launcher's commands (X2r by name).

## 3. X2b and X2r: brands and celebrities

### 3.1 Self-checks

| # | check | brands | celebrities |
|---|---|---|---|
| 1 | sweep `self` base_kept ≥ 0.99 / `paraphrase` ≥ 0.97 / eligible ≥ 50% | pass: 1.0000 / 1.0000 / 102/144 | **self 0.9808 marginal fail** / paraphrase 0.9867 pass / 142/192 pass |
| 2 | `--window` printed PATCH/BLOCKS | pass: 15 / 15-22 | pass: 16 / 16-22 |
| 3 | trace phase 0 ≥ 95% | pass: 100/100 (golds kept 494, dropped 270) | pass: 100/100 (kept 682, dropped 326) |
| 4 | phase 1b reached; phase 1c accepted verdict | pass (omits 23-27) | pass (omits 23-27) |
| 5 | k=0 question patch ≥ 80%; phase 3 k=0 = unhooked | pass: 98.8% | pass: 100.0% |
| 6 | compose clean ≥ 95% C.X; compose_random ≥ 85% C.X | pass: 98.2 / 94.6 | pass: 100.0 / 98.4 (n=64, partial) |
| 7 | tec n_eligible ≥ 100 per condition; crossmodal X1 checks 1-3, 5 | pass: 171 / 177; crossmodal pass | pass: 240 / 240; crossmodal pass |
| 8 | no KeyError / AssertionError / ValueError | pass | pass (the only traceback is the intended SIGINT, section 2.4) |
| X2r-1 | `self` cell base_q1 ≥ 99% | pass: 99.2% (all 3 runs) | **97.2% marginal fail** (c10 matched and step0) |
| X2r-2 | own-answers line | founded_year 82/123, hq_country 117/123, industry 97/122 | birth_year 56/78, nationality 63/78, native_language 67/78, occupation 74/78 |
| X2r-3 | informative n ≥ 60 per base question | pass: 163 / 183 | pass: nationality 79, native_language 61 |
| X2r-4 | `question` moved to A's Q2 ≤ 5% | pass: 0.0% | pass: 0.0% |

**Both celebrities marginal fails are near-tie wording drift, not the hooks.**
- *Sweep self control:* all 447 non-kept self rows are one celebrity (rows 120-122, nationality). Clean says `' United States'`, while every self arm, at every layer including 1, says `' USA'`. That is the serial versus cached-prefill numerics, not the patch.
- *X2r self cell:* in all 6 failing rows the self-patched text is identical to that row's clean run. It differs from the separately batched own-answer gold: Henry Ford / Beyoncé `' United States (of America).'` vs `' the United States of America.'`, and Kafka `' German.'` vs `' Czech.'`.

### 3.2 Routers (attribute-switch sweep, then `attr_head_trace`)
The handoff replicates. The question-token swap falls and the last-token swap rises in blocks 17-19, as on flags (crossing 17-21):

| layer | brands question / last | celebrities question / last |
|---|---|---|
| 17 | 100.0 / 0.0 | 97.2 / 0.0 |
| 18 | 85.3 / 10.8 | 94.4 / 1.4 |
| 19 | 27.5 / 64.7 | 27.5 / 58.5 |
| 20 | 9.8 / 82.4 | 10.6 / 79.6 |
| 21 | 0.0 / 100.0 | 0.0 / 98.6 |

- **Attention spans:** 3-block attention spans covering blocks 18-20 and 19-21 read 99.0 / 97.1% on brands and 94.4 / 95.8% on celebrities, against flags' 96.6-100%.
- **Window:** the rule picked 15 / 15-22 on brands (identical to flags) and 16 / 16-22 on celebrities.

| trace | brands (15 / 15-22) | celebrities (16 / 16-22) |
|---|---|---|
| top-8 blocks | 19x2, 20x3, 21x2, 22x1 | 19x1, 20x3, 21x3, 22x1 |
| top-8 | 21.0, 20.17, 20.18, 19.27, 21.17, 19.21, 22.25, 20.2 | 20.17, 21.0, 20.18, 19.21, 22.8, 21.25, 21.17, 20.2 |
| flags' router top-8 recurring | **5/8** (21.0, 20.17, 20.18, 19.21, 19.27) | **5/8** (21.0, 20.17, 20.18, 19.21, 21.25) |
| phase 2: restore top-4 / 8 / 16 (random 8) | removes 56.6 / 21.9 / 53.3% (8 random: ~0%) | removes 65.0 / 33.4 / 81.5% (8 random: ~0%) |
| phase 3: top-4 / 8 / 16 (all heads) | 39.1 / 15.8 / 32.0% (98.4%) | 49.4 / 23.5 / 68.2% (99.7%) |
| phase 4 verdict | CONFIRMED at ALL 224 (donor 97.0%, own 10.1%) | CONFIRMED at ALL 196 (donor 98.5%, own 2.9%) |

Flags' top-8 is 21.0, 20.17, 20.18, 19.21, 21.25, 21.4, 21.5, 19.27. Four heads (21.0, 20.17, 20.18, 19.21) are in all three top-8s, and brands and celebrities share 6/8.

The routers are in the same blocks and largely the same heads across entities. As on flags, they are redundant: phases 2-3 are non-monotone in k, and phase 4 confirms only with all heads.

### 3.3 Composition (eligible rows with four distinct golds)

| arm | predicts | brands (n=112) | celebrities (n=64, partial) | flags E4 (n=548) |
|---|---|---|---|---|
| clean | C.X | 98.2 | 100.0 | 99.1 |
| movers | B.X | 25.0 | 42.2 | 91.8 |
| routers | C.A | 8.9 | 21.9 | 85.6 |
| compose | B.A | **0.0** | **15.6** | 78.8 |
| compose_random | C.X | 94.6 | 98.4 | 99.1 |
| target_window | B.A | 82.1 | 87.5 | 92.2 |
| target_complement | C.X | 51.8 | 65.6 | 91.8 |

Whole-window patching still composes the new answer (82-88% B.A). Each entity's own traced movers and routers carry only part of it, and composing them does not reproduce B.A as on flags. The `other` column is large: 24-38% on brands' head arms.

### 3.4 Text contrast and crossmodal

| | brands | celebrities |
|---|---|---|
| clean accuracy, image / text | hq_country 96.1/96.1, founded_year 60.2/63.3, industry 85.8/83.5 | nationality 79.8/88.1, occupation 96.4/88.1, birth_year 66.7/95.2, native_language 83.3/85.7 |
| L_e, image / text | 12 / 13 | 13 / 13 |
| entity patch at L_e, image / text | 96.5 / 74.6% | 97.5 / 84.2% |
| install image movers, into image / text | 31.6 / 23.2% | 37.9 / 25.8% |
| install top-20 heads, into image / text | 39.8 / 40.7% | 55.8 / 46.7% |
| random-10 | 0.0 / 0.0% | 0.4 / 0.0% |
| image ∩ text top-10 heads | 7/10 | 8/10 |
| freeze readout 22-27 @base, image / text | 8.2 / 1.7% | 2.5 / 0.0% |
| freeze readout 16-22 @base, image / text | 81.3 / 54.2% | 64.6 / 45.8% |

- **Same heads in both modalities:** image and text runs share 7-8 of their top-10 heads (21.4, 22.15, 23.3 and the 26.x group).
- **The entity arrives at the same depth** whether shown or named: L_e is 12-13 in both conditions.
- **Late readout:** freezing the last-token readout 22-27 at base kills the transfer in both modalities, as on flags.

Crossmodal (mover-head install; rows = install into image / into text):

| | movers from the other modality | from the same modality | norm-matched | self / random |
|---|---|---|---|---|
| brands (n=175 / 181) | 35.4 / 30.9% | 30.3 / 33.1% | 34.9 / 26.5% | 0.0 / ≤0.6% |
| celebrities (n=244 / 244) | 43.9 / 43.0% | 36.5 / 47.1% | 38.1 / 43.0% | 0.0 / ≤0.4% |

The mover payload is modality-invariant on both entities (cross ≈ same), as in X1 on flags, but at about 30-47% transfer rather than flags' 89-92%. `resid@0` top-1 is 0.0%, at or below chance.

### 3.5 X2r: cross-question with common10 and own golds
B's answer to Q1 on informative rows (base and donor answer Q1 differently):

| run | Q1 | same question | another question | shift |
|---|---|---|---|---|
| brands common10, matched | hq_country | 85.9% (n=163) | 44.3% (n=183) | −41.6 |
| brands common10, **step0** | hq_country | 76.1% | 42.1% | **−34.0** |
| brands top-8, matched | hq_country | 62.0% | 9.8% | −52.1 |
| celebrities common10, matched | nationality | 73.4% (n=79) | 38.9% (n=175) | −34.6 |
| celebrities common10, **step0** | nationality | 67.1% | 37.1% | **−29.9** |
| celebrities common10, matched | native_language | 70.5% (n=61) | 47.4% (n=137) | −23.0 |
| celebrities common10, **step0** | native_language | 68.9% | 47.4% | **−21.4** |
| flags R11, matched | (all) | 81.2% | 65.1% | −16.1 (step0 removes most) |

Excluded rows (a needed own answer was wrong): 458/1,529 on brands and 688/2,048 on celebrities. The 4-way cell tables are in `logs/paper/x2r_cross_*.log`.

**Reading** (runbook table): the "same" cell now reaches flags' level with common10 and own golds, but **the cross-question shift survives step0 on both entities** (−21 to −34 points). That is "question-conditioned content on these entities": at the answer column, a large part of what the common10 heads carry depends on the question the donor was asked. On flags this is not so. Brands' own top-8 carries almost nothing across questions (9.8%).

## 4. X3: Idefics3-8B-Llama3

### 4.1 Self-checks

| # | check | result |
|---|---|---|
| 0 | smoke tests 26 passed, 0 skipped | pass. The smoke itself needed the section 2.1 workaround |
| 1 | gate | **pass**: flags/capital 77.2% (n=504), flags/language 75.8% (504), brands/hq_country 81.1% (640) |
| 2 | ≥ 64 test cause rows per traced attribute | pass: flags 2,568-2,832, brands hq_country 2,557 |
| 3 | scoring guard (layer-0 full_image ≥ 85% source match) | pass: all five at 96.9-100% |
| 4 | phase 1c MATCHES, no `k=0 differs` | **does not apply as written**: with the user's window 16-24, every trace reads `diverges, omits 25-31` (only a trace through 31 can MATCH). No `k=0 differs` line. Phase 1b passed on all five |
| 5 | ROUTER GATE PASSED; `/self` donor_first ≤ 5%; no circuit_windows WARNING | first run **failed** (8/96, casing, section 2.3); own-golds rerun **pass**: 68/96, `/self` donor_first 2.9%. circuit_windows printed no WARNING, but its window was not used |
| 6 | severed identity bit-exact | pass: all 5 runs, read-back 0.0 |
| 7 | only new paths written | pass; plus `results/image_head_pipeline_Idefics3-8B-Llama3_patchfrac095/` and `logs/..._patchfrac095/` (the moved-aside auto run) and `logs/Idefics3-8B-Llama3/` (the ceiling cache) |

### 4.2 Movers (pipeline at patch 16, blocks 16-24)
Stage 2 (capital, at patch 16) found that installing all heads of 16-24 transfers 81.2% of the image patch's 82.8%, while **no window of 3 blocks or fewer exceeds 4.7%**. The full-coverage identity check passed (16-31: 82.8% vs 82.8%). So on Idefics3 the image→text read is distributed over blocks 16-24, not concentrated in 3 blocks as on Qwen (21-23).

| attribute | top-8 movers | phase 4 shuffled-donor verdict |
|---|---|---|
| flags capital | 24.3, 16.9, 24.1, 22.20, 16.11, 24.2, 19.25, 16.18 | CONFIRMED at top-16 (donor 65.6%, own 1.6%) |
| flags currency | 24.3, 16.9, 24.1, 24.26, 16.18, 16.11, 17.23, 22.20 | CONFIRMED at top-16 (59.4 / 0.0) |
| flags language | 24.3, 16.9, 24.1, 16.11, 16.8, 24.2, 16.18, 22.20 | CONFIRMED at top-16 (79.7 / 10.9) |
| flags calling_code | 24.3, 16.9, 19.25, 24.1, 19.27, 22.20, 16.11, 16.18 | CONFIRMED at **top-8** (75.0 / 4.7) |
| brands hq_country | 24.3, 22.23, 16.9, 21.15, 23.27, 18.17, 21.13, 20.3 | CONFIRMED at top-32 (56.2 / 10.9) |

- **Within-model overlap** (`results/x3_idefics3/head_overlap/`): six heads, **16.9, 16.11, 16.18, 22.20, 24.1, 24.3**, are in at least 4 of the 5 top-8s (p = 0.0005 under both nulls). That's the same question-blind consensus pattern as Qwen's common10.
- **Two groups of heads:** 16.x reads at the first step, 22.20 / 24.x at the second.

**Depth** (`results/x3_idefics3/depth.json`):

| model | window (relative depth) | read ends (relative depth) |
|---|---|---|
| Idefics3, all five | 16-24 (0.53-0.78) | 0.78; calling_code 0.63, currency 0.94 |
| Qwen-7B, brands / celebrities | 19-23/24 (0.71-0.89) | 0.82-0.86 |
| InternVL3-8B, flags | 21/22-23 (0.79-0.86) | 0.82-0.86 |
| Qwen-3B | 27-33 (0.78-0.94) | 0.78-0.92 |

The read ends near ~0.8 on Idefics3 too, but it starts much earlier, so the window is wider (0.25 of depth vs about 0.15).

### 4.3 Routers (`--own_golds`; user's window patch 12, blocks 12-17)
Crossover on the full sweep (68 eligible rows):
- question-token swap: 92.6-94.1% through layer 12, then 67.6 / 50.0 / 50.0% at 13-15, and 11.8% by 16;
- last-token swap: 13.2% at 13, 66.2% at 16, then 70.6% at 18-21, rising further later.

So the crossover is **blocks 12-15, relative depth 0.38-0.50**. On Qwen-7B flags it was 17-21 of 28 (0.61-0.75).

Trace checks all pass:
- golds kept 122, dropped 70;
- phase 0: 100 / 100;
- read-back 0.00e+00; 1c: accepted "omits 18-31";
- k=0 question patch: 91.8%;
- phase 3 k=0 = unhooked.

| phase | result |
|---|---|
| top-8 routers | 14.13, 15.27, 16.25, 17.5, 17.6, 13.8, 16.8, 13.18 (blocks 13x2, 14, 15, 16x2, 17x2); per-block mass peaks in 13-15 |
| 2: restore top-4 / 8 / 16 | removes 53.6 / 49.1 / 44.6% (8 random: about 10%; gap +36.1%) |
| 3: top-4 / 8 / 16 alone | 15.6 / 23.0 / 23.8% first token (all 192 heads: 77.0%, a partial ceiling) |
| 4: shuffled donor | CONFIRMED at ALL 192 (donor 71.3%, own 4.1%); top-8 also follows the donor (27.0% vs 2.5%) |

Routers exist on the Llama language model and are redundant, as on Qwen. They sit **before** the movers (12-17 vs 16-24) with little overlap. On Qwen the two overlap (routers 19-21, movers 21-23).

### 4.4 MLP lookup (severed; each attribute's own top-8; freeze spans derived: MLP 23-31, attention 25-31)

| attribute | top-8 install | + MLP 23-31 @base (kept) | MLP @other → 3rd item | + attention 25-31 @base (kept) |
|---|---|---|---|---|
| flags calling_code | 74.2% | **0.0%** (0.00) | 78.9% | 71.1% (0.96) |
| flags currency | 46.8% | **0.0%** (0.00) | 63.0% | 74.6% (1.59) |
| flags language | 40.8% | **4.0%** (0.10) | 27.2% | 45.6% (1.12) |
| flags capital | 32.8% | **0.8%** (0.02) | 14.3% | 46.9% (1.43) |
| brands hq_country | 17.4% | 8.3% (0.48) | 18.0% | 21.5% (1.24) |

- **The late-MLP lookup replicates on flags.** Freezing the last-token MLPs 23-31 kills the install, and freezing them at another country writes that country's value (calling_code 78.9%, currency 63.0%).
- **No single MLP dominates.** Capital singles at 24-30 each cost 12-50%.
- **Late attention behaves differently from Qwen.** Freezing attention 25-31 never hurts and often helps (kept 0.96-1.59); on Qwen it barely mattered.
- **Brands is weak throughout.**

Against Qwen's E7/P10: the same circuit shape, but the 8-head install is much weaker (17-74% vs about 90% for common10), consistent with the distributed read.

**How to read X3** (runbook table): "movers in one window, top-8 transfer high, freezing late MLPs kills it" holds for flags, with a wider window and a weaker 8-head install. Routers exist, at a crossover the controlled prompts do elicit (with own golds).

## 5. Things to know

1. `scripts/run_x3_idefics3.sh smoke` is broken as described in section 2.1, and is unfixed.
2. The plateau rules in `image_head_pipeline` (stage 1) and `circuit_windows router` both misfired on Idefics3's 32-row curves, at one- or two-row dips. Every X3 window in this report was chosen by the user from the curves.
3. The `--own_golds` change is uncommitted, in `methods/attribute_switch_sweep.py`, `methods/attr_head_trace.py` and `tests/test_x2b_entities.py`. CLAUDE.md's X2b paragraph does not yet mention it.
4. 137 of this session's output files are staged in the git index. They were not staged by this session, and nothing has been committed.

## 6. Where everything is
- **X2b / X2r:** `results/paper/x2b_*`, `results/paper/x2r_*`, `results/paper/smoke2/`, and `logs/attr_head_trace/{brands,celebrities}/`.
- **X3 results:** `results/x3_idefics3/`; `results/image_head_pipeline_Idefics3-8B-Llama3/` (SUMMARY.md per entity); and the auto-selected run moved aside as `..._patchfrac095/`.
- **X3 logs:** `logs/image_head_pipeline_Idefics3-8B-Llama3/` and `logs/Idefics3-8B-Llama3/`.
- **VADE outputs:** `../VADE/models/Idefics3-8B-Llama3/`.
- **Step logs:** `logs/paper/` (one per step), and `logs/paper/gpu_usage_batch2.csv`.
- **Hand-run scripts:** the scripts run instead of launcher steps are copied to `/workspace/runlogs/batch2_handrun_scripts/` and included in the package below.
