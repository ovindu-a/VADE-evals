# Candidate claims beyond the current six: experiments, results, next steps (2026-10-17)

Each question below has four parts:
- **Done:** the experiment that addresses it (script, settings, n).
- **Result:** what came out.
- **Gap → run:** what is still missing, and the concrete experiment that would close it.
- **Expect:** what that experiment should show, and how to read each outcome.

Numbers were spot-checked against `results/` and `logs/`.

Conventions:
- Blocks are 0-based.
- "Layer L" in ceiling sweeps is the residual after block L-1.
- "first" = first-answer-token transfer; "cause" = full-answer match.
- Strength labels: **S** strong, **M** moderate, **W** weak, **?** suggestive.

The six claims already made (as refined in the Ideate chat) are not repeated:
1. Movers carry the entity.
2. Grounding heads are separate from the movers.
3. Image and text share the circuit.
4. There are two routes, with the question route as a fallback.
5. Movers and routers compose.
6. MLPs 23-26 do the lookup, and the readout edit raises VADE final.

**Fix Z1 (DBM) before anything that cites DBM.**

**CPU results, 2026-10-17.** Entries marked **CPU result** were run with
`methods/cpu_followups.py`. It recomputes from files already on disk and needs no
model. Numbers are in `results/cpu_followups/REPORT.md` and the per-analysis JSONs
there.

---

## A. The mover heads, beyond "they move the entity"

### A1. Do the movers only work as a conjunction? (S)
- **Done:**
  - `head_trace.py --patch_layer 21 --blocks 21-23`, phase 3 (sufficiency). The image is not patched; the patched values of the top-k ranked heads are installed at the last prompt column. k = 1, 2, 4, 8; 64 rows × 4 flags attributes.
  - `head_attribute_edit` random sets that happen to contain strong movers.
- **Result:**
  - 1 head 0-1.6%, 2 heads ≤4.7%, 4 heads 20-61%, 8 heads 61-100%.
  - A random-16 set that contains 23.4 and 21.1 (the two strongest movers) gives 3-6%. A random-8 set containing 22.19 and 21.5 gives 0%.
- **Gap → run:** greedy text is a thresholded readout, so a smooth additive effect also looks like "nothing until k=4". Re-run the `common10` install measuring the **logit difference** (source minus base first token) for:
  - each single head,
  - every pair (45 pairs),
  - leave-one-out (10 arms),
  - all 10.
- **Expect:**
  - **True conjunction:** the joint logit effect is much larger than the sum of the single-head effects, and each leave-one-out arm costs a lot.
  - **Additive below a threshold:** the logit effect grows roughly linearly in k, and the "conjunction" is a generation artefact.
  - Given the k=4 jump, superadditivity is the more likely outcome. Either way, report the logit curve, not only the text.

### A2. Are the movers necessary at the answer column, not only sufficient? (S for language and capital)
- **Done:** `head_trace.py` phase 2 (necessity). The image residual is patched to the source at L21; the top-k heads at the last column are then restored to base.
- **Result:**
  - Restoring the top-8: transfer falls to 0.0 / 0.0 / 4.7 / 28.1% (language / capital / currency / calling_code). The base answer returns 95 / 92 / 66 / 8%.
  - Restoring 8 random heads: transfer stays at 84-94%.
  - This is in the trace JSONs but not written up in R8.
- **Gap → run:** calling_code's 28% left over. Repeat phase 2 restoring the heads at **every generated step** (the continuous mode, as in `head_swap_vade`), not only the prompt column.
- **Expect:** calling_code falls toward ~0%. That would show the 28% is later answer tokens re-reading the patched image through the KV cache (the multi-token artefact), not a second route.

### A3. Do 10 of 784 head slots carry about 95% of the whole image's effect? (S)
- **Done:** `head_swap_vade.py` arms clean / common5 / common10 / top8 / random (size-matched) / full_image. VADE templates, test split, continuous patch, 2,688-3,600 rows per cell.
- **Result:**
  - `common10` 88.5% vs `full_image` 93.0% mean transfer.
  - Dose-response for calling_code: 69 / 87 / 96% at 5 / 8 / 10 heads.
  - Random size-matched sets: 0%, base kept 98.6-100%.
- **Gap → run:** the heads were chosen from traces that may share items with the test split. Re-select them on half the countries:
  - run `head_trace.py` on 42 countries;
  - take the top-10 by consensus;
  - evaluate with `head_swap_vade.py` on the other 42.
- **Expect:**
  - ≥8 of 10 heads recovered, and transfer ≥85% of `full_image`: the claim is clean.
  - A large drop would mean the selection was overfit to the evaluation items.

### A4. Do the movers carry the whole entity, so every attribute flips together? (S)
- **Done:**
  - R10 iso columns.
  - `head_attribute_edit/flags/matrix/cross_attribute_matrix.json`: each attribute's own top-8 and top-16 sets, installed and then queried on all four attributes, n=32 per cell. This is written up nowhere.
  - Learned coordinate masks inside those sets (`head_attribute_edit/flags/masks/*/eval`).
- **Result:**
  - When calling_code is the target and the model is asked for the capital, it gives the source's capital 98% of the time.
  - Every attribute's set transfers every asked attribute at 72-91%. The diagonal is no higher than the off-diagonal. Random sets: 0-6%.
  - Hard masks stay non-selective (56-88% on every attribute). Only calling_code's soft top-8 mask is partly selective: 46.9% on target vs 3-19% off-target.
- **Gap → run:** none for the claim. Optionally sweep mask sparsity (only 0.01 was tried). Note that the gates got *denser* during training (0.5 → 0.62-0.70).
- **Expect:** no selective mask for capital, currency or language at any sparsity. That would be consistent with C5: attributes only separate after the MLP lookup.

### A5. Do the movers write the entity's name directly, but almost none of the answer logit? (S)
- **Done:**
  - `head_vocab_projection.py`: 84 items × 4 questions. Each head's exact write z_h W_Oᵀ is passed through the frozen final norm and the unembedding, and the own name is ranked among the candidate tokens.
  - `dla.py --direction gold_minus_mean`: 144 rows per attribute.
- **Result:**
  - Eight of ten movers rank the own name at 1.4-9.6.
  - The name z-score is the same whichever question was asked (2.95-3.23).
  - Attention blocks 21-23 together write 0.13-1.2 of a 21.6-24.9-logit answer; MLPs 23-27 write 17.5-22.3.
- **Gap → run:** the stated chance level is wrong (Z3). The set has 156 candidates (two spellings each), and best-of-two gives a null rank of ≈52, not 39. Recompute the null empirically from the saved `rows.jsonl` by scoring a *random other* country's name under the same head. This is CPU-only.
- **Expect:** movers stay far below the null (1.4-9.6 vs ≈52), and router8 lands at the null (42-59). The conclusion survives; the stated chance level changes.
- **CPU result (`name_rank_null`):**

  | entity | candidates | best-of-2 null rank | null top-1 | null z | movers' rank (z) | routers' rank (z) |
  |---|---|---|---|---|---|---|
  | flags | 156 | 52.4 | 1.2% | 0.56 | 8.8 (3.08), 6× better | 49.0 (0.77), at the null |
  | brands | 222 | 75.0 | — | — | 29.9 (2.07), 2.5× better | — |
  | celebrities | 145 | 48.7 | — | — | 30.2 (1.30), 1.6× better | — |

  - **Verdict:** supported, and the stored chance level (78.5) should be replaced by 52.4.
  - The movers write the entity's name strongly on flags, moderately on brands and weakly on celebrities. The routers write no name.
  - The null assumes both spellings are distinct tokens. A few collapse to one, so 52.4 is slightly optimistic for the null.

### A6. Must the movers re-assert the entity at every generated token? (S)
- **Done:** `head_cross.py` (R11.3b): the head patch at the first answer token only, vs the first-token values replayed at every step.
- **Result:**
  - First token only: 81.2 → 51.6% overall, and calling_code 93.8 → 1.6%.
  - Replaying at every step: 81.6%, with the self-patch staying benign (93.4%).
- **Gap → run:** the "replications" are not independent (Z4). Run `head_cross.py --seed 1 --n_pairs 64`, which gives new pairs.
- **Expect:** the same double dissociation within ±5pp. Multi-token attributes (calling_code) should show the largest first-token-only loss.

### A7. Is the moved payload portable across prompts, and independent of the donor's question? (S for language)
- **Done:** `head_cross.py` cells: same question in a different prompt; a donor asked a different question; same flag with the donor asked a different question.
- **Result:**
  - Different prompt: 81.2% (n=256).
  - Donor asked a different question: 65.1%. Excluding calling_code, the gap is −0.9pp.
- **Gap → run:** the 64 pairs cover only 44 bases and 45 sources, so the ±4.8pp interval is optimistic. Run the seed-1 rerun (A6) and bootstrap by country, not by pair.
- **Expect:**
  - Language: the gap stays around 0.
  - calling_code: the gap persists, as the matched-step-replay artefact (M3).
  - Capital and currency: intervals of about ±7pp that include 0.
- **CPU result (`cross_bootstrap`):**

  | cell | rate | pair-bootstrap CI | country-cluster CI |
  |---|---|---|---|
  | different prompt | 81.2% | ±4.1 | ±5.1 |
  | donor asked a different question | 65.1% | ±4.8 | ±5.7 |
  | same flag, donor asked a different question | 83.2% | ±3.3 | ±3.9 |
  | self-patch control | 93.0% | ±2.9 | ±3.6 |

  - **Verdict:** the intervals widen by about 1pp; every comparison in R11 survives.

### A8. Is the moved entity code partly regional rather than country-exact? (?)
- **Done:** nothing systematic. These are incidental observations:
  - Chinese region tokens at 23.6: '非洲' (Africa) ×62, '阿拉伯' (Arabia) ×29.
  - 35.8% of wrong calling codes share the source's first digit (the ITU zone).
  - Currency errors go to neighbouring countries (IQ→KWD, SD→EGP, NP→INR).
- **Gap → run** (all CPU on existing JSONLs except step c):
  - **(a) Error geography.** For every "other" (neither base nor source) answer in R10, E4, head_cross and the readout edits, map the emitted answer to its country. Compare how often it falls in the same UN subregion as the source against a null from the region distribution of all other countries.
  - **(b) Dose.** Install `common10` scaled by α ∈ {0.25, 0.5, 0.75}, and see whether the intermediate doses produce region-mates of the source.
  - **(c) Probe.** On `attr_capture`'s `common10` columns, compare a leave-one-country-out continent probe with a nearest-centroid country identification.
- **Expect:**
  - **If regional:** same-region errors well above the null (e.g. >50% vs ~15-20%), low doses landing on region-mates, and a continent probe near ceiling at doses where country identity fails.
  - **If not:** errors at the null rate, in which case the Chinese tokens are just a lexical quirk.
- **CPU result (`error_geography`):**
  - **Method.** Answers that match neither base nor source are mapped back to the country owning the emitted value. They are deduplicated to one record per (file, base, source, queried) and compared against a uniform-other-value null. A placebo reference country is included to check the null.
  - **Calibration.** The placebo sits near its null (9.7% vs 7.7% pooled) for capital, currency and calling_code. Language is excluded: its shared values push even the placebo to 19-30%.

  | condition | same subregion as source (null ≈6%) | same subregion as base |
  |---|---|---|
  | R11 head_cross (n=263 clusters) | 56% (z=21) | 6% (at null) |
  | readout edit at the heads | 39% | 4% |
  | E4 compose (only 15 mapped) | 47% | 0% |
  | image-site dictionaries | 28% | 28% |
  | DBM | 25% | 26% |
  | dense MLP masks | 8% | 42% |
  | E9 zero:`common10` (vs the item's own subregion) | 23% | — |

  - **Reading:**
    - In clean entity installs, a failed transfer lands on the source's regional neighbours.
    - Interventions that mix source and base (image-site methods) land near either country equally often, i.e. they blend the two.
    - Dense MLP masks land near the base.
    - With the movers zeroed, wrong answers stay in the shown flag's region (continent 56% vs 23%), so a coarse, region-level identity reaches the answer without the movers.
  - **Verdict:** supported for capital, currency and calling_code.
  - **Caveats:**
    - Only 35-53% of "other" answers map to any country.
    - Calling-code zone hits (49-92%) partly reflect the first-token effect (the first digit is the first token), so they are not used as geographic evidence.
  - **Still open (GPU):** the dose sweep (b) and the continent probe (c).

### A9. Do the movers read the background more in total, but the flag more per token? (M)
- **Done:**
  - `head_vocab_projection` key-group split (A@V asserted to reconstruct o_proj's input).
  - `head_followups` `tokens.md` (n=4).
- **Result:**
  - Head 21.1 puts 60% of its attention on the background and 19% on the 8 flag tokens. Per token, that is about 5× more on the flag.
  - The direct name write per token is about 4.4× higher on the flag.
- **Gap → run:**
  - First, compute per-token-normalized attention and write for all 84 items from `rows.jsonl` (CPU).
  - Causal version: knock out the answer column's attention to **object** keys vs to **background** keys in blocks 21-23. The question → image edges must be cut too; otherwise the question-token fallback (E5) hides the effect. This needs a key-group option in `route_knockout.py`'s `EdgeKnockout`.
- **Expect:**
  - Cutting background edges costs more in total.
  - Cutting object edges costs more per token.
  - If cutting object edges alone costs almost nothing, the "flag-specific" reading is wrong, and identity is mainly read from the ring around the flag (consistent with E11).
- **CPU result (`per_token`):**
  - Every one of the 10 flags movers attends more to the background **in total** (20-60% vs 10-26% on the flag).
  - Per token, every one of them attends 5.4-14.1× more to the flag, and its direct entity write is 5.0-15.2× higher per flag token.
  - Brands: 4.7-9.9×. Celebrities: 3.7-8.9× (with 16 face tokens).
  - **Verdict:** confirmed for all three entities. "Background-dominated" is true only of totals, because there are 17× more background tokens.
  - The causal object-vs-background edge knockout still needs a GPU.

### A10. Is `common10` an entity-identity conduit that category attributes don't use? (M)
- **Done:**
  - `head_overlap.py` (E10; 20,000 permutations; pool and block nulls).
  - `image_head_pipeline` on animals.
- **Result:**
  - brands and celebrities share 7 of 8 heads with `common10`.
  - animals/native_continent shares 5/8; body_covering shares ~0.
  - Animal top-8 sufficiency is 3-41%, and it takes 392-532-head windows to reach ≥90%.
- **Gap → run:** this is overlap, not causation. Install `common10` itself on animals (`head_swap_vade.py --entity animals --head_sets common10=...`, text-matched).
- **Expect:**
  - Entity-type attributes (native_continent) get high transfer.
  - Category attributes (body_covering, reproduction) get low transfer.
  - A clean split confirms the claim causally.

---

## B. The attribute route, beyond "routers compose with movers"

### B1. Are the routers necessary and sufficient, and also only jointly? (S)
- **Done:** `attr_head_trace.py`: image fixed, question swapped. `--patch_layer 15 --blocks 15-22 --question_positions all_text`, n=1,008 rows over 84 items.
- **Result:**
  - Restoring the top-8 to base: 95.4 → 3.8%. Restoring 8 random heads: 95.3%.
  - Installing only the top-k: k=1 0.4%, k=2 4.6%, k=4 59.3%, k=8 87.0%, k=16 94.2%. Random-8: 0.4%.
  - With a shuffled donor, the answer follows the donor at 90.1%.
- **Gap → run:** (i) the heads are selected and evaluated on the same items; (ii) the traced blocks stop at 22, so with every traced head patched 18.2% still differs, and the identity check diverges by 8.3%. Run `attr_head_trace.py --blocks 15-27`, selecting heads on 42 countries (`--countries`) and evaluating on the other 42.
- **Expect:**
  - router8 reproduced (≥6/8).
  - Held-out k=8 sufficiency ~85%.
  - With blocks to 27, the all-heads arm equals the image patch (an exact identity, as in head_trace).

### B2. Do movers and routers doubly dissociate? (S)
- **Done:**
  - `circuit_compose.py` arms: movers only, routers only, composed, random, complement.
  - `head_vocab_projection` on router8.
- **Result:**
  - Movers alone: B.X 91.8%, C.A 0%. Routers alone: C.A 85.6%, B.X 0% (n=548).
  - The routers' name rank is at the null; they put 1-8% of attention on the image and 24-59% on the question.
  - Only 21.5 is in both sets.
- **Gap → run:** none needed.

### B3. Is "which attribute was asked" one entity-independent direction you can steer with? (S for the first token)
- **Done:** `attr_steer.py`. Directions are attribute-centroid differences estimated on 32 train countries, applied by `add` / `project` at the last token, L14-28, on 16 held-out countries (192 rows). Matched-norm random-direction controls.
- **Result:**
  - L22 `add` α=1: 95.8% first token.
  - `project` α=0.5: 91.7%, against 39.1% for `add` at the same α.
  - Random directions ≤1.6%.
  - Full-answer match is 58%.
- **Gap → run:**
  - **(a)** Three train/test splits (`--seed`) for an interval.
  - **(b)** Apply the steer at every generated step, not only the prompt column.
  - **(c)** Report sdpa and eager side by side (G7).
- **Expect:**
  - (a) ±5pp.
  - (b) Full answer rises to about first-token level (~90%), since the sweep_report shows the first token is the decision point.
  - (c) Eager may be ~10pp lower on base-kept; report both.

### B4. Does the attribute code rotate with depth, so the early decodable code isn't the one used? (S, descriptive)
- **Done:**
  - `attr_capture.py` + `attr_directions.py` (blocks 15-27, 84 items × 4 questions).
  - Recomputed centroid cosines.
- **Result:**
  - Cosine between attribute centroids at block 16 and block 21: 0.25; block 16 vs block 27: 0.01.
  - Adjacent blocks 20/21: 0.63.
  - Decoding is 1.000 at every block, while steering at L14-16 does 0%.
- **Gap → run:** make it causal. At L22, steer with the **block-16** direction vs the **block-21** direction, at matched norm.
- **Expect:**
  - The block-16 direction at L22 behaves like a random one (≤ a few %).
  - The matched block-21 direction gives ~95%.
  - That would establish that the early code is a different, causally unused direction.

### B5. Does answer-position attention switch from the question to the entity within about two blocks? (S, descriptive)
- **Done:** variance decomposition of the clean `attn_output` captures (no intervention).
- **Result:**
  - Attribute share 0.94 at block 20 → 0.58 at block 21.
  - Item share 0.75 at block 22 and 0.84 at block 23.
  - This brackets the routers (19-21) and the movers (21-23) from activations alone.
- **Gap → run:** the capture used the controlled `Answer:` prompt (I16). Recapture on VADE's own templates (`attr_capture.py` with VADE prompts).
- **Expect:** the same switch at blocks 20-22. If it moves, the R13 comparison with R10/R12 has a prompt-format confound.

### B6. Does the model stop reading the question after a sharp depth? (S)
- **Done:** `attribute_switch_sweep.py`, `earlier_text` residual patching, L1-28.
- **Result:** from L21 onward, patching the question tokens gives 2.4% (= the paraphrase floor), with base kept 100% (n=84).
- **Gap → run:** the "question" scope is effectively tokens 207-211, 213 and 215-217, which includes "Answer". Patch the **field word only**, then "Answer" only, then ":" only, across L1-28 (field-word-only patching is designed but has not been run).
- **Expect:**
  - **If there is a relay:** the field word goes dead earlier (~L17-18), and "Answer"/":" carry the attribute up to L21.
  - **If the field word is read directly:** the field-word curve matches the full question scope.

### B7. Is "block 19 is the keystone" an artefact? (S, re-reading)
- **Done:** `attribute_switch` attention-span sufficiency.
- **Result:**
  - {19,20} = 72.6% and {20,21} = 72.6% (n=84). In the 64-country run: 65.3 vs 69.5%.
  - {19,20,21} = 97.6%.
  - The asymmetry comes from blocks 16-18 partly substituting for block 21, while blocks ≥22 route nothing.
- **Gap → run:** necessity. Patch attention in blocks 16-23, but restore **one** block to base (19, 20 or 21 in turn).
- **Expect:**
  - **No keystone:** restoring 19, 20 or 21 costs about the same.
  - **Keystone:** restoring 19 costs much more.
  - Current data predict similar costs; if so, FINDINGS §2 should be reworded.

### B8. Does one shared router serve every attribute? (?, open tension)
- **Done:**
  - Per-attribute phase 1 rankings in `attr_head_trace`.
  - FINDINGS §3 single-block peaks.
- **Result:**
  - The per-attribute top-8 lists overlap 7-8/8.
  - Yet the single-block peaks differ: calling_code block 19 (45%), capital block 20, currency blocks 19 and 21, language none.
- **Gap → run:**
  - `attr_head_trace.py --per_attribute` with phase-3 sufficiency per attribute.
  - Then install router8 for each (donor attribute → base attribute) pair separately.
- **Expect:**
  - **Shared router:** router8 sufficiency is similar across pairs. The block peaks then track pair geometry (language↔calling_code are the most opposed directions, B4/R3), not different heads.
  - **Otherwise:** a pair whose sufficiency is much lower is using different heads.

### B9. Does the answer format come partly from the base question's tokens rather than the routers? (M)
- **Done:** failure analysis of E4.
- **Result:** 17.7% of composed outputs are the right country in the wrong format (' ITA' or ' SEK' when asked a language under a currency base; ' 392' for a calling code).
- **Gap → run:** split E4's rows by whether the base question X and target A share an answer format (word vs code vs number). Then add an arm with routers expanded to the top-16 (`--router_k 16`).
- **Expect:**
  - Wrong-format failures concentrate in format-mismatched pairs.
  - If the top-16 routers cut "other" sharply, format lives in the extra routers. If not, it comes from the base question tokens via the movers' or other heads.
- **CPU result (`compose_failures`, n=548 eligible rows):**
  - "Other" is 21.0% when X and A have different answer formats (n=452), against 2.1% when they share one (n=96).
  - Of the 95 different-format failures:
    - 80 are unmapped renderings of B in X's format (' ITA', ' SEK', digits);
    - 14 are another country's A value;
    - 1 is B's third attribute.
  - The worst pairs: calling_code→language 41.9%, currency→language 40.0%, calling_code→currency 34.9%. Every →capital pair is 0%.
  - **Verdict:** supported. The routers carry *which* attribute; the output format comes from the base question.
  - The `--router_k 16` arm (GPU) would test whether more routers carry the format.

### B10. Do edits before the handoff give complete multi-token answers, and edits after it only the first token? (S)
- **Done:** `attribute_switch` prefill vs continuous.
- **Result:**
  - Question residual at L1-17: full ≈ first (97.6-100%).
  - Last token at L22-28: first 100%, full 65-73%.
  - Continuous patching never rescues a site whose first token doesn't already transfer.
- **Gap → run:** none needed. This is a methodological claim ready to state.

---

## C. The MLP lookup: refinements of claim 6

### C1. Is the specific fact first created in last-token MLPs 24-26? (S, descriptive)
- **Done:** `attr_capture` variance decomposition of the clean last-token captures (no patching).
- **Result:**
  - The item × attribute interaction share of mlp_output is 0.03-0.05 at blocks 15-22, then 0.08 / 0.19 / 0.27 / 0.40 at blocks 23 / 24 / 25 / 26.
  - Residual interaction is ≤3.4% through block 22.
- **Gap → run:** recapture on VADE templates (same as B5).
- **Expect:** the same rise in blocks 24-26. This gives a no-intervention corroboration of E2 and E3.

### C2. Is the lookup spread over several late blocks rather than one? (M)
- **Done:**
  - `ndm/ceiling_sweep.py` with `blocks:N` at `last_token`.
  - `head_severed.py --freeze_values base mean other` for single blocks (R16.5).
- **Result:**
  - `blocks:1` at L25-28: ≤9.4%. `blocks:3` (25-27): 75%. `blocks:4` (24-27): 94% for language.
  - Freezing a single MLP at the mean recovers most of the transfer (capital 66-81%, language 89-98%). So R16.2's "each block costs 30-100%" was mostly the base answer being re-imposed.
- **Gap → run:** leave-one-out over the span. Freeze 23-26 at the mean **except** one block, in turn (`--freeze_mlp_span 24-26`, `23,25-26`, ...; non-contiguous spans need a small parser change). Then try pairs.
- **Expect:**
  - **Distributed and redundant:** no block alone restores the answer, but any 3 of 4 come close.
  - **Serial:** removing any single block breaks it.
  - The current data predict distributed and redundant, except MLP 25 for calling_code (C3).

### C3. Is MLP 25 a single-block bottleneck for calling_code (and currency)? (M)
- **Done:** `head_severed` single-block freezes at base, mean and other.
- **Result:**
  - Frozen at the mean: calling_code 25.5%, currency 29.7%; the other blocks recover.
  - Frozen at a third country: that country's code 100% at MLP 25.
- **Gap → run:** a second seed with n=128 (`--n_pairs 128 --seed 1`). Score currency with text matching.
- **Expect:** the bottleneck holds for calling_code and is weaker for currency, where the many shared currencies make the mean less damaging.

### C4. Does MLP 27 amplify and format the answer rather than decide it? (M)
- **Done:**
  - `dla.py`.
  - `head_severed` with MLP 27 frozen at base.
- **Result:**
  - MLP 27 is the largest single direct writer for capital, currency and calling_code (38-43%) and for language (23%).
  - Frozen at base, first-token transfer stays at 78-88%, but 30-72% of answers become "other".
- **Gap → run:** freeze MLP 27 at a third country (`--freeze_mlp_singles 27 --freeze_values other`).
- **Expect:**
  - **Amplifier:** the third country's answer appears on few rows (<20%), and the source answer still dominates the first token.
  - If the third country wins, MLP 27 decides too, and the "23-26" framing is wrong.

### C5. Do attributes separate only at the MLP neurons, with calling_code reading private directions? (M)
- **Done:** `readout_jacobian.py --site mlp_hidden --blocks 23-26 --max_rank 128`, with two-fold item-held-out geometry and VADE-scored eval arms.
- **Result:**
  - calling_code's readout rank is 6-8 at blocks 25-26, with ≤0.6% overlap with the others.
  - Capital, currency and language overlap each other 10-29% (chance ≤0.7%).
  - The isolated edit keeps 44-62% cause with 91-95% iso.
- **Gap → run:** sweep the rank (32 / 64 / 128 / 256). The rank-128 eval (`flags_rank128`) is partial: capital only.
- **Expect:**
  - calling_code saturates by rank ~8.
  - The other three keep gaining up to the cap, so their shared fraction is a lower bound.

### C6. On brands/celebrities, does late attention work against the head install? (?/M; contradicts README E7)
- **Done:** `head_severed.py --gold own`, freezing attention 24-27 (128 pairs each).
- **Result:**
  - Freezing late attention **raises** transfer: brands 64.6 → 73.2% (base) / 82.7% (mean) / 84.3% (other); celebrity nationality 76.2 → 81-85%.
  - It is flat for native_language and flags.
- **Gap → run:** cut the answer column's attention to the **image** in blocks 24-27 during the install (the `route_knockout.py` knockout `readout:24-27`). That script is flags-only today and needs an `--entity` option.
- **Expect:**
  - **Late attention re-reads the base image:** cutting image edges in 24-27 reproduces the 73-84% gain.
  - **Something else:** cutting them changes nothing, meaning it is re-reading text positions.
  - The first outcome would extend claim 4: the direct route keeps being consulted after the movers.

### C7. Does steering decay late because item-specific content grows there? (M)
- **Done:** R4 steering curve, read against the `attr_capture` decomposition.
- **Result:**
  - `add` vs full swap: L24 92 vs 100, L26 79 vs 100, L28 44 vs 100.
  - The item-independent share of question variance falls from 0.97 (block 21) to 0.66 (block 26).
- **Gap → run:** at L26-28, steer with a **per-country** direction (each country's own attribute-centroid difference) vs the global one.
- **Expect:** the per-country direction restores ~100% at L26-28 while the global one decays. That confirms the decay is item-specific content, not a weaker attribute signal.

---

## D. Where the entity lives on the image side

### D1. Is the image-side entity code set by the vision encoder and never locally rewritten by the language model? (S for language)
- **Done:** `ndm/ceiling_sweep.py` at `flag_ring1` and `full_image`, on every site: residual, attn_output, mlp_output, attn_head_output, mlp_hidden, joint, `blocks:1..5`. L1-28.
- **Result:**
  - Every single sublayer, the joint attn+mlp swap and every 1-5-block span read 0% at image positions at every layer.
  - A residual swap reads 62-100%.
- **Gap → run:**
  - **(a)** Run `blocks:N` for capital, currency and calling_code.
  - **(b)** The decisive test: at L10 and L20, swap **every** language-model sublayer write at the image positions (`blocks:L`, i.e. everything except the layer-0 vision embedding).
- **Expect:**
  - (a) 0% throughout.
  - (b) ~0% cause, while swapping only the layer-0 embedding gives 100% (already measured).
  - Together these would show that the entity at the image tokens *is* the vision encoder's output, carried unchanged by the residual stream.

### D2. Does identity spread outward from the object within the language model? (M)
- **Done:** E11 position-set ceilings (96 rows).
- **Result:**
  - `~entity_only` rises 34% (L10) → 68% (L18), while `entity_only` falls 34 → 16%.
  - Most of the gain is in the one-token ring around the object.
- **Gap → run:** a `ring:K` sweep (K = 0..4) and its complements, at L2-22.
- **Expect:**
  - ring:1 rises over L10-18 while ring:0 falls; ring:≥3 stays small.
  - Report each ring alongside its complement, since partial patches measure a conflict between source and base regions, not a share of identity.

### D3. Is celebrity identity more confined to the object than flag or logo identity? (M)
- **Done:** E11.
- **Result:** `entity_only` at L10 is celebrities 83%, brands 50%, flags 34%.
- **Gap → run:** match token counts by patching a random 8 of the face's 16 tokens.
- **Expect:**
  - **Faces really are more local:** 8 face tokens still clearly exceed flags' 34%.
  - If they fall to ~34%, the difference was token count.

### D4. Does the vision encoder leak object information into the preceding image tokens? (M; overturns the design assumption)
- **Done:** E11 `seq:before` vs `seq:after`.
- **Result:** before 15-24% > after 4-12% in all three entities, and before *rises* within the language model on brands and celebrities.
- **Gap → run:** compare the layer-0 residual at `seq:before` tokens between base and source images (bit-identity). This is an open item in the README.
- **Expect:**
  - Nonzero differences (the vision encoder's full attention), and the layer-0 `seq:before` cause ≈ 15-24%.
  - Any later rise under causal masking must then come from the encoder-mixed tokens interacting, not from the object.

### D5. Do category attributes leave the image 7-12 layers earlier than entity facts? (M)
- **Done:** `image_head_pipeline.py` stage 1 on animals, celebrities and brands (n=32 per layer).
- **Result:**
  - Image cause falls to ≤50% by L10-15 for body_covering, primary_habitat, reproduction, taxonomic_class, occupation and industry.
  - It falls only at L20-23 for nationality, native_language, hq_country and native_continent.
  - Within celebrities: occupation by L14 vs nationality by L22.
- **Gap → run:** re-run stage 1 at n=128 with `--text_match`.
- **Expect:** the 7-12-layer gap persists. The within-entity contrast (celebrities, brands) rules out entity-type confounds.

### D6. For category attributes, is there a window where neither the image nor the last token matters (a two-hop route)? (?)
- **Done:** only inferred, from the stage-1 curves.
- **Result:**
  - body_covering at L16-20: image 6%, last token 6%.
  - taxonomic_class at L16-19: 12-16% vs 0%.
- **Gap → run:** `ceiling_sweep.py --entity animals --attribute body_covering --positions_list phrase:attribute vision_end tok:-2 tok:-5 pre_image --layers 10-24 --text_match`.
- **Expect:**
  - **Two-hop:** some text position (most likely the question's attribute words, or `vision_end`) carries high cause in the L14-20 window.
  - If every position reads ~0, the attribute is spread too thinly for a single-position-set swap to move it.

### D7. Do image tokens hold the facts late but leave them unused? (M)
- **Done:** E1 logit lens over all 144 image tokens and the object's 8.
- **Result:**
  - At L26: name top-1 97%, capital 53%, language 50% (best token).
  - Image patches are causally dead from L23.
- **Gap → run:** take the max over fixed-size sets (8 object vs 8 random background tokens, averaged over draws), not 144 vs 8.
- **Expect:** facts decodable at both, at similar rates. Since MLPs act per position, the same lookup runs wherever the entity is represented.
- **CPU result (`late_facts`):** with size-fair pooled readouts, the object and background decode facts at similar rates.

  | L26 top-1 | name | capital | language |
  |---|---|---|---|
  | object, pooled | 0.52 | 0.30 | 0.38 |
  | background, pooled | 0.75 | 0.38 | 0.33 |
  | object, max over tokens | 0.60 | — | — |
  | background, max over tokens | 0.97 | — | — |

  - The max-over-tokens readout inflates the larger region, which is the selection bias.
  - Up to L23, every region reads ≤0.06.
  - **Verdict:** supported. The facts appear at every image position at L24-26, after the handoff.

### D8. Does the answer position linearly encode the attribute before it matters causally? (M)
- **Done:** E1 leave-one-country-out probes at the last token.
- **Result:** at L20, language 0.86 (chance 0.31), currency-EUR 0.93, calling zone 0.76, while a last-token residual swap at L20 = 0%.
- **Gap → run:** pre-register a layer (or correct for choosing the best one), and report every layer.
- **Expect:** the same conclusion, as a third instance of G3 (decodable but not causal).
- **CPU result (`lasttoken_probe`), all layers:**
  - Language at the last token reads 0.31 (L0), 0.47-0.59 (L2-18), **0.86 at L20**, and 0.92-0.94 (L24-26).
  - currency-EUR: 0.50 → 0.93 at L20. calling_zone: 0.24 → 0.76 at L20 → 0.90 at L26.
  - Above-chance decodability starts by L2, and the sharp jump (L18 → L20) comes two to three layers before the last token becomes causal (L23-24).
  - **Verdict:** supported at every layer, not just the best one. This is the cleanest instance of G3 (decodable but not causal).

### D9. Is a text-named entity gathered onto the name's last token within two blocks, and does it persist longer? (M)
- **Done:** E6 curves.
- **Result:**
  - Text patch: 88% at L0, 94% at L1, 99% at L2.
  - At L22, text 87% vs image 67%.
- **Gap → run:** the Geva-style early enrichment test (`--entity_layer` early, e.g. 4 4) is listed as open.
- **Expect:** the freeze phase becomes informative. Freezing the name position's MLPs in blocks 4-15 at the mean lowers text recall but not image recall, if the text route has a subject-enrichment stage that the image route skips.

---

## E. Routes: refinements of claim 4

### E1. Is the question-token fallback read in the same narrow window as the image? (S)
- **Done:** `route_knockout.py`: a question-position patch plus the `readout_all` knockout, with `--q_layers` 4-24 (n=472).
- **Result:** ≤4.4% up to L18, 17.8% at L20, 83.9% at L22, 1.5% at L24.
- **Gap → run:** none needed. State it.

### E2. Is currency the attribute most dependent on the direct image route, and does that explain the Park result? (M)
- **Done:** E5, E9, E4 and the ceilings.
- **Result:**
  - With the readout cut: currency 68.6% vs 85-96% for the others.
  - Question fallback at L22: currency 61% vs 87-94%.
  - Zeroing Park's heads: currency −20.8pp vs −3 to −9.5 for random sets.
  - Currency also persists longest at `flag_ring1` (75-81% to L19).
- **Gap → run:** re-score these arms on the rows where base and source currencies are unique (excluding EUR/USD/XOF clusters). Re-run `vrh_ablation.py --attributes currency --n_random 10`.
- **Expect:**
  - **Real route dependence:** the gap persists on unique-currency rows.
  - If it vanishes, shared currency values were inflating it.
- **CPU result (`currency_unique`):** restricted to rows where both the base and the source currency are unique (excluding EUR ×10, the only shared currency):

  | arm | all rows | unique-currency rows |
  |---|---|---|
  | readout cut + image patch | 68.6% | 68.8% (n=93) |
  | question fallback at L22 | 61.0% | 59.1% |
  | zero Park's heads (accuracy) | — | 56.8% |
  | zero random sets (accuracy) | — | 78-79% |
  | clean (accuracy) | — | 82.4% |

  - The route_knockout row pairs were rebuilt from the seed and checked against every row's attribute.
  - **Verdict:** the currency-specific dependence is **not** an artefact of shared currencies.

### E3. Does language have a route that doesn't need the movers, which calling_code lacks? (M)
- **Done:** E9, zeroing `common10`.
- **Result:**
  - Language 75.6% (clean 93.5%), capital 37.5%, currency 27.4%, calling_code 11.3% (n=168 per attribute).
  - There is no matched random control: `vrh_ablation.py` draws block-matched random sets only for the published lists, not for `--movers`.
- **Gap → run:**
  - **(a)** Add block-matched random draws for the movers list (about three lines next to `block_matched_random`), 5 draws in blocks 21-23.
  - **(b)** Check whether language's surviving answers concentrate on high-frequency languages (English, Spanish, French, Arabic).
- **Expect:**
  - (a) Random-10 costs ≤5pp, which makes "38%" mover-specific.
  - (b) **Guessing from visual similarity or priors:** survivors are mostly common languages, matching the layer-0 probe of 0.71. A second circuit would instead preserve rare languages too.

---
- **CPU result (`survivors`, part b; clean-correct rows):**

  | attribute | survives zero:`common10` | what the wrong answers default to |
  |---|---|---|
  | language | 80.3% (n=157); shared languages 86.6% (n=97), unique 70.0% (n=60) | mostly 'English' |
  | capital | 42.9% | Santiago ×24, Madrid ×13, Buenos Aires ×11, Brasília ×7 |
  | currency | 31.7%; EUR countries only 5.9% (n=17) | — |
  | calling_code | 11.7% | 91, 237, 230 |

  - **Verdict:** language survives more for shared languages, consistent with guessing from priors or visual similarity. But even unique languages survive 70%, so language has a real mover-independent route.
  - The wrong capitals default to Spanish-speaking ones.
  - The matched random-10 control (part a) still needs a GPU.

## F. Across models

### F1. Is the handoff at a fixed relative depth (~0.8 of the stack)? (W)
- **Done:** E8 stage 1 (7B 28 layers, 3B 36 layers, InternVL 28 layers).
- **Result:** 7B and InternVL drop at ~0.79-0.82 of depth; 3B at 0.78 and then 0.89.
- **Gap → run:** a model with a different depth, e.g. Qwen2.5-VL-32B (64 layers). This needs a VADE accuracy sweep and pruning first.
- **Expect:**
  - **Relative depth holds:** the drop is at ~L50-52.
  - **Absolute depth matters:** it is near L21-23.

### F2. Does the 3B model hand off in two separate steps, each with its own heads? (M)
- **Done:** E8 stages 1 and 3 (n=32 per layer).
- **Result:**
  - Image cause 100% through L27, 28% at L28-31, 6% from L32.
  - Consensus heads 27.10-12 and 31.1/31.3/31.7/32.13.
- **Gap → run:** stage 1 at n=128. Then run `head_trace.py` separately on windows 27-28 and 31-33, and install each step's heads alone.
- **Expect:**
  - The first step's heads give ~60-70% first-token transfer, and the second step's heads supply the rest.
  - The two steps compose (both installed ≈ the whole window).

### F3. Is calling_code never carried by 8 heads in any model? (S, but likely an artefact)
- **Done:** E8.
- **Result:** top-8 gives 1.6% in all three models; the whole block window gives 72-98%.
- **Gap → run:** a continuous (every-step) head install for calling_code on 3B and InternVL, as on 7B.
- **Expect:** a jump to ~90%+, as on 7B (6.2% → 95.5%). The claim then becomes "a last-column patch can't steer a multi-token answer" (G5), not "calling_code is distributed".

### F4. Does the vision encoder affect how often the entity is recognised? (W)
- **Done:** pruning logs (`logs/tier2/e8_*_prune_*`).
- **Result:** InternVL prunes 44-48% of celebrity tuples vs 28-31% for Qwen-3B. For flags, InternVL prunes 34.8% vs 24.3% for 3B.
- **Gap → run:** the same accuracy sweep and pruning for Qwen-7B, then per-entity recognition accuracy for all three models.
- **Expect:**
  - InternVL lowest on celebrities.
  - Since the language model is shared with Qwen-7B, any gap there is attributable to the vision encoder.

---

## G. Intervention methods (the VADE-facing claims)

### G1. Is any intervention at image positions an entity swap rather than an attribute edit? (S)
- **Done:** `intervene.py` (PCA and SAE, both position sets, 14,052 predictions each) and the DBM test summaries, all scored with `score.py`.
- **Result:**
  - PCA `flag_ring1`: language cause 51%, while the other three attributes flip to the source 61-67% of the time.
  - Kaggle-pool SAE `flag_ring1`: 19.2% cause vs 20-22% source matches on the others.
  - DBM's iso failures are also source matches (but see Z1).
- **Gap → run:**
  - **(a)** A random feature subset of the same size in `intervene.py`.
  - **(b)** Report an "entity-swap index" for every method: the mean rate at which non-target attributes match the source, divided by cause.
- **Expect:**
  - (a) Random features give ~0 cause.
  - (b) An index ≥1 for every image-position method, and ≪1 for the E3 MLP readout edit (iso 76-92%).
  - That contrast is the headline VADE result: at image positions only entity swaps are possible; attribute-specific edits exist only after the lookup.
- **CPU result (`entity_swap`, every VADE-format file on disk, 108 file × target cells):** median entity-swap index by family.

  | family | index | reading |
  |---|---|---|
  | image-site PCA/SAE | 1.07-1.15 | entity swap |
  | `full_image` patch | 0.94 | entity swap |
  | head installs (`common5` / `common10` / `top8`) | 0.96 (0.64-1.64) | entity swap |
  | full swaps at heads or MLPs | 0.93-0.95 | entity swap |
  | readout edit at the heads | 0.83 | entity swap |
  | **readout edit at MLPs 23-26** | **0.00-0.05** | attribute edit, non-target "neither" 6-22% |
  | trained masks at MLPs | 0.02-0.18 | attribute edit |
  | trained DAS at the heads, K ≤ 256 | 0.05-0.37 | see below |
  | DBM at image positions, L6-8 | 0.85-0.96 | entity swap |
  | DBM at image positions, L14-22 | 0.38-0.50 | partial; Z1 pending |

  - **The head-site DAS refinement:** its low index comes with 13-35% of non-target answers *destroyed* (neither base nor source). The selectivity at the heads is bought by damage, not achieved by attribute-specific transfer. The three void das_rotated arms reproduce `k512` exactly (index 0.94), as R12.3 says.
  - **Verdict:** supported. Only edits after the MLP lookup separate attributes cleanly.
  - The random-feature control in `intervene.py` (part a) still needs a GPU.

### G2. Does the feature-selection proxy rank methods in reverse order of real cause, mostly by choosing layers past the handoff? (M)
- **Done:** `select_features.py` winners vs `intervene.py` scores vs ceiling logs.
- **Result:**

  | method | proxy cause | real cause | full-swap ceiling at that layer |
  |---|---|---|---|
  | Kaggle SAE `flag_only` (L23) | 0.755 | 2.2% | 0% |
  | real-only SAE (L24) | — | 1.5% | 0% |
  | Kaggle SAE `flag_ring1` (L4) | — | 19.2% | 65-67% |
  | PCA (L2/L7) | — | 30-51% | 62-72% |

- **Gap → run:** compute, from `sweep.jsonl` and the ceiling logs, the rank correlation between proxy score and full-swap ceiling across layers (CPU). Then run `intervene.py` for each method at one fixed early layer (e.g. L4) to compare dictionaries at matched depth.
- **Expect:**
  - The proxy is uncorrelated or negatively correlated with the ceiling.
  - At a matched layer, the dictionary comparison changes. Real-only vs Kaggle SAE was confounded by layer (L24 vs L4).
- **CPU result (`proxy_ceiling`, flags/language, 29 layers):**
  - **Spearman correlation**, proxy cause vs the full-swap ceiling:

    | selection | vs flag_ring1 ceiling | vs full_image ceiling |
    |---|---|---|
    | PCA | +0.41 / +0.60 | — |
    | Kaggle-pool SAE | −0.38 / −0.43 | −0.52 / −0.54 |

  - **Winners:**

    | winner | layer | ceiling there | real cause |
    |---|---|---|---|
    | `flag_only_sae` | L23 | 0% (`full_image` 15.6%) | 2.2% |
    | `flag_ring1_sae` | L4 | 63% | 19.2% |
    | PCA | L2/L7 | 62-63% | 30-51% |

  - **Verdict:** partly supported. The SAE proxy is anti-correlated with causal headroom. Layer choice explains `flag_only_sae`'s failure completely, but not `flag_ring1_sae`'s: at L4 the headroom is there (63%) and the SAE features still capture only 19%.
  - **Caveat:** no flag_only ceiling exists, so the flag_ring1 one stands in.

### G3. Does being decodable fail, repeatedly, to predict causal effect? (S)
- **Done / Result:** four separate instances.
  - R2: 1.000 decoding at blocks where steering does 0%.
  - R11.5: a 7-dim variance subspace decodes language at 100% but gives 0% causal effect.
  - G2: the proxy vs the real gap.
  - D8: the last-token probe at L20.
- **Gap → run:** none needed. This is a methodological claim ready to state.

### G4. Do trained edits at the mover heads plateau around 57-58% VADE final? (M)
- **Done:** `head_das.py` das_fixed (K = 8-512) and das_rotated (four `mask_coef` values).
- **Result:** 14 non-void arms land at 57.2-58.3% final, against full-swap finals of 38.5-50.1%.
- **Gap → run:**
  - das_fixed is compared at 40 optimizer steps vs das_rotated's 375 (Z6). Re-run das_fixed with `--grad_accum_steps 1 --train_rows 6000 --eval_rows 2000`.
  - Evaluate `language_das_fixed_k128_full` (`--skip_train`).
- **Expect:**
  - das_fixed rises to the same ~57-60%. The "2.8× from sparsification" claim disappears, and the plateau claim gets stronger.
  - A clear break above 60% would mean the plateau was under-training.

### G5. Is a last-token edit capped by the fraction of an attribute's answers that are a single token? (S)
- **Done / Result:**
  - The ranking holds 4/4: 79 / 37 / 24 / 0% single-token golds ↔ 100 / 70 / 61 / 6% sufficiency.
  - Continuous patching lifts calling_code from 6.2% to 95.5%.
- **Gap → run:** none needed. State it as a measurement caveat for any last-token method.

### G6. Does the logit lens lag the causal location by 2-4 layers? (S)
- **Done:** `logit_lens.py` vs the ceilings.
- **Result:** a last-token swap is decisive at L24, but the lens top-1 rises at L26-27 (calling_code 0.02 → 0.97 at L27).
- **Gap → run:** none needed.

### G7. Do the attention backend and batch shape change clean answers? (S)
- **Done:**
  - `attr_steer` sdpa vs eager on the same 192 rows.
  - `attribute_switch` serial vs cached decoding.
- **Result:**
  - Base kept 90.6% (sdpa) vs 78.1% (eager).
  - All 12,867 self-control mismatches sit in 4 (country, attribute) cells (e.g. MU currency 'MGF' vs 'RSD', HU ' 366' vs ' 36').
- **Gap → run:** the paired serial-vs-cached check, and `--verify_cached` on GPU (both designed, not run).
- **Expect:** agreement outside those 4 cells. Report the backend for every number in the paper.

---

## Z. Things to fix before claiming anything near them

| # | Problem | What to do | Where |
|---|---|---|---|
| Z1 | **DBM cause exceeds its own full-swap ceiling** (L16: 60-62% vs 31% at `flag_ring1`; L22: 58.5% vs 15.6%). A mask cannot do that. The runs (2026-09-09/10) predate the BuildBatchCache fix (518cf31, 2026-09-14), and each built a `full_image` cache first. The DBM curve matches the `full_image` ceiling's shape. | Re-run `dbm/eval.py` on the existing checkpoints with current code; expect cause to drop to ≤31% at L16. Don't cite DBM until then. | `results/Qwen2.5-VL-7B-Instruct/flags/dbm/language/*`, `logs/dbm_*` |
| Z2 | brands/founded_year is called "inert", but its image patch moves it 78-94% (L5-12). The 0% is first-token scoring of `['Ġ','1','9',...]`. | Score it at the point where the base and source years diverge, or by text match. | README Tier-2 caveat 5; `image_head_pipeline/brands/SUMMARY.md` |
| Z3 | The name-rank chance level is wrong. There are 156 candidates (two spellings each), so the best-of-two null is ≈52, not 39. The same applies to brands (≈74) and celebrities (≈49). | Recompute the null (A5). | R16.1, README, CLAUDE.md |
| Z4 | The n=16 "replication" is a byte-identical subset of n64. n256 adds no new pairs. | Run seed 1 (A6). | R11.4 / R11.2b |
| Z5 | R10.4's "clean scores highest" is false for 2 of 3 targets. The currency figure is 85.4%, not 87%. | Edit the doc. | R10.4/R10.5 |
| Z6 | R14.2's 2.8× gain comes with ~9× more optimizer steps. | Do G4. | R14.2 |
| Z7 | R13.3 and the R16.5 follow-ups are listed as "not run", but both have run. | Edit the doc. | ATTRIBUTE_HEAD_EXPERIMENTS |
| Z8 | "MLP 27 ≈40% for every attribute" is wrong: language is 23%. | Edit the doc. | README E2 |
| Z9 | "Late attention does not matter" (brands/celebrities) is contradicted (C6). | Edit, then run the C6 test. | README E7 |
| Z10 | RESULTS.md lists the real-only SAE comparison as not done, but it exists (49cf0b2: 1.5% cause, 47.9% final). | Edit the doc. | RESULTS.md |
| Z11 | CLAUDE.md says the L14 residual "flipped the answer outright". One log shows 0%, newer ones 47-50%. | Edit the doc. | `logs/ndm_ceiling_sweep_manual.log` |
| Z12 | HEAD_FOLLOWUPS says token-edge knockout has "no measurable effect", but it drops log-probs by 0.2-3.7 nats, against ≤0.03 for random edges. | Report the logit result (n=4; extend to 64). | `head_followups/flags/*/token_knockout` |
| Z13 | sweep_report and FINDINGS report the same arms with different indexing (1- vs 0-based) and different metrics (full vs first). | Pick one convention. | — |
| Z14 | "Image encoding is question-independent" (E0b) is guaranteed by causal masking when the question follows the image. | Call it a sanity check, not a finding. | README E1 |
| Z15 | The head_trace table says image positions read 100% at L22 / 0% at L23. The logs show 87.5 / 15.6% (`full_image`), and E6 shows 67 / 19%. | Edit the doc. | README head_trace section |
