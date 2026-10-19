# Paper TODO: experiments for the final paper

How to use this file:
- Each item is an experiment, check or write-up step that the paper needs.
- When an item is done, tick it and strike it through: `- [x] ~~item~~`. Then add one line saying
  where the result is and what it showed.
- Add new items under the claim they serve. Never delete an item; strike it out instead.
- IDs (A3, P6, ...) refer to CLAIM_CANDIDATES.md and FOLLOWUP_PLAN.md.
- Priority: **[P0]** needed for submission, **[P1]** strongly expected by reviewers, **[P2]** nice to have.
- GPU commands for queued experiments live in `PAPER_RUNBOOK.md` (X-numbers), run as one batch.

Working story: **see → move → recall.** Movers (blocks 21-23) carry the whole entity. Routers (blocks
19-21) carry which attribute was asked. Last-token MLPs 23-26 look the fact up. The contrast with
text-LLM recall (Geva et al. 2023) is that the entity is moved first and the fact is recalled at the
answer position, not enriched at the subject.

---

## 0. Generality (biggest reviewer risk)

- [ ] **[P0]** Replicate movers, routers and the MLP lookup on a model whose language model is not
  Qwen (Gemma-3, Llama-3.2-Vision or Idefics3). Cover flags plus one other entity. InternVL3 shares
  Qwen2.5-7B's language model, so it does not count as generalization.
  - Chosen: **Idefics3-8B-Llama3** (Llama-3.1 LM; every architecture assumption holds).
    Llama-3.2-Vision is ruled out: its images enter by cross-attention, so there are no image
    positions. Gemma-3 is second choice (head_dim ≠ hidden/heads, post-sublayer norms).
  - **X3** in PAPER_RUNBOOK.md §5: needs an adapter plus generic prompt-builder fixes before it
    can be queued.
- [ ] **[P0]** Run the full story on brands and celebrities, not just head localization:
  cross-question mover patching, router tracing, MLP 23-26 freezing (`head_severed --gold own`).
  - Already done: head localization, `head_severed` on hq_country / nationality /
    native_language, `head_vocab_projection`.
  - **X2** (queued, runs today): continuous mover install, cross-question patching, MLP
    freezing on founded_year / industry / occupation / birth_year.
  - **X2b** (PAPER_RUNBOOK.md §5): routers, composition and text contrast. Needs per-entity
    controlled prompts and own-answer golds first.
- [ ] **[P1]** Relative-depth (~0.8) claim: Qwen2.5-VL-32B stage 1 (P34, F1). Otherwise weaken the
  claim to "late, block window per model".
- [ ] **[P1]** 3B two-step handoff: separate windows plus a continuous install for calling_code
  (P18, F2, F3).
- [ ] **[P2]** Per-model recognition accuracy, to attribute differences to the vision encoder (F4).

## 1. Movers carry the entity

- [x] ~~Held-out mover selection (P3, A3)~~ Selected on 59 countries, evaluated on 25: the selected
  set is identical to common10; transfer 72-98%, random 0-2%. `results/tier3/heldout_heads/`,
  `results/tier3/head_swap_vade_heldout/`
- [x] ~~Matched random null for zeroing the movers (P2)~~ −51 points vs ≤1.3 for random.
  `results/tier3/vrh_mover_random/`
- [x] ~~Replication of head_cross on new pairs (P5, Z4)~~ Every cell is within the CI.
  `results/tier3/head_cross_seed1/`
- [x] ~~Logit-level conjunction test (P9, A1)~~ `results/tier3/mover_subsets/`
- [ ] **[P1]** Write up P9's verdict: is the movers' effect superadditive (a true conjunction) or
  additive below a threshold? Compare singles, pairs and leave-one-out `ld` in `summary.json`.
- [x] ~~Continuous restore for calling_code (P11, A2)~~ Prefill 6.2% → continuous 95.3%. The
  multi-token artefact, not a second route. `results/tier3/decode_calling_code_k10/`
- [x] ~~Object vs background key knockout (P27, A9)~~ Object keys 94.4%, background 90.6%, all
  mover keys 45.1% (question route cut). `results/tier3/route_ko_groups/`
- [ ] **[P1]** Regional entity code: dose (α) install and continent probe (P28, A8).
- [ ] **[P1]** Settle the P5 caveat: under `--align matched`, the −16pp "donor asked a different
  question" shift is not yet attributable to content.

## 2. Image and text share the circuit

- [x] ~~Early entity-position freeze for text vs image (P22)~~ Text depends on early entity MLPs
  (0.9%), images much less (32.3%). `results/tier3/text_Le4_freeze/`
- [ ] **[P0]** Image-vs-text convergence: at which layer do image-run activations at the answer
  position (and at the entity) become similar to the text-named run? CKA or cosine per layer, plus
  cross-patching text→image runs per layer. This answers Nikankin et al. 2025 directly.
  - Implemented as `text_entity_contrast.py --phases crossmodal` (2026-10-19, not yet run;
    queued as **X1** in PAPER_RUNBOOK.md):
    cross-modal retrieval per layer / per mover head (centered and held-out ridge), plus
    cross-condition mover installs (text-named B's movers into image-of-A and the reverse) with
    same-condition, norm-matched, random-10 and `self` controls. Offline tests in
    `tests/test_crossmodal.py` have not been run (no local torch).
- [ ] **[P1]** Same-heads claim with a text-run head trace on brands/celebrities, not just flags.

## 3. Category vs entity attributes

- [x] ~~common10 installed on animals (P14, A10)~~ native_continent 49.6%; body_covering,
  habitat, reproduction and taxonomic_class 1-7%. `results/tier3/head_swap_animals/`
- [ ] **[P1]** Stage-1 curves at n=128 for the category-attribute timing (P16, D5).
- [ ] **[P1]** Animals text-position ceilings, looking for the two-hop route (P17, D6).
- [ ] **[P1]** Write a precise definition of "category attribute" (shared by a class and readable
  from appearance) vs "entity fact", and list which VADE attributes fall in each.

## 4. Routers carry the queried attribute

- [x] ~~Held-out routers and routers traced to block 27 (P6, B1)~~
  `results/tier3/attr_heads/`
- [x] ~~Reconcile router8 with P6's "top-32"~~ This is a ranking shift caused by late blocks. Router8
  holds: 87.3% held-out first-token sufficiency, 96% removed by restoring it. Write "8 heads", first
  token. `results/tier3/attr_heads/RECONCILE_ROUTER8.md`
- [x] ~~Per-attribute routers (P19, B8)~~ Every attribute's top-8 shares 7-8 heads with router8.
- [x] ~~Is block 19 a keystone? (P25, B7)~~ No; reword ATTRIBUTE_SWITCH_FINDINGS §2.
- [x] ~~Code rotation made causal (P12, B4)~~ The layer-17 direction does 0% at L22; L22's own
  direction gives 95.8%.
- [x] ~~Composition with router_k 16 (P20, B9)~~ The answer-format failure shrinks only slightly.
- [ ] **[P1]** Steering robustness: seeds, every-step application, sdpa vs eager (P24, B3).
- [ ] **[P1]** Attribute capture on VADE templates, for the prompt-format confound (P32, B5).
- [ ] **[P2]** Per-country steering directions (P33, C7).

## 5. Late MLPs do the lookup

- [x] ~~MLP 23-26 leave-one-out, MLP 25/27 at third-country values, new seed (P10)~~ The lookup is
  distributed over 23-26; MLP 25 is needed for calling_code; MLP 27 formats but does not decide.
  `results/tier3/severed_loo_n128_s1/`
- [x] ~~Gate=0.5 control for the trained DBM masks (P4)~~ Trained masks beat it by 18-43 final
  points. `results/tier3/head_das_gate0.5/`
- [ ] **[P1]** Readout rank sweep at the MLPs (P13, C5).
- [ ] **[P2]** Brands: cut late attention to the image during the install (P15, C6).

## 6. Payoff: mechanism → better disentangling edit (VADE)

- [ ] **[P0]** Headline comparison on VADE final score: the readout edit at MLPs 23-26 vs DAS, DBM,
  SAE and PCA at image positions, on every entity, at matched training budget.
- [ ] **[P0]** DAS at a matched training budget (P29, G4, Z6).
- [ ] **[P1]** Dictionary controls: random features, fixed layer (P30, G1, G2).

## 7. Methodological findings (side contributions)

- [x] ~~DBM above the full-swap ceiling (P1, Z1)~~ The eval path is sound: a learned blend beats the
  full swap at image positions. `results/tier3/dbm_identity/`
- [x] ~~Does the vision encoder spread the object into earlier tokens? (P8, D4)~~ `seq:before`
  reads 15.6%, so the encoder is not null before the object.
- [ ] **[P1]** Cached-decoding engine: P26 `--verify_cached` failed (max |dlogit| 0.23 sdpa). Either
  loosen the tolerance with a justification or fix the engine before citing attribute-switch numbers
  that used it.
- [ ] **[P2]** Token-edge knockout at n=64 (P31, Z12).
- [ ] **[P2]** Currency with 10 random draws (P21, E2).

## 8. Writing and positioning

- [ ] **[P0]** Related work and positioning: Geva et al. 2023 (dissecting recall), Neo et al.
  ICLR 2025, Zhang et al. (cross-modal information flow), Nikankin et al. 2025 (same task, different
  circuits), Park et al. (visual retrieval heads), Lietzow et al. Verify each one's exact claims.
- [ ] **[P0]** Cut to 3-4 headline claims; move the rest to the appendix.
- [ ] **[P1]** Fix stale numbers in docs before quoting them: Z2, Z3 (name-rank null 52.4), Z5, Z7,
  Z8, Z9 (CLAIM_CANDIDATES.md §Z).
- [ ] **[P1]** Report confidence intervals clustered by country throughout.
