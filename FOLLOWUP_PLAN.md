# Tier-3 follow-up plan: the GPU experiments left open in CLAIM_CANDIDATES.md (2026-10-18)

Every command below was checked against the current code (HEAD `fb0a6dd`): argparse
flags, spec syntax, default output paths, and what each script actually measures.
Experiments that need a code change say so and point to section 2. Those changes must
land, with tests, **before** the pod session. GPU_RUNBOOK.md's rule stands: no code
edits on the pod.

Claim IDs (A1, Z1, ...) refer to CLAIM_CANDIDATES.md.

---

## 1. Rules that apply to every run

1. **Always pass an explicit `--out` / `--out_dir`.** No script encodes seed, split,
   alpha, router_k, sites or training budget in its default output name. These scripts
   **overwrite committed results** if run with defaults:
   - `vrh_ablation`, `head_cross`, `circuit_compose`, `head_severed`, `readout_jacobian`
     and `head_das` (they open outputs with `"w"`);
   - `text_entity_contrast` (it truncates `rows.jsonl`);
   - `attr_head_trace` and `attr_steer` (their tags lack seed and countries);
   - `ceiling_sweep` (its tag lacks the site list).

   **Append-and-resume scripts silently skip finished rows.** `dbm/eval.py` and
   `intervene.py` pointed at an existing file return the old result.
2. **`.log` files are appended to the committed path.** `head_trace` always tees to its
   tag path, even with `--out` (`run_logging.tee_to_log`). After pod runs, either
   `git checkout` those logs or apply change C11 (the split-in-tag fix).
3. **Nothing `*.pt` exists in this checkout.** That covers DBM checkpoints,
   `readout_fit.pt`, head_das `intervention.pt`, the PCA/SAE dictionaries and VADE's
   source-activation cache. Experiments needing one are marked **[pod file]**.
   Pruned flags train/test tuples for 7B exist locally. The pruned 3B and InternVL
   tuples exist only on the pod.
4. **Attention backend.**
   - sdpa: `attr_head_trace`, `attr_capture`, `ceiling_sweep`, `head_trace`.
   - eager: `vrh_ablation`, `head_followup_common.Runner`, `route_knockout`.
   - `attr_steer` is eager only if `--image_trace` is passed.
   - `attribute_switch_sweep` follows `--attn_impl`.

   The two backends give different clean answers on some rows (G7: 90.6% vs 78.1%
   base-kept), so compare arms only within a backend.
5. **Flags' train/test split is already country-disjoint** (59 / 25 countries, 0
   overlap; checked on every attribute's tuples). Held-out selection = select on
   `--split train`, evaluate on `--split test`. No item filter is needed.
6. **Scoring.** Last-column prefill-only patches steer the first answer token only, so
   read `first` / `donor_first`, not full-answer match. Non-flags entities need text
   matching.
7. Cost buckets on an A100 80GB: **XS** <10 min, **S** 10-30 min, **M** 30-90 min,
   **L** 1.5-4 h, **XL** >4 h. These are estimates from loop counts, not measurements.

---

## 2. Code changes to make off-pod first

**Status (2026-10-18): C1-C14 are implemented.**
- 109 new offline tests in `tests/test_tier3_{heads,attr,mlp_positions,route}.py`. The full suite passes: 401 tests.
- Every new flag defaults to the old behaviour, checked against the pre-change code (same arms, names, output paths and values).
- The launcher is `scripts/run_tier3.sh`, with step names `p1`-`p33`, `session1`/`2`/`3` and `smoke`.
  - Every launcher command was checked against its script's real parser.
  - Every step that supports `--dry_run` was also dry-run (P6/P31's second commands need their first command's output).

**Where the launcher differs from section 4, it is the authoritative version:**
- **P1:** masks are written to `results/tier3/dbm_identity/`.
- **P3:** the held-out consensus (in all four train top-16 lists) is computed automatically and saved to `heldout_consensus.txt`.
- **P11:** passes `--heads` with the exact `common10` (C13).
- **P15:** now concrete: `route_knockout.py --entity brands --install_heads <brands top-8> --knockouts late=readout:24-27`.
- **P23:** uses the two fixed column halves of the face (`idx:` left / right 8), not random draws.
- **P29:** uses `--mask_lr 1e-2` (C2a makes it reach dbm's mask).
- **P31:** only calling_code.

**Small additions beyond the spec:**
- **head_das:**
  - `_mlr<x>` goes in the dbm tag when `--mask_lr` is passed, and `_init` when `--init_from` is.
  - Every summary.json now carries `init_from` (null by default).
- **head_severed:** refuses duplicate arm names.
- **route_knockout:**
  - Stacked parts are one `EdgeKnockout.union`, because two separate knockouts on a shared query row would double-subtract.
  - Installs that share a block with readout/text knockouts are refused.
  - Non-flags entities score against the model's own answers.
- **head_swap_vade:** dose arms use `torch.lerp`, which is exact at α=0 and α=1.
- **head_decode_trace, attribute_switch_sweep, attr_capture:** the implementation hash changed, so old output directories do not resume. Always use new ones.


Each change is small and must come with an offline test in `tests/`. Sizes are lines
of code.

| ID | File | Change | Size | Used by |
|---|---|---|---|---|
| C1 | `vrh_ablation.py` | `--n_random_movers N`: block-matched random draws for the `--movers` list, from a separate RNG (`random.Random(seed+1)`) so the Park draws stay identical. Append after the arms loop (~line 264). | ~5 | P2 |
| C2 | `head_das.py` | (a) The mask param group matches `n == "masks"` only, but pyvene's DBM parameter is `mask`, so `--mask_lr` is silently ignored for `--method dbm` (line ~703). Accept both names. (b) Optional `--init_from <intervention.pt>` loaded after `build_interventions` (line ~1114). `--skip_train` currently never loads a checkpoint. | 1 + ~6 | P4b, P29 |
| C3 | `head_severed.py` | `--freeze_mlp_span` takes several spans (`nargs="*"`, each `a-b[,c-d]`); `parse_span` (line 128) learns commas. Keep `str` input working: `run_tier1/2.sh` and two tests pass one string. | ~10 | P10 |
| C4 | `text_entity_contrast.py` | `--mover_subsets`: add `single:*`, `pair:*` (45) and `loo:*` (10) sets of `image_movers` to `sets` (line ~431). Write per-row `head_recs` to a `heads_rows.jsonl` (today only aggregates are written). | ~12 | P9 |
| C5 | `head_swap_vade.py` | (a) Use the entity's own attribute list (`entity_attributes(ground_truth)`) at lines 807 and 260, not the flags `ATTRIBUTES`. Add `--tuples_dir`: `load_jobs` (line 332) always reads VADE's UNPRUNED `data/<entity>/tuples`, and pruned tuples keep `row_index`. (b) `--alphas a1 a2 ...`: arms `heads_<name>_a<a>` that pass `transform={b: have + a*(want-have)}` to `patched_generate` (hook exists, line 542). | ~5 + ~12 | P14, P28 |
| C6 | `attr_head_trace.py` | (a) `--rank_from <trace.json>`: take the head ORDER from a previous run's phase 1, and still install this run's own captured values (insert after line 387; assert the same head table). (b) `--donor_attributes ...` filter in `sample_rows` (line 198) for per-attribute phases 2/3. | ~8 + ~4 | P6, P19 |
| C7 | `attr_steer.py` | (a) `--direction_layer K`: estimate at K, apply at `--layers` (lines 139-149, 302-303, 359-360). (b) `--continuous`: apply `fn` on decode steps too (line 156). (c) `--attn_impl {sdpa,eager}` (lines 277-283). Each flag goes into the tag. | ~10 + ~8 + ~4 | P12, P24 |
| C8 | `attribute_switch_sweep.py` | (a) Scopes `field` (the one token where base and donor prompts differ) and `tok:-K`. (b) Kind `attention_set:16-18.20-23` for gapped block sets. New `--out_dir`, since the implementation hash changes. | ~15 + ~12 | P25 |
| C9 | `common/position_sets.py`, `probe_common.py`, `ndm/ceiling_sweep.py` | (a) `idx:i,j,...` explicit image-index spec (`_GRID_HEADS`, `grid_indices` line 148). (b) `ATTRIBUTE_KEYWORDS` for the five animals attributes (exact keyword lists in P17), else `phrase:attribute` asserts. (c) `--out_suffix` (`--out` is forbidden with `--positions_list`). | ~6 + ~6 + ~2 | P17, P23 |
| C10 | `route_knockout.py` | (a) Knockout key groups and stacking: `name=queries:a-b[@image|@object|@background][+queries:a-b[@...]]` (`@image`, the default, is today's behaviour) (`parse_ko` line 69, `run_batch` line ~217; one `EdgeKnockout` per part). (b) A `common10` install combined with an edge knockout: capture the movers' o_proj inputs in the source forward, install them with a pre-hook (copy `text_entity_contrast.head_install`), run inside `EdgeKnockout`. (c) `--entity` for brands/celebrities: needs `--gold own`-style golds and per-attribute chunking. | ~15 / ~30 / ~50 | P27, P15 |
| C11 | `head_trace.py` | Put the split in the tag (`if args.split != "test": tag += f"_{args.split}"`, after line 435) so train-split traces never append to committed test logs. | 1 | P3 |
| C12 | `attr_capture.py` | `--prompts {controlled,vade}` + `--template_index`, chunked attribute-major (VADE templates differ in length). | ~25 | P32 |
| C13 | `head_decode_trace.py` | `--heads` to add an explicit head set (exact `common10`) next to the trace's top-k (line ~59). | ~3 | P11 |
| C14 | `scripts/run_tier3.sh` | A launcher in the run_tier1/2 style: one log per step, explicit `--out` everywhere, a `GPU=a100` batch profile. | ~150 | all |

Not needing code:
- the DBM identity test (P1);
- the gate=0.5 control (P4);
- `head_cross` seed 1 (P5);
- `blocks:N` / `seq:before` ceilings (P7, P8);
- MLP 27 at a third country (inside P10);
- the 3B windows (P18);
- `--router_k 16` (P20);
- currency with 10 random draws (P21);
- the text early-entity freeze (P22);
- the cached-decoding checks (P26).

---

## 3. Priority order

The order follows three rules:
- first, anything that can **void or overturn a claim already written**;
- then cheap **decisive** mechanistic tests;
- then generality, then expensive or optional runs.

| # | Experiment | Claims | Code | Cost | Why here |
|---|---|---|---|---|---|
| P1 | DBM identity-mask test | Z1 | none | S | Decides whether any DBM number is usable, and whether a learned blend can beat the full swap |
| P2 | Matched random-10 null for zeroing the movers | E3, claim 1 | C1 | XS | The "movers are the bottleneck" ablation has no null |
| P3 | Held-out mover selection | A3 | C11 | M | `common10` was selected on the test countries it is evaluated on |
| P4 | Gate=0.5 control for E3's DBM masks | E3 / claim 6 | none (C2 for P4b) | M | E3's masks are untrained-looking; their scores may be the 0.5 blend |
| P5 | `head_cross` with new pairs (seed 1) | A6, A7, Z4 | none | XS | The "replications" re-used the same pairs |
| P6 | Held-out routers, and routers traced to block 27 | B1 | C6a | S | Same in-sample issue as P3, on the router side |
| P7 | `blocks:L` vs embedding at image positions | D1 | none | XS | Decisive for "the entity code is the vision encoder's output" |
| P8 | `seq:before` at layer 0 | D4 | none | XS | Shows whether the encoder itself spreads the object |
| P9 | Logit-level single / pair / leave-one-out installs | A1 | C4 | S | Separates a true conjunction from a generation threshold |
| P10 | MLP span leave-one-out, MLP 25/27 at third-country values, new seed | C2, C3, C4 | C3 | M | Tests "distributed lookup", the MLP 25 bottleneck and "27 amplifies" in one run |
| P11 | Continuous restore of the heads for calling_code | A2 | C13 (optional) | M | Explains the 28% calling_code residual |
| P12 | Steering with a direction from another layer | B4 | C7a | S | Makes the "the code rotates" claim causal |
| P13 | Readout rank sweep at the MLPs | C5 | none, [pod file] | L | Rank-capped geometry |
| P14 | `common10` installed on animals | A10 | C5a (incl. `--tuples_dir`) | M | Causal version of the overlap claim |
| P15 | Brands: cut late attention to the image during the install | C6 | C10b+c | M | Tests the README E7 contradiction |
| P16 | Stage-1 curves at n=128 | D5, D6 | none | M | The category-attribute timing rests on n=32 |
| P17 | Animals text-position ceilings | D6 | C9b | S | Looks for the two-hop route |
| P18 | 3B two-step windows + continuous install for calling_code | F2, F3 | none | S | The 3B two-step story and the calling_code artefact |
| P19 | Per-attribute routers | B8 | C6b | S | Shared router vs per-attribute block peaks |
| P20 | Composition with `--router_k 16` | B9 | none | S | Is the answer format carried by more routers? |
| P21 | Currency with 10 random draws | E2 | none | XS | Tightens the Park-currency null |
| P22 | Text entity freeze at an early entity layer | D9 | none | XS | Early subject-enrichment test |
| P23 | Ring sweep + a token-count control for faces | D2, D3 | C9a,c | S | Position claims |
| P24 | Steering: seeds, continuous application, attention backend | B3, G7 | C7b,c | S | Robustness of B3 |
| P25 | Field-word scopes; gapped attention sets | B6, B7 | C8 | M | Relay and keystone questions |
| P26 | `--verify_cached` + paired serial/cached runs | G7 | none | S | Engine trust for the attribute-switch results |
| P27 | Object vs background edge knockout | A9 | C10a | S | Causal version of the per-token result |
| P28 | Dose (α) install | A8 | C5b | L | Regional-code mechanism |
| P29 | DAS at a matched training budget (+ DBM at a higher lr) | G4, E3 | C2 | L | Removes the budget confound |
| P30 | Dictionary controls: random features, fixed layer | G1, G2 | none, [pod file] | L | Needs dictionaries rclone'd or refit |
| P31 | Token-edge knockout at n=64 | Z12 | none | XL (L for calling_code only) | Extends n=4 |
| P32 | Attribute capture on VADE templates | B5, C1 | C12 | S | Prompt-format confound, with a caveat |
| P33 | Per-country steering directions | C7 | C7 (+~20) | S | Item-specific decay |
| P34 | Qwen2.5-VL-32B | F1 | small | XL | Relative-depth claim |

**Suggested sessions:**
- **Session 1:** P1-P12 (~4-6 h). Every P1-P8 result can change text already in
  README / CLAIM_CANDIDATES.
- **Session 2:** P13-P27.
- **Session 3:** P28-P34, as needed.

Run P1, P2 and P5 before anything else; they are each under 30 minutes.

---

## 4. The experiments

Every command block is ready to paste on the pod once its code change has landed.
`$OUT` = a new directory under `results/tier3/`.

### P1. DBM identity-mask test (Z1). Cost: S, 4 evals × 190 batches.

**Correction to Z1.** The 518cf31 BuildBatchCache bug **cannot** affect DBM. DBM builds
one cache per call with a single constant position set (`dbm/eval.py:55,87`,
`train.py:196,356`). The "full_image" in its log is the separate source-activation
cache, which stores all 144 image tokens and is indexed by the freshly resolved
`flag_ring1` indices (`source_cache.py:132-145`). A mismatch there would have crashed
on shape, not changed numbers.

The anomaly is real on identical rows: on the ceiling sweep's own 32 rows (reconstructed
from its seed), DBM L16 gets 22-24/32 where the full swap gets 10/32.

```bash
python - <<'EOF'
import sys, torch; sys.path.insert(0, '.')
from methods.dbm.intervention import SigmoidMaskIntervention
for name, v in [("ones", 1.0), ("half", 0.0)]:          # sigmoid(0/T) = 0.5 at any T
    iv = SigmoidMaskIntervention(embed_dim=3584); iv.mask.data.fill_(v); iv.temperature.data = torch.tensor(1e-7)
    torch.save(iv.state_dict(), f"/workspace/dbm_{name}.pt")
EOF
for c in ones half; do
  python methods/dbm/eval.py --entity flags --attribute language --layer 16 --positions flag_ring1 --split test \
    --cause_only --ckpt_path /workspace/dbm_$c.pt --out_dir results/tier3/dbm_identity/L16_$c
  python methods/dbm/eval.py --entity flags --attribute language --layer 16 --positions flag_ring1 --split test \
    --cause_only --no_source_cache --ckpt_path /workspace/dbm_$c.pt --out_dir results/tier3/dbm_identity/L16_${c}_nocache
done
```

- **Read** cause on the 3,036 cause rows.
- **If `ones` ≈ 31%** (the ceiling): the eval path is sound, and a trained per-dimension blend genuinely beats the full swap. Z1 becomes a *finding* (a partial blend avoids the source-vs-base conflict E11 describes), and DBM numbers are citable with that caveat. `half` then shows how much of DBM's 60% any blend gets.
- **If `ones` ≈ 60%:** the eval path is at fault. If only the cached arm is high, the source cache is stale: it is keyed on item only and never invalidated. Delete it and re-run.
- **If any DBM checkpoint survives on the pod** (`.../flags/dbm/language/<tag>/layer16_intervention.pt`), re-evaluate it into a new `--out_dir` as a third arm.

### P2. Matched random-10 null for zeroing the movers (E3, claim 1). Cost: XS. Needs C1.

```bash
python methods/vrh_ablation.py --lists park_vrh --n_random 0 --n_random_movers 5 --index_items 0 \
  --batch_size 128 --out_dir results/tier3/vrh_mover_random
```

- 8 arms × 672 generations. `clean`, `zero:park_vrh` and `zero:common10` repeat the committed run as a check.
- **Expect** each block-matched random-10 set (blocks 21-23, 2/4/4 heads) to cost ≤5 points, against −51 for `common10`. That makes "the movers are the bottleneck" a controlled result.
- If some random sets cost ≥20 points, blocks 21-23 are generally fragile, and the claim must become "zeroing heads in the handoff window".
- Also report `mover_name_rank` for the random arms: the movers stay intact there, so their name rank should stay ~1.6.

### P3. Held-out mover selection (A3). Cost: M. Needs C11.

Rule used for `common10` (R8): **heads in all four attributes' top-16**. Repeat that rule
on the 59 train countries, then evaluate on the 25 test countries.

```bash
for a in language capital currency calling_code; do
  python methods/head_trace.py --entity flags --attribute $a --patch_layer 21 --positions full_image \
    --blocks 21 22 23 --n_rows 64 --split train --out results/tier3/heldout_heads/${a}_train_b21-23.json
done
# CPU: heldout10 = intersection of the four phase1_ranked[:16] lists (print it and its overlap with common10)
python methods/head_swap_vade.py --head_sets "heldout=<B.H list from the CPU step>" \
  "heldout_per_attr=results/tier3/heldout_heads/{attribute}_train_b21-23.json#10" \
  --arms heads random_heads --out_dir results/tier3/head_swap_vade_heldout
```

- `--positions full_image --n_rows 64` match the committed traces (the defaults are `flag_ring1` / 32).
- Compare with the committed `heads_common10` and `full_image` on the same test rows (same job grid, same `row_index`).
- **Expect** ≥8/10 overlap with `common10`, and held-out transfer ≥85% of `full_image`.
- A drop to ≤70% means `common10` was fit to its evaluation countries. Then every `common10` number needs this caveat or a re-run with `heldout`.

### P4. Gate=0.5 control for E3's DBM masks (claim 6). Cost: M, 4 × 4,000-row evals.

An untrained pyvene `SigmoidMaskIntervention` has mask=0, i.e. gate 0.5 on every neuron
at any temperature. `--skip_train` then evaluates exactly the 0.5 blend on E3's own eval
rows (seed 0, 4,000 rows).

```bash
for a in language capital currency calling_code; do
  python methods/head_das.py --attribute $a --method dbm --site mlp_hidden --blocks 23-26 \
    --l1_coef 1e-4 --temperature_start 1e-2 --temperature_end 1e-2 --eval_rows 4000 --batch_size 32 --seed 0 \
    --skip_train --out_dir results/tier3/head_das_gate0.5/${a}_mlp23-26
done
```

- Score with `../VADE/eval/score.py` exactly as the E3 masks were scored.
- **Expect** gate=0.5 to reproduce most of E3's finals (88.2 / 60.3 / 59.4 / 59.3). That would mean the 63 updates did nothing, and E3's DBM row should be relabelled "half-swap at MLPs 23-26".
- If the trained masks clearly beat 0.5, they learned something, and P4b is worth running.
- **P4b (needs C2a):** a real mask-training run, `--lr 1e-2 --epochs 2` (equivalently `--mask_lr 1e-2` after C2a). Use the command in P29, then compare `mask_stats` `n_selected` against 0.

### P5. `head_cross` with new pairs (A6, A7, Z4). Cost: XS, 384 generate calls.

Seed 1 gives 64 pairs with **0 overlap** with seed 0's first 64 (or first 256).

```bash
python methods/head_cross.py --seed 1 --n_pairs 64 --out results/tier3/head_cross_seed1/cross_n64_seed1.jsonl
```

- Re-run `methods/cpu_followups.py`'s `cross_bootstrap` logic on the new file. Point its path at the new file, or add a `--path` option.
- **Expect** every cell within the seed-0 country-cluster interval (±4-6pp): 81 / 65 / 83 / 93%.

### P6. Held-out routers, and routers traced to block 27 (B1). Cost: S. Needs C6a.

Country split A/B (`random.Random(0).shuffle(sorted(ids))`; each half has 42):
- **A:** AL AM AR BD BE BR BY CM CO DE ES ET FI FJ GT HN IE IS KE KH KR LA LB LR LY MK NA NI OM PA PT RO RS SA SD SG TH TN TZ UA VN YE
- **B:** AF AO AU AZ BH BZ CL CU CV CY DK DZ EG GH GR GY HT HU IN IQ IT JO JP KP KW LU MA MU MX NP NZ PE PK PY QA RW SE SN UG UY UZ ZM

```bash
python methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
  --countries <A> --out results/tier3/attr_heads/sel_A.json
python methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
  --countries <B> --rank_from results/tier3/attr_heads/sel_A.json --out results/tier3/attr_heads/eval_B_rankA.json
# closes the coverage gap (phase 3 ALL must then equal the question patch's `first`):
python methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 23 24 25 26 27 \
  --n_countries 84 --knockout_ks 1 2 4 8 16 32 64 128 364 --out results/tier3/attr_heads/patch15_blocks15-27_n84.json
```

- **Expect** held-out k=8 sufficiency ~85% and necessity ≤5% transfer.
- With blocks 15-27, the all-heads arm becomes an exact identity. A gap would point to a bug, not to coverage.

### P7. `blocks:L` vs the vision embedding at image positions (D1). Cost: XS.

`blocks:N` at layer L = attn_output + mlp_output of blocks L-N … L-1. So `blocks:L` at
layer L swaps every language-model write in blocks 0 … L-1 but keeps the base's layer-0
embedding (the vision tower + merger output).

```bash
for a in capital currency calling_code language; do
  python methods/ndm/ceiling_sweep.py --entity flags --attribute $a --positions full_image --layers 0 10 20 \
    --sites residual blocks:1 blocks:2 blocks:3 blocks:4 blocks:5 blocks:10 blocks:20 \
    --out results/tier3/ceiling_blockspan/${a}_full_image_L0-10-20.json
done
```

- **Expect** `blocks:10`@10 and `blocks:20`@20 ≈ 0% cause, against `residual`@0 = 100%. Together these would show the image-token entity code is the encoder output, carried unchanged by the residual stream.
- Caveat: text positions downstream read the hybrid image residual, so the spans are not perfectly nested. The comparison with `residual@0` is what matters.

### P8. `seq:before` at layer 0 (D4). Cost: XS.

Layer 0 is the decoder input, i.e. the vision encoder output at image positions.

```bash
python methods/ndm/ceiling_sweep.py --entity flags --attribute capital --positions seq:before --layers 0 1 \
  --sites residual --out results/tier3/seqbefore/flags_capital_L0-1.json
python methods/ndm/ceiling_sweep.py --entity brands --attribute hq_country --positions seq:before --layers 0 1 \
  --sites residual --text_match --max_new_tokens 8 --out results/tier3/seqbefore/brands_hq_L0-1.json
python methods/ndm/ceiling_sweep.py --entity celebrities --attribute nationality --positions seq:before --layers 0 1 \
  --sites residual --text_match --max_new_tokens 8 --out results/tier3/seqbefore/celebrities_nat_L0-1.json
```

- **Expect** 15-24% already at layer 0, the E11 level. That pins the leak on the vision encoder's full attention.
- If layer 0 reads ~0 and the value only grows later, the language model is moving object information backwards. That cannot happen under causal masking, so it would indicate a bug.

### P9. Logit-level mover conjunction (A1). Cost: S, ~3.5k forwards. Needs C4.

```bash
python methods/text_entity_contrast.py --phases heads --entity_layer 21 21 --mover_subsets \
  --out_dir results/tier3/mover_subsets
```

- **Read** `ld` (the logit difference at the divergence point) per set: 10 singles, 45 pairs, 10 leave-one-outs, all 10. Use the image condition. The text condition also runs, as a free comparison.
- **Conjunction (superadditive):** the joint ld is much larger than the sum of single-head lds, and each leave-one-out costs a lot.
- **Additive below a threshold:** ld grows roughly linearly in set size.
- Report both. Text-level "0% at k≤2" stands either way.

### P10. MLP 23-26: leave-one-out, MLP 25/27 at third-country values, new seed (C2, C3, C4). Cost: M. Needs C3.

```bash
python methods/head_severed.py --n_pairs 128 --seed 1 \
  --freeze_mlp_span 23-26 24-26 23,25-26 23-24,26 23-25 --freeze_mlp_singles 25 27 \
  --freeze_attn_span '' --freeze_values base mean other --out_dir results/tier3/severed_loo_n128_s1
```

Arms: clean, heads, 5 spans × 3 values, 2 singles × 3 values. That is 23 arms × 128 pairs × 4 attributes.

| question | expected | if not |
|---|---|---|
| **C2: distributed?** Any 3-of-4 span frozen at mean | most of the 4-block loss | removing any one block breaks it: serial |
| **C3: MLP 25 for calling_code/currency**, at mean | ~25% / ~30% (n=64 values), reproduced | — |
| **C4: does MLP 27 decide?** `mlp[27]@other` | <20% third-country answers (it amplifies) | ≥50%: it decides too, so revise "23-26" |

### P11. Continuous restore of the heads for calling_code (A2). Cost: M, ~15k forwards at batch 1. C13 optional.

```bash
python methods/head_decode_trace.py \
  --trace logs/Qwen2.5-VL-7B-Instruct/flags/ndm/calling_code/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json \
  --head_ks 10 --out_dir results/tier3/decode_calling_code_k10
```

- Without C13 it restores that trace's own top-10: `common10` minus 23.17, plus 23.7.
- **Read** `necessity_top10_continuous`.
- **Expect** calling_code transfer falls from 28% (prefill-only) to ~0%. The 28% is later answer tokens re-reading the patched image through the KV cache.

### P12. Steering with a direction from another layer (B4). Cost: S. Needs C7a.

```bash
python methods/attr_steer.py --entity flags --layers 22 --direction_layer 17 --mode add --alpha 0.5 1 2 4 \
  --out results/tier3/attr_steer/dir17_at22.json
python methods/attr_steer.py --entity flags --layers 22 --direction_layer 22 --mode add --alpha 0.5 1 2 4 \
  --out results/tier3/attr_steer/dir22_at22.json
```

- "Block 16" = residual **layer 17** (layer L is block L-1's output).
- Use `add`, not `project`: project would install layer-17 coordinates on the layer-22 axis.
- **Expect** the layer-17 direction to behave like the random control (≤ a few %) at every α, while the matched direction gives ~95%. That would make the rotation causal, not just descriptive.

### P13. Readout rank sweep at MLPs 23-26 (C5). Cost: L, 4 × 28.7k generations. [pod file: `readout_fit.pt`]

```bash
for R in 32 64 128 256; do
  python methods/readout_jacobian.py --phase eval --site mlp_hidden --blocks 23-26 --max_rank $R --energy 1.0 \
    --fit_path results/readout_jacobian/flags__mlp_hidden_b23-26/readout_fit.pt \
    --out_dir results/tier3/readout_rank/r${R}_e1
done
```

- `--energy 1.0` is required. The default 0.9 also caps the rank, so calling_code stopped at 89 in the committed run.
- **Expect** calling_code to saturate by rank ~8. The other three keep gaining and their overlap grows, so their shared fraction is a lower bound.

### P14. `common10` installed on animals (A10). Cost: M. Needs C5a.

```bash
python methods/head_swap_vade.py --entity animals \
  --tuples_dir ../VADE/models/Qwen2.5-VL-7B-Instruct/animals/tuples \
  --head_sets "common10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17" \
  --arms clean heads random_heads full_image --patch_layer 10 --out_dir results/tier3/head_swap_animals
```

- `--patch_layer 10` replaces the default 21, which is past the animals handoff.
- **Expect** native_continent transfer to be high (an entity fact), and body_covering, reproduction and taxonomic_class to be low (category attributes, read earlier and elsewhere).
- If all are low, `common10` is flags/brands/celebrities-specific, and A10 should be worded that way.

### P15. Brands: cut late attention to the image during the install (C6). Cost: M. Needs C10b+c.

- On brands/hq_country: install the entity's top-8 (`--gold own` scoring), with and without `readout:24-27` image-edge knockout.
- **Expect**, if late attention re-reads the base image: the knockout reproduces the 64.6 → 73-84% gain seen when attention 24-27 is frozen.
- If the knockout changes nothing, the late heads read text positions instead.
- This is the most code in Session 2. It can be dropped if the README E7 wording is simply corrected.

### P16. Stage-1 curves at n=128 (D5). Cost: M, ~22k generations.

For each of the 12 non-flags attributes (animals ×5, celebrities ×4, brands ×3), run
both specs with `--out` into a new root, then let the pipeline read them:

```bash
python methods/ndm/ceiling_sweep.py --entity animals --attribute body_covering --sites residual --positions full_image \
  --layers $(seq 0 28) --n_rows 128 --text_match --max_new_tokens 8 \
  --out results/image_head_pipeline_n128/animals/body_covering/ceiling_full_image_text.json
#   (same with --positions last_token -> ceiling_last_token_text.json)
python methods/image_head_pipeline.py --entity animals celebrities brands --stages ceiling --n_rows_ceiling 128 \
  --out_root results/image_head_pipeline_n128
```

- **Do not** run the pipeline at the default `--out_root`. It silently re-uses the committed n=32 ceiling JSONs (it never checks their row count).
- **Expect** the 7-12-layer gap between category and entity-fact attributes to persist, with tighter curves.

### P17. Animals text-position ceilings (D6). Cost: S. Needs C9b.

Keywords for `ATTRIBUTE_KEYWORDS`. Matching is a substring match over the whole
question, which includes the answer-options list:
- native_continent: `["continent"]`
- primary_habitat: `["habitat", "environment", "live"]`
- taxonomic_class: `["taxonomic class", "classes", "class"]`
- reproduction: `["reproductive mode", "reproductive strategy", "reproduce", "offspring", "lay eggs"]` (not "live birth", which is one of the options)
- body_covering: `["body covering", "outer covering", "covering", "covers", "covered"]`

```bash
python methods/ndm/ceiling_sweep.py --entity animals --attribute body_covering \
  --positions_list phrase:attribute vision_end tok:-2 pre_image --text_match --max_new_tokens 8 \
  --sites residual --layers $(seq 0 28)
```

- `pre_image` is a true null.
- `tok:-5` is dropped: on iso rows it lands on different words because prefills tokenize differently.
- **Expect**, if there is a two-hop route: one text position (most likely `phrase:attribute` or `vision_end`) carries high cause over L14-20. If all read ~0, the attribute is spread too thin for a single-position swap.

### P18. 3B two-step windows + continuous install for calling_code (F2, F3). Cost: S. [pod file: pruned 3B tuples]

```bash
for a in language capital; do
  python methods/head_trace.py --entity flags --attribute $a --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --patch_layer 27 --positions full_image --blocks 27 28 --n_rows 64 --batch_size 32 \
    --out results/tier3/qwen3b/${a}_patch27_blocks27-28.json
  python methods/head_trace.py --entity flags --attribute $a --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --patch_layer 31 --positions full_image --blocks 31 32 33 --n_rows 64 --batch_size 32 \
    --out results/tier3/qwen3b/${a}_patch31_blocks31-33.json
done
python methods/head_swap_vade.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --patch_layer 27 \
  --head_sets "cc10=results/image_head_pipeline_Qwen2.5-VL-3B-Instruct/flags/calling_code/head_trace_patch27_blocks27.json#10" \
  --arms clean heads random_heads full_image --out_dir results/tier3/head_swap_qwen3b
```

- `head_trace` reads 36 layers × 16 heads from the model. Do not pass `--n_layers`.
- InternVL is **not** supported by `head_swap_vade` / `head_cross` / `vrh_ablation` (they read `image_grid_thw`), so its calling_code check needs an adapter port. Skip it.
- **Expect** each 3B step's heads to be sufficient for its own share of the transfer. The continuous calling_code install should jump to ~90% as on 7B, which would turn F3 into the G5 artefact.

### P19. Per-attribute routers (B8). Cost: S, 4 runs. Needs C6b.

```bash
for a in capital currency language calling_code; do
  python methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
    --n_countries 84 --donor_attributes $a --out results/tier3/attr_heads/donor_${a}.json
done
```

- **Expect** a top-8 overlap of ≥6/8 across attributes, with similar sufficiency. The per-block peaks then come from direction geometry, not from separate heads.

### P20. Composition with `--router_k 16` (B9). Cost: S.

```bash
python methods/circuit_compose.py --router_k 16 --n_items 64 --batch_size 64 --out_dir results/tier3/compose_rk16
```

- The window widens to blocks 17-23, so the `target_window` / `target_complement` arms are not comparable to the committed run. Compare `compose` only.
- **Expect**: if format comes from the base question, the wrong-format "other" in format-mismatched pairs stays ~20%. If it drops sharply, the extra routers carry format.
- Re-run `cpu_followups.py`'s `compose_failures` on the new rows.

### P21. Currency with 10 random draws (E2). Cost: XS.

```bash
python methods/vrh_ablation.py --attributes currency --n_random 10 --index_items 0 --batch_size 128 \
  --out_dir results/tier3/vrh_currency_r10
```

- **Expect** `zero:park_vrh` (−21) to sit outside the range of all 10 random sets.

### P22. Text entity freeze at an early entity layer (D9). Cost: XS.

```bash
python methods/text_entity_contrast.py --phases freeze --entity_layer 4 4 --out_dir results/tier3/text_Le4_freeze
```

- **Expect** freezing `entity:4-9` at the mean to lower text transfer more than image transfer, if text has a subject-enrichment stage that the image route skips. Equal effects mean no such asymmetry.

### P23. Ring sweep + a token-count control for faces (D2, D3). Cost: S. Needs C9a,c.

- **Rings.** `--positions_list ring:0 ring:1 ring:2 ring:3 ring:4 ring:0+ring:2+ring:3+ring:4` at `--layers 2 6 10 14 18 22`, with an `--out_suffix`. `ring_2` at layers 0-28 already exists, so the new layer set plus the suffix avoids a collision. `~ring:K` is not valid syntax; use the union.
- **Faces.** Patch 8 of the 16 face tokens with `--positions idx:<8 of 52-55,64-67,76-79,88-91>`, on two disjoint draws, at `--layers 10`.
- **Expect** ring:1 to rise over L10-18 while ring:0 falls.
- **Expect** the 8-token face patch to stay clearly above flags' 34%, if faces really are more local.

### P24. Steering: seeds, continuous application, attention backend (B3, G7). Cost: S. Needs C7b,c.

- `--seed 1`, `--seed 2`: the test sets overlap across seeds. For disjoint folds, add `--train_items/--test_items`.
- `--continuous` at `--layers 21 22 23`.
- `--attn_impl eager` on the seed-0 rows.
- **Expect** ±5pp across seeds, full-answer transfer rising toward first-token transfer under `--continuous`, and the eager-vs-sdpa base-kept gap to reproduce.

### P25. Field-word scopes and gapped attention sets (B6, B7). Cost: M. Needs C8.

- **Field words.** Scopes `field`, `tok:-2` ("Answer") and `last_token` (":") with residual, L1-28, `--execution cached --batch_size 64`.
- **Keystone.** `attention_set` arms leaving out block 19, 20 or 21 from 16-23, at `--layers 24 --scopes last_token`.
- "Left out" means that block recomputes from the edited residual, not a frozen restore.
- **Expect** the field word to go dead earlier (~L17-18) than the full question scope, if there is a relay through "Answer"/":". Leaving out 19, 20 or 21 should cost similar amounts if block 19 is no keystone.

### P26. Cached-decoding checks (G7). Cost: S.

```bash
python methods/attribute_switch_sweep.py --n_countries 1 --attributes capital currency --scopes last_token --sites attention \
  --layers 22 --block_spans --attention_spans 3 --execution cached --batch_size 2 --verify_cached \
  --out_dir results/tier3/switch_verify_sdpa
#   repeat with --attn_impl eager --no_share_prefix -> results/tier3/switch_verify_eager_noshare
#   paired check: identical config with --execution serial vs --execution cached --batch_size 64 (and --batch_size 1),
#   --countries AO DE --layers 18 20 22 24 --sites residual attention --scopes earlier_text last_token --block_spans
#   --attention_spans 3 --attn_impl sdpa; join rows.jsonl on (row_index, arm)
```

- `--verify_cached` raises on the first mismatch, so keep it tiny.
- **Expect** agreement outside the four known cells (AU language, QA / HU calling_code, MU currency).

### P27. Object vs background edge knockout (A9). Cost: S. Needs C10a.

```bash
python methods/route_knockout.py --knockouts \
  "obj_movers=readout:21-23@object" "bg_movers=readout:21-23@background" \
  "obj_movers_noq=readout:21-23@object+question:0-27@image" "bg_movers_noq=readout:21-23@background+question:0-27@image" \
  "movers_noq=readout:21-23+question:0-27" --q_layers 16 --out_dir results/tier3/route_ko_groups
```

- `--knockouts` replaces the default list, and `--q_layers` needs at least one value.
- The `_noq` arms also cut question → image edges. Without them the question-token fallback (E5) masks the effect: cutting all answer → image edges in 21-23 alone costs only 5 points.
- **Expect**: cutting background edges costs more in total, cutting object edges more per token. If cutting object edges alone costs ~0, identity is read from the ring around the object (consistent with E11).

### P28. Dose (α) install (A8). Cost: L. Needs C5b.

```bash
python methods/head_swap_vade.py --head_sets "common10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17" \
  --alphas 0 0.25 0.5 0.75 1 --arms heads --limit_pairs 600 --out_dir results/tier3/head_swap_dose
```

- α=0 must equal clean, and α=1 must equal the committed `heads_common10`.
- Then run `cpu_followups.py`'s `error_geography` on the α files.
- **Expect**, if the code is regional, intermediate α to produce source-region third countries well above the ~6% null. Uniform garbage would argue against a regional code.

### P29. DAS at a matched training budget, and DBM at a higher learning rate (G4, E3). Cost: L. Needs C2a for the DBM half.

```bash
python methods/head_das.py --attribute language --method das_fixed --subspace_dim 159,332,282 \
  --batch_size 16 --grad_accum_steps 1 --train_rows 6000 --eval_rows 2000 --seed 0 \
  --out_dir results/tier3/head_das/language_das_fixed_k159-332-282_bs16ga1_tr6000
#   also K = 13,14,25 / 49,54,75 / 103,161,172 (the learned das_rotated widths) and 8 / 64 / 256
python methods/head_das.py --attribute language --method dbm --site mlp_hidden --blocks 23-26 --l1_coef 1e-4 \
  --temperature_start 1e-2 --temperature_end 1e-2 --lr 1e-2 --epochs 2 --train_rows 4000 --eval_rows 4000 \
  --batch_size 32 --grad_accum_steps 2 --out_dir results/tier3/head_das/language_dbm_lr1e-2_ep2_mlp23-26
```

- **`--batch_size 16` is required.** The default is 4, which gives a different budget.
- The das_fixed commands use exactly das_rotated's budget (375 optimizer steps) and eval rows.
- **Expect** das_fixed to rise to ~57-60%. The "2.8× from sparsification" then disappears and the plateau claim strengthens.
- For the DBM run, read `n_selected` > 0 before reading any mask overlap.
- `language_das_fixed_k128_full` has no checkpoint (it was killed at step 109/129), so it can only be retrained (~2 h 50 min).

### P30. Dictionary controls: random features, fixed layer (G1a, G2). Cost: L, each run 14,052 rows. [pod file: dictionaries]

- **Get the dictionaries:** rclone `methods/dictionaries_kaggle_pool` per README, or refit PCA (`sae.py` + `fit_dictionaries.py --method pca`, which is cheap).
- **Make winners files** (CPU): random `feature_indices` of the winner's size, variance-binned for PCA, plus an empty-set control that isolates reconstruction error. Fixed-layer L4 entries can be copied verbatim from `sweep.jsonl`.
- **Run** `intervene.py --winners <file> --token_set ... --dict_method ... --dictionaries_dir ... --out methods/interventions/controls/<name>_predictions.jsonl`, always to a **new** `--out`.
- **Expect** the random subset to give ~0 cause. The empty set shows the reconstruction-error baseline.
- At L4, PCA vs SAE gives the matched-depth dictionary comparison that RESULTS.md lacks.

### P31. Token-edge knockout at n=64 (Z12). Cost: XL for all four attributes (~100k forwards at batch 1); L for calling_code only.

```bash
OUT_ROOT=results/tier3/head_followups_n64 ATTRIBUTES="calling_code" bash scripts/run_head_followups.sh identify
OUT_ROOT=results/tier3/head_followups_n64 ATTRIBUTES="calling_code" bash scripts/run_head_followups.sh knockout
```

- **Expect** the n=4 log-prob drops (0.05 → 0.48 nats against ≤0.03 for random edges) to hold.

### P32. Attribute capture on VADE templates (B5, C1). Cost: S. Needs C12.

- **Caveat (built into the design):** under VADE templates the last token is not question-neutral. Prefills end in " is" / " +", right after the attribute word. Part of the last-token "attribute content" is then local context.
- Read the item/attribute/interaction split relative to that.
- **Expect** the block-20 → 22 switch and the blocks 24-26 interaction rise to survive.

### P33. Per-country steering directions (C7). Cost: S. Needs C7 + `--direction_scope item` (~20 lines).

- Use `project` mode. Per-item `add` at α=1 reproduces the full residual swap, so it tests nothing.
- **Expect** per-item directions to keep ~100% first-token transfer at L26-28, where the global direction decays (79% → 44%).

### P34. Qwen2.5-VL-32B (F1). Cost: XL.

- It fits on one 80 GB card in bf16 (~67 GB) at batch ≤16.
- **First:** run VADE's accuracy sweep and pruning for it.
- **Then:** run `image_head_pipeline.py --n_layers 64 --n_heads 40`.
- `vrh_ablation.py:257` and `circuit_compose.py:283` hard-code 28 heads.
- **Expect**, if relative depth holds: the handoff at ~L50-52.

---

## 5. Already settled, no GPU needed

- **Z2 (brands/founded_year).** `results/image_head_pipeline/brands/founded_year/windows_patch9_text.json` already scores at the divergence point (n_div = 64):
  - the image patch moves the year (div_src 100%);
  - every prefill-only window installs 0%;
  - base and source never differ at the first answer token (n_first = 0).

  So the year is carried, but a last-column prefill patch cannot reach the digit where the two years diverge. A continuous install would be the next step: `head_severed --entity brands --attributes founded_year --gold own`.
- **7B accuracy sweep and pruning (F4 input):** already complete in `../VADE/models/Qwen2.5-VL-7B-Instruct/stats/`.
- **E3's DBM predictions** were never written to `predictions_summary.json`. They are scored in `results/cpu_followups/entity_swap.json`; run `score.py` on them for the official numbers.
