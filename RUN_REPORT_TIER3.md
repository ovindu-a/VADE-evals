# Tier-3 run report: Session 1 (P1-P12) and trimmed Session 2

Pod run on 2026-10-18, 05:24-14:47 UTC. Session 1 and the trimmed Session 2 both completed.
Every step reached its end. One self-check failed, and that failure is a designed diagnostic
(P26's `--verify_cached`, see section 3). No step ran out of memory or was skipped, except the
seven Session 2 steps the 2026-10-18 runbook deferred (P13, P15, P16, P17, P18, P21, P24).

## 1. Environment

| | |
|---|---|
| GPU | 1x NVIDIA A100-SXM4-80GB (400 W limit), driver 580.105.08, CUDA 13.0; 96 CPU cores, 1.1 TB RAM |
| Python stack | torch 2.9.1+cu130 (the pod's own build), transformers 5.16.1, pyvene 0.1.8 (`--no-deps`, plus matplotlib to import) |
| VADE | `b8dfaf9` |
| VADE-evals | Session 1 code at `21ad15b`. Session 2 lane 1 ran from an untracked launcher copy of `2734fbd`; the other Session 2 lanes ran after the pull. `methods/` and `tests/` are identical across all of these. Results are in `688bab8`, `7b1e4a4` and `d61ea67`. |
| Tests | `pytest tests/`: 401 passed, before Session 1 and again after the pull |

**How it ran:**
- Session 1 ran as the runbook's three parallel lanes at default batch sizes (`p1 p4 p10` / `p2 p5 p3 p11` /
  `p6 p7 p8 p9 p12`). A sequential `GPU=a100` attempt was stopped about 20 minutes in, partway through P1;
  P1 resumed from its partial predictions file.
- Session 2 ran as the trimmed plan's three lanes:
  - lane 1, `p14 p20 p27 p23`, at `GPU=a100`;
  - lane 2, `p19 p25`;
  - lane 3, `p26 p22`.

  Lane 1 overlapped the end of Session 1, as the runbook's "moving from a running Session 1" section
  allows. MPS was not used.

## 2. Per-step status

Times are each step's own logged `done in N min`, measured while sharing the GPU with the other lanes.

| step | status | wall time | self-check |
|---|---|---|---|
| smoke | ok | ~10 min | All 14 steps finished. `smoke_dbm_eval` failed by design, because it needs P1's masks. |
| P1 | ok | 4 + 7 + 1 + 14 min | Cached and uncached arms agree: ones 37.5 / 37.4%, half 4.6 / 4.5% |
| P2 | ok | 20 min | `clean` / `zero:park_vrh` / `zero:common10` match the committed E9 within one row of 672 (default batch) |
| P3 | ok | 4x 4-5 min + install 217 min | `heldout_consensus.txt` = common10, all 10 heads shared |
| P4 | ok | 4x 20-22 min | No check listed. E3 re-scored for comparison. |
| P5 | ok | 21 min + bootstrap | none listed |
| P6 | ok | 42 + 43 + 114 min | Phase 3's ALL-heads first = question patch first = 95.3%. Read-back MATCHES. |
| P7 | ok | 4x 8-9 min | none listed |
| P8 | ok | 3x 1 min | none listed (unhooked `matches_source` = 0%) |
| P9 | ok | 73 min | `entity_patch` 97.0 / 96.1%; `image_movers` 90.9 / 87.5% (E6: 87-91%) |
| P10 | ok | 42 min | `readback_worst` 0.0, `identity` bit-exact, liveness 2.625 |
| P11 | ok | 54 min | `all_downstream_continuous` = `image_patch` (first 1.000, full 0.906); asserted |
| P12 | ok | 5 + 4 min | none listed |
| P14 | ok | 126 min (a100) | `clean` 76-85% on animals; `full_image` 27-96% cause vs `random_heads` 0-4% |
| P19 | ok | 4x 17-19 min | Top-8 shares with router8: language 7, capital 8, currency 7, calling_code 8 |
| P20 | ok | 180 min (a100) | Read-back exact (7 sets), self-install 0.00e+00; `clean` and `movers` reproduce `circuit_compose/flags` exactly |
| P22 | ok | 17 min | Freeze specs printed; no L_e warning |
| P23 | ok | 5 + 0 + 0 min | `ring:0` L10 / L18 = 31.2 / 21.9% (E11: 34 / 16%, n=96; here n=32); both face halves ran |
| P25 | ok | 42 + 7 min | `field` raised no assertion; every gapped set <= the full blocks 16-23 arm |
| P26 | **verify FAILED** (designed); paired runs ok | 0 + 0 + 64 + 7 + 7 min | `--verify_cached` raised in both verify runs; see section 3 |
| P27 | ok | 6 min (a100) | Empty and stacked-empty knockouts give max \|dlogit\| = 0.00e+00; `img_patch` 99.6% (E5: ~98%) |

## 3. Headline numbers

**P1 (DBM identity masks, flags/language, L16, flag_ring1, cause rows, n=3036):**
- The all-ones mask, equal to a full swap, reads **37.5%** cached and 37.4% uncached (matches_base 57.5%).
- The 0.5 mask reads 4.6% and 4.5%.
- Against the ceiling's 31.2%, DBM's eval path gives a full swap slightly above the ceiling and far
  below DBM's 60-62%. So DBM's reported number exceeds a full swap through its own eval path.

**P2:** zeroing common10 drops accuracy to 38.1%. Five block-matched random-10 mover sets read
87.4-89.6% (clean 88.7%), with mover rank about 1.6 (clean 1.61).

**P3 (movers selected on the 59 train countries, installed on the 25 test countries):**

| arm | cause: lang / cap / cur / cc | iso |
|---|---|---|
| held-out consensus (= common10) | 93.8 / 98.0 / 72.2 / 95.4% | 0.9-2.5% |
| per-attribute held-out top-10 | 93.9 / 97.8 / 74.4 / 95.6% | 0.9-2.3% |
| block-matched random | 2.2 / 0 / 0 / 0% | 92-98% |

**P4 (untrained 0.5 gate vs E3's trained DBM masks at MLP 23-26, final score):**

| | language | capital | currency | calling_code |
|---|---|---|---|---|
| gate 0.5 | 16.4 | 37.0 | 36.9 | 70.0 |
| E3 trained | 59.3 | 60.3 | 59.4 | 88.2 |

FOLLOWUP_PLAN expected gate 0.5 to reproduce E3; it does not. E3's masks report `n_selected = 0`,
but their raw mask values span about ±0.04 at T = 0.01, so the gates range from about 0.02 to 0.98.
They learned a non-uniform blend that sits below the selection threshold. P4b is therefore worth
running.

**P5 (head_cross, seed-1 pairs, 64 pairs; 95% CIs about ±4-5 pp, pair and cluster):**
- Same question: 81.2%. Different question: 65.2% (shift -16.0 pp).
- Question-only cell: 78.8% stayed, 0.0% moved.
- Self: 91.8%.
- The script's caveat applies: under `--align matched`, the shift is not yet attributable to content.

**P6 (routers, blocks 15-27, 84 countries):**
- Top-32 heads alone reach 99.1% of the ceiling, and restoring the top-32 removes 99.4% of the effect.
  The top-8 does almost nothing either way.
- Shuffled donors at ALL 364 heads: the answer follows the donor at 95.2% (own question 0.5%).
- Phase 3's full-answer column reads 76.3%, against the question patch's 92.7%. The check is defined on
  the first-token column, which matches exactly.

**P7 (flags, full_image; cause ceiling):**

| | residual@0 (vision embedding) | blocks:5@10 | blocks:10@10 | blocks:10@20 | blocks:20@20 |
|---|---|---|---|---|---|
| language | 100.0 | 15.6 | 96.9 | 6.2 | 100.0 |
| capital | 100.0 | 21.9 | 93.8 | 0.0 | 90.6 |
| currency | 100.0 | 12.5 | 78.1 | 0.0 | 96.9 |
| calling_code | 100.0 | 15.6 | 87.5 | 3.1 | 96.9 |

**P8:** `seq:before` at layers 0 and 1 reads **15.6%** cause for flags/capital, brands/hq_country and
celebrities/nationality. The unhooked `matches_source` is 0%, so the effect is real, not a baseline
artefact. The vision encoder's own output is therefore not null before the object.

**P9:** see section 2. The image movers carry 87.5-90.9% of transfer at logit level in both conditions.

**P10 (seed 1, n=128; `src 1st` at base values):**

| freeze at base | language | capital | currency | calling_code |
|---|---|---|---|---|
| heads only | 100.0 | 96.9 | 79.5 | 99.0 |
| + MLP 23-26 | 0 | 0 | 0 | 0 |
| + any 3 of 23-26 (leave-one-out) | 0-0.8 | 0 | 0-3.1 | 0 |
| + MLP 25 alone | 70.2 | 43.8 | 7.9 | **0** |
| + MLP 27 alone | 86.3 | 86.7 | 50.4 | 95.0 |

`@mean` and `@other` columns are in `results/tier3/severed_loo_n128_s1/summary.json`. For example,
`heads+mlp[23-26]@other` lands on the third country 100% of the time for capital.

**P11 (calling_code, k=10):**
- `image_patch` = `all_downstream_continuous` (full 0.906).
- Top-10 install, prefill only: full 6.2%. Continuous, every decode step: **95.3%**. custom10: 96.9%.
- Random-10 continuous: 1.6%.

**P12 (steering at L22):**
- The layer-17 direction does nothing at any α (first 0.0%, same as random).
- L22's own direction reads **95.8%** first at α = 1 (58.9% full), 90.1% at α = 2, and 44.3% at α = 4.

**P14 (common10 on animals, patch L10; cause %):**

| | body_covering | native_continent | primary_habitat | reproduction | taxonomic_class |
|---|---|---|---|---|---|
| heads_common10 | 4.3 | **49.6** | 6.1 | 6.9 | 1.4 |
| random_heads | 1.1 | 2.5 | 3.6 | 1.6 | 0.0 |
| full_image | 61.7 | 90.9 | 90.2 | 27.4 | 96.0 |

The entity fact (continent) transfers through the movers; category traits barely move.

**P19:** every donor attribute's top-8 sits in blocks 19-21 and shares 7-8 of 8 heads with router8.
Restoring the top-8 removes 90.0-99.1% of the effect.

**P20 (router_k 16 vs E4's 8, eligible rows, B.A = composed answer):**

| arm | E4 (k=8) | P20 (k=16) |
|---|---|---|
| compose | 78.8% (other 17.7%) | 82.5% (other 15.7%) |
| routers | 85.6% | 92.2% |
| target_circuit | 80.3% | 85.9% |

The answer-format failure shrinks only slightly.

**P22 (entity patch at L_e = 4; transfer):**

| freeze | image | text |
|---|---|---|
| entity MLPs 4-9 @ base | 32.3% | **0.9%** |
| entity MLPs 4-9 @ mean | 84.3% | 80.1% |
| readout 23-26 @ base | 1.7% | 1.7% |

Text depends on the early entity-position MLPs much more than images do.

**P23 (flags/capital ceilings):**
- `ring:0` reads 40.6 / 40.6 / 31.2 / 28.1 / 21.9 / 9.4% at L2-L22.
- `ring:1` reads at most 12.5%, and rings 3 and 4 read 0% at every layer.
- `ring:0+2+3+4` reads 62.5-68.8% early on.
- So identity does not spread outward ring by ring.
- Celebrity face halves (8 tokens each): 37.5% at L2 and 22-25% at L10/L18 for both halves.

**P25:**

| arm (L24, n=96) | first token | full answer |
|---|---|---|
| blocks 16-23 | 97.9% | 76.0% |
| without block 19 | 97.9% | 72.9% |
| without block 20 | 91.7% | 67.7% |
| without block 21 | 96.9% | 75.0% |

Block 19 is no keystone. `field` / `tok:-2` / `last_token` curves are in `results/tier3/switch_field_scopes/`.

**P26:**
- `--verify_cached` raised in both runs, on the first token of arm `last_token/attention/L22/prefill/switch`:
  - sdpa: max |dlogit| 0.227, mean 0.043;
  - eager with `--no_share_prefix`: max 0.563, mean 0.096;

  against atol = rtol = 0.02. The top token is identical in both runs (64871).
- Paired runs over 1,776 rows: cached batch 64 differs from serial on **7** texts (2 scores), cached
  batch 1 on 9 (3 scores), and batch 64 vs batch 1 on 8.
- The cached engine's logits drift beyond tolerance, but its answers almost never change.

**P27 (n=233):**
- `img_patch` 99.6%.
- Knockout of object keys: 99.1%. Background keys: 97.4%.
- With the question route cut as well: object 94.4%, background 90.6%, all mover keys **45.1%**.

## 4. Unexpected, verbatim

- **P26 verify** (sdpa):
  `AssertionError: Cached/reference mismatch: last_token/attention/L22/prefill/switch, token 0; model_dtype=torch.bfloat16, logits_dtype=torch.bfloat16, max_abs=0.226562, mean_abs=0.0425631, actual_top=[64871], expected_top=[64871], atol=rtol=0.02`
  - The eager / no-share run gave the same message with `max_abs=0.5625, mean_abs=0.096028`.
  - `set -e` stopped the lane, so P26's four remaining commands were run as written in `run_tier3.sh`,
    with the same log naming, followed by `run_tier3.sh p22`. No code or script was edited.
- **Scoring:** `score.py --attribute all` raises `KeyError: 'target_attribute'`, because VADE's
  `data/flags/tuples/identity/` (600 rows) has no such field. P3 and P14 were scored per attribute into
  `results/tier3/*/scores/`, as earlier `head_swap_vade` results were (P14 against the pruned 7B
  animals tuples).
- **A committed log was appended to:** `logs/attr_head_trace/flags/attr_head_trace_patch15_blocks15-22_all_text.log`
  gained 160 lines from P6, because the log name does not encode `--countries`. Nothing was removed.
  The change is left uncommitted; a clean copy is in `logs/tier3/p6_*`.
- **Shared cache:** the smoke run wrote `../VADE/results/Qwen2.5-VL-7B-Instruct/flags/source_activation_cache/full_image.pt`.
  It did not exist before. P1's cached arm read it, and the uncached arm agrees with it.
- **P16 was started, then stopped** when the runbook deferred it. Its partial log and ceiling log are
  left untracked and unpackaged.
- **Benign warnings:**
  - `PYTORCH_CUDA_ALLOC_CONF is deprecated` (Session 1 lanes);
  - `--cause_only` scoring warns "10296 tuple rows had no matching prediction";
  - `head_swap_vade` notes hook calls that ran past the donor's EOS.

## 5. GPU usage

`nvidia-smi` sampled every 30 s from 06:36 to 14:47 (982 samples): `/workspace/runlogs/gpu_usage.csv`.
The Session 1 snapshot is in `logs/tier3/GPU_USAGE_session1.md`.

| period | util | memory mean / max | power mean / p90 / max |
|---|---|---|---|
| S1, 3 default lanes | 97% | 50.3 / 51.2 GB | 208 / 310 / 389 W |
| S1 2 lanes + S2 a100 lane | 97% | 55.5 / 57.4 GB | 208 / 315 / 395 W |
| S2, 3 lanes (a100 + 2 default) | 99% | 65.0 / **78.8** GB | 204 / 300 / 371 W |
| S2 lane 1 alone (P20, a100) | 83% | 29.7 / 30.9 GB | 191 / 396 / 410 W |

- **Per-lane peaks:**
  - default lanes: 16.4-18.2 GB;
  - a100 lanes: P16 `ceiling_sweep` 23.1 GB, P14 `head_swap_vade` 21.9 GB, P20 `circuit_compose` 31.6 GB;
  - P26's batch-64 cached run: 24.5 GB.
- **Memory came close to the card's limit.** The 78.8 GB peak exceeded the runbook's ~70 GB target, as
  P26's batch-64 run overlapped P20 and P25. Nothing ran out of memory.
- **The work is CPU-dispatch bound throughout.** Mean power stayed near 205 W of 400 W, with one Python
  core at 100% per lane.
- **Lanes:** three lanes gave about 1.5x throughput. One P1 eval took 7 min alone and 14 min with three
  lanes.
- **The a100 profile helps the steps it batches.** P20 drew up to 410 W running alone, yet still used
  only 83% of the GPU's time and took 180 min. The long steps (P3 install 217 min, P20 180 min, P14
  126 min, P6 114 min) dominate the wall time.
- **For the next session:** MPS from a clean start, and a per-step memory budget for runs that set
  their own batch 64 (P26).

## 6. Outputs

Committed in `688bab8` (Session 1: P1, P2, P4, P5, P10, smoke), `7b1e4a4` (rest of Session 1) and
`d61ea67` (Session 2). Package: `/workspace/tier3_outputs.tgz`, containing `logs/tier3 results/tier3`
and the `_tier3`-suffixed ceiling logs. P16's partial files are excluded.
