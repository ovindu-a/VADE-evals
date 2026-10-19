# Run report: paper batch (PAPER_RUNBOOK.md X1, X2)

Pod session 2026-10-19, 06:15-07:46 UTC. Everything in PAPER_RUNBOOK.md section 3 ran to completion.
Two things need reading before the numbers:

- **X1 crashed on the first attempt (smoke), was fixed off-pod in `0897212`, and reran clean.**
- **X2 self-check 3 (cross self-patch ≥ 85%) fails on all four cross runs (80.9-81.2%).** The patch
  itself is exact: every self row reproduces the model's own unhooked answer (640/640). The
  shortfall is `head_cross` scoring against the stored lowercase labels, not the model's own
  answers (section 4). Every brands/celebrities cross number is deflated by the same mismatch.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, driver 580.105.08, CUDA 13.0 |
| torch / transformers | 2.9.1+cu130 (pod image) / 5.16.1 |
| VADE | `b8dfaf9` |
| VADE-evals | `0e81f4c` at start; `0897212` (X1 fix) pulled at ~06:37, before lanes 2-3 and the X1 rerun. Lane 1 started on `0e81f4c`; `0897212` only touches `text_entity_contrast.py`, which no X2 step imports. |
| MPS | **on** (daemon started cleanly; a test CUDA client ran under it; shut down after the lanes) |
| Tests | `pytest tests/`: 408 passed, 0 failed, 0 skipped (incl. `test_crossmodal.py`). After the pull: `test_crossmodal.py` 8 passed. |

The pod was missing `transformers`, so GPU_RUNBOOK.md section 1 was applied: the image's CUDA torch
was kept; `transformers==5.16.*`, `accelerate scikit-learn pillow numpy tqdm huggingface_hub joblib
pytest` installed; `pyvene --no-deps`, then `matplotlib`, the one module its import named.

## 2. Steps: status, lane, wall time

| step | lane | status | wall |
|---|---|---|---|
| smoke_x1 | smoke | **crashed** (section 3) | ~4 min |
| smoke_x2_swap / _cross / _severed | smoke (by hand) | ok | 0 / 0 / 0 min |
| x1_crossmodal | own window (beside lane 1) | ok, after the fix | 5 min |
| x2_swap_celeb_a | 1 | ok | 28 min |
| x2_swap_brands | 2 | ok | 23 min |
| x2_cross_brands_matched / _step0 | 2 | ran; self-check 3 fails (section 4) | 7 / 6 min |
| x2_severed_brands | 2 | ok | 10 min |
| x2_swap_celeb_b | 3 | ok | 21 min |
| x2_cross_celeb_matched / _step0 | 3 | ran; self-check 3 fails (section 4) | 11 / 9 min |
| x2_severed_celeb | 3 | ok | 6 min |
| x2_score | CPU | ok, 64 summaries (8×3 brands + 6×4 + 4×4 celebrities) | ~1 min |

Lanes: lane 1 started 06:34, lane 2 06:45, lane 3 06:55; the last lane finished 07:44.

**Deviations from the runbook, all recorded here:**
1. Smoke stopped at `smoke_x1` (`set -e`), so the other three smoke commands never ran. They were
   run by hand, verbatim from `smoke()`, from a scratch script (logs `logs/paper/smoke_x2_*`).
2. After the user pushed the X1 fix, X1 was run on its own (`GPU=a100 bash scripts/run_paper.sh x1`),
   as the updated runbook says, alongside lane 1. Lane 3 was therefore started as its three X2 steps
   by name (`x2_swap_celeb_b x2_cross_celeb x2_severed_celeb`) rather than `lane3`, so X1 did not run
   twice.
3. No step ran out of memory; nothing was rerun without `GPU=a100`.

The runbook's 2.5-4 h per lane (extrapolated from batch-16 Tier-3 rates) was far off: at batch 64
each lane took 28-60 min, and the whole GPU batch took 70 min wall.

## 3. X1: mover-head content, image vs text

### What failed first

`smoke_x1` crashed after the capture, in `ridge_retrieval` at `resid@0`:

```
  File "/workspace/VADE-evals/methods/text_entity_contrast.py", line 230, in ridge_retrieval
    W = Ac.T @ torch.linalg.solve(K + lam * torch.eye(len(tr), dtype=K.dtype), Bt - mu_b)
torch._C._LinAlgError: torch.linalg.solve: The solver failed because the input matrix is singular.
```

Diagnosed from the smoke's `crossmodal_reps.pt`: at layer 0 every row's last-column residual is
bit-identical (max |row − mean| = 0.0 in both conditions; 0.045 at layer 1), so `Ac = 0`, `K = 0`,
`lam = alpha·mean(diag K) = 0`. Deterministic and size-independent (the representation part always
uses all 84×4 rows), so the full step would have crashed identically. Not patched on the pod; fixed
off-pod in `0897212` (λ floored at 1e-6, plus a test), then rerun.

### Self-checks (full run)

| # | check | result |
|---|---|---|
| 1 | `clean` 0%, `source_clean` 100%, n_eligible ~220-240 | **pass**: 0.0 / 100.0% in both tables; n = 230 (image), 232 (text) — E6 had 230/232 |
| 2 | `random10<-same`, `random10<-cross` ≤ 5% | **pass**: 0.0% on all four |
| 3 | `movers<-cross:self` keeps base ≥ 85% | **pass**: 99.1% (image), 100.0% (text) |
| 4 | `movers<-same` within ~15 pts of E6's 87-91% | **pass**: 91.7% (image), 90.9% (text) |
| 5 | `resid@0` top-1 ≈ chance (1.2%) | **pass**: 0.0% centered and ridge — every candidate ties, which `retrieval` scores as 0% top-1 (the new test asserts exactly this). Nothing above chance. |

### Install tables (as printed)

```
CROSSMODAL install into the image condition (donor = cross/same condition), n=230 eligible
  arm                                              n  transfer        95% CI    base   other   meanLD  normLD
  clean                                          230     0.0%         0%-2% 100.0%   0.0%   -10.59    0.00
  source_clean                                   230   100.0%      98%-100%   0.0%   0.0%    10.98    1.00
  install:movers<-same                           230    91.7%       87%-95%   2.2%   6.1%     7.41    0.83
  install:random10<-same                         230     0.0%         0%-2% 100.0%   0.0%   -10.59    0.00
  install:movers<-cross                          230    91.7%       87%-95%   2.2%   6.1%     7.67    0.85
  install:random10<-cross                        230     0.0%         0%-2%  99.6%   0.4%   -10.55    0.00
  install:movers<-cross:normmatch                230    90.4%       86%-94%   3.0%   6.5%     7.17    0.82
  install:movers<-cross:self                     230     0.0%         0%-2%  99.1%   0.9%   -10.85   -0.01

CROSSMODAL install into the text condition (donor = cross/same condition), n=232 eligible
  arm                                              n  transfer        95% CI    base   other   meanLD  normLD
  clean                                          232     0.0%         0%-2% 100.0%   0.0%   -11.09    0.00
  source_clean                                   232   100.0%      98%-100%   0.0%   0.0%    11.24    1.00
  install:movers<-cross                          232    88.8%       84%-92%   2.6%   8.6%     7.41    0.83
  install:random10<-cross                        232     0.0%         0%-2% 100.0%   0.0%   -11.10   -0.00
  install:movers<-cross:normmatch                232    86.6%       82%-90%   3.4%   9.9%     6.97    0.81
  install:movers<-cross:self                     232     0.0%         0%-2% 100.0%   0.0%   -10.89    0.01
  install:movers<-same                           232    90.9%       87%-94%   1.3%   7.8%     7.66    0.84
  install:random10<-same                         232     0.0%         0%-2% 100.0%   0.0%   -11.07    0.00
```

**Reading: cross ≈ same — the moved payload is modality-invariant.** Image←text movers install at
91.7%, identical to image←image; text←image 88.8% vs text←text 90.9%. Norm-matching does not help
(nothing to recover).

### Representation (n = 336 rows, 84 countries, chance top-1 1.2%)

`resid` top-1, centered / ridge image→text, by layer:

| L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctr | 0.0 | 8.9 | 14.0 | 18.2 | 6.8 | 7.7 | 6.5 | 6.0 | 6.8 | 8.0 | 7.7 | 6.2 | 7.4 | 8.6 |
| ridge | 0.0 | 12.2 | 15.2 | 21.1 | 7.4 | 7.1 | 8.6 | 10.1 | 10.1 | 17.6 | 15.8 | 15.2 | 14.6 | 15.5 |

| L | 14 | 15 | 16 | 17 | 18 | 19 | **20** | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctr | 11.6 | 19.6 | 15.8 | 18.5 | 24.4 | 32.1 | **92.0** | 89.0 | 99.4 | 100 | 100 | 100 | 100 | 100 |
| ridge | 17.0 | 18.2 | 15.8 | 16.4 | 21.1 | 21.1 | **74.4** | 62.2 | 96.7 | 98.5 | 91.7 | 94.0 | 92.0 | 92.6 |

**Where the image run becomes the text run: layer 20** (first layer with `resid` ridge top-1 > 50%:
74.4%; centered 92.0%). It is ≤ 32% at every earlier layer.

`entity` site (image: object tokens; text: name token): centered top-1 79.8% at L1, 92-99% from L3
on; ridge only 20-60% (peak 59.5% at L10, falling to 37-44% by L22-27).

Mover heads (centered / ridge):

| site | ctr | ridge |
|---|---|---|
| movers (concat z) | 100.0 | 100.0 |
| movers (joint write) | 100.0 | 100.0 |
| 21.1 | 99.4 | 99.1 |
| 21.5 | 96.7 | 88.7 |
| 22.13 | 92.3 | **46.7** |
| 22.15 | 99.1 | 97.9 |
| 22.17 | 97.3 | 89.3 |
| 22.19 | 100.0 | 100.0 |
| 23.3 | 94.6 | 88.7 |
| 23.4 | 100.0 | 98.8 |
| 23.6 | 98.2 | 97.3 |
| 23.17 | 97.6 | **68.2** |

The text→image ridge direction and same−diff cosines are in `summary.json` (`crossmodal.representation`).
Full per-layer printout: `logs/paper/x1_crossmodal_20261019_063743.log`.

## 4. X2: brands and celebrities

### Self-checks

| # | check | result |
|---|---|---|
| 0 | `head check ok` | **pass**: printed by every step that calls `heads()` (smoke, lane 1 ×1, lane 2 ×3, lane 3 ×2). `x2_swap_celeb_b` does not call it by design (it uses only the fixed cons6/common10 lists). |
| 1 | `random_heads_*` ≤ 5% cause on every attribute; `clean` ≈ 0 | **celebrities pass** (random 0.0-1.8%, clean 0.0-1.8%). **brands `industry` fails**: random 6.0 / 7.2 / 8.3% — but `clean` itself reads 5.7% there, so the nulls sit at the clean baseline (+0.3 to +2.6 pts). All other brands cells 0.0%. |
| 2 | `full_image` ≥ 70% on hq_country, nationality, native_language | **pass**: 99.4 / 96.4 / 99.8% |
| 3 | cross self-patch keeps base ≥ 85% | **fail**, all four runs: brands 81.2 / 81.2%, celebrities 80.9 / 80.9% (matched / step0). See below. |
| 4 | severed `identity: bit-exact` | **pass**, both runs (brands logit liveness 7.94, celebrities 11.42; read-back 0.0) |
| 5 | no `KeyError` | **pass**: none in any step |

**Self-check 3 in detail.** In every self row of all four runs, the patched generation equals the
model's own unhooked answer (`generated == clean`: brands 192/192 ×2, celebrities 256/256 ×2). The
missing ~19% are rows where the model's clean answer does not match the *stored* label under
VADE's `contains_label` — e.g. THE_GUARDIAN hq ` the UK.`, INTEL industry ` Technology sector.`,
META founded ` 2012.`; misses by attribute, brands matched: founded_year 19, industry 12, hq_country 5
of 64 each; celebrities: birth_year 22, nationality 15, native_language 9, occupation 3. So the self-patch is
an exact identity, and the 81% is the clean model's stored-label accuracy. `head_cross` has no
`--gold own` (unlike `head_severed`), so **every brands/celebrities cross cell below is scored
against stored labels and deflated by the same mismatch.** Per the ground rules nothing was patched;
the runs were left to complete and are kept. The rows store `generated` and the base's `clean`
answer but not the source's own answers, so a full own-answer rescore needs a rerun once
`head_cross` gains own-answer golds (add to fix G).

### Install: cause / iso / final per arm × attribute (VADE `score.py`, pruned tuples)

**brands** (n = 612 / 710 / 723 scored rows)

| arm | founded_year | hq_country | industry |
|---|---|---|---|
| clean | 0.0 / 96.0 / 48.0 | 0.0 / 92.0 / 46.0 | 5.7 / 98.0 / 51.9 |
| heads_hq_top8 | 14.0 / 62.9 / 38.4 | 61.8 / 51.8 / 56.8 | 14.3 / 28.8 / 21.5 |
| heads_cons6 | 7.0 / 69.0 / 38.0 | 50.6 / 66.6 / 58.6 | 10.6 / 50.2 / 30.4 |
| heads_common10 | 11.7 / 50.3 / 31.0 | **96.0** / 51.0 / 73.5 | 13.6 / 13.7 / 13.6 |
| full_image | 43.9 / 35.1 / 39.5 | 99.4 / 33.8 / 66.6 | 38.9 / 2.2 / 20.6 |
| random_heads_hq_top8 | 0.0 / 96.4 / 48.2 | 0.0 / 92.0 / 46.0 | 7.2 / 98.5 / 52.9 |
| random_heads_cons6 | 0.0 / 96.0 / 48.0 | 0.0 / 90.8 / 45.4 | 8.3 / 95.7 / 52.0 |
| random_heads_common10 | 0.0 / 96.6 / 48.3 | 0.0 / 92.3 / 46.2 | 6.0 / 97.8 / 51.9 |

**celebrities** (n = 1994 / 1863 / 1589 / 1505)

| arm | birth_year | nationality | native_language | occupation |
|---|---|---|---|---|
| clean | 0.0 / 99.7 / 49.9 | 1.8 / 97.2 / 49.5 | 0.0 / 96.7 / 48.4 | 0.0 / 97.3 / 48.6 |
| heads_nat_top8 | 3.8 / 56.0 / 29.9 | 74.9 / 53.2 / 64.0 | 66.0 / 50.8 / 58.4 | 0.9 / 32.7 / 16.8 |
| heads_lang_top8 | 7.2 / 42.4 / 24.8 | 88.5 / 39.7 / 64.1 | 93.5 / 41.2 / 67.4 | 0.0 / 14.9 / 7.5 |
| heads_cons6 | 3.0 / 75.6 / 39.3 | 42.1 / 70.7 / 56.4 | 45.3 / 72.5 / 58.9 | 0.0 / 57.7 / 28.9 |
| heads_common10 | 8.3 / 42.2 / 25.2 | 89.1 / 41.4 / 65.2 | 91.3 / 41.5 / 66.4 | 0.4 / 16.7 / 8.5 |
| full_image | 31.8 / 32.3 / 32.1 | 96.4 / 31.1 / 63.8 | 99.8 / 32.8 / 66.3 | 5.5 / 2.3 / 3.9 |
| random_heads_nat_top8 | 0.0 / 99.9 / 49.9 | 1.6 / 97.4 / 49.5 | 0.0 / 97.0 / 48.5 | 0.0 / 97.9 / 48.9 |
| random_heads_lang_top8 | 0.0 / 99.5 / 49.7 | 1.8 / 97.8 / 49.8 | 0.0 / 97.5 / 48.7 | 0.0 / 97.7 / 48.8 |
| random_heads_cons6 | 0.0 / 99.6 / 49.8 | 1.6 / 97.0 / 49.3 | 0.0 / 97.0 / 48.5 | 0.0 / 97.3 / 48.7 |
| random_heads_common10 | 0.0 / 99.5 / 49.8 | 1.6 / 94.8 / 48.2 | 0.0 / 94.8 / 47.4 | 0.0 / 94.8 / 47.4 |

Flags' `common10` installs at 96.0% on brands/hq_country and 89-91% on celebrities nationality/
native_language — higher than each entity's own top-8 (61.8%; 74.9% / 93.5%). Read-back was exact
(0.00e+00) for every head set in all three installs.

### Cross-question (stored-label scoring; see self-check 3)

| | brands matched | brands step0 | celeb matched | celeb step0 | flags R11 |
|---|---|---|---|---|---|
| self → base_q1 | 81.2 | 81.2 | 80.9 | 80.9 | 91.8-93.4 |
| donor asked SAME question → source_q1 | 22.4 | 16.1 | 22.3 | 20.7 | 81.2 |
| donor asked ANOTHER question → source_q1 | 5.2 | 5.5 | 12.0 | 12.2 | 65.1 |
| shift (pp) | −17.2 | −10.7 | −10.3 | −8.5 | −16.1 |
| question-only cell → A's Q2 answer | 0.0 | 0.0 | 0.0 | 0.0 | |

Full cell tables (n, base_q1, source_q1, source_q2, base_q2, other, ambiguous):

```
brands matched                         brands step0
  self     192  81.2  0.0 0 0 18.8 0     self     192  81.2  0.0 0 0 18.8 0
  flag     192  46.9 22.4 0 0 30.7 0     flag     192  67.2 16.1 0 0 16.7 0
  question 384  74.7  0.0 0 0 25.3 0     question 384  81.0  0.0 0 0 19.0 0
  both     384  71.1  5.2 0 0 23.7 0     both     384  78.6  5.5 0 0 15.9 0
celebrities matched                    celebrities step0
  self     256  80.9  0.0 0 0 19.1 0     self     256  80.9  0.0 0 0 19.1 0
  flag     256  50.0 22.3 0 0 27.7 0     flag     256  62.5 20.7 0 0 16.8 0
  question 768  74.6  0.0 0 0 25.4 0     question 768  80.6  0.0 0 0 19.4 0
  both     768  64.5 12.0 0 0 23.6 0     both     768  71.5 12.2 0 0 16.3 0
```

Absolute transfer is ~4× below flags even allowing for the ~19-pt label deflation, with the
single top-8 heads (`head_cross` takes BH / CN, not common10). The shift survives step0 on both
entities (−10.7, −8.5 pp), so at face value the content is partly question-conditioned. Treat these
as provisional until the own-answer rescore.

### Severed (`--gold own`, n = 128; heads = BH for brands, CN for celebrities)

Columns: src/base/3rd = first-token divergence hits; `kept` = src div ÷ `heads` src div; txt = VADE
text match. Note `kept` is meaningless for occupation, where `heads` itself moves 1 row (0.8%).

**brands / founded_year**

```
                     arm    n  src div  base div  3rd div   kept  src txt  base txt  other  3rd txt
                   clean  128     0.0%    100.0%        -   0.00     0.0%     54.7%  45.3%        -
                   heads  128    22.7%     56.2%        -   1.00     9.4%     10.9%  79.7%        -
        heads+mlp[22-27]  128     0.0%     99.2%        -   0.00     0.0%     55.5%  44.5%        -
   heads+mlp[22-27]@mean  128    35.2%     32.0%        -   1.55     0.8%      0.8%  98.4%        -
  heads+mlp[22-27]@other  128    25.8%     30.5%    96.9%   1.14     0.0%      0.0%  48.4%    51.6%
           heads+mlp[23]  128     9.4%     76.6%        -   0.41     0.8%     28.9%  70.3%        -
      heads+mlp[23]@mean  128    24.2%     56.2%        -   1.07     5.5%      7.8%  86.7%        -
     heads+mlp[23]@other  128    18.8%     54.7%    10.2%   0.83     1.6%      4.7%  90.6%     3.1%
           heads+mlp[24]  128     6.2%     79.7%        -   0.28     0.0%     28.9%  71.1%        -
      heads+mlp[24]@mean  128    23.4%     48.4%        -   1.03     3.9%      4.7%  91.4%        -
     heads+mlp[24]@other  128    19.5%     48.4%    14.8%   0.86     0.8%      3.1%  93.0%     3.1%
           heads+mlp[25]  128     2.3%     93.0%        -   0.10     0.0%     50.0%  50.0%        -
      heads+mlp[25]@mean  128    26.6%     43.0%        -   1.17     1.6%      0.0%  98.4%        -
     heads+mlp[25]@other  128    21.1%     40.6%    55.5%   0.93     0.8%      0.8%  75.0%    23.4%
           heads+mlp[26]  128    13.3%     77.3%        -   0.59     3.1%     32.0%  64.8%        -
      heads+mlp[26]@mean  128    31.2%     50.0%        -   1.38    10.9%     12.5%  76.6%        -
     heads+mlp[26]@other  128    25.8%     46.1%    34.4%   1.14     6.2%      3.1%  76.6%    14.1%
       heads+attn[24-27]  128    10.2%     77.3%        -   0.45     6.2%     32.8%  60.9%        -
  heads+attn[24-27]@mean  128    29.7%     53.1%        -   1.31    10.2%     13.3%  76.6%        -
 heads+attn[24-27]@other  128    23.4%     46.1%    26.6%   1.03     8.6%      5.5%  81.2%     4.7%
```

**brands / industry**

```
                     arm    n  src div  base div  3rd div   kept  src txt  base txt  other  3rd txt
                   clean  128     0.0%    100.0%        -   0.00     3.1%     78.9%  18.0%        -
                   heads  128     4.1%     90.9%        -   1.00     7.0%     77.3%  15.6%        -
        heads+mlp[22-27]  128     0.0%     95.0%        -   0.00     3.1%     79.7%  17.2%        -
   heads+mlp[22-27]@mean  128     6.6%     72.7%        -   1.60     7.8%     45.3%  46.9%        -
  heads+mlp[22-27]@other  128     6.6%     46.3%    42.3%   1.60     6.2%     26.6%  29.7%    37.5%
           heads+mlp[23]  128     1.7%     91.7%        -   0.40     4.7%     78.1%  17.2%        -
      heads+mlp[23]@mean  128     5.0%     89.3%        -   1.20     7.8%     75.0%  17.2%        -
     heads+mlp[23]@other  128     1.7%     90.9%     3.6%   0.40     3.9%     76.6%  15.6%     3.9%
           heads+mlp[24]  128     2.5%     94.2%        -   0.60     5.5%     78.9%  15.6%        -
      heads+mlp[24]@mean  128     3.3%     90.1%        -   0.80     5.5%     76.6%  18.0%        -
     heads+mlp[24]@other  128     1.7%     89.3%     3.6%   0.40     3.9%     76.6%  14.1%     5.5%
           heads+mlp[25]  128     2.5%     92.6%        -   0.60     5.5%     78.1%  16.4%        -
      heads+mlp[25]@mean  128     2.5%     91.7%        -   0.60     5.5%     78.1%  16.4%        -
     heads+mlp[25]@other  128     2.5%     87.6%     3.6%   0.60     5.5%     76.6%  12.5%     5.5%
           heads+mlp[26]  128     3.3%     90.1%        -   0.80     6.2%     79.7%  14.1%        -
      heads+mlp[26]@mean  128     5.0%     88.4%        -   1.20     7.8%     78.9%  13.3%        -
     heads+mlp[26]@other  128     3.3%     88.4%     0.9%   0.80     6.2%     78.1%  12.5%     3.1%
       heads+attn[24-27]  128     7.4%     87.6%        -   1.80    10.2%     75.0%  14.8%        -
  heads+attn[24-27]@mean  128    18.2%     69.4%        -   4.40    19.5%     64.1%  16.4%        -
 heads+attn[24-27]@other  128    16.5%     56.2%    18.9%   4.00    17.2%     50.8%  15.6%    16.4%
```

**celebrities / occupation**

```
                     arm    n  src div  base div  3rd div   kept  src txt  base txt  other  3rd txt
                   clean  128     0.0%    100.0%        -   0.00     0.0%    100.0%   0.0%        -
                   heads  128     0.8%     98.4%        -   1.00     0.8%     99.2%   0.0%        -
        heads+mlp[22-27]  128     0.0%    100.0%        -   0.00     0.0%    100.0%   0.0%        -
   heads+mlp[22-27]@mean  128    18.0%     73.4%        -  23.00     9.4%     69.5%  21.1%        -
  heads+mlp[22-27]@other  128    13.3%     48.4%    33.6%  17.00     0.8%     36.7%  22.7%    39.8%
           heads+mlp[23]  128     0.0%     99.2%        -   0.00     0.0%    100.0%   0.0%        -
      heads+mlp[23]@mean  128     0.0%     98.4%        -   0.00     0.0%    100.0%   0.0%        -
     heads+mlp[23]@other  128     0.0%     96.9%     0.8%   0.00     0.0%     99.2%   0.0%     0.8%
           heads+mlp[24]  128     0.0%    100.0%        -   0.00     0.0%    100.0%   0.0%        -
      heads+mlp[24]@mean  128     0.8%     95.3%        -   1.00     0.8%     99.2%   0.0%        -
     heads+mlp[24]@other  128     0.0%     93.8%     0.8%   0.00     0.0%     98.4%   0.8%     0.8%
           heads+mlp[25]  128     0.0%    100.0%        -   0.00     0.0%    100.0%   0.0%        -
      heads+mlp[25]@mean  128     0.8%     95.3%        -   1.00     0.8%     98.4%   0.8%        -
     heads+mlp[25]@other  128     1.6%     93.0%     0.0%   2.00     0.8%     97.7%   0.8%     0.8%
           heads+mlp[26]  128     0.0%    100.0%        -   0.00     0.0%    100.0%   0.0%        -
      heads+mlp[26]@mean  128     0.8%     93.8%        -   1.00     0.8%     97.7%   1.6%        -
     heads+mlp[26]@other  128     2.3%     91.4%     2.3%   3.00     1.6%     95.3%   0.8%     2.3%
       heads+attn[24-27]  128     0.8%     96.1%        -   1.00     0.8%     97.7%   1.6%        -
  heads+attn[24-27]@mean  128    10.2%     78.9%        -  13.00    10.2%     83.6%   6.2%        -
 heads+attn[24-27]@other  128     6.2%     71.1%    15.6%   8.00     3.9%     77.3%   2.3%    16.4%
```

**celebrities / birth_year**

```
                     arm    n  src div  base div  3rd div   kept  src txt  base txt  other  3rd txt
                   clean  128     0.0%    100.0%        -   0.00     0.0%     68.0%  32.0%        -
                   heads  128     7.8%     76.6%        -   1.00     0.8%     22.7%  76.6%        -
        heads+mlp[22-27]  128     0.0%    100.0%        -   0.00     0.0%     68.0%  32.0%        -
   heads+mlp[22-27]@mean  128    33.6%     28.9%        -   4.30     3.9%      3.9%  92.2%        -
  heads+mlp[22-27]@other  128    25.8%     24.2%    98.4%   3.30     0.0%      0.0%  43.0%    57.0%
           heads+mlp[23]  128     4.7%     88.3%        -   0.60     0.0%     46.1%  53.9%        -
      heads+mlp[23]@mean  128    13.3%     62.5%        -   1.70     1.6%     13.3%  85.2%        -
     heads+mlp[23]@other  128    19.5%     50.8%    10.9%   2.50     0.0%      4.7%  93.8%     1.6%
           heads+mlp[24]  128     3.1%     89.8%        -   0.40     0.0%     39.1%  60.9%        -
      heads+mlp[24]@mean  128    18.0%     57.0%        -   2.30     1.6%      2.3%  96.1%        -
     heads+mlp[24]@other  128    21.1%     43.0%    33.6%   2.70     0.0%      1.6%  89.1%     9.4%
           heads+mlp[25]  128     2.3%     94.5%        -   0.30     0.0%     61.7%  38.3%        -
      heads+mlp[25]@mean  128    21.1%     50.0%        -   2.70     5.5%      7.0%  87.5%        -
     heads+mlp[25]@other  128    29.7%     28.1%    68.0%   3.80     0.8%      1.6%  60.2%    37.5%
           heads+mlp[26]  128     3.1%     94.5%        -   0.40     0.0%     54.7%  45.3%        -
      heads+mlp[26]@mean  128    13.3%     68.8%        -   1.70     0.8%     20.3%  78.9%        -
     heads+mlp[26]@other  128    26.6%     42.2%    39.8%   3.40     1.6%      7.0%  68.0%    23.4%
       heads+attn[24-27]  128     4.7%     84.4%        -   0.60     0.0%     46.9%  53.1%        -
  heads+attn[24-27]@mean  128    16.4%     64.1%        -   2.10     2.3%     19.5%  78.1%        -
 heads+attn[24-27]@other  128    31.2%     40.6%    28.9%   4.00     3.9%      7.8%  76.6%    11.7%
```

The two years behave like flags' entity facts: freezing last-token MLPs 22-27 at base removes the
install entirely (kept 0.00), and `@other` lands on the third item's year (96.9% brands, 98.4%
celebrities), so those MLPs write the answer.

### Category attributes (industry, occupation)

**They behave like animals' category traits (P14), not like entity facts.** The movers barely
carry them: install cause industry 10.6-14.3% (clean baseline 5.7%), occupation 0.0-0.9%, against
50-96% for hq_country / nationality / native_language — the same shape as P14 (common10 on animals:
1.4-6.9% on four traits vs 49.6% on native_continent). Occupation goes further than P14: even
`full_image` at L19 transfers only 5.5% (P14's full_image was 27-96%), and the severed `heads` arm
moves 1/128 rows, so there is nothing for the freezes to remove. Industry's full_image is 38.9%.

## 5. GPU usage (`logs/paper/gpu_usage.csv`, 30 s samples, 06:33-07:44)

| period | jobs on the card | peak memory (MiB) | mean util | mean power |
|---|---|---|---|---|
| 06:33-06:37 | lane 1 | 21,821 | 55% | 189 W |
| 06:37-06:43 | lane 1 + X1 | 45,846 | 84% | 276 W |
| 06:43-06:44 | lane 1 | 23,141 | 50% | 239 W |
| 06:44-06:54 | lanes 1 + 2 | 47,832 | 88% | 219 W |
| 06:54-07:02 | lanes 1 + 2 + 3 | **72,633** (at 06:59:57) | 92% | 241 W |
| 07:02-07:44 | lanes 2 + 3 | 72,633 (07:02:27, as lane 1 was exiting) | 88% | 218 W |
| whole run | | 72,633 | 86% | 223 W of 400 |

Per-lane peaks are not separable from the CSV (most windows overlap another job). The one clean
reading: lane 1's `head_swap_vade` alone held ~22-23 GB; adding lane 2 added ~25 GB, and lane 3
another ~25 GB, landing right at the runbook's ~72 GB ceiling (below Tier-3's 78.8 GB).

**Three lanes vs Tier-3's ~1.5×:** the ten GPU steps sum to 126 min of per-step wall time, delivered
in 70 min of wall clock — 1.8×. That is an upper bound on the speed-up (each step ran slower under
contention than it would alone), so this does not establish that three lanes plus MPS beat 1.5×.
Mean power 223 W is close to Tier-3's ~205 W: still CPU-dispatch bound. Three lanes overlapped
for only ~7 min (lane 1 was short), so most of the run was two lanes.

## 6. Unexpected, verbatim

- The X1 smoke crash (section 3).
- `head_swap_vade`, all three installs: `note: 234 hook calls ran past the donor's recorded steps
  (donor hit EOS first); the last donor step was held.` (celeb_a; 66 for brands, 204 for celeb_b).
  Documented behaviour.
- `score.py`, every install arm: `WARNING: 5405 tuple rows had no matching prediction (of 6017 total)
  -- scoring on the remaining 612.` Expected from `--limit_pairs 128`.
- `head_severed` prints `third country from the same template` for brands/celebrities items too
  (flags wording; cosmetic).
- X1's output directory also contains an empty `rows.jsonl` (0 bytes) next to `crossmodal_rows.jsonl`.

## 7. Outputs

- `results/paper/` — `x2_swap_*` (+ `scores/`), `x2_cross_*`, `x2_severed_*`, `smoke/`
- `results/text_entity_contrast/flags_crossmodal/` — incl. `crossmodal_reps.pt` (276 MB, gitignored)
- `logs/paper/` — one log per step, lane consoles, `gpu_usage.csv`
- Packaged: `/workspace/paper_outputs.tgz` (all three trees, `.pt` files included)

Nothing existing under `results/` or `logs/` was modified. Nothing was committed or pushed.
