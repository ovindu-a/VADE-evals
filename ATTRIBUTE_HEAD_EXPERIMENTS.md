# Attribute-side head experiments

Three new scripts, all downstream of [ATTRIBUTE_SWITCH_FINDINGS.md](ATTRIBUTE_SWITCH_FINDINGS.md).
They ask about the **question -> readout** path — which heads carry WHICH ATTRIBUTE
was asked — as opposed to `methods/head_trace.py`, which holds the question fixed
and swaps the image to trace the **image -> text** path. Different circuits;
do not quote one window as the other.

**Indexing.** `--patch_layer L` patches the residual at the output of block `L-1`.
`--blocks` are 0-based block indices. A traced block must be `>= patch_layer`, or
its attention has already run by the time the patch lands; this is rejected at
argument time rather than reported as a row of zeros.

## 1. `methods/attr_head_trace.py` — which heads carry the attribute selection

Same phase structure as `head_trace.py`, and literally the same code: the phase
machinery (`capture_head_outputs`, `probe_identity`, `head_patches`,
`per_head_delta`, `generate_with_patches`) is imported, not copied, so a fix
lands in both and the two results stay comparable. What differs is the ~60 lines
that build a same-image/different-question batch and patch the QUESTION tokens
instead of the image.

```bash
# Validate the grid and images; no torch, no model.
python methods/attr_head_trace.py --dry_run --n_countries 16

# Phase 1 + 1c only: the ranking and the self-test, no generation.
python methods/attr_head_trace.py --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
    --n_countries 16 --skip_knockout

# Everything, with the per-attribute ranking (free -- same forward passes).
python methods/attr_head_trace.py --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
    --n_countries 24 --per_attribute
```

Phases, and what each is for:

| phase | question | cost |
|---|---|---|
| 0 | clean baselines: does the model answer both questions? | 2 generations/batch |
| 1 | per-head `delta_z` / `delta_resid` / `delta_dla` when the question is swapped | 2 forwards/batch |
| 1c | **read-back identity** — does the patch address the tensor the capture reads, and does installing every traced head reproduce the question patch? | 3 forwards, one batch |
| 1b | connectivity telemetry + block-everything | 1 generation pass |
| 2 | cumulative knockout (necessity) | 1 pass per k |
| 3 | sufficiency — install only these heads into a clean run | 1 pass per k |
| 4 | shuffled-donor control | 1 pass per k in `--control_k` |

**Read phase 3, not phase 2.** The sweep's span table shows this mechanism is
redundant across blocks (any span covering 19-21 reads ~97-100%, and dropping
block 21 costs almost nothing if earlier blocks remain). Necessity is exactly
what redundancy defeats, so expect a flat phase-2 curve even for real conduits.
Phase 4 then decides whether a steep phase-3 curve is localization or an
artefact of perturbing the site — if rolling the donor one row does not carry
the answer with it, phase 3 is measuring the perturbation, not transfer.

**Read `first`, not `full`.** Printed side by side. A last-token intervention
cannot steer an answer past its first token; `capital` and `calling_code` golds
are 2-3 tokens, so their full-match rate is capped by answer length. `--per_attribute`
matters for the same reason the sweep's per-attribute table does: the
single-block peak moves with the attribute (block 19 for `currency` and
`calling_code`, 20 for `capital`, nowhere at all for `language`), so a pooled
ranking can hide an attribute-specific head.

## 2. `methods/attr_capture.py` — the crossed activation grid

One clean forward per (item, attribute) over a fully crossed grid, capturing
four sites at every traced block at the answer position. No intervention,
nothing trained.

```bash
python methods/attr_capture.py --dry_run --n_items 64          # sizing only
python methods/attr_capture.py --entity flags --n_items 64 --blocks 15 16 17 18 19 20 21 22
```

Writes `results/attr_capture/<entity>/<tag>/`:

| file | contents |
|---|---|
| `meta.json` | model, blocks, sites, shapes, prompts, `head_proj_norms`, hashes |
| `index.jsonl` | one row per run: `row`, `item`, `attribute`, `condition`, `gold_ids` |
| `acts_<site>.npy` | memmap `[n_runs, n_blocks, n_positions, hidden]` float32 |

Sites: `residual` (block output, pre-norm), `attn_output`, `mlp_output`,
`attn_head_output`. The last is stored at **full width** so the 28x128 head
structure survives — selecting k heads is a slice at analysis time, whereas
storing a chosen subset would freeze a head ranking into the data.

64 items x 4 attributes x 8 blocks is ~117 MB. float32 rather than float16 on
purpose: the model computes in bf16, which has float32's exponent range, and a
residual-stream outlier coordinate above 65504 would silently become `inf`.

`head_proj_norms` stores `||W_O_h||_F` per (block, head) at capture time, so the
analysis can distinguish a head whose output varies a lot from one whose
variation actually lands in the residual stream — without loading the model.

## 3. `methods/attr_directions.py` — the analysis (no model, no GPU)

```bash
python methods/attr_directions.py results/attr_capture/flags/blocks15-22_n64 \
    --trace logs/attr_head_trace/flags/attr_head_trace_patch15_blocks15-22_all_text.json \
    --image_trace logs/.../head_trace_patch21_blocks18-27.json \
    --top_k 16 --json_out results/attr_capture/flags/directions.json
```

**The centering is the method.** Raw activations are dominated by which flag is
in the image. Subtracting each item's own mean over its four questions removes
the item main effect exactly, and what survives is what the question changed:

```
Y[c,a] = X[c,a] - mean_a' X[c,a']        m_a = mean_c Y[c,a]
```

Because `sum_a Y[c,a] = 0` by construction, the four centroids sum to zero and
span at most `A-1 = 3` dimensions. A reported effective rank of 3 is therefore
the **ceiling, not a finding**; a rank near 1 would be a finding.

What it reports:

- **separability** per (site, block): `attr_var_explained` (between/(between+within)
  on the demeaned grid) and a leave-one-**item**-out nearest-centroid accuracy.
  Held out by item, never by row — the four rows of one item are dependent after
  centering, so holding out rows would leak.
- **geometry**: cosines between the four centroids and their singular values.
  Four near-orthogonal directions is a different claim from one axis the
  attributes sit along.
- **heads**: the same separability per (block, head) on `attn_head_output` — a
  **correlational** ranking. `--trace` compares it against `attr_head_trace`'s
  **causal** one (rank correlation + top-k overlap). Low overlap is a result,
  not an error: it is the same gap RESULTS.md records between the Phase B
  classifier proxy and real interventions.
- **image_heads** (`--image_trace`): the variance decomposition below.

### The image-head question

For each of an image `head_trace`'s top-k heads, decompose its variance over the
crossed grid — an exact orthogonal split on a balanced design:

```
X[c,a] = m + alpha_c + beta_a + gamma[c,a]
SS_total = A*sum||alpha||^2 + C*sum||beta||^2 + sum||gamma||^2
```

- **item ~ 1** — the head ferries the same image content whatever is asked.
- **large attribute** — the head carries a question-dependent component that is
  the same for every flag, so it is not image data at all.
- **large interaction** — the head reads *this* flag differently depending on
  what was asked. This is the term that means attribute-selective image reading,
  and it is the one the experiment exists to measure.

Every one of these numbers is printed against the same decomposition over **all**
captured heads, so "item=0.9" can be read against what an arbitrary head does
rather than in a vacuum. Heads whose block falls outside the captured window are
listed as skipped rather than silently dropped.

## Tests

`python -m pytest tests/test_attr_head_trace.py tests/test_attr_directions.py`

The head-trace tests run a real (randomly initialized) tiny Qwen2.5-VL and check
the question patch is a no-op when self-applied, that it reaches the read column
at all, and that the phase-1c read-back identity holds under full downstream
coverage — the check phases 2-4 depend on. The analysis tests run against grids
with a **planted** answer (known attribute direction, known variance split, a
known single head carrying the signal), which is the only way to catch a
decomposition that is self-consistent but wrong.

## 4. `methods/attr_steer.py` — does a *direction* substitute for a swap?

The three scripts above ask *where* the attribute selection lives. This one asks
whether what lives there is **one vector**. Estimate a single direction per
ordered attribute pair on a set of TRAIN items and apply it to **disjoint** TEST
items:

```
d(a -> a') = mean over TRAIN items of ( resid[item, a'] - resid[item, a] )
```

```bash
python methods/attr_steer.py --dry_run --layers 16 20 22 24 26
python methods/attr_steer.py --layers 14 16 18 20 21 22 23 24 26 28 \
    --mode both --alpha 0.5 1.0 2.0 --n_train_items 32 --n_test_items 16 --n_random 2
python methods/attr_steer.py --layers 20 21 22 24 --mode project --alpha 1.0 \
    --image_trace logs/.../head_trace_patch21.json --top_k 16 --attention_batch_size 2
```

**Held-out items are enforced, not optional.** Including a test item in its own
mean folds a fraction of that item's activation into the vector applied to it,
and the arm silently degrades into a weak full swap. The split is the experiment.

**Two modes, because they test different claims.**

```
add       x' = x + alpha*d                          leaves the base component in place
project   x' = x - (x.u)u + alpha*t*u               DISCARDS it and installs the donor's
```

`project` cannot overshoot, so it is the arm that tests whether the axis *carries*
the attribute rather than whether shoving along it changes the answer. `add`
working while `project` does not would mean the model reads magnitude along the
axis rather than position.

**`--n_random` is not optional either.** Pushing a residual stream along any large
vector degrades the answer. A matched-norm random direction must sit at the
unsteered floor, or the layer is merely fragile and its attribute arm proves
nothing. This matters more with depth: the perturbation is 19% of the residual
norm at block 16 and 80% at block 27.

Phase 3 (`--image_trace`) reads the attention from the readout position onto the
image columns for an image trace's top-k heads, clean vs steered — the question
a logit-level result cannot answer. It forces eager attention; sdpa returns
`None` for attention weights, which would read as "the heads stopped attending".

---

# Results (flags, Qwen2.5-VL-7B-Instruct)

All numbers below are real-model runs. Raw artifacts:
`logs/attr_head_trace/flags/`, `logs/attr_steer/flags/`,
`results/attr_capture/flags/blocks15-27_n84/`, `results/attr_capture/flags/directions_n84.json`.

## R1. The attribute selection is routed by ~8 heads in blocks 19-21

`attr_head_trace.py --patch_layer 15 --blocks 15..22`, 84 items x 12 directed
pairs = 1008 rows. Baselines: base and donor questions both answered 95.5%
first / 92.9% full.

Per-block total `|delta_resid|`: 15->19.8, 16->38.1, 17->46.3, 18->50.2,
19->75.4, 20->101.3, **21->133.5**, 22->57.1. Top heads `21.0` (19.74),
`20.17` (16.36), `20.18` (14.64), `19.21` (13.21), `21.25`, `21.4`, `21.5`,
`19.27`. The top-8 sit in blocks 19x2, 20x2, 21x4.

| arm | k=4 | k=8 | k=16 | random-8 |
|---|---|---|---|---|
| phase 2 necessity (`first`, from 95.4%) | 31.9% | 3.8% | 0.9% | 95.3% |
| phase 3 sufficiency (`first`, ceiling 95.4%) | 59.3% | 87.0% | 94.2% | 0.4% |
| phase 4 shuffled donor (`own` / `donor`) | 0.4 / 62.8 | 0.5 / 90.1 | 0.5 / 94.7 | — |

**Necessity and sufficiency agree, which is rare.** Restoring 8 of 224 heads
destroys 96% of the effect while 8 random heads destroy nothing (+91.6pp gap);
installing the same 8 reaches 91% of the all-heads ceiling. Phase 4 confirms at
k=4: the answer follows **whichever row supplied the values**, so phase 3 is
measuring transfer, not perturbation.

Two readings that matter downstream:

- **`delta_dla` is ~0 and mostly negative** (-0.08 to +0.02) for every top head.
  These heads contribute almost nothing *directly* to the answer logit. They are
  **routers, not writers** — the answer is written later.
- **The per-attribute top-8 lists are nearly identical.** `21.0` and `20.17` are
  top-2 for all four attributes. **There are no per-attribute heads.** One shared
  circuit routes "which attribute was asked"; the attribute identity rides
  through it as a value. That is what makes R4 possible.

**Caveat.** Phase 1b's block-everything arm reads 18.2%, not the 0.4% clean
floor, and phase 1c's identity diverges 8.3%. Both are explained by `--blocks`
stopping at 22 while blocks 23-27 still read the clean question — so "these 8
heads are the mechanism" holds only up to that coverage gap.

## R2. The correlational probe cannot localize anything

`attr_capture` n=84 (fully crossed 84 items x 4 attributes, `last_token`,
blocks 15-27, 4 sites) analysed by `attr_directions`.

**Leave-one-item-out accuracy is 1.000 in all 52 cells** (chance 0.25), and
`attr_var_explained` is *highest at block 15* (97-98%), declining to 66% by
block 26 — i.e. strongest exactly where causal effect is zero.

| block | residual | attn_output | mlp_output | attn_head_output |
|---|---|---|---|---|
| 15 | 97.5 | 98.0 | 96.8 | 98.4 |
| 19 | 96.5 | 96.2 | 94.2 | 95.9 |
| 21 | 96.6 | 92.7 | 96.2 | 92.6 |
| 22 | 96.0 | **51.2** | 94.7 | **58.0** |
| 23 | 92.3 | **41.0** | 86.9 | **43.9** |
| 26 | 65.8 | 65.0 | 47.0 | 59.1 |

The one thing the table does say: the two attention sites **collapse at block 22**
while the two residual-width sites hold. Attention stops carrying
attribute-discriminative content at exactly the block where R1's per-block mass
halves — two methods, one boundary.

Per head, the same inversion. Correlational top-16 (`20.18, 19.21, 20.17, 16.20,
21.0, 15.27, 18.21, 16.6, 15.17, 16.15, ...`) vs R1's causal ranking over the 224
shared heads: **rank correlation +0.220, top-16 overlap 8/16**. Seven of sixteen
are blocks 15-18 heads the causal trace scores near zero. The probe finds the
causal heads *and* a large set of decoys, with no way to tell them apart from
inside its own framework. This is RESULTS.md's proxy-vs-real gap reproduced on a
single shared grid.

## R3. Four separate directions, not one shared axis

`residual` at block 16, after item-demeaning:

```
                capital  currency  language  calling_code
capital           1.000    -0.448    -0.348      -0.197
currency         -0.448     1.000    -0.287      -0.131
language         -0.348    -0.287     1.000      -0.565
calling_code     -0.197    -0.131    -0.565       1.000

norms 10.2 / 9.2 / 11.7 / 9.8      singular values 13.89, 11.79, 9.56, 0.00
effective rank 2.93 (ceiling 3)
```

**Read the cosines against -1/(A-1) = -0.333, not 0** — centering forces the
centroids to sum to zero. Against that baseline `language`<->`calling_code`
(-0.565) and `capital`<->`currency` (-0.448) are *more* opposed than chance,
while `currency`<->`calling_code` (-0.131) is *less* opposed. Effective rank
2.93 of a ceiling of 3 means the centroids fill the space available: each
attribute has its own direction. The fourth singular value is exactly 0.00, as
the centering constraint requires.

## R4. One direction, estimated on 32 countries, flips 16 unseen ones at 96%

`attr_steer.py`, directions from 32 TRAIN items applied to 16 disjoint TEST
items (192 directed rows). `donor_first`; unsteered floor 0.0%, base_kept 90.6%.

| layer | add 0.5 | add 1.0 | add 2.0 | proj 0.5 | proj 1.0 | proj 2.0 | RANDOM |
|---|---|---|---|---|---|---|---|
| L14 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| L16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| L18 | 0.0 | 0.0 | 38.5 | 0.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| L20 | 14.1 | 68.2 | 88.0 | 42.7 | 67.2 | 90.1 | 0.0 / 0.0 |
| L21 | 34.9 | **94.8** | 91.1 | 67.7 | 92.2 | 92.7 | 0.0 / 0.0 |
| L22 | 39.1 | **95.8** | 89.6 | **91.7** | 94.8 | 93.2 | 0.0 / 0.0 |
| L23 | 33.9 | 95.3 | 85.4 | 90.6 | 95.3 | 92.2 | 0.0 / 0.0 |
| L24 | 29.7 | 91.7 | 79.7 | 84.4 | 92.2 | 89.6 | 0.0 / 0.5 |
| L26 | 14.1 | 78.6 | 67.7 | 62.5 | 77.6 | 79.2 | 0.5 / 1.6 |
| L28 | 13.0 | 44.3 | 30.2 | 28.6 | 41.1 | 45.3 | 0.0 / 0.5 |

The sweep's *full residual swap* at these blocks reads 100%. **One fixed
direction recovers ~96% of it on countries it has never seen**, so the attribute
selection is essentially entity-independent and one-dimensional per attribute.

The layer profile — dead through L16, onset L20, plateau L21-23, decay from L24
— is the third independent reproduction of the sweep's blocks 17-21 window
(after span knockouts and R1's per-head trace).

**Why it is not a magnitude effect**, three ways:

1. Matched-norm **random directions read 0.0-1.6%** at every layer.
2. `||d||/||resid||` grows monotonically **0.19 (b16) -> 0.64 (b21) -> 0.80
   (b27)** while efficacy *peaks* at b21-22 and falls to 44% at b27. The largest
   relative perturbation has the weakest effect.
3. `base_kept` -> 0.0% wherever `first` is high: the model moves to a *specific*
   other answer, not to noise. L20 alpha=0.5 (first 14.1%, base_kept 73.4%) is a
   clean dose-response.

**`project` beating `add` at low alpha is the decisive control.** At L22
alpha=0.5: project **91.7%** vs add 39.1%. Half-strength *replacement* works
where additive pushing does not, so the axis carries the attribute. `project`
is also stable at alpha=2.0 where `add` degrades, exactly as a bounded operation
should against an unbounded one.

`donor_full ~ 0.55-0.60 x donor_first` throughout — the documented `last_token`
multi-token cap, not a second finding.

## R5. The image heads are attribute-*selective*, not image-*rich*

> **RETRACTED — the head set came from a VOID trace.** The source file
> `.../mlp_hidden_pruned/head_trace_patch21.json` (blocks 21-27, n_rows=128) is dated
> **2026-09-13 23:31**, 79 minutes BEFORE commit `518cf31` (2026-09-14 00:50) fixed
> `BuildBatchCache` returning another spec's positions. It carries the exact signature
> CLAUDE.md records for pre-fix runs: a phase-3 ceiling of **1.6%** with ALL 224 heads
> patched (against a 98.4% image patch), flat at every k, and a phase-1 ranking whose
> mass sits in blocks 26-27. The 16 heads decomposed below were selected by a ranking
> that was tracing image columns while reporting last-token. **The decomposition
> arithmetic is sound; the head set it was applied to is not.** See R8 for the valid
> head sets. Only 2 of the 55 head traces on disk are affected (both n_rows=128); all
> 53 post-fix traces are valid.

Top-16 heads of the image trace (`head_trace_patch21.json`, attribute=language,
blocks 21-27, n_rows=128), decomposed over the crossed grid:

```
block.head   item   attribute  interaction   scale
27.3        0.726     0.063       0.211      14.30   <- the only true image head
27.0        0.551     0.190       0.259       8.09
26.24       0.379     0.344       0.277      22.92
27.20       0.388     0.385       0.227       8.44
26.15       0.368     0.418       0.214       8.04
26.25       0.273     0.357       0.370      13.21   <- most attribute-selective
27.13/11/12/10 0.194  0.628       0.178      ~19.6   <- near-duplicates, see below
27.2        0.130     0.739       0.131      15.54
21.6        0.145     0.821       0.034       8.56   <- nearly pure question

mean over these 16:   item=0.298  attribute=0.508  interaction=0.193
ALL 364 heads:        item=0.296  attribute=0.595  interaction=0.109
```

**They carry no more image content than an arbitrary head** (item 0.298 vs
0.296 — indistinguishable). Half their variance is a question-dependent
component identical across flags, which is not image data at all. What
distinguishes them is **interaction: 0.193 vs 0.109, ~1.8x** — reading *this*
flag differently depending on what was asked. Attribute-selective image reading
is the whole of what makes these heads special.

**Verified duplication.** Heads 27.10-27.13 have identical decompositions to
three decimals because they are near-duplicates: centered-activation cosines
**0.9992-0.9998** among 27.11/12/13 and 0.918-0.923 against 27.10. The top-16
list is ~13 distinct signals with one group in four slots; **de-duplicate before
quoting any mean over it.**

## R6. Steering changes the answer without changing where the model looks

> **PARTIALLY RETRACTED.** `attr_steer --image_trace` was pointed at the same void file
> as R5, so the 16 heads whose attention was measured are the wrong heads. The attention
> numbers are real measurements of those heads, and the causality check below still
> validates the instrumentation — but nothing here licenses a claim about "the image
> heads". Re-run with an R8 head set.

`attr_steer` phase 3, `project` alpha=1.0, steering at L21/L22/L24, reading
attention onto image columns at the readout position.

A causality check falls out of the table and passes: steering at L24 leaves
every head at or below block 23 (`23.20`, `23.8`, `21.6`) at cosine **exactly
1.0000** and L1 shift **exactly 0.0000**; steering at L22 pins only `21.6`;
steering at L21 moves `21.6` too. Correct causal ordering in all three arms —
nobody asserted this, it is a consequence, and it validates the head indexing.

Steering at L22:

```
block.head   mass clean   mass steer   cosine   L1 shift
27.12           0.3335       0.0442    0.1670    0.3072   <- collapses
27.3            0.3833       0.4240    0.9009    0.3030
27.2            0.1668       0.1756    0.5837    0.1770
26.13           0.0759       0.0812    0.8521    0.0479
21.6            0.0150       0.0150    1.0000    0.0000
```

For 12 of 13 distinct heads the image mass barely moves and cosines sit
0.49-0.91: **the model looks at the same pixels and reports something else**.
Selection happens downstream of the image read. The exception is real and sharp
— `27.12` collapses (mass 0.3335 -> 0.04-0.09, cosine 0.17-0.34 in all three
arms), so one head's image read is *gated* by the attribute direction.

## R7. Known-void arm: `--positions question`

`attr_steer --positions question --layers 8..16` reads 0.0% everywhere. **This is
a bug, not a negative result.** `attr_steer.py:110` hardcodes
`pos = batch_ids.shape[1] - 1`; the `read_last` parameter at line 85 is declared
and never used, and the call site passes `True` unconditionally. So the run
estimated directions at the **last token** and applied them at **question**
columns. The logs confirm it: the L14/L16 norms are bit-identical to the
`last_token` run (4.939 / 3.469 / 5.459 / 4.430). Fixing it needs a pooling rule
— question positions are 67 columns, not one.

## The account these add up to

1. **Blocks 15-18** — the attribute is maximally *decodable* (R2) and causally
   inert (R1 low mass, R4 0% steering). Present and unused.
2. **Blocks 19-21** — ~8 heads, the *same* 8 for all four attributes (R1), route
   the selection into the readout residual. What they route is one direction per
   attribute (R3), entity-independent enough to transfer between disjoint country
   sets at 96% (R4).
3. **Block 22 on** — attention stops carrying attribute-discriminative content
   (R2 collapse, R1 mass halving). The image heads report an already-selected
   attribute from an unchanged image read (R5, R6).
4. **`delta_dla` ~ 0 for every routing head** (R1): the answer is *written* later
   still, by the late MLPs the sweep flagged from block 21 on.

## R8. The entity-reading heads (image side): blocks 21-23, shared across attributes

**Experiment.** `methods/head_trace.py --patch_layer 21 --positions full_image`,
n_rows=64, `rank_by=delta_resid`, run independently for **all four flags
attributes** over a systematic sweep of block windows (every single block 21-27
and every 2-3 block span). 53 valid traces. Files:

```
logs/Qwen2.5-VL-7B-Instruct/flags/ndm/<attribute>/
  L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/
    head_trace_patch21_blocks21-23.json          <- the winning window
```

**Blocks 21-23 is the window, for every attribute.** Phase-3 sufficiency — patch
ONLY these heads into a clean run, no image patch — across windows:

| window | language | capital | currency | calling_code |
|---|---|---|---|---|
| [21] / [22] / [23] alone | 0.0 / 0.0 / 6.2 | 0.0 / 0.0 / 9.4 | 0.0 / 1.6 / 4.7 | 0.0 / 0.0 / 1.6 |
| [21,22] | 28.1 | 17.2 | 9.4 | 1.6 |
| [22,23] | 87.5 | 50.0 | 53.1 | 6.2 |
| **[21,22,23]** | **98.4** | **76.6** | **67.2** | 6.2 |
| [22,23,24] | 89.1 | 54.7 | 75.0 | 6.2 |
| [23,24,25] and beyond | <=9.4 | <=14.1 | <=12.5 | <=1.6 |

Single blocks are dead; the span is the unit. Nothing from block 24 on carries it.

Within the [21,22,23] window:

| attribute | image patch alone | ceiling (84 heads) | k=4 | k=8 | k=16 | phase 4 own/donor |
|---|---|---|---|---|---|---|
| language | 98.4% | 98.4% | 60.9% | **100.0%** | 98.4% | 15.6 / 98.4 |
| capital | 96.9% | 76.6% | 35.9% | 70.3% | 76.6% | 6.3 / 79.7 |
| currency | 95.3% | 67.2% | 20.3% | 60.9% | 68.8% | 0.0 / 70.3 |
| calling_code | 92.2% | **6.2%** | 1.6% | 6.2% | 6.2% | 3.1 / 14.1 |

**`calling_code`'s 6.2% is an ANSWER-LENGTH artefact, not a dead window — and it
makes the whole column non-comparable.** `head_trace.py` scores with
`exact_match` over the FULL answer (`common/targets.py:82`) while the head patch
fires only at the last prompt column (`make_cache_aware_patch_hook` is a no-op
once decoding collapses to one column, `common/hooks.py:44`). Measured gold
lengths over all 6 templates x 84 countries:

| attribute | 1-token golds | mean len | k=8 sufficiency |
|---|---|---|---|
| language | 78.6% | 1.37 | 100.0% |
| capital | 36.9% | 2.01 | 70.3% |
| currency | 23.8% | 1.82 | 60.9% |
| calling_code | **0.0%** | 2.68 | 6.2% |

The ranking is monotone in single-token fraction, 4 of 4, and `calling_code` is
the only attribute with *zero* single-token golds. `swap_trace.py` already
recorded the same illusion at a different site: calling_code reads 3.4% full but
**99.7% first-token**. **Do not read this column as "which attributes this window
carries"** until `head_trace` reports a first-token score the way
`attr_head_trace.score_switch` does (`gold_len` clamped to 1) — it currently has
no such metric.

### The head sets

Per-attribute top-8 by `delta_resid`:

```
language      23.4  21.1  23.3  22.19 23.6  21.5  22.17 22.15
capital       23.4  23.3  21.1  23.6  22.17 23.17 22.19 22.13
currency      23.4  23.3  21.1  23.6  22.19 22.17 23.11 22.15
calling_code  23.17 23.4  23.3  21.1  23.7  21.5  22.19 23.6
```

**Common heads — in all four attributes' top-8 (5):**

```
21.1   22.19   23.3   23.4   23.6
```

**Common heads — in all four attributes' top-16 (10):**

```
21.1   21.5   22.13   22.15   22.17   22.19   23.3   23.4   23.6   23.17
```

(`22.0` and `23.11` appear in 3/4.)

**The ranking is near-identical across attributes** — `23.4`, `23.3`, `21.1`,
`23.6` are top-4 for every one. This mirrors R1's result on the question side:
**there are no per-attribute heads on the image side either.** One shared circuit
reads the entity out of the image; which attribute gets extracted is decided
elsewhere (R1/R4, blocks 19-21). That is the cleanest statement of the
disentanglement the benchmark is about, and it is the head set any VADE-scored
head-swap experiment should use.

## R9. The entity heads ARE image-rich (supersedes the retracted R5)

Same decomposition as R5, run against the **valid** blocks 21-23 head sets of R8,
one run per attribute (`attr_directions --image_trace <attr>/head_trace_patch21_blocks21-23.json`).
No model; outputs at `results/attr_capture/flags/image_heads_21-23_<attr>.json`.

| head set | item | attribute | interaction |
|---|---|---|---|
| language (21-23) | **0.667** | 0.167 | 0.166 |
| capital (21-23) | **0.609** | 0.218 | 0.173 |
| currency (21-23) | **0.689** | 0.133 | 0.178 |
| calling_code (21-23) | **0.645** | 0.199 | 0.156 |
| ALL 364 captured heads | 0.296 | 0.595 | 0.109 |
| *(retracted R5, void head set)* | *0.298* | *0.508* | *0.193* |

**`item` is 0.61-0.69 against a 0.296 all-head baseline — more than double.**
The real entity heads carry flag identity, and they carry it far more than an
arbitrary head does. The five heads common to all four attributes are close to
pure conduits:

```
block.head   item   attribute  interaction
23.4        0.971     0.006       0.023      <- 97% "which flag", question-independent
21.1        0.931     0.022       0.047
23.3        0.920     0.012       0.068
22.17       0.874     0.016       0.110
22.19       0.862     0.039       0.099
23.6        0.720     0.158       0.122
```

`interaction` is 0.156-0.178 against a 0.109 baseline — mildly enriched, but an
order of magnitude less than `item`. **These heads are an entity conduit, not an
attribute-selective reader.** R5's opposite conclusion was an artefact of the
void ranking, which had selected heads in blocks 26-27 that carry mostly question
identity.

**Consequence for a VADE-scored head swap.** An `item`-dominated conduit
transplants the whole country, so every attribute moves together: expect **cause
high, iso near 0, final_score ~50%** — the selectivity null. VADE pins any
attribute-agnostic edit at 50% (cause and iso trade off exactly), and beating it
requires the edit to flip the queried attribute more than the un-queried ones.
On these numbers this head set should NOT beat it. That is a clean negative
result about where attribute disentanglement can be done, not a failed run —
but the run must use continuous patching or an image-position patch, because
VADE's scorer matches whole labels and a last-token patch cannot produce them
(see R8).

## 5. `methods/head_swap_vade.py` — the entity-head swap, scored by VADE

R8's heads, installed into the base run, generated freely, and written in
`VADE/eval/score.py`'s format so the result lands on `cause` / `iso` /
`final_score` rather than an internal flip rate.

```bash
python methods/head_swap_vade.py --dry_run                       # sizing, no torch
python methods/head_swap_vade.py --limit_pairs 20 --out_dir results/head_swap_vade/smoke
python methods/head_swap_vade.py --batch_size 16                 # full test split
python methods/head_swap_vade.py --donor_question target --arms clean heads
python ../VADE/eval/score.py --entity flags --attribute all \
    --predictions results/head_swap_vade/flags/heads.jsonl
```

**Continuous patching is required, not a refinement.** `head_trace` patches only
the last PROMPT column — `make_cache_aware_patch_hook` returns the tensor
untouched once generation collapses to one column — so the intervention reaches
the first answer token and no further. VADE's scorer matches a whole label
anywhere in the generated text, so a first-token-only flip produces NEITHER
label: it scores as a miss on the cause row AND a false success on the iso rows.
This script therefore records the donor's head outputs at **every generated
step** and replays them step-by-step.

The hook addresses **column -1** in both passes. Batches are left-padded, so the
last real prompt token is the final column, and a decode step has exactly one
column — one rule covering prefill and decode, which is what keeps step
alignment checkable.

**Cost.** Under `--donor_question queried` the edit does not depend on
`target_attribute`, so one generation answers one row in each of the four target
files: the 56,208-row test split costs **14,052 generations**. `target` freezes
the donor at the target attribute's question and does not collapse (4x).

**The 50% null — read before interpreting any number.** VADE builds one
intervention per `target_attribute` then asks all four questions under it. An
edit with no attribute selectivity is pinned near `final_score = 50%`: it either
transplants the country (cause ~100%, iso ~0%) or does nothing (cause 0%, iso
~100%), and both average to 50%. Beating 50% requires moving the queried
attribute MORE than the un-queried ones. **R9 predicts this head set will not**,
because it is `item`-dominated. A ~50% result is the expected finding about where
disentanglement can be done, not a failed run; the informative quantities are
which corner it lands in and the distance from 50%.

| arm | what it is | expected |
|---|---|---|
| `clean` | no patch | cause ~0, iso ~100 |
| `heads` | the experiment | ? |
| `random_heads` | as many random heads, same blocks, drawn once | ≈ clean |
| `full_image` | source residual at every image token at `--patch_layer` | cause high, iso ~0 |

Both 50% corners are measured, so any deviation is readable against them.

**Two guards.** `heads_from_trace` **refuses** a trace whose phase-3 ceiling is
near zero against a live image patch — the signature of the two pre-`518cf31`
runs whose rankings are void (R5). And the run opens with a **read-back check**:
one generation with an observer registered AFTER the patch, asserting the site
returns what was installed at every step. Observers register last for the reason
`head_trace` documents — a hook added before the patch reads the tensor the patch
is about to overwrite.

`tests/test_head_swap_vade.py` (10 tests) covers the pair that has to hold
together: replaying a run's own capture into itself is bit-identical, while a
different donor changes the answer. Neither alone is sufficient — a completely
dead patch passes the first.
