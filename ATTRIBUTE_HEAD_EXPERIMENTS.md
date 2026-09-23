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

**Several head sets in one run.** `--head_sets NAME=SPEC ...` scores each set as
its own arm and its own predictions file. SPEC is a `BLOCK.HEAD` list, or a
`head_trace` JSON path with an optional `#k` for its top-k; a path containing
`{attribute}` expands per attribute. Every set shares ONE donor capture because
they all live in blocks 21-23, so k=8 and k=16 cost extra generations only:

```bash
T='logs/Qwen2.5-VL-7B-Instruct/{entity}/ndm/{attribute}/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json'
python methods/head_swap_vade.py --batch_size 16 \
    --head_sets "common5=21.1,22.19,23.3,23.4,23.6" "top8=$T#8" "top16=$T#16"
```

Each set gets its OWN size-matched random null -- 16 random heads is a bigger
perturbation than 5, so a single shared null would under-control the large sets.
Batches never mix donor attributes, because a per-attribute set installs a
different mask per attribute and the hook applies one mask per batch.

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

**Two guards.** `heads_from_trace` **refuses** a trace whose ranking is inert —
judged by whether patching EVERY traced head DISLODGED the base answer, not by
whether it transferred. Those are different failures and only the first indicts
the ranking: the void trace reads cause 1.6% / base_kept **95.3%** (the patch did
nothing), while `calling_code`'s perfectly valid trace reads cause 6.2% /
base_kept **26.6%** (the patch destroyed the answer without transferring — the
answer-length artefact). A guard on `cause` alone rejects the good trace. And the
run opens with a **read-back check**:
one generation with an observer registered AFTER the patch, asserting the site
returns what was installed at every step. Observers register last for the reason
`head_trace` documents — a hook added before the patch reads the tensor the patch
is about to overwrite.

`tests/test_head_swap_vade.py` (10 tests) covers the pair that has to hold
together: replaying a run's own capture into itself is bit-identical, while a
different donor changes the answer. Neither alone is sufficient — a completely
dead patch passes the first.

## R10. On the benchmark: 10 heads ARE the entity, and that is worth 50%

> **RUN STATUS — 39,552 of 56,208 scored rows (70.4%).** All four *target*
> attributes are present; the gap is one whole *queried* column: **`language` was
> never asked**. Consequences: `target=language` has no cause row and no
> `final_score`, and every other target's `iso_mean` averages two attributes
> rather than three. Numbers below are re-scored from the committed JSONLs with
> `VADE/eval/score.py`. The committed `results/head_swap_vade/flags/partial_scores/`
> summaries are STALE (they describe the earlier 20,288-row state) and should be
> regenerated before being quoted.

`methods/head_swap_vade.py`, `--donor_question queried`, continuous patching.
`cause` is `matches_source` on the cause pool; `iso keep` is `pct_accuracy` (=
stayed at base) averaged over the iso pool; `final` is `score.py`'s own
`final_score`. Columns cc / cap / cur = calling_code / capital / currency.

| arm | heads | cause cc | cause cap | cause cur | iso keep (cc/cap/cur) | final cc | final cap | final cur |
|---|---|---|---|---|---|---|---|---|
| clean | — | 0.0 | 0.0 | 0.0 | 91.3 / 91.3 / 100.0 | 45.6 | 45.6 | 50.0 |
| random_heads_common5 | 5 rnd | 0.0 | 0.0 | 0.0 | 91.3 / 90.6 / 99.3 | 45.6 | 45.3 | 49.6 |
| random_heads_top8 | 8 rnd | 0.0 | 0.0 | 0.0 | 91.4 / 91.4 / 100.0 | 45.7 | 45.7 | 50.0 |
| random_heads_common10 | 10 rnd | 0.0 | 0.0 | 0.0 | 91.5 / 91.1 / 99.6 | 45.8 | 45.5 | 49.8 |
| heads_common5 | 5 | 69.4 | 92.0 | 49.1 | 9.2 / 10.7 / 3.8 | 39.3 | 51.3 | 26.5 |
| heads_top8 | 8/attr | 87.4 | 96.5 | 70.7 | 2.9 / 3.3 / 0.8 | 45.1 | 49.9 | 35.7 |
| **heads_common10** | **10** | **95.5** | **98.0** | **72.1** | 2.1 / 2.2 / 0.2 | 48.8 | 50.1 | 36.1 |
| full_image | all image | 97.1 | 97.3 | 84.4 | 0.2 / 0.1 / 0.1 | 48.6 | 48.7 | 42.2 |

Mean cause over the three measurable attributes: `heads_common10` **88.5%** vs
`full_image` **93.0%**, `heads_top8` 84.9%, `heads_common5` 70.2%, every random
arm **0.0%**.

> `clean` does not read 100% iso keep because the model is only **82.6% correct
> on `currency` unintervened**. Every `currency` number in this table — cause and
> iso alike — is against that degraded ceiling, not against 100.

### R10.1 `calling_code`'s 6.2% was the metric, exactly as suspected

`head_trace` put `calling_code`'s blocks 21-23 sufficiency ceiling at **6.2%**.
The same heads under continuous patching reach **95.5%**, against a `full_image`
reference of 97.1%. R8's caveat is now settled empirically: that column was
measuring **answer length**, not which attributes the window carries. The window
carries all of them.

### R10.2 Ten heads out of 784 reproduce a whole-image swap

Cause rises 69.4 -> 87.4 -> 95.5 (cc), 92.0 -> 96.5 -> 98.0 (cap) and
49.1 -> 70.7 -> 72.1 (cur) as the set grows 5 -> 8 -> 10, and `heads_common10`
**matches `full_image`** — slightly exceeding it on `capital` (98.0 vs 97.3),
and reaching 95% of it in mean cause (88.5 vs 93.0). The conduit is ~10 heads wide, and the
10 attribute-INDEPENDENT heads beat the 8 per-attribute ones, consistent with R8's
finding that the per-attribute rankings are near-identical.

The size-matched random nulls read **0.0% cause at every size**, with base kept
98.6-100.0%. The effect is about WHICH heads, not about perturbing the site.

### R10.3 iso does not degrade — it LEAKS

The `iso -> SOURCE` column is the one to read. Target `calling_code`, then ask
for the capital, and the model returns the SOURCE country's capital **98.0%** of
the time. The base answer is not destroyed or confused; it is replaced by the
donor country's. R9 predicted exactly this from `item` = 0.61-0.69, and this is
its causal confirmation.

### R10.4 Why the scores sit at 50%, and why the best arm is doing nothing

Under `--donor_question queried` the edit is attribute-agnostic, so ONE generation
serves the cause row of one target file and an iso row of another — the identical
text, scored against opposite rules. The full grid now shows this as an exact
identity rather than an approximation: **for a fixed queried attribute,
`matches_source` is the same number in all four target files**, whether that row
is scored as a cause or as an iso. For `heads_common10`:

| queried | target=cc | target=cap | target=cur | target=lang |
|---|---|---|---|---|
| calling_code | **95.5** (cause) | 95.5 (iso) | 95.5 (iso) | 95.5 (iso) |
| capital | 98.0 (iso) | **98.0** (cause) | 98.0 (iso) | 98.0 (iso) |
| currency | 72.1 (iso) | 72.1 (iso) | **72.1** (cause) | 72.1 (iso) |

One generation per (queried, row); the target column only changes the scoring
rule applied to it. Hence

```
final_score = 1/2( cause_T + mean_{q != T} keep_q ),   keep_q ~= 1 - cause_q
```

and if every attribute transferred equally the result would be exactly 50%. The
deviations below 50 are the DESTROYED fraction (rows matching neither label,
~4-5% at k=10, ~25% for `calling_code` at k=5) plus the asymmetry between
attributes' transfer rates. A better conduit does not score better — it converges
on 50 from below.

**The highest score in the table is `clean`.** On VADE's own metric these heads
are the worst available intervention site: maximal `cause`, zero `iso`. That is
the finding, not a failed run.

### R10.5 A real difficulty gradient between attributes

`capital` transfers more readily than `calling_code` at small k (92.0 vs 69.4 at
k=5), and the gap closes by k=10 (98.0 vs 95.5). This is NOT the old last-token
artefact — that is gone. The plausible cause is answer length still costing
something: `calling_code` averages 2.68 tokens to `capital`'s 2.01, so the donor
must stay correct for more consecutive steps.

`currency` is the hard one at every k (49.1 / 70.7 / 72.1) and the only attribute
where `heads_common10` falls clearly short of `full_image` (72.1 vs 84.4) — but
its unintervened ceiling is 82.6%, so the 10 heads recover 87% of what the whole
image recovers, in line with the other two.

### R10.6 What the remaining 30% changes

The whole `language` **queried** column is missing, so `target=language` has no
cause row and the other three targets each average iso over two attributes rather
than three. `language` had the 100% `head_trace` sufficiency and is the attribute
where the last-token measurement was ALREADY accurate (78.6% single-token golds),
so it is the clean control that continuous patching did not change something it
should not have. It is also the only attribute with a predominantly single-token
gold, which makes it the least contaminated cause measurement available — worth
finishing for that reason alone. Adding it moves `iso_mean` by at most a third of
whatever `language`'s own keep rate is; with iso at 0.1-4.2% throughout, little
movement is expected.

Do not quote `overall accuracy` from the summaries — it pools cause and iso rows
into one number, which is why `score.py` reports `final_score` separately.

### R10.7 What `--donor_question target` would do, and why it is the only arm that can move

R10.4's identity is the whole reason this run cannot exceed 50%, and it comes
from ONE design choice: under `queried`, the donor is captured asking the SAME
question the base is asked, so the edit never learns which attribute the VADE row
is targeting. Take the pair

```
row A   target=calling_code  queried=calling_code   (cause: must flip to source)
row B   target=capital       queried=calling_code   (iso:   must stay at base)
```

Under `queried` both rows are answered by the *same generation* — source image
asked "what is the calling code", heads installed, base asked the same. One text,
scored as a required flip in one file and a required non-flip in the other. It
cannot satisfy both. That is the 50% pin, and it is arithmetic, not a property of
the model.

**`target` breaks the identity by changing what the donor was asked.** For an iso
row it captures the source's head outputs under the TARGET attribute's question
and freezes that one vector across all four queries:

```
row B under --donor_question target
  donor   SOURCE image + "what is the capital"        <- the target attribute
  base    BASE image   + "what is the calling code"   <- the queried attribute
          with the donor's frozen head outputs installed
```

So the question becomes: **is what these heads write conditioned on the question
that produced it?**

* If they carry pure entity identity (`item` ~ 1), the installed vector says
  "this is Argentina" no matter which question produced it. The calling code
  leaks anyway, iso stays ~0, and the score stays at 50%.
* If they carry a question-conditioned read (the `interaction` term), the vector
  is "Argentina-as-read-for-capital", which is a poor answer to a calling-code
  question. The base's calling code survives, iso rises, and the score exceeds 50.

`final_score - 50%` is therefore a direct causal measurement of `interaction`,
the same way R10 is a direct causal measurement of `item`. R9 puts `interaction`
at 0.156-0.178 against `item` at 0.61-0.69, so the expectation is a real but
modest rise — iso somewhere in the tens of percent, not a solved disentanglement.

**A free consistency check comes with it.** On a CAUSE row the target and queried
attributes are equal, so the donor question is the same in both modes and
`donor_template()` resolves to the same template — the generation must be
IDENTICAL to this run's. Cause rates that move between modes mean a bug in the
donor-prompt plumbing, not a finding. Only the iso rows may change.

**Cost:** the edit now depends on `target_attribute`, so the 4x collapse does not
apply — 56,208 generations instead of 14,052.

```bash
python methods/head_swap_vade.py --donor_question target --arms clean heads \
    --head_sets "common10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
```

## 6. `methods/head_cross.py` — the cross-prompt design

R10 establishes that ~10 heads carry *something* that determines all four
answers. It cannot say **what**, because in `head_swap_vade` the donor and the
receiving run are always asked the same question. Two hypotheses survive it:

* **ENTITY** — the heads write "this is Cambodia", and every attribute is
  recomputed downstream from that.
* **ATTRIBUTE** — the heads write "the capital is Phnom Penh", i.e. an answer
  already selected by the question that produced it.

`head_cross.py` separates them by crossing the two things the donor can differ
in. For a flag pair (A = base, B = source) and a question pair (Q1 = what the
receiving run is asked, Q2 = what the donor was asked) it runs both donors:

|  | donor Q = Q1 | donor Q = Q2 |
|---|---|---|
| **donor flag = A (base)** | `self` | `question` |
| **donor flag = B (source)** | `flag` | `both` |

Each generation is classified against four golds — `base_q1` (A,Q1),
`source_q1` (B,Q1), `source_q2` (B,Q2), `base_q2` (A,Q2) — plus `other` and
`ambiguous`. It imports the machinery from `head_swap_vade` (capture, patch,
read-back, batching) rather than forking it, and loads VADE's own
`contains_label` for matching so the labels agree with `eval/score.py`.

**Only `both` has four distinct golds.** This is the key structural fact and the
reason the other three cells are controls, not results:

| cell | A vs B | Q1 vs Q2 | distinct golds | what it can possibly distinguish |
|---|---|---|---|---|
| `self` | same | same | 1 | only "did anything change at all" |
| `flag` | differ | same | 2 | entity transfer; blind to the question axis |
| `question` | same | differ | 2 | question transfer; blind to the entity axis |
| `both` | differ | differ | **4** | all four hypotheses simultaneously |

`classify()` collapses golds that resolve to the same string before calling
anything ambiguous, so a `source_q1` of 0.0% in the `self`/`question` cells is an
*identity*, not a measurement — in those cells `donor_flag == base`, so
`source_q1` and `base_q1` are literally the same gold. Do not read those two
zeros as evidence of anything.

**A second, subtler unreachability — and the design's real limit.** The "4
distinct golds" in the `both` row above is a count of distinct *strings*, not of
*attainable outcomes*. Every VADE template ends in a prefill that fixes the
answer's TYPE (`"The capital city is"`, `"The currency code is"`,
`"The calling code is +"`), and that prefill belongs to the RECEIVING prompt,
which this design never patches. Both Q2-indexed golds (`source_q2`, `base_q2`)
are therefore Q2-typed strings that the model cannot syntactically emit under a
Q1 prefill — measured at 767/768 = 99.9% type-correct-for-Q1 in `both`. So the
`both` cell really discriminates on **two** outcomes, `base_q1` vs `source_q1`,
and its power comes from `source_q1` being HIGH rather than from the Q2 golds
being zero. See R11.2. A design that could reach the Q2 golds would have to
patch the prefill too, which would change what is being asked and is a different
experiment.

```bash
# the full 2x2
python methods/head_cross.py --n_pairs 64 --batch_size 64

# only the cells that carry information (saves the two diagonal cells)
python methods/head_cross.py --n_pairs 64 --batch_size 64 --cells flag question both

# same design, but patch only a k-dim subspace of the head columns
python methods/head_cross.py --n_pairs 16 --batch_size 64 --subspace_dim 7

# break the step correspondence -- see R11.3b. Same --n_pairs/--seed draws the
# identical rows, so these are paired with the matched run row for row.
python methods/head_cross.py --n_pairs 64 --align step0 --cells flag self both
python methods/head_cross.py --n_pairs 64 --align hold0 --cells flag self both
```

Cost is `n_pairs x 4 x 4 x 2` rows and ~3 model passes per row (donor capture,
patched generation, clean reference). At `n_pairs=64` that is 2,048 rows;
`--cells flag self both` drops 768 of them, and R11.4 shows the `question` cell
carries nothing beyond its own inertness.

### `--align`: which donor step lands at which base step

`head_swap_vade.donor_index(align, step, n_steps)` is the whole mechanism, and
the three modes exist to separate a CONTENT effect from a STEP-ALIGNMENT one.
`matched` installs donor step t at base step t; a donor asked a DIFFERENT
question therefore contributes, at t >= 1, its own answer's continuation, and a
multi-token result cannot tell the two apart. `hold0` holds step 0 everywhere;
`step0` patches once and lets the base free-run (the hook returns `None`, which
leaves a pre-hook's input untouched).

The mode is in the OUTPUT FILENAME (`cross_n64_step0.jsonl`) — three arms of the
same `--n_pairs` would otherwise overwrite each other, the mistake R12.3's
`--mask_coef` arms made. Rows carry `align` and `first_token`; `first_token` is
what makes the modes comparable, since they are identical at t=0 by construction
and so must agree there row for row. Check that before reading any score:

```bash
python3 -c "
import json
f={m:{(r['base'],r['donor_flag'],r['q1'],r['q2'],r['cell']):r['first_token']
      for r in map(json.loads,open(p))}
   for m,p in [('step0','results/head_cross/flags/cross_n64_step0.jsonl'),
               ('hold0','results/head_cross/flags/cross_n64_hold0.jsonl')]}
k=set(f['step0'])&set(f['hold0'])
print(sum(f['step0'][x]==f['hold0'][x] for x in k), '/', len(k), 'agree at t=0')"
```

`verify_readback` takes `align` too and checks only the steps that mode actually
installed — `step0` skips t >= 1 rather than failing on them — but it still
catches a dead patch at t=0 (`tests/test_head_swap_vade.py`).

## R11. The heads carry the ENTITY, not the answer

`methods/head_cross.py`, flags, `common10` (21.1, 21.5, 22.13, 22.15, 22.17,
22.19, 23.3, 23.4, 23.6, 23.17), continuous patching, read-back exact at
`0.00e+00`. n = 64 flag pairs x 16 question pairs x 2 donors = **2,048
generations** (`results/head_cross/flags/cross_n64.jsonl`).

### R11.1 The four cells

| cell | donor flag | donor question | n | base_q1 | source_q1 | source_q2 | base_q2 | other |
|---|---|---|---|---|---|---|---|---|
| `self` | base | same | 256 | 93.0 | — | — | — | 7.0 |
| `flag` | **source** | same | 256 | 5.1 | **81.2** | — | — | 13.7 |
| `question` | base | **different** | 768 | 83.2 | — | — | 0.0 | 16.7 |
| `both` | **source** | **different** | 768 | 5.2 | **65.1** | **0.0** | 0.0 | 29.7 |

(— marks a gold that coincides with another in that cell and is therefore
unreachable by construction; `question` also carries 0.1% ambiguous.)

Reading the cells in order:

* `self` is a **no-op**, as it must be — 93.0% unchanged. The 7.0% `other` is the
  noise floor of this pipeline and recurs, to the tenth of a percent, in every
  inert arm below.
* `flag` is **cross-prompt transfer**: capture A's heads, install them into a
  *different prompt about B*, and B's run answers about A, 81.2% of the time.
  R10's result was within one prompt; this shows the payload is portable.
* `question` is the first real finding. Same flag, donor captured while a
  **different** question was being asked: **83.2% unchanged, 0.0% `base_q2`**.
  Swapping which question produced the head outputs changes nothing. The heads
  carry no question identity whatsoever.

### R11.2 `both`: an answer that existed in neither prompt

The receiving run is asked Q1. The donor was asked Q2 about a different flag.
The output is **the donor's flag answered for Q1** — a combination present in
neither prompt, at 65.1%, with **`source_q2` at exactly 0.0%**.

The model did not copy the donor's answer; it *recomputed* an answer from the
donor's entity against the receiving prompt's question.

> **CORRECTION (audit, this revision).** An earlier version of this section
> called `source_q2` = 0.0% "the load-bearing one". **It is not, and it rejects
> nothing.** Every VADE template ends in a *prefill* that fixes the answer's
> TYPE — `"The capital city is"`, `"The currency code is"`, `"The calling code
> is +"` — and the prefill belongs to the RECEIVING prompt, which is never
> patched. A `source_q2`/`base_q2` gold is by construction a Q2-typed string
> (a city when Q1 asked for a currency code, and so on), so the model is
> syntactically unable to emit it whatever the heads contain. Measured on the
> 768 off-diagonal `both` rows: **767/768 = 99.9% of generations are
> type-correct for Q1**, including every failure —
>
> ```
> q1=currency     q2=capital      -> ' USD.'
> q1=calling_code q2=capital      -> '856.'
> q1=language     q2=currency     -> ' German.'
> ```
>
> Two of the design's four golds are therefore unreachable in the cell that was
> supposed to make all four reachable. The conclusion survives intact, but it
> rests on ONE number, not four:

| observed | rejects | still valid? |
|---|---|---|
| `base_q1` 5.2% | "nothing transfers" | **yes** — `base_q1` is Q1-typed and reachable |
| `base_q2` 0.0% | "the donor's question transfers" | **no** — prefill-forced |
| `source_q2` 0.0% | "the donor's whole prompt/answer transfers" | **no** — prefill-forced |
| `source_q1` **65.1%** | leaves only: the donor's **entity** transfers | **yes — this is the whole result** |

The surviving argument is short and still sufficient: the donor was asked **Q2
and only Q2**, so under the ATTRIBUTE hypothesis its head outputs encode "flag
B's answer to Q2" and contain nothing about Q1. For the receiving run to then
emit **flag B's answer to Q1** — which it does 65.1% of the time — the payload
must identify B well enough for the model to compute a Q1 answer it was never
given. That is the ENTITY hypothesis, and no prefill can manufacture it.

**Error bar.** The 768 rows come from only 64 flag pairs (12 off-diagonal
question pairs each), so they are not independent. Clustering by pair:
65.1% +/- **4.8pp** (95%), against a naive binomial +/- 3.4pp. Quote the
clustered one. The 64 pairs cover 64 distinct countries.

### R11.2b The donor's question is causally irrelevant — the ENTIRE entity crosses

The single 65.1% shows the payload is entity-like. Breaking `both` out by BOTH
questions shows something stronger: the transfer rate is set by the question the
RECEIVING run asks, and is essentially independent of the question the donor was
asked.

```
   donor asked |  calling_code       capital      currency      language   row mean
  calling_code |            --         96.9%         43.8%         85.9%     75.5%
       capital |         31.2%            --         54.7%         84.4%     56.8%
      currency |         39.1%         96.9%            --         85.9%     74.0%
      language |         25.0%         92.2%         45.3%            --     54.2%
   column mean |         31.8%         95.3%         47.9%         85.4%
                        +/-7.5        +/-4.6       +/-11.4        +/-8.6   (pair-clustered 95%)
```

Every cell is far from zero: a donor asked **only** about `calling_code` lets the
receiving run answer `capital` at 96.9%, `language` at 85.9% and `currency` at
43.8%. The row means vary only because each row excludes a different column —
predicting each row mean as the average of the three columns it covers gives
76.2 / 55.0 / 70.8 / 58.3 against the observed 75.5 / 56.8 / 74.0 / 54.2. **The
row carries no information; the column carries all of it.**

Against the `flag` cell (same question, different flag), i.e. the *upper bound*
for a given column:

| Q1 | `flag` (same-question donor) | `both` (different-question donor) | shift |
|---|---|---|---|
| capital | 98.4 | 95.3 | -3.1 |
| language | 85.9 | 85.4 | -0.5 |
| currency | 46.9 | 47.9 | +1.0 |
| calling_code | 93.8 | 31.8 | **-62.0** |

`flag` and `both` are measured on the SAME 64 pairs, so the honest test is
paired, and all of its power sits in the pairs where the two cells DISAGREE:

| Q1 | both hit | both miss | flag only | both only | McNemar exact p | MDE @ 80% |
|---|---|---|---|---|---|---|
| calling_code | 4 | 3 | **56** | 1 | **8.1e-16** | 12.3pp |
| capital | 59 | 1 | 4 | 0 | 0.125 | 5.0pp |
| currency | 25 | 27 | 5 | 7 | 0.774 | 8.3pp |
| language | 54 | 9 | 1 | 0 | 1.000 | 1.5pp |

For three of four attributes the donor's question makes **no detectable
difference**: 4-vs-0, 5-vs-7 and 1-vs-0 discordant pairs. `currency`'s 5-vs-7 is
as balanced as noise gets, which is why its wide +/-8.3pp MDE is not a worry —
the *direction* is undetermined, not merely the size.

**State the precision honestly.** The point estimates are +3.1 / -1.0 / +0.5pp,
but the paired 95% CIs are [-0.4, +6.6], [-6.9, +4.8] and [-0.5, +1.5]. Only
`language` is tight enough to call equivalent outright; `capital` and `currency`
are bounded to roughly +/-7pp and +/-5pp. Quadrupling to 256 pairs halves every
MDE (1.5 -> 0.8, 5.0 -> 2.5, 8.3 -> 4.2) and is the run that would let the claim
be made without qualification.

The fourth attribute, `calling_code`, is 56-vs-1 discordant at p = 8e-16 — a real
and large effect. **It is not a failure of entity transfer.** Classifying what
the failures actually say:

| cell | exact source | source's FIRST digit, wrong after | exact base | base's first digit |
|---|---|---|---|---|
| `flag` (same-question donor) | 93.8% | 6.2% | 0.0% | 0.0% |
| `both` (different-question donor) | 31.8% | **66.7%** | 1.0% | 0.5% |

**98.5% of `both` rows emit the SOURCE's first digit** — higher than `language`'s
whole transfer rate. The base country's code appears in 1.5% of rows. Matched
examples, same pair, only the donor's question differing:

```
UY->KH (true +855)  donor asked capital  | flag: 855   both: 856
FI->AZ (true +994)  donor asked capital  | flag: 994   both: 998
CO->NA (true +264)  donor asked capital  | flag: 264   both: 267
```

The entity crossed intact; the *continuation* drifted. `patched_generate`
installs the donor's head output step-for-step, so donor step 1 is whatever the
donor's state was while emitting ITS second token — `" Penh"`, not the second
digit of `855`. Answers that fit in one step are immune (`language` is 78.6%
single-token golds); `calling_code` has **0/384** single-token golds and needs
2-3 aligned steps, so it is the only attribute the misalignment can reach. See
R11.3.

This is the strongest form of the claim, and it is what licenses "the heads carry
the whole entity" rather than the weaker "the heads carry something entity-like":
one capture, taken while the model was answering **one** question, supplies
enough to recompute **every** attribute the benchmark asks for.

### R11.3 The -16.1pp CONTENT SHIFT is a decode artefact, not question-conditioning

The script's headline contrast is `flag` (81.2%) vs `both` (65.1%) = **-16.1pp**,
which it interprets as question-conditioned content. Broken out by the question
the receiving run was asked, that shift lives in exactly one attribute:

| Q1 | `flag` | `both` | shift |
|---|---|---|---|
| capital | 98.4 | 95.3 | -3.1 |
| language | 85.9 | 85.4 | -0.5 |
| currency | 46.9 | 47.9 | **+1.0** |
| **calling_code** | **93.8** | **31.8** | **-62.0** |
| all four | 81.2 | 65.1 | **-16.1** |
| **excluding calling_code** | **77.1** | **76.2** | **-0.9** |

`calling_code` is the attribute with **0/384 single-token golds**. Its failures
are not wrong countries — against the same-question donor's own output for the
same pair, the `both` rows agree on the **first digit in 191/192 = 99.5%** of
cases (and on the *base's* first digit in only 17.7%):

```
UY->KH  q2=capital    got '856.'   same-question donor said '855.'
FI->AZ  q2=currency   got '998.'   same-question donor said '994.'
PY->AM  q2=language   got '359.'   same-question donor said '374.'
```

Same source country, trailing digits drifting. This is the continuous-patch
version of the multi-token artefact `swap_trace.py` documents: the donor's
step-2+ head outputs were captured while it answered a *different* question, so
after the first token the installed sequence no longer tracks the digits being
emitted. **The corrected content shift is -0.9pp.**

### R11.3b The artefact, confirmed by a double dissociation (`--align`)

R11.3 infers the artefact from *where* the shift lives (one attribute, the only
one with 0/384 single-token golds) and from first-digit agreement. `--align`
tests it directly by changing WHICH donor step gets installed at base step t.
`patched_generate` patches the last column of **every** forward, so the default
`matched` mode is a step-for-step replay (`idx = min(t, T-1)`); two alternatives
break that correspondence (`head_swap_vade.donor_index`):

| mode | donor step installed at base step t |
|---|---|
| `matched` | `min(t, T-1)` — the R10/R11 intervention |
| `hold0` | `0` — the donor's readout column, held at every step |
| `step0` | `0` at t=0, then the hook returns None and the base free-runs |

All three are identical at t=0 **by construction**, which is a free self-test:
the first generated token must agree across modes, row for row. It does —
1200/1200 exact `first_token` agreement between `step0` and `hold0`, and
1200/1200 on the first character against the `matched` run (which predates the
`first_token` field). `n_pairs 64 --seed 0` draws identical rows in all three.

`calling_code`, `source_q1` %:

| mode | `flag` | `both` | shift |
|---|---|---|---|
| `matched` | 93.8 | 31.8 | **-62.0** |
| `step0` | **1.6** | **1.6** | **+0.0** |
| `hold0` | 95.3 | 89.1 | -6.2 |

**Removing steps >=1 kills both cells identically.** `flag` falls 93.8 -> 1.6%,
so every point of its calling_code success was carried by the steps after 0, and
with those gone the two cells are indistinguishable. The gap was never content.

**Replacing steps >=1 with a question-independent signal rescues `both` at no
cost to `flag`.** 31.8 -> 89.1% (+57.3pp) while `flag` goes 93.8 -> 95.3%. So the
step-0 column's content is question-independent *and* sufficient, and `matched`'s
step >=1 was not merely useless but actively destructive. The -6.2pp residual in
the table is a difference of cell rates; paired by base item it is **-5.3pp, 95%
CI [-12.5, +0.8]pp over 44 items** — not significant. Do not report it as a
remaining content effect, and do not report it as exactly zero either.

The source's first digit arrives at **100.0% (`flag`) / 99.5% (`both`) in all
three modes** — the t=0 identity, unchanged, visible at the answer level.

Whole-design content shift: **-16.1pp (`matched`) -> -1.2pp (`hold0`) -> +0.4pp
(`step0`)**. R11.3's -0.9pp correction is reproduced by an intervention that does
not drop an attribute.

**A second result, not a bug fix.** `step0` retention tracks how much token 0
pins the rest of the answer:

| attribute | `matched` | `step0` | retained |
|---|---|---|---|
| language | 85.9% | 82.8% | 96.4% |
| capital | 98.4% | 85.9% | 87.3% |
| currency | 46.9% | 35.9% | 76.7% |
| calling_code | 93.8% | 1.6% | **1.7%** |

Single-token `language` barely moves — the control, predicted in advance. `capital`
mostly survives because `" Phnom"` leaves the base context almost no freedom;
`"8"` is consistent with 855, 856 and 880, so calling_code collapses. And `flag`
needs `hold0` (95.3%) where `step0` gives 1.6%: **the heads must keep asserting
the entity at every decode step.** The answer is not written at the readout
column and then unrolled from the KV cache.

The control that licenses all of this is the `self` cell: 93.0% (`matched`) ->
93.4% (`hold0`). Holding a step-0 column at every step is *benign on its own*, so
`hold0`'s +57pp recovery on `both` is signal, not the intervention accidentally
helping.

**`hold0` is not a replacement headline.** It is off-manifold and exists as
mechanism evidence. The reportable number stays `matched`, either excluding
calling_code (R11.3) or quoted as first-token agreement, with these two modes
cited as the proof that the exclusion is principled rather than convenient.

### R11.4 The `question` cell's zeros, and what they are worth

`question` reads 83.2% `base_q1` / 0.0% `base_q2`: a donor captured under a
different question, same flag, moves nothing. **Two** of this cell's golds are
unreachable, for two different reasons: `source_q1` because `donor_flag == base`
there, so that gold is literally the same string as `base_q1`; and `base_q2`
because of the prefill blocking described in R11.2. So the cell's only real
content is that it is INERT — 89.9% unchanged excluding `calling_code`, against
`self`'s 91.1%, a 1.2pp gap. That inertness is the evidence that the heads carry
no question identity; the `0.0% base_q2` next to it is not.

Excluding `calling_code` throughout, the whole design reads:

| cell | n | base_q1 | source_q1 | source_q2 | base_q2 | other |
|---|---|---|---|---|---|---|
| `self` | 192 | 91.1 | — | — | — | 8.9 |
| `flag` | 192 | 6.8 | **77.1** | — | — | 16.1 |
| `question` | 576 | 89.9 | — | — | 0.0 | 9.9 |
| `both` | 576 | 6.6 | **76.2** | **0.0** | 0.0 | 17.2 |

`both`'s 17.2% `other` is ~7pp above the inert floor — the real cost of the
mismatched continuation, and small.

Replicated independently at n=16 (`cross_n16.jsonl`): `flag` 85.9%, `both`
71.9%, shift -14.1pp, same shape.

### R11.5 The 7-dim value subspace is causally dead — two independent tests

R9's decomposition gave a 7-dim per-block subspace spanning `language`'s value
centroids, and a decoding check put language recovery from those 7 dims at 100%.
Both causal tests of it come back null.

**(a) `head_cross --subspace_dim 7`** (n=16, `cross_n16_sub7.jsonl`). All four
cells become indistinguishable:

| cell | base_q1 | source_q1 | other |
|---|---|---|---|
| `self` | 92.2 | — | 7.8 |
| `flag` | **92.2** | **0.0** | 7.8 |
| `question` | 92.2 | — | 7.8 |
| `both` | 91.7 | **0.0** | 8.3 |

The `flag` cell — which transfers at 81.2% with the full head columns —
transfers **0.0%**. 472/512 generations are bit-identical to clean, and the 40
that differ match `self`'s noise rate exactly.

**(b) `head_swap_vade --subspace_dim 7 --subspace_attribute language`**
(`results/head_swap_vade/flags_sub7/`, target=language, n=1434):

| arm | cause | generations differing from clean |
|---|---|---|
| `clean` | 1.7 | — |
| `sub7_heads_common10` | **1.7** | 4.2% |
| `sub7_random_heads_common10` | 1.7 | **6.6%** |
| `full_image` | — | 99.6% |

The selected-head subspace patch perturbs **fewer** generations than the
random-head subspace patch. Its read-back check passes at `0.00e+00`, so the
patch is genuinely applied — it simply carries nothing.

**What this does and does not mean.** It does not refute DAS. It kills one
shortcut: a subspace read off capture-time *variance* is not the causal carrier,
and the precondition test we used to justify it (7 dims -> 100% language
decoding) is **not predictive of causal sufficiency**. That is the same failure
`RESULTS.md` already records for the Phase B selection proxy, now reproduced at
the head site. A subspace that steers has to be found *against the generation
objective*, not against a decoder — which is exactly what DAS does and what this
test was not.

### R11.6 Summary of the head-site account

Six results, in the order they constrain each other:

1. **R8** — the image->text read is ~10 attention heads in blocks 21-23, and the
   same set serves all four attributes.
2. **R9** — those heads are image-*rich* (`item` 0.61-0.69), not
   attribute-selective; R5's opposite claim came from a void trace.
3. **R10** — on the benchmark, 10 of 784 head-slots (1.3%) reproduce a whole-image
   swap: 88.5% mean cause vs `full_image`'s 93.0%, with size-matched random head
   sets at 0.0%. Dose-response is monotone in k.
4. **R10.3/R10.4** — iso does not degrade, it **leaks to the source**, and
   `final_score` is pinned at 50% by arithmetic: one generation is scored as a
   required flip in one target file and a required non-flip in another.
5. **R11** — the payload is the **entity**, not the answer: a donor asked only
   Q2 yields flag B's answer to **Q1**, 65.1% +/- 4.8pp, an answer present in
   neither prompt. It is portable across prompts (81.2%) and carries no question
   identity (the `question` cell is inert to within 1.2pp of `self`). The
   apparent -16.1pp question-dependence is -0.9pp once `calling_code`'s
   multi-token decode artefact is removed. **Do not cite `source_q2` = 0.0%** —
   R11.2 shows it is forced by the receiving prompt's prefill.
6. **R11.5** — a variance-derived 7-dim subspace of those heads transfers 0.0%.

Taken together: these heads are a high-bandwidth entity conduit, VADE's metric
punishes exactly that, and separating "which attribute" out of the conduit — if
it is separable at all — requires a subspace learned against generation. That is
the motivation for the two training experiments that follow.

---

## 7. `methods/head_das.py` — learning a subspace of the conduit

R11 leaves one question open: the ten heads move every attribute together, but
does the payload **decompose**? Is there a subspace of those 1,280 columns that
carries "which language" separably from "which country"? R11.5 shows the cheap
answer fails (a variance-derived 7-dim subspace transfers 0.0%), so the only way
left is to learn the subspace *against the generation objective*.

Three hypothesis classes, one loop, all modules imported rather than
reimplemented:

| `--method` | module | what is learned |
|---|---|---|
| `das_fixed` | VADE's `FixedSubspaceIntervention` | a D x K semi-orthogonal `R`, K fixed up front |
| `das_rotated` | VADE's `RotatedSpaceIntervention` | a full D x D rotation + annealed sigmoid mask, so K is learned |
| `dbm` | `methods/dbm/`'s `SigmoidMaskIntervention` | an axis-aligned mask over raw head dims + L1 |

One intervention is trained **per (entity, attribute)** and is block-diagonal by
necessity — blocks run sequentially, so a single joint rotation across 21/22/23
is not expressible in one hook. Widths are `n_heads_in_block * head_dim`
(common10: 256 / 512 / 512).

The objective IS VADE's metric made differentiable: each row is supervised
toward the SOURCE gold when `rule == match_source` and the BASE gold when
`rule == match_base`. Trained on VADE's `train` split, evaluated on `test` —
**split by item**, so 59 countries train and a disjoint 25 evaluate (asserted in
`--dry_run`). The patch covers every answer token, not just the last prompt
token, via `build_teacher_forced_extension`; a last-token patch cannot steer past
the first answer token (see `ndm/swap_trace.py`).

### `--subspace_dim`: one value, per block, or `full`

```bash
python methods/head_das.py --attribute language --subspace_dim 128          # all blocks
python methods/head_das.py --attribute language --subspace_dim 128,256,256  # per block
python methods/head_das.py --attribute language --subspace_dim full         # the ceiling
```

Per-block values matter because common10's heads are spread **2/4/4** over
blocks 21/22/23, so a single K is a different *fraction* of each block's space.

**The zero-DOF guard, and why it is not cosmetic.** `FixedSubspaceIntervention`'s
output depends on `R` only through `R^T R`, so the hypothesis class is the
Grassmannian Gr(K, D) of dimension **K(D-K)**, not the K*D stored parameters. At
`K == D` that is **zero**: `R^T R = I` identically, the module IS the full swap
whatever the weights say, and every gradient is pure gauge. Verified:

```
K= 128  max|out - source| = 5.250e+00   max|grad| = 1.008e+02
K= 512  max|out - source| = 6.229e-06   max|grad| = 1.831e-04
```

So `--subspace_dim full` is a *measurement*, not a run; training is skipped
automatically. `--dry_run` and `summary.json` report per-block DOF, which also
surfaces **mixed** arms: `--subspace_dim 256` silently clamps block 21 (width
256) to a full swap while 22 and 23 train, so its point on a K curve is not
comparable to arms where every block trains.

Run tags encode the knob that varies (`_k128`, `_k128-256-256`, `_kfull`,
`_m0.001`, `_l10.001`) so a sweep cannot overwrite itself — the gotcha CLAUDE.md
records for `select_features.py`.

### `--mask_lr` / `--mask_init` / `--temperature_*`: making the mask movable

`das_rotated` and `dbm` learn their width through `sigmoid(m / T)`, so **every**
gradient reaching `masks` is scaled by `sigmoid'(m/T)/T = s(1-s)/T`. Once `m/T`
is large that factor underflows to *exactly* 0 in float32 and the mask is frozen
for the rest of training — silently, with the run still producing fluent text
and a plausible score. That is R12.3.

Adam normalizes by gradient magnitude, so while the factor is non-zero the mask
moves ~`lr` per step. `mask_travel_budget(mask_init, temps, mask_lr)` turns that
into a checkable number and every `das_rotated` run prints it:

```
  param groups: rotation lr=0.001, masks lr=4
  mask anneal T 50 -> 0.1; mask_init=150
  mask gradient survives 80/375 steps -> travel budget 320.0 vs 150.0 needed
```

`travel < need` prints a WARNING naming the arm as an (almost) full swap
whatever `--mask_coef` says. Sized at three realistic step counts:

| opt steps | config | live | travel | need | |
|---|---|---|---|---|---|
| 94 | R12.3's runs (`lr 1e-3`, `--grad_accum_steps 4`) | 20/94 | 0.02 | 150 | **PINNED** |
| 375 | `--mask_lr 1.0` | 80/375 | 80 | 150 | **PINNED** |
| 375 | `--mask_lr 4.0` | 80/375 | 320 | 150 | OK |
| 375 | `--mask_init 3 --temperature_start 1 --mask_lr 0.2` | 206/375 | 41 | 3 | OK |

Three things to know before tuning against it:

* **`--grad_accum_steps` is the hidden variable.** R12.3 used 4, which turned 375
  batches into 94 optimizer steps. Dropping to 1 is *free* — same batches, same
  wall clock, 4x the steps.
* **Only the RATIO `mask_init / T` is visible to the sigmoid**, so `--mask_init`
  and `--temperature_start` must move together. Lowering `mask_init` alone walks
  into the `masks = 0` degeneracy VADE's own docstring warns about, where
  `mixed = 0.5*(rotated_source + rotated_base)` makes the rotation cancel.
* **The guard is conservative and is a floor, not a target.** It evaluates
  liveness at the *initial* mask value; as the mask falls so does `m/T`, so the
  real window is longer. R14's arms were still moving at step 115 of a predicted
  80. Do not raise `--mask_lr` to make the printed number larger.

`masks` gets its own Adam parameter group because its step size is set by the
DISTANCE it must travel, which has nothing to do with the rotation's step size.
Every knob that changes the run is in the output tag (`_m3e-4_mlr4_mi3_T1`), so a
sweep cannot overwrite itself — R12.3's arms did, because only `--mask_coef` was
encoded.

### `methods/head_das_table.py` — read SLACK, not `final_score`

```bash
python methods/head_das_table.py                       # flags/language, every arm
python methods/head_das_table.py --attribute currency
```

`final_score` hides the result. Every arm at this site lies on a monotone
cause<->iso trade-off, so arms that behave very differently (cause 46.6 / iso
67.7 versus cause 73.2 / iso 39.9) score within noise of each other.

> **slack** = the arm's `iso`, minus the `iso` a *probabilistic* full swap would
> reach at that `cause` — i.e. the straight line from the clean run (cause 0, iso
> 94.2) to the full-swap ceiling (cause 94.7, iso 1.7). A coin flip between "swap
> everything" and "swap nothing" has slack 0 by construction, so slack is the
> only column that measures whether an arm is doing something a random mixture
> could not.

The table also reports **width** — summed `subspace_dim` for `das_fixed`, summed
`mask_sum` for the masked methods — because a rotated arm and a fixed arm are
only comparable when they spend the same number of dimensions. It sorts by slack,
prints an SE on `final_score` with a settable design effect for item clustering
(default 2.0), and lists incomplete runs rather than silently omitting them.

## R12. The conduit decomposes, but only to ~57%

`methods/head_das.py`, flags, `common10`, `das_fixed`, trained on `train` and
evaluated on the disjoint `test` items.

### R12.1 The four ceilings, and the completed grid

`--subspace_dim full` costs one eval pass and no training. Running it for all
four attributes fills the target x queried grid that R10 could not complete —
the head-swap run never asked the `language` question:

```
       target |  calling_code       capital      currency      language   final
 calling_code |        95.1*         98.6          71.7          92.7    48.7%
      capital |        95.6          97.9*         72.0          93.0    50.1%
     currency |        94.5          98.4          76.1*         93.2    38.5%
     language |        95.0          98.4          72.5          94.7*   48.2%
  (* = the cause cell; every other entry is an iso cell. pct_matches_source.)
```

**Read the column HEIGHTS, not the row-to-row flatness.** The four runs apply a
*byte-identical* edit — a full swap has no access to `target_attribute` — and
VADE's four tuple files are the **same 14,052 (base, source, queried,
template_id) rows**, differing only in the `rule` label (verified: 100% pairwise
key overlap). So switching `--attribute` is pure relabelling plus a different
stratified subsample, and flat columns are guaranteed by construction. They are
worth exactly two narrower things:

* a **no-leakage check** on the harness — of the 1,790 keys generated by more
  than one of the four runs, 95.1% produced character-identical text, and every
  disagreement is a trailing-continuation difference from batch padding, not a
  different answer. Nothing in the donor capture or the teacher-forced extension
  conditions on the target;
* an **error bar** — four independent 2,000-row subsamples put sampling noise at
  0.7-4.4pp per cell at n ~ 500, which is the yardstick for reading everything
  below.

What IS new is the `language` column: language transfers at 92.7-94.7% when it is
not the target. That completes R10's "one edit moves all four attributes at
72-98%", which is readable from any **single row**. The four per-attribute
ceilings (48.7 / 50.1 / 38.5 / 48.2) are the baselines every trained arm must
beat; `currency`'s low 38.5% is mostly the model, since clean currency accuracy
is only 82.6%.

### R12.2 The K sweep: a flat plateau at ~57%

`slack` is the distance above the straight line joining clean to the measured
ceiling — what a K-indexed dial with no selectivity at all would give. Both
endpoints sit on that line by construction, so any bulge is real.

| arm | cause | iso | final | slack | swapped dims |
|---|---|---|---|---|---|
| clean | 0.0% | 94.2% | 47.1% | +0.0pp | 0 |
| K=8 | 14.0% | 85.7% | 49.9% | +5.2pp | 24 |
| K=16 | 20.8% | 79.1% | 50.0% | +5.2pp | 48 |
| K=32 | 33.9% | 71.5% | 52.7% | +10.4pp | 96 |
| K=64 | 33.0% | 75.3% | 54.1% | +13.3pp | 192 |
| **K=128** | 46.6% | 67.7% | **57.2%** | +19.0pp | 384 |
| **128,256,256** | 62.8% | 52.1% | **57.4%** | **+19.2pp** | 640 |
| 192,256,256 | 67.9% | 45.5% | 56.7% | +17.6pp | 704 |
| K=256 (mixed) | 73.2% | 39.9% | 56.5% | +17.2pp | 768 |
| `full` (ceiling) | 94.7% | 1.7% | 48.2% | -0.0pp | 1280 |

Two readings, and the second is the finding:

1. **The conduit does decompose, a little.** Every interior arm sits above the
   trivial line, peaking at +19.2pp. There is a subspace of roughly half of each
   block that carries more `language` than `capital`.
2. **`final_score` is pinned at 56.5-57.4% across a 27-point range of `cause`
   (46.6% -> 73.2%)**, over three different K parameterisations. On n=2000 the
   standard error on `final_score` is ~ +/-1.3pp, so `128,256,256`'s 57.4% and
   K=128's 57.2% are the same number. Per-block allocation bought nothing, and
   past the peak the extra dimensions buy only full-swap behaviour: iso
   `matches_source` climbs 6.9% -> 26.3% between K=128 and K=256 while inert
   (base,source) pairs collapse 159/598 -> 38/598.

The cap is not a training-budget artefact. K=8 has **converged** (tail slope
-0.7% per 5 steps, CE flat at 2.6) and still reads 14.0%; a 52-step run on the
full 32,811-row split reached CE 0.425 against 0.466 at 40 steps. Longer training
moves along this curve, not above it.

### R12.3 `das_rotated` as VADE ships it cannot sparsify — VOID

Three runs at `--mask_coef` 3e-4 / 1e-3 / 3e-3 produced **byte-identical
predictions** (md5 `43371e15...`), all equal to the full swap, with
`mask_sum == dim` exactly on every block at final temperature 0.1.

VADE never instantiates `RotatedSpaceIntervention` — its trainer only ever builds
`FixedSubspaceIntervention` — so this path had never been exercised. Two
compounding failures, measured:

| T | mask | \|out-src\| | grad `masks` | grad `R` |
|---|---|---|---|---|
| 50.0 (start) | 0.95257 | 2.45e-01 | 5.64e-02 | 4.20e-05 |
| 13.4 | 0.99999 | 7.08e-05 | 7.12e-05 | 3.43e-05 |
| 1.0 | 1.00000 | 1.91e-06 | **0.00e+00** | 3.43e-05 |
| 0.1 (end) | 1.00000 | 1.43e-06 | **0.00e+00** | 4.20e-05 |

* `mask_init = 150` with Adam at `lr = 1e-3`: Adam's per-step move is bounded by
  ~`lr`, so reaching the decision boundary at 0 needs **~150,000 steps**. The
  runs did 94. Observed rate, from a killed run: `maskK` 1280 -> 1279.9656 in 22
  steps, i.e. ~580,000 steps to reach 384.
* The temperature anneal makes it **worse**: by T <= 1, `sigmoid'(150/T)`
  underflows and the mask gradient is *exactly* zero.
* And at `mask == 1`, `mixed = rotated_source`, so `output = rotated_source @ R
  = source` — **the rotation cancels** and `R` has no useful gradient either.
  This is the exact mirror of the `masks = 0` degeneracy VADE's own docstring
  warns about.

`dbm` does **not** have this trap: it inits at `mask = 0.5`, the maximally
informative point, with gradients of 4e2 -> 4e7 as T falls.

**This has since been fixed and the diagnosis confirmed exactly — see R14.** The
checkpoints of these three void arms carry `mask_sum == dim` on every block
(1280/1280), so they were not DAS that underperformed: they were the **full
swap** wearing a DAS label, which is why all three scored 94.7 / 1.7 / 48.2,
identical to `das_fixed --subspace_dim full`. They appear in R14's table at width
1280, slack -0.0, and should be cited only as the negative control.

### R12.4 Where this leaves the head site

The `das_fixed` sweep is finished; no configuration left will move it. The site
admits ~+19pp of slack over trivial and caps `final_score` near 57%, against
endpoints at 47-48%. Two questions remained:

* does a **learned** width agree with K ~ 128-256 per block? **Answered in R14**:
  it does not. A learned width traces the same trade-off but reaches a *better*
  slack at every width, and the advantage grows as the width shrinks.
* is ~57% a property of *these heads* or of the model? **R13 answers this
  structurally** — these heads carry 77-89% entity variance and 3-15% question
  variance, so a question-blind intervention here has no input on which to be
  selective. The cap is the site, not the method.

## R13. Why no intervention at this site can be selective — measured, and partly a theorem

R12's plateau is usually read as a training result. It is not: it follows from
what the intervention can *see*. This section measures that, off the crossed
activation grid `attr_capture.py` already wrote (84 items x 4 attributes at
`last_token`, blocks 15-27) with **no model and no GPU**.

```bash
python methods/attr_variance.py --site residual                                  # R13.1 top
python methods/attr_variance.py --site attn_head_output --heads common10 \
    --blocks 21 22 23                                                            # R13.1 bottom
python methods/attr_variance.py --per_head --blocks 21 22 23 --site attn_head_output
```

`methods/attr_variance.py` asserts the grid is balanced before decomposing -- the
exactness below depends on it, and a capture with a missing cell would otherwise
produce plausible percentages. It takes any entity's capture via `--capture_dir`,
so R13 reruns on brands/animals for the cost of an `attr_capture.py` run.

### R13.1 The decomposition

The grid is balanced (one row per (item, attribute) cell) and the activations are
deterministic — one forward pass per cell, no replication — so the two-way
decomposition `SS_total = SS_item + SS_attr + SS_interaction` is exact and the
residual term **is** the interaction, not noise.

The interaction is the quantity VADE is about. `item` alone is "which country";
`attribute` alone is "which question"; only `item x attribute` is *this country's
this attribute*, and only a representation carrying it can be edited for one
attribute without moving the others.

`residual` @ `last_token`, % of variance explained:

| block | item (entity) | attribute (question) | **item x attribute** |
|---|---|---|---|
| 15 | 26.5 | 71.7 | 1.9 |
| 17 | 13.2 | 84.9 | 2.0 |
| 19 | 21.0 | 76.2 | 2.8 |
| 21 | 12.8 | 84.3 | **2.9** |
| 22 | 15.0 | 81.6 | **3.4** |
| 23 | 32.1 | 62.6 | **5.2** |
| 24 | 40.3 | 50.3 | **9.3** |
| 25 | 31.9 | 51.1 | **17.1** |
| 26 | 29.6 | 46.3 | **24.1** |
| 27 | 28.0 | 49.4 | 22.6 |

`attn_head_output`, restricted to the `common10` columns — the site R12 trained
on:

| block | item | attribute | item x attribute |
|---|---|---|---|
| 21 | **77.0** | 14.7 | 8.3 |
| 22 | **82.1** | 5.3 | 12.7 |
| 23 | **88.7** | 3.3 | 8.0 |

Two readings:

* **The DAS runs were three to five blocks upstream of the quantity VADE scores.**
  The bound representation is ~3% of the variance through block 22 and does not
  become substantial until 25-26.
* **The `common10` heads are 77-89% "which country" and 3-15% "which question".**
  This is R11's -1.2pp content shift measured directly in the activations rather
  than inferred from generations, on a different quantity, off a capture the
  head_cross runs never touched.

### R13.2 The identifiability argument

A DAS/DBM intervention here is one fixed trained function `f(base_head_output,
donor_head_output)`, applied identically to every row. On a cause row it must
transfer; on an iso row it must not. Those rows differ only in which question was
asked — and the question is 3-15% of what `f` can see.

**The information required to be selective is not in the intervention's input.**
`cause` and `iso` are therefore yoked, which is exactly R12.2's curve: `final_score`
pinned at 56.5-58.3% while `cause` ran 41.5 -> 73.3%, and still only 54.7% at the
sparsest arm's `cause` of 31.4%. No K, no learning rate, no mask schedule
addresses this, and R14 confirms it for a second hypothesis class.

What DAS *does* buy is real and should be reported: at cause 46.6%, a
probabilistic full swap predicts iso 48.7% and `fixed_k128` measures 67.7%. The
~+19-21pp of slack is the honest ceiling for a question-blind intervention at an
entity-coding site.

### R13.3 At image positions this is not a measurement but a theorem

`build_prompt` puts the image before the text question in the same user turn
(`head_swap_vade.py`). Every image token therefore precedes every question token,
and under causal masking **image-position activations at every layer are
bit-identical across the four questions** — not "mostly", not "-1.2pp".

So any intervention at image positions is question-blind *by architecture*, for
every method, forever, unless the attributes are separately encoded in the visual
representation itself. VADE's own design patches the object's image tokens; this
is a structural cap on that design, and R11/R12/R14 are its downstream shadow at
the heads that read those positions.

This is one line to verify and has **not** been run — it is the same check
`seq:before` uses in `methods/common/position_sets.py`: capture image-position
activations under two different questions and assert bit-identity. Until it is
run, treat R13.3 as an argument from the prompt layout, not a measurement.

### R13.4 What this does and does not license

It does **not** say the model is incapable of the task, nor that a different site
cannot be selective — R13.1's own table shows the interaction rising to 24.1% by
block 26.

It does say that a *question-blind* intervention at an *entity-coding* site is
structurally capped, and that the three places this project has trained one
(image positions, `common10`, and `residual` at the image span) are all such
sites. The natural control is the same machinery at `residual`/`last_token`/L24,
where the interaction is 3x block 21's and the full-swap ceiling is already
measured at 100% for language. That arm is **outside the benchmark's intended
site** — at the last text token you have left the visual pathway, and a success
there is a claim about factual recall, not visual attribute encoding. It is worth
running as a **positive control** that disambiguates "our trainer is broken" from
"the site cannot support selectivity", and it should be labelled as one.

## R14. `das_rotated`, fixed: a canonical ~50-dim ENTITY subspace

R12.3's three arms were void. With `--mask_lr 4.0 --grad_accum_steps 1` the mask
moves and the hypothesis class is tested for the first time. Five arms,
flags/language, `common10`, 6000 train / 2000 eval rows, 375 optimizer steps.

```bash
for C in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  python methods/head_das.py --attribute language --method das_rotated --mask_coef $C \
    --train_rows 6000 --eval_rows 2000 --batch_size 16 --grad_accum_steps 1 --mask_lr 4.0
done
python methods/head_das_table.py                                      # R14.1
python methods/das_subspace_geometry.py --checkpoints <dir of *.pt>   # R14.3, R14.4
```

R14.3 and R14.4 are read off the **checkpoints**, not the predictions: two arms
that found completely different geometry can score identically on a trade-off
curve, so `final_score` and even `slack` cannot see any of what follows.

### R14.1 The sweep, and the retraction it completes

| `--mask_coef` | width | cause | iso | final | slack | slack/dim |
|---|---|---|---|---|---|---|
| 1e-4 | 922 | 73.3% | 43.1% | 58.2% | +20.5 | 0.022 |
| **3e-4** | 773 | 65.8% | 50.8% | **58.3%** | **+20.9** | 0.027 |
| 1e-3 | 436 | 53.2% | 62.5% | 57.9% | +20.3 | 0.047 |
| 3e-3 | 178 | 41.5% | 72.1% | 56.8% | +18.4 | 0.103 |
| 1e-2 | 52 | 31.4% | 77.9% | 54.7% | +14.4 | **0.277** |
| *(void, R12.3)* | *1280* | *94.7%* | *1.7%* | *48.2%* | *-0.0* | *0.000* |

The void arms land at width 1280 — `mask_sum == dim` on every block. They were
the full swap, confirming R12.3's diagnosis from the checkpoints rather than from
the gradient table.

Width falls monotonically with `--mask_coef`; slack peaks at 773 and falls. So on
the score axis **there is no compact subspace that wins**, and `final_score`'s
+0.9pp over `das_fixed`'s best is inside the error bar (SE 1.8pp at design effect
2.0). The plateau is intact across two hypothesis classes.

### R14.2 Annealed sparsification beats fixed-K, and the gap grows as K shrinks

At matched width — the only comparison that separates "the method helps" from
"this arm sits further along the same curve":

| width | `das_fixed` | `das_rotated` | gap |
|---|---|---|---|
| ~50 | +5.2 (K=16) | **+14.4** | **+9.2** (2.8x) |
| ~185 | +13.3 (K=64) | +18.4 | +5.1 |
| ~410 | +19.0 (K=128) | +20.3 | +1.3 |
| ~770 | +17.2 (K=256) | +20.9 | +3.7 |

Both classes learn an arbitrarily-oriented subspace (`FixedSubspaceIntervention`
is Grassmannian — see §7 — and `das_rotated`'s masks are **perfectly binary**:
zero dimensions anywhere in 0.01-0.99 at final temperature, on every arm). So the
difference is not soft-versus-hard transfer and not axis alignment. It is
optimization: sparsifying down from a working full swap finds a much better small
subspace than optimizing a hard K directly. The rotated arm also *allocates*
across blocks (13/14/25 of 256/512/512 at the tightest budget) where a single K
cannot.

### R14.3 The subspaces are nested, far beyond chance

Fraction of the sparse arm's span captured by the denser arm's, against the
random-subspace expectation (`observed / chance`):

| sparse | dims | vs 178 | vs 436 | vs 773 | vs 922 |
|---|---|---|---|---|---|
| 1e-2 | 52 | **0.910** / 0.147 | 0.920 / 0.347 | 0.936 / 0.595 | 0.941 / 0.699 |
| 3e-3 | 178 | — | 0.619 / 0.348 | 0.786 / 0.600 | 0.851 / 0.706 |
| 1e-3 | 436 | — | — | 0.705 / 0.603 | 0.805 / 0.716 |
| 3e-4 | 773 | — | — | — | 0.787 / 0.726 |

The 52-dim subspace sits **91% inside** the 178-dim one where chance is 15% — a
factor of 6.2. The nesting is strongest at the sparse end and **dissolves at the
dense end** (0.787 against 0.726 chance). The dense arms are a canonical core plus
several hundred dimensions of essentially arbitrary padding, which is why slack
per dimension falls 10x from 52 to 773 dims.

**Two controls rule out a shared-initialization artefact.** The five rotations are
*mutually random*: mean |cos| between same-index rotated axes is 0.033-0.054,
matching sqrt(2/pi*D) = 0.050 (D=256) and 0.035 (D=512) to two digits. And the
mask index Jaccard is at chance in all ten pairs (e.g. 0.044 vs 0.039). Five
independent trainings, from mutually random bases, keeping different coordinates,
converge on the same **span**.

### R14.4 But the core is entity structure, not attribute structure

Fraction of the sparse core's span captured by three reference subspaces built
from the R13 grid at the same columns:

| block | dims | top-k PCA | **item means** | **attribute means** | chance |
|---|---|---|---|---|---|
| 21 | 13 | 0.321 | **0.342** | 0.010 | 0.051 |
| 22 | 14 | 0.434 | **0.433** | 0.030 | 0.027 |
| 23 | 25 | 0.338 | **0.354** | 0.011 | 0.049 |

The core aligns with the **entity** directions at 7-16x chance and with the
**attribute** directions at roughly chance. The attribute-mean subspace has rank
3 (four attributes), so a 13-dim core could capture at most 3/13 = 0.23 of its
own span from it; the observed 0.010 is **4% of that maximum**. Item and PCA
overlaps are near-identical, consistent with R13.1's finding that these columns
are 77-89% item variance — the top principal directions here *are* the entity
directions.

So DAS, given a free rotation and a free width, found a compact canonical
**entity code**. It did not find an attribute subspace, because R13 says there
isn't one here to find.

### R14.5 What is established, and the one control still missing

Established:

* a **canonical ~50-dimensional entity subspace** at the `common10` heads,
  recovered by five independent runs from mutually random bases, carrying 69% of
  the best arm's slack (+14.4 of +20.9) in 52 of the site's 1,280 columns (4.1%);
* **annealed sparsification > fixed-K optimization** at this site, by 2.8x in
  slack at ~50 dims;
* the plateau survives a second hypothesis class, and the thing DAS converges on
  is entity-aligned — which is R13's prediction, tested against geometry the
  score table cannot see.

Missing: **all five arms share the row-sampling seed**, so they saw the same data
in the same order. Mutually random rotations rule out shared *initialization*, not
shared *data order*.

```bash
python methods/head_das.py --attribute language --method das_rotated --mask_coef 1e-2 \
  --train_rows 6000 --eval_rows 2000 --batch_size 16 --grad_accum_steps 1 \
  --mask_lr 4.0 --seed 1
```

If that arm's 52-dim span still lands ~0.9 inside the seed-0 178-dim arm, the
canonical core is a property of the model and R14.3-R14.5 stand. If it drops
toward 0.15, the nesting was data order and R14.3 onward collapse to a statement
about one training run. **Do not cite the canonical core before this runs.**

## R15. The account, after R11-R14

Four independent lines now say the same thing, and they were measured on
different quantities with different tooling:

1. **R1** — one shared router in blocks 19-21 carries *which attribute was asked*;
   the per-attribute top-8 lists are nearly identical. No per-attribute heads.
2. **R8/R9** — one shared circuit in blocks 21-23 reads the entity out of the
   image; again no per-attribute heads.
3. **R11/R11.3b** — what those heads transfer is question-independent: content
   shift -1.2pp (`hold0`) / +0.4pp (`step0`), and the -16.1pp headline was a
   decode artefact, proved by a double dissociation rather than inferred.
4. **R13/R14** — the same columns are 77-89% entity variance and 3-15% question
   variance, and the best subspace DAS can find in them is entity-aligned at
   7-16x chance and attribute-aligned at chance.

**The model does not disentangle visual attributes: it stores an entity identity
at blocks 21-23 and computes attributes on demand at 25-27.** For this checkpoint
VADE's premise does not hold, and R13.3 argues the benchmark's chosen patch site
cannot test it either way, because image-position activations are question-blind
under causal masking.

That is a finding, not a failed experiment — and it predicts R12/R14's plateau
quantitatively rather than after the fact. The positive results that come with it
are the ~+19-21pp slack ceiling for question-blind interventions at this site,
the canonical ~50-dim entity subspace (pending R14.5) — 4.1% of the site's
columns carrying 69% of the best arm's slack — and the sparsification result in
R14.2.

**What would move it, in order of cost:**

| | experiment | cost | what it settles |
|---|---|---|---|
| 1 | R14.5's `--seed 1` rerun | 1 run | whether the canonical core is real |
| 2 | R13.3's bit-identity check | no GPU | turns the image-side cap from argument into measurement |
| 3 | `residual`/`last_token`/L24, labelled a positive control | 1 run | disambiguates "trainer broken" from "site cannot support it" |
| 4 | the oracle arm — give `head_das` the `target_attribute` | 1 run | upper bound at `common10` under a question-blind input |

