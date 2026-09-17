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
