# Same-image attribute-switching sweep

This diagnostic uses VADE flag images and ground-truth labels, keeping the image
identical while changing the requested attribute. It does not train masks or
change the country. These are controlled experimental prompts, not VADE's
original/pruned prompt tuples. The earlier image ceilings changed the image
under a fixed question; this experiment changes the donor question instead.

## Run

Use the existing GPU environment from the head experiments, from the repo root.

```bash
# Metadata-only validation; no model or torch required.
python3 methods/attribute_switch_sweep.py --dry_run

# First check whether the clean model answers the controlled prompts correctly.
python methods/attribute_switch_sweep.py --baselines_only --out_dir results/attribute_switch/baselines

# Small sweep: two countries, all 12 directed attribute switches.
python methods/attribute_switch_sweep.py --n_countries 2 --layers 8 16 24 28 --block_spans 2 --out_dir results/attribute_switch/smoke

# Full depth, 8 countries, all sites, spans 2/4/8, prefill intervention.
python methods/attribute_switch_sweep.py --out_dir results/attribute_switch/prefill

# Compare prefill and continuous interventions at interesting layers.
python methods/attribute_switch_sweep.py --layers 16 20 24 28 --modes prefill continuous --out_dir results/attribute_switch/continuous

# New text-swap smoke run: all three scopes, all four sites, both window families.
python methods/attribute_switch_sweep.py --n_countries 2 --attributes capital currency \
  --scopes earlier_text last_token all_text --sites residual attention mlp joint \
  --layers 8 16 20 24 28 --block_spans 2 4 --attention_spans 2 4 \
  --out_dir results/attribute_switch/text_smoke

# Full depth, all attributes, independent single layers and consecutive windows.
python methods/attribute_switch_sweep.py \
  --scopes earlier_text last_token all_text --sites residual attention mlp joint \
  --block_spans 1 2 3 4 5 8 --attention_spans 1 2 3 4 5 8 \
  --out_dir results/attribute_switch/text_full
```

Defaults include self-patching and same-attribute paraphrase controls at every
site/layer. This makes the full sweep expensive: the command prints the number
of generation runs before loading the model. Narrow with `--attributes capital
currency`, `--countries FR DE`, `--sites last_residual last_joint`, or
`--block_spans 2`. Pass `--block_spans` alone for no additional spans. Keep the
same seed/countries when comparing runs. Rerun an identical command to resume;
use another output directory when changing configuration or code.

### Malformed clean answers and multimodal positions

The runner supplies `mm_token_type_ids` when the installed Qwen implementation
requires them for multimodal rotary positions. The shared input adapter's
historical pixel/grid-only output omitted these labels; newer Transformers can
silently fall back to text positions in their absence. Labels are rebuilt for
every full prefix, including generated text. Older implementations continue
using their own token-ID inference.

After this fix, rerun clean baselines in a **new output directory** before the
sweep. Existing results cannot be resumed across the code change. This corrects
position handling, but clean answer quality must still be checked on the actual
pretrained model and prompts. Offline multi-image-token tests compare the runner
with explicit spatial positions and cached generation; self-swap identity alone
would not detect a position bug shared by clean and patched forwards.

Console `first` and `full` score the donor answer. `base_full` scores the
recipient answer, so a correct self-swap normally has `full=0 base_full=1` when
the two answers differ. Code-like output in a clean run is a separate problem
from those donor scores being zero.

## Faster prefill sweeps on a 24 GB GPU

Add `--execution cached` to batch independent prefill arms on one model instance.
The default remains `--execution serial` for reference comparisons.

Three things make the cached engine fast, and only the first is old:

1. **Batched arms + per-arm KV caches** (`--batch_size`). Independent arms share
   one prefill; each keeps its own cache through decoding.
2. **A shared prompt prefix** (on by default; `--no_share_prefix` opts out).
   Every prompt variant of a row agrees on a long token prefix -- on the flag
   task 205 of 219 tokens, with the whole image inside it. That prefix's KV
   cache is built once per row and per batch size and reused by every donor and
   every arm, so each prefill computes only the divergent tail and the vision
   tower runs **once per row** instead of once per lane per chunk. Patch columns
   below the divergence are dropped: donor and recipient read the same cache
   entries there, so writing one onto the other is a no-op by construction.
3. **SDPA attention** (`--attn_impl`, default `sdpa`). Nothing in this script
   reads attention weights, and `eager` materializes a `[batch, heads, T, T]`
   tensor per layer. Pass `--attn_impl eager` to reproduce pre-2026-09 runs.

`--summary_every N` (default 16) controls how often `switch_summary.json` is
rebuilt. It used to be rewritten after every row over every record written so
far, which made the CPU side quadratic in rows while the GPU idled; the summary
is a pure function of `rows.jsonl`, so the cadence only affects how stale an
interrupted run's summary is.

**These changes invalidate resume into existing directories.** `config.json`
records `attn_impl`/`no_share_prefix`, `runtime.json` now records the actual
attention backend, and `implementation_sha256` covers both edited modules. The
partially complete `attention_18_24_64_cached_b96` and `text_continuous_cached_b2`
runs therefore need a new `--out_dir` (or `--attn_impl eager --no_share_prefix`,
which restores the old identity on everything except the implementation hash).

Verify before trusting a long run -- this compares every cached token logit and
greedy choice against full-prefix serial execution, and intentionally removes the
speed benefit:

```bash
python methods/attribute_switch_sweep.py \
  --n_countries 1 --attributes capital currency \
  --scopes last_token --sites attention --layers 22 \
  --block_spans --attention_spans 3 \
  --execution cached --batch_size 2 --verify_cached \
  --out_dir results/attribute_switch/cached_verify
```

The CPU test suite (`python -m pytest tests/`) covers the same equivalence on a
tiny randomly-initialized VLM at batch sizes 1/2/4 across every site, scope and
control, plus the shared-prefix path, its opt-out, and the dropped-column
bookkeeping. It establishes mechanical equivalence; actual speed, memory use and
BF16 numerical agreement still need measuring on the real model.

The cached engine, in detail:

- Captures each needed donor question variant per row and active batch size,
  storing only the requested text positions/sites on CPU. Donor values at prompt positions cannot
  depend on subsequent generated tokens, so they are reused across arms/steps.
  Donor prefills use the same batch size, cache settings, and position construction
  as recipients; each lane's activations are retained separately. A shorter final
  batch rebuilds the bank at that size. Comparing batch-one donor activations to
  batched BF16 recipients can otherwise fail self-swap identity checks.
- Runs multiple recipient arms in one batched prefill. Each batch element has
  its own interventions, including when scopes, layers, sites, or donors differ.
- Builds recipient KV caches while the patches are active, removes hooks after
  prefill, and decodes one new token per step. It does not rerun the vision tower
  during decoding. Explicit multimodal rotary positions prevent donor forwards
  from contaminating cached position state.
- Stops recording each arm independently at EOS. Finished batch lanes stay
  inactive until the batch finishes. This is fixed-size batching, not a serving
  scheduler that continuously refills lanes.
- Still executes every self and paraphrase control. Clean baselines and any
  continuous intervention arms use the serial reference path. Cached execution
  currently supports the runner's image-only Qwen inputs.

First verify the GPU runtime on a small configuration. This diagnostic checks
every cached next-token logit and greedy choice against full-prefix serial
execution. It intentionally removes the speed benefit:

Identity/verification tolerances follow the model's computation dtype, not a
possibly upcast logit tensor. A changed greedy token always fails. Failures report
the arm, batch lane (for prefill identity), dtypes, absolute errors, and top-token
IDs; they do not silently discard a failed arm or substitute a serial result.

```bash
python methods/attribute_switch_sweep.py \
  --n_countries 1 --attributes capital currency \
  --scopes last_token --sites attention --layers 22 \
  --block_spans --attention_spans 3 \
  --execution cached --batch_size 2 --verify_cached \
  --out_dir results/attribute_switch/cached_verify
```

Then run the focused overnight sweep **without** `--verify_cached`:

```bash
mkdir -p logs
nohup python -u methods/attribute_switch_sweep.py \
  --n_countries 8 \
  --attributes capital currency language calling_code \
  --scopes earlier_text last_token --sites attention \
  --layers 18 19 20 21 22 23 24 \
  --block_spans --attention_spans 1 2 3 4 5 6 \
  --modes prefill --execution cached --batch_size 2 \
  --out_dir results/attribute_switch/attention_18_24_cached_b2 \
  > logs/attribute_switch_attention_18_24_cached_b2.log 2>&1 &
```

Use a new output directory when changing code, execution engine, verification,
or batch size. Old serial records are not silently mixed into a cached run.
Identical commands resume at the completed-arm level, including an interrupted
batch. Records store the execution engine and actual batch size. Reducing to
`--batch_size 1` retains donor reuse and KV caching if VRAM is tight.

CPU tests cover batch sizes 1/2/4, all sites/scopes/controls, spatial image tokens,
individual EOS, interrupted resumption, and hook cleanup. They establish
mechanical equivalence; actual speed, memory use, and BF16 numerical agreement
must be measured on the pretrained GPU model. No speedup factor is assumed.

## Controlled questions and alignment

Every question shares field definitions, including three-letter currency codes
and digits-only calling codes, followed by `Report the {field} of this country.`
The fields are `capital`, `currency`, `language`, and `calling`. All use the same
assistant prefill, `Answer:`. The paraphrase control changes `Report` to `Return`
while keeping the recipient attribute. No country name or answer is in the prompt.

The processor must produce equal-length donor/recipient/paraphrase sequences
with identical final tokens, image positions and context through the image.
Vision tensors must be exactly equal. Checks run before intervening on each
pair; misalignment raises an error instead of padding or shifting tokens. The
metadata dry run cannot establish tokenizer alignment or clean model accuracy.
Clean records save all three tokenized prompts and the earlier-text positions
for inspection. Every country gets every directed pair of selected attributes.

## Intervention families

| CLI site | Donor values installed in the recipient |
|---|---|
| `earlier_text` | Residuals of all ordinary text tokens before the final prompt token; excludes image and tokenizer special tokens |
| `last_residual` | Residual at the final prompt token |
| `last_attention` | Attention output at the final prompt token |
| `last_mlp` | MLP output at the final prompt token |
| `last_joint` | Both attention and MLP outputs at that token in the same block |
| `--block_spans N` | Both sublayer outputs across N consecutive blocks ending at the selected layer |
| `--attention_spans N` | Only attention outputs across N consecutive blocks; recipient MLPs compute normally |

Use `--scopes earlier_text last_token all_text` to cross each requested site and
window with explicit position scopes. `earlier_text` excludes the final prompt
token; `all_text` includes it. Both exclude image and tokenizer special tokens.
With explicit scopes, use the generic site names `residual attention mlp joint`;
legacy site names are also accepted and reduced to their tensor type. Without
`--scopes`, old commands retain their existing position semantics. Generic site
names without scopes default to the last token. Width-1 windows and duplicate
site aliases are evaluated only once.

Layer L means after block L−1, consistent with the existing ceiling sweep. Spans
longer than L are skipped; span 1 aliases `last_joint`. All contributions in a
joint/span arm come from one unpatched donor forward and are installed together.
Only the selected tensors are changed; the recipient input IDs stay unchanged.

Prefill mode patches only the original prompt positions. Continuous mode adds
every answer position to last-token and all-text interventions. Earlier-text scope stays
fixed and is evaluated once, since both modes would perform the same operation
there. In serial execution every generation step recomputes the full prefix with
caching disabled, so previous edits remain in force. The donor sees its own question followed by
the recipient's freely generated suffix, never the gold answer or an independent
donor rollout. Cached prefill execution uses the equivalent prompt-only donor
activations and retains edited history in recipient KV caches as described above.
Gold answers are used for scoring only.

## Controls and interpretation

Each question pair gets recipient-clean and donor-clean baselines. Every selected
site also gets the requested control donors: `self` must leave logits unchanged;
`paraphrase` tests sensitivity to same-attribute wording. Self-patch logit identity
is checked at runtime, as is final-layer residual identity with the donor whenever
the current readout position is patched. These checks fail loudly.

Final-layer earlier-text replacement is a next-token null: no downstream attention
remains to communicate those edits to the final position. Conversely, successful
late final-token replacement can copy an already-computed answer; it does not
establish an abstract attribute representation. Joint block success versus failed
single sublayers can motivate finer localization, but the full-swap curve is not
a mathematical upper bound on every possible selective edit.

## Outputs

- `rows.jsonl`: generated IDs/text, full gold suffixes, first/per-token/full scores,
  country, switch direction, site, layer, scope, and control. Clean rows also save
  prompt IDs and intervention positions.
  Intervention rows include the position scope, original prompt positions, and
  exact zero-based decoder block list; continuous generated positions are added
  dynamically to last-token/all-text scopes.
- `switch_summary.json`: one cell per arm and directed switch, with donor-answer,
  original-answer, neither-answer, and donor first-token rates. Reports both all
  examples and the subset where **both clean answers are correct**; an empty
  eligible subset is null, not zero accuracy. Country counts are explicit.
- `summary.json`: the shared follow-up token/length summaries grouped by arm.
- `config.json` and `runtime.json`: exact rows, prompts, image/metadata/code hashes,
  model config, dtype and library versions, checked on resume.

Full-answer matching uses all gold suffix tokens, allowing trailing generated
text, consistent with the other head follow-ups. Insufficient generation budgets
raise an error. It is exact token matching against VADE's canonical label, not an
alias-aware semantic scorer; interpret language variants and formatting errors
using the saved decoded answers. No clean-accuracy filtering silently removes
countries from the unconditional results.

Next refinements, after identifying useful layers, are attribute-phrase-only
patching, replication on original VADE prompts with explicit span alignment,
and learned switches evaluated on unseen countries. They are not part of this
initial full-swap sweep.
