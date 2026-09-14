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
```

Defaults include self-patching and same-attribute paraphrase controls at every
site/layer. This makes the full sweep expensive: the command prints the number
of generation runs before loading the model. Narrow with `--attributes capital
currency`, `--countries FR DE`, `--sites last_residual last_joint`, or
`--block_spans 2`. Pass `--block_spans` alone for no additional spans. Keep the
same seed/countries when comparing runs. Rerun an identical command to resume;
use another output directory when changing configuration or code.

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

Layer L means after block L−1, consistent with the existing ceiling sweep. Spans
longer than L are skipped; span 1 aliases `last_joint`. All contributions in a
joint/span arm come from one unpatched donor forward and are installed together.
Only the selected tensors are changed; the recipient input IDs stay unchanged.

Prefill mode patches only the original prompt positions. Continuous mode adds
every answer position to last-token interventions. Earlier-text scope stays
fixed and is evaluated once, since both modes would perform the same operation
there. Every generation step recomputes the full prefix with caching disabled,
so previous edits remain in force. The donor sees its own question followed by
the recipient's freely generated suffix, never the gold answer or an independent
donor rollout. Gold answers are used for scoring only.

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
