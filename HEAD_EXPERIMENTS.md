# Head-trace follow-up experiments

These experiments use a saved `head_trace` ranking without fitting or re-ranking
heads. They address two questions: whether continuing head substitution fixes
multi-token answers, and which prompt-token reads of the selected heads matter.
Both use the same source-image residual patch as `head_trace`, with the same
patch layer and image positions from its JSON.

The launcher defaults to flags' language, capital, currency, and calling_code
traces at image patch layer 21 / blocks 21–23. Run it from the repo checkout on
the GPU machine with the same Python environment as the existing experiments.
The new code needs torch, transformers and Pillow; loading Qwen's AutoProcessor
also needs the existing torchvision dependency. It does not need pyvene.

## Experiment 1: continue substitution during answer generation

```bash
# Validate trace/tuples/image paths locally without loading a model.
PY=python3 bash scripts/run_head_followups.sh decode --limit 4 --dry_run

# Run all four attributes, initially on four rows each.
bash scripts/run_head_followups.sh decode --limit 4

# Or focus on calling code.
ATTRIBUTES=calling_code bash scripts/run_head_followups.sh decode --limit 4
```

`methods/head_decode_trace.py` compares:

| Arm | Image patch on recipient | Head values supplied | Positions substituted |
|---|---|---|---|
| Clean | No | None | None |
| Image patch | Yes | None | Image positions |
| Sufficiency / prefill | No | Image-patched donor | Last prompt position |
| Sufficiency / continuous | No | Image-patched donor | Last prompt position and every generated position |
| Necessity / prefill | Yes | Clean donor | Last prompt position |
| Necessity / continuous | Yes | Clean donor | Last prompt position and every generated position |

The default head sizes are 8 and 16. Random head sets of each size are evaluated
in the same directions and modes (`--head_ks 8`, `--random_repeats 3`, and
`--skip_necessity` adjust these). The random sets are fixed across rows/modes.
An additional **all-downstream continuous** arm includes every head from the
patch layer through the final block. It checks numerical logit agreement with
the image-patched donor at every answer step and records generation agreement.

**Donor alignment:** before predicting answer token j, the donor runs on the
recipient's actual generated prefix through j−1. Its activations are captured at
the matching positions and installed into the recipient. The first-token donor
vector is never repeated at later positions, and no gold answer is fed to either
forward pass. Necessity uses the same prefix alignment with a clean donor.

These runs recompute the full prefix with `use_cache=False`, reapplying earlier
patches, so changing the scope does not leave stale unpatched KV entries behind.
This is deliberately more expensive than cached generation. Each row is run
individually without padding. The prefill arm is checked against cached
generation in the offline tests. All arms use greedy raw-logit decoding and the
same eager attention backend; they do not apply extra generation logits processors.

Outputs include first-token accuracy, the existing up-to-three-token match,
**full uncapped gold-token match**, per-position accuracy, results grouped by
gold answer length, and per-step gold-token log probabilities. Gold-token scores
are read after the forward pass; they do not affect generation. Later-token
probabilities are conditioned on the arm's own generated prefix, not on gold.
An answer matches when its generated prefix matches the gold suffix; extra
explanation after the gold answer is not penalized. `--max_new_tokens` defaults
to 12, and rows whose gold is longer than that budget are explicitly counted.
First-token accuracy is also reported on rows whose base/source first tokens
differ, so shared leading digits do not masquerade as successful transfer.

Evidence for the continuation hypothesis would be similar first-token scores
but higher later-token/full-answer accuracy in continuous versus prefill arms,
especially among longer answers. Failure to improve would leave head coverage
and other information paths as possibilities; it would not establish that the
model lacks the attribute. A large restoration effect can reflect disrupted
generation as well as clean base-answer recovery, so read both source and base
scores.

## Experiment 2, phase 1: identify tokens for each head

```bash
bash scripts/run_head_followups.sh identify --limit 4
```

`methods/head_token_trace.py identify` records the top eight heads independently:

- Exact softmax attention probabilities for every key position, in both clean
  and image-patched runs, at the last prompt query and subsequent answer queries.
- Token IDs/text and absolute sequence positions; image keys additionally have
  their image-relative index and grid row/column, with object/background labels.
- Attention mass by group, including generated answer context in later steps.
- Per-key `||A[q,k] V[k] W_O_head||`, the magnitude of the direct residual
  contribution. This is distinct from attention probability and from the original
  head-trace `delta_resid` ranking (which measures a head's output change).

Inspect `results/head_followups/flags/<attribute>/tokens_identified/tokens.md`.
The companion `identification.json` freezes a **separate ranking for each head
and row**, taken from the image-patched last prompt query. Raw maps for all
generation steps remain in `rows.jsonl`. Phase 1 does not run any knockouts.

`--head_k`, `--key_scope all|image|object|background|text`, and
`--token_rank_by attention|contribution` control what is identified. The default
scope includes special tokens, since attention sinks may be informative. Phase 2
targets saved **prompt** positions; generated-context attention is described but
is not assigned a fixed knockout identity across diverging answer rollouts.

## Experiment 2, phase 2: knock out the identified token reads

Run this separately after inspecting phase 1:

```bash
bash scripts/run_head_followups.sh knockout

# Alternatively, keep the token-edge removals active at every answer query.
# Use a separate output root/directory for a different configuration (see below).
python methods/head_token_trace.py knockout \
  --identification_dir results/head_followups/flags/calling_code/tokens_identified \
  --query_scope continuous \
  --out_dir results/head_followups/flags/calling_code/token_knockout_continuous
```

Phase 2 consumes `identification.json` and the copied trace. It verifies the
tuples and tokenization still match the saved absolute positions. It never
re-ranks tokens after a removal. All arms retain the image residual patch:

1. No knockout, required to reproduce phase 1's image-patched generation.
2. Each of the top three tokens **individually for each head**.
3. Cumulative removal of the top 1, 2, 4, and 8 keys **per head**, jointly across
   the selected heads. Different heads may have different top keys.
4. Matched random-key removals from the same candidate scope, with nested
   random sets across K. `--random_repeats` adds independent controls.

`--per_head_curves` additionally runs cumulative and random curves for each head
alone. `--single_top_n` and `--knockout_ks` control the sweep. The default
`--query_scope prefill` removes edges at the final prompt query only;
`continuous` applies the same frozen prompt-key removals to every answer query.

**What zero means:** set the selected post-softmax attention edges to zero,
without redistributing their probability mass. The token stays in the input and
other heads can still read it. Implementation subtracts the selected `A V W_O`
contributions from the attention output before the residual addition, which is
algebraically equivalent to modifying the weights. An optional
`--ablation_mode renormalize` normalizes the surviving weights; it is a separate
experimental condition. Removing every available key yields zero output for
that head rather than NaNs.

On every forward, `A @ V` must reconstruct the actual pre-projection head output,
including grouped-query value-head mapping. The offline tests also compare the
ablation to a direct edit of Qwen's actual eager attention weights. Each result
records exactly which head/key edges were removed, the generated answer, token
accuracy, and log probabilities. Read the paired first-source-token log-probability
drop even when greedy text stays unchanged; compare cumulative curves with the
matched random removals. A null knockout does not exclude redundant paths.

## Reproduction, resumption, and larger samples

Each directory stores `config.json`, `runtime.json`, append-and-resume
`rows.jsonl`, and `summary.json`. Completed row/arm pairs are skipped. Different
trace contents, source code, row samples, parameters, or runtimes are rejected
when reusing a directory. Standalone CLIs choose a config-hashed directory if
`--out_dir` is omitted. The launcher's stable directory names are for resuming
the same command. Keep smoke tests and full runs in separate roots:

```bash
# No --limit means the original trace's full sample size (64 for these traces).
OUT_ROOT=results/head_followups/flags_full bash scripts/run_head_followups.sh decode
OUT_ROOT=results/head_followups/flags_full bash scripts/run_head_followups.sh identify
OUT_ROOT=results/head_followups/flags_full bash scripts/run_head_followups.sh knockout
```

`--sample_seed` changes the evaluation sample while keeping the original ranking
fixed. A different random sample may overlap the original; it is not a guaranteed
held-out split. Phase 2 inherits phase 1's rows and generation budget. Neither
experiment measures attribute isolation, and neither retrains the head ranking.

Offline validation (a temporary CPU environment is sufficient):

```bash
python -m pytest -q tests/test_head_followups.py
```

The tests use a tiny randomly initialized Qwen2.5-VL with actual image tensors,
grouped-query attention, and several generated positions. They validate mechanics,
not the scientific outcome on the pretrained 7B model.
