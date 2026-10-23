# Running the molab notebooks (handoff, 2026-10-21)

This is a handoff for whichever agent runs the notebooks. Two things are shared:
- **Code:** the `molab` branch of https://github.com/ovindu-a/VADE-evals (public).
- **Data:** the public HF bucket `ovindu-a/counterfact-data`, which holds VADE `data/` plus
  `models/Qwen2.5-VL-7B-Instruct/` (the pruned tuples).

Molab only syncs the notebook file itself. The notebook clones the repo and pulls the bucket on its own.

## X5: `notebooks/x5_gate_knockout.py` (entity-gate knockout)

**Question.** An ICLR 2027 submission ("Localizing and Disrupting Entity Identification in VLMs")
reports an "entity-recognition gate" at Qwen2.5-VL-7B layers 16-17 (0-based). Is that gate really
entity recognition? Or is it an answer-side signal: abstention, or the question not being routed?

**Method.** `methods/gate_knockout.py` applies the paper's exact intervention to our flags data:
each head's `o_proj` input in the ablated blocks is replaced by its mean over the prompt's positions.

Arms:
- `clean`
- `ko:16-17`, plus control windows `ko:12-13`, `ko:19-20` and `ko:24-25`
- `ko:gate5`, the paper's five heads 16.1, 16.14, 16.17, 16.20 and 17.21
- `ko:gate5:random*`, block-matched random five-head sets

Probes:
- our 4 fact questions about the flag image;
- the paper's `known,<name>` identity contract;
- an open "which country?" question;
- a dominant-colour perceptual control;
- the same fact questions with the country **named in text** (no image).

**Links.**
- Notebook, synced from GitHub (read-only preview):
  https://molab.marimo.io/github/ovindu-a/VADE-evals/blob/molab/notebooks/x5_gate_knockout.py
- The user's fork, already made in "Ovindu Atukorala's Workspace", with the RTX Pro 6000 GPU attached:
  https://molab.marimo.io/notebooks/nb_5x6mHrXpxm1cgNTb5o82VT

  Setup (button 1) was started there and was cloning the repo when the previous session stopped.
  If the session has since shut down (90 minutes idle), just press the buttons again.

### Steps

1. Open the fork above, signed in as the user. Check that the header reads `4 CPU · 32 GiB · RTX Pro 6000`.
   If it does not, open that menu, set **GPUs** to RTX Pro 6000 (Blackwell), then click **Save and restart**.
2. If cells show no output, press the bottom-right "run all" (▶) button once.
   The notebook then waits at the buttons.
3. Press **1. Set up (clone repo + pull data)**. It does four things:
   - clones `molab` into `/tmp/x5/VADE-evals` (about 2,157 files);
   - syncs the bucket into `/tmp/x5/VADE` (513 files, 371 MB);
   - runs `gate_knockout.py --dry_run`, which must print `Grid valid.`;
   - prints the GPU name.

   **Self-check:** the cell ends with `GPU: **NVIDIA RTX PRO 6000 ...**`, not "none".
4. Leave **run size = smoke**, then press **2. Run gate_knockout**. The smoke run covers 8 flags, the
   `16-17` window only and one random draw; expect a few minutes, including the ~16 GB model download.

   **Self-checks** in the smoke output:
   - `clean` rows exist for every probe;
   - `ko:16-17` changes something compared with clean;
   - no traceback.
5. If the smoke run is sane, set **run size = full** and press **2** again. The full run covers all 84 flags,
   4 windows, gate5 and 3 random draws, about 10-20 minutes on that GPU.
6. Use the download buttons at the bottom for `x5_summary.json` and `x5_rows.jsonl`.
   Save them in this repo as `results/gate_knockout/flags_full/summary.json` and `rows.jsonl`
   (and `flags_smoke/` for the smoke run).
   - Do not commit them unless the user asks.
   - Do not write to the HF bucket: it is the user's, and is meant for data only.

### What to report back (copy the notebook's tables)

- **image / fact (pooled)** and **text / fact (pooled)**: per arm, `n`, `correct`, `refusal`,
  `other_entity`, `wrong_attr` and `other`, all as % of rows the clean model got right.
- **image/identify**, **image/identify_open**, and **image/color % changed vs clean**.
- The mover name rank (`mover_rank_mean` in `summary.json`) for `clean` vs `ko:16-17` vs `ko:gate5`.
- A verdict, using the table in the notebook's first cell:

  | pattern | reading |
  |---|---|
  | image drops, text survives, failures = other entity | visual recognition |
  | image **and** text drop, failures = refusal | answer-side abstention |
  | failures = wrong attribute | the question is not routed |

  Also give the contrast with the random-head draws and the control windows. The paper's effect should be
  much larger than either.
- Anything unexpected, verbatim.

### Troubleshooting

| Problem | Fix |
|---|---|
| `hf: command not found` or `No such command 'buckets'` | The `hf` CLI must come from `huggingface_hub>=1.24`, which the notebook's inline dependencies pin. In molab's package sidebar install `huggingface_hub>=1.24`, then rerun setup. |
| CUDA out of memory | Lower the **batch size** number in the notebook (32 → 16 → 8). |
| A transformers version error on loading Qwen2.5-VL | The notebook pins `transformers==5.16.*`, the version the pipeline was verified on. Do not upgrade. |
| A cell says "Ancestor stopped" | Normal: it is waiting for a button press. |
| A code change is needed | Edit on the `molab` branch, push, then in molab pull the synced file. Do not touch `main`: it has the user's uncommitted LLaVA / runbook work. |

### Afterwards (for the user / main session)

- Strike X5 through in PAPER_TODO.md with the result.
- Use the verdict to finish the entry on the ICLR paper in PAPER_CLAIM_AUDIT.md §4.5.
- If X5 is worth a full-scale or multi-entity run, add it to PAPER_RUNBOOK.md for the next pod batch.

## X7: `notebooks/x7_mlp_lookup.py` (how entity and question combine in the late MLPs)

**Question.** The movers write *which country* and the routers write *which attribute was asked*
into the answer column. Do last-token MLPs 23-26 then compute the fact from a plain SUM of the two?

**Method.** `methods/mlp_lookup.py` (flags, controlled prompts, last prompt column) has three phases:
- **capture:** one forward per cell of the crossed 84 x 4 grid, plus a clean generation. It stores
  the residual at layers 18-28 and the MLP neurons and output at blocks 21-27 (~265 MB of `.npy`).
- **compose:** on a base run (C, asked X), it replaces the last-column residual at layers 22-27 by a
  vector built from the grid without the target cell. Arms:
  - `actual`;
  - `additive` = rowmean(B) + colmean(A) - mean;
  - `swap_entity`;
  - `swap_question`;
  - `random_offset`.

  It reads the first answer token against the four golds B.A / C.X / B.X / C.A.
- **neurons** (CPU): the exact two-way (item x attribute) variance decomposition per layer and block,
  neuron classes, and the value test. The value test asks whether countries that share a language,
  EUR or a calling zone group together in a neuron only when that attribute is asked; it is scored
  against a permutation null. It also writes a heatmap of the top interaction neurons.

Part A of the notebook is a CPU toy model: one attention layer and one MLP trained on a lookup table.
It shows the same mechanism (a mover head, a router head, an additive residual into the MLP, and
composition and cross-question transfer at 100%) before any GPU time is spent.

**Links.**
- Notebook (read-only preview, synced from GitHub):
  https://molab.marimo.io/github/ovindu-a/VADE-evals/blob/molab/notebooks/x7_mlp_lookup.py
- To run it, fork it into the workspace and attach the RTX Pro 6000, as for X5.

### Steps
1. Press **1. Set up**. It clones `molab` into `/tmp/x7`, syncs the bucket, and runs
   `mlp_lookup.py --dry_run`. That must print `Grid valid.`
2. **Part A:** press **A. Train + analyse the toy** (CPU, about 1 min).
   **Self-checks:**
   - accuracy ~100%;
   - one head mainly attends to the entity and one to the relation;
   - the residual's interaction share is ~0;
   - compose ~100%.
3. **Part B:** leave **run size = smoke**, then press **2. Run**. Smoke covers 12 flags, 48 compose
   rows and layers 23-25. **Self-checks:**
   - no traceback;
   - `clean` reads ~100% C.X;
   - `actual` differs from `clean` at every layer (the patch is live).
4. If the smoke run looks sane, set **full** and press **2** again (84 flags, 384 rows, layers 22-27;
   about 20-40 min).
5. Download `summary.json`, `compose_rows.jsonl`, `index.jsonl` and `heatmap.npz`. Save them under
   `results/mlp_lookup/flags_full/` in this repo. Do not commit them unless the user asks.

### What to report back
- The **compose** table, per layer: each arm's first-token % on B.A / C.X / B.X / C.A / other.
  The headline is `additive` against `actual` at the same layer.
- The interaction share by depth for the residual, the MLP neurons and the MLP output. The residual
  should stay ≤ ~3.4% through layer 22 (R13.1).
- The value-test table: blocks and attributes whose diff exceeds the null's 95th percentile.
- Anything unexpected, verbatim.

| pattern | reading |
|---|---|
| `additive` ≈ `actual` ≫ `random_offset` | the lookup acts on entity + question as a sum |
| `swap_entity` → B.X and `swap_question` → C.A | the two components can be edited independently |
| `additive` ≈ clean | the late blocks need more than the sum (e.g. an interaction already built upstream) |

## Next notebooks (not written yet)

These are the next candidates in priority order. Each follows X5's pattern: a thin wrapper over an existing
script, run at smoke size first.

1. **X2b rerun with `--movers common10`** (composition and crossmodal on brands / celebrities).
   It needs VADE's `eval/score.py`, which is not in the bucket yet; ask the user first.
2. **Mean-ablation arm for the movers**, alongside `vrh_ablation`'s zero-ablation.
3. **Early/late MLP freezes on text-named entities** (`text_entity_contrast`). These separate a
   Qwen-specific effect from a modality effect.
4. **A random-site control for the trained MLP masks** (`head_das --method dbm` at, e.g., MLPs 10-13).

## X8: `notebooks/x8_llava_window.py` (LLaVA-1.5-7B: can the head window be found?)

**Question.** Before the full X4 pod run, can stage 1 find LLaVA-1.5-7B's image->text handoff window,
and can stage 2 narrow it to a few blocks? Stage 1 is the residual swap at every image position vs at the
last prompt token, at layers 0-32. Stage 2 installs every head of each window of ≤3 blocks at the last
token.

**Method.** `methods/image_head_pipeline.py --stages ceiling [windows]` on flags, with:
- `--model_id llava-hf/llava-1.5-7b-hf --n_layers 32 --n_heads 32`;
- `--scoring text` (LLaVA's casing need not match the labels);
- `--allow_unpruned --skip_scoring_check`. VADE's accuracy and pruning scripts are not in the bucket, so
  the curves are read against `source_read`, the model's own accuracy on these rows.

The setup cell also downloads LLaVA's processor files (no weights) and runs
`tests/test_llava_adapter.py`, which must show 19 passed and 0 skipped.

**Links.**
- Notebook (read-only preview, synced from GitHub):
  https://molab.marimo.io/github/ovindu-a/VADE-evals/blob/molab/notebooks/x8_llava_window.py
- To run it, fork it into the workspace and attach the RTX Pro 6000, as for X5.

### Steps
1. Press **1. Set up**. The cell ends with the GPU name and `LLaVA adapter tests: passed`.
2. Leave attributes = capital, language and rows = medium (64), then press **2. Run**. The first run
   downloads ~14 GB of weights.
3. If the curves cross cleanly, tick **stage 2** and press **2** again. Stage 1 is reused.
4. Download the zip. Save it under `results/image_head_pipeline_llava-1.5-7b-hf/` in this repo, and do not
   commit it unless the user asks.

### What to report back
- The stage-1 table per attribute: `source_read`, `patch_layer`, `read_end`, handoff blocks, relative
  depth, and warnings.
- Both curves by layer, raw and ÷ `source_read`.
- If stage 2 ran, its table and chosen window, plus the identity check.
- Compare against Qwen-7B (21-23 of 28, depth 0.75-0.82) and Idefics3 (16-24 of 32, no ≤3-block window).
