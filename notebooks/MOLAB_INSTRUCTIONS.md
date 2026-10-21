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

## Next notebooks (not written yet)

These are the next candidates in priority order. Each follows X5's pattern: a thin wrapper over an existing
script, run at smoke size first.

1. **X2b rerun with `--movers common10`** (composition and crossmodal on brands / celebrities).
   It needs VADE's `eval/score.py`, which is not in the bucket yet; ask the user first.
2. **Mean-ablation arm for the movers**, alongside `vrh_ablation`'s zero-ablation.
3. **Early/late MLP freezes on text-named entities** (`text_entity_contrast`). These separate a
   Qwen-specific effect from a modality effect.
4. **A random-site control for the trained MLP masks** (`head_das --method dbm` at, e.g., MLPs 10-13).
