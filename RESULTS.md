# Kaggle-pool SAE reproduction run — flags entity (2026-09-08)

This documents a full run of the README's ["Reproducing results on a fresh
machine from the Kaggle-pool SAE checkpoints"](README.md) runbook (steps
0-8), done on a fresh RunPod GPU box, plus what had to be fixed along the
way and what the actual numbers came out to.

## What was done

### Environment setup
- Cloned `VADE` and `VADE-evals` as siblings, installed
  `transformers>=4.49`, `tqdm`, `accelerate`, `scikit-learn`.
- The box's preinstalled `torch 2.4.1+cu124` is too old for
  `transformers 5.16.1` (requires torch>=2.5) — upgraded to
  `torch==2.13.0+cu126` / `torchvision==0.28.0+cu126` (matching the
  README's pinned versions) via `--index-url
  https://download.pytorch.org/whl/cu126`. Dropped `torchaudio` rather
  than chase a matching build — nothing in either repo imports it.
- The root filesystem here is a small 20GB overlay (not the 407TB
  `/workspace` mount) — redirected `HF_HOME` to `/workspace/fyp/.hf_cache`
  before downloading model weights, after a first attempt filled the root
  disk and crashed mid-download. (2026-09-11: this run had left behind a
  second, incomplete cache at the container's preset `HF_HOME`, plus a third
  stray one here — all three got consolidated into a single sibling
  `hf_home/` next to `VADE/`/`VADE-evals/`, matching the `VADE_ROOT`
  convention; see README's step 4. `.hf_cache` no longer exists.)

### Step 3 — Kaggle-pool SAE checkpoints via rclone
- `rclone authorize "drive"` was run by the user on a browser-capable
  machine; the resulting OAuth token was used to `rclone config create`
  a `gdrive-transfer` remote here.
- The README's suggested restart-loop (`while ! timeout 90 rclone copy
  ...; do sleep 3; done`) **corrupted 5 of the 59 checkpoint files** —
  the hard `timeout 90` kill lands mid-write on Drive's chunked download
  path without going through rclone's normal temp-file-then-rename
  protection, leaving a truncated file at the final name rather than a
  `.partial` file. Caught via `rclone check` (`Sizes differ` on 5
  files); fixed by killing the loop and running one clean, uninterrupted
  `rclone copy` pass. Final state verified: **59/59 files, 11.104 GiB,
  0 differences** against the Drive source.

### Step 4 — model weights
- Pre-fetched `Qwen/Qwen2.5-VL-7B-Instruct` (16GB, 16 files) into the
  redirected `HF_HOME`.

### Step 5 — re-extract real flag activations
- `python methods/sae.py --entity flags` →
  `methods/activations/flags_Qwen2.5-VL-7B-Instruct_all_layers.pt`
  (1.1GB, shapes `flag_only: (84,29,8,3584)`, `flag_ring1: (84,29,24,3584)`).

### Step 6 — feature selection against the Kaggle-pool SAE
- Ran both token sets (`flag_only`, `flag_ring1`) against
  `--dictionaries_dir methods/dictionaries_kaggle_pool`, sweeping all 29
  fitted layers (vs. the 4/14/24 subset in the basic real-only example).
- Added a one-line, easily-reverted progress instrumentation to
  `select_features.py` (`[progress] layer=... done (i/N, rows so far)`,
  printed with `flush=True` at the end of each layer's inner loop) since
  the script otherwise prints nothing until the entire 29-layer sweep
  finishes — no incremental progress or per-layer file writes exist
  upstream of that patch.
- **Caveat:** `select_features.py`'s default output path
  (`selections/<entity>/<token_set>_<dict_method>/`) doesn't encode
  which `--dictionaries_dir` was used, so this run **overwrote** the
  previously-committed real-only-SAE selection results for both token
  sets. They're recoverable from git history (commit `49cf0b2`, e.g.
  `git show 49cf0b2:methods/selections/flags/flag_only_sae/winners.json`)
  — real-only `flag_only`'s winner was layer=24, forward, C=0.1,
  119 features. Use `--output_dir` to avoid this collision on a repeat.

### Step 7 — intervention
- Ran `intervene.py` for both token sets against the Kaggle-pool
  dictionaries. `flag_only` and `flag_ring1`'s intervention runs were
  chained (not parallelized) since, unlike the CPU-only selection step,
  both load the full 7B model onto the same GPU — `flag_only` alone hit
  99% GPU util / 18.7GB of 24.5GB VRAM, leaving no headroom for a second
  concurrent model load. The chain (`pgrep`-poll then launch) was run as
  a plain shell script inside `tmux` rather than tied to the Claude
  session, so it would survive a session disconnect.
- 14,052 predictions written per token set (only `language` had a
  winner from step 6 — see below for why).

### Step 8 — scoring
- Scored both against `VADE/eval/score.py --attribute all`.

## Why only `language` ever gets a winner

Verified directly against `VADE/data/flags/ground_truth.json` (84
countries): `capital` and `calling_code` are unique per country (0
classes with 2+ members), `currency` has exactly one repeated class
(EUR × 10, still leaves only 1 class after dropping singletons — not
enough for a held-out split), and only `language` has real repetition
(8 classes with 2+ members, 49 of 84 rows). `select_features.py`'s
`cv_probe` requires ≥2 classes with ≥2 members to do a stratified
held-out split at all, so `capital`/`calling_code`/`currency` are
reported "no valid combination scored" by construction, not from a bug
— this matches the README's own note.

## Results

### Feature-selection winners (proxy score, held-out CV on encoded features)

| token_set | attribute | layer | direction | C | n_features | cause | iso | combined |
|---|---|---|---|---|---|---|---|---|
| flag_only | language | 23 | inverse | 1.0 | 170 | 0.755 | 1.000 | 0.878 |
| flag_ring1 | language | 4 | forward | 0.3 | 108 | 0.735 | 1.000 | 0.867 |

`iso=1.000` on every layer for both token sets is **not** a real "zero
leakage" finding — the other three attributes can never be scored (see
above), so the leakage list is always empty and `iso_score = 1 -
mean([]) = 1.0` unconditionally. The sweep is effectively only
discriminating on `cause_score` here.

### Real generation-time intervention scores (`eval/score.py`, single winning layer only)

| token_set | layer | n | overall acc | cause | mean(iso) | **final_score** |
|---|---|---|---|---|---|---|
| flag_only | 23 | 14,052 | 73.1% | 2.2% | 94.7% | **48.5%** |
| flag_ring1 | 4 | 14,052 | 54.0% | 19.2% | 64.6% | **41.9%** |

`flag_only` iso breakdown: calling_code 99.5%, capital 100.0%, currency
84.6% (all "matches_base", i.e. the intervention barely disturbs these).
`flag_ring1` iso breakdown: calling_code 64.8%, capital 72.2%, currency
56.9%.

### The key finding: proxy cause massively overstates real cause

The feature-selection proxy (a classifier reading the target attribute
off the dictionary-encoded features) scored ~0.74-0.76 `cause` for both
token sets' winning layer. The real generation-time intervention —
actually patching those features into the forward pass and letting the
model generate — only flips the model's stated language **2.2%**
(flag_only) and **19.2%** (flag_ring1) of the time. The features are
linearly separable enough for a classifier to decode the attribute, but
patching them in doesn't reliably steer what the model *says*. This is
a real property of this SAE/patching method on this task, not a
pipeline bug — full per-layer proxy tables are in
`methods/selections/flags/{flag_only,flag_ring1}_sae/sweep.jsonl` if
you want to look for a layer where the proxy/real gap is smaller.

`flag_ring1` (24 image-token positions, the wider ring) noticeably
outperforms `flag_only` (8 positions) on real `cause` (19.2% vs 2.2%)
but underperforms on `iso` (64.6% vs 94.7%) — i.e. patching more tokens
makes the intervention actually take effect more often, at the cost of
disturbing other attributes more.

## Not yet done

- **Real-only (non-Kaggle-pool) SAE comparison** — the README's
  suggested next step, to see whether the Kaggle augmentation pool
  actually helped *causal* intervention accuracy or just unsupervised
  reconstruction quality. Needs steps 6-8 rerun with
  `--dictionaries_dir methods/dictionaries` (default, no `_kagglepool`
  suffix) and — this time — a distinct `--output_dir` so it doesn't
  clobber these results the way it clobbered the original real-only run
  (see caveat above; the original real-only winner is only in git
  history now, not re-derivable without rerunning `fit_dictionaries.py`
  against the small real-only activation set).
- **DAS baseline comparison** — the README points at VADE's own
  `results/` directory for this, but that directory is gitignored and
  isn't present in this checkout (no run has produced it here yet).
- Real per-layer intervention scores beyond the single winning layer
  (would need `intervene.py` pointed at each candidate layer manually —
  expensive, ~14k rows per layer per token_set).
