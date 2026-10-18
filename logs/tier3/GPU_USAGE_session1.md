# GPU usage: Tier 3 Session 1 (3 parallel lanes)

Snapshot taken 2026-10-18 08:40 UTC, while Session 1 was still running. Raw samples are in
`gpu_usage_session1.csv`: `nvidia-smi` every 30 s, 249 samples from 06:36 to 08:40. The
columns are timestamp, util %, mem-util %, MiB used, W, SM MHz.

## Setup

- 1x A100-SXM4-80GB (400 W power limit), 96 CPU cores, 1.1 TB RAM.
- Session 1 ran as the runbook's three lanes, all at default batch sizes (no `GPU=a100`), with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - lane 1: `p1 p4 p10`
  - lane 2: `p2 p5 p3 p11`
  - lane 3: `p6 p7 p8 p9 p12`
- Lanes 1 and 2 started at 05:55. Lane 3 started at 06:06, after a 10-minute memory check that
  peaked at 34.8 GB with two lanes running.

## Summary (06:36-08:40, three lanes running)

| metric | mean | p10 | median | p90 | max |
|---|---|---|---|---|---|
| GPU util (%) | 97.2 | 99 | 100 | 100 | 100 |
| memory used (GB) | 47.8 | 34.4 | 50.1 | 51.2 | 51.2 |
| power (W, limit 400) | 213 | 121 | 199 | 323 | 389 |
| memory-bandwidth util (%) | 21.7 | 6 | 22 | 38 | 48 |

| 30-min window | util | memory | mean power (max) |
|---|---|---|---|
| 06:30 | 99% | 50.0 GB | 194 W (389) |
| 07:00 | 99% | 49.8 GB | 225 W (360) |
| 07:30 | 96% | 50.4 GB | 210 W (368) |
| 08:00 | 95% | 46.2 GB | 211 W (380) |
| 08:30 | 100% | 34.4 GB | 235 W (362) |

## Reading

- **The GPU is busy but not saturated.** "Util" only measures the fraction of time some kernel
  is running. Mean power is about 213 W, roughly half the 400 W limit, and only 38 of 249
  samples went above 300 W. A compute-saturated A100 sits at 300-400 W.
- **Each lane is limited by its CPU thread.** Every lane's Python process holds one core at
  90-100%, while about 93% of the 96 cores are idle. The work is token-by-token generation at
  small default batches, with a hook callback on every layer. That means many small kernels
  launched one after another from one Python thread.
- **Memory is not the constraint.** Each lane uses 16.5-18 GB, about 17 GB on average. Three
  lanes peaked at 52 GB, leaving roughly 28 GB free.
- **Lanes do increase throughput, by about 1.5x, not 3x.** Running the same work alone and with
  three lanes: `dbm/eval.py --no_source_cache`, 3,036 rows, P1, took 7 min alone and 14 min with
  three lanes. Each step runs at about half speed, while three run at once.

## Step wall times so far (three lanes, default batch)

FOLLOWUP_PLAN buckets are for an A100 running one job: XS under 10 min, S 10-30, M 30-90.
Observed times here are about 2-4x the bucket.

| step | plan | observed |
|---|---|---|
| p1 (4 dbm evals) | S | 4 + 7 alone, then 1 (resume) + 14 with lanes |
| p2_vrh_mover_random | XS | 20 min |
| p5_head_cross (+ CPU bootstrap) | XS | 21 min |
| p3_trace_* (x4) | (M total) | 4-5 min each |
| p3_install | (in M) | running; about 4,500 of 14,052 generations after about 60 min |
| p4_gate05_* (x4, 4,000 rows each) | M | 20-22 min each, 84 min total |
| p6_sel_A, p6_eval_B | S (whole P6) | 42 and 43 min |
| p10_severed (first sub-run) | M | 42 min |

## Implications for Session 2 (input for whoever plans it)

- Memory would fit 4 lanes at default batch (about 4 x 17 GB, roughly 68 GB). The limit is
  per-process CPU dispatch, so each extra lane should still add throughput, though less for
  each lane added, because all lanes share the GPU's compute.
- `GPU=a100` as a single job uses larger batches and therefore does more work per kernel
  launch. That helps steps whose batch it raises (vrh, ceiling, severed, tec, route, compose,
  das). It does nothing for `head_trace`, `head_window_sweep` and `attr_head_trace`, which keep
  their default batch, or for batch-1 steps.
- Not measured: a single `GPU=a100` job's power and memory profile. The first session1 attempt
  ran that way for only about 20 minutes, entirely within P1, which passes no batch flag. So
  there is no a100-profile data point yet.
