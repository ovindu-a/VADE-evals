#!/usr/bin/env bash
# Tier-1 circuit experiments (E1-E6), sequentially, one log per step.
#
#   bash scripts/run_tier1.sh smoke              # every NEW script at tiny size (~10 min)
#   bash scripts/run_tier1.sh e1 e5 e2 e6 e4 e3  # the real runs, in the recommended order
#   bash scripts/run_tier1.sh all                # same as the line above
#
# Run inside tmux: the full set is several hours. Each step loads the model once.
# PY overrides the interpreter (default: python). Logs: logs/tier1/<step>_<stamp>.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/tier1

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/tier1/${step}_${STAMP}.log"
  echo "=== [$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

e1() {  # attributes at the image tokens? (+ E0b identity check)
  run e1 $PY methods/image_enrichment.py
}

e2() {  # do the late MLPs WRITE the answer? + their direct logit share
  run e2_severed $PY methods/head_severed.py --n_pairs 64 --freeze_values base mean other \
      --freeze_mlp_span 23-26 --freeze_mlp_singles 23 24 25 26 --freeze_attn_span ''
  for a in capital currency language calling_code; do
    run "e2_dla_$a" $PY methods/dla.py --entity flags --attribute "$a" --direction gold_minus_mean --limit 24
  done
}

e3() {  # attribute-specific lookup at the neuron level (ceiling first, then masks)
  run e3_jacobian $PY methods/readout_jacobian.py --phase all --site mlp_hidden --blocks 23-26 --max_rank 128
  for a in calling_code currency language capital; do
    run "e3_das_$a" $PY methods/head_das.py --attribute "$a" --method dbm --site mlp_hidden --blocks 23-26 \
        --l1_coef 1e-4 --temperature_start 1e-2 --temperature_end 1e-2 --train_rows 4000 --eval_rows 4000
  done
  run e3_overlap $PY methods/mask_overlap.py results/head_das/flags/*_dbm_l10.0001_T0.01-0.01_mlp23-26/mask_selected.json
}

e4() { run e4 $PY methods/circuit_compose.py; }
e5() { run e5 $PY methods/route_knockout.py; }
e6() { run e6 $PY methods/text_entity_contrast.py; }

smoke() {
  run smoke_e1 $PY methods/image_enrichment.py --identity_items 2 --probe_layers 0 16 22 --n_perm 5 \
      --out_dir results/image_enrichment/smoke
  run smoke_e4 $PY methods/circuit_compose.py --n_items 2 --out_dir results/circuit_compose/smoke
  run smoke_e5 $PY methods/route_knockout.py --n_pairs 6 --out_dir results/route_knockout/smoke
  run smoke_e6 $PY methods/text_entity_contrast.py --n_pairs 6 --layers 0 8 14 18 20 22 24 26 \
      --out_dir results/text_entity_contrast/smoke
}

[ $# -eq 0 ] && { sed -n 2,9p "$0"; exit 1; }
for step in "$@"; do
  case $step in
    all) e1; e5; e2; e6; e4; e3 ;;
    e1|e2|e3|e4|e5|e6|smoke) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
