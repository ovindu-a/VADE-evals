#!/usr/bin/env bash
# Sweep head_trace's --blocks window at a FIXED --patch_layer.
#
# WHY --patch_layer IS FIXED AND WHY IT IS 21.
#   --patch_layer must sit BELOW the handoff, i.e. where an image-position residual swap still
#   moves the answer; above it there is no effect left to trace and every arm downstream of
#   "image patch only" is meaningless. Measured for flags/language at full_image (see
#   ceiling_sweep_..._full_image_...log, `residual` rows):
#
#       layer      21      22      23      24      25
#       cause    100.0%   87.5%   15.6%    0.0%    0.0%
#
#   So 21 is the only layer with a full-strength effect, 22 is usable, and 23+ are dead. Holding
#   it at 21 also keeps the traced effect constant across every run, which is what makes the
#   windows comparable to each other -- varying patch_layer AND the window at the same time
#   confounds window position with effect size.
#
#   Re-measure this table before reusing the script on another entity/attribute/position set.
#   PATCH_LAYER is overridable precisely so that a re-measured value can be used.
#
# ORDER. The full-coverage reference runs FIRST: it is the only run whose phase 1c identity must
# hold, so it validates the plumbing, and its phase-2 "image patch only" row is the denominator
# every windowed run is read against. Partial windows will report "diverges, blocks missing" in
# phase 1c -- correct, not a failure: the untraced blocks' attention still reads the clean image.
#
# Usage:
#   scripts/run_head_trace_sweep.sh              # run everything, skipping finished runs
#   DRY=1 scripts/run_head_trace_sweep.sh        # print the commands, run nothing
#   FORCE=1 scripts/run_head_trace_sweep.sh      # re-run even where a result JSON exists
#   N_ROWS=128 CONTROL_K="8 16" scripts/run_head_trace_sweep.sh
set -uo pipefail

PY="${PY:-python}"
ENTITY="${ENTITY:-flags}"
ATTRIBUTE="${ATTRIBUTE:-language}"
POSITIONS="${POSITIONS:-full_image}"
PATCH_LAYER="${PATCH_LAYER:-21}"
N_ROWS="${N_ROWS:-64}"
CONTROL_K="${CONTROL_K:-1 2 4 8 16 32}"
LOG_DIR="${LOG_DIR:-logs/scheduled}"
DRY="${DRY:-0}"
FORCE="${FORCE:-0}"

# Window list. Full coverage first (the reference), then 1-, 2- and 3-block windows across the
# downstream range. Every block here must be >= PATCH_LAYER: a block below it runs BEFORE the
# patch, so its captured values are identical to base and installing them is a no-op.
WINDOWS=(
  "21 22 23 24 25 26 27"
  "21" "22" "23" "24" "25" "26" "27"
  "21 22" "22 23" "23 24" "24 25" "25 26" "26 27"
  "21 22 23" "22 23 24" "23 24 25" "24 25 26" "25 26 27"
)

mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/run_head_trace_sweep.log"
say() { echo "[$(date -u '+%a %b %d %H:%M:%S UTC %Y')] $*" | tee -a "$BATCH_LOG"; }

span() {  # "21 22 23" -> "21-23";  "21" -> "21"
  local -a b=($1)
  if [ "${#b[@]}" -eq 1 ]; then echo "${b[0]}"; else echo "${b[0]}-${b[${#b[@]}-1]}"; fi
}

say "Starting head_trace window sweep: entity=$ENTITY attribute=$ATTRIBUTE positions=$POSITIONS \
patch_layer=$PATCH_LAYER n_rows=$N_ROWS control_k='$CONTROL_K' windows=${#WINDOWS[@]}"

failed=(); skipped=(); ran=()
for W in "${WINDOWS[@]}"; do
  # Every block in the window must be downstream of the patch, or its arm is a silent no-op.
  for b in $W; do
    if [ "$b" -lt "$PATCH_LAYER" ]; then
      say "SKIP blocks=$W -- block $b is below --patch_layer $PATCH_LAYER (runs before the patch)"
      continue 2
    fi
  done

  S="$(span "$W")"
  NAME="head_trace_${ENTITY}_${ATTRIBUTE}_patch${PATCH_LAYER}_${POSITIONS}_blocks${S}"
  # head_trace writes its own JSON/log under logs/<model>/<entity>/ndm/<attribute>/<config tag>/.
  # The config tag is built by ndm/config.py, so glob for it rather than reconstructing it here.
  EXISTING=$(ls logs/*/"$ENTITY"/ndm/"$ATTRIBUTE"/*attn_head_output*/head_trace_patch"${PATCH_LAYER}"_blocks"${S}".json 2>/dev/null | head -1)
  if [ -n "$EXISTING" ] && [ "$FORCE" != "1" ]; then
    say "SKIP $NAME -- already have $EXISTING (FORCE=1 to re-run)"
    skipped+=("$NAME"); continue
  fi

  CMD=("$PY" methods/head_trace.py --entity "$ENTITY" --attribute "$ATTRIBUTE"
       --patch_layer "$PATCH_LAYER" --positions "$POSITIONS" --blocks $W
       --n_rows "$N_ROWS" --control_k $CONTROL_K)
  if [ "$DRY" = "1" ]; then echo "${CMD[*]}"; continue; fi

  say "START $NAME: ${CMD[*]}"
  "${CMD[@]}" >"$LOG_DIR/$NAME.log" 2>&1
  rc=$?
  say "END $NAME (exit $rc)"
  # Keep going on failure: one bad window should not cost the other seventeen. The summary below
  # is the thing to read, not the scrollback.
  if [ "$rc" -ne 0 ]; then
    failed+=("$NAME"); say "  tail of $LOG_DIR/$NAME.log:"; tail -15 "$LOG_DIR/$NAME.log" | tee -a "$BATCH_LOG"
  else
    ran+=("$NAME")
  fi
done

[ "$DRY" = "1" ] && exit 0
say "DONE: ${#ran[@]} ran, ${#skipped[@]} skipped, ${#failed[@]} failed"
if [ "${#failed[@]}" -gt 0 ]; then
  for f in "${failed[@]}"; do say "  FAILED $f -> $LOG_DIR/$f.log"; done
  exit 1
fi
