#!/usr/bin/env bash
# Run ONE phase across the four saved flag attributes. Identification never launches knockouts.
# Usage: bash scripts/run_head_followups.sh {decode|identify|knockout} [CLI arguments...]
# Examples:
#   ATTRIBUTES=calling_code bash scripts/run_head_followups.sh decode --limit 4
#   bash scripts/run_head_followups.sh identify --limit 4
#   bash scripts/run_head_followups.sh knockout --query_scope continuous
# Use a different OUT_ROOT when changing configuration; each output validates its provenance.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?Expected decode, identify, or knockout}"
shift
PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-results/head_followups/flags}"
TRACE_ROOT="${TRACE_ROOT:-logs/Qwen2.5-VL-7B-Instruct/flags/ndm}"
read -r -a attrs <<< "${ATTRIBUTES:-language capital currency calling_code}"
for attribute in "${attrs[@]}"; do
  trace="$TRACE_ROOT/$attribute/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json"
  case "$MODE" in
    decode)
      "$PY" methods/head_decode_trace.py --trace "$trace" --out_dir "$OUT_ROOT/$attribute/decode" "$@"
      ;;
    identify)
      "$PY" methods/head_token_trace.py identify --trace "$trace" --out_dir "$OUT_ROOT/$attribute/tokens_identified" "$@"
      ;;
    knockout)
      "$PY" methods/head_token_trace.py knockout --identification_dir "$OUT_ROOT/$attribute/tokens_identified" \
        --out_dir "$OUT_ROOT/$attribute/token_knockout" "$@"
      ;;
    *) echo "Expected decode, identify, or knockout; got $MODE" >&2; exit 2 ;;
  esac
done
