#!/usr/bin/env bash
# Both phases use all four queries, even when training only one target mask.
set -euo pipefail
cd "$(dirname "$0")/.."
phase="${1:?Expected matrix or train}"
shift
traces=()
for attribute in language capital currency calling_code; do
  traces+=("${TRACE_ROOT:-logs/Qwen2.5-VL-7B-Instruct/flags/ndm}/$attribute/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json")
done
"${PY:-python}" methods/head_attribute_edit.py "$phase" --traces "${traces[@]}" "$@"
