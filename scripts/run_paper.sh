#!/usr/bin/env bash
# Paper experiments (PAPER_RUNBOOK.md X-numbers), one log per step.
#
#   Batch 2 (X2b, X3, X2 cross-question rerun):
#   bash scripts/run_paper.sh smoke2                 # every batch-2 step at tiny size (X3's downloads the model)
#   GPU=a100 bash scripts/run_paper.sh b2_lane1      # X3: Idefics3 (scripts/run_x3_idefics3.sh all)
#   GPU=a100 bash scripts/run_paper.sh b2_lane2      # X2b brands, then the brands cross rerun
#   GPU=a100 bash scripts/run_paper.sh b2_lane3      # X2b celebrities, then the celebrities cross rerun
#
#   Batch 1 (X1, X2; ran 2026-10-19, RUN_REPORT_PAPER.md):
#   bash scripts/run_paper.sh smoke                  # every batch-1 step at tiny size, plus the head-list check
#   GPU=a100 bash scripts/run_paper.sh b1_lane1      # (was lane1; lane1..3 still work)
#   bash scripts/run_paper.sh x2_score               # CPU only, after the batch-1 lanes
#   bash scripts/run_paper.sh x1 x2_cross_brands     # any steps by name
#
# Run inside tmux. PY overrides the interpreter (default: python). Logs: logs/paper/<step>_<stamp>.log
# Every step writes under results/paper/, never over a committed file.
# GPU=a100 (any 80GB card) raises batch sizes; that only changes how many rows share a forward pass.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VADE=${VADE_ROOT:-../VADE}
T=$VADE/models/Qwen2.5-VL-7B-Instruct
STAMP=$(date +%Y%m%d_%H%M%S)
R=results/paper
P=results/image_head_pipeline
mkdir -p logs/paper "$R"

BS_TEC= BS_SWAP= BS_CROSS= BS_SEV= BS_CC=
if [ "${GPU:-}" = a100 ]; then
  BS_TEC=64 BS_SWAP=64 BS_CROSS=64 BS_SEV=64 BS_CC=64
  echo "GPU=a100 batch profile: tec=$BS_TEC swap=$BS_SWAP cross=$BS_CROSS severed=$BS_SEV compose=$BS_CC"
fi
bs() { [ -n "$1" ] && echo "--batch_size $1"; return 0; }   # bs <value> -> flag or nothing

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/paper/${step}_${STAMP}.log"
  local prev="" x
  for x in "$@"; do   # head_cross opens --out without creating its directory
    [ "$prev" = "--out" ] && mkdir -p "$(dirname "$x")"
    prev=$x
  done
  echo "=== [$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

top8() {  # top8 <entity> <attribute> -> "23.3,21.4,..." off that trace's phase-1 ranking
  $PY - "$P" "$@" <<'EOF'
import glob, json, sys
root, entity, attribute = sys.argv[1:4]
paths = sorted(glob.glob(f"{root}/{entity}/{attribute}/head_trace_*.json"))
assert paths, f"no head_trace JSON under {root}/{entity}/{attribute} -- run image_head_pipeline first"
print(",".join(f"{b}.{h}" for b, h in json.load(open(paths[0]))["phase1_ranked"][:8]))
EOF
}

COMMON10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17
C6=21.1,22.15,22.17,23.3,23.4,23.6          # shared by all three entity top-8s
# Expected head lists (2026-10-19 audit). Any drift means the traces changed under us: stop.
BH_EXPECT=23.3,21.4,22.17,23.4,23.6,22.15,23.17,21.1
CN_EXPECT=23.3,22.17,23.4,23.6,21.4,22.15,21.5,21.1
CL_EXPECT=22.17,23.4,21.1,23.6,23.3,22.19,22.15,19.6

heads() {  # sets BH/CN/CL and refuses to continue if they differ from the expected lists
  BH=$(top8 brands hq_country); CN=$(top8 celebrities nationality); CL=$(top8 celebrities native_language)
  local ok=1
  [ "$BH" = "$BH_EXPECT" ] || { echo "HEAD CHECK FAILED: BH=$BH, expected $BH_EXPECT"; ok=0; }
  [ "$CN" = "$CN_EXPECT" ] || { echo "HEAD CHECK FAILED: CN=$CN, expected $CN_EXPECT"; ok=0; }
  [ "$CL" = "$CL_EXPECT" ] || { echo "HEAD CHECK FAILED: CL=$CL, expected $CL_EXPECT"; ok=0; }
  [ $ok = 1 ] || exit 1
  echo "head check ok: BH=$BH CN=$CN CL=$CL"
}

# ---------------------------------------------------------------- X1: mover content, image vs text

x1() {
  run x1_crossmodal $PY methods/text_entity_contrast.py --phases crossmodal $(bs "$BS_TEC") \
      --out_dir results/text_entity_contrast/flags_crossmodal
}

# ---------------------------------------------------------------- X2: brands + celebrities
# The celebrities install is ~2x any other job (2,308 generations per arm vs 1,394 for brands), so
# it is split by head set over two lanes, into two directories (each arm is its own .jsonl).
# Each invocation draws its random-head nulls from its own seed-0 RNG, in head-set order.

SEV_FLAGS=(--n_pairs 128 --gold own --freeze_values base mean other --freeze_mlp_span 22-27
           --freeze_mlp_singles 23 24 25 26 --freeze_attn_span 24-27)

x2_swap_brands() {
  heads
  run x2_swap_brands $PY methods/head_swap_vade.py --entity brands --tuples_dir $T/brands/tuples \
      --head_sets "hq_top8=$BH" "cons6=$C6" "common10=$COMMON10" \
      --arms clean heads random_heads full_image --patch_layer 19 --limit_pairs 128 $(bs "$BS_SWAP") \
      --out_dir $R/x2_swap_brands
}

x2_swap_celeb_a() {   # clean + full_image + the two attribute top-8s
  heads
  run x2_swap_celeb_a $PY methods/head_swap_vade.py --entity celebrities --tuples_dir $T/celebrities/tuples \
      --head_sets "nat_top8=$CN" "lang_top8=$CL" \
      --arms clean heads random_heads full_image --patch_layer 19 --limit_pairs 128 $(bs "$BS_SWAP") \
      --out_dir $R/x2_swap_celebrities_a
}

x2_swap_celeb_b() {   # the cross-entity sets: the 6 shared heads and flags' common10
  run x2_swap_celeb_b $PY methods/head_swap_vade.py --entity celebrities --tuples_dir $T/celebrities/tuples \
      --head_sets "cons6=$C6" "common10=$COMMON10" \
      --arms heads random_heads --limit_pairs 128 $(bs "$BS_SWAP") \
      --out_dir $R/x2_swap_celebrities_b
}

x2_cross_brands() {   # seed 0, n<=64: larger samples draw AIRBUS (no industry label), which crashes head_cross
  heads
  local al
  for al in matched step0; do
    run x2_cross_brands_$al $PY methods/head_cross.py --entity brands --heads "$BH" --n_pairs 64 --seed 0 \
        --align $al $(bs "$BS_CROSS") --out $R/x2_cross_brands/cross_n64_$al.jsonl
  done
}

x2_cross_celeb() {
  heads
  local al
  for al in matched step0; do
    run x2_cross_celeb_$al $PY methods/head_cross.py --entity celebrities --heads "$CN" --n_pairs 64 --seed 0 \
        --align $al $(bs "$BS_CROSS") --out $R/x2_cross_celebrities/cross_n64_$al.jsonl
  done
}

x2_severed_brands() {
  heads
  run x2_severed_brands $PY methods/head_severed.py --entity brands --attributes founded_year industry \
      --heads "$BH" "${SEV_FLAGS[@]}" $(bs "$BS_SEV") --out_dir $R/x2_severed_brands_year_industry
}

x2_severed_celeb() {
  heads
  run x2_severed_celeb $PY methods/head_severed.py --entity celebrities --attributes occupation birth_year \
      --heads "$CN" "${SEV_FLAGS[@]}" $(bs "$BS_SEV") --out_dir $R/x2_severed_celebrities_occ_year
}

x2_score() {   # CPU only: every install arm, per attribute, against the same pruned tuples
  local e d f a
  for e in brands celebrities; do
    for d in $R/x2_swap_${e}*; do
      [ -d "$d" ] || continue
      for f in "$d"/*.jsonl; do
        for a in $(ls $T/$e/tuples); do
          run x2_score_${e} $PY $VADE/eval/score.py --predictions "$f" --entity $e --attribute $a \
              --tuples_dir $T/$e/tuples --out_dir "$d/scores/$(basename "$f" .jsonl)/$a"
        done
      done
    done
  done
}

# ---------------------------------------------------------------- X2 rerun: cross-question, own golds
# Batch 1's head_cross scored against stored lowercase labels (self cell read 81%) and used each
# entity's top-8. This reruns it the way the flags result was made: common10, golds = the model's own
# answers (--gold own), base question restricted to the entity facts a transfer can show up in, and a
# larger n because two items sharing an answer (61/128 brands are US-headquartered) carry no signal.

x2r_cross_brands() {
  heads
  local al
  for al in matched step0; do
    run x2r_cross_brands_c10_$al $PY methods/head_cross.py --entity brands --heads "$COMMON10" --gold own \
        --q1_attributes hq_country --n_pairs 256 --seed 0 --align $al $(bs "$BS_CROSS") \
        --out $R/x2r_cross_brands/common10_n256_own_$al.jsonl
  done
  run x2r_cross_brands_top8_matched $PY methods/head_cross.py --entity brands --heads "$BH" --gold own \
      --q1_attributes hq_country --n_pairs 256 --seed 0 --align matched $(bs "$BS_CROSS") \
      --out $R/x2r_cross_brands/top8_n256_own_matched.jsonl
}

x2r_cross_celeb() {
  heads
  local al
  for al in matched step0; do
    run x2r_cross_celeb_c10_$al $PY methods/head_cross.py --entity celebrities --heads "$COMMON10" --gold own \
        --q1_attributes nationality native_language --n_pairs 128 --seed 0 --align $al $(bs "$BS_CROSS") \
        --out $R/x2r_cross_celebrities/common10_n128_own_$al.jsonl
  done
  run x2r_cross_celeb_top8_matched $PY methods/head_cross.py --entity celebrities --heads "$CN" --gold own \
      --q1_attributes nationality native_language --n_pairs 128 --seed 0 --align matched $(bs "$BS_CROSS") \
      --out $R/x2r_cross_celebrities/top8_n128_own_matched.jsonl
}

# ---------------------------------------------------------------- X2b: routers, compose, text contrast
# One entity per lane; each stage needs the previous one's output. The trace window comes from the
# sweep by a fixed rule (attribute_switch_report.py --window, which reproduces flags' 15 / 15-22).

x2b_entity() {  # x2b_entity <entity> <movers> <sweep items> <trace items>
  local e=$1 mv=$2 ns=$3 nt=$4
  run x2b_switch_$e $PY methods/attribute_switch_sweep.py --entity $e --n_countries $ns \
      --sites residual attention --scopes earlier_text last_token --block_spans --attention_spans 3 \
      --execution cached --batch_size 64 --out_dir $R/x2b_switch_$e
  local PATCH="" BLOCKS=""
  # --window raises when the sweep shows no handoff; that must not stop the lane (set -e), since the
  # text contrast and the cross rerun do not depend on it.
  if run x2b_switch_report_$e $PY methods/attribute_switch_report.py $R/x2b_switch_$e --window \
      --json_out $R/x2b_switch_$e/report.json; then
    eval "$(grep -h '^PATCH=' logs/paper/x2b_switch_report_${e}_${STAMP}.log | tail -n 1)"
  fi
  if [ -z "$PATCH" ]; then
    echo "x2b $e: no trace window (self-check 2) -- skipping trace and compose; running the text contrast"
  else
    echo "x2b $e: PATCH=$PATCH BLOCKS=$BLOCKS"
    # shellcheck disable=SC2086
    run x2b_attr_trace_$e $PY methods/attr_head_trace.py --entity $e --patch_layer $PATCH --blocks $BLOCKS \
        --n_countries $nt --out $R/x2b_attr_trace_$e.json
    run x2b_compose_$e $PY methods/circuit_compose.py --entity $e --movers "$mv" \
        --router_trace $R/x2b_attr_trace_$e.json --n_items 64 $(bs "$BS_CC") --out_dir $R/x2b_compose_$e
  fi
  run x2b_tec_$e $PY methods/text_entity_contrast.py --entity $e --movers "$mv" --n_pairs 128 --batch_size 48 \
      --out_dir $R/x2b_tec_$e
  run x2b_crossmodal_$e $PY methods/text_entity_contrast.py --entity $e --movers "$mv" --phases crossmodal \
      --n_pairs 128 $(bs "$BS_TEC") --out_dir $R/x2b_crossmodal_$e
}

x2b_brands() { heads; x2b_entity brands "$BH" 24 128; }
x2b_celeb() { heads; x2b_entity celebrities "$CN" 16 84; }

x3() {  # X3 has its own launcher (logs logs/paper/x3_*); GPU passes through
  bash scripts/run_x3_idefics3.sh all
}

smoke2() {
  heads
  local S=$R/smoke2 e mv
  for e in brands celebrities; do
    if [ $e = brands ]; then mv=$BH; else mv=$CN; fi
    run smoke_x2b_switch_$e $PY methods/attribute_switch_sweep.py --entity $e --n_countries 2 --layers 8 16 24 28 \
        --sites residual --scopes earlier_text last_token --block_spans --execution cached --batch_size 64 \
        --out_dir $S/x2b_switch_$e
    run smoke_x2b_report_$e $PY methods/attribute_switch_report.py $S/x2b_switch_$e
    run smoke_x2b_trace_$e $PY methods/attr_head_trace.py --entity $e --patch_layer 15 --blocks 15 16 \
        --n_countries 2 --knockout_ks 1 2 --n_random 2 --control_k 2 --out $S/x2b_trace_$e.json
    run smoke_x2b_compose_$e $PY methods/circuit_compose.py --entity $e --movers "$mv" \
        --router_trace $S/x2b_trace_$e.json --router_k 4 --n_items 2 --out_dir $S/x2b_compose_$e
    run smoke_x2b_tec_$e $PY methods/text_entity_contrast.py --entity $e --movers "$mv" --n_pairs 6 \
        --layers 0 8 14 18 20 22 24 26 --out_dir $S/x2b_tec_$e
    run smoke_x2b_crossmodal_$e $PY methods/text_entity_contrast.py --entity $e --movers "$mv" \
        --phases crossmodal --n_pairs 8 --out_dir $S/x2b_crossmodal_$e
  done
  run smoke_x2r_cross $PY methods/head_cross.py --entity brands --heads "$COMMON10" --gold own \
      --q1_attributes hq_country --n_pairs 4 --seed 0 --out $S/x2r_cross/cross_n4_own.jsonl
  bash scripts/run_x3_idefics3.sh smoke
}

# ---------------------------------------------------------------- smoke

smoke() {
  heads
  local S=$R/smoke
  run smoke_x1 $PY methods/text_entity_contrast.py --phases crossmodal --n_pairs 4 --out_dir $S/x1
  run smoke_x2_swap $PY methods/head_swap_vade.py --entity brands --tuples_dir $T/brands/tuples \
      --head_sets "hq_top8=$BH" --arms clean heads --limit_pairs 4 --out_dir $S/x2_swap
  run smoke_x2_cross $PY methods/head_cross.py --entity celebrities --heads "$CN" --n_pairs 4 --seed 0 \
      --out $S/x2_cross/cross_n4.jsonl
  run smoke_x2_severed $PY methods/head_severed.py --entity brands --attributes founded_year --heads "$BH" \
      --n_pairs 8 --gold own --freeze_values base --freeze_mlp_span 23-26 --freeze_attn_span '' \
      --out_dir $S/x2_severed
}

[ $# -eq 0 ] && { sed -n 2,19p "$0"; exit 1; }
for step in "$@"; do
  case $step in
    # Batch 2 (PAPER_RUNBOOK.md section 2): one model chain per lane.
    b2_lane1) x3 ;;
    b2_lane2) x2b_brands; x2r_cross_brands ;;
    b2_lane3) x2b_celeb; x2r_cross_celeb ;;
    # Batch 1, balanced by generation count (~13-14k each). lane1..3 kept as aliases.
    b1_lane1|lane1) x2_swap_celeb_a ;;
    b1_lane2|lane2) x2_swap_brands; x2_cross_brands; x2_severed_brands ;;
    b1_lane3|lane3) x2_swap_celeb_b; x2_cross_celeb; x2_severed_celeb; x1 ;;
    smoke|smoke2|x1|x2_swap_brands|x2_swap_celeb_a|x2_swap_celeb_b|x2_cross_brands|x2_cross_celeb|x2_severed_brands|x2_severed_celeb|x2_score) $step ;;
    x2r_cross_brands|x2r_cross_celeb|x2b_brands|x2b_celeb|x3) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
