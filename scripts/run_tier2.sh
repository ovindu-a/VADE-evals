#!/usr/bin/env bash
# Tier-2 generality experiments (E7-E11), sequentially, one log per step.
#
#   bash scripts/run_tier2.sh smoke                    # every NEW GPU script at tiny size (~10 min)
#   bash scripts/run_tier2.sh e10 e7 e9 e11            # Qwen2.5-VL-7B steps (~1.5 h)
#   bash scripts/run_tier2.sh e8_3b e8_internvl        # second models: prep + pipeline (~5-6 h)
#   bash scripts/run_tier2.sh all                      # everything above, in that order
#
# Run inside tmux. PY overrides the interpreter (default: python). Logs: logs/tier2/<step>_<stamp>.log
# GPU=a100 (any 80GB card) raises every batch size that only changes how many rows share a
# forward pass; unset = each script's own default (sized for 24GB).
# e8_* write pruned tuples into the sibling VADE repo (VADE/models/<model slug>/), VADE's own convention.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VADE=${VADE_ROOT:-../VADE}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/tier2

BS_SEV= BS_VRH= BS_CEIL= BS_PIPE= BS_ACC=
N_SEV=64 N_CEIL=48       # pairs per severed run (E7), rows per ceiling sweep (E11)
if [ "${GPU:-}" = a100 ]; then
  N_SEV=128 N_CEIL=96
  # PIPE stays at 32, not 64: head_trace/head_window_sweep build their shuffled-donor control by
  # rolling rows WITHIN a batch, and a single 64-row batch would leave no batch boundary at all.
  BS_SEV=64 BS_VRH=128 BS_CEIL=64 BS_PIPE=32 BS_ACC=128
  echo "GPU=a100 batch profile: severed=$BS_SEV vrh=$BS_VRH ceiling=$BS_CEIL pipeline=$BS_PIPE accuracy=$BS_ACC"
  echo "GPU=a100 sample sizes: e7 n_pairs=$N_SEV e11 n_rows=$N_CEIL"
fi
bs() { [ -n "$1" ] && echo "--batch_size $1"; return 0; }   # bs <value> -> flag or nothing

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/tier2/${step}_${STAMP}.log"
  echo "=== [$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

top8() {  # top8 <pipeline_root> <entity> <attribute> -> "23.3,21.4,..." off that trace's phase-1 ranking
  $PY - "$@" <<'EOF'
import glob, json, sys
root, entity, attribute = sys.argv[1:4]
paths = sorted(glob.glob(f"{root}/{entity}/{attribute}/head_trace_*.json"))
assert paths, f"no head_trace JSON under {root}/{entity}/{attribute} -- run image_head_pipeline first"
print(",".join(f"{b}.{h}" for b, h in json.load(open(paths[0]))["phase1_ranked"][:8]))
EOF
}

# ---------------------------------------------------------------- Qwen2.5-VL-7B

e10() {  # head-set overlap vs chance: across entities (E7 stats) + vs Park/Lietzow lists (E10). CPU, ~1 min
  run e10_overlap $PY methods/head_overlap.py --n_perm 20000
}

e7() {  # do brands/celebrities use the same movers AND the same downstream MLP lookup?
  local P=results/image_head_pipeline
  local BH CN CL
  BH=$(top8 $P brands hq_country); CN=$(top8 $P celebrities nationality); CL=$(top8 $P celebrities native_language)
  run e7_vocab_brands $PY methods/head_vocab_projection.py --entity brands \
      --head_sets "hq_country_top8=$BH" "common10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
  run e7_vocab_celebrities $PY methods/head_vocab_projection.py --entity celebrities \
      --head_sets "nationality_top8=$CN" "native_language_top8=$CL" \
                  "common10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
  # --gold own: brands/celebrities store lowercase labels, so first-token columns must use the
  # model's own answers (read at their divergence point). Text columns use VADE's matcher either way.
  local SEV=(--n_pairs "$N_SEV" --gold own --freeze_values base mean other --freeze_mlp_span 22-27
             --freeze_mlp_singles 23 24 25 26 --freeze_attn_span 24-27 $(bs "$BS_SEV"))
  run e7_severed_brands_hq $PY methods/head_severed.py --entity brands --attributes hq_country --heads "$BH" \
      "${SEV[@]}" --out_dir results/head_severed/brands_hq_country__gold-own
  run e7_severed_celeb_nat $PY methods/head_severed.py --entity celebrities --attributes nationality --heads "$CN" \
      "${SEV[@]}" --out_dir results/head_severed/celebrities_nationality__gold-own
  run e7_severed_celeb_lang $PY methods/head_severed.py --entity celebrities --attributes native_language \
      --heads "$CL" "${SEV[@]}" --out_dir results/head_severed/celebrities_native_language__gold-own
}

e9() {  # zero Park et al.'s Visual Retrieval Heads on flags: index check, recall, entity at the movers
  run e9_vrh $PY methods/vrh_ablation.py $(bs "$BS_VRH")
}

e11() {  # object vs background tokens: does identity leave the object before the movers read it?
  # seq:before MUST read 0% (bit-identical tokens under causal attention); if not, discard the run.
  local L=(--layers 10 14 18 19 20 21 22 23 --sites residual --n_rows "$N_CEIL" $(bs "$BS_CEIL"))
  run e11_flags $PY methods/ndm/ceiling_sweep.py --entity flags --attribute capital "${L[@]}" \
      --positions_list entity_only "~entity_ring1" "~entity_only" full_image seq:before seq:after
  run e11_brands $PY methods/ndm/ceiling_sweep.py --entity brands --attribute hq_country "${L[@]}" --text_match \
      --positions_list entity_only "~entity_ring1" "~entity_only" full_image seq:before seq:after
  run e11_celebrities $PY methods/ndm/ceiling_sweep.py --entity celebrities --attribute nationality "${L[@]}" \
      --text_match --positions_list entity_only "~entity_ring1" "~entity_only" full_image seq:before seq:after
}

# ---------------------------------------------------------------- second models (E8)

prep_model() {  # prep_model <step> <model_id>: VADE baseline accuracy -> pruned tuples for that model
  run "$1_accuracy" $PY "$VADE/models/run_accuracy_sweep.py" --model_id "$2" --entities flags brands celebrities \
      $(bs "$BS_ACC")
  # identity/ holds only test.jsonl and no `rule` field, which prune_tuples.py (VADE b8dfaf9) cannot
  # read; nothing in E8 uses it, so prune every OTHER tuple dir.
  local attrs
  attrs=$(find "$VADE"/data/{flags,brands,celebrities}/tuples -mindepth 1 -maxdepth 1 -type d \
          ! -name identity -printf '%f\n' | sort -u)
  run "$1_prune" $PY "$VADE/models/prune_tuples.py" --model_id "$2" --entities flags brands celebrities \
      --attributes $attrs
}

pipeline() {  # pipeline <step> <model_id> <n_layers> <n_heads>: image->text heads on the key attributes
  local step=$1 model=$2 nl=$3 nh=$4
  local C=(--model_id "$model" --n_layers "$nl" --n_heads "$nh" $(bs "$BS_PIPE"))
  run "${step}_flags" $PY methods/image_head_pipeline.py --entity flags "${C[@]}"
  run "${step}_brands" $PY methods/image_head_pipeline.py --entity brands --attributes hq_country "${C[@]}"
  run "${step}_celebrities" $PY methods/image_head_pipeline.py --entity celebrities \
      --attributes nationality native_language "${C[@]}"
}

e8_3b() {  # same family, smaller: 36 blocks x 16 heads, so head IDs do not transfer -- compare the pattern
  prep_model e8_3b Qwen/Qwen2.5-VL-3B-Instruct
  pipeline e8_3b_pipeline Qwen/Qwen2.5-VL-3B-Instruct 36 16
  run e8_3b_overlap $PY methods/head_overlap.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --n_layers 36 --n_heads 16 \
      --pipeline_root results/image_head_pipeline_Qwen2.5-VL-3B-Instruct --flags_trace_glob ''
}

e8_internvl() {  # different family, SAME 28x28 Qwen2.5-7B text stack: head IDs are directly comparable
  local M=OpenGVLab/InternVL3-8B-hf
  prep_model e8_internvl $M
  pipeline e8_internvl_pipeline $M 28 28
  run e8_internvl_overlap $PY methods/head_overlap.py --model_id $M \
      --pipeline_root results/image_head_pipeline_InternVL3-8B-hf --flags_trace_glob '' \
      --compare_pipeline_root results/image_head_pipeline --compare_label qwen7b
}

# ---------------------------------------------------------------- smoke

smoke() {
  run smoke_e10 $PY methods/head_overlap.py --n_perm 500 --out_dir results/head_overlap/smoke
  run smoke_e9 $PY methods/vrh_ablation.py --limit_items 6 --n_templates 1 --index_items 3 --n_random 1 \
      --out_dir results/vrh_ablation/smoke
  run smoke_e7 $PY methods/head_severed.py --entity brands --attributes hq_country \
      --heads "$(top8 results/image_head_pipeline brands hq_country)" --n_pairs 8 --gold own \
      --freeze_values base mean other --freeze_mlp_span 23-26 --freeze_mlp_singles 25 --freeze_attn_span '' \
      --out_dir results/head_severed/smoke
  run smoke_e8 $PY methods/image_head_pipeline.py --entity flags --attributes language \
      --model_id OpenGVLab/InternVL3-8B-hf --allow_unpruned --dry_run
}

[ $# -eq 0 ] && { sed -n 2,12p "$0"; exit 1; }
for step in "$@"; do
  case $step in
    all) e10; e7; e9; e11; e8_3b; e8_internvl ;;
    e7|e8_3b|e8_internvl|e9|e10|e11|smoke) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
