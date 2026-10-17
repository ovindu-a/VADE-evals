#!/usr/bin/env bash
# Tier-3 follow-ups (FOLLOWUP_PLAN.md P1-P33), sequentially, one log per step.
#
#   bash scripts/run_tier3.sh smoke              # every changed code path at tiny size (~15 min)
#   bash scripts/run_tier3.sh session1           # P1-P12: results that can void or overturn written claims
#   bash scripts/run_tier3.sh session2           # P13-P27: generality
#   bash scripts/run_tier3.sh session3           # P28-P33: expensive / optional
#   bash scripts/run_tier3.sh p1 p2 p5           # any steps by name
#
# Run inside tmux. PY overrides the interpreter (default: python). Logs: logs/tier3/<step>_<stamp>.log
# Every step writes under results/tier3/ (or a new *_n128 / suffixed name), never over a committed file.
# GPU=a100 (any 80GB card) raises batch sizes that only change how many rows share a forward pass.
# P30 (dictionary controls) and P34 (Qwen2.5-VL-32B) are manual: see FOLLOWUP_PLAN.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VADE=${VADE_ROOT:-../VADE}
STAMP=$(date +%Y%m%d_%H%M%S)
R=results/tier3
mkdir -p logs/tier3 "$R"

BS_VRH= BS_CEIL= BS_SEV= BS_TEC= BS_RK= BS_CC= BS_DAS=
if [ "${GPU:-}" = a100 ]; then
  # head_trace / head_window_sweep / attr_head_trace keep their defaults: their shuffled-donor
  # controls roll rows WITHIN a batch (see run_tier2.sh).
  BS_VRH=128 BS_CEIL=64 BS_SEV=64 BS_TEC=64 BS_RK=64 BS_CC=64 BS_DAS=32
  echo "GPU=a100 batch profile: vrh=$BS_VRH ceiling=$BS_CEIL severed=$BS_SEV tec=$BS_TEC route=$BS_RK compose=$BS_CC das=$BS_DAS"
fi
bs() { [ -n "$1" ] && echo "--batch_size $1"; return 0; }   # bs <value> -> flag or nothing

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/tier3/${step}_${STAMP}.log"
  local prev="" x
  for x in "$@"; do   # several scripts open --out without creating its directory -- AFTER the GPU work
    [ "$prev" = "--out" ] && mkdir -p "$(dirname "$x")"
    prev=$x
  done
  echo "=== [$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

COMMON10=21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17
ATTRS="language capital currency calling_code"
FLAGS_TRACE='logs/Qwen2.5-VL-7B-Instruct/flags/ndm/{a}/L1_0.0_T0-0_LR0_MLR0_CLIP1_full_image_attn_head_output_pruned/head_trace_patch21_blocks21-23.json'
trace_of() { echo "${FLAGS_TRACE//\{a\}/$1}"; }

top8() {  # top8 <pipeline_root> <entity> <attribute> -> "23.3,21.4,..." off that trace's phase-1 ranking
  $PY - "$@" <<'EOF'
import glob, json, sys
root, entity, attribute = sys.argv[1:4]
paths = sorted(glob.glob(f"{root}/{entity}/{attribute}/head_trace_*.json"))
assert paths, f"no head_trace JSON under {root}/{entity}/{attribute} -- run image_head_pipeline first"
print(",".join(f"{b}.{h}" for b, h in json.load(open(paths[0]))["phase1_ranked"][:8]))
EOF
}

# ================================================================ session 1

p1() {  # Z1: is DBM's eval path sound? all-ones mask (= full swap) and 0.5 mask through dbm/eval.py
  mkdir -p $R/dbm_identity
  $PY - <<'EOF'
import sys, torch; sys.path.insert(0, ".")
from methods.dbm.intervention import SigmoidMaskIntervention
for name, v in [("ones", 1.0), ("half", 0.0)]:          # sigmoid(0 / T) = 0.5 at any T
    iv = SigmoidMaskIntervention(embed_dim=3584); iv.mask.data.fill_(v); iv.temperature.data = torch.tensor(1e-7)
    torch.save(iv.state_dict(), f"results/tier3/dbm_identity/mask_{name}.pt")
    print("wrote", name)
EOF
  for c in ones half; do
    run "p1_dbm_${c}" $PY methods/dbm/eval.py --entity flags --attribute language --layer 16 --positions flag_ring1 \
        --split test --cause_only --ckpt_path $R/dbm_identity/mask_$c.pt --out_dir $R/dbm_identity/L16_$c
    run "p1_dbm_${c}_nocache" $PY methods/dbm/eval.py --entity flags --attribute language --layer 16 \
        --positions flag_ring1 --split test --cause_only --no_source_cache --ckpt_path $R/dbm_identity/mask_$c.pt \
        --out_dir $R/dbm_identity/L16_${c}_nocache
  done
}

p2() {  # E3/claim 1: block-matched random-10 null for zeroing the movers
  run p2_vrh_mover_random $PY methods/vrh_ablation.py --lists park_vrh --n_random 0 --n_random_movers 5 \
      --index_items 0 $(bs "$BS_VRH") --out_dir $R/vrh_mover_random
}

p3() {  # A3: select movers on the 59 TRAIN countries (the common10 rule), install on the 25 test countries
  for a in $ATTRS; do
    run "p3_trace_$a" $PY methods/head_trace.py --entity flags --attribute "$a" --patch_layer 21 \
        --positions full_image --blocks 21 22 23 --n_rows 64 --split train \
        --out $R/heldout_heads/${a}_train_b21-23.json
  done
  local HELD
  HELD=$($PY - <<'EOF'
import json
attrs = "language capital currency calling_code".split()
common10 = {(21,1),(21,5),(22,13),(22,15),(22,17),(22,19),(23,3),(23,4),(23,6),(23,17)}
tops = [[tuple(h) for h in json.load(open(f"results/tier3/heldout_heads/{a}_train_b21-23.json"))["phase1_ranked"][:16]]
        for a in attrs]
held = sorted(set(tops[0]).intersection(*tops[1:]))   # R8's rule: in ALL four attributes' top-16
assert held, "no head is in all four train top-16 lists"
import sys
print(f"[p3] heldout = {len(held)} heads, {len(set(held) & common10)} shared with common10", file=sys.stderr)
print(",".join(f"{b}.{h}" for b, h in held))
EOF
)
  echo "$HELD" > $R/heldout_heads/heldout_consensus.txt
  run p3_install $PY methods/head_swap_vade.py --head_sets "heldout=$HELD" \
      "heldout_per_attr=$R/heldout_heads/{attribute}_train_b21-23.json#10" \
      --arms heads random_heads --out_dir $R/head_swap_vade_heldout
}

p4() {  # E3/claim 6: does a fixed gate=0.5 (an untrained dbm mask) reproduce E3's DBM finals?
  for a in $ATTRS; do
    run "p4_gate05_$a" $PY methods/head_das.py --attribute "$a" --method dbm --site mlp_hidden --blocks 23-26 \
        --l1_coef 1e-4 --temperature_start 1e-2 --temperature_end 1e-2 --eval_rows 4000 --seed 0 \
        $(bs "$BS_DAS") --skip_train --out_dir $R/head_das_gate0.5/${a}_mlp23-26
    run "p4_score_$a" $PY $VADE/eval/score.py --predictions $R/head_das_gate0.5/${a}_mlp23-26/predictions.jsonl \
        --entity flags --attribute "$a"
  done
  for a in $ATTRS; do   # E3's own DBM predictions were never scored into a summary
    local d=results/head_das/flags/${a}_dbm_l10.0001_T0.01-0.01_mlp23-26
    [ -f "$d/predictions.jsonl" ] && run "p4_score_e3_$a" $PY $VADE/eval/score.py --predictions "$d/predictions.jsonl" \
        --entity flags --attribute "$a" --out_dir $R/head_das_e3_scores/$a
  done
  return 0
}

p5() {  # A6/A7/Z4: head_cross on 64 NEW pairs (seed 1 shares none with seed 0)
  run p5_head_cross $PY methods/head_cross.py --seed 1 --n_pairs 64 --out $R/head_cross_seed1/cross_n64_seed1.jsonl
  run p5_bootstrap $PY methods/cpu_followups.py --analyses cross_bootstrap \
      --cross_rows $R/head_cross_seed1/cross_n64_seed1.jsonl --out_dir $R/cpu_followups_seed1
}

SPLIT_A="AL AM AR BD BE BR BY CM CO DE ES ET FI FJ GT HN IE IS KE KH KR LA LB LR LY MK NA NI OM PA PT RO RS SA SD SG TH TN TZ UA VN YE"
SPLIT_B="AF AO AU AZ BH BZ CL CU CV CY DK DZ EG GH GR GY HT HU IN IQ IT JO JP KP KW LU MA MU MX NP NZ PE PK PY QA RW SE SN UG UY UZ ZM"

p6() {  # B1: routers selected on half the countries, evaluated on the other half; then traced through block 27
  run p6_sel_A $PY methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
      --countries $SPLIT_A --out $R/attr_heads/sel_A.json
  run p6_eval_B $PY methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 17 18 19 20 21 22 \
      --countries $SPLIT_B --rank_from $R/attr_heads/sel_A.json --out $R/attr_heads/eval_B_rankA.json
  run p6_blocks15-27 $PY methods/attr_head_trace.py --entity flags --patch_layer 15 \
      --blocks 15 16 17 18 19 20 21 22 23 24 25 26 27 --n_countries 84 --knockout_ks 1 2 4 8 16 32 64 128 364 \
      --out $R/attr_heads/patch15_blocks15-27_n84.json
}

p7() {  # D1: every LM write at the image (blocks:L) vs the vision embedding alone (residual@0)
  for a in $ATTRS; do
    run "p7_blockspan_$a" $PY methods/ndm/ceiling_sweep.py --entity flags --attribute "$a" --positions full_image \
        --layers 0 10 20 --sites residual blocks:1 blocks:2 blocks:3 blocks:4 blocks:5 blocks:10 blocks:20 \
        $(bs "$BS_CEIL") --out $R/ceiling_blockspan/${a}_full_image_L0-10-20.json
  done
}

p8() {  # D4: is seq:before already non-null at layer 0 (= the vision encoder's output)?
  run p8_flags $PY methods/ndm/ceiling_sweep.py --entity flags --attribute capital --positions seq:before \
      --layers 0 1 --sites residual $(bs "$BS_CEIL") --out $R/seqbefore/flags_capital_L0-1.json
  run p8_brands $PY methods/ndm/ceiling_sweep.py --entity brands --attribute hq_country --positions seq:before \
      --layers 0 1 --sites residual --text_match --max_new_tokens 8 $(bs "$BS_CEIL") \
      --out $R/seqbefore/brands_hq_country_L0-1.json
  run p8_celebrities $PY methods/ndm/ceiling_sweep.py --entity celebrities --attribute nationality \
      --positions seq:before --layers 0 1 --sites residual --text_match --max_new_tokens 8 $(bs "$BS_CEIL") \
      --out $R/seqbefore/celebrities_nationality_L0-1.json
}

p9() {  # A1: logit-level single / pair / leave-one-out mover installs
  run p9_mover_subsets $PY methods/text_entity_contrast.py --phases heads --entity_layer 21 21 --mover_subsets \
      $(bs "$BS_TEC") --out_dir $R/mover_subsets
}

p10() {  # C2/C3/C4: MLP 23-26 leave-one-block-out, MLP 25/27 singles, at base/mean/other, new seed
  run p10_severed $PY methods/head_severed.py --n_pairs 128 --seed 1 \
      --freeze_mlp_span 23-26 24-26 23,25-26 23-24,26 23-25 --freeze_mlp_singles 25 27 \
      --freeze_attn_span '' --freeze_values base mean other $(bs "$BS_SEV") --out_dir $R/severed_loo_n128_s1
}

p11() {  # A2: restore the heads at EVERY decode step (calling_code's 28% left over)
  run p11_decode $PY methods/head_decode_trace.py --trace "$(trace_of calling_code)" --head_ks 10 \
      --heads $COMMON10 --out_dir $R/decode_calling_code_k10
}

p12() {  # B4: steer at L22 with the layer-17 (block-16) direction vs its own layer's direction
  run p12_dir17_at22 $PY methods/attr_steer.py --entity flags --layers 22 --direction_layer 17 --mode add \
      --alpha 0.5 1 2 4 --out $R/attr_steer/dir17_at22.json
  run p12_dir22_at22 $PY methods/attr_steer.py --entity flags --layers 22 --direction_layer 22 --mode add \
      --alpha 0.5 1 2 4 --out $R/attr_steer/dir22_at22.json
}

# ================================================================ session 2

p13() {  # C5: readout rank sweep at MLPs 23-26 (needs the pod's readout_fit.pt)
  local FIT=results/readout_jacobian/flags__mlp_hidden_b23-26/readout_fit.pt
  [ -f "$FIT" ] || { echo "p13: SKIPPED -- $FIT missing (gitignored; it lives on the pod that ran E3)"; return 0; }
  for r in 32 64 128 256; do
    run "p13_rank$r" $PY methods/readout_jacobian.py --phase eval --site mlp_hidden --blocks 23-26 --max_rank $r \
        --energy 1.0 --fit_path "$FIT" --out_dir $R/readout_rank/r${r}_e1
  done
}

p14() {  # A10: common10 installed on animals (image patch at layer 10, before the animals handoff)
  run p14_animals $PY methods/head_swap_vade.py --entity animals \
      --tuples_dir $VADE/models/Qwen2.5-VL-7B-Instruct/animals/tuples --head_sets "common10=$COMMON10" \
      --arms clean heads random_heads full_image --patch_layer 10 --out_dir $R/head_swap_animals
}

p15() {  # C6: brands -- does cutting late attention to the image during the head install raise transfer?
  local BH; BH=$(top8 results/image_head_pipeline brands hq_country)
  run p15_brands_late $PY methods/route_knockout.py --entity brands --attributes hq_country --install_heads "$BH" \
      --knockouts late=readout:24-27 --q_layers 16 $(bs "$BS_RK") --out_dir $R/route_ko_brands_late
}

N128="animals:body_covering animals:native_continent animals:primary_habitat animals:reproduction animals:taxonomic_class
celebrities:birth_year celebrities:nationality celebrities:native_language celebrities:occupation
brands:founded_year brands:hq_country brands:industry"

p16() {  # D5: stage-1 curves at n=128 into a NEW root (the default root would reuse the n=32 files)
  local ea e a
  for ea in $N128; do
    e=${ea%%:*} a=${ea##*:}
    for pos in full_image last_token; do
      run "p16_${e}_${a}_${pos}" $PY methods/ndm/ceiling_sweep.py --entity "$e" --attribute "$a" --sites residual \
          --positions $pos --layers $(seq 0 28) --n_rows 128 --text_match --max_new_tokens 8 $(bs "$BS_CEIL") \
          --out results/image_head_pipeline_n128/$e/$a/ceiling_${pos}_text.json
    done
  done
  run p16_pipeline $PY methods/image_head_pipeline.py --entity animals celebrities brands --stages ceiling \
      --n_rows_ceiling 128 --out_root results/image_head_pipeline_n128
}

p17() {  # D6: animals -- does a text position carry the attribute in the L14-20 gap?
  run p17_animals_text $PY methods/ndm/ceiling_sweep.py --entity animals --attribute body_covering \
      --positions_list phrase:attribute vision_end tok:-2 pre_image --text_match --max_new_tokens 8 \
      --sites residual --layers $(seq 0 28) $(bs "$BS_CEIL") --out_suffix _tier3
}

p18() {  # F2/F3: Qwen2.5-VL-3B's two handoff steps traced separately; continuous install for calling_code
  local M=Qwen/Qwen2.5-VL-3B-Instruct
  for a in language capital; do
    run "p18_3b_${a}_27-28" $PY methods/head_trace.py --entity flags --attribute "$a" --model_id $M \
        --patch_layer 27 --positions full_image --blocks 27 28 --n_rows 64 --out $R/qwen3b/${a}_patch27_blocks27-28.json
    run "p18_3b_${a}_31-33" $PY methods/head_trace.py --entity flags --attribute "$a" --model_id $M \
        --patch_layer 31 --positions full_image --blocks 31 32 33 --n_rows 64 --out $R/qwen3b/${a}_patch31_blocks31-33.json
  done
  run p18_3b_install $PY methods/head_swap_vade.py --model_id $M --patch_layer 27 \
      --head_sets "cc10=results/image_head_pipeline_Qwen2.5-VL-3B-Instruct/flags/calling_code/head_trace_patch27_blocks27.json#10" \
      --arms clean heads random_heads full_image --out_dir $R/head_swap_qwen3b
}

p19() {  # B8: one router trace per donor attribute
  for a in $ATTRS; do
    run "p19_donor_$a" $PY methods/attr_head_trace.py --entity flags --patch_layer 15 \
        --blocks 15 16 17 18 19 20 21 22 --n_countries 84 --donor_attributes "$a" --out $R/attr_heads/donor_${a}.json
  done
}

p20() {  # B9: compose with 16 routers; does the answer-format failure shrink?
  run p20_compose $PY methods/circuit_compose.py --router_k 16 --n_items 64 $(bs "$BS_CC") --out_dir $R/compose_rk16
  run p20_failures $PY methods/cpu_followups.py --analyses compose_failures --compose_rows $R/compose_rk16/rows.jsonl \
      --out_dir $R/cpu_followups_rk16
}

p21() {  # E2: currency under Park's heads against 10 block-matched random draws
  run p21_currency $PY methods/vrh_ablation.py --attributes currency --n_random 10 --index_items 0 $(bs "$BS_VRH") \
      --out_dir $R/vrh_currency_r10
}

p22() {  # D9: text vs image entity freeze at an EARLY entity layer
  run p22_freeze_Le4 $PY methods/text_entity_contrast.py --phases freeze --entity_layer 4 4 $(bs "$BS_TEC") \
      --out_dir $R/text_Le4_freeze
}

p23() {  # D2/D3: ring bands by layer; 8 of the 16 face tokens (token-count control)
  run p23_rings $PY methods/ndm/ceiling_sweep.py --entity flags --attribute capital \
      --positions_list ring:0 ring:1 ring:2 ring:3 ring:4 ring:0+ring:2+ring:3+ring:4 \
      --layers 2 6 10 14 18 22 --sites residual $(bs "$BS_CEIL") --out_suffix _tier3
  # celebrities entity_only = 52-55 64-67 76-79 88-91; two disjoint 8-token halves (left / right columns)
  run p23_face_left $PY methods/ndm/ceiling_sweep.py --entity celebrities --attribute nationality \
      --positions idx:52,53,64,65,76,77,88,89 --layers 2 10 18 --sites residual --text_match --max_new_tokens 8 \
      $(bs "$BS_CEIL") --out $R/face_tokens/celebrities_nationality_left8.json
  run p23_face_right $PY methods/ndm/ceiling_sweep.py --entity celebrities --attribute nationality \
      --positions idx:54,55,66,67,78,79,90,91 --layers 2 10 18 --sites residual --text_match --max_new_tokens 8 \
      $(bs "$BS_CEIL") --out $R/face_tokens/celebrities_nationality_right8.json
}

p24() {  # B3/G7: steering robustness -- new splits, every-step application, eager backend
  for s in 1 2; do
    run "p24_seed$s" $PY methods/attr_steer.py --entity flags --seed $s --out $R/attr_steer/seed$s.json
  done
  run p24_continuous $PY methods/attr_steer.py --entity flags --layers 21 22 23 --continuous \
      --out $R/attr_steer/continuous_L21-23.json
  run p24_eager $PY methods/attr_steer.py --entity flags --attn_impl eager --out $R/attr_steer/eager_seed0.json
}

p25() {  # B6/B7: field word vs 'Answer' vs ':' ; gapped attention sets (leave one of blocks 19-21 out)
  run p25_scopes $PY methods/attribute_switch_sweep.py --sites residual --scopes field tok:-2 last_token \
      --block_spans --layers $(seq 1 28) --execution cached --batch_size 64 --out_dir $R/switch_field_scopes
  run p25_sets $PY methods/attribute_switch_sweep.py --sites attention --scopes last_token --block_spans \
      --attention_spans 8 --layers 24 --attention_sets 16-18.20-23 16-19.21-23 16-20.22-23 \
      --execution cached --batch_size 64 --out_dir $R/switch_attention_sets
}

p26() {  # G7: the cached engine against serial decoding
  run p26_verify $PY methods/attribute_switch_sweep.py --n_countries 1 --attributes capital currency \
      --scopes last_token --sites attention --layers 22 --block_spans --attention_spans 3 --execution cached \
      --batch_size 2 --verify_cached --out_dir $R/switch_verify_sdpa
  run p26_verify_eager $PY methods/attribute_switch_sweep.py --n_countries 1 --attributes capital currency \
      --scopes last_token --sites attention --layers 22 --block_spans --attention_spans 3 --execution cached \
      --batch_size 2 --verify_cached --attn_impl eager --no_share_prefix --out_dir $R/switch_verify_eager_noshare
  local P=(--countries AO DE --layers 18 20 22 24 --sites residual attention --scopes earlier_text last_token
           --block_spans --attention_spans 3 --attn_impl sdpa)
  run p26_pair_serial $PY methods/attribute_switch_sweep.py "${P[@]}" --execution serial --out_dir $R/pair_serial_sdpa
  run p26_pair_cached64 $PY methods/attribute_switch_sweep.py "${P[@]}" --execution cached --batch_size 64 \
      --out_dir $R/pair_cached_b64_sdpa
  run p26_pair_cached1 $PY methods/attribute_switch_sweep.py "${P[@]}" --execution cached --batch_size 1 \
      --out_dir $R/pair_cached_b1_sdpa
}

p27() {  # A9: answer-column edges to OBJECT vs BACKGROUND keys in the mover blocks (question route cut too)
  run p27_groups $PY methods/route_knockout.py --knockouts \
      "obj_movers=readout:21-23@object" "bg_movers=readout:21-23@background" \
      "obj_movers_noq=readout:21-23@object+question:0-27" "bg_movers_noq=readout:21-23@background+question:0-27" \
      "movers_noq=readout:21-23+question:0-27" --q_layers 16 $(bs "$BS_RK") --out_dir $R/route_ko_groups
}

# ================================================================ session 3

p28() {  # A8: dose (alpha) install of common10; then the geography of the 'other' answers
  run p28_dose $PY methods/head_swap_vade.py --head_sets "common10=$COMMON10" --alphas 0 0.25 0.5 0.75 1 \
      --arms heads --out_dir $R/head_swap_dose
  local X=() a
  for a in 0 0.25 0.5 0.75 1; do X+=("dose_a$a=$R/head_swap_dose/heads_common10_a$a.jsonl"); done
  run p28_geography $PY methods/cpu_followups.py --analyses error_geography entity_swap \
      --extra_predictions "${X[@]}" --out_dir $R/cpu_followups_dose
}

p29() {  # G4/E3: das_fixed at das_rotated's budget (375 steps, bs16 ga1); dbm masks with a 10x mask lr
  local K
  for K in 159,332,282 13,14,25 49,54,75 103,161,172 8 64 256; do
    run "p29_das_fixed_k${K//,/-}" $PY methods/head_das.py --attribute language --method das_fixed --subspace_dim "$K" \
        --batch_size 16 --grad_accum_steps 1 --train_rows 6000 --eval_rows 2000 --seed 0 \
        --out_dir $R/head_das/language_das_fixed_k${K//,/-}_bs16ga1_tr6000
  done
  run p29_dbm_mlr $PY methods/head_das.py --attribute language --method dbm --site mlp_hidden --blocks 23-26 \
      --l1_coef 1e-4 --temperature_start 1e-2 --temperature_end 1e-2 --mask_lr 1e-2 --epochs 2 \
      --train_rows 4000 --eval_rows 4000 --batch_size 32 --grad_accum_steps 2 \
      --out_dir $R/head_das/language_dbm_mlr1e-2_ep2_mlp23-26
}

p31() {  # Z12: token-edge knockout at n=64 (calling_code only; all four is ~100k batch-1 forwards)
  run p31_identify env OUT_ROOT=$R/head_followups_n64 ATTRIBUTES=calling_code PY="$PY" \
      bash scripts/run_head_followups.sh identify
  run p31_knockout env OUT_ROOT=$R/head_followups_n64 ATTRIBUTES=calling_code PY="$PY" \
      bash scripts/run_head_followups.sh knockout
}

p32() {  # B5/C1: the attribute capture on VADE's own templates (prefill-word confound: see the plan)
  run p32_capture $PY methods/attr_capture.py --prompts vade --template_index 0
}

p33() {  # C7: per-country (item) steering directions in project mode at the late layers
  run p33_item $PY methods/attr_steer.py --entity flags --layers 24 26 27 28 --mode project --direction_scope item \
      --out $R/attr_steer/item_project_L24-28.json
}

# ================================================================ smoke

smoke() {  # tiny sizes, numbers meaningless: every changed path must run end to end
  local S=$R/smoke
  run smoke_vrh $PY methods/vrh_ablation.py --lists park_vrh --limit_items 4 --n_templates 1 --index_items 0 \
      --n_random 0 --n_random_movers 1 --out_dir $S/vrh
  run smoke_trace_train $PY methods/head_trace.py --entity flags --attribute capital --patch_layer 21 \
      --positions full_image --blocks 21 22 23 --n_rows 8 --split train --out $S/trace_train.json
  run smoke_swap_alpha $PY methods/head_swap_vade.py --head_sets "common10=$COMMON10" --alphas 0 1 --arms heads \
      --limit_pairs 2 --out_dir $S/swap_alpha
  run smoke_tec $PY methods/text_entity_contrast.py --phases heads --entity_layer 21 21 --mover_subsets --n_pairs 4 \
      --out_dir $S/tec
  run smoke_severed $PY methods/head_severed.py --n_pairs 4 --freeze_mlp_span 23-26 23,25-26 \
      --freeze_mlp_singles 27 --freeze_attn_span '' --freeze_values base other --out_dir $S/severed
  run smoke_decode $PY methods/head_decode_trace.py --trace "$(trace_of calling_code)" --head_ks 10 \
      --heads $COMMON10 --limit 1 --out_dir $S/decode
  run smoke_steer $PY methods/attr_steer.py --entity flags --layers 22 --direction_layer 17 --continuous \
      --n_train_items 8 --n_test_items 4 --out $S/steer.json
  run smoke_attr_trace $PY methods/attr_head_trace.py --entity flags --patch_layer 15 --blocks 15 16 \
      --n_countries 2 --donor_attributes capital --out $S/attr_trace.json
  run smoke_switch $PY methods/attribute_switch_sweep.py --n_countries 1 --sites residual attention \
      --scopes field tok:-2 --block_spans --attention_spans --attention_sets 19-20 --layers 21 \
      --execution cached --batch_size 8 --out_dir $S/switch
  run smoke_ceiling $PY methods/ndm/ceiling_sweep.py --entity flags --attribute capital --positions idx:64,65 \
      --layers 10 --sites residual --n_rows 4 --out $S/ceiling_idx.json
  run smoke_route_groups $PY methods/route_knockout.py --n_pairs 4 --knockouts \
      "obj=readout:21-23@object" "bg=readout:21-23@background" "stack=readout:21-23@object+question:0-27" \
      --q_layers 16 --out_dir $S/route_groups
  run smoke_route_install $PY methods/route_knockout.py --n_pairs 4 --install_heads common10 \
      --knockouts "late=readout:24-27" --q_layers 16 --out_dir $S/route_install
  run smoke_gate $PY methods/head_das.py --attribute language --method dbm --site mlp_hidden --blocks 23-26 \
      --eval_rows 8 --skip_train --out_dir $S/gate05
  run smoke_dbm_eval $PY methods/dbm/eval.py --entity flags --attribute language --layer 16 --positions flag_ring1 \
      --cause_only --limit_rows 16 --ckpt_path $R/dbm_identity/mask_ones.pt --out_dir $S/dbm_ones \
      || echo "    (smoke_dbm_eval needs p1's masks: run p1 first, or ignore)"
}

[ $# -eq 0 ] && { sed -n 2,13p "$0"; exit 1; }
for step in "$@"; do
  case $step in
    session1) p1; p2; p5; p3; p4; p6; p7; p8; p9; p10; p11; p12 ;;
    session2) p13; p14; p15; p16; p17; p18; p19; p20; p21; p22; p23; p24; p25; p26; p27 ;;
    session3) p28; p29; p31; p32; p33 ;;
    p1|p2|p3|p4|p5|p6|p7|p8|p9|p10|p11|p12|p13|p14|p15|p16|p17|p18|p19|p20|p21|p22|p23|p24|p25|p26|p27|p28|p29|p31|p32|p33|smoke) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
