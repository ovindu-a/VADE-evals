#!/usr/bin/env bash
# X4 (PAPER_RUNBOOK.md): the mover / router / MLP-lookup story, plus X1's crossmodal test, on
# LLaVA-1.5-7B (Vicuna = Llama-2 language model) -- the reference VLM of the papers this work is
# positioned against. Same stages as scripts/run_x3_idefics3.sh, with what X3 taught built in:
#   * stage-1 ceiling curves at n=128 rows (X3's 32-row curves had one-row dips that broke the
#     automatic window rules, so every X3 window ended up hand-picked);
#   * routers always use --own_golds (X3's gate failed on lowercase answers);
#   * windows can be overridden by environment variables, so a hand choice is RECORDED in the
#     command line and the log rather than typed by hand:
#       X4_PATCH=16 X4_BLOCKS="16 17 18 19 20" ... pipeline
#       X4_ROUTER_PATCH=12 X4_ROUTER_BLOCKS="12 13 14 15" ... routers
#   * the smoke ceiling step no longer passes --out with --positions_list (X3's smoke bug).
#
#   bash scripts/run_x4_llava.sh smoke        # every stage at tiny size on the real model
#   bash scripts/run_x4_llava.sh gate         # cheap accuracy gate; STOP if it fails
#   bash scripts/run_x4_llava.sh prep pipeline overlap depth routers severed crossmodal
#   bash scripts/run_x4_llava.sh all          # gate, then everything above, in that order
#   X4_SEVERED_ATTRS="capital language" bash scripts/run_x4_llava.sh severed   # skip voided traces
#
# Run inside tmux. PY overrides the interpreter. Logs: logs/paper/x4_<step>_<stamp>.log.
# Outputs: results/x4_llava/ (+ the pipeline's own results/image_head_pipeline_llava-1.5-7b-hf/
# and logs/image_head_pipeline_llava-1.5-7b-hf/). The accuracy sweep and prune_tuples write new
# files under ../VADE/models/llava-1.5-7b-hf/ (VADE's own convention, as for E8 and X3); no VADE
# code is touched -- the sweep runs unmodified through methods/vade_accuracy_sweep.py.
# GPU=a100 (any 80GB card) raises batch sizes / pair counts exactly as run_x3_idefics3.sh does.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VADE=${VADE_ROOT:-../VADE}
M=${X4_MODEL:-llava-hf/llava-1.5-7b-hf}
SLUG=${M##*/}
NL=${X4_N_LAYERS:-32} NH=${X4_N_HEADS:-32}      # 7B; the 13B is 40 x 40
N_CEIL=${X4_N_CEIL:-128}                        # stage-1 rows per curve (X3 used 32)
N_ROUTER=${X4_N_ROUTER_COUNTRIES:-16}           # attribute-switch items (X3 used the default 8)
R=results/x4_llava
P=results/image_head_pipeline_$SLUG
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/paper "$R"

BS_PIPE= BS_ACC= BS_SEV= BS_TEC=
N_SEV=64
if [ "${GPU:-}" = a100 ]; then
  # PIPE stays at 32: head_trace/head_window_sweep's shuffled-donor control rolls rows WITHIN a batch.
  BS_PIPE=32 BS_ACC=128 BS_SEV=64 BS_TEC=64 N_SEV=128
  echo "GPU=a100 batch profile: pipeline=$BS_PIPE accuracy=$BS_ACC severed=$BS_SEV tec=$BS_TEC; severed n_pairs=$N_SEV"
fi
bs() { [ -n "$1" ] && echo "--batch_size $1"; return 0; }

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/paper/x4_${step}_${STAMP}.log"
  echo "=== [x4_$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [x4_$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

trace_of() {  # trace_of <entity> <attribute> -> that attribute's head_trace JSON from the pipeline
  local f
  f=$(ls "$P/$1/$2"/head_trace_*.json 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "no head_trace JSON under $P/$1/$2 -- run the pipeline step first" >&2; exit 1; }
  echo "$f"
}

consensus() {  # consensus <min count> -> heads in >= that many flags top-8s, "b.h,..." (X1's movers)
  $PY - "$P" "$1" $FLAGS_ATTRS <<'EOF'
import collections, glob, json, sys
root, need, attrs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
count = collections.Counter()
for a in attrs:
    paths = sorted(glob.glob(f"{root}/flags/{a}/head_trace_*.json"))
    if paths:
        count.update(tuple(h) for h in json.load(open(paths[0]))["phase1_ranked"][:8])
heads = sorted(h for h, c in count.items() if c >= need)
assert len(heads) >= 4, f"only {len(heads)} heads in >= {need} flags top-8s ({dict(count)}) -- no consensus set"
print(",".join(f"{b}.{h}" for b, h in heads))
EOF
}

FLAGS_ATTRS="capital currency language calling_code"
ACC=(--model_id "$M" $(bs "$BS_ACC"))           # no --processor_kwargs: 336px canvas = native input
WIN=()
[ -n "${X4_PATCH:-}" ] && WIN+=(--patch_layer "$X4_PATCH")
# shellcheck disable=SC2206
[ -n "${X4_BLOCKS:-}" ] && WIN+=(--blocks $X4_BLOCKS)

# ---------------------------------------------------------------- smoke (numbers mean nothing)

smoke() {
  local S=$R/smoke
  run smoke_accuracy $PY methods/vade_accuracy_sweep.py "${ACC[@]}" --entities flags --attributes capital \
      --out_dir "$S/accuracy"
  run smoke_tests $PY -m pytest -q -rs tests/test_llava_adapter.py
  run smoke_ceiling $PY methods/ndm/ceiling_sweep.py --model_id "$M" --entity flags --attribute capital \
      --allow_unpruned --sites residual --positions_list full_image last_token --layers 0 16 32 --n_rows 4
  run smoke_trace $PY methods/head_trace.py --model_id "$M" --entity flags --attribute capital --allow_unpruned \
      --patch_layer 20 --positions full_image --blocks 20 21 --n_rows 8 --batch_size 4 --out "$S/head_trace.json"
  run smoke_switch $PY methods/attribute_switch_sweep.py --model_id "$M" --own_golds --n_countries 1 \
      --sites residual --scopes earlier_text last_token --block_spans --layers 16 --controls self \
      --out_dir "$S/switch"
  run smoke_attr_trace $PY methods/attr_head_trace.py --model_id "$M" --own_golds --entity flags \
      --patch_layer 16 --blocks 16 17 --n_countries 2 --donor_attributes capital --out "$S/attr_trace.json"
  run smoke_severed $PY methods/head_severed.py --model_id "$M" --entity flags --attributes capital --allow_unpruned \
      --heads 20.0,21.1 --n_pairs 4 --gold own --freeze_values base --freeze_mlp_span 22-31 \
      --freeze_mlp_singles 31 --freeze_attn_span '' --out_dir "$S/severed"
  run smoke_crossmodal $PY methods/text_entity_contrast.py --model_id "$M" --movers 20.0,21.1,22.2,23.3 \
      --phases crossmodal --n_pairs 4 --out_dir "$S/crossmodal"
}

# ---------------------------------------------------------------- the cheap gate, then prep

gate() {  # one attribute per entity the pipeline traces; resumes into the SAME predictions file as prep
  run gate_accuracy $PY methods/vade_accuracy_sweep.py "${ACC[@]}" --entities flags brands \
      --attributes capital language hq_country
  run gate_check $PY - "$VADE/models/$SLUG/stats" <<'EOF'
import json, sys
stats, floor = sys.argv[1], 50.0
need = {("flags", "capital"), ("flags", "language"), ("brands", "hq_country")}
ok = True
for entity in ("flags", "brands"):
    s = json.load(open(f"{stats}/{entity}_summary.json"))
    for row in s["by_attribute"]:
        if (entity, row["attribute"]) in need:
            good = row["pct_accuracy"] >= floor
            ok &= good
            print(f"  {entity}/{row['attribute']}: {row['pct_accuracy']:.1f}% (n={row['n']}) "
                  f"{'ok' if good else f'BELOW {floor:.0f}%'}")
print("GATE PASSED" if ok else "GATE FAILED: too few eligible rows to trace -- stop X4 here and report")
sys.exit(0 if ok else 1)
EOF
}

prep() {  # full accuracy sweep (resumes the gate's rows) -> pruned tuples
  run accuracy $PY methods/vade_accuracy_sweep.py "${ACC[@]}" --entities flags brands
  local attrs
  attrs=$(find "$VADE"/data/{flags,brands}/tuples -mindepth 1 -maxdepth 1 -type d ! -name identity \
          -printf '%f\n' | sort -u)
  run prune $PY "$VADE/models/prune_tuples.py" --model_id "$M" --entities flags brands --attributes $attrs
}

# ---------------------------------------------------------------- movers (image -> text heads)

pipeline() {
  # X4_SCORING=text if flags' scoring guard fails on casing (LLaVA may not write flags labels as
  # stored); auto = token for flags, text for brands, as for every earlier model.
  local C=(--model_id "$M" --n_layers $NL --n_heads $NH --n_rows_ceiling $N_CEIL --scoring "${X4_SCORING:-auto}"
           $(bs "$BS_PIPE") "${WIN[@]}")
  [ ${#WIN[@]} -gt 0 ] && echo "x4 pipeline: window OVERRIDDEN by X4_PATCH/X4_BLOCKS: ${WIN[*]}"
  run pipeline_flags $PY methods/image_head_pipeline.py --entity flags --attributes $FLAGS_ATTRS "${C[@]}"
  run pipeline_brands $PY methods/image_head_pipeline.py --entity brands --attributes hq_country "${C[@]}"
}

overlap() {  # WITHIN LLaVA: do flags and brands share movers?
  run overlap $PY methods/head_overlap.py --model_id "$M" --n_layers $NL --n_heads $NH \
      --pipeline_root "$P" --flags_trace_glob '' --out_dir "$R/head_overlap"
}

depth() {  # handoff / mover window as a fraction of depth, next to every model so far (CPU)
  run depth $PY methods/circuit_windows.py depth --out "$R/depth.json" --root "$P:$NL" \
      results/image_head_pipeline:28 results/image_head_pipeline_InternVL3-8B-hf:28 \
      results/image_head_pipeline_Qwen2.5-VL-3B-Instruct:36 results/image_head_pipeline_Idefics3-8B-Llama3:32
}

# ---------------------------------------------------------------- routers (question -> readout heads)

routers() {  # flags, own golds (the model's own clean answers; LLaVA's casing need not match the labels)
  run switch_baselines $PY methods/attribute_switch_sweep.py --model_id "$M" --own_golds --baselines_only \
      --n_countries $N_ROUTER --out_dir "$R/switch_baselines"
  run switch_gate $PY - "$R/switch_baselines" <<'EOF'
import sys
sys.path.insert(0, ".")
from methods.attribute_switch_report import baselines
b = baselines(sys.argv[1])
for a, v in sorted(b["clean_by_attribute"].items()):
    print(f"  clean {a}: full {v['full']:.1%} first {v['first']:.1%} (n={v['n']})")
print(f"  eligible rows (base AND donor clean-correct): {b['eligible']}/{b['rows_configured']}")
if b["eligible"] < 24:
    print("ROUTER GATE FAILED: fewer than 24 eligible rows -- skip the router stage and report")
    sys.exit(1)
print("ROUTER GATE PASSED")
EOF
  run switch_sweep $PY methods/attribute_switch_sweep.py --model_id "$M" --own_golds --sites residual \
      --scopes earlier_text last_token --block_spans --layers $(seq 1 $NL) --controls self \
      --n_countries $N_ROUTER --out_dir "$R/switch_residual"
  local W
  if [ -n "${X4_ROUTER_PATCH:-}" ]; then
    W="--patch_layer $X4_ROUTER_PATCH --blocks ${X4_ROUTER_BLOCKS:?set X4_ROUTER_BLOCKS with X4_ROUTER_PATCH}"
    echo "router window OVERRIDDEN by X4_ROUTER_PATCH/X4_ROUTER_BLOCKS: $W"
    $PY methods/circuit_windows.py router --switch_dir "$R/switch_residual" --n_layers $NL \
        --out "$R/router_window_rule.json" || true                 # still record what the rule says
  else
    W=$($PY methods/circuit_windows.py router --switch_dir "$R/switch_residual" --n_layers $NL \
          --out "$R/router_window.json")
    echo "router window from this model's crossover: $W"
  fi
  # shellcheck disable=SC2086
  run attr_trace $PY methods/attr_head_trace.py --model_id "$M" --own_golds --entity flags $W \
      --n_countries $N_ROUTER --out "$R/attr_heads/flags_trace.json"
}

# ---------------------------------------------------------------- MLP lookup (severed tracing)

severed() {  # each attribute's own top-8 movers; freeze spans from their deepest block
  local a A
  for a in ${X4_SEVERED_ATTRS:-$FLAGS_ATTRS}; do
    A=$($PY methods/circuit_windows.py severed --trace "$(trace_of flags $a)" --n_layers $NL \
          --out "$R/severed_args/flags_$a.json")
    echo "flags/$a severed args: $A"
    # shellcheck disable=SC2086
    run "severed_flags_$a" $PY methods/head_severed.py --model_id "$M" --entity flags --attributes $a $A \
        --n_pairs $N_SEV --gold own --freeze_values base mean other $(bs "$BS_SEV") \
        --out_dir "$R/severed/flags_$a"
  done
  A=$($PY methods/circuit_windows.py severed --trace "$(trace_of brands hq_country)" --n_layers $NL \
        --out "$R/severed_args/brands_hq_country.json")
  echo "brands/hq_country severed args: $A"
  # shellcheck disable=SC2086
  run severed_brands_hq_country $PY methods/head_severed.py --model_id "$M" --entity brands \
      --attributes hq_country $A --n_pairs $N_SEV --gold own --freeze_values base mean other $(bs "$BS_SEV") \
      --out_dir "$R/severed/brands_hq_country"
}

# ---------------------------------------------------------------- X1 on LLaVA: image vs text mover content

crossmodal() {  # movers = heads in >= 3 of the 4 flags top-8s (or X4_MOVERS), never Qwen's common10
  local MV=${X4_MOVERS:-}
  [ -n "$MV" ] || MV=$(consensus 3)
  echo "x4 crossmodal movers: $MV"
  run crossmodal $PY methods/text_entity_contrast.py --model_id "$M" --movers "$MV" --phases crossmodal \
      $(bs "$BS_TEC") --out_dir "$R/crossmodal_flags"
}

[ $# -gt 0 ] || { echo "usage: $0 smoke|gate|prep|pipeline|overlap|depth|routers|severed|crossmodal|all ..."; exit 1; }
for step in "$@"; do
  case $step in
    all) gate; prep; pipeline; overlap; depth; routers; severed; crossmodal ;;
    smoke|gate|prep|pipeline|overlap|depth|routers|severed|crossmodal) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
