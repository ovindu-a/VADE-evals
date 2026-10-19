#!/usr/bin/env bash
# X3 (PAPER_RUNBOOK.md): the mover / router / MLP-lookup story on Idefics3-8B-Llama3, a Llama-3.1-8B
# language model -- the first non-Qwen text stack here. Mirrors run_tier2.sh's e8_internvl, plus the
# router and MLP stages, with every head/block argument DERIVED from this model's own runs
# (methods/circuit_windows.py), never Qwen's.
#
#   bash scripts/run_x3_idefics3.sh smoke        # every stage at tiny size on the real model (~15 min)
#   bash scripts/run_x3_idefics3.sh gate         # cheap accuracy gate (~15 min); STOP if it fails
#   bash scripts/run_x3_idefics3.sh prep pipeline overlap routers severed depth
#   bash scripts/run_x3_idefics3.sh all          # gate, then everything above, in that order
#   X3_SEVERED_ATTRS="capital language" bash scripts/run_x3_idefics3.sh severed   # skip voided traces
#
# Run inside tmux. PY overrides the interpreter. Logs: logs/paper/x3_<step>_<stamp>.log.
# Outputs: results/x3_idefics3/ (+ the pipeline's own results/image_head_pipeline_Idefics3-8B-Llama3/
# and logs/image_head_pipeline_Idefics3-8B-Llama3/), and VADE/models/Idefics3-8B-Llama3/ (accuracy
# predictions + pruned tuples, VADE's own convention, as for E8).
# GPU=a100 (any 80GB card) raises batch sizes / pair counts exactly as run_tier2.sh does.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VADE=${VADE_ROOT:-../VADE}
M=HuggingFaceM4/Idefics3-8B-Llama3
SLUG=Idefics3-8B-Llama3
NL=32 NH=32
PK='{"do_image_splitting": false}'      # the adapter's single 364px tile; pruning must see the same input. Passed by
                                        # methods/vade_accuracy_sweep.py, which runs VADE's sweep UNMODIFIED.
R=results/x3_idefics3
P=results/image_head_pipeline_$SLUG
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/paper "$R"

BS_PIPE= BS_ACC= BS_SEV=
N_SEV=64
if [ "${GPU:-}" = a100 ]; then
  # PIPE stays at 32: head_trace/head_window_sweep's shuffled-donor control rolls rows WITHIN a batch.
  BS_PIPE=32 BS_ACC=128 BS_SEV=64 N_SEV=128
  echo "GPU=a100 batch profile: pipeline=$BS_PIPE accuracy=$BS_ACC severed=$BS_SEV; severed n_pairs=$N_SEV"
fi
bs() { [ -n "$1" ] && echo "--batch_size $1"; return 0; }

run() {  # run <step> <command...>
  local step=$1; shift
  local log="logs/paper/x3_${step}_${STAMP}.log"
  echo "=== [x3_$step] $* ==="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  echo "    -> [x3_$step] done in $(( (SECONDS - t0) / 60 )) min ($log)"
}

trace_of() {  # trace_of <entity> <attribute> -> that attribute's head_trace JSON from the pipeline
  local f
  f=$(ls "$P/$1/$2"/head_trace_*.json 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "no head_trace JSON under $P/$1/$2 -- run the pipeline step first" >&2; exit 1; }
  echo "$f"
}

FLAGS_ATTRS="capital currency language calling_code"
ACC=(--model_id "$M" --processor_kwargs "$PK" $(bs "$BS_ACC"))

# ---------------------------------------------------------------- smoke (numbers mean nothing)

smoke() {
  local S=$R/smoke
  # accuracy first: it downloads the model, so the tokenizer tests below RUN instead of skipping
  run smoke_accuracy $PY methods/vade_accuracy_sweep.py "${ACC[@]}" --entities flags --attributes capital \
      --out_dir "$S/accuracy"
  run smoke_tests $PY -m pytest -q -rs tests/test_idefics3_adapter.py
  run smoke_ceiling $PY methods/ndm/ceiling_sweep.py --model_id "$M" --entity flags --attribute capital \
      --allow_unpruned --sites residual --positions_list full_image last_token --layers 0 16 32 --n_rows 4 \
      --out "$S/ceiling.json"
  run smoke_trace $PY methods/head_trace.py --model_id "$M" --entity flags --attribute capital --allow_unpruned \
      --patch_layer 20 --positions full_image --blocks 20 21 --n_rows 8 --batch_size 4 --out "$S/head_trace.json"
  run smoke_switch $PY methods/attribute_switch_sweep.py --model_id "$M" --n_countries 1 --sites residual \
      --scopes earlier_text last_token --block_spans --layers 16 --controls self --out_dir "$S/switch"
  run smoke_attr_trace $PY methods/attr_head_trace.py --model_id "$M" --entity flags --patch_layer 16 --blocks 16 17 \
      --n_countries 2 --donor_attributes capital --out "$S/attr_trace.json"
  run smoke_severed $PY methods/head_severed.py --model_id "$M" --entity flags --attributes capital --allow_unpruned \
      --heads 20.0,21.1 --n_pairs 4 --gold own --freeze_values base --freeze_mlp_span 22-31 \
      --freeze_mlp_singles 31 --freeze_attn_span '' --out_dir "$S/severed"
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
print("GATE PASSED" if ok else "GATE FAILED: too few eligible rows to trace -- stop X3 here and report")
sys.exit(0 if ok else 1)
EOF
}

prep() {  # full accuracy sweep (resumes the gate's rows) -> pruned tuples
  run accuracy $PY methods/vade_accuracy_sweep.py "${ACC[@]}" --entities flags brands
  # identity/ has no `rule` field, which prune_tuples.py cannot read (same as run_tier2.sh prep_model).
  local attrs
  attrs=$(find "$VADE"/data/{flags,brands}/tuples -mindepth 1 -maxdepth 1 -type d ! -name identity \
          -printf '%f\n' | sort -u)
  run prune $PY "$VADE/models/prune_tuples.py" --model_id "$M" --entities flags brands --attributes $attrs
}

# ---------------------------------------------------------------- movers (image -> text heads)

pipeline() {
  local C=(--model_id "$M" --n_layers $NL --n_heads $NH $(bs "$BS_PIPE"))
  run pipeline_flags $PY methods/image_head_pipeline.py --entity flags --attributes $FLAGS_ATTRS "${C[@]}"
  run pipeline_brands $PY methods/image_head_pipeline.py --entity brands --attributes hq_country "${C[@]}"
}

overlap() {  # WITHIN Idefics3: do flags and brands share movers? (head IDs do not transfer across models)
  run overlap $PY methods/head_overlap.py --model_id "$M" --n_layers $NL --n_heads $NH \
      --pipeline_root "$P" --flags_trace_glob '' --out_dir "$R/head_overlap"
}

depth() {  # where each model's handoff / mover window sits, as a fraction of its depth (CPU)
  run depth $PY methods/circuit_windows.py depth --out "$R/depth.json" --root "$P:$NL" \
      results/image_head_pipeline:28 results/image_head_pipeline_InternVL3-8B-hf:28 \
      results/image_head_pipeline_Qwen2.5-VL-3B-Instruct:36
}

# ---------------------------------------------------------------- routers (question -> readout heads)

routers() {  # flags only: the controlled prompts are flags-only until X2b lands
  run switch_baselines $PY methods/attribute_switch_sweep.py --model_id "$M" --baselines_only \
      --out_dir "$R/switch_baselines"
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
  # The crossover: question-token vs last-token residual swap at every layer (ATTRIBUTE_SWITCH_FINDINGS sec. 1).
  run switch_sweep $PY methods/attribute_switch_sweep.py --model_id "$M" --sites residual \
      --scopes earlier_text last_token --block_spans --layers $(seq 1 $NL) --controls self \
      --out_dir "$R/switch_residual"
  local W
  W=$($PY methods/circuit_windows.py router --switch_dir "$R/switch_residual" --n_layers $NL \
        --out "$R/router_window.json")
  echo "router window from this model's crossover: $W"
  run attr_trace $PY methods/attr_head_trace.py --model_id "$M" --entity flags $W \
      --out "$R/attr_heads/flags_trace.json"
}

# ---------------------------------------------------------------- MLP lookup (severed tracing)

severed() {  # each attribute's own top-8 movers; freeze spans from their deepest block
  # X3_SEVERED_ATTRS="capital language" drops flags attributes whose trace failed a self-check.
  local a A
  for a in ${X3_SEVERED_ATTRS:-$FLAGS_ATTRS}; do
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

[ $# -gt 0 ] || { echo "usage: $0 smoke|gate|prep|pipeline|overlap|depth|routers|severed|all ..."; exit 1; }
for step in "$@"; do
  case $step in
    all) gate; prep; pipeline; overlap; depth; routers; severed ;;
    smoke|gate|prep|pipeline|overlap|depth|routers|severed) $step ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac
done
