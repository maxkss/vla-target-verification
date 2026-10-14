#!/usr/bin/env bash
# One experiment from start to end, or the rest of an existing run.
#
#   ./run.sh experiments/<name>.json [--only PATTERNS]   new run (PATTERNS: file of task/sentence/initNN globs)
#   ./run.sh <run folder> [--from STAGE]                 continue a run; --from also redoes STAGE and later
#
# Stages: collect label select_head segment task_objects track alarm evaluate.
# Every stage skips what it already wrote, so an interrupted run continues where it stopped.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env/local.sh"
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$here"
# Determinism: no XLA autotuning (otherwise kernel choice, and so the actions, can differ between processes) and
# deterministic XLA ops (no numeric change for this model, a guard); deterministic cuDNN / cuBLAS for SAM and the
# VLM; fixed hash seed; models read from the local cache only.
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.19   # ~9 GB per policy process
export MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 SAM2_BUILD_CUDA=0
read -r -a gpus <<< "${GPUS:-0}"
STAGES=(collect label select_head segment task_objects track alarm evaluate)

target=$1; shift
only="" from=""
while [ $# -gt 0 ]; do
  case $1 in
    --only) only=$2; shift 2 ;;
    --from) from=$2; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

if [ -f "$target" ]; then
  run=$("$POLICY_PYTHON" -c "
import sys, common
only = [l.strip() for l in open(sys.argv[2]) if l.strip() and not l.startswith('#')] if sys.argv[2] else None
print(common.Run.create(sys.argv[1], only).path)" "$target" "$only")
  echo "new run: $run"
else
  run=$(cd "$target" && pwd)
fi
mkdir -p "$run/logs"

stage_outputs() {   # files a stage writes, for --from
  case $1 in
    label) echo "label.json" ;;
    select_head) echo "head.json head.png" ;;
    segment) echo "candidates.npz" ;;
    task_objects) echo "task_objects.json numbered.png" ;;
    track) echo "tracks.npz" ;;
    alarm) echo "alarm.json" ;;
    evaluate) echo "summary.json summary.md summary.png rows.json" ;;
  esac
}
if [ -n "$from" ]; then
  [ "$from" = collect ] && { echo "--from collect would discard every rollout: start a new run"; exit 2; }
  redo=0
  for s in "${STAGES[@]}"; do
    [ "$s" = "$from" ] && redo=1
    if [ $redo = 1 ]; then
      for f in $(stage_outputs "$s"); do find "$run" -name "$f" -delete; done
      rm -f "$run/stages/$s".*json "$run/stages/$s.json"
    fi
  done
fi

parallel() {   # parallel STAGE PYTHON PER_GPU: one worker per slot, a failing worker stops the run
  local stage=$1 python=$2 per_gpu=$3 pids=() n=$(( ${#gpus[@]} * $3 )) i=0
  for g in "${gpus[@]}"; do
    for _ in $(seq "$per_gpu"); do
      CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g "$python" "$here/$stage.py" "$run" --worker "$i/$n" \
        > "$run/logs/$stage.w$i.log" 2>&1 &
      pids+=($!); i=$((i + 1))
    done
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "$stage failed (see $run/logs/$stage.w*.log)"; exit 1; }; done
}

skip=1; [ -z "$from" ] && skip=0
for s in "${STAGES[@]}"; do
  [ "$s" = "$from" ] && skip=0
  [ $skip = 1 ] && continue
  echo "$(date '+%F %T') $s"
  case $s in
    collect) parallel collect "$POLICY_PYTHON" 2 ;;              # 2 policy processes per GPU (~9 GB each)
    segment) parallel segment "$SAM_PYTHON" 1 ;;
    task_objects) parallel task_objects "$VLM_PYTHON" 1 ;;      # one Qwen3.5-9B (~19 GB) per GPU
    track) parallel track "$SAM_PYTHON" 1 ;;
    *) CUDA_VISIBLE_DEVICES="" "$POLICY_PYTHON" "$here/$s.py" "$run" > "$run/logs/$s.log" 2>&1 \
         || { echo "$s failed (see $run/logs/$s.log)"; exit 1; } ;;
  esac
done
echo "$(date '+%F %T') done: $run/results/summary.md"
