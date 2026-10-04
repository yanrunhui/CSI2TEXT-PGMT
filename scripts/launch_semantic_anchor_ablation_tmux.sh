#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-semantic_anchor_ablation_seed0}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/semantic_anchor_ablation_seed${SEED}}"
TRAIN_DATA="${TRAIN_DATA:-/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt}"
TEST_DATA="${TEST_DATA:-/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt}"
UNSEEN_NF_TESTS="${UNSEEN_NF_TESTS:-}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux_not_found=1" >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux_session_already_exists=$SESSION_NAME" >&2
  echo "attach_command=tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

if [[ "$OUTPUT_ROOT" = /* ]]; then
  output_path="$OUTPUT_ROOT"
else
  output_path="$ROOT/$OUTPUT_ROOT"
fi
mkdir -p "$output_path"
tmux_log="$output_path/tmux.log"

printf -v run_command \
  'set -o pipefail; cd %q && GPU_ID=%q SEEDS=%q EPOCHS=%q BATCH_SIZE=%q EVAL_BATCH_SIZE=%q OUTPUT_ROOT=%q TRAIN_DATA=%q TEST_DATA=%q UNSEEN_NF_TESTS=%q bash scripts/run_semantic_anchor_ablation.sh 2>&1 | tee %q' \
  "$ROOT" \
  "$GPU_ID" \
  "$SEED" \
  "$EPOCHS" \
  "$BATCH_SIZE" \
  "$EVAL_BATCH_SIZE" \
  "$OUTPUT_ROOT" \
  "$TRAIN_DATA" \
  "$TEST_DATA" \
  "$UNSEEN_NF_TESTS" \
  "$tmux_log"

tmux new-session -d -s "$SESSION_NAME" "$run_command"

echo "tmux_session=$SESSION_NAME"
echo "tmux_log=$tmux_log"
echo "attach_command=tmux attach -t $SESSION_NAME"
echo "tail_command=tail -f $tmux_log"
