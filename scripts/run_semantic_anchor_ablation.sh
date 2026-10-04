#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/train.yaml}"
TRAIN_DATA="${TRAIN_DATA:-/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt}"
TEST_DATA="${TEST_DATA:-/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/semantic_anchor_ablation}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SAVE_EVERY="${SAVE_EVERY:-20}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-1}"

# Optional whitespace-separated NAME=PATH entries, for example:
# UNSEEN_NF_TESTS="nf64=/data/upa8x8_nf64_test.pt nf96=/data/upa8x8_nf96_test.pt"
UNSEEN_NF_TESTS="${UNSEEN_NF_TESTS:-}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "train_data_not_found=$TRAIN_DATA" >&2
  exit 1
fi
if [[ ! -f "$TEST_DATA" ]]; then
  echo "test_data_not_found=$TEST_DATA" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

evaluate_split() {
  local checkpoint="$1"
  local split_name="$2"
  local data_path="$3"
  local run_dir="$4"
  local eval_dir="$run_dir/evaluation/$split_name"

  if [[ ! -f "$data_path" ]]; then
    echo "evaluation_data_not_found=$data_path" >&2
    exit 1
  fi
  mkdir -p "$eval_dir"

  echo "evaluation_start split=$split_name data_path=$data_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u scripts/evaluate.py \
    --data-path "$data_path" \
    --checkpoint "$checkpoint" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --verbose-diagnostics \
    --signal-description-correction relational \
    --save-signal-descriptions "$eval_dir/signal_descriptions.pt" \
    --save-regression-pairs "$eval_dir/regression_pairs.csv" \
    --save-prototype-diagnostics "$eval_dir/prototype_diagnostics.csv" \
    2>&1 | tee "$eval_dir/evaluate_output.txt"
  echo "model_evaluation_complete split=$split_name"

  python -u scripts/evaluate_signal_descriptions.py \
    --payload "$eval_dir/signal_descriptions.pt" \
    --output-dir "$eval_dir" \
    2>&1 | tee "$eval_dir/text_evaluate_output.txt"
  echo "text_evaluation_complete split=$split_name"
}

for seed in $SEEDS; do
  for mode_spec in \
    "fixed_semantic_anchors:fixed" \
    "adaptive_prototypes:adaptive_prototypes" \
    "joint_semantic_learning:joint"
  do
    variant="${mode_spec%%:*}"
    anchor_mode="${mode_spec#*:}"
    run_dir="$OUTPUT_ROOT/$variant/seed_$seed"
    mkdir -p "$run_dir"
    checkpoint="$run_dir/checkpoint_last.pt"

    if [[ "$REUSE_CHECKPOINTS" == "1" && -f "$checkpoint" ]]; then
      echo "reusing_checkpoint=$checkpoint"
    else
      echo "=== train variant=$variant seed=$seed mode=$anchor_mode ==="
      CUDA_VISIBLE_DEVICES="$GPU_ID" python -u scripts/pretrain.py \
        --config "$CONFIG" \
        --data-path "$TRAIN_DATA" \
        --output-dir "$run_dir" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --save-every "$SAVE_EVERY" \
        --seed "$seed" \
        --semantic-anchor-mode "$anchor_mode" \
        --first-path-aod-weight 0 \
        --first-path-aod-nlos-weight 0 \
        2>&1 | tee "$run_dir/train.log"
      echo "training_complete variant=$variant seed=$seed"
    fi

    if [[ ! -f "$checkpoint" ]]; then
      echo "checkpoint_not_found=$checkpoint" >&2
      exit 1
    fi

    evaluate_split "$checkpoint" "train_nf128" "$TRAIN_DATA" "$run_dir"
    evaluate_split "$checkpoint" "test_nf128" "$TEST_DATA" "$run_dir"

    for unseen_spec in $UNSEEN_NF_TESTS; do
      unseen_name="${unseen_spec%%=*}"
      unseen_path="${unseen_spec#*=}"
      if [[ "$unseen_name" == "$unseen_spec" || -z "$unseen_path" ]]; then
        echo "invalid_UNSEEN_NF_TESTS_entry=$unseen_spec" >&2
        exit 1
      fi
      evaluate_split "$checkpoint" "unseen_$unseen_name" "$unseen_path" "$run_dir"
    done
  done
done

grep -H -E \
  '^(first_path_delay_context_MAE|first_path_angle_MAE|first_path_angle_los_MAE|first_path_angle_nlos_MAE|final_first_power_MAE|los_first_power_MAE|nlos_first_power_MAE|physical_description_los_status_accuracy|csi_to_learnable_prototype_R@1|csi_to_learnable_prototype_MRR|prototype_to_initial_text_cosine_mean|prototype_drift_Dp|prototype_to_final_text_cosine_mean|final_text_to_initial_text_cosine_mean)=' \
  "$OUTPUT_ROOT"/*/seed_*/evaluation/*/evaluate_output.txt \
  > "$OUTPUT_ROOT/key_metrics_summary.txt"

{
  for metrics_path in "$OUTPUT_ROOT"/*/seed_*/evaluation/*/signal_description_text_metrics.csv; do
    [[ -f "$metrics_path" ]] || continue
    echo "=== $metrics_path ==="
    grep -E \
      'description_factual_accuracy|entire_record_factual_accuracy|physical_consistency_rate,predicted_description' \
      "$metrics_path"
  done
} > "$OUTPUT_ROOT/description_factuality_summary.txt"

echo "saved_key_metrics_summary=$OUTPUT_ROOT/key_metrics_summary.txt"
echo "saved_description_factuality_summary=$OUTPUT_ROOT/description_factuality_summary.txt"
