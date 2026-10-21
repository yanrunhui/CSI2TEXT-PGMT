# 基于预测总接收功率的链路中断判断

该目录实现一条完全可解释、无下游黑盒分类器的链路：

```text
CSI -> 预测总接收功率/dBW -> 固定接收机噪声 -> 预测 SNR -> Normal/Outage
```

模型只预测总接收功率。真实与预测中断状态均由同一个固定规则生成：

```text
SNR_true = P_total,true,dBW - P_noise,dBW
SNR_pred = P_total,pred,dBW - P_noise,dBW
Outage = SNR < required_snr_db
```

默认噪声参数为 100 MHz、噪声系数 7 dB、热噪声谱密度 -174 dBm/Hz，因此噪声功率为 -87 dBm（-117 dBW）；默认所需 SNR 为 5 dB。

## 1. 将原 test10k 均衡拆分为 validation5k/test5k

```bash
cd /home/yrh/CSI_model/CSI_model

python -m downstream_outage_prediction.prepare_data \
  --input artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt \
  --output-dir downstream_outage_prediction/artifacts/seed23421 \
  --validation-per-status 2500 \
  --test-per-status 2500 \
  --seed 23421
```

这会生成：

- `validation.pt`：2500 LoS + 2500 NLoS；
- `test.pt`：2500 LoS + 2500 NLoS；
- `split_manifest.json`：数量、哈希和跨集合重叠检查。

## 2. 在 validation5k 上确认中断比例

```bash
python -m downstream_outage_prediction.run_experiment \
  --data downstream_outage_prediction/artifacts/seed23421/validation.pt \
  --checkpoint artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt \
  --output-dir downstream_outage_prediction/outputs/seed23421/validation \
  --bandwidth-hz 100000000 \
  --noise-figure-db 7 \
  --required-snr-db 5 \
  --batch-size 128
```

先检查 `true_outage_fraction_pct`，确认 5 dB 门限没有造成几乎全 Normal 或全 Outage。若需要确定非零 `--safety-margin-db`，只能根据 validation 结果选择，选定后不要再根据 test 调参。

## 3. 固定参数后评估 test5k

```bash
python -m downstream_outage_prediction.run_experiment \
  --data downstream_outage_prediction/artifacts/seed23421/test.pt \
  --checkpoint artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt \
  --output-dir downstream_outage_prediction/outputs/seed23421/test \
  --bandwidth-hz 100000000 \
  --noise-figure-db 7 \
  --required-snr-db 5 \
  --safety-margin-db 0 \
  --batch-size 128
```

`--safety-margin-db` 默认是 0。设为正数会让预测规则更保守，一般降低漏检、增加误报。真实标签始终使用原始 `--required-snr-db`，不会随安全裕量改变。

## 输出

每次运行生成：

- `metrics.json`：功率 MAE/RMSE/Bias/Pearson，中断 Accuracy、Recall、F1、漏检率、误报率，以及 LoS/NLoS 分组结果；
- `predictions.csv`：逐样本功率、SNR、中断状态和错误类型；
- `predictions.npz`：无损数组结果；
- `power_scatter.png`：预测与真实总功率散点图；
- `outage_confusion_matrix.png`：Normal/Outage 混淆矩阵；
- `snr_histogram.png`：真实与预测 SNR 分布。

其中最重要的安全指标是 `missed_outage_rate_pct`：真实为 Outage、预测为 Normal 的比例。

## 4. LLM baseline：CSI -> 总功率 -> Outage

这个 baseline 不是 beam selection。它用完整 CSI 训练 LLM adapter/head 直接回归
`total_received_power_dbw`，再使用与上面完全相同的噪声、SNR 和 outage 判决规则。

先从原始训练集生成两个 LoS/NLoS 均衡训练子集：

```bash
cd /home/yrh/CSI_model/CSI_model

python -m downstream_outage_prediction.make_training_subsets \
  --input artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt \
  --output-dir downstream_outage_prediction/artifacts/llm_subsets_seed23421 \
  --train-sizes 1000 10000 \
  --seed 23421
```

再从原验证/测试集各抽一个 LoS/NLoS 均衡的 smoke 子集：

```bash
python -m downstream_outage_prediction.make_eval_subsets \
  --validation-input downstream_outage_prediction/artifacts/seed23421/validation.pt \
  --test-input downstream_outage_prediction/artifacts/seed23421/test.pt \
  --output-dir downstream_outage_prediction/artifacts/smoke_eval_seed23421 \
  --validation-size 500 \
  --test-size 500 \
  --seed 23421
```

公共路径：

```bash
TRAIN_1K=downstream_outage_prediction/artifacts/llm_subsets_seed23421/train_1000.pt
TRAIN_10K=downstream_outage_prediction/artifacts/llm_subsets_seed23421/train_10000.pt
VAL=downstream_outage_prediction/artifacts/seed23421/validation.pt
TEST=downstream_outage_prediction/artifacts/seed23421/test.pt
SMOKE_VAL=downstream_outage_prediction/artifacts/smoke_eval_seed23421/validation_500.pt
SMOKE_TEST=downstream_outage_prediction/artifacts/smoke_eval_seed23421/test_500.pt
OUT=downstream_outage_prediction/outputs/llm_outage_seed23421
mkdir -p "$OUT/logs"
```

1k 只用于检查输出是否正常，使用 `train1k + val500 + test500`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_1K" --val-data "$SMOKE_VAL" --test-data "$SMOKE_TEST" \
  --model deepseek_r1_qwen3_8b \
  --model-path models/DeepSeek-R1-0528-Qwen3-8B \
  --output-dir "$OUT/deepseek_r1_qwen3_8b_train1k" \
  --backbone-training frozen --quantization 4bit \
  --epochs 1 --patience 1 --batch-size 1 --grad-accum 16 \
  --learning-rate 2e-4 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/deepseek_r1_qwen3_8b_train1k.log"
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_1K" --val-data "$SMOKE_VAL" --test-data "$SMOKE_TEST" \
  --model internlm3_8b \
  --model-path models/InternLM3-8B-Instruct \
  --output-dir "$OUT/internlm3_8b_train1k" \
  --backbone-training frozen --quantization 4bit \
  --epochs 1 --patience 1 --batch-size 1 --grad-accum 16 \
  --learning-rate 2e-4 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/internlm3_8b_train1k.log"
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_1K" --val-data "$SMOKE_VAL" --test-data "$SMOKE_TEST" \
  --model qwen3_5_2b \
  --model-path models/Qwen3.5-2B \
  --output-dir "$OUT/qwen3_5_2b_train1k" \
  --backbone-training full --quantization none \
  --epochs 1 --patience 1 --batch-size 1 --grad-accum 16 \
  --learning-rate 2e-4 --backbone-learning-rate 1e-5 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/qwen3_5_2b_train1k.log"
```

10k 用于正式训练，使用 `train10k + val5k + test5k`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_10K" --val-data "$VAL" --test-data "$TEST" \
  --model deepseek_r1_qwen3_8b \
  --model-path models/DeepSeek-R1-0528-Qwen3-8B \
  --output-dir "$OUT/deepseek_r1_qwen3_8b_train10k" \
  --backbone-training frozen --quantization 4bit \
  --epochs 10 --patience 3 --batch-size 2 --grad-accum 16 \
  --learning-rate 2e-4 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/deepseek_r1_qwen3_8b_train10k.log"
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_10K" --val-data "$VAL" --test-data "$TEST" \
  --model internlm3_8b \
  --model-path models/InternLM3-8B-Instruct \
  --output-dir "$OUT/internlm3_8b_train10k" \
  --backbone-training frozen --quantization 4bit \
  --epochs 10 --patience 3 --batch-size 2 --grad-accum 16 \
  --learning-rate 2e-4 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/internlm3_8b_train10k.log"
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m downstream_outage_prediction.llm_baselines \
  --train-data "$TRAIN_10K" --val-data "$VAL" --test-data "$TEST" \
  --model qwen3_5_2b \
  --model-path models/Qwen3.5-2B \
  --output-dir "$OUT/qwen3_5_2b_train10k" \
  --backbone-training full --quantization none \
  --epochs 10 --patience 3 --batch-size 1 --grad-accum 32 \
  --learning-rate 2e-4 --backbone-learning-rate 1e-5 --adapter-width 512 \
  --bandwidth-hz 100000000 --noise-figure-db 7 --required-snr-db 5 \
  --num-workers 0 --gradient-checkpointing \
  2>&1 | tee "$OUT/logs/qwen3_5_2b_train10k.log"
```
