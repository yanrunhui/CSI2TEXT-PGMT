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

