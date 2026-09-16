# MCS 链路自适应下游任务

该目录实现独立的七分类 MCS 下游实验：

```text
现有 CSI + total_received_power_dbw
        -> 训练集完整链路仿真校准 EESM/BLER
        -> Outage、MCS0...MCS5 oracle 标签
        -> Scratch+Tx / Scratch+Tx+Physics / Full embedding+Tx / Full+Tx+Physics
```

该流程不会重新读取或重新提取：

```text
/home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data
```

它直接复用服务器上的现有文件：

```text
/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt
/home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt
```

默认通信配置为 23 dBm 发射功率、100 MHz 带宽、7 dB 噪声系数、
-174 dBm/Hz 热噪声密度，即总噪声功率 -87 dBm（-117 dBW）。MCS 表位于
`mcs_table.json`。其中没有伪造的 EESM beta 或 SNR 门限；正式生成标签前必须完成训练集校准。

## 1. 选择训练集校准链路

```bash
cd /home/yrh/CSI_model/CSI_model

python -m downstream_mcs_adaptation.select_calibration \
  --train-pt artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt \
  --mcs-table downstream_mcs_adaptation/mcs_table.json \
  --output-dir downstream_mcs_adaptation/artifacts/calibration_seed23421 \
  --num-links 2000 \
  --seed 23421 \
  --effective-channel-mode best_beam
```

选择器按照 LoS/NLoS、总接收功率、时延扩展和频率选择性分层抽样。它只读取
train，不读取 test，并生成：

- `selection.pt`：原训练集索引和 `group_id`；
- `selected_links.csv`：校准链路物理分布；
- `effective_channels.npz`：单位平均功率的等效频域信道 `[link, 128]`；
- `simulation_requests.csv`：建议的 MCS/SNR 扫描点；
- `selection_summary.json`：抽样检查结果。

## 2. 运行完整 PHY 链路仿真

使用 MATLAB、Sionna 或已有链路仿真器处理 `simulation_requests.csv`。每一行给出：

```text
group_id, source_index, mcs_index, average_snr_db
```

仿真器必须固定以下条件：单用户、单流、完美信道估计、MMSE 均衡、足够长 CP，
并使用 `mcs_table.json` 中的调制方式和编码率。`best_beam` 的含义是：从 CSI 的
64 个 DFT 波束中按跨子载波平均功率选择一个波束，然后在所有 128 个子载波上
固定使用该波束。可通过 `downstream_mcs_adaptation.mcs.complex_beam_csi(sample)`
恢复 `[beam, subcarrier]` 复数 CSI。

为保证可复现，当前 Sionna 实现固定使用 5G NR LDPC、1536 bit 码字、RV0、无额外
CRC、20 次 BP 译码迭代。门限标定与最后的 test 抽查必须原样复用这些配置；若修改
任一项，必须重新完成校准和标签生成。

若外部仿真器不读取 PyTorch `.pt`，可直接读取 `effective_channels.npz` 中的
`normalized_frequency_response`。它已满足跨子载波平均功率为 1；每个 request 的
`average_snr_db` 决定噪声标定，不能再次按原 CSI 幅度重复缩放。

完整仿真结果必须保存为 `simulation_results.csv`，字段为：

```text
group_id,mcs_index,average_snr_db,block_errors,total_blocks
```

`block_errors` 是错误传输块数，`total_blocks` 是总传输块数。可以使用自适应停止，
不要求完成 requests 文件中的每一行，但每种 MCS 必须同时覆盖高 BLER 和低 BLER
区域。严禁加入 test 链路。

本目录提供 Sionna 2.x 实现 `simulate_sionna.py`。它使用 5G NR LDPC、Gray QAM、
频域 OFDM 等效信道和完美 CSI 单流 MMSE。先用少量数据做接口 smoke test：

```bash
conda activate sionna_phy311

python -m downstream_mcs_adaptation.simulate_sionna \
  --max-links 2 \
  --snr-stride 9 \
  --batch-blocks 2 \
  --min-block-errors 2 \
  --max-blocks 4 \
  --output downstream_mcs_adaptation/artifacts/calibration_seed23421/simulation_results_smoke.csv \
  --device cuda
```

smoke test 通过后，用 200 条链路及间隔 SNR 网格做 pilot：

```bash
conda activate sionna_phy311

python -m downstream_mcs_adaptation.simulate_sionna \
  --max-links 200 \
  --snr-stride 2 \
  --batch-blocks 64 \
  --min-block-errors 20 \
  --max-blocks 200 \
  --output downstream_mcs_adaptation/artifacts/calibration_seed23421/simulation_results_pilot.csv \
  --device cuda
```

脚本逐行写入并刷新结果。中断后使用完全相同的参数并增加 `--resume` 即可续跑。
pilot 能顺利覆盖每种 MCS 的高低 BLER 区域后，再扩大到 2000 条链路和完整 SNR
网格，输出正式的 `simulation_results.csv`。不要把 smoke/pilot 结果直接拼接到正式
CSV。

## 3. 拟合 EESM beta、10% BLER 门限及 BLER 曲线

```bash
python -m downstream_mcs_adaptation.fit_calibration \
  --selection downstream_mcs_adaptation/artifacts/calibration_seed23421/selection.pt \
  --results downstream_mcs_adaptation/artifacts/calibration_seed23421/simulation_results.csv \
  --mcs-table downstream_mcs_adaptation/mcs_table.json \
  --output downstream_mcs_adaptation/artifacts/calibration_seed23421/calibrated_mcs.json
```

该脚本对每种 MCS 搜索 EESM beta，并用按传输块数加权的二项 Logistic 曲线拟合
BLER。`snr_threshold_db` 是拟合曲线上 BLER 恰好为 10% 的有效 SNR。若门限没有
随 MCS 严格升高，脚本会拒绝输出，要求检查或增加 PHY 仿真数据。

## 4. 生成 40k/5k/5k oracle 标签

```bash
python -m downstream_mcs_adaptation.prepare_data \
  --train-pt artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt \
  --test-pt artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt \
  --calibrated-mcs downstream_mcs_adaptation/artifacts/calibration_seed23421/calibrated_mcs.json \
  --output downstream_mcs_adaptation/artifacts/d2los_50k_mcs_labels.pt \
  --seed 23421
```

现有 train 全部作为训练集；现有 10k test 按 `group_id` 的稳定哈希拆为 5k
validation 和 5k final test。同一物理链路不能跨集合。输出只保存到原 `.pt` 的
引用，不复制 50k 份 CSI。

标签类别映射为：

```text
class 0 = Outage / MCS -1
class 1 = MCS 0
...
class 6 = MCS 5
```

每条链路同时保存平均 SNR、逐 MCS 有效 SNR、逐 MCS 估计 BLER 和 oracle MCS。
`.summary.json` 会报告各集合类别分布；若单一类别超过 80%，会明确告警。此时应先
检查校准，再考虑对同一 CSI 增加多个功率/噪声档位，且同一物理链路的所有版本必须
保留在同一 split。

若固定 23 dBm 数据主要集中在 Outage 与 MCS5，可保留上述文件作为固定功率基线，
再生成 0、5、10、15、20、23 dBm 六个实际发射功率档位：

```bash
python -m downstream_mcs_adaptation.prepare_data \
  --train-pt artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt \
  --test-pt artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt \
  --calibrated-mcs downstream_mcs_adaptation/artifacts/calibration_seed23421/calibrated_mcs.json \
  --tx-power-levels-dbm 0 5 10 15 20 23 \
  --output downstream_mcs_adaptation/artifacts/d2los_300k_mcs_power_profiles.pt \
  --seed 23421
```

输出包含 300k 个链路—功率样本，但仍只有 50k 条物理链路。传播 CSI 保持原样，
`tx_power_dbm` 作为显式输入提供给每一种下游分类器；标签中的接收功率和平均 SNR
增加相同的 dB 偏移。全部功率版本沿用原物理链路的 split，因此物理划分仍为
40k/5k/5k，不会发生跨集合泄漏。

预训练模型始终接收原始 23 dBm 数据分布下的 CSI。Full+Physics 先预测基准功率下的
归一化总接收功率，再做
`predicted_rx_current_norm = predicted_rx_base_norm + power_offset_db / 30`；K 因子、LoS
概率、时延扩展和路径数不随发射功率改变。这样不会出现相同模型输入对应六个不同标签，
也不会依赖预训练 Encoder 对人为缩放 CSI 的域外外推。

## 5. 训练四种模型

默认 checkpoint 应当包含已经训练的总接收功率头和 `use_power_branch=True`。
如果实际 checkpoint 路径不同，必须显式传入 `--checkpoint`。

```bash
python -m downstream_mcs_adaptation.run_experiment \
  --data downstream_mcs_adaptation/artifacts/d2los_300k_mcs_power_profiles.pt \
  --checkpoint artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt \
  --output-dir downstream_mcs_adaptation/outputs/seed23421 \
  --models scratch scratch_physics full_embedding full_physics \
  --device cuda \
  --batch-size 128 \
  --epochs 100 \
  --patience 15 \
  --seed 23421
```

四种模型为：

- `scratch`：随机初始化 CSI Encoder，使用原始 CSI token 的对数幅度统计和显式发射
  功率；不使用预训练权重或真实物理标签。
- `scratch_physics`：随机初始化且参与下游训练的 CSI Encoder，加显式发射功率和冻结
  预训练模型预测的物理量。它不使用原始 CSI 幅度统计；物理量与 `full_physics` 完全
  相同，因此二者可用于隔离预训练 embedding 的贡献。
- `full_embedding`：冻结预训练模型，使用归一化的 256 维 embedding 加显式发射功率
  训练 MCS 头。
- `full_physics`：冻结预训练模型，使用 embedding、显式发射功率，以及模型预测的
  当前功率下总接收功率、K 因子、LoS 概率、时延扩展和路径数。不会把真实物理标签
  输入 MCS 分类头。

已有三模型实验完成后，可以复用其冻结特征缓存，单独补跑 `scratch_physics`：

```bash
python -u -m downstream_mcs_adaptation.run_experiment \
  --data downstream_mcs_adaptation/artifacts/d2los_300k_mcs_power_profiles.pt \
  --checkpoint artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt \
  --feature-cache downstream_mcs_adaptation/outputs/seed23421_explicit_tx/frozen_features.pt \
  --output-dir downstream_mcs_adaptation/outputs/seed23421_scratch_physics \
  --models scratch_physics \
  --device cuda:0 \
  --batch-size 128 \
  --epochs 100 \
  --patience 15 \
  --seed 23421
```

缓存只提供冻结预训练模型的预测物理量；`scratch_physics` 自己的 CSI Encoder 仍从
随机权重开始训练。运行时应出现 `loaded_feature_cache=...`，否则程序会重新提取特征。

默认损失是带训练集类别权重的交叉熵。`--safety-penalty-weight 0` 表示纯交叉熵；
如需在验证集上研究更保守的选择，可设置一个正值，但选定后不能根据 final test 调参。

## 输出指标

`metrics.csv` 和 `metrics.json` 对每个模型报告：

- MCS Accuracy；
- aggressive selection rate，即选择高于 oracle 的比例；
- conservative selection rate；
- BLER > 10% violation rate；
- 平均选择 BLER；
- 平均频谱效率；
- 平均估计 Goodput；
- Oracle Goodput 及 Goodput ratio；
- Outage recall 和 false-alarm rate。

此外保存逐测试样本 CSV/NPZ、混淆矩阵、最佳 checkpoint 和冻结特征缓存。
所有最终指标只在哈希拆分出的 final test 5k 上计算。

## 6. 用完整 PHY 抽查快速 oracle 标签

主实验结束后，可从 final test 固定抽取 500 条链路；该步骤不参与门限拟合或调参：

```bash
python -m downstream_mcs_adaptation.export_test_validation \
  --data downstream_mcs_adaptation/artifacts/d2los_50k_mcs_labels.pt \
  --output-dir downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421 \
  --num-links 500 \
  --seed 23421
```

对生成的 `simulation_requests.csv` 逐链路、逐 MCS 运行与校准阶段完全相同的完整
PHY 仿真，写入 `block_errors` 和 `total_blocks`，保存为 `simulation_results.csv`。
随后计算快速 EESM oracle 与完整 PHY oracle 的一致性：

```bash
python -m downstream_mcs_adaptation.score_test_validation \
  --selection downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421/selection.pt \
  --results downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421/simulation_results.csv \
  --output downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421/validation_metrics.json
```

输出包括快速 oracle 的一致率、过高/过低率、实际 BLER 违反率和逐链路审核表。
这是独立测试审核，不得再用结果修改 EESM beta、MCS 门限或模型超参数。

## 重要边界

本目录负责校准样本选择、EESM/BLER 参数拟合、50k 快速标签生成和三模型训练评估。
它不伪装成标准 PHY 编码/译码器；完整 LDPC/Polar 编码、调制、OFDM、加噪声、MMSE
均衡、软解调和译码必须由服务器上可信的 MATLAB/Sionna/现有仿真器执行，并通过规定
的 CSV 接口返回真实块错误统计。这一步是论文中 MCS 门限可信性的来源，不能用手填门限
代替。
