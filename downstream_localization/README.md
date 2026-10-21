# CSI2TEXT-PGMT 下游二维定位实验

这个目录是独立的定位实验，不会向仓库原有的 `scripts/`、`data/`、`models/` 写入任何文件。它复用原项目的 CSI encoder 和 D2Los 预处理定义，完成以下模型的公平比较：

1. `scratch`：随机初始化 CSI encoder，与定位头一起端到端训练；
2. `scratch_physics`：训练随机初始化的 CSI encoder，同时拼接 Full 模型冻结预测的四维物理特征；
3. `pretrained_finetune`：用预训练 CSI encoder 初始化，然后和定位头一起端到端微调；
4. `full_embedding`：冻结预训练 CSI2TEXT-PGMT，仅使用 256 维 CSI embedding；
5. `full_physics`：冻结预训练模型，使用 256 维 embedding，加预测的 LoS 概率、首径时延、首径角度正余弦，共 260 维。

物理特征全部来自模型预测，不使用真实 LoS、真实时延或真实角度。定位标签是 `(RX_x - TX_x, RX_y - TX_y)`，坐标归一化统计量只由训练集计算。

## 文件说明

- `prepare_data.py`：从 RayVerse `.propbin.gz` 流式生成 CSI token、TX/RX 坐标和固定划分；
- `prepare_ultra_dense.py`：从 KU Leuven ultra-dense `.npy` 实测 CSI 生成定位迁移验证数据；
- `convert_existing.py`：给已有预处理 train/val/test CSI 回填坐标，不重新生成 CSI；
- `data_io.py`：定位数据集和 batch 拼接；
- `pretrained.py`：从 checkpoint 自动还原 CSI2TEXT-PGMT，并提取 embedding/物理预测；
- `run_experiment.py`：训练四个模型、early stopping、测试、汇总指标和绘制 CDF。
- `run_label_efficiency.py`：用嵌套的 1%/5%/10%/25%/100% 位置标签比较 Scratch 与 Scratch + Frozen Physics。
- `diagnose_test_errors.py`：读取已有测试预测，按 LoS/NLoS 分组统计并列出最大误差链路。
- `run_llm_baseline.py`：用 DeepSeek-Qwen3-8B、InternLM3-8B 和 Qwen3.5-2B 做连续 CSI soft-prefix 定位 baseline。
- `run_classical_agreement.py`：第二层实测验证；用 IFFT-PDP / raw CSI energy 作为独立经典估计器，检查模型物理输出的一致性。

## 1. 准备 100k 定位数据

在服务器的仓库根目录执行：

```bash
cd /home/yrh/CSI_model/CSI_model

python -m downstream_localization.prepare_data \
  --d2los-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data \
  --output downstream_localization/artifacts/d2los_upa8x8_nf128_100k.pt \
  --max-samples 100000 \
  --sampling map_uniform \
  --array-rows 8 \
  --array-cols 8 \
  --subcarriers 128 \
  --bandwidth-hz 100000000 \
  --seed 23421
```

程序直接从 propbin 头部读取 TX 三维坐标，从 RX record 读取 RX 三维坐标。它按 source 文件小批处理，避免一次在内存中堆放全部原始复数信道。输出仍会较大，因为 100k 条预处理 token 本身约占数 GB。

划分使用 `seed + 完整 map/source/rx 链路 ID` 的稳定哈希，比例为 80%/10%/10%。同一链路只会进入一个 split，且三个模型读取同一份 split。

如果要严格排除预训练阶段见过的链路，可以附加一个或多个原预训练 `.pt`：

```bash
  --exclude-seen-pt artifacts/pretraining_train.pt \
  --exclude-seen-pt artifacts/pretraining_validation.pt
```

不提供这个选项时，代码无法保证定位测试链路未被 CSI2TEXT-PGMT 预训练见过；因此该结果只能作为第一版可行性实验，不能声称未知地图或严格未见链路泛化。

### 复用已有的 40k/5k/5k 预处理数据（更快）

如果已经有同一批 50k 数据的 train、validation、test 三个 `.pt`，不需要重新 preprocess CSI。只需回到 propbin 按 `group_id` 读取坐标：

```bash
python -m downstream_localization.convert_existing \
  --train-pt artifacts/d2los_50k_balanced_los_angle_seed23421_train40k.pt \
  --val-pt artifacts/d2los_50k_balanced_los_angle_seed23421_validation5k.pt \
  --test-pt artifacts/d2los_50k_balanced_los_angle_seed23421_test5k.pt \
  --d2los-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data \
  --output downstream_localization/artifacts/d2los_50k_localization.pt
```

验证集文件名请以服务器上的实际名称为准。脚本会原样复用三份文件的 split，并检查 `map/source/rx` 是否跨 split 泄漏。单独一个 `train40k.pt` 不足以计算验证和测试指标。

如果现有文件是 `train40k + test10k`，省略 `--val-pt` 即可。脚本会根据链路 ID 和 seed 将 `test10k` 稳定拆成 `val5k + test5k`，最终仍为 80%/10%/10%：

```bash
python -m downstream_localization.convert_existing \
  --train-pt artifacts/d2los_50k_balanced_los_angle_seed23421_train40k.pt \
  --test-pt artifacts/d2los_50k_balanced_los_angle_seed23421_test10k.pt \
  --d2los-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data \
  --output downstream_localization/artifacts/d2los_50k_localization.pt \
  --seed 23421
```

## 2. 运行四个模型

将下面 checkpoint 替换为你的 Full CSI2TEXT-PGMT checkpoint：

```bash
python -m downstream_localization.run_experiment \
  --data downstream_localization/artifacts/d2los_upa8x8_nf128_100k.pt \
  --checkpoint /path/to/full_csi2text_pgmt_checkpoint.pt \
  --output-dir downstream_localization/outputs/first_run \
  --models all \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --head-lr 1e-3 \
  --scratch-lr 3e-4 \
  --seed 23421
```

默认使用 CUDA 和混合精度（CUDA 可用时）。可用 `--device cuda:1` 选择 GPU，或用 `--no-amp` 关闭混合精度。第一次运行冻结模型时会生成 `frozen_features.pt`；之后可直接复用，使用 `--rebuild-feature-cache` 可强制重算。

也可以只运行指定模型：

```bash
--models scratch
--models scratch_physics
--models full_embedding full_physics
```

## 输出

输出目录中包括：

- `metrics.csv` / `metrics.json`：Mean、Median、P90、≤5 m、≤10 m；
- `localization_error_cdf.png`：所选模型的误差 CDF；
- `*_best.pt`：按验证集 Median error 选出的 checkpoint；
- `*_test_predictions.npz`：测试样本索引、相对坐标、RX 坐标和逐样本误差；
- `frozen_features.pt`：冻结 Full 模型提取的 256 维 embedding 和四维预测物理特征。

首径时延在送入定位头前除以 3000 ns，与 CSI2TEXT-PGMT 内部物理目标尺度一致；预测角度向量会归一化为单位 `(sin, cos)`。

### 用 KU Leuven ultra-dense 实测数据做第一层验证

这个实验只使用真实可标定的用户位置 GT，回答的是：

```text
DeepMIMO 上预训练出的 CSI 表示，迁移到真实 measured CSI 后，是否能提升定位？
```

如果需要 LoS/nLoS 数量严格平衡，推荐直接生成一个 combined corpus。下面命令会生成 50k 样本：train 40k，其中 LoS 20k、nLoS 20k；val 5k，其中 LoS 2.5k、nLoS 2.5k；test 5k，其中 LoS 2.5k、nLoS 2.5k。脚本会自动读取 `user_positions.npy`、`antenna_positions.npy` 和真实存在的 `samples/channel_measurement_*.npy`，跳过缺失 CSI 文件。

抽样不是简单连续取样：`--split-tile-size-m 0.25` 让 train/val/test 使用互不重叠的粗空间 tile，`--sample-cell-size-m 0.01` 让每个类别在 1 cm 小格内最多取一个样本，并且默认不让同一 split 里的 LoS/nLoS 共用同一个 1 cm 小格，避免 5 mm 超密网格里抽到一堆几乎同位置的点。

```bash
python -m downstream_localization.prepare_ultra_dense_balanced \
  --los-data-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/ultra_dense/URA_lab_LoS \
  --nlos-data-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/ultra_dense/URA_lab_nLoS \
  --output downstream_localization/artifacts/ultra_dense_ura_lab_los_nlos_balanced_40k_5k_5k.pt \
  --train-per-class 20000 \
  --val-per-class 2500 \
  --test-per-class 2500 \
  --split-tile-size-m 0.25 \
  --sample-cell-size-m 0.01 \
  --carrier-hz 2.61e9 \
  --bandwidth-hz 20e6 \
  --target-nf 100 \
  --seed 23421
```

然后跑四组方法：

```bash
python -m downstream_localization.run_experiment \
  --data downstream_localization/artifacts/ultra_dense_ura_lab_los_nlos_balanced_40k_5k_5k.pt \
  --checkpoint /path/to/full_csi2text_pgmt_checkpoint.pt \
  --output-dir downstream_localization/outputs/ultra_dense_balanced_first_layer \
  --models all \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --head-lr 1e-3 \
  --scratch-lr 3e-4 \
  --seed 23421
```

对应四个方法就是表里的：

| 表中名字 | 命令里的模型名 |
|---|---|
| Scratch + frozen physics | `scratch_physics` |
| Scratch | `scratch` |
| Pretrained finetune | `pretrained_finetune` |
| Pretrained embedding | `full_embedding` |
| Pretrained embedding + physics | `full_physics` |

注意这里的实测数据是 uplink 单发多收，脚本按 TDD 互易性的方式把 64 根接收天线当成模型的 8×8 空间阵列输入。输出指标除了 Mean/Median/P90，还会报告 `≤0.1m`、`≤0.25m`、`≤0.5m`、`≤1m`，更适合这个超密定位场景。

### 第三层：实测物理一致性验证

没有 path-level GT 时，不把模型输出叫 accuracy，而是验证它是否符合真实信道的物理趋势。下面脚本会在实测 test split 上输出：

- LoS/NLoS 的 `p_LoS`、K-factor、delay spread、power 分布；
- LoS 场景下 first-path delay vs distance 的线性斜率，和 `1/c = 3.336 ns/m` 对比；
- LoS 场景下 predicted power / CSI energy vs `log10(distance)` 的路径损耗趋势；
- 空间一致性曲线：位置越远，embedding/物理预测差异是否越大。

```bash
python -m downstream_localization.run_physical_consistency \
  --data downstream_localization/artifacts/ultra_dense_ura_lab_los_nlos_balanced_40k_5k_5k.pt \
  --checkpoint /home/yrh/CSI_model/CSI_model/artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_last.pt \
  --output-dir downstream_localization/outputs/ultra_dense_physical_consistency_test \
  --splits test \
  --batch-size 256 \
  --pair-count 200000 \
  --seed 23421
```

主要输出：

- `physical_consistency_metrics.json`：回归斜率、相关性、LoS/NLoS 分布差异、空间一致性相关性；
- `los_nlos_distribution_summary.csv`：LoS/NLoS 分组统计；
- `spatial_consistency_bins.csv`：不同空间距离 bin 下的预测差异；
- `physical_consistency_summary.png`：delay-distance、power-distance、K-factor、delay spread 四联图；
- `spatial_consistency.png`：空间一致性曲线。

### 第二层：与经典 PDP 估计器的一致性

这个实验不需要 path-level GT，也不要写成 accuracy。它比较的是：

- 模型 `first_delay_ns` vs IFFT-PDP 的 first delay，只建议报告 Pearson/Spearman correlation；
- 模型 `delay_spread_ns` vs PDP RMS delay spread；
- 模型 `k_factor_db` vs PDP dominant peak-to-tail ratio；
- 模型 power head vs raw CSI energy。

```bash
python -m downstream_localization.run_classical_agreement \
  --data downstream_localization/artifacts/ultra_dense_ura_lab_los_nlos_balanced_40k_5k_5k.pt \
  --checkpoint /home/yrh/CSI_model/CSI_model/artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_last.pt \
  --output-dir downstream_localization/outputs/ultra_dense_classical_agreement_test \
  --splits test \
  --batch-size 256 \
  --n-fft-factor 4 \
  --pdp-threshold-db 20 \
  --max-delay-ns 3000 \
  --seed 23421
```

主要输出：

- `classical_agreement_metrics.json`：整体、LoS、nLoS 三组的一致性指标；
- `classical_agreement_summary.csv`：便于直接整理成表格的扁平版指标；
- `classical_agreement_predictions.npz`：逐样本模型输出和经典 PDP 估计；
- `classical_agreement_scatter.png`：四联散点图。

## 3. 标签效率实验

以下命令以 40k 定位训练集为基准，依次使用 400、2k、4k、10k、40k 条位置标签。所有子集严格嵌套，两个模型在同一比例使用完全相同的样本；val5k 和 test5k 始终不变。

单个下游随机种子的第一轮实验：

```bash
python -m downstream_localization.run_label_efficiency \
  --data downstream_localization/artifacts/d2los_50k_localization.pt \
  --checkpoint artifacts/ablations_d2los_100k_upa8x8_nf128_los50k_nlos50k/full_multitask/seed_0/checkpoint_last.pt \
  --output-dir downstream_localization/outputs/label_efficiency_seed23421 \
  --label-fractions 0.01 0.05 0.10 0.25 1.0 \
  --seeds 23421 \
  --subset-seed 23421 \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --scratch-lr 3e-4
```

若已有四模型实验生成的冻结特征缓存，可避免重新提取 Full 物理预测：

```bash
  --feature-cache downstream_localization/outputs/d2los_50k_full_seed0/frozen_features.pt
```

正式结果建议运行三个下游随机种子；位置标签子集仍由固定的 `subset-seed` 定义，所以不同模型和训练种子之间可直接比较：

```bash
python -m downstream_localization.run_label_efficiency \
  --data downstream_localization/artifacts/d2los_50k_localization.pt \
  --checkpoint artifacts/ablations_d2los_100k_upa8x8_nf128_los50k_nlos50k/full_multitask/seed_0/checkpoint_last.pt \
  --output-dir downstream_localization/outputs/label_efficiency_3seeds \
  --feature-cache downstream_localization/outputs/d2los_50k_full_seed0/frozen_features.pt \
  --label-fractions 0.01 0.05 0.10 0.25 1.0 \
  --seeds 23421 23422 23423 \
  --subset-seed 23421 \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --scratch-lr 3e-4
```

主要输出：

- `detailed_metrics.csv`：每个模型、标签比例、训练种子的完整测试指标；
- `summary_metrics.csv`：跨种子的均值和标准差；
- `label_efficiency.json`：机器可读汇总；
- `label_efficiency.png`：Mean、Median、P90 随标签数量变化的三联图；
- `*_best.pt`：各比例按验证集 Median error 选择的最佳 checkpoint；
- `*_test_predictions.npz`：各比例的最终测试预测。

## 4. LoS/NLoS 与长尾误差诊断

无需重新训练。对于原四模型实验的预测文件，运行：

```bash
python -m downstream_localization.diagnose_test_errors \
  --data downstream_localization/artifacts/d2los_50k_localization.pt \
  --predictions-dir downstream_localization/outputs/d2los_50k_full_seed0 \
  --top-k 20
```

如果 `scratch_physics` 的预测单独存放在另一个输出目录，再运行一次：

```bash
python -m downstream_localization.diagnose_test_errors \
  --data downstream_localization/artifacts/d2los_50k_localization.pt \
  --predictions-dir downstream_localization/outputs/scratch_physics_seed23421 \
  --models scratch_physics \
  --top-k 20
```

每次运行都会在预测目录下生成 `diagnostics/`：

- `los_nlos_metrics.csv`：每个模型在全体、LoS、NLoS 上的 Mean、Median、P90、≤5 m、≤10 m，以及 >50 m、>100 m 比例；
- `tail_composition.csv`：误差最大约 10% 样本中的 NLoS 比例，并与整个测试集的 NLoS 比例比较；
- `worst_samples.csv`：每个模型误差最大的 20 条链路，含 `map/source/rx` ID、LoS/NLoS、TX 坐标、真实/预测 RX 坐标；
- `diagnostics.json`：同样信息的机器可读版本。

脚本会确认预测文件中的样本索引与当前数据包的 test split 完全一致，并重新计算 RX 定位误差；若传错数据集或预测文件，将直接报错。查看结果可用：

```bash
column -s, -t downstream_localization/outputs/d2los_50k_full_seed0/diagnostics/los_nlos_metrics.csv
column -s, -t downstream_localization/outputs/d2los_50k_full_seed0/diagnostics/tail_composition.csv
head -n 21 downstream_localization/outputs/d2los_50k_full_seed0/diagnostics/worst_samples.csv
```

## 5. LLM 定位 baseline

脚本：[run_llm_baseline.py](run_llm_baseline.py)。它不是把 CSI 写成文字让大模型 zero-shot 回答坐标，而是：

```text
CSI -> 随机初始化 CSIEncoder -> soft-prefix projector -> LLM -> localization head -> (dx, dy)
```

默认训练策略：

- `deepseek_r1_qwen3_8b`：冻结 8B LLM，只训练 CSIEncoder、soft-prefix projector、readout 和定位输出头；
- `internlm3_8b`：冻结 8B LLM，只训练 CSIEncoder、soft-prefix projector、readout 和定位输出头；
- `qwen3_5_2b`：全量微调文本主干，同时训练 CSIEncoder、soft-prefix projector、readout 和定位输出头。

先用 1k/500/500 检查输出是否正常。以下命令会在 tmux 中后台运行，Qwen3.5 用卡 0，DeepSeek 和 InternLM3 用卡 1；两个 8B 都放在卡 1，但建议不要同时跑，避免显存相互挤占。

```bash
cd /home/yrh/CSI_model/CSI_model
mkdir -p downstream_localization/outputs/llm_smoke

tmux new-session -d -s loc_qwen35_1k "cd /home/yrh/CSI_model/CSI_model && CUDA_VISIBLE_DEVICES=0 conda run -n csi_qwen35 --no-capture-output python -u -m downstream_localization.run_llm_baseline --data downstream_localization/artifacts/d2los_50k_localization.pt --checkpoint artifacts/ablations_d2los_100k_upa8x8_nf128_los50k_nlos50k/full_multitask/seed_0/checkpoint_last.pt --model qwen3_5_2b --model-path models/Qwen3.5-2B --output-dir downstream_localization/outputs/llm_smoke/qwen35_2b_1k --train-size 1000 --val-size 500 --test-size 500 --batch-size 4 --grad-accum 8 --epochs 1 --patience 1 --quantization none --backbone-training full --gradient-checkpointing --no-save-backbone 2>&1 | tee downstream_localization/outputs/llm_smoke/qwen35_2b_1k.log"

tmux new-session -d -s loc_deepseek_1k "cd /home/yrh/CSI_model/CSI_model && CUDA_VISIBLE_DEVICES=1 conda run -n csi_qwen35 --no-capture-output python -u -m downstream_localization.run_llm_baseline --data downstream_localization/artifacts/d2los_50k_localization.pt --checkpoint artifacts/ablations_d2los_100k_upa8x8_nf128_los50k_nlos50k/full_multitask/seed_0/checkpoint_last.pt --model deepseek_r1_qwen3_8b --model-path models/DeepSeek-R1-0528-Qwen3-8B --output-dir downstream_localization/outputs/llm_smoke/deepseek_qwen3_8b_1k --train-size 1000 --val-size 500 --test-size 500 --batch-size 4 --grad-accum 8 --epochs 1 --patience 1 --backbone-training frozen --quantization 4bit --gradient-checkpointing 2>&1 | tee downstream_localization/outputs/llm_smoke/deepseek_qwen3_8b_1k.log"

tmux new-session -d -s loc_internlm3_1k "cd /home/yrh/CSI_model/CSI_model && CUDA_VISIBLE_DEVICES=1 conda run -n csi_internlm3 --no-capture-output python -u -m downstream_localization.run_llm_baseline --data downstream_localization/artifacts/d2los_50k_localization.pt --checkpoint artifacts/ablations_d2los_100k_upa8x8_nf128_los50k_nlos50k/full_multitask/seed_0/checkpoint_last.pt --model internlm3_8b --model-path models/InternLM3-8B-Instruct --output-dir downstream_localization/outputs/llm_smoke/internlm3_8b_1k --train-size 1000 --val-size 500 --test-size 500 --batch-size 4 --grad-accum 8 --epochs 1 --patience 1 --backbone-training frozen --quantization 4bit --gradient-checkpointing 2>&1 | tee downstream_localization/outputs/llm_smoke/internlm3_8b_1k.log"
```

查看运行状态：

```bash
tmux ls
tmux attach -t loc_qwen35_1k
tmux attach -t loc_deepseek_1k
tmux attach -t loc_internlm3_1k
```

查看 1k smoke 结果：

```bash
cat downstream_localization/outputs/llm_smoke/qwen35_2b_1k/metrics.json
cat downstream_localization/outputs/llm_smoke/deepseek_qwen3_8b_1k/metrics.json
cat downstream_localization/outputs/llm_smoke/internlm3_8b_1k/metrics.json
```

确认输出正常后，把 `--train-size 1000 --val-size 500 --test-size 500 --epochs 1 --patience 1` 改成 `--train-size 10000 --val-size 5000 --test-size 5000 --epochs 10 --patience 3`，并换一个新的 `--output-dir`，即可做正式 LLM baseline。

## 依赖和约束

使用原项目环境即可，额外只需要常见的 `numpy`、`matplotlib` 和 `torch`。数据 token 宽度必须与 checkpoint 一致；例如 UPA 8×8、`2×2` patch 对应 `d_token=8`。checkpoint 必须带有同时包含 LoS/NLoS 的 `prototype_keys`，否则无法构造预测的 `p_LoS`，程序会明确报错。

LLM baseline 额外需要对应环境里的 `transformers`、`accelerate`、`bitsandbytes` 和 `safetensors`。Qwen3.5 建议使用 `csi_qwen35` 环境；InternLM3 使用 `csi_internlm3` 环境，并会启用 `trust_remote_code=True` 加载本地模型代码。
