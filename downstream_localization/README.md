# CSI2TEXT-PGMT 下游二维定位实验

这个目录是独立的定位实验，不会向仓库原有的 `scripts/`、`data/`、`models/` 写入任何文件。它复用原项目的 CSI encoder 和 D2Los 预处理定义，完成以下四个模型的公平比较：

1. `scratch`：随机初始化 CSI encoder，与定位头一起端到端训练；
2. `scratch_physics`：训练随机初始化的 CSI encoder，同时拼接 Full 模型冻结预测的四维物理特征；
3. `full_embedding`：冻结预训练 CSI2TEXT-PGMT，仅使用 256 维 CSI embedding；
4. `full_physics`：冻结预训练模型，使用 256 维 embedding，加预测的 LoS 概率、首径时延、首径角度正余弦，共 260 维。

物理特征全部来自模型预测，不使用真实 LoS、真实时延或真实角度。定位标签是 `(RX_x - TX_x, RX_y - TX_y)`，坐标归一化统计量只由训练集计算。

## 文件说明

- `prepare_data.py`：从 RayVerse `.propbin.gz` 流式生成 CSI token、TX/RX 坐标和固定划分；
- `convert_existing.py`：给已有预处理 train/val/test CSI 回填坐标，不重新生成 CSI；
- `data_io.py`：定位数据集和 batch 拼接；
- `pretrained.py`：从 checkpoint 自动还原 CSI2TEXT-PGMT，并提取 embedding/物理预测；
- `run_experiment.py`：训练四个模型、early stopping、测试、汇总指标和绘制 CDF。
- `run_label_efficiency.py`：用嵌套的 1%/5%/10%/25%/100% 位置标签比较 Scratch 与 Scratch + Frozen Physics。

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

## 依赖和约束

使用原项目环境即可，额外只需要常见的 `numpy`、`matplotlib` 和 `torch`。数据 token 宽度必须与 checkpoint 一致；例如 UPA 8×8、`2×2` patch 对应 `d_token=8`。checkpoint 必须带有同时包含 LoS/NLoS 的 `prototype_keys`，否则无法构造预测的 `p_LoS`，程序会明确报错。
