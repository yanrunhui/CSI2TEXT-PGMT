# 基于 CSI 的自适应循环前缀选择

这个目录实现一个独立的下游五分类任务：根据 CSI 选择能够覆盖主要真实多径能量的最短 CP。它不会修改原有训练、定位或预处理代码。

默认候选配置为 `40/80/160/320/640 ns`，带宽 100 MHz 时分别对应 `4/8/16/32/64` 个 10 ns 采样点。这里把它们视为受控 OFDM 配置，不声称是某个具体标准的 numerology。

## 实现内容

- `labeling.py`：根据真实路径的相对时延和相对能量生成最短可行 CP 标签；
- `prepare_data.py`：匹配已有 40k/10k CSI 样本与 D2Los `.propbin(.gz)` 路径，生成固定 train/val/test 标签清单；
- `data_io.py`：按清单引用原 `.pt` 样本，不额外复制数 GB 的 CSI；
- `pretrained.py`：提取冻结 Full embedding 和冻结模型预测的物理量；
- `run_experiment.py`：在完全相同的数据与标签上训练和测试五个模型，并导出完整指标和混淆矩阵。
- `diagnose_intervals.py`：按真实 CP 区间分析召回率、错误方向、标签边界难度和物理预测误差。

五种模型是：

1. `scratch`：随机初始化 CSI encoder，端到端训练 encoder 和 CP 分类头；
2. `scratch_physics`：随机初始化并训练 CSI encoder，同时拼接冻结教师模型预测的物理量；
3. `full_embedding`：冻结预训练模型，只训练基于 256 维 Full embedding 的分类头；
4. `full_physics`：冻结预训练模型，只训练基于 Full embedding 与预测物理量的分类头。
5. `full_physics_finetune`：以 Full 的 CSI encoder 参数为起点，解冻最后若干 Transformer 层和输出投影，与 CP 分类头一起训练；物理量仍来自冻结教师模型。

若要进行整个 CSI encoder 全量微调（输入投影、位置/频率编码、全部 Transformer 层和输出投影），使用 `--unfreeze-all-encoder`。物理预测仍来自冻结教师缓存，不参与更新。

`full_physics` 使用以下四个附加量：

```text
p_LoS
predicted_delay_spread_ns / 200
predicted_n_paths / 10
predicted_reflection_path_count / 10
```

这些值全部来自冻结模型的预测，不会把真实 delay spread、路径数或反射数泄漏给分类器。微调模型也不会更新物理预测头，避免 CP 标签破坏原有物理量语义。

## 第一步：生成 CP 标签

在服务器仓库根目录运行：

```bash
cd /home/yrh/CSI_model/CSI_model

python -m downstream_cp_selection.prepare_data \
  --d2los-root /home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data \
  --train-pt /home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_balanced_los_angle_seed23421_train40k.pt \
  --test-pt /home/yrh/CSI_model/CSI_model/artifacts/d2los_50k_balanced_los_angle_seed23421_test10k.pt \
  --output /home/yrh/CSI_model/CSI_model/downstream_cp_selection/artifacts/d2los_50k_cp_labels.pt \
  --cp-ns 40 80 160 320 640 \
  --max-outside-energy-ratio 0.01 \
  --seed 23421
```

程序会保持原 `train40k` 为训练集，并按 `seed + group_id` 的稳定哈希把 `test10k` 均分为 validation/test。划分单位是完整 `map/source/rx` 物理链路，并检查重复或跨集合泄漏。

对每条链路先计算：

```text
excess_delay = path_delay - minimum_path_delay
outside_ratio(CP) = energy(excess_delay > CP) / total_path_energy
```

标签是 `outside_ratio <= 1%` 的最短 CP。路径文件保存的是 path loss，但同一链路的公共发射功率会在能量比中抵消，所以代码使用数值稳定的相对线性功率。

### 最长 CP 仍不够时

默认 `--overflow-policy drop`：若 640 ns 下 CP 外能量仍超过 1%，样本不会获得错误的 CP-5 标签，而是从五个模型共同使用的数据中删除；程序会报告 overflow 数量和各 split 的最终分布。

如果 overflow 较多，推荐重新加入第六类：

```bash
--cp-ns 40 80 160 320 640 1280
```

这时训练脚本会自动变成六分类，无需修改代码。若确实想把超长样本截断到最后一类，可以显式使用 `--overflow-policy last`，但此时最后一类不再严格满足 1% 门限。`--overflow-policy error` 可用于严格审计。

## 第二步：运行五模型实验

默认使用带 reflection-path-count 预测头的 Full CSI2TEXT-PGMT checkpoint：

```bash
python -m downstream_cp_selection.run_experiment \
  --data /home/yrh/CSI_model/CSI_model/downstream_cp_selection/artifacts/d2los_50k_cp_labels.pt \
  --checkpoint /home/yrh/CSI_model/CSI_model/artifacts/pretrain_d2los_100k_upa8x8_nf128_los50k_nlos50k_reflection_path/checkpoint_last.pt \
  --output-dir /home/yrh/CSI_model/CSI_model/downstream_cp_selection/outputs/seed23421 \
  --models all \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --head-lr 1e-3 \
  --scratch-lr 3e-4 \
  --finetune-lr 3e-5 \
  --unfreeze-last-n-layers 2 \
  --seed 23421
```

这个 checkpoint 已写为训练脚本默认值，因此也可以省略 `--checkpoint`。第一次运行冻结模型时会生成 `frozen_features.pt`，以后相同数据和 checkpoint 会自动复用；可用 `--rebuild-feature-cache` 强制重建。

默认使用普通交叉熵，测试集始终保持自然 CP 分布。只有训练类别严重失衡并且你决定做第二轮实验时，才建议加入：

```bash
--class-weight balanced
```

### CP 外能量安全惩罚

全量或部分微调模型可以在交叉熵之外加入安全惩罚：

```text
L = CE + lambda_safe * sum_c p(c|CSI) * max(0, r_out(c) - 0.01)
```

它使用训练样本中已经保存的每类真实 `outside_energy_ratios`，惩罚模型分配给不安全短 CP 的概率；真实最短安全类别仍由 CE 约束，因此不会单纯把所有样本推向 640 ns。验证集 checkpoint 选择也使用相同联合目标。

全量解冻并加入安全惩罚的推荐起点：

```bash
--models full_physics_finetune \
--unfreeze-all-encoder \
--finetune-lr 1e-5 \
--head-lr 1e-3 \
--safety-loss-weight 5.0
```

`5.0` 是首轮实验值，不保证恰好得到某个不足率。正式选择权重时，应只根据 validation 的准确率、不足率、平均 CP 和 CP 外能量比较 `2/5/10`，最后固定配置评估测试集。

### 移除反射路径数的消融

如果诊断发现 `predicted_reflection_path_count` 误差较大，可使用：

```bash
--exclude-reflection-path-count
```

分类器输入会从 `[p_LoS, delay spread, n_paths, reflection path count]` 改为 `[p_LoS, delay spread, n_paths]`。原冻结特征缓存仍然复用，只在送入分类器前移除最后一列；checkpoint、数据划分、训练设置均保持不变，因此可以严格判断反射路径数特征的净贡献。

### 移除时延扩展的消融

如果预训练 checkpoint 跨带宽迁移后 `predicted_delay_spread` 明显失准，可使用：

```bash
--exclude-delay-spread
```

分类器输入会从 `[p_LoS, delay spread, n_paths, reflection path count]` 改为 `[p_LoS, n_paths, reflection path count]`。完整冻结特征缓存仍然复用，只在送入分类器前移除 delay spread 列，因此可与保留该特征的实验进行严格比较。该开关也可与 `--exclude-reflection-path-count` 同时使用。

也可以单独运行模型：

```bash
--models scratch
--models scratch_physics
--models full_embedding
--models full_physics
--models full_physics_finetune
```

CUDA 可用时默认使用 CUDA 和混合精度。指定 GPU 可用 `--device cuda:1`，关闭混合精度可用 `--no-amp`。

## 输出

输出目录包含：

- 标签文件旁的 `d2los_50k_cp_labels.summary.json`：各 split 的类别、overflow 和最长 CP 外能量统计；
- `metrics.csv` / `metrics.json`：所选模型的汇总指标；
- `*_best.pt`：按 validation cross-entropy early stopping 得到的最优模型；
- `*_test_predictions.npz`：逐样本概率、预测/真实 CP、group ID、预测 CP 下的真实外部能量；
- `*_confusion_matrix.csv/png`：五类（或自定义类数）混淆矩阵；
- `frozen_features.pt`：Full embedding 与四个冻结预测物理量的缓存。

汇总指标包括：

- `accuracy_pct`：CP 分类准确率；
- `underprediction_rate_pct`：预测 CP 短于最优 CP 的比例；
- `overprediction_rate_pct`：预测 CP 长于最优 CP 的比例；
- `mean_predicted_cp_ns`：平均预测 CP 开销；
- `mean_cp_outside_energy_pct`：使用预测 CP 后的平均真实外部能量；
- `p95_cp_outside_energy_pct`：外部能量的 P95；
- `unsafe_outside_energy_rate_pct`：预测后真实外部能量仍超过标签门限的比例。

## 实验解释注意事项

- 五个模型读取同一个标签清单，因此训练、验证、测试样本完全一致。
- Full+Physics 会检查 checkpoint 参数中是否能看出 delay spread、总路径数和反射路径数预测头接受过监督；若看不出来会给出 warning，而不是静默把随机物理头当作有效预测。
- 若该 50k 数据中的测试链路在 Full 模型预训练阶段已经出现过，实验只能作为初步可行性结果。正式论文结果需要额外审计预训练集合与下游测试 `group_id` 的交集，或使用严格未见地图/链路划分。
- 当前标签基于理想定时同步后的相对时延，首径绝对传播时延不会影响 CP 类别。

## 大语言模型 CP baseline

先用 `make_llm_subsets.py` 从完整 CP manifest 的各原始 split 内按类别分层抽样，生成两个嵌套数据集：

- smoke：train/val/test = 1000/500/500；
- formal：train/val/test = 10000/5000/5000。

smoke 的每个 split 都是 formal 对应 split 的真子集。抽样只在父 manifest 原有的 train、val、test 内分别进行，不会把验证或测试链路放入训练集，并继续保证 `group_id` 无跨集泄漏。

`run_llm_baseline.py` 使用统一结构：

```text
CSIEncoder -> soft-prefix projector -> language backbone -> 5-class CP head
```

- Qwen3.5-2B 使用 `--full-lm`，训练文本语言主干、CSI encoder、projector 和 CP head；视觉/MTP 辅助模块保持冻结。
- DeepSeek-R1-0528-Qwen3-8B 与 InternLM3-8B-Instruct 不使用 `--full-lm`，冻结语言模型，仅训练随机初始化的 CSI encoder、soft-prefix projector 和 CP head。梯度仍通过冻结语言模型传回输入 projector。
- 所有模型使用相同的 CP 标签、交叉熵、验证集 early stopping 和测试指标，且不使用 oracle 物理量。
- `--max-train-steps` 用于 smoke test；正式实验应省略该参数并增加 epochs/patience。

冻结 LLM 出现多数类坍缩时，可用 `--diagnostics` 将首个 batch 的 CSI、
mapped prefix、LLM pooled hidden、logits 样本间差异，以及首次反向传播的梯度范数和
首次 optimizer step 的参数改变量写入 `diagnostics.json`。`--csi-checkpoint` 可加载已训练的
CSI encoder；`--align-prefix-to-token-rms` 将连续 prefix 对齐到原生 token embedding 的
近似 RMS；`--append-query-token` 会追加一个由 BOS embedding 初始化的可训练查询 token，
并从查询位置做 CP 分类。这三个改动均为显式开关，因此可以和原始 frozen-soft-prefix
baseline 做独立消融。

## 区间与原因诊断

对指定模型的逐样本测试结果运行：

```bash
python -m downstream_cp_selection.diagnose_intervals \
  --data downstream_cp_selection/artifacts/d2los_50k_cp_labels.pt \
  --predictions downstream_cp_selection/outputs/EXPERIMENT/full_physics_finetune_test_predictions.npz \
  --feature-cache downstream_cp_selection/outputs/seed23421/frozen_features.pt \
  --output-dir downstream_cp_selection/outputs/EXPERIMENT/interval_diagnostics
```

输出包括 `interval_diagnostics.csv`、`confusion_pairs.csv`、`top_error_pairs.csv`、`diagnostics.json` 和 `interval_diagnostics.png`。`boundary_hard_rate_pct` 表示标签距离 1% CP 外能量门限不超过 0.2 个百分点的样本比例；物理预测误差用于判断错分是否与 delay spread、路径数或反射路径数估计误差相关。这些是关联性证据，严格原因仍需通过平衡采样、oracle physics 替换或删除边界样本等受控消融确认。
