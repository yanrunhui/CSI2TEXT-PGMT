# 三个大语言模型的完整 CSI 波束预测 baseline

脚本：[llm_baselines.py](llm_baselines.py)。它复用 `full_csi_50k_beams.pt` 的同一 40k/5k/5k 物理链路划分和 `oracle_beam` 标签，分别比较：

| 参数 `--model` | 官方权重 ID |
| --- | --- |
| `deepseek_r1_qwen3_8b` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |
| `internlm3_8b` | `internlm/internlm3-8b-instruct` |
| `qwen3_5_2b` | `Qwen/Qwen3.5-2B` |

脚本支持两种监督学习模式：默认的 `--backbone-training frozen` 冻结 LLM、训练 CSI 投影器/位置向量/读出向量/64 类分类头；`--backbone-training full` 从预训练 LLM 权重开始，**更新全部主干参数**及上述 CSI 模块。后一种模式叫全参数微调，不叫 Scratch：Scratch 意味着 LLM 主干也从随机权重开始。两种模式每条输入都是原始 16×8×128 的完整 beamspace CSI token；投影为 16 个连续 LLM embedding，追加一个读出 token，取其最终隐藏状态分类。没有使用文字形式逐个列出 64 个波束增益；`beam_gains` 仅用于测试阶段计算选错波束的增益损失。也没有把真实 AoD、时延或功率作为输入。**这不是零样本提示词实验，也不能说明 LLM 理解了无线电物理。**

## 环境

三个模型分别运行，尤其 Qwen3.5 要使用 Transformers 5.x，而 InternLM3 推荐 4.x；不要把两套版本装进同一个环境。先安装与你服务器 CUDA 匹配的 PyTorch，然后在各环境中安装 `numpy`、`scipy`、`pyyaml`、`accelerate`、`bitsandbytes`、`safetensors` 和相应 Transformers 版本。DeepSeek-Qwen3 建议 `transformers>=4.51,<5`；InternLM3 官方要求 `transformers>=4.48`，因此**不要直接照搬仓库旧的 `requirements-internlm3.txt`（它固定为 4.47.1）**；Qwen3.5 可先用仓库 `requirements-qwen.txt`，若该版本尚不支持下载的权重，按模型卡要求安装最新 Transformers，并加装 `torchvision`、`Pillow`。首次运行若权重已在 HF 缓存中，官方 ID 即可；若没有缓存，显式添加 `--allow-download`。也可用 `--model-path /服务器上的模型目录` 指向本地权重，尤其适合离线服务器。

`InternLM3` 加载时需要 `trust_remote_code=True`，因此只使用可信的官方或已审阅过的本地权重。Qwen3.5 是多模态模型，但本实验仅使用它的文本主干处理连续 CSI embedding，不输入图像。

## 冻结主干训练命令

以下以仓库在服务器的路径 `/home/yrh/CSI_model/CSI_model` 为例。先在终端激活对应环境，三个命令的输出目录不能复用；每次用一张空闲 GPU，通过 `CUDA_VISIBLE_DEVICES` 指定。`--model-path` 可添加到任一命令。

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_deepseek
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model deepseek_r1_qwen3_8b \
  --output-dir downstream_beam_selection/outputs/llm_deepseek_r1_qwen3_8b \
  --batch-size 2 --grad-accum 16 --epochs 10 --patience 3 \
  --gradient-checkpointing
```

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_internlm3
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model internlm3_8b \
  --output-dir downstream_beam_selection/outputs/llm_internlm3_8b \
  --batch-size 2 --grad-accum 16 --epochs 10 --patience 3 \
  --gradient-checkpointing
```

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_qwen35
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model qwen3_5_2b \
  --output-dir downstream_beam_selection/outputs/llm_qwen3_5_2b \
  --batch-size 2 --grad-accum 16 --epochs 10 --patience 3 \
  --gradient-checkpointing
```

tmux 中可执行相同命令，例如 `tmux new -s llm_beam` 后粘贴命令，按 `Ctrl+b` 再按 `d` 脱离。若 GPU 不支持 BF16，追加 `--compute-dtype fp16`。显存不足时将 `--batch-size` 改为 1，增大 `--grad-accum` 以维持有效 batch，并保留 `--gradient-checkpointing`。8B 主干即使 4-bit 冻结，反向传播到投影器也可能需要较大显存、训练时间明显长于原 256 维 CSIEncoder。

## 全参数微调命令

下列三条命令会更新三个预训练 LLM 的**全部主干权重**。必须指定 `--quantization none`；4-bit 量化权重不能用此脚本做全参数梯度更新。`--learning-rate` 控制新建的 CSI 模块，`--backbone-learning-rate` 控制预训练主干。优先在空闲的大显存卡上尝试 Qwen3.5-2B；两个 8B 模型的参数、梯度和 AdamW 优化器状态在单卡上可能需要数十 GB 以上，24 GB 卡通常不能运行。脚本当前仅支持**单卡**全参数微调，不会自动分片到多卡；若服务器没有足够大的单卡，应保持冻结模式，或另做 FSDP/ZeRO 分布式训练，不能将 LoRA 称为“主干全部解冻”。

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_qwen35
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model qwen3_5_2b \
  --output-dir downstream_beam_selection/outputs/llm_qwen3_5_2b_full \
  --backbone-training full --quantization none \
  --batch-size 1 --grad-accum 32 --epochs 10 --patience 3 \
  --backbone-learning-rate 1e-5 --learning-rate 2e-4 \
  --gradient-checkpointing
```

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_deepseek
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model deepseek_r1_qwen3_8b \
  --output-dir downstream_beam_selection/outputs/llm_deepseek_r1_qwen3_8b_full \
  --backbone-training full --quantization none \
  --batch-size 1 --grad-accum 32 --epochs 10 --patience 3 \
  --backbone-learning-rate 1e-5 --learning-rate 2e-4 \
  --gradient-checkpointing
```

```bash
cd /home/yrh/CSI_model/CSI_model
conda activate csi_llm_internlm3
CUDA_VISIBLE_DEVICES=1 python -u downstream_beam_selection/llm_baselines.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --model internlm3_8b \
  --output-dir downstream_beam_selection/outputs/llm_internlm3_8b_full \
  --backbone-training full --quantization none \
  --batch-size 1 --grad-accum 32 --epochs 10 --patience 3 \
  --backbone-learning-rate 1e-5 --learning-rate 2e-4 \
  --gradient-checkpointing
```

每个输出目录保存 `best_adapter.pt`、`results.csv`、`results.json`、`test_predictions.csv` 和 `class_distribution.json`。全参数模式另存 `best_backbone/`（体积接近完整模型），并在测试前重新加载验证集最佳主干；冻结模式的 checkpoint 只保存小型可训练参数。验证集只用于选最佳 checkpoint，测试集只在训练结束后评估一次。结果沿用现有 Top-1、Top-3、macro accuracy、LoS/NLoS 分层指标及增益损失。对照现有 CSIEncoder 的 50 epoch 结果时，应注明这里默认最多 10 epoch、有效 batch 为 32，以及不同模型参数量与训练成本；如果要做严格的训练预算对照，请统一预算再重跑。

因为完整 CSI 已包含所有候选波束的复响应，直接算 64 个 `beam_gains` 的 `argmax` 是确定性 oracle。这个 LLM 实验衡量的是从 CSI 学习该映射的能力，不是降低波束扫描开销。预训练 LLM 的语料是否含有相似文字无法像 CSI 物理链路那样审计，但这份任务的 40k/5k/5k `group_id` 在数据加载时会检查互不重叠。
