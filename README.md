# CrossConfig-CSI

Reference implementation scaffold for Cross-Configuration CSI pretraining with physics-grounded text supervision.

## What's included

- Offline CSI preprocessing and unified beamspace tokenization
- Semantic key construction and caption generation
- Caption tokenizer and PyTorch dataset utilities
- CSI encoder, text encoder, observation classifier, and full model wrapper
- Losses, staged schedules, and a simple trainer
- Config files and runnable entrypoint scripts

## Quick start

```bash
python scripts/pretrain.py --smoke-test
```

That command builds a synthetic batch, runs a forward pass, computes losses, and executes a tiny training loop to verify the pipeline.

## D2Los / RayVerse data

RayVerse dataset repository: https://github.com/UNIC-Lab/RayVerse.git

Convert the imported `D2Los_Data` tree to the unified training `.pt` format first:

```bash
python scripts/preprocess_all.py \
  --d2los-root deepmimo_scenarios/D2Los_Data \
  --output artifacts/d2los.pt \
  --max-samples 10000
```

The dataset is large, so set `--max-samples` or D2Los-specific limits such as
`--max-maps`, `--max-sources-per-map`, and `--max-rx-per-source` for each run.
Then train with:

```bash
python scripts/pretrain.py --data-path artifacts/d2los.pt
```

To inspect how balanced the path-related labels are before training:

```bash
python scripts/diagnose_path_distribution.py \
  --input artifacts/d2los.pt \
  --samples-per-key 4 \
  --num-keys-per-batch 8
```

## Layout

See the folder structure in the accompanying design document for the intended expansion path.

## Qualitative CSI-to-Text examples

After producing aligned `signal_descriptions.pt` payloads for the proposed model and a direct-LLM
baseline, export the three paper cases with:

```bash
python scripts/export_qualitative_channel_reports.py \
  --ours artifacts/ours/evaluation/signal_descriptions.pt \
  --direct-llm InternLM3-8B=artifacts/internlm3_8b_csi_prefix/seed_0/evaluation/signal_descriptions.pt \
  --output-dir artifacts/qualitative_channel_reports
```

The script intersects the payloads by sample index and verifies that every aligned target record is
identical, so a full-cohort proposed-model payload can be compared safely with a shorter direct-LLM
subset. It then deterministically selects a typical LoS case, a complex NLoS case, and a challenging
case. It audits both the structured records and values actually stated in the reports. The output
directory contains JSON and CSV audit data, a Markdown preview, and a double-column LaTeX table that
can be included in the paper with `\input{qualitative_channel_reports.tex}`. Field tolerances default
to those used by `evaluate_signal_descriptions.py` and can be overridden with the generated
`--tolerance-*` options.
