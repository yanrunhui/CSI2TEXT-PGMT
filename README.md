# CSI2TEXT-PGMT

CSI2TEXT-PGMT is a physics-grounded framework for learning semantic representations from channel state information (CSI).

The repository contains the core preprocessing, pretraining, evaluation, robustness experiments, cross-configuration generalization experiments, and downstream-task implementations used by the project.

## Highlights

- CSI-to-text representation learning
- Physics-grounded multi-task training
- Support for different receiver-array configurations
- Cross-array and cross-numerology generalization
- AWGN and channel-condition robustness evaluation
- Real-world BUPT CSI/CIR processing
- Multiple downstream communication and sensing tasks

## Repository Structure

```text
CSI_model/
├── configs/
│   ├── train.yaml                 # Canonical training configuration
│   └── experiments/
│       └── siso.yaml              # SISO/BUPT experiment configuration
├── data/                          # Dataset loading and CSI preprocessing modules
├── models/                        # CSI2TEXT-PGMT model components
├── training/                      # Training objectives and trainer implementation
├── scripts/                       # Preprocessing, training and evaluation scripts
├── tests/                         # Automated tests
├── downstream_beam_selection/     # Beam-selection task
├── downstream_cp_selection/       # Cyclic-prefix selection task
├── downstream_localization/       # Localization task
├── downstream_mcs_adaptation/     # MCS adaptation task
├── downstream_outage_prediction/  # Outage-prediction task
└── requirements*.txt              # Core and optional dependencies
```

Generated datasets, checkpoints and experiment outputs are intentionally excluded from Git.

## Environment

Python 3.10 or 3.11 is recommended.

```bash
conda create -n csi2text python=3.11 -y
conda activate csi2text

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For plotting, dataset conversion and paper experiments:

```bash
python -m pip install -r requirements-experiments.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

Optional model-specific environments are documented by:

```text
requirements-internlm3.txt
requirements-qwen.txt
requirements-deepseek.txt
requirements-sionna.txt
```

These optional dependencies should preferably be installed in separate environments.

## Quick Start

Run the built-in smoke test:

```bash
python scripts/pretrain.py --smoke-test
```

### 1. Preprocess CSI Data

For a DeepMIMO D2Los dataset:

```bash
python scripts/preprocess_all.py \
  --d2los-root deepmimo_scenarios/D2Los_Data \
  --output artifacts/d2los.pt \
  --max-samples 10000
```

Use `--help` to inspect all preprocessing options:

```bash
python scripts/preprocess_all.py --help
```

### 2. Train CSI2TEXT-PGMT

The canonical release configuration is `configs/train.yaml`.

```bash
python scripts/pretrain.py \
  --config configs/train.yaml \
  --data-path artifacts/d2los.pt
```

Paths in the configuration file are examples and may need to be adjusted for your local dataset and output directories.

### 3. Evaluate a Checkpoint

```bash
python scripts/evaluate.py \
  --data-path artifacts/d2los.pt \
  --checkpoint path/to/checkpoint.pt
```

Available evaluation arguments can be inspected with:

```bash
python scripts/evaluate.py --help
```

## Experiment Entry Points

The `scripts/` directory contains the reproducibility entry points for the major experiment groups.

| Experiment | Representative entry points |
|---|---|
| AWGN robustness | `evaluate_noise_robustness.py`, `train_noise_augmented_full.sh` |
| Cross-array generalization | `run_strict_cross_array_generalization.sh`, `validate_cross_array_splits.py` |
| Cross-numerology generalization | `run_full_multitask_cross_numerology_nf128.sh`, `validate_cross_numerology_splits.py` |
| Semantic-anchor ablation | `run_semantic_anchor_ablation.sh` |
| Other ablations | `run_ablation_experiments.py` |
| BUPT/SISO experiments | `preprocess_bupt_cir.py`, `evaluate_bupt_delay_spread.py` |
| BUPT data diagnostics | `diagnose_bupt_beam_scan.py`, `validate_bupt_scan_states.py` |

Most experiment scripts expose additional options through `--help`. Shell scripts may contain dataset and output paths that must be adapted to the local environment.

## Downstream Tasks

Task-specific instructions are provided in the corresponding directories:

- [Beam selection](downstream_beam_selection/README.md)
- [Adaptive cyclic-prefix selection](downstream_cp_selection/README.md)
- [CSI-based localization](downstream_localization/README.md)
- [MCS adaptation](downstream_mcs_adaptation/README.md)
- [Outage prediction](downstream_outage_prediction/README.md)

The downstream tasks are kept separate from the core pretraining workflow so that they can be reproduced and extended independently.

## BUPT Real-World Data

Utilities for processing and validating the BUPT measurements include:

```text
scripts/build_bupt_medium_pool.py
scripts/diagnose_bupt_beam_scan.py
scripts/make_bupt_experiment_subsets.py
scripts/preprocess_bupt_cir.py
scripts/validate_bupt_scan_states.py
configs/experiments/siso.yaml
```

Raw measurement data is not distributed in this repository.

## Qualitative Examples

Export qualitative model predictions with:

```bash
python scripts/export_qualitative_examples.py \
  --checkpoint path/to/checkpoint.pt \
  --data-path artifacts/d2los.pt \
  --output artifacts/qualitative_examples.json
```

## Testing

Run the test suite from the repository root:

```bash
python -m pytest -q
```

## Data and Checkpoints

Large datasets, generated artifacts and trained checkpoints are not tracked by Git. Store them in ignored directories such as:

```text
artifacts/
checkpoints/
outputs/
results/
```

Release links for pretrained checkpoints and processed datasets can be added here when they become publicly available.

## Citation

Citation metadata will be provided in `CITATION.cff` before the public `v1.0.0` release.

## License

No open-source license is currently granted for this repository. Unless otherwise stated, all rights are reserved.

The source code may be viewed for academic reference. Copying, modification, redistribution, commercial use, or incorporation into other projects requires prior written permission from the copyright holder(s).
