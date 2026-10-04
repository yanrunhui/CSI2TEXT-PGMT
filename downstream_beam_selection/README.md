# Full-CSI Best Transmit Beam Prediction

The primary experiment uses **all 16 CSI patch tokens** (all 64 complex DFT
beams) to predict one of 64 fixed transmit beams. The beam-only baselines are
Scratch (train encoder + head), Frozen (pretrained encoder, train head only),
and Fine-tune (pretrained encoder + head both trainable). Two matched auxiliary
supervision variants, `scratch_physics` and `fine_tune_physics`, use the same
beam model plus randomly initialized physical heads. They differ only in
whether the CSI encoder starts from pretrained weights. This experiment does
not mask CSI, model a beam-probing procedure, or report scanning-overhead savings.
Since the complete CSI contains every beam response, direct gain argmax is a
deterministic 100% oracle by definition; the neural-network comparison tests
representation learning/transfer, not whether a network beats that rule.

For the three LLM CSI-adapter baselines (DeepSeek-R1-Qwen3-8B,
InternLM3-8B-Instruct, Qwen3.5-2B) in frozen or full-fine-tune mode,
see [LLM_BASELINES.md](LLM_BASELINES.md).

`beam_gains[b]` is the mean across 128 subcarriers of the unit-norm DFT beam
power. The target `oracle_beam` is its argmax. Existing `[16,8,128]` patch
tokens are unpacked to `[64,128]` complex beam responses in row-major 8x8
order; the preparation script verifies the first few unpack/repack examples.

Use the **same** 40k pretraining train split and disjoint 10k holdout split.
The latter becomes balanced 5k validation and 5k final test links. This
preserves physical-link IDs and avoids independently resampling from the 200k
pool, which could leak pretraining links into the downstream test set.

```bash
python downstream_beam_selection/prepare_data.py \
  --train-input artifacts/d2los_50k_upa8x8_nf128_aod_balanced/d2los_50k_upa8x8_nf128_aod_train.pt \
  --holdout-input artifacts/d2los_50k_upa8x8_nf128_aod_balanced/d2los_50k_upa8x8_nf128_aod_test.pt \
  --output downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --train-size 40000 --val-size 5000 --test-size 5000 --seed 23421

python downstream_beam_selection/inspect_labels.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --output-dir downstream_beam_selection/outputs/full_csi_label_check

python downstream_beam_selection/full_csi_experiment.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --checkpoint artifacts/pretrain_d2los_50k_aod_context/checkpoint_last.pt \
  --output-dir downstream_beam_selection/outputs/full_csi_50k \
  --models scratch frozen fine_tune --require-clean-pretrain \
  --batch-size 128 --epochs 50 --patience 8 --seed 23421
```

To run only the two new physics comparisons without repeating the beam-only
baselines, use a new output directory and select their names explicitly:

```bash
python downstream_beam_selection/full_csi_experiment.py \
  --data downstream_beam_selection/artifacts/full_csi_50k_beams.pt \
  --checkpoint artifacts/pretrain_d2los_50k_aod_context/checkpoint_last.pt \
  --output-dir downstream_beam_selection/outputs/full_csi_50k_physics \
  --models scratch_physics fine_tune_physics \
  --require-clean-pretrain --physics-weight 0.1 \
  --batch-size 128 --epochs 50 --patience 8 --seed 23421
```

For a fully matched new run, choose `--models scratch fine_tune
scratch_physics fine_tune_physics` instead. Each model gets the same shuffled
train order. Physics supervision uses the 13 normalized targets defined in
`data/dataset.py` (including delay, power, first-path AoA/AoD sin/cos and path
statistics), plus a LoS binary head. Missing regression values are masked.
The joint loss is beam cross-entropy plus `--physics-weight` times the sum of
masked smooth-L1 regression and LoS BCE. Only training links supply physical targets;
validation selects by beam cross-entropy alone, and test inference never reads
ground-truth physical labels. Physical heads start from scratch in both
variants, so this comparison isolates the encoder initialization. The output
`training_physics_coverage.json` records finite target counts for the train
split, and `history.json` records beam and physical training loss separately.

The runner audits the checkpoint's recorded training sources and transfer
ancestors by `group_id` before loading the full CSI corpus. If any source is
missing, the audit is `unknown`; if overlap exists, it is `overlap`. The
`--require-clean-pretrain` flag stops unless the audit is `clean`.
If an ancestor checkpoint or its training `.pt` was moved or deleted, restore
it at the recorded path for a clean audit; otherwise omit this flag and report
the result as provenance-unknown rather than unseen-link transfer. Results
include Top-1, Top-3, Top-1 beam gain-loss mean/median/P90, the fraction within
1 dB, per-LoS/NLoS metrics, and 64-class support/majority-baseline reports.
The test split is evaluated once per model after validation selects its best
epoch. The Top-3 figure is a classification hit rate, not a claim that three
beams were measured.

## Earlier sparse-observation experiment

The code below remains available for a separate sparse-probing study and is
not part of the full-CSI experiment above.

### Adaptive 64-Beam Selection

This folder implements transmit-beam prediction and Top-K adaptive search for the
UPA-8x8, single-Rx CSI data. It does not rerun RayVerse: the unshifted 8x8 DFT
beam response is reconstructed exactly from the existing 16 `[8, Nf]` patch
tokens.

## Task contract

- Array: UPA-8x8, 64 unshifted DFT beams.
- Beam ID: `beam_id = row * 8 + col`.
- Oracle: beam with the largest mean unit-norm DFT power across subcarriers.
- Sparse input: 8/16/32 observed beams, represented by 2/4/8 observed 2x2 patch
  tokens. Unobserved tokens are zeroed and excluded by `token_mask`.
- Prediction: 64 logits followed by Top-K candidate search.

`first_path_aod` remains an auxiliary physical feature. It is not used as the
oracle beam label.

## 1. Generate labels and balanced splits

The command can read the 200k candidate pool directly. It selects 25k LoS and
25k NLoS links, then creates leakage-free 40k/5k/5k splits.

```bash
python downstream_beam_selection/prepare_data.py \
  --input artifacts/d2los_200k_upa8x8_nf128_aod_seed23421.pt \
  --output downstream_beam_selection/artifacts/d2los_50k_beam_labels.pt \
  --train-size 40000 \
  --val-size 5000 \
  --test-size 5000 \
  --seed 23421
```

The command also materializes `d2los_50k_beam_labels_samples.pt` so later
training loads only the selected 50k CSI samples rather than reopening the full
200k pool. The label file stores references to that compact sample file, 64 beam
gains, `oracle_beam`, `oracle_top5`, oracle gain, the 1 dB acceptable-beam mask,
and split indices.

Optional label inspection:

```bash
python downstream_beam_selection/inspect_labels.py \
  --data downstream_beam_selection/artifacts/d2los_50k_beam_labels.pt \
  --output-dir downstream_beam_selection/outputs/label_inspection
```

## 2. Main Sparse-16 / Top-3 experiment

Use a checkpoint trained with the AoD context branch enabled. The default
`configs/train.yaml` now sets `use_first_path_aod_context_encoder: true` and
`first_path_aod_weight: 0.02` plus `first_path_aod_nlos_weight: 0.01`,
independently of `aux_regression_weight: 0.005`.
Older checkpoints still load, but do not contain the new AoD context branch;
retrain or transfer-train before comparing Full+Tx Physics.

```bash
python downstream_beam_selection/run_experiment.py \
  --data downstream_beam_selection/artifacts/d2los_50k_beam_labels.pt \
  --checkpoint artifacts/YOUR_PRETRAIN_RUN/checkpoint_last.pt \
  --output-dir downstream_beam_selection/outputs/sparse16_top3 \
  --models all \
  --observed-beams 16 \
  --top-k 1 3 5 8 \
  --observation-strategy uniform \
  --batch-size 128 \
  --epochs 50 \
  --patience 8 \
  --seed 23421
```

## 3. Full observation sweep

```bash
python downstream_beam_selection/run_experiment.py \
  --data downstream_beam_selection/artifacts/d2los_50k_beam_labels.pt \
  --checkpoint artifacts/YOUR_PRETRAIN_RUN/checkpoint_last.pt \
  --output-dir downstream_beam_selection/outputs/observation_sweep \
  --models all \
  --observed-beams 8 16 32 64 \
  --top-k 1 3 5 8 \
  --observation-strategy uniform \
  --class-weight balanced \
  --seed 23421
```

Models:

- `scratch`: randomly initialized CSI encoder and beam head.
- `full_embedding`: frozen pretrained 256-D CSI embedding.
- `full_los_k`: frozen embedding plus predicted LoS probability and K-factor.
- `full_tx_physics`: frozen embedding plus predicted first-path AoD sin/cos,
  LoS probability, K-factor, and path count.

The physics variants use predictions from the checkpoint, never ground-truth
test labels. Frozen features are cached once per observation setting. The runner
rejects `full_tx_physics` when checkpoint metadata does not confirm AoD
supervision; `--allow-untrained-tx-physics` exists only for debugging.

## Metrics and outputs

`results.json` and `results.csv` contain:

- Top-1/3/5/8 oracle hit rate.
- Top-K hit rate for any beam within 1 dB of oracle.
- Mean, median, P90, and P95 gain loss.
- Fractions within 1 dB and 3 dB.
- Nominal search measurements and overhead reduction, both including and
  excluding the initial observation cost.
- Wideband rate retention using the per-subcarrier power and
  `--noise-power-dbw` (default `-117 dBW`).

Use `--skip-rate-metric` when memory is constrained. The default test split
stores its temporary per-subcarrier SNR-in-dB tensor as float16 (about 82 MB for
5k samples), avoiding both watt-domain underflow and linear-SNR overflow.
