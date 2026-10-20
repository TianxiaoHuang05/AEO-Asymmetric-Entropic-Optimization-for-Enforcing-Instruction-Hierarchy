# Asymmetric Entropic Optimization (AEO)

**NeurIPS 2026 · Spotlight**

Code for **Structure vs. Chaos: Asymmetric Entropic Optimization for Enforcing Instruction Hierarchy**.

AEO detects instruction-hierarchy violations by combining two hidden-state views: the final response-token representation and the mean representation over response tokens. Its asymmetric training objective encourages confident predictions for compliant responses and high predictive entropy for violations, while keeping the underlying language model frozen.

This repository provides the core implementation, the IHEval training and evaluation workflow, six main-table detector checkpoints, and score-conditioned post-hoc correction.

## Repository layout

- `core/MyDataProcessing.py` generates model replies, applies the response checker, and extracts `hd`/`hd3` features. `core/label_utils.py` and `core/template_utils.py` are called by this pipeline.
- `scripts/train_detector.py` trains the AEO MLP; `scripts/score_detector.py` scores feature caches and reports ROC metrics; `scripts/check_checkpoints.py` checks the included detector files.
- `core/MyCorrection_SCPC_anchored.py` is the recommended score-conditioned correction entry point.
- `configs/iheval/` pairs each backbone with its method configuration and recorded training metadata; `configs/iheval/checkpoints.csv` is the checkpoint index.
- `data/examples/` contains a synthetic feature-only input for an interface check.

Run commands from the repository root. Helper modules under `core/` are imported by the entry points and do not need to be run separately.

## Install

The environment specification uses Python 3.9, PyTorch 2.0.1, NumPy 1.26.4, scikit-learn 1.6.1, Transformers 4.43.1, Accelerate 1.10.1, SentencePiece 0.2.1, and tqdm 4.67.3.

```bash
conda env create -f environment.yml
conda activate MIND-AEO
```

Detector training, scoring, and cached-feature evaluation support CPU operation. Reply generation, hidden-state extraction, and correction require the selected causal language model and are intended for CUDA-capable hardware. The environment file is based on the versions available for development; installation in a clean environment has not been validated.

## Data, models, and feature format

Obtain IHEval records from the dataset's official distribution and preserve its frozen train/validation/test membership. The pipeline expects JSON files directly inside `--iheval_dir`; each file is a JSON array (or a single JSON object) with `system` or `system_message`, and `user` or `user_message` plus optional `task`. Use the original per-backbone split records when generating the corresponding main experiment. A compact split-ID manifest or public redistribution location for the frozen split is not provided here; do not replace it with a newly randomized split.

Download each base model from its official model card, accept its terms, and provide the model ID or local snapshot directory as `--model_path`. The pipeline saves actual generated text in `model_response` and a checker-derived response-level `response_label` (`follow_system` or `violate_system`). Labels refer to the model's response, not merely whether the input contains an attack or what an ideal/reference answer says.

The generated feature JSON retains the input fields and adds `hd` and `hd3`, numeric arrays of the backbone's last-layer hidden size. `hd` is the final hidden state at the last token of the concatenated prompt and response; `hd3` is the mean hidden state over the response-token span. The detector input is `hd` followed by `hd3` (`hd || hd3`), so dimensions are 3,072 for Qwen2.5-1.5B, 6,144 for Phi-3-mini, 8,192 for Mistral-7B/Llama-2-7B/Llama-3.1-8B, and 10,240 for Qwen2.5-14B.

If you already have feature caches, the scoring and training scripts accept JSON arrays with `hd`, `hd3`, and `response_label` (or the supported binary `label` values). Example:

```json
[{"sample_id":"example-1","response_label":"follow_system","hd":[0.0],"hd3":[0.1]}]
```

The actual vectors must match the selected backbone's hidden size; the one-dimensional illustrative values above only show the field structure.

## Quick CPU check

```bash
python scripts/score_detector.py \
  --checkpoint checkpoints/iheval/Qwen2.5-1.5B-Instruct.pt \
  --features data/examples/iheval_smoke_qwen15b.json \
  --output outputs/smoke_scores.json
```

This synthetic two-row file checks checkpoint loading and score output only. The output contains per-sample labels and class-1 scores, plus ROC-AUC and FPR95 when defined; these synthetic metrics are not method results.

## IHEval main experiment

### Generate replies and features

Place the frozen split JSONs directly in a split directory and run the following for the matching backbone. This operation generates sampled model responses (temperature 0.7, top-p 0.9, up to 256 new tokens), assigns response labels with the task checker, and extracts features; it requires model downloads and GPU inference.

```bash
python core/MyDataProcessing.py \
  --model_path /models/Qwen2.5-1.5B-Instruct \
  --iheval_dir /data/iheval/Qwen2.5-1.5B-Instruct \
  --new_data_dir /data/empty-new-data --gpu 0
```

The script writes generated responses and feature JSONs in model-named subdirectories beneath each input directory. Existing feature files are skipped. For subsequent training, pass the resulting train and validation feature JSON paths explicitly.

### Train a detector

The recorded main-table training settings are 20 epochs, batch size 16, learning rate `5e-4`, weight decay `1e-5`, entropy weight `1.0`, and seed 42. The trainer selects the checkpoint with the best validation ROC-AUC, breaking ties by validation accuracy. The matched `train_meta.json` records selected epoch and run metadata. Supply the corresponding frozen feature splits:

```bash
python scripts/train_detector.py \
  --train_paths /data/features/train.json \
  --val_paths /data/features/val.json \
  --output_ckpt outputs/Qwen2.5-1.5B-Instruct.pt \
  --metrics_out outputs/Qwen2.5-1.5B-Instruct.metrics.json \
  --epochs 20 --batch_size 16 --lr 5e-4 --weight_decay 1e-5 \
  --entropy_weight 1.0 --seed 42
```

These are the recorded detector-training parameters, not a replacement for the matching backbone-specific frozen split and feature inputs.

### Score and evaluate

```bash
python scripts/score_detector.py \
  --checkpoint checkpoints/iheval/Qwen2.5-1.5B-Instruct.pt \
  --features /data/features/test.json \
  --output outputs/Qwen2.5-1.5B-Instruct.test_scores.json
```

The score is `softmax(logits)[:, 1]`, where class 1 is response-level violation. ROC-AUC and FPR95 are calculated from the scored set's ROC curve; they are distinct from FPR/TPR measured at a threshold frozen on validation data. The six checkpoint/configuration pairs and SHA256 values are listed in `configs/iheval/checkpoints.csv`.

| Backbone | Input dimension | Checkpoint | Matching method configuration |
|---|---:|---|---|
| Qwen2.5-1.5B-Instruct | 3,072 | `checkpoints/iheval/Qwen2.5-1.5B-Instruct.pt` | `configs/iheval/Qwen2.5-1.5B-Instruct.method_aeo.json` |
| Phi-3-mini-128k-instruct | 6,144 | `checkpoints/iheval/Phi3-mini-128k-instruct.pt` | `configs/iheval/Phi3-mini-128k-instruct.method_aeo.json` |
| Mistral-7B-Instruct-v0.2 | 8,192 | `checkpoints/iheval/Mistral-7B-Instruct-v0.2.pt` | `configs/iheval/Mistral-7B-Instruct-v0.2.method_aeo.json` |
| Llama-2-7B-chat | 8,192 | `checkpoints/iheval/Llama-2-7b-chat-hf.pt` | `configs/iheval/Llama-2-7b-chat-hf.method_aeo.json` |
| Llama-3.1-8B-Instruct | 8,192 | `checkpoints/iheval/Meta-Llama-3.1-8B-Instruct.pt` | `configs/iheval/Meta-Llama-3.1-8B-Instruct.method_aeo.json` |
| Qwen2.5-14B-Instruct | 10,240 | `checkpoints/iheval/Qwen2.5-14B-Instruct.pt` | `configs/iheval/Qwen2.5-14B-Instruct.method_aeo.json` |

The matching `.train_meta.json` file gives training parameters and selected-epoch metadata for each row.

## Score-conditioned correction

Run `core/MyCorrection_SCPC_anchored.py` with the same backbone as the detector, an evaluation JSON containing `system`/`user`, `model_response`, `response_label`, `hd`, and `hd3`, and the desired detector. With defaults, scores below 0.1 do not trigger correction, scores from 0.1 to below 0.4 use the moderate reminder, and scores at least 0.4 use the instruction-anchored reminder. The script partitions the input pool into a 20% stratified subset and a disjoint 80% correction/evaluation pool. In the current implementation the lower trigger threshold is the fixed `--tau_mild` value; the 20% subset is not used to estimate that threshold. `--tau_strong 0.4` selects the anchored strong reminder tier.

Example for the Qwen2.5-1.5B detector and a local Qwen model:

```bash
python core/MyCorrection_SCPC_anchored.py \
  --llm_path /models/Qwen2.5-1.5B-Instruct \
  --detector_path checkpoints/iheval/Qwen2.5-1.5B-Instruct.pt \
  --eval_files /data/iheval/features/test_with_responses.json \
  --output_dir outputs/correction_qwen15b \
  --tau_mild 0.1 --tau_strong 0.4 --cal_ratio 0.2 \
  --calibration_mode stratified --gpu 0
```

Results and a summary report are written under `--output_dir`. Use `python core/MyCorrection_SCPC_anchored.py --help` for all supported arguments.

## Sources and citation

IHEval, base models, and their associated assets remain subject to their own terms. The repository's project-wide redistribution license has not been specified; no license should be inferred from included third-party materials. The data processing imports Transformers model/tokenizer implementations and the included environment records their required versions.

```text
Structure vs. Chaos: Asymmetric Entropic Optimization for Enforcing Instruction Hierarchy.
NeurIPS 2026.
```
