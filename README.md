# FirstDelayNet

Code accompanying the manuscript:

**Predicting the First Domino: Early Delay Detection in Transport Chains Using a Teacher–Student Framework**

**Authors:** Manuel Pimpão and Henrik Leopold  
**Affiliation:** Kühne Logistics University, Hamburg, Germany

## Overview

This repository contains the anonymized implementation of **FirstDelayNet**, a teacher–student framework for detecting the onset of the first delay in transport-driven industrial processes.

At an observation time \(\tau\) within an ongoing transport interval, the model predicts whether the upcoming transport checkpoint will be the first delayed checkpoint of the case. The implementation combines:

- a Transformer teacher operating on heterogeneous event sequences;
- leakage-safe teacher logits and latent sequence representations;
- PCA-compressed teacher embeddings;
- engineered \(\tau\)-anchored operational features;
- an XGBoost student trained under first-delay hazard supervision.

Only information available at or before the observation time \(\tau\) is used as model input.

## Repository Contents

```text
.
├── data/
│   └── anonymized/
│       ├── train_events_anonymized.parquet
│       ├── val_events_anonymized.parquet
│       ├── tau_train_anonymized.parquet
│       ├── tau_val_anonymized.parquet
│       ├── tau_test_anonymized.parquet
│       ├── raw_train_anonymized.parquet
│       ├── raw_val_anonymized.parquet
│       └── raw_test_anonymized.parquet
├── model_artifacts/
│   └── teacher/
├── results/
├── data_loader_anonymized.py
├── teacher_dataset_anonymized.py
├── teacher_model_anonymized.py
├── train_teacher_anonymized.py
├── distill_features_anonymized.py
├── features_xgb_anonymized.py
├── model_xgb_anonymized.py
├── main_train_anonymized.py
├── test_anonymized.py
├── requirements.txt
├── LICENSE
└── README.md
```

The data files and trained artifacts are included only when their publication has been approved.

## Script Description

### `data_loader_anonymized.py`

Loads event-log data from one CSV file or a directory of CSV files. It supports optional Parquet caching, PM4Py-compatible column names, passthrough loading, and optional sequence-model enrichment.

### `teacher_dataset_anonymized.py`

Constructs masked and leakage-safe sequence datasets for the teacher. It creates categorical token sequences, numerical feature sequences, masks, labels, vocabulary files, configuration files, and keys linking sequences to their source cases and prefixes.

### `teacher_model_anonymized.py`

Defines the Transformer teacher model, its configuration, masked binary loss utilities, representation extraction, training support, metrics, and diagnostic outputs.

### `train_teacher_anonymized.py`

Trains and validates the Transformer teacher using:

```text
data/anonymized/train_events_anonymized.parquet
data/anonymized/val_events_anonymized.parquet
```

The script saves the trained teacher, vocabulary, configuration, normalization statistics, metrics, logs, keys, and plots under:

```text
model_artifacts/teacher/
```

### `distill_features_anonymized.py`

Loads the trained teacher and computes teacher logits and optional PCA-compressed embeddings for event-log prefixes or \(\tau\)-anchored snapshots. It uses length-sorted, token-budgeted batches to reduce memory usage. PCA is fitted on the training split and reused for validation and test data.

### `features_xgb_anonymized.py`

Performs lightweight feature engineering and assembles the tabular input matrices used by the XGBoost student. It includes numerical normalization, cyclical time encoding, congestion-related interactions, and a denylist for target, future, identifier, and leakage-prone variables.

### `model_xgb_anonymized.py`

Defines and trains the XGBoost student classifier. It exports the fitted model, validation operating threshold, classification metrics, threshold analyses, calibration results, feature importance, and diagnostic plots.

### `main_train_anonymized.py`

Runs the main student-training pipeline:

1. Load the \(\tau\)-level training and validation tables.
2. Load the matching raw event histories.
3. retain only events available up to each observation time \(\tau\);
4. generate teacher logits and PCA-compressed embeddings;
5. merge teacher-derived features into the \(\tau\)-level tables;
6. construct XGBoost features;
7. train and save the student model;
8. determine and save the validation operating threshold.

The main student model is saved as:

```text
results/xgb_student_tau.joblib
```

The validation operating threshold is saved as:

```text
results/val_operating_threshold.txt
```

### `test_anonymized.py`

Evaluates the trained teacher–student pipeline on the held-out test split. It reports row-level, decision-point, pre-first-window, case-level, false-alarm, calibration, and early-warning metrics.

Important outputs include:

```text
results/test_tau/test_tau_predictions.parquet
results/test_tau/test_tau_metrics.json
results/test_tau/tau_rows_with_alerts.parquet
results/test_tau/case_level_alert_summary.csv
results/test_tau/plots_test_tau/
```

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The main packages are:

```text
numpy
pandas
pyarrow
scikit-learn
xgboost
torch
joblib
matplotlib
pm4py
```

PM4Py is optional when only DataFrame-based loading is used.

## Expected Data Files

The default scripts expect the following files:

| File | Purpose |
|---|---|
| `train_events_anonymized.parquet` | Event-level teacher training data |
| `val_events_anonymized.parquet` | Event-level teacher validation data |
| `tau_train_anonymized.parquet` | Student training observations |
| `tau_val_anonymized.parquet` | Student validation observations |
| `tau_test_anonymized.parquet` | Student test observations |
| `raw_train_anonymized.parquet` | Raw training histories used for distillation |
| `raw_val_anonymized.parquet` | Raw validation histories used for distillation |
| `raw_test_anonymized.parquet` | Raw test histories used for evaluation |

All paths can be adapted directly in the configuration sections of the corresponding scripts. The test script additionally supports command-line path overrides.

## Running the Pipeline

Run all commands from the repository root.

### 1. Train the Transformer teacher

```bash
python train_teacher_anonymized.py
```

This creates the teacher artifacts in:

```text
model_artifacts/teacher/
```

### 2. Train the XGBoost student

```bash
python main_train_anonymized.py
```

This generates teacher-derived features at each \(\tau\), trains the XGBoost student, and saves the model and validation outputs in `results/`.

### 3. Evaluate on the test split

```bash
python test_anonymized.py \
  --tau-test data/anonymized/tau_test_anonymized.parquet \
  --raw-test data/anonymized/raw_test_anonymized.parquet \
  --model results/xgb_student_tau.joblib \
  --result-dir results \
  --art-dir model_artifacts/teacher \
  --pca-path model_artifacts/teacher/teacher_pca_tau.pkl \
  --out-dir results/test_tau
```

On Windows Command Prompt, use `^` instead of `\` for line continuation:

```bat
python test_anonymized.py ^
  --tau-test "data\anonymized\tau_test_anonymized.parquet" ^
  --raw-test "data\anonymized\raw_test_anonymized.parquet" ^
  --model "results\xgb_student_tau.joblib" ^
  --result-dir "results" ^
  --art-dir "model_artifacts\teacher" ^
  --pca-path "model_artifacts\teacher\teacher_pca_tau.pkl" ^
  --out-dir "results\test_tau"
```

To evaluate without teacher-derived features:

```bash
python test_anonymized.py --student-only
```

This mode is only appropriate when the saved student was trained without teacher features.

## Evaluation Configurations

The evaluation code supports the operational perspectives used in the manuscript:

- **Config A – GlobalTau:** global ranking across all \(\tau\) prefixes;
- **Config B – PreFirstDelayed:** stability and first-delay detection before the first delay in delayed cases;
- **Config C – Kminus1PerCase:** one prediction per case at the operational decision point;
- **Config D – PreFirstAllCases:** pre-first evaluation including non-delayed cases;
- **Config E – CaseAlarm:** case-level alarm quality.

Reported measures include AUPRC, AUROC, precision, recall, F1-score, Brier score, decision-point recall, pre-first precision, false-alarm burden, and early-warning lead time.

## Leakage Prevention

The implementation includes several safeguards against temporal leakage:

- sequence prefixes contain only events available up to the relevant prefix;
- teacher features at \(\tau\) are computed only from events with timestamps \(t \leq \tau\);
- future checkpoint outcomes are used only for offline label construction;
- PCA is fitted on the training split and reused unchanged;
- direct targets, future variables, and leakage-prone columns are excluded from student features;
- post-first-delay observations are excluded from hazard-model training;
- validation thresholds are fixed before test evaluation.

## Reproducibility Notes

The scripts use a default random seed of `42` where applicable. Exact reproducibility may still depend on:

- Python and package versions;
- CPU or GPU execution;
- CUDA and cuDNN versions;
- XGBoost version;
- available memory;
- hardware-dependent nondeterminism.

For a fixed archival release, record exact package versions in `requirements.txt`.

## Data Anonymization

The publicly released data must contain only approved anonymized information. Direct identifiers, confidential internal identifiers, credentials, personal data, and unapproved operational details must not be included.

The repository should include a separate data description, preferably:

```text
data/README.md
```

It should document the released files, column meanings, anonymization procedure, removed attributes, transformations, and limitations.

## Data and Code Availability

Suggested manuscript statement:

> The anonymized data and Python implementation supporting the findings of this study are publicly available in the accompanying GitHub repository: https://github.com/manuelpimpao/firstdelaynet/.

## License

A license should be added only after confirming that it is compatible with the applicable Kühne Logistics University and Volkswagen publication requirements.

Without an explicit license, the repository remains protected by copyright and reuse is not automatically permitted.

## Contact

**Manuel Pimpão**  
Kühne Logistics University  
Hamburg, Germany
