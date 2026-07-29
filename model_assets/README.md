# Model Assets

This folder contains the trained model files and supporting artifacts used by the FirstDelayNet teacher–student pipeline.

## Folder Structure

```text
model_assets/
├── xgb_next_delay_tau.joblib
├── teacher_pca_tau.pkl
├── teacher_artifacts/
│   ├── teacher_model_best.pt
│   ├── teacher_vocab.json
│   ├── teacher_config.json
│   ├── teacher_model_config.json
│   ├── teacher_num_cols.json
│   ├── num_feature_norm_stats.npz
│   ├── teacher_pca_tau.pkl
│   ├── teacher_train_keys.parquet
│   ├── teacher_val_keys.parquet
│   ├── teacher_train_log.jsonl
│   └── summary.json
└── clusters/
    ├── kmeans_case_profile.joblib
    └── scaler_case_profile.joblib
```

## Main Student Model

### `xgb_next_delay_tau.joblib`

Trained XGBoost student model for predicting whether the upcoming transport checkpoint will constitute the first delay of the case.

The model expects the same engineered features and teacher-derived inputs used during training.

Load it with:

```python
import joblib

model = joblib.load("model_assets/xgb_next_delay_tau.joblib")
```

## Teacher Artifacts

The `teacher_artifacts/` folder contains all files required to reconstruct and execute the Transformer teacher.

### `teacher_model_best.pt`

Best saved Transformer teacher checkpoint.

This file contains the trained PyTorch model weights used to generate:

- teacher logits;
- latent sequence embeddings;
- teacher-derived features for the XGBoost student.

### `teacher_vocab.json`

Categorical vocabulary used by the teacher model.

It maps event tokens to the integer identifiers expected by the trained embedding layer.

### `teacher_config.json`

Dataset-related teacher configuration, including fields such as:

- categorical feature column;
- numerical feature columns;
- case identifier column;
- prefix column;
- mask column;
- maximum sequence length;
- timestamp column.

### `teacher_model_config.json`

Transformer architecture configuration, including parameters such as:

- embedding and hidden dimensions;
- number of attention heads;
- number of Transformer layers;
- feed-forward dimension;
- dropout;
- maximum sequence length;
- number of numerical features.

### `teacher_num_cols.json`

Ordered list of numerical input columns used by the teacher.

The order must remain unchanged because it must match the numerical channels used during training.

### `num_feature_norm_stats.npz`

Training-set normalization statistics for the teacher's numerical inputs.

It typically contains the mean and standard deviation applied before inference.

### `teacher_pca_tau.pkl`

PCA transformation fitted on teacher embeddings from the training split.

It converts high-dimensional teacher representations into the compressed components used by the XGBoost student.

Load it with:

```python
import joblib

pca = joblib.load(
    "model_assets/teacher_artifacts/teacher_pca_tau.pkl"
)
```

### `teacher_train_keys.parquet`

Metadata linking teacher training sequences or prefixes to their corresponding cases and positions.

This file is mainly useful for reproducibility, diagnostics, and alignment checks.

### `teacher_val_keys.parquet`

Metadata linking teacher validation sequences or prefixes to their corresponding cases and positions.

### `teacher_train_log.jsonl`

Training log generated during teacher training.

It may contain epoch-level or step-level information such as:

- training and validation loss;
- AUPRC and AUROC;
- selected thresholds;
- early-stopping information;
- runtime diagnostics.

This file is not required for inference.

### `summary.json`

Compact summary of the teacher training run and selected model.

This file is mainly intended for documentation and reproducibility.

## Root-Level PCA File

### `teacher_pca_tau.pkl`

A second copy of the PCA artifact is currently stored directly under `model_assets/`.

To avoid ambiguity, keep only one canonical copy or clearly ensure that both files are identical.

Recommended canonical location:

```text
model_assets/teacher_artifacts/teacher_pca_tau.pkl
```

All scripts and README commands should refer to the same location.

## Clustering Artifacts

The `clusters/` folder contains two serialized preprocessing artifacts.

### `scaler_case_profile.joblib`

Feature scaler fitted on case-profile variables before clustering.

### `kmeans_case_profile.joblib`

Trained K-means model used to assign case-profile clusters.

These files should only be included when clustering features are used by the released training or inference pipeline.

Load them with:

```python
import joblib

scaler = joblib.load(
    "model_assets/clusters/scaler_case_profile.joblib"
)

kmeans = joblib.load(
    "model_assets/clusters/kmeans_case_profile.joblib"
)
```

If the published scripts do not use these artifacts, remove the `clusters/` folder from the release or explain separately where it is used.

## Required Files for Inference

The minimum set normally required for full teacher–student inference is:

```text
xgb_next_delay_tau.joblib
teacher_artifacts/teacher_model_best.pt
teacher_artifacts/teacher_vocab.json
teacher_artifacts/teacher_config.json
teacher_artifacts/teacher_model_config.json
teacher_artifacts/teacher_num_cols.json
teacher_artifacts/num_feature_norm_stats.npz
teacher_artifacts/teacher_pca_tau.pkl
```

The following files are primarily for reproducibility and diagnostics:

```text
teacher_train_keys.parquet
teacher_val_keys.parquet
teacher_train_log.jsonl
summary.json
```

The clustering files are required only if clustering-derived features are part of the released pipeline.

## Compatibility

These artifacts must be used with:

- the same feature definitions;
- the same categorical vocabulary;
- the same numerical-column order;
- the same PCA transformation;
- compatible versions of PyTorch, XGBoost, scikit-learn, NumPy, pandas, and joblib.

Exact package versions should be listed in the repository-level `requirements.txt`.

## Security and Portability

Files serialized with `joblib`, `pickle`, or PyTorch should only be loaded from trusted sources.

Some serialized artifacts may depend on the package versions used when they were created. When necessary, retrain the models using the released code and anonymized data.

## Reproducing the Models

The teacher model can be retrained with:

```bash
python code/train_teacher_anonymized.py
```

The student model and PCA artifact can be generated with:

```bash
python code/main_train_anonymized.py
```

Update these paths when the code folder has a different name.

## License and Use

The model artifacts are released only under the license and institutional conditions stated in the repository.

They must not be used to reconstruct confidential industrial information or to infer the identity of systems, locations, vehicles, materials, employees, or facilities.

