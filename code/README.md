# First-delay prediction: processed-input reproducibility candidate

**Working candidate, not a release authorization.** The directory contains reviewed portable source modules and curated aggregate evidence. No row-level data or trained model binaries are included. Access to the separately supplied, matching processed dataset and frozen asset bundle is required for model reproduction. Use only trusted artifacts; joblib loads executable serialized Python objects. This is a retrospective historical reanalysis, not a new independent holdout or a deployment demonstration.

## Aggregate-only reproduction (no data or model assets)

Run from this directory, with a new output directory outside this directory for each command:

```sh
python retrospective_tables.py --aggregates retrospective_aggregates.json --out-dir ../results/retrospective
python development_tables.py --aggregates development_aggregates.json --out-dir ../results/development
python cohort_tables.py --aggregates cohort_aggregates.json --out-dir ../results/cohorts
python manuscript_figures.py --retrospective-aggregates retrospective_aggregates.json --figure-aggregates figure_aggregates.json --out-dir ../results/figures
```

The development, robustness, runtime, SHAP, tolerance, subgroup and population aggregates are archived evidence, not fresh execution of the original experiments. The four aggregate figures were reproduced at 72 dpi. The two schematic figures have manuscript LaTeX source; the original pooled boxplot is archived, with its descriptive counts independently checked. The PR recipe accepts a saved-score file with a known SHA-256; exact AP replay was verified on the original historical score export, not on newly generated pseudonym-order predictions.

## Processed-input reproduction

Use `DATA` and `BUNDLE` below as placeholders for paths to the matched candidate directories. These are **not included**. The manifest inventories ten processed Parquet inputs and the 26-file frozen inference bundle. It is an inventory for the candidate, not proof of permission to distribute it. Output directories must be new. Do not copy secret mappings, private order manifests, raw extraction files, prediction rows, private audit reports or arbitrary project folders into a release.

```sh
python observed_features.py --data-dir DATA --out-dir ../results/observed
python teacher_oos.py --data-dir DATA --bundle BUNDLE --out-dir ../results/teacher --device cuda
python primary_student.py prepare --data-dir DATA --teacher-export ../results/teacher/teacher_oos.parquet --pca BUNDLE/pca.joblib --out-dir ../results/student
python tabular_baselines.py validate --matrices-dir ../results/student/matrices --out-dir ../results/tabular
python neural_baselines.py validate --data-dir DATA --bundle BUNDLE --matrices-dir ../results/student/matrices --out-dir ../results/neural
python sequence_training.py validate --data-dir DATA --bundle BUNDLE --out-dir ../results/sequence
python replay.py --bundle BUNDLE --device cuda --metrics-out ../results/frozen_metrics.json
```

The teacher export and student preparation use strict byte pins, so different Parquet serialization or GPU numerics can block the chain even when values are close. Frozen historical inference was validated for all nine models at the metric level; eight had exact native-order probabilities, while the sequence LSTM showed small batch-order-dependent probability differences with identical alert decisions. Do not assert exact probabilities or original bootstrap intervals for a new pseudonym-order run. The score-only statistical script has been validated using the original saved scores and historical events; an end-to-end public score-schema adapter from `replay.py` output to that script is still outstanding. Do not publish original order or score exports.

## Fresh fitting (optional, not exact historical model reproduction)

The `fit` commands in `primary_student.py`, `tabular_baselines.py`, `fit_pca.py`, `sequence_training.py` and the `reproduce` command in `neural_baselines.py` expose bounded fitting recipes. They are separate from frozen replay; run only into fresh output directories. Real input compatibility and synthetic training tests were performed; full fresh-model equivalence is not claimed. PCA parameters were reproduced exactly, but five transformed values exceeded the strict 1e-6 tolerance. A fresh primary student fit differed from the archived model on both original and pseudonymized inputs; the original-source control reproduced the fresh-fit discrepancy. Retain the frozen assets for manuscript reproduction. No retuning or threshold search is needed.

## Scope and dependencies

The supported starting point is the processed event log, prefix table, fixed queries and role manifests. Raw industrial extraction, upstream deduplication and source-clock construction, secret identifier mapping and some row-level development computations are restricted or not reproduced by this candidate. Fixed query cohorts do not establish prospective query scheduling, historical plan versions or receipt-time availability. Cross-release event and case linkage and learned-parameter disclosure risks remain. A separately documented release-scope decision is needed before distribution; prior approval scope is not independently verified here. See `requirements.txt` for known historical versions and dependencies whose exact original version is not yet verified. No license or repository URL is assumed.
