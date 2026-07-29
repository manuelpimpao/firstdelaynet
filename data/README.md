# Data

This folder contains the anonymized datasets used for training, validation, and testing of the FirstDelayNet teacher–student pipeline.

## Files

| File | Description |
|---|---|
| `raw_train_anonymized.parquet` | Raw anonymized event history for the training cases. Used to reconstruct leakage-safe event prefixes and compute teacher-derived features at observation time \(\tau\). |
| `raw_val_anonymized.parquet` | Raw anonymized event history for the validation cases. Used for teacher-feature generation and validation of the student model. |
| `raw_test_anonymized.parquet` | Raw anonymized event history for the held-out test cases. Used only for final evaluation. |
| `tau_train_anonymized.parquet` | Anonymized \(\tau\)-level training table. Each row represents one observation time within an ongoing checkpoint interval and contains the hazard target and operational features available at that time. |
| `tau_val_anonymized.parquet` | Anonymized \(\tau\)-level validation table. Used for model selection, threshold calibration, and validation metrics. |
| `tau_test_anonymized.parquet` | Anonymized \(\tau\)-level test table. Used for the final held-out evaluation reported in the accompanying study. |

## Dataset Roles

The six files form two aligned dataset groups.

### Raw event-log files

The `raw_*` files contain the anonymized chronological event histories for the corresponding data split. They are used by the teacher model and by the distillation pipeline to construct event prefixes containing only information available up to the respective observation time.

### Tau-level files

The `tau_*` files contain the decision-point representation used by the XGBoost student model. Each row corresponds to an observation time \(\tau\) and is aligned with the event history of the same case in the matching `raw_*` file.

The pairing is:

```text
raw_train_anonymized.parquet  <->  tau_train_anonymized.parquet
raw_val_anonymized.parquet    <->  tau_val_anonymized.parquet
raw_test_anonymized.parquet   <->  tau_test_anonymized.parquet
```

## Data Splits

The datasets are separated into:

- **Training split:** used to fit the teacher model, PCA transformation, and XGBoost student.
- **Validation split:** used for model selection, early stopping, threshold calibration, and parameter validation.
- **Test split:** reserved for final evaluation.

The splits should not be recombined before model development or evaluation.

## File Format

All datasets are stored in Apache Parquet format.

Parquet preserves column types and is efficient for large tabular datasets. The files can be loaded with pandas:

```python
import pandas as pd

df = pd.read_parquet("data/anonymized/tau_train_anonymized.parquet")
```

The `pyarrow` package is recommended:

```bash
pip install pandas pyarrow
```

## Expected Use

The default pipeline uses the files as follows:

```text
train_teacher_anonymized.py
    -> train_events / validation events, where applicable

main_train_anonymized.py
    -> raw_train_anonymized.parquet
    -> raw_val_anonymized.parquet
    -> tau_train_anonymized.parquet
    -> tau_val_anonymized.parquet

test_anonymized.py
    -> raw_test_anonymized.parquet
    -> tau_test_anonymized.parquet
```

Depending on the final repository version, the teacher-training files may be generated from the raw event-log splits or provided separately.

## Anonymization

The released datasets are anonymized versions of the original industrial event data.

The anonymization process should ensure that no directly identifying or confidential information is publicly exposed. Depending on the original column, the applied protection may include:

- hashing or replacing case and event identifiers;
- anonymizing material, vehicle, station, source, destination, and subsystem identifiers;
- removing direct personal identifiers;
- removing credentials, internal URLs, file paths, and system secrets;
- suppressing or grouping rare categorical values;
- transforming timestamps while preserving relative temporal order and durations;
- removing columns that are not required for reproducibility;
- retaining only features necessary for training and evaluating the published method.

The released data must not be used to infer the identity of employees, systems, locations, vehicles, materials, or industrial facilities.

## Temporal Integrity and Leakage Prevention

The datasets are designed for leakage-safe predictive process monitoring.

For every observation time \(\tau\):

- only events with timestamps less than or equal to \(\tau\) may be used as model input;
- future event outcomes are used only for offline label construction;
- post-first-delay observations are excluded where required by the hazard formulation;
- the PCA transformation must be fitted only on the training split;
- validation and test data must not influence model fitting.

## Important Columns

The exact set of columns may differ between the raw and tau-level tables. Typical groups include:

### Raw event-log data

- anonymized case identifier;
- event timestamp;
- event activity or category;
- subsystem or event-source information;
- transport-related indicators;
- numerical timing and congestion features;
- prefix index;
- masked teacher target information.

### Tau-level data

- anonymized case identifier;
- observation timestamp `tau_ts`;
- target `y_next_delay`;
- interval-level operational features;
- congestion and timing features;
- first-delay or decision-point indicators;
- teacher-derived features after distillation.

To inspect the exact schema:

```python
import pandas as pd

for path in [
    "raw_train_anonymized.parquet",
    "raw_val_anonymized.parquet",
    "raw_test_anonymized.parquet",
    "tau_train_anonymized.parquet",
    "tau_val_anonymized.parquet",
    "tau_test_anonymized.parquet",
]:
    df = pd.read_parquet(path)
    print(path)
    print(df.dtypes)
    print()
```

## Limitations

These datasets are anonymized research artifacts and may not preserve every operational attribute of the original system.

Possible limitations include:

- generalized or replaced categorical values;
- removed location or resource semantics;
- transformed timestamps;
- suppression of rare events;
- omitted confidential attributes;
- limited interpretability of anonymized identifiers;
- inability to reconstruct the original industrial process in full detail.

The datasets are intended for reproducing the published experiments, not for operational deployment.

## License and Use Restrictions

The applicable data license or terms of use should be added only after approval by Kühne Logistics University and Volkswagen.

Until an explicit license is included, redistribution, commercial use, or use beyond research reproduction is not automatically permitted.

## Contact

**Manuel Pimpão**  
Kühne Logistics University  
Hamburg, Germany
