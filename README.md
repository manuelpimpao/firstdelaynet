# First-delay prediction — reproducibility package

This package supports the retrospective first-delay study using processed, pseudonymized event data and frozen prediction models. It is not a live deployment or a fresh independent holdout. The current package is a candidate pending final distribution-scope and inventory review.

## Directory layout

- `data/`: ten processed Parquet inputs (see `docs/DATA_DICTIONARY.md`).
- `code/`: model recipes, feature reconstruction, frozen replay, archived statistical/figure regeneration, dependencies and file inventory.
- `models/`: frozen model artifacts, preprocessing contracts and calibrators; includes the standalone replay script and two historical data files duplicated from `data/` for compatibility. The copies must have identical fingerprints.
- `docs/`: data dictionary, feature definitions and reproduction scope.

Use a trusted environment and trusted, verified assets only; serialized joblib objects may execute code when loaded. Do not upload locally generated row-level scores or outputs, original case ordering, raw extraction sources or pseudonymization secrets. Keep all generated outputs in a separate directory outside the package.

## Running the code

From `code/`, follow `README.md` for exact command signatures. Substitute `../data` for `DATA`, `../models` for `BUNDLE`, and a new output directory **outside the submission root** for `../results`. Install dependencies from `requirements.txt` in an isolated environment; the historical neural runtime used torch 2.7.0+cu128. Exact versions for pyarrow, joblib and matplotlib were not established. No network access is required by the reproduction commands after dependencies and verified assets are available.

Start with aggregate-only commands if you want to reproduce published tables and plots without model execution. For the historical frozen inference, use `python replay.py --bundle ../models --device cuda --metrics-out ../../results/frozen_metrics.json` after creating the external results directory. Output must not already exist. The frozen models and fixed thresholds are the authoritative evaluation assets. Fresh fitting is optional and not claimed to be bitwise equivalent to historical training. See `docs/REPRODUCTION_SCOPE.md` for verified and unverified steps.

The data contain shifted timestamps and consistent pseudonyms, not fully anonymous records. Relative timings and structural patterns remain; linking to earlier exports and model-based disclosure remain possible. No claim of irreversible anonymization, new approval or permission for arbitrary redistribution is made here. No license is assumed.
