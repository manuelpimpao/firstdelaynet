# End-to-end frozen retrospective evaluation

Run from the package root, with the dependencies in `code/requirements.txt` installed in a trusted environment:

```bash
python -I code/evaluate.py --submission-root . --out-dir ../reproduction_run --device cuda --acknowledge-retrospective
```

Use a new output directory outside the package. The script verifies the original pinned asset inventory and the documentation, joins predicted query keys to independently checked historical outcomes, and writes aggregate metrics and paired case-cluster bootstrap summaries. It does not write row-level predictions or fit models. It is not a new independent holdout. Use only trusted serialized assets. The original model bundle includes two duplicate data files for compatibility.

The V4.502 local integration run on all 60 prior pinned files reproduced the core metrics of all nine frozen models within the existing absolute 1e-6 metric tolerance; all confusion, first-alarm counts and fixed thresholds matched. At the stricter 1e-10 aggregate comparison, 14 of 1,407 values differed (all sequence LSTM; maximum absolute difference 3.1143e-7). The original row-level native-order sequence probability comparison was not exact. All 252 bootstrap comparison values differed under native pseudonym order (maximum absolute difference 0.0046677; 200 valid draws). The original case ordering is deliberately not distributed; the private archived-score replay reproduced all 1,407 point estimates and 252 bootstrap values exactly. Archived aggregate tables remain the reported numerical reference. Exact fresh refitting is not established. These checks do not certify anonymity or unrestricted redistribution.

The evaluation adapter has the same computational functions as the tested V4.502 gate; its packaging checks additionally account for this file and the installed adapter. The adapter's own integrity is checked by the external allowlist audit, not by self-authentication. Do not recursively publish unlisted files from the working directory.
