# Reproduction scope and limitations

| Component | Evidence / boundary |
|---|---|
| Operational feature reconstruction | All 6,299,184 values across 331,536 fixed TRAIN queries matched the reference exactly. Raw industrial extraction and prospective scheduling are not reproduced. |
| Frozen teacher export | All 82,786 logits and 15,894,912 embedding values exactly matched in the validated environment; output bytes matched. Different GPU or Parquet writer may break the strict export pin. |
| Primary student matrices | All nine fitting/calibration/threshold-selection matrix files matched reference bytes. |
| Tabular and neural fitting recipes | Input compatibility and synthetic fitting verified; exact archived weights after fresh fitting are not established. |
| PCA | All fitted parameters matched exactly; five of 2,167,248 transformed values exceeded 1e-6 (max 1.90735e-6). Later serialization check stopped; frozen asset remains authoritative. |
| Primary student fresh refit | Both original and pseudonymized source refits agreed with each other, not with archived trees; historical cause unresolved. Do not replace archived model. |
| Frozen historical replay | All nine models matched reported metrics within 1e-6; eight native-order probability vectors exact, sequence LSTM tiny batch-order differences with identical alert decisions. No original private case order is distributed. |
| Retrospective statistics | 1,407 point estimates and 252 interval values reproduced exactly on original saved predictions and original case order. Original score rows and original ordering are not distributed. Direct pseudonymized replay-to-statistics adapter remains outstanding. |
| Supplementary analysis | Archived aggregate tables and four rendered plots regenerated; not all underlying development, robustness, runtime or attribution experiments can be rerun from the included processed inputs. |
| Precision-recall and historical description | All nine APs and four pooled category counts checked on original historical inputs. Original graphical render of PR and boxplot not established. |

The package contains fixed thresholds and retrospective first-delay risk-set logic. It does not establish live receipt-time availability, historical plan version provenance, deployment efficacy, causal effects, or independent holdout generalization. Rows with unknown subsequent completion are not verified outcome negatives. Source-level extraction and private linkage/order information are excluded. Reproducibility is conditional on access to the matching data and models; pseudonymization retains timing and structure that can support linkage, and trained parameters may retain information. Existing confidentiality approvals are relied upon within their original scope; coverage of newly added artifacts is not independently verified. Inventory and static pattern screening are not confidentiality certification.
