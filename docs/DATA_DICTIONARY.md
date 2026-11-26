# Processed dataset dictionary

The ten inputs below are the fixed processed reproduction population, not the raw industrial source. Counts denote rows and retained columns, not independent prediction examples. `case_seg`, `k_idx`, `tau_ts` form the query key; in the historical query export repeated keys must be reconciled by the frozen replay's duplicate rule. Event-log joins use a segmented case and event timestamp where the replay enforces uniqueness. Role manifests are case-level. Prefix pointers refer to the delivered teacher event table ordering, not the original export. Do not reorder that table without remapping the pointers.

## Files and columns

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `revised_train_events.parquet` | 874,844 | 10 | Minimized TRAIN event log |
| `historical_test_events.parquet` | 258,929 | 24 | Historical evaluation event log |
| `teacher_prefix_events.parquet` | 874,844 | 26 | Observed teacher event predictors |
| `train_role_manifest.parquet` | 82,930 | 2 | Case-level outer TRAIN roles |
| `teacher_inner_manifest.parquet` | 21,461 | 2 | Case-level teacher fit/early-stop roles |
| `train_query_index.parquet` | 283,842 | 9 | Observation keys and prefix pointers |
| `train_student_candidates.parquet` | 331,536 | 22 | 19 operational predictors and query keys |
| `train_query_labels.parquet` | 331,536 | 7 | Retrospective labels and weights; not inputs |
| `train_retrospective_metadata.parquet` | 331,536 | 6 | Retrospective times; not inputs |
| `historical_test_queries.parquet` | 531,363 | 10 | Historical observation queries and reconstruction fields |

### `revised_train_events.parquet`

`case_seg`, `case:concept:name`, `event_id`, `time:timestamp`, `planned:timestamp`, `is_delayed`, `is_delayed_dyn`, `category`, `concept:name`, `event_end`.

### `historical_test_events.parquet`

`case:concept:name`, `event_id`, `case_seg`, `time:timestamp`, `category`, `is_delayed`, `is_delayed_dyn`, `planned:timestamp`, `event_end`, `concept:name`, `inter_event_time`, `hour_of_day`, `weekday_0`, `weekday_1`, `weekday_2`, `weekday_3`, `weekday_4`, `weekday_5`, `weekday_6`, `concept:name_repeat_flag`, `prefix_length`, `schichtwechsel`, `time_since_start`, `unique_stations_so_far`.

### `teacher_prefix_events.parquet`

`case_seg`, `time:timestamp`, `concept:name`, `inter_event_time`, `roll_ie_mean`, `roll_ie_min`, `roll_ie_max`, `hour_of_day`, `weekday_0`, `weekday_1`, `weekday_2`, `weekday_3`, `weekday_4`, `weekday_5`, `weekday_6`, `case_length_prefix`, `case_mean_iet_prefix`, `concept:name_repeat_flag`, `prefix_length`, `schichtwechsel`, `time_since_start`, `unique_stations_so_far`, `service_rate_event_end_1h`, `in_flight_event_end_1h`, `starts_5m_event_end`, `starts_15m_event_end`.

### `train_role_manifest.parquet`

`case_seg`, `part`.

### `teacher_inner_manifest.parquet`

`case_seg`, `teacher_inner_role`.

### `train_query_index.parquet`

`case_seg`, `k_idx`, `tau_ts`, `part`, `last_event_global_index`, `observed_event_count`, `last_event_timestamp`, `query_seconds_since_last_event`, `query_seconds_since_observed_transport_t0`.

### `train_student_candidates.parquet`

`case_seg`, `k_idx`, `tau_ts`, `starts_5m_event_end`, `burstiness_cv_5m_event_end`, `starts_15m_event_end`, `burstiness_cv_15m_event_end`, `starts_60m_event_end`, `burstiness_cv_60m_event_end`, `new_cases_1h`, `active_cases_1h`, `ewma_if_event_end`, `momentum_if_event_end`, `observed_transport_5m_event_end`, `observed_transport_15m_event_end`, `observed_transport_60m_event_end`, `observed_transport_global_5m`, `observed_transport_global_15m`, `observed_transport_global_60m`, `observed_transport_count_case_at_tau`, `seconds_since_last_observed_transport`, `seconds_since_observed_transport_t0`.

### `train_query_labels.parquet`

`case_seg`, `k_idx`, `tau_ts`, `y_next_delay`, `weight_full_clock`, `weight_predeadline`, `is_predeadline`.

### `train_retrospective_metadata.parquet`

`case_seg`, `k_idx`, `tau_ts`, `t0`, `t_next`, `next_static_plan`.

### `historical_test_queries.parquet`

`case_seg`, `k_idx`, `tau_ts`, `t0`, `t_next`, `y_next_delay`, `has_post`, `events_since_t0`, `station`, `time_since_t0_s`.

## Shared fields and usage restrictions

| Field | Meaning / use |
|---|---|
| `case_seg` | Pseudonymous segmented case key. |
| `case:concept:name` | Pseudonymous parent case identifier. |
| `event_id` | Pseudonymous event identifier. |
| `time:timestamp` | Shifted recorded event timestamp (UTC); not authenticated receipt time. |
| `planned:timestamp`, `next_static_plan` | Archived plan timestamps, not verified historical plan versions. Retrospective outcome/evaluation fields, not model inputs. |
| `category` | Event category; transport remains literal; other category values are pseudonymized. |
| `concept:name` | Pseudonymized activity token. |
| `event_end`, `station` | Consistent pseudonymized station/location categories. |
| `is_delayed`, `is_delayed_dyn` | Historical static/dynamic delay indicators; outcome fields, excluded from observed features. |
| `k_idx` | Transport anchor index within segmented case (zero-based in reconstruction). |
| `tau_ts` | Fixed observation timestamp. |
| `t0`, `t_next` | Observed anchor completion and subsequent completion respectively; `t_next` is outcome information. |
| `y_next_delay` | Historical binary next-delay label, not a predictor. |
| `part`, `teacher_inner_role` | Outer TRAIN and inner teacher partition roles. |
| `last_event_global_index`, `last_event_timestamp`, `observed_event_count` | Prefix endpoint pointer, timestamp and number of observed events. |
| `query_seconds_since_last_event`, `query_seconds_since_observed_transport_t0` | Query-time age in seconds. |
| `weight_full_clock`, `weight_predeadline` | Supervision/evaluation weights for the respective observation population. |
| `is_predeadline` | Retrospective archived-deadline eligibility flag; not a predictor. |
| `has_post` | Historical post-observation availability indicator; not a predictor. |
| `events_since_t0`, `time_since_t0_s` | Legacy recorded observation-history summaries; use only as specified by the replay adapter. |

The 19 operational channels are listed in `FEATURE_DICTIONARY.md`. The sequence channel order is defined in `code/replay.py` (`EVENT_FEATURES`, `QFEATURES`) and the two frozen preprocessing contracts, rather than inferred from arbitrary column order. The teacher sequence uses 23 numeric event channels and three query channels in the frozen input contract. The input contracts specify activity vocabulary and normalization; do not fit these on the historical evaluation population. Retained legacy numeric fields are not automatically eligible predictors.

## Pseudonymization and boundaries

A keyed one-to-one pseudonym mapping preserves case and event boundaries within separate namespaces. Station and activity vocabularies are mapped consistently. A constant whole-week UTC shift preserves time differences and weekday/hour structure. Keys, inverse mappings, exact time offset and original execution-order manifest are not distributed. This is pseudonymization, not guaranteed unlinkability; relative time and sequence fingerprints can enable cross-release linkage. The earlier export used a rare-value treatment that merged case identifiers; the current candidate does not merge rare identities. Historical evaluation is a previously exposed retrospective cohort. No new independent test is claimed.
