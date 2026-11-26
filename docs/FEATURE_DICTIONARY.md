# Operational and derived feature dictionary

All 19 operational features use recorded event history at or before the fixed observation time tau. The window convention is `(tau-W, tau]`; source timestamps are event times, not verified arrival times. Station windows refer to the transport anchor's observed station, not an inferred future destination. Counts refer to observed event rows, not orders forecast to arrive. Implementation: `code/observed_features.py` and `code/replay.py`.

| Feature | Definition | Unit |
|---|---|---|
| `starts_5m_event_end` | All observed event rows at anchor station in prior 5 minutes. Historical name retained. | count |
| `burstiness_cv_5m_event_end` | Sample standard deviation divided by mean of 5 one-minute station event-count bins; zero when mean is zero. | ratio |
| `starts_15m_event_end` | All observed event rows at anchor station in prior 15 minutes. Historical name retained. | count |
| `burstiness_cv_15m_event_end` | Sample standard deviation divided by mean of 15 one-minute station event-count bins; zero when mean is zero. | ratio |
| `starts_60m_event_end` | All observed event rows at anchor station in prior 60 minutes. Historical name retained. | count |
| `burstiness_cv_60m_event_end` | Sample standard deviation divided by mean of 60 one-minute station event-count bins; zero when mean is zero. | ratio |
| `new_cases_1h` | Cases whose first recorded event falls in preceding hour. | count |
| `active_cases_1h` | Distinct cases with at least one recorded event in preceding hour. | count |
| `ewma_if_event_end` | EWMA (alpha 0.3, adjust=False) of station's preceding-hour observed event counts at station event-clock ticks, carried forward to tau. | smoothed count |
| `momentum_if_event_end` | Difference between consecutive values of that station EWMA, carried forward to tau; first tick zero. | smoothed count difference |
| `observed_transport_5m_event_end` | Transport-category event count at anchor station over 5 minutes. | count |
| `observed_transport_15m_event_end` | Transport-category event count at anchor station over 15 minutes. | count |
| `observed_transport_60m_event_end` | Transport-category event count at anchor station over 60 minutes. | count |
| `observed_transport_global_5m` | Transport-category event count across stations over 5 minutes. | count |
| `observed_transport_global_15m` | Transport-category event count across stations over 15 minutes. | count |
| `observed_transport_global_60m` | Transport-category event count across stations over 60 minutes. | count |
| `observed_transport_count_case_at_tau` | Transport-category events in current case observed up to tau. | count |
| `seconds_since_last_observed_transport` | Time since latest observed transport in case; -1 if none. | seconds |
| `seconds_since_observed_transport_t0` | Time elapsed since observed transport anchor. | seconds |

## Teacher sequence and student-derived features

The frozen sequence contracts define exact scaling and feature order. `inter_event_time` and rolling gap summaries describe inter-event times; the reconstructed rolling mean/min/max use the latest three gaps including current event. `case_length_prefix` is observed event count including current event; `case_mean_iet_prefix` is cumulative mean of observed gaps, with initial gap zero. Time-context, prefix and activity fields are carried according to the frozen event-channel contract. `service_rate_event_end_1h` and `in_flight_event_end_1h` both use the observed station event count in the preceding hour in the audited reconstruction; the historical names do not denote measured future service or true work in process. `starts_5m_event_end` and `starts_15m_event_end` are corresponding observed counts at event time. Other retained sequence channel definitions and numerical transforms are governed by the frozen code and contracts; this glossary is not a replacement for them.

`teacher_logit` is the scalar output of the frozen binary teacher. `teacher_embed_000` through `teacher_embed_191` are its learned representation, and `teacher_pca_00` through `teacher_pca_47` are the 48 principal component projections. PCA is fitted on predeadline student-fit rows only (31,080). The primary student's 68 inputs are 19 operational channels + one logit + 48 PCA coordinates; soft supervision uses lambda 0.25 and temperature 2. Teacher training cases and student-fit cases are separate. Labels, weights and retrospective timestamps are stored separately from predictors. No claim that the learned components have direct physical interpretations is made.
