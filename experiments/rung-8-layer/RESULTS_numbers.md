| perturbation | A policy | B filter | C adapt | D oracle |
|---|---|---|---|---|
| healthy | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| payload_add | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| spring_soft | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| spring_stiff | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| damping_high | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| motor_weak | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| gravity_tilt | 0% (0–11) | 0% (0–11) | 0% (0–11) | 0% (0–11) |
| sensor_delay | 0% (0–11) | 0% (0–11) | 0% (0–11) | 100% (89–100) |

| perturbation | falls A / B / C / D | B reject rate | C steps adapted | C mean scale |
|---|---|---|---|---|
| healthy | 0 / 30 / 0 / 0 | 35.9% | 80.0% | 1.45 |
| payload_add | 0 / 30 / 0 / 0 | 29.2% | 34.6% | 1.62 |
| spring_soft | 0 / 30 / 0 / 0 | 32.6% | 67.4% | 1.56 |
| spring_stiff | 0 / 30 / 0 / 0 | 44.1% | 78.0% | 1.28 |
| damping_high | 0 / 30 / 0 / 0 | 37.6% | 70.2% | 1.48 |
| motor_weak | 0 / 30 / 0 / 0 | 31.9% | 47.8% | 1.61 |
| gravity_tilt | 30 / 30 / 30 / 30 | 89.9% | 5.8% | 1.21 |
| sensor_delay | 11 / 30 / 28 / 0 | 41.0% | 44.2% | 1.14 |

- Claim 1_policy_degrades_>=30pp: value 0.0, **FAIL**
- Claim 2_oracle_degrades_<=10pp: value 0.0, **PASS**
- Claim 3_selfmodel_beats_policy_>=15pp: value 0.0, **FAIL**
- Claim 4_adapt_beats_filter_somewhere: value ['payload_add', 'spring_soft', 'spring_stiff', 'damping_high', 'motor_weak'], **PASS**
