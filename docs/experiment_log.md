# Experiment Log

## Project Goal

Participate in the AI CUP 2026 VeriPromiseESG competition.

Predict the following labels from ESG commitment statements:

- `promise_status`
- `verification_timeline`
- `evidence_status`
- `evidence_quality`

## Experiment Timeline

### Early Baseline

| Version | Public Score |
|---|---:|
| v15 | 0.5761173 |
| v16_1 | 0.5791360 |
| v17c | 0.5785303 |

Main work:

- Baseline model construction
- Initial threshold tuning
- Validation pipeline verification

### Quality Optimization Stage

| Version | Public Score |
|---|---:|
| v19_2 | 0.5857160 |
| v19_3 | 0.5843682 |

Main work:

- Evidence quality optimization
- Rule adjustment
- Prediction consistency improvement

### Metadata and Hybrid Stage

| Version | Public Score |
|---|---:|
| v20_1 | 0.5879202 |
| v20_4 | 0.5869557 |
| v20_5 | 0.5862315 |

### Strong Model Stage

| Version | Public Score |
|---|---:|
| Sub_v22_1.csv | 0.5896736 |
| Sub_v22_2B.csv | 0.5955661 |

### Consensus and Timeline Experiments

| Version | Public Score |
|---|---:|
| Sub_v23_3C_consensus.csv | 0.5946937 |
| Sub_v24_1B_timeline.csv | 0.5950099 |
| Sub_v25_2E_ultra_safe.csv | 0.5945326 |

### RoBERTa-Large Ensemble Stage

| Version | Public Score |
|---|---:|
| Sub_v26_1.csv | 0.6057791 |
| Sub_v26_2C_large_base_quality_q035.csv | 0.6038160 |
| Sub_v26_3.csv | 0.6059663 |

### Final Hybrid Stage

| Version | Public Score |
|---|---:|
| Sub_v28_1.csv | 0.6079660 |

Best public rank: 50 / 141
