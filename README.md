# AI-CUP-ESG-2026Public
NLP-based ESG Promise Verification using MacBERT and Transformer Models

# AI CUP 2026 VeriPromiseESG

## Competition Result

Best Public Score

0.6079660

Best Submission

Sub_v28_1.csv

Public Ranking

50 / 141

---

## Project Overview

This repository contains the source code, experiment records, and submission generation scripts used in the AI CUP 2026 VeriPromiseESG competition.

The task is to predict four ESG-related fields:

- promise_status
- verification_timeline
- evidence_status
- evidence_quality

from ESG commitment statements.

---

## Environment

### Hardware

- NVIDIA Tesla T4
- Google Colab
- Kaggle Notebook

### Software

- Python 3.10+
- PyTorch
- Transformers
- Pandas
- NumPy
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Official datasets are provided by the competition organizer and are not included in this repository.

Expected directory structure:

```text
data/

├── vpesg_4k_train_1000.json
├── vpesg4k_val_1000.json
└── vpesg4k_test_2000.json
```

---

## Repository Structure

```text
AICUP-2026-VeriPromiseESG/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── v26_1_config.json
│   ├── v26_3_config.json
│   └── v28_1_config.json
├── scripts/
│   ├── train_v26_1_roberta_large_combined_384len_2seed.py
│   ├── train_v26_3_roberta_large_add_seed3407_3seed.py
│   ├── train_v27_1_macbert_large_combined_384len_2seed.py
│   └── make_v28_1_public_field_hybrid.py
├── summaries/
│   ├── public_leaderboard_history.csv
│   ├── 26-1SumForGpt.json
│   ├── 26-3SumForGpt.json
│   ├── 27-1SumForGpt.json
│   └── 28-1Summary.json
├── submissions/
│   ├── Sub_v26_1.csv
│   ├── Sub_v26_3.csv
│   └── Sub_v28_1.csv
└── docs/
    ├── experiment_log.md
    └── how_to_reproduce.md
```

---

## Main Experiments

### Sub_v26_1.csv

Public Score:

```text
0.605779
```

Characteristics:

- RoBERTa-based model
- Multi-seed training
- Automatic logic repair

---

### Sub_v26_3.csv

Public Score:

```text
0.605966
```

Characteristics:

- Improved ensemble strategy
- Better performance on:
  - promise_status
  - verification_timeline
  - evidence_status

---

### Sub_v28_1.csv

Public Score:

```text
0.607966
```

This is the final best-performing public submission.

Generation strategy:

Base submission:

```text
Sub_v26_3.csv
```

Use predictions from:

```text
Sub_v26_1.csv
```

for:

```text
evidence_quality
```

when:

```text
evidence_status == Yes
```

Automatic consistency repair:

```text
promise_status == No
    -> verification_timeline = N/A
    -> evidence_status = N/A
    -> evidence_quality = N/A

evidence_status != Yes
    -> evidence_quality = N/A
```

Generation script:

```bash
python Scripts/make_v28_1_public_field_hybrid.py
```

Output:

```text
Sub_v28_1.csv
```

---

## Reproducing v28

Place:

```text
Sub_v26_1.csv
Sub_v26_3.csv
```

inside:

```text
submissions/
```

Run:

```bash
python Scripts/make_v28_1_public_field_hybrid.py
```

Generated file:

```text
Sub_v28_1.csv
```

---

## Best Public Scores

| Submission | Public Score |
|------------|-------------:|
| Sub_v26_1.csv | 0.605779 |
| Sub_v26_3.csv | 0.605966 |
| Sub_v28_1.csv | 0.607966 |

Best Public Submission:

```text
Sub_v28_1.csv
```

---

## Notes

Model checkpoints (.pt files), competition datasets, and large output files are excluded from this repository due to size limitations.

Only source code, summaries, and final submission files are provided.
