# AI CUP 2026 VeriPromiseESG

## Competition Result

Best Public Score: **0.6079660**

Best Submission: **Sub_v28_1.csv**

Public Ranking: **50 / 141**

---

## Project Overview

This repository contains the source code, experiment records, and submission generation scripts used for the AI CUP 2026 VeriPromiseESG competition.

The task is to predict four ESG-related labels from ESG commitment statements:

- `promise_status`
- `verification_timeline`
- `evidence_status`
- `evidence_quality`

The main goal of this project was not only to train a single model, but also to iteratively improve the submission through model selection, multi-seed training, ensemble strategies, and automatic logic-based post-processing.

---

## Repository Structure

```text
AICUP-2026-VeriPromiseESG/

├── README.md
├── requirements.txt
├── .gitignore

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
    └── experiment_log.md
```

Large model checkpoint files, official datasets, and full output folders are not included in this repository because of file size limitations.

---

## Environment

The project was mainly developed and tested on Google Colab and Kaggle Notebook.

Recommended environment:

- Python 3.10+
- CUDA GPU, preferably NVIDIA Tesla T4 or better
- PyTorch
- HuggingFace Transformers
- Pandas
- NumPy
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The official dataset is provided by the AI CUP 2026 VeriPromiseESG competition organizer and is not included in this repository.

Expected local structure:

```text
data/
├── vpesg_4k_train_1000.json
├── vpesg4k_val_1000.json
└── vpesg4k_test_2000.json
```

The submission generation scripts assume that the prediction files are placed under `submissions/`.

---

## Main Experiments

### Sub_v26_1.csv

Public Score: **0.6057791**

Main characteristics:

- Chinese RoBERTa-large model
- Combined train + validation split
- Two-seed ensemble
- Automatic consistency repair

---

### Sub_v26_3.csv

Public Score: **0.6059663**

Main characteristics:

- Extension of v26_1
- Added one more large-model seed
- Improved performance on:
  - `promise_status`
  - `verification_timeline`
  - `evidence_status`

---

### Sub_v28_1.csv

Public Score: **0.6079660**

This is the best public submission in this project.

v28_1 is an automatic public-field hybrid generated from:

- `Sub_v26_3.csv`
- `Sub_v26_1.csv`

Hybrid rule:

```text
Use Sub_v26_3.csv as the base submission.

Keep from Sub_v26_3.csv:
- promise_status
- verification_timeline
- evidence_status

Use evidence_quality from Sub_v26_1.csv when it is logically compatible.
```

Automatic consistency repair:

```text
If promise_status == No:
    verification_timeline = N/A
    evidence_status = N/A
    evidence_quality = N/A

If evidence_status != Yes:
    evidence_quality = N/A
```

This process does not use manual labeling or human correction of individual test samples. The final prediction file is generated automatically by program rules.

---

## Reproducing the Final Submission

Place the following files under `submissions/`:

```text
submissions/Sub_v26_1.csv
submissions/Sub_v26_3.csv
```

Run:

```bash
python scripts/make_v28_1_public_field_hybrid.py
```

Expected output:

```text
submissions/Sub_v28_1.csv
```

---

## Public Leaderboard History

A full record of public submissions and scores is available at:

```text
summaries/public_leaderboard_history.csv
```

Key milestones:

| Version | Submission | Public Score | Note |
|---|---|---:|---|
| v15 | submission_v15_final.csv | 0.5761173 | Early baseline |
| v19 | submission_v19_2.csv | 0.5857160 | Quality-focused refinement |
| v22 | Sub_v22_2B.csv | 0.5955661 | Timeline-quality hybrid |
| v26 | Sub_v26_1.csv | 0.6057791 | RoBERTa-large breakthrough |
| v26 | Sub_v26_3.csv | 0.6059663 | Large 3-seed ensemble |
| v28 | Sub_v28_1.csv | 0.6079660 | Final hybrid submission |

---

## Best Public Scores

| Submission | Public Score | Rank |
|---|---:|---:|
| Sub_v26_1.csv | 0.6057791 | - |
| Sub_v26_3.csv | 0.6059663 | 53 / 141 |
| Sub_v28_1.csv | 0.6079660 | 50 / 141 |

Best public submission:

```text
Sub_v28_1.csv
```

---

## Notes for Reviewers and Teachers

This repository focuses on:

1. Model training scripts
2. Experiment summaries
3. Submission generation logic
4. Public leaderboard history
5. Reproducible final submission generation

Model checkpoint files are not committed to GitHub because each large-model checkpoint is about 1.3 GB. If full reproduction from checkpoints is required, the checkpoints should be downloaded separately and placed under the corresponding output directory.
