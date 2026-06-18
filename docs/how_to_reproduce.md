# How To Reproduce

## Environment

Python 3.10+ is recommended.

Required packages:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the official competition datasets under `data/`:

```text
data/
├── vpesg_4k_train_1000.json
├── vpesg4k_val_1000.json
└── vpesg4k_test_2000.json
```

## Training

Example:

```bash
python scripts/train_v27_1_macbert_large_combined_384len_2seed.py
```

## Generate Final Submission

Required files:

```text
submissions/
├── Sub_v26_1.csv
└── Sub_v26_3.csv
```

Run:

```bash
python scripts/make_v28_1_public_field_hybrid.py
```

Expected output:

```text
Sub_v28_1.csv
```

## Final Result

Best public score: 0.6079660

Best submission: `Sub_v28_1.csv`

Public rank: 50 / 141
