# How to Reproduce

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

```bash
python scripts/train_v26_1_roberta_large_combined_384len_2seed.py
python scripts/train_v26_3_roberta_large_add_seed3407_3seed.py
python scripts/train_v27_1_macbert_large_combined_384len_2seed.py
python scripts/make_v28_1_public_field_hybrid.py
```

Update this document with dataset paths, preprocessing steps, checkpoints, and final submission-generation commands.
