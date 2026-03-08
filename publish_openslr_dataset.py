"""
OpenSLR Sinhala Spelling Correction Dataset Publisher
======================================================
Publishes the OpenSLR spelling correction dataset to Hugging Face Hub.

- Train split : data/openslr-60000.csv  (~60 k rows)
- Eval  split : data/openslr-7000.csv   (~7 k rows)
- Columns     : dyslexic_sentence (noisy input), clean_sentence (ground truth)

Dataset repo  : SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference

Usage:
    python publish_openslr_dataset.py
    python publish_openslr_dataset.py --token hf_xxxx
"""

import argparse
import csv
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: Required libraries not found. Install them with:")
    print("    pip install datasets huggingface_hub")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID      = "SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference-60000"
TRAIN_CSV    = Path("data/openslr-60000.csv")
EVAL_CSV     = Path("data/openslr-7000.csv")

DATASET_CARD = """\
---
language:
- si
tags:
- spelling-correction
- sinhala
- asr
- openslr
- text2text-generation
license: cc-by-4.0
task_categories:
- translation
size_categories:
- 10K<n<100K
---

# OpenSLR Sinhala Spelling Correction – Prediction / Reference

This dataset contains Sinhala sentence pairs intended for training and evaluating
spelling-correction models on ASR output.

| Column              | Description                                              |
|---------------------|----------------------------------------------------------|
| `dyslexic_sentence` | Noisy / dyslexic sentence (model input – ASR hypothesis) |
| `clean_sentence`    | Clean / correct sentence (ground truth)                  |

## Splits

| Split | Source file          | Rows (approx.) |
|-------|----------------------|----------------|
| train | openslr-60000.csv    | ~60 000        |
| eval  | openslr-7000.csv     | ~7 000         |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference")
print(ds["train"][0])
# {'dyslexic_sentence': '...', 'clean_sentence': '...'}
```

## Citation

If you use this dataset please cite the original OpenSLR Sinhala corpus and the
SPEAK-PP spelling-corrector project.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """Read a CSV with `prediction` and `reference` columns, renaming them for HF."""
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pred = (row.get("prediction") or "").strip()
            ref  = (row.get("reference")  or "").strip()
            if pred and ref:
                rows.append({"dyslexic_sentence": pred, "clean_sentence": ref})
    return rows


def build_dataset_dict(train_path: Path, eval_path: Path) -> DatasetDict:
    print(f"Loading train split from  : {train_path}")
    train_rows = load_csv(train_path)
    print(f"  → {len(train_rows):,} rows")

    print(f"Loading eval  split from  : {eval_path}")
    eval_rows  = load_csv(eval_path)
    print(f"  → {len(eval_rows):,} rows")

    train_ds = Dataset.from_list(train_rows)
    eval_ds  = Dataset.from_list(eval_rows)

    return DatasetDict({"train": train_ds, "eval": eval_ds})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Publish OpenSLR dataset to Hugging Face")
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env variable / use `huggingface-cli login`)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public)",
    )
    args = parser.parse_args()

    # ── Authentication ──────────────────────────────────────────────────────
    if args.token:
        login(token=args.token)
    else:
        # Will use cached credentials from `huggingface-cli login`
        print("No token provided – using cached HF credentials.")

    api = HfApi()

    # ── Ensure the repo exists ──────────────────────────────────────────────
    print(f"\nEnsuring dataset repo exists: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    # ── Build DatasetDict ───────────────────────────────────────────────────
    dataset_dict = build_dataset_dict(TRAIN_CSV, EVAL_CSV)
    print(f"\nDataset overview:\n{dataset_dict}")

    # ── Push to Hub ─────────────────────────────────────────────────────────
    print(f"\nPushing to Hub → {REPO_ID} …")
    dataset_dict.push_to_hub(
        REPO_ID,
        commit_message="Add OpenSLR Sinhala spelling-correction dataset (train + eval)",
    )

    # ── Upload dataset card ─────────────────────────────────────────────────
    print("Uploading README / dataset card …")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"\nDone!  Dataset published at:")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
