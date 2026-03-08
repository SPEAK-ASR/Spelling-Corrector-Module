"""
Hugging Face Dataset Publisher - Dyslexic/Clean CSV
===================================================
Publishes prediction/reference CSV datasets to Hugging Face Hub as
dyslexic_sentence/clean_sentence pairs.

Default dataset:
    new_dataset/prediction_openslr-sinhala-asr-norm-noise-rem - prediction_openslr-sinhala-asr-norm-noise-rem.csv

Usage:
    python new_dataset/publish_prediction_dataset.py
    python new_dataset/publish_prediction_dataset.py --repo-name SPEAK-PP/prediction_openslr-sinhala-asr-norm-noise-rem
    python new_dataset/publish_prediction_dataset.py --private
"""

import argparse
import csv
import random
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: Required libraries not found. Install them with:")
    print("    pip install datasets huggingface_hub")
    raise SystemExit(1)


DEFAULT_INPUT_PATH = (
    "new_dataset/prediction_openslr-sinhala-asr-norm-noise-rem "
    "- prediction_openslr-sinhala-asr-norm-noise-rem.csv"
)
DEFAULT_REPO_NAME = "SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference"


def load_prediction_reference_csv(csv_path: str) -> list:
    """
    Load rows from CSV and map them to dyslexic/clean fields.

    Accepts these possible column aliases:
      - prediction: prediction, predicted, output, noisy_sentence, dyslexic_sentence
      - reference:  reference, target, label, clean_sentence, correct_sentence
    """
    data = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dyslexic_sentence = (
                row.get("prediction")
                or row.get("predicted")
                or row.get("output")
                or row.get("noisy_sentence")
                or row.get("dyslexic_sentence")
                or ""
            ).strip()

            clean_sentence = (
                row.get("reference")
                or row.get("target")
                or row.get("label")
                or row.get("clean_sentence")
                or row.get("correct_sentence")
                or ""
            ).strip()

            if dyslexic_sentence and clean_sentence:
                data.append({
                    "dyslexic_sentence": dyslexic_sentence,
                    "clean_sentence": clean_sentence,
                })

    return data


def split_dataset(data: list, seed: int = 42) -> tuple:
    """Split rows into 80% train, 10% eval, 10% test."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.8)
    eval_end = train_end + int(total * 0.1)

    train_rows = shuffled[:train_end]
    eval_rows = shuffled[train_end:eval_end]
    test_rows = shuffled[eval_end:]
    return train_rows, eval_rows, test_rows


def generate_dataset_card(
    repo_name: str,
    total_samples: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> str:
    """Create README.md content for HF dataset repo."""
    return f"""---
language:
- si
license: mit
task_categories:
- text2text-generation
tags:
- asr
- spelling-correction
- sinhala
- prediction-reference
size_categories:
- 1K<n<10K
---

# Sinhala ASR Dyslexic-Clean Dataset

## Dataset Description

This dataset contains sentence pairs from ASR output post-processing.
Fields are provided in this order:

- `dyslexic_sentence`: model/system predicted text
- `clean_sentence`: ground-truth reference text

## Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | {train_samples:,} |
| Validation | {validation_samples:,} |
| Test  | {test_samples:,} |
| **Total** | **{total_samples:,}** |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}")
print(dataset["train"][0])
# {{'dyslexic_sentence': '...', 'clean_sentence': '...'}}
```

## Citation

```bibtex
@dataset{{sinhala_asr_prediction_reference,
  title={{Sinhala ASR Prediction-Reference Dataset}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```
"""


def publish_dataset(
    input_csv_path: str,
    repo_name: str,
    seed: int,
    private: bool,
    token: str | None,
) -> None:
    """Load, split, and publish dataset to Hugging Face Hub."""
    print("=" * 72)
    print("Hugging Face Dataset Publisher (Dyslexic/Clean)")
    print("=" * 72)

    input_path = Path(input_csv_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_csv_path}")
        return

    print(f"\n1. Loading CSV: {input_csv_path}")
    rows = load_prediction_reference_csv(input_csv_path)
    print(f"   Loaded {len(rows):,} valid rows")

    if not rows:
        print("Error: No usable rows found. Check your CSV column names/content.")
        return

    print(f"\n2. Splitting dataset (80% train, 10% validation, 10% test; seed={seed})")
    train_rows, validation_rows, test_rows = split_dataset(rows, seed=seed)
    print(f"   Train rows: {len(train_rows):,}")
    print(f"   Validation rows: {len(validation_rows):,}")
    print(f"   Test rows:  {len(test_rows):,}")

    print("\n3. Building Hugging Face dataset objects")
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(validation_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    print(dataset_dict)

    print("\n4. Authenticating with Hugging Face")
    if token:
        login(token=token)
    else:
        print("   No --token provided, using interactive login / cached credentials")
        login()

    print(f"\n5. Pushing to Hub: {repo_name} (private={private})")
    dataset_dict.push_to_hub(
        repo_name,
        private=private,
        commit_message="Initial upload: dyslexic/clean sentence dataset",
    )

    print("\n6. Uploading dataset card (README.md)")
    card_content = generate_dataset_card(
        repo_name=repo_name,
        total_samples=len(rows),
        train_samples=len(train_rows),
        validation_samples=len(validation_rows),
        test_samples=len(test_rows),
    )

    local_card_path = input_path.parent / "dataset_card_prediction_reference.md"
    local_card_path.write_text(card_content, encoding="utf-8")
    print(f"   Saved local card: {local_card_path}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print("\n" + "=" * 72)
    print("Dataset published successfully")
    print(f"URL: https://huggingface.co/datasets/{repo_name}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publish dyslexic/clean CSV dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Input CSV path",
    )
    parser.add_argument(
        "--repo-name",
        "-r",
        type=str,
        default=DEFAULT_REPO_NAME,
        help="HF dataset repo id (e.g., SPEAK-PP/dataset-name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic 80/10/10 shuffling",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset as private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (optional; otherwise interactive login is used)",
    )

    args = parser.parse_args()

    publish_dataset(
        input_csv_path=args.input,
        repo_name=args.repo_name,
        seed=args.seed,
        private=args.private,
        token=args.token,
    )
