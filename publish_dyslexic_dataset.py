"""
Hugging Face Dataset Publisher - Dyslexic Correct Pairs
========================================================
Publishes the dyslexic_correct_pairs.csv dataset to Hugging Face Hub.

Features:
- Splits dataset into train and test sets
- Uploads to Hugging Face with proper metadata
- Creates dataset card with description

Usage:
    python publish_dyslexic_dataset.py --repo-name your-username/sinhala-dyslexic-correction
    python publish_dyslexic_dataset.py --repo-name your-username/sinhala-dyslexic-correction --test-size 0.1
"""

import csv
import random
import argparse
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: Required libraries not found. Install them with:")
    print("    pip install datasets huggingface_hub")
    exit(1)


# Default input file path
DEFAULT_INPUT_PATH = "data/dyslexic_correct_pairs.csv"


# ============================================================================
# DATASET LOADING AND SPLITTING
# ============================================================================

def load_dyslexic_csv_data(csv_path: str) -> list:
    """
    Load data from the dyslexic_correct_pairs.csv file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries with dyslexic_sentence and correct_sentence
    """
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dyslexic = row.get('dyslexic_sentence', '').strip()
            correct = row.get('correct_sentence', '').strip()
            
            if dyslexic and correct:
                data.append({
                    'dyslexic_sentence': dyslexic,
                    'correct_sentence': correct
                })
    
    return data


def split_dataset(data: list, test_size: float = 0.1, seed: int = 42) -> tuple:
    """
    Split dataset into train and test sets.
    
    Args:
        data: List of data dictionaries
        test_size: Fraction of data to use for test set (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    split_idx = int(len(shuffled_data) * (1 - test_size))
    
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data


# ============================================================================
# DATASET CARD GENERATION
# ============================================================================

def generate_dataset_card(
    repo_name: str,
    total_samples: int,
    train_samples: int,
    test_samples: int,
    test_size: float
) -> str:
    """
    Generate a dataset card (README.md) for the Hugging Face dataset.
    
    Args:
        repo_name: Repository name
        total_samples: Total number of samples
        train_samples: Number of training samples
        test_samples: Number of test samples
        test_size: Test split ratio
        
    Returns:
        Dataset card content as string
    """
    card = f"""---
language:
- si
- en
license: mit
task_categories:
- text2text-generation
- text-generation
tags:
- spelling-correction
- dyslexia
- sinhala
- code-mixed
- text-correction
- sinhala-english
size_categories:
- 10K<n<100K
---

# Sinhala Dyslexic Spelling Correction Dataset

## Dataset Description

This dataset contains Sinhala and code-mixed (Sinhala-English) text pairs for training spelling correction models, specifically designed to address dyslexia-like spelling errors.

### Features

- `dyslexic_sentence`: Input text with dyslexia-like spelling errors (string)
- `correct_sentence`: Corrected output text (string)

### Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | {train_samples:,} |
| Test  | {test_samples:,} |
| **Total** | **{total_samples:,}** |

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_name}")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Example
print(train_data[0])
# {{'dyslexic_sentence': 'මහවවැලි ගඟට ගොස් ආපුස එයන ගමනේදී', 'correct_sentence': 'මහවැලි ගඟට ගොස් ආපසු එන ගමනේදී'}}
```

## Dataset Creation

This dataset was created to address spelling correction challenges in Sinhala text, including:

### Error Types Included

- **Character substitution**: Similar-looking Sinhala characters swapped
- **Character deletion**: Missing characters in words
- **Character insertion**: Extra characters added
- **Character transposition**: Swapped adjacent characters
- **Phonetic errors**: Similar sounding Sinhala characters (e.g., ණ/න, ෂ/ස)
- **Diacritic errors**: Vowel sign (matra) mistakes
- **Word order errors**: Shuffled word positions in sentences
- **Code-mixed errors**: Errors in Sinhala-English mixed text

### Languages

- Primary: Sinhala (සිංහල)
- Secondary: English (code-mixed content)

## Example Pairs

| Dyslexic (Input) | Correct (Output) |
|------------------|------------------|
| මහවවැලි ගඟට ගොස් ආපුස එයන ගමනේදී | මහවැලි ගඟට ගොස් ආපසු එන ගමනේදී |
| ඓවන් ශ්‍රේෂ්ඨ ජාථියක් බිහි කිරීමට | එවන් ශ්‍රේෂ්ඨ ජාතියක් බිහි කිරීමට |
| යම්කිසි මනුෂ්‍යයෙක් ෂිල් රකිනවානම් | යම්කිසි මනුෂ්‍යයෙක් සිල් රකිනවානම් |

## Model Training

This dataset is suitable for fine-tuning:
- T5-based models
- mBART models
- Sequence-to-sequence transformers
- Custom encoder-decoder models

### Recommended Preprocessing

```python
def preprocess_function(examples):
    inputs = ["correct: " + text for text in examples['dyslexic_sentence']]
    targets = examples['correct_sentence']
    return {{'input': inputs, 'target': targets}}
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{sinhala_dyslexic_spelling_correction,
  title={{Sinhala Dyslexic Spelling Correction Dataset}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

This dataset is released under the MIT License.
"""
    return card


# ============================================================================
# PUBLISHING FUNCTION
# ============================================================================

def publish_dataset(
    input_csv_path: str,
    repo_name: str,
    test_size: float = 0.1,
    seed: int = 42,
    private: bool = False,
    token: str = None
) -> None:
    """
    Publish dataset to Hugging Face Hub.
    
    Args:
        input_csv_path: Path to the input CSV file
        repo_name: Hugging Face repository name (e.g., 'username/dataset-name')
        test_size: Fraction of data for test set
        seed: Random seed for reproducibility
        private: Whether to make the dataset private
        token: Hugging Face API token (optional, will prompt if not provided)
    """
    print("=" * 60)
    print("Dyslexic Dataset Publisher - Hugging Face")
    print("=" * 60)
    
    # Check if input file exists
    if not Path(input_csv_path).exists():
        print(f"Error: Input file '{input_csv_path}' does not exist!")
        return
    
    # Load data
    print(f"\n1. Loading data from: {input_csv_path}")
    data = load_dyslexic_csv_data(input_csv_path)
    print(f"   Loaded {len(data):,} samples")
    
    if len(data) == 0:
        print("Error: No data loaded from CSV!")
        return
    
    # Split data
    print(f"\n2. Splitting dataset (test_size={test_size}, seed={seed})")
    train_data, test_data = split_dataset(data, test_size=test_size, seed=seed)
    print(f"   Train samples: {len(train_data):,}")
    print(f"   Test samples:  {len(test_data):,}")
    
    # Create Hugging Face datasets
    print(f"\n3. Creating Hugging Face Dataset objects")
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    print(f"   Dataset structure:")
    print(f"   {dataset_dict}")
    
    # Show samples
    print(f"\n4. Sample data:")
    print(f"\n   Train sample:")
    dyslexic_sample = train_data[0]['dyslexic_sentence'][:80]
    correct_sample = train_data[0]['correct_sentence'][:80]
    print(f"      Dyslexic: {dyslexic_sample}...")
    print(f"      Correct:  {correct_sample}...")
    print(f"\n   Test sample:")
    dyslexic_test = test_data[0]['dyslexic_sentence'][:80]
    correct_test = test_data[0]['correct_sentence'][:80]
    print(f"      Dyslexic: {dyslexic_test}...")
    print(f"      Correct:  {correct_test}...")
    
    # Login to Hugging Face
    print(f"\n5. Authenticating with Hugging Face Hub")
    if token:
        login(token=token)
    else:
        print("   Please login to Hugging Face (or set HF_TOKEN environment variable)")
        login()
    
    # Push to hub
    print(f"\n6. Pushing dataset to: {repo_name}")
    print(f"   Private: {private}")
    
    dataset_dict.push_to_hub(
        repo_name,
        private=private,
        commit_message="Initial dataset upload - Dyslexic Correct Pairs"
    )
    
    # Generate and upload dataset card
    print(f"\n7. Generating dataset card")
    card_content = generate_dataset_card(
        repo_name=repo_name,
        total_samples=len(data),
        train_samples=len(train_data),
        test_samples=len(test_data),
        test_size=test_size
    )
    
    # Save card locally for reference
    card_path = Path(input_csv_path).parent / "dyslexic_dataset_card.md"
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card_content)
    print(f"   Dataset card saved to: {card_path}")
    
    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
    
    print(f"\n" + "=" * 60)
    print("Dataset published successfully!")
    print(f"   URL: https://huggingface.co/datasets/{repo_name}")
    print("=" * 60)


def save_splits_locally(
    input_csv_path: str,
    output_dir: str,
    test_size: float = 0.1,
    seed: int = 42
) -> None:
    """
    Save train/test splits to local CSV files without publishing.
    
    Args:
        input_csv_path: Path to the input CSV file
        output_dir: Output directory for split files
        test_size: Fraction of data for test set
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Local Dataset Splitter - Dyslexic Correct Pairs")
    print("=" * 60)
    
    # Check if input file exists
    if not Path(input_csv_path).exists():
        print(f"Error: Input file '{input_csv_path}' does not exist!")
        return
    
    # Load data
    print(f"\n1. Loading data from: {input_csv_path}")
    data = load_dyslexic_csv_data(input_csv_path)
    print(f"   Loaded {len(data):,} samples")
    
    if len(data) == 0:
        print("Error: No data loaded from CSV!")
        return
    
    # Split data
    print(f"\n2. Splitting dataset (test_size={test_size}, seed={seed})")
    train_data, test_data = split_dataset(data, test_size=test_size, seed=seed)
    print(f"   Train samples: {len(train_data):,}")
    print(f"   Test samples:  {len(test_data):,}")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train split
    train_path = output_path / "dyslexic_train.csv"
    print(f"\n3. Saving train split to: {train_path}")
    with open(train_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dyslexic_sentence', 'correct_sentence'])
        writer.writeheader()
        writer.writerows(train_data)
    
    # Save test split
    test_path = output_path / "dyslexic_test.csv"
    print(f"   Saving test split to: {test_path}")
    with open(test_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dyslexic_sentence', 'correct_sentence'])
        writer.writeheader()
        writer.writerows(test_data)
    
    print(f"\n" + "=" * 60)
    print("Splits saved successfully!")
    print(f"   Train: {train_path} ({len(train_data):,} samples)")
    print(f"   Test:  {test_path} ({len(test_data):,} samples)")
    print("=" * 60)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publish dyslexic_correct_pairs.csv dataset to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Publish to Hugging Face (uses default input file)
    python publish_dyslexic_dataset.py --repo-name your-username/sinhala-dyslexic-correction
    
    # With custom input file and test size
    python publish_dyslexic_dataset.py --input data/dyslexic_correct_pairs.csv --repo-name your-username/sinhala-dyslexic-correction --test-size 0.15
    
    # Save splits locally without publishing
    python publish_dyslexic_dataset.py --local-only --output-dir data/dyslexic_splits
    
    # Make dataset private
    python publish_dyslexic_dataset.py --repo-name your-username/sinhala-dyslexic-correction --private
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_PATH,
                        help=f'Input CSV file path (default: {DEFAULT_INPUT_PATH})')
    parser.add_argument('--repo-name', '-r', type=str,
                        help='Hugging Face repository name (e.g., username/sinhala-dyslexic-correction)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--private', action='store_true',
                        help='Make the dataset private')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face API token (optional)')
    parser.add_argument('--local-only', action='store_true',
                        help='Only save splits locally, do not publish to Hugging Face')
    parser.add_argument('--output-dir', type=str, default='data/dyslexic_splits',
                        help='Output directory for local splits (default: data/dyslexic_splits)')
    
    args = parser.parse_args()
    
    if args.local_only:
        save_splits_locally(
            input_csv_path=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            seed=args.seed
        )
    else:
        if not args.repo_name:
            parser.error("--repo-name is required when publishing to Hugging Face")
        
        publish_dataset(
            input_csv_path=args.input,
            repo_name=args.repo_name,
            test_size=args.test_size,
            seed=args.seed,
            private=args.private,
            token=args.token
        )
