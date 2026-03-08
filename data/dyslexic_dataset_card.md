---
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
| Train | 37,056 |
| Test  | 9,265 |
| **Total** | **46,321** |

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("SPEAK-ASR/openslr-sinhala-synthetic-spell-errors-quarter")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Example
print(train_data[0])
# {'dyslexic_sentence': 'මහවවැලි ගඟට ගොස් ආපුස එයන ගමනේදී', 'correct_sentence': 'මහවැලි ගඟට ගොස් ආපසු එන ගමනේදී'}
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
    return {'input': inputs, 'target': targets}
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{sinhala_dyslexic_spelling_correction,
  title={Sinhala Dyslexic Spelling Correction Dataset},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/SPEAK-ASR/openslr-sinhala-synthetic-spell-errors-quarter}
}
```

## License

This dataset is released under the MIT License.
