---
language:
- si
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
size_categories:
- 10K<n<100K
---

# Sinhala Spelling Correction Dataset

## Dataset Description

This dataset contains Sinhala text pairs for training spelling correction models. It includes:
- **Dyslexic/Noisy sentences**: Text with spelling errors, typos, and dyslexia-like mistakes
- **Clean sentences**: Corrected versions of the text

### Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | 37,712 |
| Test  | 9,428 |
| **Total** | **47,140** |

### Features

- `dyslexic_sentence`: Input text with errors (string)
- `clean_sentence`: Corrected output text (string)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("SPEAK-ASR/sinhala-spelling-correction")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Example
print(train_data[0])
# {'dyslexic_sentence': '...', 'clean_sentence': '...'}
```

## Dataset Creation

This dataset was created by combining:
1. Synthetically generated noisy Sinhala text
2. Code-mixed (Sinhala-English) text with errors
3. Existing dyslexia correction datasets

### Error Types Included

- **Spelling errors**: Character substitution, deletion, insertion, transposition
- **Phonetic errors**: Similar sounding Sinhala characters
- **Diacritic errors**: Matra/vowel sign mistakes
- **Grammar errors**: Word order, suffix errors

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{sinhala_spelling_correction,
  title={Sinhala Spelling Correction Dataset},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/SPEAK-ASR/sinhala-spelling-correction}
}
```

## License

This dataset is released under the MIT License.
