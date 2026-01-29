# Sinhala Spelling Corrector Dataset

## Dataset Description

A code-mixed Sinhala-English dataset designed for training spelling correction models, specifically targeting dyslexic error patterns in Sinhala text.

### Dataset Summary

This dataset contains pairs of noisy (misspelled) and clean (correct) Sinhala text chunks, suitable for sequence-to-sequence spelling correction tasks. The dataset includes synthetically generated dyslexic errors that simulate common spelling mistakes made by Sinhala speakers, including those with dyslexia.

### Supported Tasks

- **Spelling Correction**: Correcting misspelled Sinhala text to its correct form
- **Text Normalization**: Normalizing noisy text input
- **Sequence-to-Sequence Generation**: Training encoder-decoder models for text correction

### Languages

- **Primary**: Sinhala (සිංහල) - `si`
- **Secondary**: English (code-mixed) - `en`

---

## Dataset Structure

### Data Instances

```json
{
  "noisy_sentence": "මම පාසැලට යනවා",
  "correct_sentence": "මම පාසලට යනවා"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `noisy_sentence` / `dyslexic_chunk` | string | The misspelled/noisy version of the text |
| `correct_sentence` / `clean_chunk` | string | The correct/clean version of the text |

### Data Splits

| Split | Number of Examples | Percentage |
|-------|-------------------|------------|
| Train | 43,845 | 80% |
| Test | 10,962 | 20% |
| **Total** | **54,807** | 100% |

---

## Dataset Creation

### Source Data

The dataset was created from multiple sources:
1. **Sinhala text corpora** - Clean Sinhala text chunks extracted from various documents
2. **Code-mixed text** - Sinhala-English mixed content from real-world sources
3. **Hugging Face dataset** - `SPEAK-ASR/sinhala-dyslexia-corrected-id20percent`

### Error Generation Process

Synthetic errors were generated using the following techniques:

#### 1. Sinhala Spelling Errors
- **Character Substitution**: Replacing characters with visually similar Sinhala characters
- **Character Deletion**: Randomly removing characters
- **Character Insertion**: Adding extra characters
- **Character Transposition**: Swapping adjacent characters
- **Character Repetition**: Duplicating characters

#### 2. Sinhala Phonetic Errors
Common phonetic confusions in Sinhala:
- ක ↔ ග (ka ↔ ga)
- ත ↔ ද (ta ↔ da)
- ප ↔ බ (pa ↔ ba)
- ස ↔ ශ ↔ ෂ (sa ↔ sha)
- න ↔ ණ (na ↔ ṇa)
- ල ↔ ළ (la ↔ ḷa)

#### 3. Sinhala Diacritic Errors
Manipulation of vowel diacritics (පිලි):
- ා, ි, ී, ු, ූ, ෙ, ේ, ො, ෝ, ෞ, ං, ඃ, ්

### Data Processing Pipeline

```
Raw Text → Chunk Extraction (2-5 words) → Error Generation → 
Punctuation Removal → Randomization → Train/Test Split → Final Dataset
```

---

## Usage

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("YOUR_USERNAME/sinhala-spelling-corrector")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# Example usage
for example in train_data[:5]:
    print(f"Noisy: {example['noisy_sentence']}")
    print(f"Clean: {example['correct_sentence']}")
    print("---")
```

### Loading from CSV

```python
import pandas as pd

# Load train and test splits
train_df = pd.read_csv('data/splits/train.csv')
test_df = pd.read_csv('data/splits/test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
```

### Training Example (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Prepare dataset
def preprocess_function(examples):
    inputs = tokenizer(examples['noisy_sentence'], max_length=128, truncation=True, padding='max_length')
    targets = tokenizer(examples['correct_sentence'], max_length=128, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs

# Train
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(preprocess_function, batched=True)
```

---

## Dataset Statistics

### Character Distribution

| Category | Description |
|----------|-------------|
| Sinhala Unicode Range | U+0D80 - U+0DFF |
| Sinhala Consonants | ක-ෆ (25 base consonants) |
| Sinhala Vowels | අ-ඖ (18 vowels) |
| Diacritics | ් ා ි ී ු ූ ෙ ේ ො ෝ ෞ ං ඃ |

### Chunk Statistics

- **Minimum words per chunk**: 2
- **Maximum words per chunk**: 5
- **Average chunk length**: ~3-4 words
- **Code-mixed chunks**: Includes chunks with at least 1 English word

---

## Considerations

### Biases

- The dataset contains synthetically generated errors which may not capture all real-world error patterns
- Code-mixed content may have varying proportions of Sinhala and English
- Phonetic errors are based on common Sinhala speaker mistakes

### Limitations

- Limited to short text chunks (2-5 words)
- Does not include context-dependent spelling errors
- May not cover all dialectal variations of Sinhala

### Recommendations

- Fine-tune on domain-specific data for better performance
- Combine with other Sinhala NLP datasets for improved generalization
- Consider augmenting with real-world spelling error data

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{sinhala_spelling_corrector_2026,
  title={Sinhala Spelling Corrector Dataset},
  author={[Your Name]},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/YOUR_USERNAME/sinhala-spelling-corrector}}
}
```

---

## License

This dataset is released under the [MIT License](LICENSE).

---

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | January 2026 | Initial release with 54,807 samples |

