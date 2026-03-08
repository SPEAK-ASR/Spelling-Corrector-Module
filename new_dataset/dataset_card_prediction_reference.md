---
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

# Sinhala ASR Prediction-Reference Dataset (3000 no-numbers)

## Dataset Description

This dataset contains sentence pairs for spelling correction:

- `dyslexic_sentence`: noisy / predicted text
- `clean_sentence`: clean reference text

## Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | 2,400 |
| Eval  | 300 |
| Test  | 300 |
| **Total** | **3,000** |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("SPEAK-PP/sinhala-spelling-correction-already-corrected-pairs")
print(dataset["train"][0])
```
