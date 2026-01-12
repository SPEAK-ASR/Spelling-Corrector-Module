"""
Dataset Utilities

Quick utility functions for common dataset operations
"""

import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def load_local_csv_dataset(csv_file: Path) -> Dataset:
    """
    Load a CSV file as a HuggingFace Dataset
    
    Args:
        csv_file: Path to CSV file
    
    Returns:
        Dataset: Hugging Face Dataset object
    """
    return load_dataset("csv", data_files=str(csv_file))


def load_local_parquet_dataset(parquet_file: Path) -> Dataset:
    """
    Load a Parquet file as a HuggingFace Dataset
    
    Args:
        parquet_file: Path to Parquet file
    
    Returns:
        Dataset: Hugging Face Dataset object
    """
    return load_dataset("parquet", data_files=str(parquet_file))


def peek_dataset(dataset: Dataset, num_samples: int = 5) -> None:
    """
    Display first N samples from a dataset
    
    Args:
        dataset: The dataset to preview
        num_samples: Number of samples to show
    """
    df = dataset.to_pandas()
    print(f"\nShowing first {num_samples} samples:")
    print("="*60)
    for idx, row in df.head(num_samples).iterrows():
        print(f"\nSample {idx + 1}:")
        for col, val in row.items():
            print(f"  {col}: {val}")


def compare_sentences(dyslexic: str, clean: str) -> None:
    """
    Compare dyslexic and clean sentences side by side
    
    Args:
        dyslexic: The dyslexic sentence
        clean: The clean sentence
    """
    print(f"\nDyslexic: {dyslexic}")
    print(f"Clean:    {clean}")
    
    # Simple character-by-character comparison
    if len(dyslexic) != len(clean):
        print(f"Length mismatch: {len(dyslexic)} vs {len(clean)}")
    
    diffs = []
    for i, (d, c) in enumerate(zip(dyslexic, clean)):
        if d != c:
            diffs.append((i, d, c))
    
    if diffs:
        print(f"Differences found at positions: {[d[0] for d in diffs]}")
        for pos, d, c in diffs:
            print(f"  Position {pos}: '{d}' → '{c}'")


def filter_by_error_type(dataset: Dataset, error_type: str) -> Dataset:
    """
    Filter dataset by error type
    
    Args:
        dataset: The dataset to filter
        error_type: The error type to filter by
    
    Returns:
        Dataset: Filtered dataset
    """
    filtered = dataset.filter(lambda x: x["error_type"] == error_type)
    logger.info(f"Filtered to {len(filtered)} samples with error_type='{error_type}'")
    return filtered


def get_error_type_stats(dataset: Dataset) -> dict:
    """
    Get distribution of error types
    
    Args:
        dataset: The dataset to analyze
    
    Returns:
        dict: Error type distribution
    """
    df = dataset.to_pandas()
    stats = df["error_type"].value_counts().to_dict()
    return stats


def print_error_type_stats(dataset: Dataset) -> None:
    """
    Print error type statistics
    
    Args:
        dataset: The dataset to analyze
    """
    stats = get_error_type_stats(dataset)
    total = sum(stats.values())
    
    print("\nError Type Distribution:")
    print("="*40)
    for error_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"{error_type:20} | {count:4} ({percentage:5.1f}%) {bar}")
    print("="*40)


def dataset_to_dataframe(dataset: Dataset) -> pd.DataFrame:
    """
    Convert dataset to pandas DataFrame
    
    Args:
        dataset: The dataset to convert
    
    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    return dataset.to_pandas()


def export_to_format(dataset: Dataset, output_path: Path, format: str = "csv") -> None:
    """
    Export dataset to various formats
    
    Args:
        dataset: The dataset to export
        output_path: Output file path
        format: Format ('csv', 'parquet', 'json', 'excel')
    """
    output_path = Path(output_path)
    df = dataset.to_pandas()
    
    if format.lower() == "csv":
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif format.lower() == "parquet":
        df.to_parquet(output_path)
    elif format.lower() == "json":
        df.to_json(output_path, orient='records', indent=2)
    elif format.lower() == "excel":
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Exported to {output_path}")


def merge_datasets(*datasets: Dataset) -> Dataset:
    """
    Merge multiple datasets
    
    Args:
        *datasets: Variable number of datasets to merge
    
    Returns:
        Dataset: Merged dataset
    """
    if not datasets:
        raise ValueError("At least one dataset required")
    
    merged = datasets[0]
    for dataset in datasets[1:]:
        merged = merged.concatenate(dataset)
    
    return merged


def split_dataset(dataset: Dataset, train_ratio: float = 0.8) -> tuple:
    """
    Split dataset into train and test
    
    Args:
        dataset: The dataset to split
        train_ratio: Ratio for training data (0.0-1.0)
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    split = dataset.train_test_split(test_size=1-train_ratio)
    return split["train"], split["test"]
