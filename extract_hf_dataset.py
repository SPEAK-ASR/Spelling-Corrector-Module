"""
Hugging Face Dataset Extractor
==============================
Extracts data from the SPEAK-ASR/sinhala-dyslexia-corrected-id20percent dataset
on Hugging Face and saves it to a CSV file.

Output format:
- noisy_sentence (first column)
- correct_sentence (second column)
- Punctuation marks are removed

Usage:
    python extract_hf_dataset.py --output data/initial_dataset.csv
"""

import csv
import re
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install it with:")
    print("    pip install datasets")
    exit(1)


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks from text.
    
    Args:
        text: The text to clean
        
    Returns:
        Text with punctuation removed
    """
    if not text:
        return ""
    
    # Common punctuation marks to remove
    punctuation = r'[.,!?;:"\'\-\(\)\[\]\{\}<>@#$%^&*+=|\\~`/।]'
    
    # Remove punctuation
    cleaned = re.sub(punctuation, '', text)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_dataset(
    dataset_name: str = "SPEAK-ASR/sinhala-dyslexia-corrected-id20percent",
    output_csv_path: str = "data/initial_dataset.csv",
    noisy_column: str = None,
    clean_column: str = None
) -> None:
    """
    Extract dataset from Hugging Face and save to CSV.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        output_csv_path: Path for the output CSV file
        noisy_column: Column name for noisy/dyslexic text (auto-detected if None)
        clean_column: Column name for clean/correct text (auto-detected if None)
    """
    print(f"Loading dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully!")
        print(f"\nDataset structure:")
        print(dataset)
        
        # Get the split (usually 'train', but could be different)
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"\nUsing split: '{split_name}'")
        print(f"Number of examples: {len(data):,}")
        
        # Show column names
        column_names = data.column_names
        print(f"Columns: {column_names}")
        
        # Auto-detect column names if not provided
        if noisy_column is None or clean_column is None:
            # Common patterns for column names
            noisy_patterns = ['dyslexic', 'noisy', 'input', 'source', 'incorrect', 'misspelled']
            clean_patterns = ['clean', 'correct', 'target', 'output', 'corrected']
            
            for col in column_names:
                col_lower = col.lower()
                if noisy_column is None:
                    for pattern in noisy_patterns:
                        if pattern in col_lower:
                            noisy_column = col
                            break
                if clean_column is None:
                    for pattern in clean_patterns:
                        if pattern in col_lower and 'dyslexic' not in col_lower and 'noisy' not in col_lower:
                            clean_column = col
                            break
            
            # If still not found, use first two columns
            if noisy_column is None and len(column_names) >= 1:
                noisy_column = column_names[0]
            if clean_column is None and len(column_names) >= 2:
                clean_column = column_names[1]
        
        print(f"\nUsing columns:")
        print(f"   Noisy/Dyslexic: '{noisy_column}'")
        print(f"   Clean/Correct:  '{clean_column}'")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        for i in range(min(3, len(data))):
            print(f"\n   Row {i + 1}:")
            print(f"      Noisy: {data[i][noisy_column][:80]}..." if len(str(data[i][noisy_column])) > 80 else f"      Noisy: {data[i][noisy_column]}")
            print(f"      Clean: {data[i][clean_column][:80]}..." if len(str(data[i][clean_column])) > 80 else f"      Clean: {data[i][clean_column]}")
        
        # Ensure output directory exists
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract and save to CSV
        print(f"\nExtracting and cleaning data...")
        
        total_rows = 0
        skipped_rows = 0
        
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['noisy_sentence', 'correct_sentence'])
            
            for i, row in enumerate(data):
                noisy_text = str(row[noisy_column]) if row[noisy_column] else ""
                clean_text = str(row[clean_column]) if row[clean_column] else ""
                
                # Remove punctuation
                noisy_text = remove_punctuation(noisy_text)
                clean_text = remove_punctuation(clean_text)
                
                # Only write if both are not empty
                if noisy_text and clean_text:
                    writer.writerow([noisy_text, clean_text])
                    total_rows += 1
                else:
                    skipped_rows += 1
                
                if (i + 1) % 5000 == 0:
                    print(f"   Processed {i + 1:,}/{len(data):,} rows...")
        
        print(f"\n" + "=" * 60)
        print(f"Extraction complete!")
        print(f"   Total rows extracted: {total_rows:,}")
        print(f"   Rows skipped (empty): {skipped_rows:,}")
        print(f"   Output file: {output_csv_path}")
        
        # Show sample of cleaned output
        print(f"\nSample cleaned output:")
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f"\n   Row {i + 1}:")
                print(f"      Noisy: {row[0][:80]}..." if len(row[0]) > 80 else f"      Noisy: {row[0]}")
                print(f"      Clean: {row[1][:80]}..." if len(row[1]) > 80 else f"      Clean: {row[1]}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure you have internet connection")
        print("   2. Check if the dataset name is correct")
        print("   3. Try: pip install --upgrade datasets")
        raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Hugging Face dataset to CSV with punctuation removed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_hf_dataset.py --output data/initial_dataset.csv
    python extract_hf_dataset.py --dataset SPEAK-ASR/sinhala-dyslexia-corrected-id20percent --output data/initial_dataset.csv
    python extract_hf_dataset.py --noisy-col dyslexic_sentence --clean-col correct_sentence
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str, 
                        default="SPEAK-ASR/sinhala-dyslexia-corrected-id20percent",
                        help='Hugging Face dataset name (default: SPEAK-ASR/sinhala-dyslexia-corrected-id20percent)')
    parser.add_argument('--output', '-o', type=str, 
                        default="data/initial_dataset.csv",
                        help='Output CSV file path (default: data/initial_dataset.csv)')
    parser.add_argument('--noisy-col', type=str, default=None,
                        help='Column name for noisy/dyslexic text (auto-detected if not provided)')
    parser.add_argument('--clean-col', type=str, default=None,
                        help='Column name for clean/correct text (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    extract_dataset(
        dataset_name=args.dataset,
        output_csv_path=args.output,
        noisy_column=args.noisy_col,
        clean_column=args.clean_col
    )
