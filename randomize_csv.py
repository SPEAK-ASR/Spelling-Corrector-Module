"""
CSV Row Randomizer
==================
Randomizes/shuffles the order of rows in a CSV file and saves to a new file.

Usage:
    python randomize_csv.py --input data/final.csv --output data/final_randomized.csv
"""

import csv
import random
import argparse
from pathlib import Path


def randomize_csv(
    input_csv_path: str,
    output_csv_path: str,
    seed: int = None
) -> None:
    """
    Randomize the order of rows in a CSV file.
    
    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path for the output randomized CSV file
        seed: Random seed for reproducibility (optional)
    """
    input_path = Path(input_csv_path)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_csv_path}' does not exist!")
        return
    
    print(f"Reading input file: {input_csv_path}")
    
    # Read all rows
    rows = []
    header = None
    
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header
        for row in reader:
            rows.append(row)
    
    print(f"   Loaded {len(rows):,} rows")
    print(f"   Columns: {header}")
    
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"   Using random seed: {seed}")
    
    # Shuffle rows
    print(f"\nRandomizing rows...")
    random.shuffle(rows)
    
    # Ensure output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to output file
    print(f"Writing to: {output_csv_path}")
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"\nDone!")
    print(f"   Total rows randomized: {len(rows):,}")
    print(f"   Output file: {output_csv_path}")
    
    # Show sample of randomized output
    print(f"\nSample (first 5 rows after randomization):")
    for i, row in enumerate(rows[:5]):
        print(f"   Row {i + 1}: {row[0][:50]}..." if len(row[0]) > 50 else f"   Row {i + 1}: {row[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomize the order of rows in a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python randomize_csv.py --input data/final.csv --output data/final_randomized.csv
    python randomize_csv.py -i data/final.csv -o data/shuffled.csv --seed 42
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for reproducibility (optional)')
    
    args = parser.parse_args()
    
    randomize_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        seed=args.seed
    )
