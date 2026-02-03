"""
Code-Mixed Small Chunks Generator
=================================
Splits sentences into small chunks (2-5 words) and includes them only if 
they contain at least one English word.

The input CSV already contains dyslexic_sentence and clean_sentence pairs,
so no error generation is needed.

Usage:
    python generate_small_chunks.py --input code-mixed-long-chunks.csv --output data/small_chunks.csv
    python generate_small_chunks.py --input code-mixed-long-chunks.csv --output out.csv --min-words 2 --max-words 5
"""

import csv
import argparse
import re
from pathlib import Path


# ============================================================================
# TEXT CLEANING
# ============================================================================

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks from text.
    
    Args:
        text: The text to clean
        
    Returns:
        Text with punctuation removed
    """
    # Common punctuation marks to remove
    punctuation = r'[.,!?;:"\'\-\(\)\[\]\{\}<>@#$%^&*+=|\\~`/]'
    
    # Remove punctuation
    cleaned = re.sub(punctuation, '', text)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


# ============================================================================
# ENGLISH WORD DETECTION
# ============================================================================

def contains_english_word(text: str) -> bool:
    """
    Check if text contains at least one English word.
    
    Args:
        text: The text to check
        
    Returns:
        True if text contains at least one English word
    """
    # Pattern to match English words (letters a-z, A-Z)
    # Must be at least 2 characters to avoid single letter matches
    english_pattern = r'\b[a-zA-Z]{2,}\b'
    
    matches = re.findall(english_pattern, text)
    return len(matches) > 0


# ============================================================================
# CHUNKING FUNCTIONS
# ============================================================================

def split_into_words(text: str) -> list:
    """
    Split text into words, preserving word boundaries.
    
    Args:
        text: The text to split
        
    Returns:
        List of words
    """
    # Split by whitespace
    words = text.split()
    return [w for w in words if w.strip()]


def create_aligned_chunks(dyslexic_text: str, clean_text: str, min_words: int = 2, max_words: int = 5) -> list:
    """
    Split both texts into aligned small chunks (2-5 words).
    Only include chunks that contain at least one English word.
    
    Args:
        dyslexic_text: The dyslexic/noisy sentence
        clean_text: The clean/correct sentence
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk
        
    Returns:
        List of tuples (dyslexic_chunk, clean_chunk)
    """
    chunks = []
    
    dyslexic_words = split_into_words(dyslexic_text)
    clean_words = split_into_words(clean_text)
    
    # Split clean text into chunks of max_words
    clean_chunks = []
    for i in range(0, len(clean_words), max_words):
        chunk_words = clean_words[i:i + max_words]
        if len(chunk_words) >= min_words:
            clean_chunks.append((i, chunk_words))
    
    # Split dyslexic text into chunks of max_words
    dyslexic_chunks = []
    for i in range(0, len(dyslexic_words), max_words):
        chunk_words = dyslexic_words[i:i + max_words]
        if len(chunk_words) >= min_words:
            dyslexic_chunks.append((i, chunk_words))
    
    # Pair up chunks by index
    # Use the minimum number of chunks from both
    num_chunks = min(len(clean_chunks), len(dyslexic_chunks))
    
    for idx in range(num_chunks):
        _, clean_chunk_words = clean_chunks[idx]
        _, dyslexic_chunk_words = dyslexic_chunks[idx]
        
        clean_chunk = ' '.join(clean_chunk_words)
        dyslexic_chunk = ' '.join(dyslexic_chunk_words)
        
        # Only include if at least one of them contains an English word
        if contains_english_word(clean_chunk) or contains_english_word(dyslexic_chunk):
            # Remove punctuation from both chunks
            clean_chunk = remove_punctuation(clean_chunk)
            dyslexic_chunk = remove_punctuation(dyslexic_chunk)
            
            # Only add if chunks are not empty after cleaning
            if clean_chunk and dyslexic_chunk:
                chunks.append((dyslexic_chunk, clean_chunk))
    
    return chunks


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def generate_small_chunks(
    input_csv_path: str,
    output_csv_path: str,
    min_words: int = 2,
    max_words: int = 5
) -> None:
    """
    Generate small chunks CSV from code-mixed long chunks.
    Only includes chunks that contain at least one English word.
    
    Args:
        input_csv_path: Path to the input CSV file (code-mixed-long-chunks.csv)
        output_csv_path: Path for the output CSV file
        min_words: Minimum words per chunk (default: 2)
        max_words: Maximum words per chunk (default: 5)
    """
    input_path = Path(input_csv_path)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_csv_path}' does not exist!")
        return
    
    print(f"Reading input file: {input_csv_path}")
    
    # Read input CSV
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"   Loaded {len(rows):,} sentence pairs")
    
    # Extract chunks
    print(f"\nSplitting into chunks ({min_words}-{max_words} words)...")
    print(f"   Only including chunks with at least 1 English word...")
    
    all_chunks = []
    
    for i, row in enumerate(rows):
        dyslexic = row.get('dyslexic_sentence', '')
        clean = row.get('clean_sentence', '')
        
        if not dyslexic or not clean:
            continue
        
        # Split into aligned chunks and filter by English word presence
        chunks = create_aligned_chunks(dyslexic, clean, min_words, max_words)
        all_chunks.extend(chunks)
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1:,}/{len(rows):,} sentences, found {len(all_chunks):,} chunks so far...")
    
    print(f"   Total chunks extracted: {len(all_chunks):,}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk[0], chunk[1])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    print(f"   Unique chunks: {len(unique_chunks):,}")
    
    # Ensure output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write output CSV
    print(f"\nWriting output to: {output_csv_path}")
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dyslexic_chunk', 'clean_chunk'])
        
        for dyslexic_chunk, clean_chunk in unique_chunks:
            writer.writerow([dyslexic_chunk, clean_chunk])
    
    print(f"\nDone! Generated {len(unique_chunks):,} small chunks")
    
    # Show statistics
    print(f"\nStatistics:")
    word_counts = {}
    for _, clean_chunk in unique_chunks:
        word_count = len(clean_chunk.split())
        word_counts[word_count] = word_counts.get(word_count, 0) + 1
    
    for wc in sorted(word_counts.keys()):
        count = word_counts[wc]
        pct = (count / len(unique_chunks)) * 100 if unique_chunks else 0
        bar = '█' * int(pct / 2)
        print(f"   {wc} words: {count:5} ({pct:5.1f}%) {bar}")
    
    # Show sample chunks
    print(f"\nSample chunks:")
    import random
    samples = random.sample(unique_chunks, min(10, len(unique_chunks)))
    for dyslexic, clean in samples:
        print(f"   Dyslexic: {dyslexic}")
        print(f"   Clean:    {clean}")
        print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split sentences into small chunks (2-5 words) containing at least one English word",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_small_chunks.py --input code-mixed-long-chunks.csv --output data/small_chunks.csv
    python generate_small_chunks.py -i code-mixed-long-chunks.csv -o out.csv --min-words 2 --max-words 5
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file path (code-mixed-long-chunks.csv)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--min-words', type=int, default=2,
                        help='Minimum words per chunk (default: 2)')
    parser.add_argument('--max-words', type=int, default=5,
                        help='Maximum words per chunk (default: 5)')
    
    args = parser.parse_args()
    
    generate_small_chunks(
        input_csv_path=args.input,
        output_csv_path=args.output,
        min_words=args.min_words,
        max_words=args.max_words
    )
