"""
Sinhala Noisy Data Generator
============================
Takes Sinhala text files from a folder, splits them into sentence-like chunks,
and creates a CSV with correct_sentence and noisy_sentence columns.

Error types implemented:
- Spelling errors (substitution, deletion, insertion, transposition, repetition)
- Phonetic errors (similar sounding characters)
- Grammar errors (suffix errors, word order, particle errors)
- Diacritic errors (matra/vowel sign mistakes)

Usage:
    python generate_noisy_data.py --input-folder data/texts/ --output data/training_data.csv
"""

import random
import re
import csv
import argparse
from pathlib import Path


# ============================================================================
# SINHALA ERROR GENERATION MAPPINGS
# ============================================================================

# Phonetically similar characters (sounds alike)
SINHALA_PHONETIC_CONFUSIONS = {
    # Aspirated vs Non-aspirated consonants
    'ක': ['ඛ', 'ග'], 'ඛ': ['ක', 'ඝ'], 'ග': ['ඝ', 'ක'], 'ඝ': ['ග', 'ඛ'],
    'ච': ['ඡ', 'ජ'], 'ඡ': ['ච', 'ඣ'], 'ජ': ['ඣ', 'ච'], 'ඣ': ['ජ', 'ඡ'],
    'ට': ['ඨ', 'ඩ'], 'ඨ': ['ට', 'ඪ'], 'ඩ': ['ඪ', 'ට'], 'ඪ': ['ඩ', 'ඨ'],
    'ත': ['ථ', 'ද'], 'ථ': ['ත', 'ධ'], 'ද': ['ධ', 'ත'], 'ධ': ['ද', 'ථ'],
    'ප': ['ඵ', 'බ'], 'ඵ': ['ප', 'භ'], 'බ': ['භ', 'ප'], 'භ': ['බ', 'ඵ'],
    # Nasals
    'ඤ': ['ඥ', 'න'], 'ඥ': ['ඤ', 'ණ'], 'ණ': ['ඥ', 'න'], 'න': ['ණ', 'ඤ'],
    # Sibilants (common confusion)
    'ස': ['ශ', 'ෂ'], 'ශ': ['ස', 'ෂ'], 'ෂ': ['ස', 'ශ'],
    # Short vs Long vowels
    'අ': ['ආ'], 'ආ': ['අ'],
    'ඉ': ['ඊ'], 'ඊ': ['ඉ'],
    'උ': ['ඌ'], 'ඌ': ['උ'],
    'එ': ['ඒ', 'ඓ'], 'ඒ': ['එ'], 'ඓ': ['එ'],
    'ඔ': ['ඕ', 'ඖ'], 'ඕ': ['ඔ'], 'ඖ': ['ඔ'],
}

# Visually similar characters (looks alike on keyboard/screen)
SINHALA_VISUAL_CONFUSIONS = {
    'ර': ['ල'], 'ල': ['ර'],
    'ම': ['භ'], 'භ': ['ම'],
    'ය': ['ර'],
    'ව': ['හ'], 'හ': ['ව'],
    'ණ': ['ඩ'], 'ඩ': ['ණ'],
}

# Common diacritic marks (matras)
SINHALA_DIACRITICS = ['ා', 'ැ', 'ෑ', 'ි', 'ී', 'ු', 'ූ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ං', 'ඃ', '්']

# Similar diacritics (commonly confused)
DIACRITIC_CONFUSIONS = {
    'ා': ['ැ', 'ෑ'],  # aa vs ae
    'ැ': ['ා', 'ෑ'],
    'ෑ': ['ා', 'ැ'],
    'ි': ['ී', 'ු'],  # i vs ii
    'ී': ['ි'],
    'ු': ['ූ', 'ි'],  # u vs uu
    'ූ': ['ු'],
    'ෙ': ['ේ', 'ෛ'],  # e vs ee
    'ේ': ['ෙ'],
    'ො': ['ෝ'],  # o vs oo
    'ෝ': ['ො'],
}

# Common Sinhala consonants for random insertion
SINHALA_CONSONANTS = 'කඛගඝඞචඡජඣඤටඨඩඪණතථදධනපඵබභමයරලවශෂසහළ'

# Common grammar patterns that get confused
GRAMMAR_SUFFIX_ERRORS = {
    'ට': ['ත', 'ද'],      # dative case marker
    'ගේ': ['ගෙ', 'ට'],    # genitive marker
    'යි': ['ය', 'වි'],     # verbal ending
    'නි': ['න', 'නු'],     # plural marker
    'ක්': ['කු', 'ක'],     # various suffix
    'ම': ['මා', 'මේ'],     # emphasis/case markers
}


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
    # Common punctuation marks to remove
    punctuation = r'[.,!?;:"\'\-\(\)\[\]\{\}<>@#$%^&*+=|\\~`/]'
    
    # Remove punctuation
    cleaned = re.sub(punctuation, '', text)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


# ============================================================================
# ERROR GENERATION FUNCTIONS
# ============================================================================

def apply_spelling_error(word: str) -> str:
    """Apply common spelling errors - character substitution, deletion, insertion, transposition"""
    if len(word) < 2:
        return word

    error_type = random.choice(['substitute', 'delete', 'insert', 'transpose', 'repeat'])

    if error_type == 'substitute':
        # Replace a character with a random Sinhala character
        idx = random.randint(0, len(word) - 1)
        new_char = random.choice(SINHALA_CONSONANTS)
        return word[:idx] + new_char + word[idx + 1:]

    elif error_type == 'delete':
        # Delete a random character
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]

    elif error_type == 'insert':
        # Insert a random character
        idx = random.randint(0, len(word))
        new_char = random.choice(SINHALA_CONSONANTS + ''.join(SINHALA_DIACRITICS[:5]))
        return word[:idx] + new_char + word[idx:]

    elif error_type == 'transpose':
        # Swap two adjacent characters
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]

    elif error_type == 'repeat':
        # Repeat a character (common typo)
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx] + word[idx] + word[idx + 1:]

    return word


def apply_phonetic_error(word: str) -> str:
    """Apply phonetically-motivated errors - sounds that are confused"""
    if len(word) < 1:
        return word

    # Find characters that can be confused phonetically
    confusable_positions = []
    for i, char in enumerate(word):
        if char in SINHALA_PHONETIC_CONFUSIONS:
            confusable_positions.append(i)

    if not confusable_positions:
        # No phonetic confusion possible, apply diacritic confusion instead
        return apply_diacritic_error(word)

    # Pick one position to introduce error
    idx = random.choice(confusable_positions)
    char = word[idx]
    replacement = random.choice(SINHALA_PHONETIC_CONFUSIONS[char])

    return word[:idx] + replacement + word[idx + 1:]


def apply_diacritic_error(word: str) -> str:
    """Apply diacritic/matra errors - common in Sinhala typing"""
    if len(word) < 1:
        return word

    error_type = random.choice(['remove', 'replace', 'add_wrong'])

    if error_type == 'remove':
        # Remove a diacritic
        new_word = ''
        removed = False
        for char in word:
            if char in SINHALA_DIACRITICS and not removed and random.random() < 0.5:
                removed = True
                continue
            new_word += char
        return new_word if new_word else word

    elif error_type == 'replace':
        # Replace a diacritic with a similar one
        new_word = ''
        for char in word:
            if char in DIACRITIC_CONFUSIONS and random.random() < 0.4:
                new_word += random.choice(DIACRITIC_CONFUSIONS[char])
            else:
                new_word += char
        return new_word

    elif error_type == 'add_wrong':
        # Add a wrong diacritic after a consonant
        new_word = ''
        added = False
        for i, char in enumerate(word):
            new_word += char
            # Add diacritic after a consonant (if next char is not already a diacritic)
            if (char in SINHALA_CONSONANTS and
                not added and
                random.random() < 0.3 and
                (i + 1 >= len(word) or word[i + 1] not in SINHALA_DIACRITICS)):
                new_word += random.choice(SINHALA_DIACRITICS[:8])  # Common diacritics
                added = True
        return new_word

    return word


def apply_grammar_error(sentence: str) -> str:
    """Apply grammar-related errors - wrong suffixes, particle errors"""
    words = sentence.split()
    if len(words) < 2:
        return sentence

    error_type = random.choice(['suffix_error', 'word_order', 'particle_drop', 'particle_wrong'])

    if error_type == 'suffix_error':
        # Change suffix of a random word
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        for suffix, wrong_suffixes in GRAMMAR_SUFFIX_ERRORS.items():
            if word.endswith(suffix):
                words[idx] = word[:-len(suffix)] + random.choice(wrong_suffixes)
                break

    elif error_type == 'word_order':
        # Swap two adjacent words
        if len(words) >= 2:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

    elif error_type == 'particle_drop':
        # Drop a short particle word (like ද, ත්, ම)
        for i, word in enumerate(words):
            if len(word) <= 2 and random.random() < 0.3:
                words[i] = ''
                break
        words = [w for w in words if w]

    elif error_type == 'particle_wrong':
        # Add a wrong particle
        idx = random.randint(0, len(words))
        particles = ['ද', 'ත්', 'ම', 'ය', 'යි']
        words.insert(idx, random.choice(particles))

    return ' '.join(words)


def generate_noisy_sentence(clean_sentence: str, num_errors: int = None) -> str:
    """
    Generate a noisy version of a clean Sinhala sentence.

    Args:
        clean_sentence: The original clean sentence
        num_errors: Number of errors to introduce (random 1-3 if None)

    Returns:
        Noisy sentence with spelling, phonetic, and/or grammar errors
    """
    if not clean_sentence or len(clean_sentence.strip()) < 3:
        return clean_sentence

    if num_errors is None:
        num_errors = random.randint(1, 3)

    noisy = clean_sentence
    words = noisy.split()

    if len(words) == 0:
        return clean_sentence

    error_types = ['spelling', 'phonetic', 'diacritic', 'grammar']

    for _ in range(num_errors):
        error_type = random.choice(error_types)

        if error_type == 'grammar':
            # Grammar errors work on the whole sentence
            noisy = apply_grammar_error(noisy)
            words = noisy.split()
        else:
            # Other errors work on individual words
            if len(words) == 0:
                break
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]

            if error_type == 'spelling':
                words[word_idx] = apply_spelling_error(word)
            elif error_type == 'phonetic':
                words[word_idx] = apply_phonetic_error(word)
            elif error_type == 'diacritic':
                words[word_idx] = apply_diacritic_error(word)

            noisy = ' '.join(words)

    return noisy


# ============================================================================
# TEXT CHUNKING FUNCTIONS
# ============================================================================

def split_into_sentences(text: str) -> list:
    """
    Split Sinhala text into sentences using common sentence delimiters.
    Sinhala uses period (.), question mark (?), and other punctuation.
    """
    # Common Sinhala sentence endings
    sentence_pattern = r'[.!?។။၊]|\n\n+'

    # Split by sentence delimiters
    sentences = re.split(sentence_pattern, text)

    # Clean up sentences
    cleaned = []
    for s in sentences:
        s = s.strip()
        # Remove extra whitespace
        s = re.sub(r'\s+', ' ', s)
        if s and len(s) >= 5:  # Minimum sentence length
            cleaned.append(s)

    return cleaned


def chunk_text(text: str, max_words: int = 15, min_words: int = 5) -> list:
    """
    Split text into chunks suitable for training.

    Args:
        text: The input text
        max_words: Maximum words per chunk
        min_words: Minimum words per chunk

    Returns:
        List of text chunks
    """
    chunks = []

    # First, try splitting by sentences
    sentences = split_into_sentences(text)

    for sentence in sentences:
        words = sentence.split()

        if len(words) <= max_words:
            if len(words) >= min_words:
                chunks.append(sentence)
        else:
            # Split long sentences into smaller chunks
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= max_words:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []

            # Add remaining words if they meet minimum
            if len(current_chunk) >= min_words:
                chunks.append(' '.join(current_chunk))
            elif current_chunk and chunks:
                # Append to previous chunk if too small
                chunks[-1] = chunks[-1] + ' ' + ' '.join(current_chunk)

    return chunks


# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

def get_error_count_by_distribution():
    """
    Get number of errors based on a realistic distribution.
    
    Distribution:
    - 10% : 0 errors (clean pairs - helps model learn identity mapping)
    - 25% : 1 error (light errors)
    - 35% : 2 errors (moderate errors)
    - 20% : 3 errors (heavy errors)
    - 10% : 4-5 errors (very noisy - challenging cases)
    
    Returns:
        int: Number of errors to introduce
    """
    rand = random.random()
    
    if rand < 0.10:       # 10% - no errors
        return 0
    elif rand < 0.35:     # 25% - 1 error
        return 1
    elif rand < 0.70:     # 35% - 2 errors
        return 2
    elif rand < 0.90:     # 20% - 3 errors
        return 3
    else:                 # 10% - 4-5 errors
        return random.randint(4, 5)


def generate_training_data(
    input_folder_path: str,
    output_csv_path: str,
    max_words_per_chunk: int = 15,
    min_words_per_chunk: int = 5,
    file_extension: str = '.txt',
    min_errors: int = None,  # Deprecated, kept for backward compatibility
    max_errors: int = None   # Deprecated, kept for backward compatibility
) -> None:
    """
    Generate training data CSV from multiple Sinhala text files in a folder.
    
    Uses a realistic error distribution:
    - 10% clean (0 errors)
    - 25% light (1 error)
    - 35% moderate (2 errors)
    - 20% heavy (3 errors)
    - 10% very noisy (4-5 errors)

    Args:
        input_folder_path: Path to the folder containing text files
        output_csv_path: Path for the output CSV file
        max_words_per_chunk: Maximum words per sentence chunk
        min_words_per_chunk: Minimum words per sentence chunk
        file_extension: File extension to filter (default: '.txt')
        min_errors: (Deprecated) Not used anymore
        max_errors: (Deprecated) Not used anymore
    """
    # Get all text files from the folder
    input_folder = Path(input_folder_path)
    
    if not input_folder.exists():
        print(f"Error: Folder '{input_folder_path}' does not exist!")
        return
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder_path}' is not a directory!")
        return
    
    # Find all text files
    text_files = sorted(input_folder.glob(f'*{file_extension}'))
    
    if not text_files:
        print(f"Error: No '{file_extension}' files found in '{input_folder_path}'!")
        return
    
    print(f"Found {len(text_files)} text file(s) in: {input_folder_path}")
    for tf in text_files:
        print(f"   - {tf.name}")
    
    # Read and combine all text files
    print(f"\nReading all text files...")
    combined_text = []
    total_chars = 0
    
    for text_file in text_files:
        print(f"   Reading: {text_file.name}")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            combined_text.append(text)
            total_chars += len(text)
            print(f"      Characters: {len(text):,}")
    
    text = '\n\n'.join(combined_text)
    print(f"\n   Total characters (combined): {total_chars:,}")

    # Split into chunks
    print(f"\nSplitting into chunks (max {max_words_per_chunk} words, min {min_words_per_chunk} words)...")
    chunks = chunk_text(text, max_words=max_words_per_chunk, min_words=min_words_per_chunk)
    print(f"   Generated {len(chunks):,} chunks")

    # Generate noisy versions and write to CSV
    print(f"\nGenerating noisy sentences with varied error distribution...")
    print(f"   Distribution: 10% clean | 25% light(1) | 35% moderate(2) | 20% heavy(3) | 10% very noisy(4-5)")

    # Ensure output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Track error distribution for statistics
    error_counts = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['noisy_sentence', 'correct_sentence'])

        for i, clean_sentence in enumerate(chunks):
            num_errors = get_error_count_by_distribution()
            
            # Track distribution
            if num_errors >= 4:
                error_counts['4+'] += 1
            else:
                error_counts[num_errors] += 1
            
            if num_errors == 0:
                noisy_sentence = clean_sentence  # Keep it clean
            else:
                noisy_sentence = generate_noisy_sentence(clean_sentence, num_errors=num_errors)

            # Remove punctuation from both sentences
            clean_sentence = remove_punctuation(clean_sentence)
            noisy_sentence = remove_punctuation(noisy_sentence)

            # Only write if sentences are not empty after cleaning
            if clean_sentence and noisy_sentence:
                writer.writerow([noisy_sentence, clean_sentence])

            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1:,}/{len(chunks):,} chunks...")

    # Print summary
    print(f"\nSaved to: {output_csv_path}")
    print(f"\nData generation complete!")
    print(f"   Total pairs: {len(chunks):,}")
    
    # Print error distribution stats
    print(f"\nError Distribution:")
    total = len(chunks)
    for errors, count in error_counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        label = f"{errors} errors" if errors != '4+' else "4+ errors"
        bar = '█' * int(pct / 2)
        print(f"   {label:10}: {count:5} ({pct:5.1f}%) {bar}")

    # Show sample pairs
    print(f"\nSample pairs (showing variety):")
    samples = random.sample(range(len(chunks)), min(8, len(chunks)))
    for idx in samples:
        chunk = chunks[idx]
        num_errors = get_error_count_by_distribution()
        if num_errors == 0:
            noisy = chunk
        else:
            noisy = generate_noisy_sentence(chunk, num_errors=num_errors)
        print(f"\n   [errors={num_errors}]")
        print(f"   Correct: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
        print(f"   Noisy:   {noisy[:50]}{'...' if len(noisy) > 50 else ''}")


def demo_error_types():
    """Demonstrate different error types"""
    print("=" * 60)
    print("ERROR TYPE DEMONSTRATIONS")
    print("=" * 60)

    demo_word = "සිංහල"
    demo_sentence = "මම ඊයේ පාසැල් ගියා"

    print(f"\nOriginal word: '{demo_word}'")
    print("-" * 40)
    print(f"   Spelling error:  '{apply_spelling_error(demo_word)}'")
    print(f"   Phonetic error:  '{apply_phonetic_error(demo_word)}'")
    print(f"   Diacritic error: '{apply_diacritic_error(demo_word)}'")

    print(f"\nOriginal sentence: '{demo_sentence}'")
    print("-" * 40)
    print(f"   Grammar error:   '{apply_grammar_error(demo_sentence)}'")
    print(f"   Full noisy:      '{generate_noisy_sentence(demo_sentence, num_errors=2)}'")

    # Show multiple variations
    print(f"\nMultiple noisy variations of: '{demo_sentence}'")
    print("-" * 40)
    for i in range(5):
        noisy = generate_noisy_sentence(demo_sentence, num_errors=random.randint(1, 2))
        print(f"   {i+1}. {noisy}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate noisy Sinhala training data from clean text files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_noisy_data.py --input-folder data/texts/ --output data/training.csv
    python generate_noisy_data.py --input-folder chunks/ --output out.csv --max-words 20
    python generate_noisy_data.py --input-folder data/ --output out.csv --extension .txt
    python generate_noisy_data.py --demo
        """
    )

    parser.add_argument('--input-folder', '-i', type=str, help='Input folder containing text files')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path')
    parser.add_argument('--extension', '-e', type=str, default='.txt', help='File extension to process (default: .txt)')
    parser.add_argument('--max-words', type=int, default=4, help='Maximum words per chunk (default: 4)')
    parser.add_argument('--min-words', type=int, default=2, help='Minimum words per chunk (default: 2)')
    parser.add_argument('--demo', action='store_true', help='Run demonstration of error types')

    args = parser.parse_args()

    if args.demo:
        demo_error_types()
    elif args.input_folder and args.output:
        generate_training_data(
            input_folder_path=args.input_folder,
            output_csv_path=args.output,
            max_words_per_chunk=args.max_words,
            min_words_per_chunk=args.min_words,
            file_extension=args.extension
        )
    else:
        parser.print_help()
        print("\nPlease provide --input-folder and --output, or use --demo")
