"""
Dyslexic Chunk Generator (Sinhala-focused)
=========================================
Reads clean_chunk from CSV and generates dyslexic/noisy versions by adding
common Sinhala digitized-text errors (phonetic, shape, diacritics, prenasalized,
encoding artifacts, merge/split) + generic spelling ops.

Upgraded to include common Sinhala errors described in:
"Sinhala Spell Correction - A Novel Benchmark with Neural Spell Correction"
(esp. Table 3 + Table 5 error categories).

Usage:
    python generate_dyslexic_chunks.py --input data/small_english_chunks.csv --output data/dyslexic_chunks.csv
"""

import random
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# HELPERS
# =============================================================================

def contains_sinhala(text: str) -> bool:
    """Check if text contains Sinhala characters (Unicode range U+0D80..U+0DFF)."""
    return bool(re.search(r'[\u0D80-\u0DFF]', text))


def build_confusion_map_from_sets(sets: List[List[str]]) -> Dict[str, List[str]]:
    """
    Build a symmetric confusion map from list of interchangeable sets.
    Example: [['න','ණ'], ['ස','ශ','ෂ']] -> {'න':['ණ'], 'ණ':['න'], 'ස':['ශ','ෂ'], ...}
    """
    conf: Dict[str, List[str]] = {}
    for group in sets:
        for ch in group:
            conf[ch] = [x for x in group if x != ch]
    return conf


# =============================================================================
# SINHALA CHARACTER GROUPS (from paper: Table 5 + described operations)
# =============================================================================

# Similar sounding / rule-based confusions (Table 5)
SIMILAR_SOUNDING_SETS = [
    ['ෂ', 'ශ', 'ස'],
    ['න', 'ණ'],
    ['ල', 'ළ'],
    ['ඟ', 'ග'],
    ['ඦ', 'ජ'],
    ['ඬ', 'ඩ'],
    ['ඳ', 'ද'],
    ['ඹ', 'බ'],
    ['ධ', 'ද'],
    ['ඛ', 'ක'],
    ['ඝ', 'ග'],
    ['ඡ', 'ච'],
    ['ඨ', 'ට'],
    ['භ', 'බ'],
    ['ඵ', 'ප'],
    ['ඩ', 'ඪ'],
    ['ථ', 'ත'],
    ['ජ', 'ඣ'],
]

# Similar shape (Table 5) + mentioned set {ඔ, ඹ} in the text
SIMILAR_SHAPE_SETS = [
    ['ඞ', 'ඬ', 'ඩ'],
    ['ඡ', 'ජ'],
    ['ඔ', 'ඹ'],     # stated in the paper text as a similar-shaped confusion set
]

# NOTE: {ඍ, සෘ} includes a multi-char form (ස + ◌ෘ)
# We'll handle this with special multi-pattern replacements below.

# Diacritics list (keep yours + add Sinhala "ෘ" and additional signs used in the paper tokens)
SINHALA_DIACRITICS = [
    'ා', 'ැ', 'ෑ', 'ි', 'ී', 'ු', 'ූ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ං', 'ඃ', '්', 'ෘ', 'ෟ', 'ෳ', 'ෲ'
]

# Similar diacritics (Table 5 sets, simplified to practical char-level swaps)
DIACRITIC_CONFUSIONS = {
    'ැ': ['ෑ'],
    'ෑ': ['ැ'],
    'ු': ['ූ'],
    'ූ': ['ු'],
    'ි': ['ී'],
    'ී': ['ි'],
    'ෙ': ['ේ'],
    'ේ': ['ෙ'],
    'ො': ['ෝ', 'ෞ'],
    'ෝ': ['ො'],
    'ෞ': ['ො'],
    # paper mentions sets like {◌ෟ, ◌ෳ} (rare), keep them swappable
    'ෟ': ['ෳ'],
    'ෳ': ['ෟ'],
}

# Consonants (same as yours)
SINHALA_CONSONANTS = 'කඛගඝඞචඡජඣඤටඨඩඪණතථදධනපඵබභමයරලවශෂසහළ'

# Build single-char confusion maps from sets
SOUND_CONFUSIONS = build_confusion_map_from_sets(SIMILAR_SOUNDING_SETS)
SHAPE_CONFUSIONS = build_confusion_map_from_sets(SIMILAR_SHAPE_SETS)

# Prenasalized vs obstruent sets (explicitly emphasized in paper text)
PRENASALIZED_SETS = [
    ['ඟ', 'ග'],
    ['ඦ', 'ජ'],
    ['ඬ', 'ඩ'],
    ['ඳ', 'ද'],
    ['ඹ', 'බ'],
]
PRENASALIZED_CONFUSIONS = build_confusion_map_from_sets(PRENASALIZED_SETS)

# Special multi-char / rendering related swaps (paper’s {ඍ, සෘ})
# - 'ඍ' (single codepoint vowel) <-> 'සෘ' (two codepoints: 'ස' + 'ෘ')
MULTI_CHAR_CONFUSIONS: List[Tuple[str, str]] = [
    ('ඍ', 'සෘ'),
    ('සෘ', 'ඍ'),
]


# =============================================================================
# ENCODING / RENDERING ERRORS (from paper Table 3 + text examples)
# =============================================================================

# Table 3 style decompositions and artifacts:
# - 'ආ' may appear as "අා"
# - 'ඇ' as "අැ"
# - 'ඈ' as "අෑ"
# - 'ඒ' as "එ්"
# - backtick artifacts for "ළු": "`ඵ" / "`එ" / "`ථ"
ENCODING_SWAPS: List[Tuple[str, str]] = [
    ('ආ', 'අා'),
    ('ඇ', 'අැ'),
    ('ඈ', 'අෑ'),
    ('ඒ', 'එ්'),
    ('ළු', '`ඵ'),
    ('ළු', '`එ'),
    ('ළු', '`ථ'),
]

# Sometimes you want the reverse as well (to simulate normalization going either way)
ENCODING_SWAPS += [(b, a) for (a, b) in ENCODING_SWAPS if a not in ['`ඵ', '`එ', '`ථ']]


def apply_sinhala_encoding_error(token: str) -> str:
    """
    Apply encoding/rendering noise:
    - Replace a substring using ENCODING_SWAPS
    - Or insert stray virama "්" at end (e.g., ඔවුන්්)
    - Or duplicate a diacritic (as mentioned in paper text)
    """
    if not token:
        return token

    ops = ['swap_pattern', 'append_virama', 'dup_diacritic']
    op = random.choice(ops)

    if op == 'swap_pattern':
        candidates = [(a, b) for (a, b) in ENCODING_SWAPS if a in token]
        if not candidates:
            # also try multi-char confusions if present
            for a, b in MULTI_CHAR_CONFUSIONS:
                if a in token:
                    return token.replace(a, b, 1)
            return token
        a, b = random.choice(candidates)
        return token.replace(a, b, 1)

    if op == 'append_virama':
        # Add virama at the end if not already there (paper shows this type)
        if token.endswith('්'):
            return token
        return token + '්'

    if op == 'dup_diacritic':
        # Duplicate a diacritic inside the token (paper mentions repeated diacritics can be hard to notice)
        diacs = [c for c in token if c in SINHALA_DIACRITICS]
        if not diacs:
            return token
        d = random.choice(diacs)
        idx = token.find(d)
        return token[:idx] + d + token[idx:]  # duplicate once

    return token


# =============================================================================
# SINHALA ERROR GENERATION FUNCTIONS
# =============================================================================

def apply_sinhala_spelling_error(word: str) -> str:
    """Generic spelling ops (substitute/delete/insert/transpose/repeat)."""
    if len(word) < 2:
        return word

    error_type = random.choice(['substitute', 'delete', 'insert', 'transpose', 'repeat'])

    if error_type == 'substitute':
        idx = random.randint(0, len(word) - 1)
        new_char = random.choice(SINHALA_CONSONANTS)
        return word[:idx] + new_char + word[idx + 1:]

    elif error_type == 'delete':
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]

    elif error_type == 'insert':
        idx = random.randint(0, len(word))
        new_char = random.choice(SINHALA_CONSONANTS + ''.join(SINHALA_DIACRITICS))
        return word[:idx] + new_char + word[idx:]

    elif error_type == 'transpose':
        if len(word) >= 2:
            idx = random.randint(0, len(word) - 2)
            return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]

    elif error_type == 'repeat':
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx] + word[idx] + word[idx + 1:]

    return word


def apply_sinhala_rule_based_confusion(word: str) -> str:
    """
    Apply Sinhala-specific confusions from Table 5:
    - similar sounding groups
    - similar shape groups
    - multi-char confusion {ඍ, සෘ}
    """
    if not word:
        return word

    # Try multi-char first (important for {ඍ, සෘ})
    multi_candidates = [(a, b) for (a, b) in MULTI_CHAR_CONFUSIONS if a in word]
    if multi_candidates and random.random() < 0.6:
        a, b = random.choice(multi_candidates)
        return word.replace(a, b, 1)

    # Collect confusable positions (sound/shape)
    positions = []
    for i, ch in enumerate(word):
        if ch in SOUND_CONFUSIONS or ch in SHAPE_CONFUSIONS:
            positions.append(i)

    if not positions:
        return word

    idx = random.choice(positions)
    ch = word[idx]

    candidates = []
    if ch in SOUND_CONFUSIONS:
        candidates.extend(SOUND_CONFUSIONS[ch])
    if ch in SHAPE_CONFUSIONS:
        candidates.extend(SHAPE_CONFUSIONS[ch])

    if not candidates:
        return word

    rep = random.choice(list(set(candidates)))
    return word[:idx] + rep + word[idx + 1:]


def apply_sinhala_prenasalized_error(word: str) -> str:
    """Confuse prenasalized consonants with their obstruent pairs (paper category D)."""
    if not word:
        return word

    positions = [i for i, ch in enumerate(word) if ch in PRENASALIZED_CONFUSIONS]
    if not positions:
        return word

    idx = random.choice(positions)
    ch = word[idx]
    rep = random.choice(PRENASALIZED_CONFUSIONS[ch])
    return word[:idx] + rep + word[idx + 1:]


def apply_sinhala_diacritic_error(word: str) -> str:
    """Apply diacritic errors (paper category F) using confusion sets."""
    if not word:
        return word

    error_type = random.choice(['remove', 'replace', 'add_wrong'])

    if error_type == 'remove':
        new_word = ''
        removed = False
        for char in word:
            if char in SINHALA_DIACRITICS and not removed and random.random() < 0.6:
                removed = True
                continue
            new_word += char
        return new_word if new_word else word

    elif error_type == 'replace':
        new_word = ''
        replaced = False
        for char in word:
            if (char in DIACRITIC_CONFUSIONS) and (random.random() < 0.6) and (not replaced):
                new_word += random.choice(DIACRITIC_CONFUSIONS[char])
                replaced = True
            else:
                new_word += char
        return new_word

    elif error_type == 'add_wrong':
        # add a vowel sign after a consonant that currently has none
        for i in range(len(word) - 1):
            if word[i] in SINHALA_CONSONANTS and word[i + 1] not in SINHALA_DIACRITICS:
                if random.random() < 0.5:
                    d = random.choice(['ා', 'ැ', 'ි', 'ු', 'ෙ', 'ො', 'ෘ'])
                    return word[:i + 1] + d + word[i + 1:]
        # fallback: append a diacritic
        return word + random.choice(['ා', 'ැ', 'ි', 'ු'])

    return word


# =============================================================================
# MERGE / SPLIT ERRORS (paper category G) - CHUNK LEVEL
# =============================================================================

def apply_merge_split_error_to_chunk(chunk: str) -> str:
    """
    Introduce word-boundary mistakes:
    - merge two adjacent tokens
    - split one token into two
    """
    words = chunk.split()
    if len(words) < 2 and len(chunk) < 4:
        return chunk

    if len(words) >= 2 and random.random() < 0.55:
        # MERGE: pick adjacent words and remove the space
        i = random.randint(0, len(words) - 2)
        merged = words[:i] + [words[i] + words[i + 1]] + words[i + 2:]
        return ' '.join(merged)

    # SPLIT: pick a word and split inside it
    if len(words) == 0:
        return chunk
    i = random.randint(0, len(words) - 1)
    w = words[i]
    if len(w) < 4:
        return chunk  # too small to split nicely

    cut = random.randint(1, len(w) - 2)
    words[i:i + 1] = [w[:cut], w[cut:]]
    return ' '.join(words)


# =============================================================================
# UNIFIED ERROR APPLICATION
# =============================================================================

# Weight error-type selection to resemble observed frequency trends:
# - diacritics are very common in digitized Sinhala
# - insertion/deletion/substitution also common
# - rule-based confusions (sound/shape/prenasalized) common
# - encoding noise occurs but less frequent
ERROR_TYPE_WEIGHTS: List[Tuple[str, float]] = [
    ('diacritic', 0.32),
    ('spelling',  0.18),
    ('rule',      0.22),
    ('prenasal',  0.12),
    ('encoding',  0.10),
    ('diacritic2',0.06),  # small extra bucket to keep diacritics prominent
]

def weighted_choice(weighted: List[Tuple[str, float]]) -> str:
    r = random.random()
    upto = 0.0
    for k, w in weighted:
        upto += w
        if r <= upto:
            return k
    return weighted[-1][0]


def apply_error_to_word(word: str) -> str:
    """Apply appropriate Sinhala error to a word token."""
    if not contains_sinhala(word):
        return word  # non-Sinhala unchanged

    et = weighted_choice(ERROR_TYPE_WEIGHTS)

    if et in ('diacritic', 'diacritic2'):
        return apply_sinhala_diacritic_error(word)
    if et == 'spelling':
        return apply_sinhala_spelling_error(word)
    if et == 'rule':
        return apply_sinhala_rule_based_confusion(word)
    if et == 'prenasal':
        return apply_sinhala_prenasalized_error(word)
    if et == 'encoding':
        return apply_sinhala_encoding_error(word)

    return word


def generate_dyslexic_chunk(clean_chunk: str, num_errors: Optional[int] = None) -> str:
    """
    Generate a dyslexic/noisy version of a clean chunk.

    Each "error" can be:
    - word-level Sinhala noise
    - or chunk-level merge/split (with small probability)
    """
    if not clean_chunk or len(clean_chunk.strip()) < 2:
        return clean_chunk

    if num_errors is None:
        num_errors = get_error_count_by_distribution()

    if num_errors == 0:
        return clean_chunk

    dyslexic = clean_chunk

    for _ in range(num_errors):
        # Occasionally apply merge/split at chunk-level (paper category G)
        if random.random() < 0.12:
            dyslexic = apply_merge_split_error_to_chunk(dyslexic)
            continue

        words = dyslexic.split()
        if not words:
            break

        word_idx = random.randint(0, len(words) - 1)
        words[word_idx] = apply_error_to_word(words[word_idx])
        dyslexic = ' '.join(words)

    return dyslexic


# =============================================================================
# ERROR DISTRIBUTION (keep your original)
# =============================================================================

def get_error_count_by_distribution() -> int:
    rand = random.random()

    if rand < 0.10:
        return 0
    elif rand < 0.35:
        return 1
    elif rand < 0.70:
        return 2
    elif rand < 0.90:
        return 3
    else:
        return random.randint(4, 5)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def generate_dyslexic_data(input_csv_path: str, output_csv_path: str) -> None:
    input_path = Path(input_csv_path)

    if not input_path.exists():
        print(f"Error: Input file '{input_csv_path}' does not exist!")
        return

    print(f"Reading input file: {input_csv_path}")

    clean_chunks = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_chunk = row.get('clean_chunk', '')
            if clean_chunk:
                clean_chunks.append(clean_chunk)

    print(f"   Loaded {len(clean_chunks):,} clean chunks")

    print("\nGenerating dyslexic chunks with varied error distribution...")
    print("   Distribution: 10% clean | 25% (1 error) | 35% (2 errors) | 20% (3 errors) | 10% (4-5 errors)")
    print("   Error types: Sinhala rule confusions (sound/shape/prenasal), diacritics, encoding noise, merge/split, + generic ops")

    error_counts = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    results = []

    for i, clean_chunk in enumerate(clean_chunks):
        num_errors = get_error_count_by_distribution()

        if num_errors >= 4:
            error_counts['4+'] += 1
        else:
            error_counts[num_errors] += 1

        dyslexic_chunk = generate_dyslexic_chunk(clean_chunk, num_errors=num_errors)
        results.append((dyslexic_chunk, clean_chunk))

        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1:,}/{len(clean_chunks):,} chunks...")

    print(f"\nWriting output to: {output_csv_path}")

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dyslexic_chunk', 'clean_chunk'])
        for dyslexic_chunk, clean_chunk in results:
            writer.writerow([dyslexic_chunk, clean_chunk])

    print(f"\nDone! Generated {len(results):,} dyslexic chunks")

    print("\nError Distribution:")
    total = len(results)
    for errors, count in error_counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        label = f"{errors} errors" if errors != '4+' else "4+ errors"
        bar = '█' * int(pct / 2)
        print(f"   {label:10}: {count:5} ({pct:5.1f}%) {bar}")

    print("\nSample pairs:")
    samples = random.sample(results, min(10, len(results)))
    for dyslexic, clean in samples:
        print(f"   Clean:    {clean}")
        print(f"   Dyslexic: {dyslexic}")
        print()


def demo_error_types():
    print("=" * 60)
    print("ERROR TYPE DEMONSTRATIONS")
    print("=" * 60)

    sinhala_word = "සිංහල"
    print(f"\nSinhala word: '{sinhala_word}'")
    print("-" * 40)
    print(f"   Generic spelling: '{apply_sinhala_spelling_error(sinhala_word)}'")
    print(f"   Rule confusion:   '{apply_sinhala_rule_based_confusion(sinhala_word)}'")
    print(f"   Prenasalized:     '{apply_sinhala_prenasalized_error(sinhala_word)}'")
    print(f"   Diacritic:        '{apply_sinhala_diacritic_error(sinhala_word)}'")
    print(f"   Encoding:         '{apply_sinhala_encoding_error(sinhala_word)}'")

    mixed_chunk = "මේ computer එක ගැන"
    print(f"\nMixed chunk: '{mixed_chunk}'")
    print("-" * 40)
    for i in range(5):
        num_errors = random.randint(1, 3)
        dyslexic = generate_dyslexic_chunk(mixed_chunk, num_errors=num_errors)
        print(f"   [{num_errors} errors]: {dyslexic}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dyslexic/noisy versions of clean chunks with Sinhala-specific error types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_dyslexic_chunks.py --input data/small_english_chunks.csv --output data/dyslexic_chunks.csv
    python generate_dyslexic_chunks.py --demo
        """
    )

    parser.add_argument('--input', '-i', type=str, help='Input CSV file path (expects column: clean_chunk)')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path')
    parser.add_argument('--demo', action='store_true', help='Run demonstration of error types')

    args = parser.parse_args()

    if args.demo:
        demo_error_types()
    elif args.input and args.output:
        generate_dyslexic_data(args.input, args.output)
    else:
        parser.print_help()
        print("\nPlease provide --input and --output, or use --demo")
