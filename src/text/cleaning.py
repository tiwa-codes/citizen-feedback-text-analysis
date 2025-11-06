"""
Text cleaning and preprocessing for citizen feedback data.

Handles tokenization, normalization, PII masking, and basic text cleaning.
"""
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Common contractions to expand
CONTRACTIONS = {
    "dont": "do not",
    "dnt": "do not",
    "wont": "will not",
    "cant": "can not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "im": "i am",
    "ive": "i have",
    "youre": "you are",
    "youve": "you have",
    "theyre": "they are",
    "theyve": "they have",
    "hes": "he is",
    "shes": "she is",
    "its": "it is",
    "thats": "that is",
    "whats": "what is",
    "wheres": "where is",
}

# Common expletives (mild ones for filtering)
EXPLETIVES = {
    "damn", "hell", "bloody", "stupid", "idiot", "foolish",
    "nonsense", "rubbish", "useless", "bastard"
}


def remove_non_printable(text: str) -> str:
    """
    Remove non-printable characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Keep printable ASCII and common Unicode characters
    return ''.join(char for char in text if char.isprintable() or char.isspace())


def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized punctuation
    """
    # Replace multiple punctuation with single
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def expand_contractions(text: str) -> str:
    """
    Expand common contractions in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with expanded contractions
    """
    words = text.split()
    expanded = []
    
    for word in words:
        word_lower = word.lower()
        # Check if word (without punctuation) is a contraction
        clean_word = word_lower.strip('.,!?;:')
        if clean_word in CONTRACTIONS:
            # Replace but preserve original punctuation
            replacement = CONTRACTIONS[clean_word]
            if word_lower != clean_word:
                # Add back punctuation
                punct = word_lower.replace(clean_word, '')
                replacement = replacement + punct
            expanded.append(replacement)
        else:
            expanded.append(word)
    
    return ' '.join(expanded)


def mask_phone_numbers(text: str) -> str:
    """
    Mask phone numbers in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with masked phone numbers
    """
    # Nigerian phone patterns: +234..., 0..., etc.
    patterns = [
        r'\+234\d{10}',  # +234XXXXXXXXXX
        r'\b0\d{10}\b',   # 0XXXXXXXXXX
        r'\b\d{11}\b',    # 11 digit numbers
        r'\+234[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}',  # With separators
    ]
    
    masked_text = text
    for pattern in patterns:
        masked_text = re.sub(pattern, '[PHONE]', masked_text)
    
    return masked_text


def mask_emails(text: str) -> str:
    """
    Mask email addresses in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with masked emails
    """
    # Email pattern
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(pattern, '[EMAIL]', text)


def mask_national_id(text: str) -> str:
    """
    Mask national ID-like sequences.
    
    Args:
        text: Input text
        
    Returns:
        Text with masked IDs
    """
    # Pattern for ID-like sequences (simple heuristic)
    patterns = [
        r'\b[A-Z]{2}\d{8,11}\b',  # XX12345678
        r'\bNIN[-:\s]?\d{11}\b',  # NIN:12345678901
        r'\bBVN[-:\s]?\d{11}\b',  # BVN:12345678901
    ]
    
    masked_text = text
    for pattern in patterns:
        masked_text = re.sub(pattern, '[ID]', masked_text, flags=re.IGNORECASE)
    
    return masked_text


def detect_url(text: str) -> bool:
    """
    Check if text contains URLs.
    
    Args:
        text: Input text
        
    Returns:
        True if URL detected
    """
    url_pattern = r'https?://\S+|www\.\S+'
    return bool(re.search(url_pattern, text, re.IGNORECASE))


def detect_number(text: str) -> bool:
    """
    Check if text contains numbers.
    
    Args:
        text: Input text
        
    Returns:
        True if number detected
    """
    return bool(re.search(r'\d+', text))


def detect_expletive(text: str) -> bool:
    """
    Check if text contains expletives.
    
    Args:
        text: Input text
        
    Returns:
        True if expletive detected
    """
    text_lower = text.lower()
    return any(expletive in text_lower for expletive in EXPLETIVES)


def tokenize_simple(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace and punctuation.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Split on whitespace and punctuation but keep words
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 0]


def clean_text(text: str, mask_pii: bool = True) -> Tuple[str, Dict]:
    """
    Clean a single text string.
    
    Args:
        text: Input text
        mask_pii: Whether to mask PII
        
    Returns:
        Tuple of (cleaned_text, metadata_dict)
    """
    if pd.isna(text) or not isinstance(text, str):
        return "", {
            'word_count': 0,
            'char_count': 0,
            'has_url': False,
            'has_number': False,
            'contains_expletive': False
        }
    
    original_text = text
    
    # Step 1: Remove non-printable characters
    text = remove_non_printable(text)
    
    # Step 2: Mask PII if requested (demonstrates practice even on synthetic data)
    if mask_pii:
        text = mask_phone_numbers(text)
        text = mask_emails(text)
        text = mask_national_id(text)
    
    # Step 3: Expand contractions
    text = expand_contractions(text)
    
    # Step 4: Normalize punctuation
    text = normalize_punctuation(text)
    
    # Step 5: Basic cleaning
    text = text.strip()
    
    # Calculate metadata from cleaned text
    tokens = tokenize_simple(text)
    metadata = {
        'word_count': len(tokens),
        'char_count': len(text),
        'has_url': detect_url(original_text),
        'has_number': detect_number(text),
        'contains_expletive': detect_expletive(text)
    }
    
    return text, metadata


def clean_text_df(df: pd.DataFrame, text_col: str = 'raw_text') -> pd.DataFrame:
    """
    Clean text in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column to clean
        
    Returns:
        DataFrame with added cleaned_text and metadata columns
    """
    print(f"Cleaning {len(df):,} text records...")
    
    # Apply cleaning
    results = df[text_col].apply(lambda x: clean_text(x, mask_pii=True))
    
    # Extract cleaned text and metadata
    df['cleaned_text'] = results.apply(lambda x: x[0])
    df['word_count'] = results.apply(lambda x: x[1]['word_count'])
    df['char_count'] = results.apply(lambda x: x[1]['char_count'])
    df['has_url'] = results.apply(lambda x: x[1]['has_url'])
    df['has_number'] = results.apply(lambda x: x[1]['has_number'])
    df['contains_expletive'] = results.apply(lambda x: x[1]['contains_expletive'])
    
    # Filter out very short messages (likely spam or noise)
    initial_count = len(df)
    df = df[df['word_count'] >= 2].copy()
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"  Removed {removed_count:,} records with < 2 words")
    
    print(f"✓ Cleaning complete: {len(df):,} records retained")
    
    return df


def main():
    """Main entry point for text cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean citizen feedback text data"
    )
    parser.add_argument(
        '--input',
        '--in',
        dest='input',
        type=str,
        default='data/raw/citizen_feedback.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        '--out',
        dest='output',
        type=str,
        default='data/processed/citizen_feedback_clean.parquet',
        help='Output file path (parquet format)'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        default='raw_text',
        help='Name of text column to clean'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    
    if Path(args.input).is_absolute():
        input_path = Path(args.input)
    else:
        input_path = project_root / args.input
    
    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        output_path = project_root / args.output
    
    # Check input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"Please generate data first:")
        print(f"  python -m src.data.generate_synthetic_feedback")
        return
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} records")
    
    # Clean text
    df_clean = clean_text_df(df, text_col=args.text_col)
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"✓ Saved cleaned data to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cleaning Summary")
    print(f"{'='*60}")
    print(f"Input records:  {len(df):,}")
    print(f"Output records: {len(df_clean):,}")
    print(f"Avg word count: {df_clean['word_count'].mean():.1f}")
    print(f"Has URL:        {df_clean['has_url'].sum():,}")
    print(f"Has number:     {df_clean['has_number'].sum():,}")
    print(f"Has expletive:  {df_clean['contains_expletive'].sum():,}")


if __name__ == '__main__':
    main()
