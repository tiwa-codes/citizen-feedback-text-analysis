"""
Sentiment analysis for citizen feedback.

Provides rule-based lexicon sentiment scoring for text.
"""
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import re


# Positive sentiment words (Nigerian English context)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'best',
    'thank', 'thanks', 'grateful', 'appreciate', 'helpful', 'kind', 'nice',
    'professional', 'efficient', 'quick', 'fast', 'clean', 'better', 'improved',
    'happy', 'satisfied', 'pleased', 'love', 'perfect', 'brilliant', 'superb',
    'outstanding', 'commend', 'praise', 'well', 'quality', 'friendly', 'polite',
    'courteous', 'respectful', 'caring', 'attentive', 'dedicated', 'committed'
}

# Negative sentiment words (Nigerian English context)
NEGATIVE_WORDS = {
    'bad', 'poor', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'angry',
    'rude', 'disrespect', 'slow', 'delay', 'long', 'wait', 'dirty', 'filthy',
    'broken', 'problem', 'issue', 'complaint', 'corrupt', 'bribe', 'illegal',
    'shortage', 'lack', 'no', 'not', 'never', 'nothing', 'nowhere', 'none',
    'inadequate', 'insufficient', 'poor', 'substandard', 'negligent', 'careless',
    'unprofessional', 'incompetent', 'useless', 'frustrating', 'disappointed',
    'disappointing', 'unacceptable', 'disgrace', 'shameful', 'neglect', 'ignore',
    'refused', 'deny', 'denied', 'closed', 'unavailable', 'absent', 'missing'
}

# Intensifiers that amplify sentiment
INTENSIFIERS = {
    'very', 'extremely', 'really', 'so', 'too', 'quite', 'absolutely',
    'completely', 'totally', 'highly', 'seriously'
}

# Negation words that flip sentiment
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'none', 'hardly', 'scarcely', 'barely', 'doesnt', 'dont', 'didnt',
    'isnt', 'arent', 'wasnt', 'werent', 'cant', 'cannot', 'wont', 'wouldnt',
    'shouldnt', 'couldnt', 'mightnt'
}


def tokenize_for_sentiment(text: str) -> List[str]:
    """
    Tokenize text for sentiment analysis.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert to lowercase and split
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def compute_lexicon_sentiment(text: str) -> Tuple[float, str, float]:
    """
    Compute sentiment using lexicon-based approach.
    
    This implements a VADER-like heuristic approach:
    - Count positive and negative words
    - Apply intensifiers (boost score)
    - Handle negations (flip sentiment in window)
    - Normalize to [-1, 1] range
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (sentiment_score, sentiment_label, confidence)
    """
    if not text or not isinstance(text, str):
        return 0.0, 'neutral', 0.0
    
    tokens = tokenize_for_sentiment(text)
    
    if len(tokens) == 0:
        return 0.0, 'neutral', 0.0
    
    positive_score = 0.0
    negative_score = 0.0
    
    # Window for negation effect (words after negation)
    negation_window = 3
    negation_active = False
    negation_counter = 0
    
    for i, token in enumerate(tokens):
        # Check if we're in a negation window
        if negation_active:
            negation_counter += 1
            if negation_counter > negation_window:
                negation_active = False
                negation_counter = 0
        
        # Check for negation
        if token in NEGATION_WORDS:
            negation_active = True
            negation_counter = 0
            continue
        
        # Base sentiment value
        sentiment_value = 1.0
        
        # Check for intensifiers in previous position
        if i > 0 and tokens[i-1] in INTENSIFIERS:
            sentiment_value = 1.5
        
        # Score positive/negative words
        if token in POSITIVE_WORDS:
            if negation_active:
                negative_score += sentiment_value  # Flip to negative
            else:
                positive_score += sentiment_value
        
        if token in NEGATIVE_WORDS:
            if negation_active:
                positive_score += sentiment_value  # Flip to positive
            else:
                negative_score += sentiment_value
    
    # Compute compound score
    total_score = positive_score - negative_score
    
    # Normalize by length (but cap the effect)
    norm_factor = len(tokens) ** 0.5
    if norm_factor > 0:
        compound_score = total_score / norm_factor
    else:
        compound_score = 0.0
    
    # Bound to [-1, 1]
    compound_score = max(-1.0, min(1.0, compound_score))
    
    # Determine label based on thresholds
    if compound_score >= 0.05:
        label = 'positive'
    elif compound_score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    # Calculate confidence (based on how far from neutral)
    confidence = abs(compound_score)
    
    return compound_score, label, confidence


def compute_simple_polarity(text: str) -> Tuple[float, str]:
    """
    Compute simple polarity by counting positive vs negative words.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (polarity_score, polarity_label)
    """
    if not text or not isinstance(text, str):
        return 0.0, 'neutral'
    
    tokens = tokenize_for_sentiment(text)
    
    if len(tokens) == 0:
        return 0.0, 'neutral'
    
    pos_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    
    total_sentiment_words = pos_count + neg_count
    
    if total_sentiment_words == 0:
        return 0.0, 'neutral'
    
    # Polarity: -1 (all negative) to 1 (all positive)
    polarity = (pos_count - neg_count) / total_sentiment_words
    
    if polarity > 0.2:
        label = 'positive'
    elif polarity < -0.2:
        label = 'negative'
    else:
        label = 'neutral'
    
    return polarity, label


def compute_sentiment(df: pd.DataFrame, text_col: str = 'cleaned_text') -> pd.DataFrame:
    """
    Compute sentiment for all texts in DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        
    Returns:
        DataFrame with added sentiment columns
    """
    print(f"Computing sentiment for {len(df):,} records...")
    
    # Apply lexicon-based sentiment
    sentiment_results = df[text_col].apply(compute_lexicon_sentiment)
    
    df['sentiment_score'] = sentiment_results.apply(lambda x: x[0])
    df['sentiment_label'] = sentiment_results.apply(lambda x: x[1])
    df['sentiment_confidence'] = sentiment_results.apply(lambda x: x[2])
    
    # Also compute simple polarity as alternative
    polarity_results = df[text_col].apply(compute_simple_polarity)
    df['polarity_score'] = polarity_results.apply(lambda x: x[0])
    df['polarity_label'] = polarity_results.apply(lambda x: x[1])
    
    print(f"✓ Sentiment computed")
    print(f"  Positive: {(df['sentiment_label'] == 'positive').sum():,}")
    print(f"  Neutral:  {(df['sentiment_label'] == 'neutral').sum():,}")
    print(f"  Negative: {(df['sentiment_label'] == 'negative').sum():,}")
    
    return df


def get_sentiment_summary(df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
    """
    Get sentiment summary statistics.
    
    Args:
        df: DataFrame with sentiment columns
        group_by: Optional column to group by
        
    Returns:
        Summary DataFrame
    """
    if group_by:
        summary = df.groupby(group_by).agg({
            'sentiment_score': ['mean', 'std'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x),
        }).round(3)
        summary.columns = ['mean_sentiment', 'std_sentiment', 'positive_ratio']
    else:
        summary = pd.DataFrame({
            'mean_sentiment': [df['sentiment_score'].mean()],
            'std_sentiment': [df['sentiment_score'].std()],
            'positive_ratio': [(df['sentiment_label'] == 'positive').sum() / len(df)],
            'negative_ratio': [(df['sentiment_label'] == 'negative').sum() / len(df)],
            'neutral_ratio': [(df['sentiment_label'] == 'neutral').sum() / len(df)],
        }).round(3)
    
    return summary


def main():
    """Example usage of sentiment analysis."""
    from pathlib import Path
    
    # Load cleaned data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'processed' / 'citizen_feedback_clean.parquet'
    
    if not data_path.exists():
        print(f"Error: Cleaned data not found at {data_path}")
        print("Please run cleaning first: python -m src.text.cleaning")
        return
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} records")
    
    # Compute sentiment
    df = compute_sentiment(df)
    
    # Save with sentiment
    output_path = data_path.parent / 'citizen_feedback_with_sentiment.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Sentiment Summary")
    print(f"{'='*60}")
    
    summary = get_sentiment_summary(df)
    print(summary)
    
    # Summary by channel
    print(f"\n{'='*60}")
    print(f"Sentiment by Channel")
    print(f"{'='*60}")
    
    channel_summary = get_sentiment_summary(df, group_by='channel')
    print(channel_summary)
    
    # Show examples
    print(f"\n{'='*60}")
    print(f"Example Sentiments")
    print(f"{'='*60}")
    
    for label in ['positive', 'negative', 'neutral']:
        print(f"\n{label.upper()} examples:")
        examples = df[df['sentiment_label'] == label].head(3)
        for idx, row in examples.iterrows():
            print(f"  [{row['sentiment_score']:.2f}] {row['cleaned_text'][:100]}...")


if __name__ == '__main__':
    main()
