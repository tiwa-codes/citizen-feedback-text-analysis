"""
Feature extraction for text analysis.

Provides TF-IDF vectorization, keyword extraction, and lexical diversity measures.
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import yaml
from pathlib import Path


def load_config() -> Dict:
    """Load analysis configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'analysis_config.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'min_df': 5,
        'max_df': 0.95,
        'n_features_tfidf': 20000,
        'ngram_range': [1, 2]
    }


def compute_tfidf_features(
    texts: List[str],
    min_df: int = None,
    max_df: float = None,
    max_features: int = None,
    ngram_range: Tuple[int, int] = None
) -> Tuple[np.ndarray, TfidfVectorizer, List[str]]:
    """
    Compute TF-IDF features for text documents.
    
    Args:
        texts: List of text documents
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        max_features: Maximum number of features
        ngram_range: N-gram range (e.g., (1, 2) for unigrams and bigrams)
        
    Returns:
        Tuple of (tfidf_matrix, vectorizer, feature_names)
    """
    # Load defaults from config if not provided
    config = load_config()
    if min_df is None:
        min_df = config.get('min_df', 5)
    if max_df is None:
        max_df = config.get('max_df', 0.95)
    if max_features is None:
        max_features = config.get('n_features_tfidf', 20000)
    if ngram_range is None:
        ngram_range = tuple(config.get('ngram_range', [1, 2]))
    
    print(f"Computing TF-IDF features...")
    print(f"  min_df={min_df}, max_df={max_df}, max_features={max_features}")
    print(f"  ngram_range={ngram_range}")
    
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return tfidf_matrix, vectorizer, feature_names


def get_top_keywords(
    tfidf_matrix: np.ndarray,
    feature_names: List[str],
    top_n: int = 20
) -> List[Tuple[str, float]]:
    """
    Extract top keywords by TF-IDF score.
    
    Args:
        tfidf_matrix: TF-IDF matrix
        feature_names: Feature names
        top_n: Number of top keywords to return
        
    Returns:
        List of (keyword, score) tuples
    """
    # Sum TF-IDF scores across all documents
    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    
    # Get top indices
    top_indices = np.argsort(tfidf_sum)[::-1][:top_n]
    
    top_keywords = [(feature_names[i], tfidf_sum[i]) for i in top_indices]
    
    return top_keywords


def get_keywords_by_group(
    df: pd.DataFrame,
    text_col: str,
    group_col: str,
    top_n: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract top keywords for each group (e.g., by state or time period).
    
    Args:
        df: DataFrame with text and grouping column
        text_col: Name of text column
        group_col: Name of grouping column
        top_n: Number of top keywords per group
        
    Returns:
        Dictionary mapping group names to list of (keyword, score) tuples
    """
    keywords_by_group = {}
    
    for group_name, group_df in df.groupby(group_col):
        if len(group_df) < 10:  # Skip very small groups
            continue
        
        texts = group_df[text_col].tolist()
        
        try:
            tfidf_matrix, vectorizer, feature_names = compute_tfidf_features(
                texts,
                min_df=min(2, len(texts) // 10),
                max_features=1000
            )
            
            keywords = get_top_keywords(tfidf_matrix, feature_names, top_n=top_n)
            keywords_by_group[group_name] = keywords
        except Exception as e:
            print(f"Warning: Could not extract keywords for {group_name}: {e}")
            continue
    
    return keywords_by_group


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (type-token ratio).
    
    Args:
        text: Input text
        
    Returns:
        Lexical diversity score (0-1)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    
    return diversity


def compute_text_statistics(df: pd.DataFrame, text_col: str = 'cleaned_text') -> pd.DataFrame:
    """
    Compute text statistics for DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        
    Returns:
        DataFrame with added statistics columns
    """
    print(f"Computing text statistics for {len(df):,} records...")
    
    # Lexical diversity
    df['lexical_diversity'] = df[text_col].apply(compute_lexical_diversity)
    
    # Already have word_count and char_count from cleaning
    # Add average word length
    df['avg_word_length'] = df.apply(
        lambda row: row['char_count'] / row['word_count'] if row['word_count'] > 0 else 0,
        axis=1
    )
    
    print(f"✓ Statistics computed")
    
    return df


def extract_keywords_from_text(text: str, top_n: int = 5) -> List[str]:
    """
    Extract simple keywords from a single text (most frequent words).
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    if not text or not isinstance(text, str):
        return []
    
    # Simple word frequency
    words = [w.lower() for w in text.split() if len(w) > 3]
    
    # Remove common stop words manually (basic list)
    stop_words = {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from', 
                  'have', 'has', 'was', 'were', 'been', 'they', 'their', 'what',
                  'when', 'where', 'which', 'will', 'would', 'could', 'should'}
    
    words = [w for w in words if w not in stop_words]
    
    # Get most common
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(top_n)]
    
    return top_words


def main():
    """Example usage of feature extraction."""
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
    
    # Compute text statistics
    df = compute_text_statistics(df)
    
    # Extract TF-IDF features
    texts = df['cleaned_text'].tolist()
    tfidf_matrix, vectorizer, feature_names = compute_tfidf_features(texts)
    
    # Get top keywords
    top_keywords = get_top_keywords(tfidf_matrix, feature_names, top_n=20)
    
    print(f"\n{'='*60}")
    print(f"Top 20 Keywords")
    print(f"{'='*60}")
    for i, (keyword, score) in enumerate(top_keywords, 1):
        print(f"{i:2d}. {keyword:30s} {score:10.2f}")
    
    # Keywords by state (top 5 states by feedback count)
    top_states = df['state'].value_counts().head(5).index.tolist()
    df_top_states = df[df['state'].isin(top_states)]
    
    print(f"\n{'='*60}")
    print(f"Keywords by Top States")
    print(f"{'='*60}")
    
    keywords_by_state = get_keywords_by_group(
        df_top_states, 
        text_col='cleaned_text',
        group_col='state',
        top_n=5
    )
    
    for state, keywords in keywords_by_state.items():
        print(f"\n{state}:")
        for keyword, score in keywords[:5]:
            print(f"  - {keyword}: {score:.2f}")


if __name__ == '__main__':
    main()
