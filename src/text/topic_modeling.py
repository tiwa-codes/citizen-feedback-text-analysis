"""
Topic modeling for citizen feedback using LDA and NMF.

Provides topic extraction, document-topic assignment, and topic labeling.
"""
import argparse
from typing import List, Dict, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# NLP imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("Warning: NLTK not fully configured. Will use basic preprocessing.")
    nltk = None

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

try:
    from gensim import corpora, models
    from gensim.models import LdaModel, CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: Gensim not available. Will use sklearn only.")
    GENSIM_AVAILABLE = False


def ensure_nltk_data():
    """Ensure NLTK data is downloaded."""
    if nltk is None:
        return False
    
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        print("Downloading required NLTK data...")
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            return True
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            return False


# Initialize NLTK
NLTK_AVAILABLE = ensure_nltk_data()


def load_config() -> Dict:
    """Load analysis configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'analysis_config.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'random_seed': 42,
        'n_topics': 10,
        'min_df': 5,
        'max_df': 0.95,
    }


def get_stopwords() -> set:
    """Get stopwords for text preprocessing."""
    if NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
    else:
        # Basic English stopwords
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
    
    # Add domain-specific stopwords
    domain_stops = {
        'facility', 'service', 'please', 'need', 'want', 'get', 'make',
        'go', 'come', 'see', 'use', 'take', 'give', 'know', 'think',
        'people', 'time', 'way', 'look', 'also', 'back', 'use', 'two',
        'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own',
        'phone', 'email', 'id'  # masked PII tokens
    }
    
    stop_words.update(domain_stops)
    
    return stop_words


class SimplePreprocessor:
    """Simple text preprocessor for topic modeling."""
    
    def __init__(self):
        self.stop_words = get_stopwords()
        self.lemmatizer = None
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except:
                pass
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Tokenize (simple split for now)
        tokens = text.lower().split()
        
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatize if available
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def preprocess_documents(self, texts: List[str]) -> List[List[str]]:
        """Preprocess multiple documents."""
        return [self.preprocess(text) for text in texts]


def fit_sklearn_nmf(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 5000,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, List[str], CountVectorizer]:
    """
    Fit NMF topic model using sklearn.
    
    Args:
        texts: List of text documents
        n_topics: Number of topics
        max_features: Maximum vocabulary size
        random_state: Random seed
        
    Returns:
        Tuple of (model, doc_topic_matrix, feature_names, vectorizer)
    """
    print(f"Fitting NMF with {n_topics} topics...")
    
    config = load_config()
    stop_words = list(get_stopwords())
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=config.get('min_df', 5),
        max_df=config.get('max_df', 0.95),
        stop_words=stop_words,
        ngram_range=(1, 2)
    )
    
    # Fit transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Fit NMF
    nmf_model = NMF(
        n_components=n_topics,
        random_state=random_state,
        init='nndsvd',
        max_iter=400,
        alpha_W=0.1,
        alpha_H=0.1,
        l1_ratio=0.5
    )
    
    doc_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
    
    print(f"✓ NMF model fitted")
    
    return nmf_model, doc_topic_matrix, feature_names, vectorizer


def fit_sklearn_lda(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 5000,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, List[str], CountVectorizer]:
    """
    Fit LDA topic model using sklearn.
    
    Args:
        texts: List of text documents
        n_topics: Number of topics
        max_features: Maximum vocabulary size
        random_state: Random seed
        
    Returns:
        Tuple of (model, doc_topic_matrix, feature_names, vectorizer)
    """
    print(f"Fitting LDA with {n_topics} topics...")
    
    config = load_config()
    stop_words = list(get_stopwords())
    
    # Create count vectorizer for LDA
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=config.get('min_df', 5),
        max_df=config.get('max_df', 0.95),
        stop_words=stop_words,
        ngram_range=(1, 1)
    )
    
    # Fit transform
    count_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Fit LDA
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=50,
        learning_method='online',
        batch_size=128,
        n_jobs=-1
    )
    
    doc_topic_matrix = lda_model.fit_transform(count_matrix)
    
    print(f"✓ LDA model fitted")
    
    return lda_model, doc_topic_matrix, feature_names, vectorizer


def get_top_terms_per_topic(
    model: Any,
    feature_names: List[str],
    n_top_words: int = 10
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extract top terms for each topic.
    
    Args:
        model: Fitted topic model (LDA or NMF)
        feature_names: Feature names from vectorizer
        n_top_words: Number of top words per topic
        
    Returns:
        Dictionary mapping topic index to list of (word, weight) tuples
    """
    topics = {}
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [(feature_names[i], topic[i]) for i in top_indices]
        topics[topic_idx] = top_words
    
    return topics


def get_dominant_topic(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """
    Get dominant topic for each document.
    
    Args:
        doc_topic_matrix: Document-topic probability matrix
        
    Returns:
        Array of dominant topic indices
    """
    return doc_topic_matrix.argmax(axis=1)


def get_representative_documents(
    df: pd.DataFrame,
    doc_topic_matrix: np.ndarray,
    text_col: str = 'cleaned_text',
    n_docs: int = 3
) -> Dict[int, List[str]]:
    """
    Get most representative documents for each topic.
    
    Args:
        df: DataFrame with text
        doc_topic_matrix: Document-topic matrix
        text_col: Name of text column
        n_docs: Number of representative docs per topic
        
    Returns:
        Dictionary mapping topic to list of representative texts
    """
    n_topics = doc_topic_matrix.shape[1]
    representative_docs = {}
    
    for topic_idx in range(n_topics):
        # Get documents with highest probability for this topic
        topic_probs = doc_topic_matrix[:, topic_idx]
        top_doc_indices = topic_probs.argsort()[-n_docs:][::-1]
        
        top_docs = df.iloc[top_doc_indices][text_col].tolist()
        representative_docs[topic_idx] = top_docs
    
    return representative_docs


def fit_topics(
    df: pd.DataFrame,
    text_col: str = 'cleaned_text',
    n_topics: int = None,
    method: str = 'lda',
    random_state: int = None
) -> Dict[str, Any]:
    """
    Fit topic model on DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        n_topics: Number of topics (default from config)
        method: 'lda' or 'nmf'
        random_state: Random seed (default from config)
        
    Returns:
        Dictionary with model, assignments, and metadata
    """
    config = load_config()
    
    if n_topics is None:
        n_topics = config.get('n_topics', 10)
    
    if random_state is None:
        random_state = config.get('random_seed', 42)
    
    print(f"Fitting topic model with method={method}, n_topics={n_topics}")
    print(f"Documents: {len(df):,}")
    
    texts = df[text_col].tolist()
    
    # Fit model
    if method == 'nmf':
        model, doc_topic_matrix, feature_names, vectorizer = fit_sklearn_nmf(
            texts, n_topics=n_topics, random_state=random_state
        )
    else:  # lda
        model, doc_topic_matrix, feature_names, vectorizer = fit_sklearn_lda(
            texts, n_topics=n_topics, random_state=random_state
        )
    
    # Extract topics and terms
    topics = get_top_terms_per_topic(model, feature_names, n_top_words=15)
    
    # Get dominant topic for each document
    dominant_topics = get_dominant_topic(doc_topic_matrix)
    
    # Get representative documents
    representative_docs = get_representative_documents(
        df, doc_topic_matrix, text_col=text_col, n_docs=5
    )
    
    # Calculate simple coherence (average topic-term weight)
    avg_coherence = np.mean([np.mean([w for _, w in terms[:10]]) 
                             for terms in topics.values()])
    
    print(f"✓ Topic modeling complete")
    print(f"  Average coherence proxy: {avg_coherence:.4f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'doc_topic_matrix': doc_topic_matrix,
        'dominant_topics': dominant_topics,
        'topics': topics,
        'representative_docs': representative_docs,
        'feature_names': feature_names,
        'coherence': avg_coherence,
        'method': method,
        'n_topics': n_topics
    }


def save_topic_assignments(
    df: pd.DataFrame,
    topic_result: Dict[str, Any],
    output_path: Path
):
    """
    Save topic assignments to CSV.
    
    Args:
        df: Original DataFrame
        topic_result: Result from fit_topics
        output_path: Output CSV path
    """
    # Create assignments DataFrame
    assignments_df = df.copy()
    
    # Add dominant topic
    assignments_df['dominant_topic'] = topic_result['dominant_topics']
    
    # Add topic probabilities
    doc_topic_matrix = topic_result['doc_topic_matrix']
    for i in range(doc_topic_matrix.shape[1]):
        assignments_df[f'topic_{i}_prob'] = doc_topic_matrix[:, i]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved topic assignments to {output_path}")


def print_topics(topic_result: Dict[str, Any], n_words: int = 10):
    """Print topics with top terms."""
    print(f"\n{'='*80}")
    print(f"Topics ({topic_result['method'].upper()})")
    print(f"{'='*80}")
    
    topics = topic_result['topics']
    representative_docs = topic_result['representative_docs']
    
    for topic_idx, terms in topics.items():
        print(f"\nTopic {topic_idx}:")
        top_terms = ', '.join([word for word, _ in terms[:n_words]])
        print(f"  Top terms: {top_terms}")
        
        print(f"  Representative examples:")
        for i, doc in enumerate(representative_docs[topic_idx][:2], 1):
            print(f"    {i}. {doc[:100]}...")


def main():
    """Main entry point for topic modeling."""
    parser = argparse.ArgumentParser(
        description="Perform topic modeling on citizen feedback"
    )
    parser.add_argument(
        '--input',
        '--in',
        dest='input',
        type=str,
        default='data/processed/citizen_feedback_clean.parquet',
        help='Input parquet file'
    )
    parser.add_argument(
        '--output',
        '--out',
        dest='output',
        type=str,
        default='data/processed/topic_assignments.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--n-topics',
        type=int,
        default=None,
        help='Number of topics (default from config)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['lda', 'nmf'],
        default='lda',
        help='Topic modeling method'
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
    
    # Check input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} records")
    
    # Fit topics
    topic_result = fit_topics(
        df,
        n_topics=args.n_topics,
        method=args.method
    )
    
    # Print topics
    print_topics(topic_result)
    
    # Save assignments
    save_topic_assignments(df, topic_result, output_path)
    
    print(f"\n{'='*80}")
    print(f"Topic modeling complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
