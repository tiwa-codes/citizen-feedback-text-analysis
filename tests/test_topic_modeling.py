"""
Unit tests for topic modeling module.
"""
import pytest
import pandas as pd
import numpy as np
from src.text.topic_modeling import (
    get_stopwords,
    SimplePreprocessor,
    fit_sklearn_lda,
    fit_sklearn_nmf,
    get_top_terms_per_topic,
    get_dominant_topic,
    get_representative_documents,
    fit_topics
)


class TestStopwords:
    """Test stopword functionality."""
    
    def test_get_stopwords(self):
        """Test getting stopwords."""
        stopwords = get_stopwords()
        
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert 'the' in stopwords
        assert 'and' in stopwords


class TestPreprocessor:
    """Test text preprocessor."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = SimplePreprocessor()
        
        assert preprocessor.stop_words is not None
        assert len(preprocessor.stop_words) > 0
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        preprocessor = SimplePreprocessor()
        text = "The hospital is very good and the staff are helpful"
        
        tokens = preprocessor.preprocess(text)
        
        assert isinstance(tokens, list)
        assert 'hospital' in tokens
        # Stopwords should be removed
        assert 'the' not in tokens
        assert 'and' not in tokens
        # Short tokens should be removed
        assert 'is' not in tokens
    
    def test_preprocess_empty(self):
        """Test preprocessing empty text."""
        preprocessor = SimplePreprocessor()
        tokens = preprocessor.preprocess("")
        
        assert tokens == []
    
    def test_preprocess_documents(self):
        """Test preprocessing multiple documents."""
        preprocessor = SimplePreprocessor()
        texts = [
            "The hospital is good",
            "The school needs improvement",
            "Water supply is broken"
        ]
        
        token_lists = preprocessor.preprocess_documents(texts)
        
        assert len(token_lists) == 3
        assert all(isinstance(tokens, list) for tokens in token_lists)


class TestTopicModeling:
    """Test topic modeling functions."""
    
    @pytest.fixture
    def sample_texts(self):
        """Fixture providing sample texts."""
        return [
            "The hospital staff are rude and unprofessional",
            "Water supply is not working in our area",
            "School building needs urgent repair",
            "Health center has no medicine available",
            "Thank you for the excellent service at the clinic",
            "The road to the market is in bad condition",
            "No electricity at the school for weeks",
            "Police station needs more officers",
            "Water borehole is broken and needs fixing",
            "Staff at the hospital are very helpful and kind"
        ] * 5  # Repeat to have enough data
    
    def test_fit_sklearn_lda(self, sample_texts):
        """Test LDA topic modeling."""
        model, doc_topic_matrix, feature_names, vectorizer = fit_sklearn_lda(
            sample_texts,
            n_topics=3,
            max_features=100,
            random_state=42
        )
        
        assert model is not None
        assert doc_topic_matrix.shape[0] == len(sample_texts)
        assert doc_topic_matrix.shape[1] == 3  # n_topics
        assert len(feature_names) > 0
        
        # Check that probabilities sum to 1 (approximately)
        row_sums = doc_topic_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_fit_sklearn_nmf(self, sample_texts):
        """Test NMF topic modeling."""
        model, doc_topic_matrix, feature_names, vectorizer = fit_sklearn_nmf(
            sample_texts,
            n_topics=3,
            max_features=100,
            random_state=42
        )
        
        assert model is not None
        assert doc_topic_matrix.shape[0] == len(sample_texts)
        assert doc_topic_matrix.shape[1] == 3  # n_topics
        assert len(feature_names) > 0
        
        # Check non-negative
        assert np.all(doc_topic_matrix >= 0)
    
    def test_get_top_terms_per_topic(self, sample_texts):
        """Test extracting top terms per topic."""
        model, _, feature_names, _ = fit_sklearn_lda(
            sample_texts,
            n_topics=3,
            random_state=42
        )
        
        topics = get_top_terms_per_topic(model, feature_names, n_top_words=5)
        
        assert len(topics) == 3  # n_topics
        for topic_idx, terms in topics.items():
            assert len(terms) == 5  # n_top_words
            assert all(isinstance(term, str) for term, _ in terms)
            assert all(isinstance(weight, (int, float)) for _, weight in terms)
    
    def test_get_dominant_topic(self, sample_texts):
        """Test getting dominant topic."""
        _, doc_topic_matrix, _, _ = fit_sklearn_lda(
            sample_texts,
            n_topics=3,
            random_state=42
        )
        
        dominant_topics = get_dominant_topic(doc_topic_matrix)
        
        assert len(dominant_topics) == len(sample_texts)
        assert all(0 <= topic < 3 for topic in dominant_topics)
    
    def test_get_representative_documents(self, sample_texts):
        """Test getting representative documents."""
        _, doc_topic_matrix, _, _ = fit_sklearn_lda(
            sample_texts,
            n_topics=3,
            random_state=42
        )
        
        df = pd.DataFrame({'cleaned_text': sample_texts})
        
        rep_docs = get_representative_documents(
            df,
            doc_topic_matrix,
            text_col='cleaned_text',
            n_docs=2
        )
        
        assert len(rep_docs) == 3  # n_topics
        for topic_idx, docs in rep_docs.items():
            assert len(docs) == 2  # n_docs
            assert all(isinstance(doc, str) for doc in docs)


class TestFitTopics:
    """Test complete topic fitting pipeline."""
    
    @pytest.fixture
    def sample_df(self):
        """Fixture providing sample DataFrame."""
        texts = [
            "The hospital staff are rude and unprofessional",
            "Water supply is not working in our area",
            "School building needs urgent repair",
            "Health center has no medicine available",
            "Thank you for the excellent service at the clinic",
            "The road to the market is in bad condition",
            "No electricity at the school for weeks",
            "Police station needs more officers",
            "Water borehole is broken and needs fixing",
            "Staff at the hospital are very helpful and kind"
        ] * 5
        
        return pd.DataFrame({'cleaned_text': texts})
    
    def test_fit_topics_lda(self, sample_df):
        """Test fitting LDA topics."""
        result = fit_topics(
            sample_df,
            text_col='cleaned_text',
            n_topics=3,
            method='lda',
            random_state=42
        )
        
        # Check all expected keys are present
        expected_keys = [
            'model', 'vectorizer', 'doc_topic_matrix', 'dominant_topics',
            'topics', 'representative_docs', 'feature_names', 'coherence',
            'method', 'n_topics'
        ]
        for key in expected_keys:
            assert key in result
        
        # Check types and shapes
        assert result['method'] == 'lda'
        assert result['n_topics'] == 3
        assert len(result['dominant_topics']) == len(sample_df)
        assert len(result['topics']) == 3
        assert len(result['representative_docs']) == 3
    
    def test_fit_topics_nmf(self, sample_df):
        """Test fitting NMF topics."""
        result = fit_topics(
            sample_df,
            text_col='cleaned_text',
            n_topics=3,
            method='nmf',
            random_state=42
        )
        
        assert result['method'] == 'nmf'
        assert result['n_topics'] == 3
    
    def test_fit_topics_small_dataset(self):
        """Test fitting topics on very small dataset."""
        df = pd.DataFrame({
            'cleaned_text': [
                "hospital staff rude unprofessional behavior attitude",
                "water supply broken not working area village",
                "school needs repair maintenance building infrastructure"
            ] * 5  # Repeat to have enough data for min_df
        })
        
        result = fit_topics(df, n_topics=2, random_state=42)
        
        # Should still work even with minimal data
        assert result is not None
        assert len(result['dominant_topics']) == len(df)


class TestTopicConsistency:
    """Test topic modeling consistency and reproducibility."""
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        texts = [
            "hospital staff service quality care patient treatment",
            "water supply broken pipe maintenance repair area",
            "school education teacher student building needs",
            "health center medicine drugs available stockout"
        ] * 10  # Mix of different texts
        
        model1, matrix1, _, _ = fit_sklearn_lda(
            texts, n_topics=2, random_state=42, max_features=100, min_df=1
        )
        model2, matrix2, _, _ = fit_sklearn_lda(
            texts, n_topics=2, random_state=42, max_features=100, min_df=1
        )
        
        # Results should be identical with same seed
        assert np.allclose(matrix1, matrix2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
