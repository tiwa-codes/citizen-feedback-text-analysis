"""
Unit tests for sentiment analysis module.
"""
import pytest
import pandas as pd
from src.text.sentiment import (
    tokenize_for_sentiment,
    compute_lexicon_sentiment,
    compute_simple_polarity,
    compute_sentiment,
    get_sentiment_summary
)


class TestTokenization:
    """Test sentiment tokenization."""
    
    def test_tokenize_for_sentiment(self):
        """Test tokenization for sentiment analysis."""
        text = "Hello, world! This is a test."
        tokens = tokenize_for_sentiment(text)
        
        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_tokenize_empty(self):
        """Test tokenization of empty text."""
        tokens = tokenize_for_sentiment("")
        assert tokens == []
    
    def test_tokenize_none(self):
        """Test tokenization of None."""
        tokens = tokenize_for_sentiment(None)
        assert tokens == []


class TestLexiconSentiment:
    """Test lexicon-based sentiment analysis."""
    
    def test_positive_sentiment(self):
        """Test detection of positive sentiment."""
        text = "The service was excellent and the staff were very helpful"
        score, label, confidence = compute_lexicon_sentiment(text)
        
        assert label == 'positive'
        assert score > 0
        assert 0 <= confidence <= 1
    
    def test_negative_sentiment(self):
        """Test detection of negative sentiment."""
        text = "The service was terrible and the staff were very rude"
        score, label, confidence = compute_lexicon_sentiment(text)
        
        assert label == 'negative'
        assert score < 0
        assert 0 <= confidence <= 1
    
    def test_neutral_sentiment(self):
        """Test detection of neutral sentiment."""
        text = "The office is located on Main Street"
        score, label, confidence = compute_lexicon_sentiment(text)
        
        assert label == 'neutral'
        assert abs(score) < 0.1  # Close to zero
    
    def test_negation_handling(self):
        """Test that negation flips sentiment."""
        positive_text = "The service was good"
        negative_text = "The service was not good"
        
        pos_score, pos_label, _ = compute_lexicon_sentiment(positive_text)
        neg_score, neg_label, _ = compute_lexicon_sentiment(negative_text)
        
        # Negation should flip or reduce positive sentiment
        assert pos_score > neg_score or pos_label != neg_label
    
    def test_intensifier_handling(self):
        """Test that intensifiers amplify sentiment."""
        normal_text = "The service was good"
        intense_text = "The service was very good"
        
        normal_score, _, _ = compute_lexicon_sentiment(normal_text)
        intense_score, _, _ = compute_lexicon_sentiment(intense_text)
        
        # Intensifier should increase sentiment magnitude
        assert intense_score >= normal_score
    
    def test_empty_text(self):
        """Test sentiment of empty text."""
        score, label, confidence = compute_lexicon_sentiment("")
        
        assert score == 0.0
        assert label == 'neutral'
        assert confidence == 0.0
    
    def test_none_text(self):
        """Test sentiment of None."""
        score, label, confidence = compute_lexicon_sentiment(None)
        
        assert score == 0.0
        assert label == 'neutral'
        assert confidence == 0.0


class TestSimplePolarity:
    """Test simple polarity calculation."""
    
    def test_positive_polarity(self):
        """Test positive polarity."""
        text = "good great excellent"
        polarity, label = compute_simple_polarity(text)
        
        assert polarity > 0
        assert label == 'positive'
    
    def test_negative_polarity(self):
        """Test negative polarity."""
        text = "bad terrible horrible"
        polarity, label = compute_simple_polarity(text)
        
        assert polarity < 0
        assert label == 'negative'
    
    def test_mixed_polarity(self):
        """Test mixed polarity."""
        text = "good but bad"
        polarity, label = compute_simple_polarity(text)
        
        assert -1 <= polarity <= 1
        assert label in ['positive', 'neutral', 'negative']
    
    def test_neutral_polarity(self):
        """Test neutral polarity."""
        text = "the building is on the street"
        polarity, label = compute_simple_polarity(text)
        
        assert polarity == 0.0
        assert label == 'neutral'


class TestComputeSentiment:
    """Test sentiment computation on DataFrame."""
    
    def test_compute_sentiment_df(self):
        """Test computing sentiment for DataFrame."""
        df = pd.DataFrame({
            'cleaned_text': [
                "The service was excellent",
                "The staff were rude",
                "The building is located downtown",
                "Thank you for the help",
                "This is very bad"
            ]
        })
        
        df_with_sentiment = compute_sentiment(df, text_col='cleaned_text')
        
        # Check that sentiment columns are added
        assert 'sentiment_score' in df_with_sentiment.columns
        assert 'sentiment_label' in df_with_sentiment.columns
        assert 'sentiment_confidence' in df_with_sentiment.columns
        assert 'polarity_score' in df_with_sentiment.columns
        assert 'polarity_label' in df_with_sentiment.columns
        
        # Check that labels are valid
        valid_labels = {'positive', 'neutral', 'negative'}
        assert all(label in valid_labels for label in df_with_sentiment['sentiment_label'])
        
        # Check score ranges
        assert all(-1 <= score <= 1 for score in df_with_sentiment['sentiment_score'])
        assert all(0 <= conf <= 1 for conf in df_with_sentiment['sentiment_confidence'])
    
    def test_compute_sentiment_empty_df(self):
        """Test computing sentiment for empty DataFrame."""
        df = pd.DataFrame({'cleaned_text': []})
        df_with_sentiment = compute_sentiment(df, text_col='cleaned_text')
        
        assert len(df_with_sentiment) == 0
        assert 'sentiment_score' in df_with_sentiment.columns


class TestSentimentSummary:
    """Test sentiment summary statistics."""
    
    def test_get_sentiment_summary_overall(self):
        """Test overall sentiment summary."""
        df = pd.DataFrame({
            'sentiment_score': [0.5, -0.5, 0.0, 0.3, -0.2],
            'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        
        summary = get_sentiment_summary(df)
        
        assert 'mean_sentiment' in summary.columns
        assert 'positive_ratio' in summary.columns
        assert 'negative_ratio' in summary.columns
        
        # Check that ratios sum to 1
        total_ratio = (
            summary['positive_ratio'].iloc[0] +
            summary['neutral_ratio'].iloc[0] +
            summary['negative_ratio'].iloc[0]
        )
        assert abs(total_ratio - 1.0) < 0.01
    
    def test_get_sentiment_summary_grouped(self):
        """Test grouped sentiment summary."""
        df = pd.DataFrame({
            'sentiment_score': [0.5, -0.5, 0.0, 0.3, -0.2],
            'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'channel': ['SMS', 'SMS', 'Web', 'Web', 'Hotline']
        })
        
        summary = get_sentiment_summary(df, group_by='channel')
        
        assert 'mean_sentiment' in summary.columns
        assert len(summary) <= df['channel'].nunique()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
