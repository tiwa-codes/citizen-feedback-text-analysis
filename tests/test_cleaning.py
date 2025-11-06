"""
Unit tests for text cleaning module.
"""
import pytest
import pandas as pd
from src.text.cleaning import (
    remove_non_printable,
    normalize_punctuation,
    expand_contractions,
    mask_phone_numbers,
    mask_emails,
    mask_national_id,
    detect_url,
    detect_number,
    detect_expletive,
    tokenize_simple,
    clean_text,
    clean_text_df
)


class TestBasicCleaning:
    """Test basic text cleaning functions."""
    
    def test_remove_non_printable(self):
        """Test non-printable character removal."""
        text = "Hello\x00World\x01Test"
        cleaned = remove_non_printable(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    def test_normalize_punctuation(self):
        """Test punctuation normalization."""
        text = "Hello...  World  !"
        cleaned = normalize_punctuation(text)
        assert "..." not in cleaned
        assert " !" not in cleaned
        assert cleaned == "Hello. World!"
    
    def test_expand_contractions(self):
        """Test contraction expansion."""
        text = "dont cant wont"
        expanded = expand_contractions(text)
        assert "do not" in expanded
        assert "can not" in expanded
        assert "will not" in expanded


class TestPIIMasking:
    """Test PII masking functions."""
    
    def test_mask_phone_numbers(self):
        """Test phone number masking."""
        text = "Call me at +2348012345678"
        masked = mask_phone_numbers(text)
        assert "[PHONE]" in masked
        assert "+2348012345678" not in masked
        
        text2 = "My number is 08012345678"
        masked2 = mask_phone_numbers(text2)
        assert "[PHONE]" in masked2
    
    def test_mask_emails(self):
        """Test email masking."""
        text = "Email me at test@example.com"
        masked = mask_emails(text)
        assert "[EMAIL]" in masked
        assert "test@example.com" not in masked
    
    def test_mask_national_id(self):
        """Test national ID masking."""
        text = "My NIN: 12345678901"
        masked = mask_national_id(text)
        assert "[ID]" in masked or "12345678901" not in masked


class TestDetection:
    """Test detection functions."""
    
    def test_detect_url(self):
        """Test URL detection."""
        assert detect_url("Visit https://example.com") is True
        assert detect_url("Visit www.example.com") is True
        assert detect_url("No URL here") is False
    
    def test_detect_number(self):
        """Test number detection."""
        assert detect_number("I have 5 items") is True
        assert detect_number("No numbers") is False
    
    def test_detect_expletive(self):
        """Test expletive detection."""
        assert detect_expletive("This is damn bad") is True
        assert detect_expletive("This is nice") is False


class TestTokenization:
    """Test tokenization."""
    
    def test_tokenize_simple(self):
        """Test simple tokenization."""
        text = "Hello, world! How are you?"
        tokens = tokenize_simple(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert len(tokens) == 5


class TestCleanText:
    """Test complete text cleaning."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello world!"
        cleaned, metadata = clean_text(text)
        
        assert cleaned != ""
        assert isinstance(metadata, dict)
        assert 'word_count' in metadata
        assert 'char_count' in metadata
        assert metadata['word_count'] == 2
    
    def test_clean_text_with_pii(self):
        """Test cleaning with PII masking."""
        text = "Call me at +2348012345678 or email test@example.com"
        cleaned, metadata = clean_text(text, mask_pii=True)
        
        assert "[PHONE]" in cleaned
        assert "[EMAIL]" in cleaned
        assert "+2348012345678" not in cleaned
        assert "test@example.com" not in cleaned
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        cleaned, metadata = clean_text("")
        assert cleaned == ""
        assert metadata['word_count'] == 0
    
    def test_clean_text_none(self):
        """Test cleaning None value."""
        cleaned, metadata = clean_text(None)
        assert cleaned == ""
        assert metadata['word_count'] == 0


class TestCleanDataFrame:
    """Test DataFrame cleaning."""
    
    def test_clean_text_df(self):
        """Test cleaning DataFrame with text."""
        df = pd.DataFrame({
            'raw_text': [
                "Hello world",
                "Test message with phone +2348012345678",
                "A",  # Too short, should be filtered
                "Another valid message"
            ]
        })
        
        df_clean = clean_text_df(df, text_col='raw_text')
        
        # Check that short messages are filtered
        assert len(df_clean) == 3  # One message too short
        
        # Check that cleaned_text column exists
        assert 'cleaned_text' in df_clean.columns
        assert 'word_count' in df_clean.columns
        assert 'char_count' in df_clean.columns
        
        # Check PII masking
        phone_row = df_clean[df_clean['raw_text'].str.contains('phone', case=False, na=False)]
        if len(phone_row) > 0:
            assert '[PHONE]' in phone_row.iloc[0]['cleaned_text']
    
    def test_clean_text_df_empty(self):
        """Test cleaning empty DataFrame."""
        df = pd.DataFrame({'raw_text': []})
        df_clean = clean_text_df(df, text_col='raw_text')
        
        assert len(df_clean) == 0
        assert 'cleaned_text' in df_clean.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
