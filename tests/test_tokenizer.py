"""
Unit tests for the tokenizer module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer import Tokenizer, simple_tokenize, tokenize_text, load_stopwords


class TestTokenizer:
    """Test suite for Tokenizer class."""
    
    def test_empty_input(self):
        """Test handling of empty string."""
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("") == []
        assert tokenizer.sentence_split("") == []
    
    def test_simple_tokenization(self):
        """Test basic word tokenization."""
        tokenizer = Tokenizer()
        text = "Hello world! This is a test."
        tokens = tokenizer.tokenize(text)
        expected = ['hello', 'world', 'this', 'is', 'a', 'test']
        assert tokens == expected
    
    def test_case_sensitivity(self):
        """Test lowercase conversion."""
        tokenizer = Tokenizer(lowercase=True)
        tokens = tokenizer.tokenize("HeLLo WoRLd")
        assert tokens == ['hello', 'world']
        
        tokenizer_upper = Tokenizer(lowercase=False)
        tokens_upper = tokenizer_upper.tokenize("HeLLo WoRLd")
        assert 'HeLLo' in tokens_upper
        assert 'WoRLd' in tokens_upper
    
    def test_stopword_filtering(self):
        """Test stopword removal."""
        stopwords = ['the', 'a', 'is', 'in']
        tokenizer = Tokenizer(stopwords=stopwords)
        text = "The cat is in the house"
        tokens = tokenizer.tokenize(text)
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'in' not in tokens
        assert 'cat' in tokens
        assert 'house' in tokens
    
    def test_sentence_splitting(self):
        """Test sentence splitting."""
        tokenizer = Tokenizer()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = tokenizer.sentence_split(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence"
        assert sentences[1] == "Second sentence"
        assert sentences[2] == "Third sentence"
    
    def test_tokenize_sentences(self):
        """Test tokenization of each sentence."""
        tokenizer = Tokenizer()
        text = "Hello world. How are you?"
        tokenized_sents = tokenizer.tokenize_sentences(text)
        assert len(tokenized_sents) == 2
        assert tokenized_sents[0] == ['hello', 'world']
        assert tokenized_sents[1] == ['how', 'are', 'you']
    
    def test_count_tokens(self):
        """Test token counting."""
        tokenizer = Tokenizer()
        assert tokenizer.count_tokens("one two three") == 3
        assert tokenizer.count_tokens("") == 0
        assert tokenizer.count_tokens("   ") == 0
    
    def test_vocabulary_size(self):
        """Test vocabulary calculation."""
        tokenizer = Tokenizer()
        texts = ["hello world", "hello there", "world is big"]
        vocab_size = tokenizer.vocabulary_size(texts)
        unique_tokens = set(['hello', 'world', 'there', 'is', 'big'])
        assert vocab_size == len(unique_tokens)
    
    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        tokenizer = Tokenizer()
        text = "Hello, world! How're you? I'm fine."
        tokens = tokenizer.tokenize(text)
        # All tokens should be alphanumeric
        assert all(t.isalnum() or t.replace('_', '').isalnum() for t in tokens)
    
    def test_numbers_and_mixed_content(self):
        """Test handling of numbers and mixed content."""
        tokenizer = Tokenizer()
        text = "I have 2 apples and 10 oranges."
        tokens = tokenizer.tokenize(text)
        assert '2' in tokens
        assert '10' in tokens
        assert 'apples' in tokens
        assert 'oranges' in tokens


class TestSimpleTokenize:
    """Test simple_tokenize convenience function."""
    
    def test_basic(self):
        """Test basic usage."""
        tokens = simple_tokenize("Hello World")
        assert tokens == ['hello', 'world']
    
    def test_preserve_case(self):
        """Test case preservation."""
        tokens = simple_tokenize("Hello World", lowercase=False)
        assert 'Hello' in tokens
        assert 'World' in tokens


class TestTokenizeText:
    """Test tokenize_text convenience function."""
    
    def test_with_stopwords(self):
        """Test with stopword removal."""
        tokens = tokenize_text("The quick brown fox", remove_stopwords=True)
        assert 'the' not in tokens
        assert 'quick' in tokens
    
    def test_without_stopwords(self):
        """Test without stopword removal."""
        tokens = tokenize_text("The quick brown fox", remove_stopwords=False)
        assert 'the' in tokens


class TestLoadStopwords:
    """Test stopword loading."""
    
    def test_default_stopwords(self):
        """Test default English stopwords are loaded."""
        stopwords = load_stopwords()
        assert isinstance(stopwords, list)
        assert len(stopwords) > 10
        assert 'the' in stopwords
        assert 'a' in stopwords
    
    def test_custom_file(self, tmp_path):
        """Test loading from custom file."""
        stopfile = tmp_path / "stopwords.txt"
        stopfile.write_text("custom1\ncustom2\n")
        stopwords = load_stopwords(stopfile)
        assert 'custom1' in stopwords
        assert 'custom2' in stopwords


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete tokenization pipeline."""
        text = """
        Natural language processing is fun!
        It helps computers understand human language.
        Let's build something amazing.
        """
        
        tokenizer = Tokenizer(lowercase=True, stopwords=[])
        sentences = tokenizer.sentence_split(text)
        assert len(sentences) >= 2
        
        all_tokens = []
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            all_tokens.extend(tokens)
        
        assert len(all_tokens) > 0
        # Check lowercase worked
        assert any(t == 'natural' for t in all_tokens)
        assert any(t == 'language' for t in all_tokens)
    
    def test_repeated_tokenization(self):
        """Test tokenizing same text multiple times yields same result."""
        tokenizer = Tokenizer()
        text = "Test repeated tokenization"
        result1 = tokenizer.tokenize(text)
        result2 = tokenizer.tokenize(text)
        assert result1 == result2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
