"""
NLP Tokenizer Module
A simple, extensible tokenization library for text processing.
"""

import re
from typing import List, Optional
from pathlib import Path


class Tokenizer:
    """Basic tokenizer for natural language processing tasks."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 stopwords: Optional[List[str]] = None):
        """
        Initialize tokenizer.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation tokens
            stopwords: List of stopwords to filter out
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.stopwords = set(stopwords) if stopwords else set()
        
        # Basic word tokenization pattern
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Sentence splitting pattern
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Apply lowercase
        if self.lowercase:
            text = text.lower()
        
        # Find all word tokens
        tokens = self.word_pattern.findall(text)
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            tokens = [t for t in tokens if t.isalnum()]
        
        # Remove stopwords
        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences (without trailing whitespace or punctuation)
        """
        if not text:
            return []
        
        # Split on sentence-ending punctuation followed by whitespace
        sentences = self.sentence_pattern.split(text.strip())
        
        # Filter out empty strings and strip trailing punctuation
        sentences = [s.strip().rstrip('.,!?;:') for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize_sentences(self, text: str) -> List[List[str]]:
        """
        Split text into sentences and tokenize each sentence.
        
        Args:
            text: Input text
            
        Returns:
            List of token lists, one per sentence
        """
        sentences = self.sentence_split(text)
        return [self.tokenize(sent) for sent in sentences]
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        return len(self.tokenize(text))
    
    def vocabulary_size(self, texts: List[str]) -> int:
        """Calculate vocabulary size from a list of texts."""
        vocab = set()
        for text in texts:
            vocab.update(self.tokenize(text))
        return len(vocab)


def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple one-function tokenizer for quick use.
    
    Args:
        text: Text to tokenize
        lowercase: Whether to lowercase
        
    Returns:
        List of tokens
    """
    tokenizer = Tokenizer(lowercase=lowercase)
    return tokenizer.tokenize(text)


def load_stopwords(filepath: Optional[Path] = None) -> List[str]:
    """
    Load stopwords from file or use default English stopwords.
    
    Args:
        filepath: Path to stopwords file (one word per line)
        
    Returns:
        List of stopwords
    """
    default_stopwords = [
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were'
    ]
    
    if filepath and filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    return default_stopwords


# Convenience function for common use
def tokenize_text(text: str, 
                  remove_stopwords: bool = False,
                  lowercase: bool = True) -> List[str]:
    """
    Quick tokenization function.
    
    Args:
        text: Text to tokenize
        remove_stopwords: Whether to filter stopwords
        lowercase: Whether to lowercase
        
    Returns:
        List of tokens
    """
    stopwords = load_stopwords() if remove_stopwords else []
    tokenizer = Tokenizer(lowercase=lowercase, stopwords=stopwords)
    return tokenizer.tokenize(text)
