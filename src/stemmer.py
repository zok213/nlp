"""
Stemmer Module
Porter Stemmer implementation for reducing words to their root form.
"""

import re
from typing import List, Optional


class PorterStemmer:
    """
    Implementation of the Porter Stemming Algorithm.
    
    The Porter stemmer removes morphological endings from words to produce
    the base or "stem" form. It uses a series of rule-based transformations.
    
    Reference: https://tartarus.org/martin/PorterStemmer/
    """
    
    def __init__(self):
        """Initialize the stemmer with consonant/vowel patterns."""
        self.cons = r'[^aeiou]'
        self.vowel = r'[aeiou]'
        self.syllable_pattern = re.compile(f'{self.cons}*{self.vowel}{self.cons}*')
    
    def _has_vowel(self, stem: str) -> bool:
        """Check if stem contains a vowel."""
        return re.search(self.vowel, stem) is not None
    
    def _ends_with(self, word: str, suffix: str) -> bool:
        """Check if word ends with the given suffix."""
        return word.endswith(suffix)
    
    def _replace_suffix(self, word: str, suffix: str, replacement: str) -> str:
        """Replace suffix with replacement."""
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement
        return word
    
    def _measure(self, stem: str) -> int:
        """
        Calculate the measure (m) of a stem.
        Count of VC sequences (vowel-consonant groups).
        """
        syllables = self.syllable_pattern.findall(stem)
        return len(syllables)
    
    def _contains_vowel(self, stem: str, start: int = 0) -> bool:
        """Check if stem contains a vowel from start position onward."""
        return re.search(self.vowel, stem[start:]) is not None
    
    # Step 1a: Plural and past tense removal
    def _step1a(self, word: str) -> str:
        """Step 1a: Remove plural and past tense markers."""
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2] if len(word) > 3 else word
        if word.endswith('ss'):
            return word
        if word.endswith('s'):
            return word[:-1]
        return word
    
    # Step 1b: -ly, -ed, -ing removal
    def _step1b(self, word: str) -> str:
        """Step 1b: Remove -ly, -ed, -ing endings."""
        if word.endswith('eed'):
            if self._measure(word[:-3]) > 0:
                return word[:-1]
            return word
        
        if word.endswith('ed') and self._contains_vowel(word[:-2]):
            word = word[:-2]
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                return word + 'e'
            if word.endswith('bb') or word.endswith('dd') or word.endswith('ff') or \
               word.endswith('gg') or word.endswith('mm') or word.endswith('nn') or \
               word.endswith('pp') or word.endswith('rr') or word.endswith('tt'):
                return word[:-1]
            if self._measure(word) == 1 and self._has_vowel(word):
                return word + 'e'
            return word
        
        if word.endswith('ing') and self._contains_vowel(word[:-3]):
            word = word[:-3]
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                return word + 'e'
            if word.endswith('bb') or word.endswith('dd') or word.endswith('ff') or \
               word.endswith('gg') or word.endswith('mm') or word.endswith('nn') or \
               word.endswith('pp') or word.endswith('rr') or word.endswith('tt'):
                return word[:-1]
            if self._measure(word) == 1 and self._has_vowel(word):
                return word + 'e'
            return word
        
        if word.endswith('y') and self._contains_vowel(word[:-1]):
            return word[:-1] + 'i'
        
        return word
    
    # Step 1c: Replace -y with -i
    def _step1c(self, word: str) -> str:
        """Step 1c: Replace suffix y with i after vowel."""
        if len(word) > 2 and word.endswith('y') and self._has_vowel(word[:-1]):
            return word[:-1] + 'i'
        return word
    
    # Step 2: Longer suffixes
    def _step2(self, word: str) -> str:
        """Step 2: Remove longer suffixes."""
        suffixes = {
            'ational': 'ate',
            'tional': 'tion',
            'enci': 'ence',
            'anci': 'ance',
            'izer': 'ize',
            'isator': 'ize',
            'biliti': 'ble',
            'logi': 'log',
            'alli': 'al',
            'entli': 'ent',
            'fulness': '',
            'ousli': 'ous',
            'ization': 'ize',
            'ation': 'ate',
            'ator': 'ate',
            'alism': 'al',
            'iveness': 'ive',
            'fulness': 'ful',
            'ousness': 'ous',
            'aliti': 'al',
            'iviti': 'ive',
            'ibiliti': 'ible',
        }
        
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement if replacement else stem
        return word
    
    # Step 3: Shorter suffixes
    def _step3(self, word: str) -> str:
        """Step 3: Remove shorter suffixes."""
        suffixes = {
            'icate': 'ic',
            'iciti': 'ic',
            'ative': '',
            'alize': 'al',
            'icile': 'ic',
            'ical': 'ic',
            'ful': '',
            'ness': '',
        }
        
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement if replacement else stem
        return word
    
    # Step 4: Remove suffixes with measure >=2
    def _step4(self, word: str) -> str:
        """Step 4: Remove suffixes when measure >= 2."""
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible',
            'ant', 'ement', 'ment', 'ent', 'ism', 'ate',
            'iti', 'ous', 'ive', 'ize', 'ion', 'al', 'er', 'ic'
        ]
        
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:  # measure >= 2
                    return stem
        return word
    
    # Step 5: Remove final -e and -l
    def _step5a(self, word: str) -> str:
        """Step 5a: Remove final -e if measure >= 2."""
        if word.endswith('e'):
            stem = word[:-1]
            if self._measure(stem) > 1 or (self._measure(stem) == 1 and not self._has_vowel(stem)):
                return stem
        return word
    
    def _step5b(self, word: str) -> str:
        """Step 5b: Remove final -l if measure >= 2 and ends with double -l."""
        if len(word) > 1 and word.endswith('ll') and self._measure(word[:-2]) > 1:
            return word[:-1]
        return word
    
    def stem(self, word: str) -> str:
        """
        Apply Porter stemming algorithm to a word.
        
        Args:
            word: Input word to stem
            
        Returns:
            Stemmed word form
        """
        if not word or len(word) < 3:
            return word
        
        # Lowercase the word
        word = word.lower()
        
        # Apply steps in order
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)
        
        return word


class Stemmer:
    """Simplified stemmer interface."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def stem(self, text: str) -> str:
        """Stem a single word."""
        return self.stemmer.stem(text)
    
    def stem_list(self, words: List[str]) -> List[str]:
        """Stem a list of words."""
        return [self.stem(word) for word in words]
    
    def stem_text(self, text: str, lowercase: bool = True) -> List[str]:
        """
        Tokenize and stem text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase before stemming
            
        Returns:
            List of stemmed tokens
        """
        import re
        if lowercase:
            text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return self.stem_list(tokens)


def simple_stem(word: str) -> str:
    """
    Quick stemming function.
    
    Args:
        word: Word to stem
        
    Returns:
        Stemmed word
    """
    stemmer = PorterStemmer()
    return stemmer.stem(word)
