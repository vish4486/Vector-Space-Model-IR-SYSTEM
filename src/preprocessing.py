import re
import os

def load_stopwords(path="data/stopwords.txt"):
    """
    Load stopwords from a plain text file (one word per line).
    this returns a set of stopwords.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stopword file not found at: {path}")
    
    with open(path, 'r') as file:
        stopwords = {line.strip().lower() for line in file if line.strip()}
    return stopwords

# Load stopwords once at import
STOPWORDS = load_stopwords()

def tokenize(text):
    """
    Tokenize the input text:
    - Lowercases the text
    - Extracts words with only alphabetic characters (a-z)
    - Removes numbers and punctuation
    Returns a list of tokens.
    """
    return re.findall(r'\b[a-z]+\b', text.lower())

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.
    Returns a filtered list.
    """
    return [token for token in tokens if token not in STOPWORDS]

def preprocess_text(text):
    """
    Complete preprocessing pipeline:
    - Tokenization
    - Stopword removal
    Returns a list of cleaned tokens.
    """
    tokens = tokenize(text)
    return remove_stopwords(tokens)
