import os
import json
import math
from collections import defaultdict
from src.preprocessing import preprocess_text
from src.utils import cosine_similarity

# === Document Reader ===
def read_documents(folder="data/raw_docs/"):
    """
    Read and preprocess all .txt documents in the specified folder.
    Returns a dictionary: {filename: list of preprocessed tokens}.
    """
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                documents[filename] = preprocess_text(file.read())
    return documents

# === Term Frequency (TF) ===
def compute_tf(documents):
    """
    Compute raw term frequency for each document.
    Returns: {doc_id: {term: freq}}.
    """
    tf = {}
    for doc_id, tokens in documents.items():
        tf_doc = defaultdict(int)
        for token in tokens:
            tf_doc[token] += 1
        tf[doc_id] = dict(tf_doc)
    return tf

# === Document Frequency (DF) ===
def compute_df(tf):
    """
    Compute document frequency for each term.
    Returns: {term: doc_count}.
    """
    df = defaultdict(int)
    for tf_doc in tf.values():
        for term in tf_doc:
            df[term] += 1
    return dict(df)


# === Inverse Document Frequency (IDF) ===
def compute_idf(df, N):
    """
    Compute IDF for each term using: log(N / df).
    """
    return {term: math.log(N / df[term]) for term in df}


# === TF-IDF Vector Computation ===
def compute_tfidf(tf, idf):
    """
    Compute TF-IDF weight for each term in each document.
    Returns: {doc_id: {term: tf-idf_weight}}.
    """
    tfidf = {}
    for doc_id, tf_doc in tf.items():
        tfidf_doc = {}
        for term, freq in tf_doc.items():
            tfidf_doc[term] = freq * idf.get(term, 0)
        tfidf[doc_id] = tfidf_doc
    return tfidf


# === Inverted Index Construction ===
def build_inverted_index(tfidf):
    """
    Build inverted index: {term: list of (doc_id, tf-idf_weight)}.
    """
    index = defaultdict(list)
    for doc_id, terms in tfidf.items():
        for term, weight in terms.items():
            index[term].append((doc_id, weight))
    return dict(index)


# === Save JSON Utility ===
def save_json(data, path):
    """
    Save any Python dict or list to a JSON file with indentation.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
