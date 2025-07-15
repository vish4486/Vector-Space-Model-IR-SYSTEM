import json
import math
from collections import Counter
from src.preprocessing import preprocess

# Load precomputed IDF and document vectors
with open("index/idf.json", "r") as f:
    IDF = json.load(f)

with open("index/doc_vectors.json", "r") as f:
    DOC_VECTORS = json.load(f)

def preprocess_query(query):
    """Preprocess the query using same pipeline as documents"""
    return preprocess(query)

def compute_query_vector(query_tokens, idf_dict):
    """Compute TF-IDF vector for the query"""
    tf = Counter(query_tokens)
    total_terms = sum(tf.values())
    
    tfidf_vector = {}
    for term, count in tf.items():
        if term in idf_dict:
            tf_score = count / total_terms
            tfidf_vector[term] = tf_score * idf_dict[term]
    return tfidf_vector

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two sparse vectors (dicts)"""
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def rank_documents(query_vector, doc_vectors, top_k=5):
    """Rank all documents based on cosine similarity to the query"""
    scores = []
    for doc_name, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((doc_name, score))
    
    # Sort documents by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search(query, top_k=5):
    """Main search function to return top K relevant documents"""
    tokens = preprocess_query(query)
    query_vector = compute_query_vector(tokens, IDF)
    return rank_documents(query_vector, DOC_VECTORS, top_k)
