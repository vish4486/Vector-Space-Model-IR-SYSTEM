import json
import math
from collections import Counter
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_text
from src import spell_correction


# Load precomputed IDF and document vectors
with open("index/idf.json", "r") as f:
    IDF = json.load(f)

with open("index/doc_vectors.json", "r") as f:
    DOC_VECTORS = json.load(f)

with open("index/champion_lists.json", "r") as f:
    CHAMPION_LISTS = json.load(f)


# Extract vocabulary from IDF keys
vocabulary = set(IDF.keys())

def preprocess_query(query):
    """Preprocess the query using same pipeline as documents"""
    return preprocess_text(query)

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


def rank_documents_champion_lists(query_vector, doc_vectors, champion_lists, top_k=5):
    """Rank documents using Champion Lists (reduced candidate set)"""
    candidate_docs = set()

    # Collect candidate documents from Champion Lists
    for term in query_vector:
        if term in champion_lists:
            candidate_docs.update(doc for doc, _ in champion_lists[term])

    scores = []
    for doc in candidate_docs:
        doc_vec = doc_vectors.get(doc, {})
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search_with_champion_lists(query, top_k=5):
    tokens = preprocess_text(query)
    corrected_tokens = spell_correction.correct_query(tokens, vocabulary)
    if tokens != corrected_tokens:
        print(f"[Info] Corrected query: {' '.join(corrected_tokens)}")

    query_vector = compute_query_vector(corrected_tokens, IDF)
    return rank_documents_champion_lists(query_vector, DOC_VECTORS, CHAMPION_LISTS, top_k)



def search(query, top_k=5, method="basic"):
    tokens = preprocess_text(query)
    corrected_tokens = spell_correction.correct_query(tokens, vocabulary)
    if tokens != corrected_tokens:
        print(f"[Info] Corrected query: {' '.join(corrected_tokens)}")
    tokens = corrected_tokens

    query_vector = compute_query_vector(tokens, IDF)

    if method == "basic":
        return rank_documents(query_vector, DOC_VECTORS, top_k)
    elif method == "champion":
        return rank_documents_champion_lists(query_vector, DOC_VECTORS, CHAMPION_LISTS, top_k)
    elif method == "cluster":
        raise NotImplementedError("Cluster pruning not implemented yet.")
    else:
        raise ValueError(f"Unknown search method: {method}")


