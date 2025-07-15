import os
import sys
import json
import math
from collections import Counter,defaultdict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_text
from src import spell_correction
from src.utils import cosine_similarity

# ==== Load Index Files ====

with open("index/idf.json", "r") as f:
    IDF = json.load(f)

with open("index/doc_vectors.json", "r") as f:
    DOC_VECTORS = json.load(f)

with open("index/champion_lists.json", "r") as f:
    CHAMPION_LISTS = json.load(f)

with open("index/leaders.json", "r") as f:
    LEADERS = json.load(f)

with open("index/leader_followers.json", "r") as f:
    LEADER_FOLLOWERS = json.load(f)

with open("index/static_quality_scores.json", "r") as f:
    STATIC_SCORES = json.load(f)

with open("index/impact_index.json", "r") as f:
    IMPACT_INDEX = json.load(f)


# ==== Vocabulary ====
vocabulary = set(IDF.keys())

# ==== TF-IDF Vector for Query ====

def compute_query_vector(query_tokens, idf_dict):
    tf = Counter(query_tokens)
    total_terms = sum(tf.values())
    tfidf_vector = {}

    for term, count in tf.items():
        if term in idf_dict:
            tf_score = count / total_terms
            tfidf_vector[term] = tf_score * idf_dict[term]
    return tfidf_vector

# ==== Retrieval Strategies ====

def rank_documents(query_vector, doc_vectors, top_k=5):
    scores = []
    for doc_name, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((doc_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def rank_with_champion_lists(query_vector, doc_vectors, champion_lists, top_k=5):
    candidate_docs = set()
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

def rank_with_cluster_pruning(query_vector, top_k=5):
    # Step 1: Choose closest leader
    best_leader = max(
        LEADERS,
        key=lambda leader: cosine_similarity(query_vector, DOC_VECTORS[leader])
    )

    # Step 2: Search only within leader's cluster
    candidate_docs = [best_leader] + LEADER_FOLLOWERS.get(best_leader, [])
    
    scores = []
    for doc in candidate_docs:
        doc_vec = DOC_VECTORS.get(doc, {})
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def rank_with_static_quality(query_vector, doc_vectors, static_scores, alpha=0.5, top_k=5):
    """
    Combines cosine similarity with static quality score.
    Final score = alpha * cosine_sim + (1 - alpha) * static_score
    """
    scores = []
    for doc, vec in doc_vectors.items():
        cosine = cosine_similarity(query_vector, vec)
        static = static_scores.get(doc, 0)
        combined = alpha * cosine + (1 - alpha) * static
        scores.append((doc, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def rank_with_impact_ordering(query_vector, top_k=5):
    """Rank documents using impact-ordered index"""
    scores = defaultdict(float)

    for term, q_weight in query_vector.items():
        if term in IMPACT_INDEX:
            for doc_name, d_weight in IMPACT_INDEX[term]:
                scores[doc_name] += q_weight * d_weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ==== Main Search Dispatcher ====

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
        return rank_with_champion_lists(query_vector, DOC_VECTORS, CHAMPION_LISTS, top_k)
    elif method == "cluster":
        return rank_with_cluster_pruning(query_vector, top_k)
    elif method == "static":
        return rank_with_static_quality(query_vector, DOC_VECTORS, STATIC_SCORES, top_k=top_k)
    elif method == "impact":
        return rank_with_impact_ordering(query_vector, top_k)
    else:
        raise ValueError(f"Unknown search method: {method}")
