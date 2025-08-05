import os
import sys
import json
import math
from collections import Counter,defaultdict

# Add parent directory to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import custom modules
from src.preprocessing import preprocess_text
from src import spell_correction
from src.utils import cosine_similarity
from src import relevance_feedback

# ==== Load Index Files ====
# Precomputed index data used by all retrieval strategies
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


# ==== Vocabulary for spell correction ====
vocabulary = set(IDF.keys())

# ==== TF-IDF Vector for Query ====
def compute_query_vector(query_tokens, idf_dict):
    """
    Build a TF-IDF vector for the input query tokens.
    """
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
    """
    Rank all documents based on cosine similarity with the query vector.
    """
    scores = []
    for doc_name, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((doc_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def rank_with_champion_lists(query_vector, doc_vectors, champion_lists, top_k=5):
    """
    Use Champion Lists to retrieve and rank only top documents for each term.
    """
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
    """
    Use Cluster Pruning to reduce the search space.
    Step 1: Find closest leader.
    Step 2: Search within that leader’s cluster.
    """
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
    """
    Rank documents using precomputed impact-ordered index (term-wise inverted list).
    """
    scores = defaultdict(float)

    for term, q_weight in query_vector.items():
        if term in IMPACT_INDEX:
            for doc_name, d_weight in IMPACT_INDEX[term]:
                scores[doc_name] += q_weight * d_weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


#====MANUAL RELEVANCE FEEDBACK====REQUIRES KNOWN RELEVANT DOCS#
def search_with_feedback(query_vector, relevant_docs, top_k=5, non_relevant_docs=None):
    """
    Manual Rocchio feedback based on user-marked relevant documents.
    """
    updated_query_vector = relevance_feedback.rocchio_feedback(query_vector, relevant_docs, DOC_VECTORS,non_relevant_docs=non_relevant_docs)
    return rank_documents(updated_query_vector, DOC_VECTORS, top_k)


#====PSEUDO RELEVANCE FEEDBACK====#
def search_with_pseudo_feedback(query_vector, top_k=5, pseudo_k=3):
    """
    Apply Rocchio using top pseudo_k docs as relevant (pseudo-relevance feedback).
    """
    """Assume top pseudo_k docs are relevant and apply Rocchio"""
    # Step 1: Initial retrieval
    initial_results = rank_documents(query_vector, DOC_VECTORS, top_k=pseudo_k)
    pseudo_relevant_docs = [doc for doc, score in initial_results if score > 0]

    # Step 2: Refine query
    updated_query_vector = relevance_feedback.rocchio_feedback(query_vector, pseudo_relevant_docs, DOC_VECTORS)

    # Step 3: Rank again using updated query
    return rank_documents(updated_query_vector, DOC_VECTORS, top_k)



# ==== Main Search Dispatcher ====

def search(query, top_k=5, method="basic"):
    """
    Dispatcher that applies the specified retrieval method on the given query.
    """
    # Step 1: Preprocess and spell-correct
    tokens = preprocess_text(query)
    corrected_tokens = spell_correction.correct_query(tokens, vocabulary)
    if tokens != corrected_tokens:
        print(f"[Info] Corrected query: {' '.join(corrected_tokens)}")
    tokens = corrected_tokens

    # Step 2: Compute query TF-IDF vector
    query_vector = compute_query_vector(tokens, IDF)

    # Step 3: Call appropriate ranking strategy
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
    elif method == "feedback":
        # Manual relevance feedback using CLI
        top_results = rank_documents(query_vector, DOC_VECTORS, top_k)
        print(f"\nTop {top_k} initial results using 'feedback' retrieval:")
        for rank, (doc_name, score) in enumerate(top_results, 1):
            print(f"{rank}. {doc_name} (Score: {score:.4f})")

        print("\nMark relevant documents (comma-separated list of doc IDs, e.g., doc2.txt,doc5.txt):")
        feedback_input = input("Relevant documents: ").strip()
        relevant_docs = [doc.strip() for doc in feedback_input.split(",") if doc.strip() in dict(top_results)]

        print("Mark non-relevant documents (comma-separated list of doc IDs, or press Enter to skip):")
        nonrel_input = input("Non-relevant documents: ").strip()
        non_relevant_docs = [doc.strip() for doc in nonrel_input.split(",") if doc.strip() in dict(top_results)]

        if not relevant_docs:
            print("No relevant documents selected. Using original results.")
            return top_results

        return search_with_feedback(query_vector, relevant_docs, top_k,non_relevant_docs)
    elif method == "pseudo":
        return search_with_pseudo_feedback(query_vector, top_k)
    else:
        raise ValueError(f"Unknown search method: {method}")
