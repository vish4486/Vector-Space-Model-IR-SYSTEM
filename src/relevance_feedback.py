import math
from collections import defaultdict

def rocchio_feedback(query_vector, relevant_docs, doc_vectors, alpha=1.0, beta=0.75, gamma=0.0, non_relevant_docs=None):
    """
    Apply Rocchio relevance feedback to update the query vector.
    Only relevant documents are used unless non_relevant_docs are also provided.
    """
    updated_query = defaultdict(float)

    # Start with alpha * original query vector
    for term, weight in query_vector.items():
        updated_query[term] += alpha * weight

    # Add beta * average of relevant documents
    if relevant_docs:
        for doc in relevant_docs:
            for term, weight in doc_vectors.get(doc, {}).items():
                updated_query[term] += (beta / len(relevant_docs)) * weight

    # Optionally subtract gamma * average of non-relevant documents
    if non_relevant_docs and gamma > 0:
        for doc in non_relevant_docs:
            for term, weight in doc_vectors.get(doc, {}).items():
                updated_query[term] -= (gamma / len(non_relevant_docs)) * weight

    return dict(updated_query)
