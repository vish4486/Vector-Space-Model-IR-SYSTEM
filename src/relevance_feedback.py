import math
from collections import defaultdict

def rocchio_feedback(query_vector, relevant_docs, doc_vectors, alpha=1.0, beta=0.75, gamma=0.25, non_relevant_docs=None):
    """
    Apply Rocchio relevance feedback to update the query vector.
    Logs every term access and catches invalid formats.
    """
    updated_query = defaultdict(float)

    #print("\n[DEBUG] Applying Rocchio feedback...")
    #print(f"Relevant docs: {relevant_docs}")
    #print(f"Non-relevant docs: {non_relevant_docs}")
    #print(f"Alpha = {alpha}, Beta = {beta}, Gamma = {gamma}")

    # Alpha * original query
    #print("[DEBUG] Scaling original query vector...")
    for term, weight in query_vector.items():
        #print(f"[query] {term}: {weight}")
        updated_query[term] += alpha * weight

    # Beta * relevant documents
    if relevant_docs:
        #print("\n[DEBUG] Adding relevant document vectors...")
        for doc in relevant_docs:
            #print(f"[relevant] Doc: {doc}")
            for term, weight in doc_vectors.get(doc, {}).items():
                try:
                    w = float(weight)
                    #print(f"  [add] {term} in {doc}: {w}")
                    updated_query[term] += (beta / len(relevant_docs)) * w
                except Exception as e:
                    print(f"[ERROR] Failed to convert weight for term '{term}' in doc '{doc}': value = {weight}")
                    raise

    # Gamma * non-relevant documents
    if non_relevant_docs and gamma > 0:
        #print("\n[DEBUG] Subtracting non-relevant document vectors...")
        for doc in non_relevant_docs:
            #print(f"[non-relevant] Doc: {doc}")
            for term, weight in doc_vectors.get(doc, {}).items():
                try:
                    w = float(weight)
                    #print(f"  [subtract] {term} in {doc}: {w}")
                    updated_query[term] -= (gamma / len(non_relevant_docs)) * w
                except Exception as e:
                    print(f"[ERROR] Failed to convert weight for term '{term}' in doc '{doc}': value = {weight}")
                    raise

    print("\n[DEBUG] Rocchio update complete.\n")
    return dict(updated_query)
