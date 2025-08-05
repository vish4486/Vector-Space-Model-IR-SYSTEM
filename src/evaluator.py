import os
import json
from collections import defaultdict


# ==== Precision@k ====
def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = [doc.lower().strip() for doc in retrieved_docs[:k]]
    relevant_set = set(doc.lower().strip() for doc in relevant_docs)
    true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
    return true_positives / k if k else 0.0


# ==== Recall@k ====
def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = [doc.lower().strip() for doc in retrieved_docs[:k]]
    relevant_set = set(doc.lower().strip() for doc in relevant_docs)
    if not relevant_set:
        return 0.0
    true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
    return true_positives / len(relevant_set)


# ==== F1@k ====
def f1_at_k(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

# ==== Precision-Recall Curve ====
def precision_recall_curve(retrieved_docs, relevant_docs):
    relevant_set = set(relevant_docs)
    precisions = []
    recalls = []
    true_positives = 0

    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_set:
            true_positives += 1
        prec = true_positives / (i + 1)
        rec = true_positives / len(relevant_set)
        precisions.append(prec)
        recalls.append(rec)

    return precisions, recalls

# ==== 11-Point Interpolated Precision ====
def eleven_point_interpolated_precision(precisions, recalls):
    interpolated = []
    for recall_level in [i / 10 for i in range(11)]:
        max_prec = max([p for p, r in zip(precisions, recalls) if r >= recall_level], default=0.0)
        interpolated.append(max_prec)
    return interpolated

# ==== Average Precision (for MAP) ====
def average_precision(retrieved_docs, relevant_docs):
    relevant_set = set(relevant_docs)
    hits = 0
    precision_sum = 0.0

    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / len(relevant_set) if relevant_set else 0.0

# ==== R-Precision ====
def r_precision(retrieved_docs, relevant_docs):
    R = len(relevant_docs)
    return precision_at_k(retrieved_docs, relevant_docs, k=R)

# ==== Main Evaluation Logic for One Query ====
def evaluate_query(query_id, query_text, relevant_docs, search_fn, method, k=5):
    retrieved = search_fn(query_text, top_k=k, method=method)
    retrieved_docs = [doc for doc, score in retrieved]

    prec = precision_at_k(retrieved_docs, relevant_docs, k)
    rec = recall_at_k(retrieved_docs, relevant_docs, k)
    f1 = f1_at_k(prec, rec)
    ap = average_precision(retrieved_docs, relevant_docs)
    rprec = r_precision(retrieved_docs, relevant_docs)
    pr_curve, rc_curve = precision_recall_curve(retrieved_docs, relevant_docs)
    interp_prec = eleven_point_interpolated_precision(pr_curve, rc_curve)

    return {
        "query": query_text,
        "method": method,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
        "r_precision": round(rprec, 4),
        "retrieved_docs": retrieved_docs,
        "relevant_docs": relevant_docs,
        "precision_curve": pr_curve,
        "recall_curve": rc_curve,
        "interpolated_precision": interp_prec
    }


# ==== Main Evaluation Logic for One Query ====
def evaluate_query(query_id, query_text, relevant_docs, search_fn, method, k=5):
    """
    Evaluate a single query using the provided search function and retrieval method.
    Returns dictionary with all relevant metrics and lists.
    """
    retrieved = search_fn(query_text, top_k=k, method=method)
    retrieved_docs = [doc for doc, score in retrieved]

    prec = precision_at_k(retrieved_docs, relevant_docs, k)
    rec = recall_at_k(retrieved_docs, relevant_docs, k)
    f1 = f1_at_k(prec, rec)

    return {
        "query": query_text,
        "method": method,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "retrieved_docs": retrieved_docs,
        "relevant_docs": relevant_docs
    }


# ==== Load Queries and Relevance Judgments ====
def load_queries_and_relevance(queries_folder, results_folder):
    """
    Loads all queries and their associated relevance judgments from disk.
    Assumes queries are in *.txt files and relevance in *_relevant.txt files.
    Returns:
        - queries: {query_id: query_text}
        - relevance: {query_id: [relevant_doc_ids]}
    """
    queries = {}
    relevance = {}

    for filename in os.listdir(queries_folder):
        if filename.endswith(".txt"):
            qid = filename.replace(".txt", "")
            with open(os.path.join(queries_folder, filename)) as f:
                queries[qid] = f.read().strip()

            rel_path = os.path.join(results_folder, f"{qid}_relevant.txt")
            if os.path.exists(rel_path):
                with open(rel_path) as rel_f:
                    relevance[qid] = [line.strip() for line in rel_f if line.strip()]

    return queries, relevance
