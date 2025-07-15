import os
import json
from collections import defaultdict


def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
    return true_positives / k


def recall_at_k(retrieved_docs, relevant_docs, k):
    relevant_set = set(relevant_docs)
    retrieved_k = retrieved_docs[:k]
    true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
    return true_positives / len(relevant_set) if relevant_set else 0.0


def f1_at_k(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def evaluate_query(query_id, query_text, relevant_docs, search_fn, method, k=5):
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


def load_queries_and_relevance(queries_folder, results_folder):
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
