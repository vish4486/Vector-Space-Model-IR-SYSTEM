import os
import sys
import time
from collections import defaultdict

# Append parent directory to allow importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import search, evaluator


# === Helper: Load query text from file ===
def load_query(query_file):
    with open(query_file, 'r') as f:
        return f.read().strip()

# === Helper: Load relevant document names from relevance file ===
def load_relevant_docs(relevant_file):
    with open(relevant_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())



# === Main evaluation script for a single query ===
def main():
    queries_dir = "queries"
    results_dir = "results"
    top_k = 20

    query_id = "query1"  # <<== CHANGE THIS if needed
    query_file = os.path.join(queries_dir, f"{query_id}.txt")
    relevant_file = os.path.join(results_dir, f"{query_id}_relevant.txt")

    if not os.path.exists(query_file):
        print(f"[Error] Query file not found: {query_file}")
        return
    if not os.path.exists(relevant_file):
        print(f"[Error] Relevant file not found: {relevant_file}")
        return

    query_text = load_query(query_file)
    relevant_docs = load_relevant_docs(relevant_file)

    print(f"\nEvaluating query: {query_id} -> '{query_text}'")

    methods = ["basic", "champion", "cluster", "static", "impact", "pseudo"]
    summary = defaultdict(dict)

    for method in methods:
        start = time.time()
        try:
            results = search.search(query_text, top_k=top_k, method=method)
            print(f"    Top-{top_k} retrieved: {[doc for doc, _ in results]}")
            print(f"    Relevant: {relevant_docs}")
        except Exception as e:
            print(f"[Error] {method} on {query_id}: {e}")
            continue
        end = time.time()

        metrics = evaluator.evaluate_query(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            search_fn=search.search,
            method=method,
            k=top_k
        )

        summary[query_id][method] = {
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Time": round(end - start, 4)
        }

        print(f"  [{method}] P@{top_k}: {metrics['precision']:.4f}, "
              f"R@{top_k}: {metrics['recall']:.4f}, "
              f"F1@{top_k}: {metrics['f1']:.4f}, "
              f"Time: {end - start:.4f}s")

if __name__ == "__main__":
    main()
