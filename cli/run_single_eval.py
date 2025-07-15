import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.search import search
from src.evaluator import precision_at_k, recall_at_k, f1_at_k

def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def evaluate_single_query(query_path, relevance_path, method="basic", k=5):
    #query_text = read_text(query_path)
    query_text = "aerodynamic heat transfer conical laminar turbulent"

    relevant_docs = set(read_text(relevance_path).splitlines())

    print(f"\n[Query]: {query_text}\n")

    start_time = time.time()
    results = search(query_text, top_k=k, method=method)
    elapsed = time.time() - start_time

    retrieved_docs = [doc for doc, _ in results]

    precision = precision_at_k(retrieved_docs, relevant_docs, k)
    recall = recall_at_k(retrieved_docs, relevant_docs, k)
    f1 = f1_at_k(precision, recall)

    print(f"[{method.upper()}] P@{k}: {precision:.4f}, R@{k}: {recall:.4f}, F1@{k}: {f1:.4f}, Time: {elapsed:.4f}s")
    print("Top Retrieved Docs:", retrieved_docs)
    print("Relevant Docs     :", list(relevant_docs))


# === 🔧 Customize this ===
evaluate_single_query(
    query_path="queries/query131.txt",
    relevance_path="results/query131_relevant.txt",
    method="basic",  # Try: basic / champion / cluster / impact / static / pseudo
    k=5
)

