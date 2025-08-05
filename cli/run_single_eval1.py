import os
import sys
import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search import search, search_with_pseudo_feedback, compute_query_vector, DOC_VECTORS
from src.evaluator import precision_at_k, recall_at_k

# === Settings ===
#query = "aerodynamic heat transfer conical laminar turbulent"
query = "aerodynamic heat transfer"
methods = ["basic", "champion", "cluster", "static", "impact", "pseudo"]
top_k = 20  # max rank for PR curve
#relevant_docs = ["doc50.txt", "doc101.txt", "doc125.txt", "doc57.txt", "doc113.txt"]  # manually define
relevant_docs = ['doc564.txt', 'doc435.txt', 'doc662.txt']  # from baseline and pseudo

print(f"\n[Query]: {query}\n")

# === 1. TIME PLOT ===
timings = {}

for method in methods:
    print(f"Running: {method}")
    start = time.time()
    _ = search(query, top_k=top_k, method=method)
    end = time.time()
    timings[method] = round(end - start, 4)

plt.figure(figsize=(10, 6))
plt.bar(timings.keys(), timings.values(), color="lightgreen")
plt.xlabel("Retrieval Method")
plt.ylabel("Time (seconds)")
plt.title("Query Execution Time per Retrieval Method")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/query_time_comparison.png")
print("Saved time plot to plots/query_time_comparison.png")

# === 2. PR CURVE: baseline vs pseudo ===
# Baseline ranking
baseline_results = search(query, top_k=top_k, method="basic")
baseline_docs = [doc for doc, _ in baseline_results]


# Pseudo feedback
idf = json.load(open("index/idf.json"))
query_vector = compute_query_vector(query.split(), idf)
pseudo_results = search_with_pseudo_feedback(query_vector, top_k=top_k, pseudo_k=5)
pseudo_docs = [doc for doc, _ in pseudo_results]

# Precision/Recall at each rank
baseline_prec, baseline_rec = [], []
pseudo_prec, pseudo_rec = [], []

print("[Baseline Retrieved Docs]:", baseline_docs)
print("[Pseudo Feedback Docs]:", pseudo_docs)
print("[Relevant Docs]:", relevant_docs)

for k in range(1, top_k + 1):
    baseline_prec.append(precision_at_k(baseline_docs, relevant_docs, k))
    baseline_rec.append(recall_at_k(baseline_docs, relevant_docs, k))
    pseudo_prec.append(precision_at_k(pseudo_docs, relevant_docs, k))
    pseudo_rec.append(recall_at_k(pseudo_docs, relevant_docs, k))


# === Interpolated 11-Point Precision ===
def interpolate_11_point(recalls, precisions):
    recall_levels = [i / 10 for i in range(11)]
    interpolated = []
    for r in recall_levels:
        precisions_at_r = [p for rec, p in zip(recalls, precisions) if rec >= r]
        interpolated.append(max(precisions_at_r) if precisions_at_r else 0)
    return recall_levels, interpolated

baseline_interp_x, baseline_interp_y = interpolate_11_point(baseline_rec, baseline_prec)
pseudo_interp_x, pseudo_interp_y = interpolate_11_point(pseudo_rec, pseudo_prec)

# === PR Curve Plot ===
plt.figure(figsize=(10, 6))
plt.plot(baseline_rec, baseline_prec, 'r--x', label="Baseline (PR)")
plt.plot(pseudo_rec, pseudo_prec, 'b-o', label="Pseudo-Relevance Feedback (PR)")

plt.plot(baseline_interp_x, baseline_interp_y, 'r--', alpha=0.6, label="Baseline (11-pt interp)")
plt.plot(pseudo_interp_x, pseudo_interp_y, 'b-', alpha=0.6, label="Pseudo (11-pt interp)")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve with 11-Point Interpolation")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plots/precision_recall_feedback.png")
print("Saved P–R curve to plots/precision_recall_feedback.png")
