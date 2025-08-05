import os
import sys
import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.search import search, search_with_pseudo_feedback, compute_query_vector
from src.evaluator import precision_at_k, recall_at_k

top_k = 20
num_queries = 5

basic_precisions = []
basic_recalls = []
pseudo_precisions = []
pseudo_recalls = []

all_baseline_recalls, all_baseline_precisions = [], []
all_pseudo_recalls, all_pseudo_precisions = [], []

idf = json.load(open("index/idf.json"))

for i in range(1, num_queries + 1):
    query_file = f"queries/query{i}.txt"
    relevant_file = f"results/query{i}_relevant.txt"

    if not os.path.exists(query_file) or not os.path.exists(relevant_file):
        print(f"[Warning] Missing files for Query {i}, skipping.")
        continue

    with open(query_file, "r") as f:
        query = f.read().strip()

    with open(relevant_file, "r") as f:
        relevant_docs = [line.strip() for line in f if line.strip()]

    print(f"\n[Query {i}]: {query}")
    print(f"[Relevant Docs]: {relevant_docs}")

    # === Basic Retrieval ===
    baseline_results = search(query, top_k=top_k, method="basic")
    baseline_docs = [doc for doc, _ in baseline_results]

    precs, recs = [], []
    for k in range(1, top_k + 1):
        precs.append(precision_at_k(baseline_docs, relevant_docs, k))
        recs.append(recall_at_k(baseline_docs, relevant_docs, k))
    basic_precisions.append(precs[-1])
    basic_recalls.append(recs[-1])
    all_baseline_recalls.extend(recs)
    all_baseline_precisions.extend(precs)

    # === Pseudo-Relevance Feedback ===
    query_vector = compute_query_vector(query.split(), idf)
    pseudo_results = search_with_pseudo_feedback(query_vector, top_k=top_k, pseudo_k=10)
    pseudo_docs = [doc for doc, _ in pseudo_results]

    precs, recs = [], []
    for k in range(1, top_k + 1):
        precs.append(precision_at_k(pseudo_docs, relevant_docs, k))
        recs.append(recall_at_k(pseudo_docs, relevant_docs, k))
    pseudo_precisions.append(precs[-1])
    pseudo_recalls.append(recs[-1])
    all_pseudo_recalls.extend(recs)
    all_pseudo_precisions.extend(precs)

# === Interpolated 11-Point Precision ===
def interpolate_11_point(recalls, precisions):
    recall_levels = [i / 10 for i in range(11)]
    interpolated = []
    for r in recall_levels:
        precisions_at_r = [p for rec, p in zip(recalls, precisions) if rec >= r]
        interpolated.append(max(precisions_at_r) if precisions_at_r else 0)
    return recall_levels, interpolated

# Interpolate across combined queries
baseline_interp_x, baseline_interp_y = interpolate_11_point(all_baseline_recalls, all_baseline_precisions)
pseudo_interp_x, pseudo_interp_y = interpolate_11_point(all_pseudo_recalls, all_pseudo_precisions)

# === Plot 1: Interpolated PR Curve ===
plt.figure(figsize=(10, 6))
plt.plot(baseline_interp_x, baseline_interp_y, 'r--x', label="Basic (11-pt interp)")
plt.plot(pseudo_interp_x, pseudo_interp_y, 'b-o', label="Pseudo-Relevance (11-pt interp)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Interpolated 11-Point PR Curve (Avg. over Queries 1–5)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/avg_precision_recall_curve_q1_to_q5.png")
print("[Saved] Interpolated PR curve to plots/avg_precision_recall_curve_q1_to_q5.png")

# === Plot 2: Mean Precision and Recall ===
mean_prec_basic = sum(basic_precisions) / len(basic_precisions)
mean_rec_basic = sum(basic_recalls) / len(basic_recalls)
mean_prec_pseudo = sum(pseudo_precisions) / len(pseudo_precisions)
mean_rec_pseudo = sum(pseudo_recalls) / len(pseudo_recalls)

labels = ["Precision", "Recall"]
basic_scores = [mean_prec_basic, mean_rec_basic]
pseudo_scores = [mean_prec_pseudo, mean_rec_pseudo]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x, basic_scores, width, label="Basic", color='salmon')
plt.bar([i + width for i in x], pseudo_scores, width, label="Pseudo", color='skyblue')
plt.xticks([i + width / 2 for i in x], labels)
plt.ylabel("Score")
plt.title("Mean Precision & Recall (Queries 1–5)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/avg_precision_recall_bar_q1_to_q5.png")
print("[Saved] Mean Precision/Recall bar plot to plots/avg_precision_recall_bar_q1_to_q5.png")
