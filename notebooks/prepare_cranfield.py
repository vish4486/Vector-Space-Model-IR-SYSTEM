import os
import re
from pathlib import Path

# Paths
raw_dir = "cranfield_raw"
output_docs_dir = "data/raw_docs"
output_queries_dir = "queries"
output_results_dir = "results"

# Ensure output directories exist
os.makedirs(output_docs_dir, exist_ok=True)
os.makedirs(output_queries_dir, exist_ok=True)
os.makedirs(output_results_dir, exist_ok=True)

# === STEP 1: Convert cran.all.1400 to doc1.txt ... doc1400.txt ===
with open(os.path.join(raw_dir, "cran.all.1400"), "r") as f:
    data = f.read()

docs = re.split(r"\.I\s+(\d+)", data)[1:]
for i in range(0, len(docs), 2):
    doc_id = int(docs[i])
    doc_content = docs[i + 1]
    cleaned = re.sub(r"\.(T|A|B|W)\n", "", doc_content).strip()
    with open(os.path.join(output_docs_dir, f"doc{doc_id}.txt"), "w") as out_f:
        out_f.write(cleaned)

print(f"[✓] Saved {len(docs)//2} documents to {output_docs_dir}")

# === STEP 2: Convert cran.qry to query1.txt ... query225.txt ===
with open(os.path.join(raw_dir, "cran.qry"), "r") as f:
    queries_raw = f.read()

queries = re.split(r"\.I\s+(\d+)", queries_raw)[1:]
for i in range(0, len(queries), 2):
    q_id = int(queries[i])
    q_text = re.sub(r"\.W\n", "", queries[i + 1]).strip()
    with open(os.path.join(output_queries_dir, f"query{q_id}.txt"), "w") as out_f:
        out_f.write(q_text)

print(f"[✓] Saved {len(queries)//2} queries to {output_queries_dir}")

# === STEP 3: Convert cranqrel to queryX_relevant.txt ===
relevance_dict = {}
with open(os.path.join(raw_dir, "cranqrel"), "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            qid, docid = int(parts[0]), int(parts[1])
            relevance_dict.setdefault(qid, []).append(f"doc{docid}.txt")  # No zero padding

for qid, docs in relevance_dict.items():
    with open(os.path.join(output_results_dir, f"query{qid}_relevant.txt"), "w") as out_f:
        for doc_name in docs:
            out_f.write(f"{doc_name}\n")

print(f"[✓] Saved relevance judgments to {output_results_dir}")
