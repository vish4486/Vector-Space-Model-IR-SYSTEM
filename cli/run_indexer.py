import os
import sys
import json
from collections import defaultdict
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import indexer
from src.utils import cosine_similarity


def main():
    print("Reading and preprocessing documents...")
    docs = indexer.read_documents()

    print("Computing term frequencies (TF)...")
    tf = indexer.compute_tf(docs)

    print("Computing document frequencies (DF)...")
    df = indexer.compute_df(tf)

    print("Computing inverse document frequencies (IDF)...")
    idf = indexer.compute_idf(df, N=len(docs))

    print("Computing TF-IDF document vectors...")
    tfidf_vectors = indexer.compute_tfidf(tf, idf)

    print("Building inverted index...")
    inverted_index = indexer.build_inverted_index(tfidf_vectors)

    print("Saving index and vectors to disk...")
    os.makedirs("index", exist_ok=True)
    indexer.save_json(tfidf_vectors, "index/doc_vectors.json")
    indexer.save_json(inverted_index, "index/tfidf_index.json")
    indexer.save_json(idf, "index/idf.json")

    # === Build and Save Champion Lists ===
    print("Building champion lists...")
    R = 5  # Top R documents per term
    champion_lists = defaultdict(list)

    for doc_name, vector in tfidf_vectors.items():
        for term, weight in vector.items():
            champion_lists[term].append((doc_name, weight))

    for term in champion_lists:
        champion_lists[term].sort(key=lambda x: x[1], reverse=True)
        champion_lists[term] = champion_lists[term][:R]

    with open("index/champion_lists.json", "w") as f:
        json.dump(champion_lists, f)

    print(f"Champion lists saved to index/champion_lists.json")

    # === Cluster Pruning ===
    print("Building cluster pruning leaders and followers...")
    random.seed(42)
    doc_names = list(tfidf_vectors.keys())
    num_leaders = max(1, len(doc_names) // 3)
    leaders = random.sample(doc_names, num_leaders)
    leader_followers = defaultdict(list)

    for doc_name in doc_names:
        if doc_name in leaders:
            continue
        best_leader = max(
            leaders,
            key=lambda l: cosine_similarity(tfidf_vectors[doc_name], tfidf_vectors[l])
        )
        leader_followers[best_leader].append(doc_name)

    indexer.save_json(leaders, "index/leaders.json")
    indexer.save_json(leader_followers, "index/leader_followers.json")
    print("Cluster pruning structures saved to index/")

    # === Generate and Save Static Quality Scores ===
    print("Generating static quality scores...")
    static_scores = {doc: round(random.uniform(0, 1), 4) for doc in tfidf_vectors}
    indexer.save_json(static_scores, "index/static_quality_scores.json")
    print("Static quality scores saved to index/static_quality_scores.json")

    # === Build and Save Impact-Ordered Index ===
    print("Building impact-ordered index...")
    impact_index = defaultdict(list)
    for doc_name, vector in tfidf_vectors.items():
        for term, weight in vector.items():
            impact_index[term].append((doc_name, weight))

    # Sort each posting list in descending order of weight
    for term in impact_index:
        impact_index[term].sort(key=lambda x: x[1], reverse=True)

    with open("index/impact_index.json", "w") as f:
        json.dump(impact_index, f)

    print("Impact-ordered index saved to index/impact_index.json")
    
    print("Indexing completed.")
    print(f"Documents indexed: {len(docs)}")
    print(f"Vocabulary size: {len(idf)}")
    print("Index files saved to: index/")

if __name__ == "__main__":
    main()
