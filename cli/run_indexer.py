import os
import sys
import json
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import indexer

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

    # === NEW: Build and Save Champion Lists ===
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

    print(" Indexing completed.")
    print(f"Documents indexed: {len(docs)}")
    print(f"Vocabulary size: {len(idf)}")
    print(f"Index files saved to: index/")

if __name__ == "__main__":
    main()
