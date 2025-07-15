import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import search

def main():
    print("=== Vector Space Model: Search Interface ===")

    # Let user choose search method
    valid_methods = {"basic", "champion", "cluster"}
    method = input("Choose retrieval method [basic/champion/cluster]: ").strip().lower()

    if method not in valid_methods:
        print(f"[Error] Invalid method '{method}'. Using default: basic")
        method = "basic"

    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting search.")
            break

        top_k = 5  # Can be made user-configurable
        try:
            results = search.search(query, top_k=top_k, method=method)
        except NotImplementedError as e:
            print(f"[Error] {e}")
            continue

        if not results or all(score == 0 for _, score in results):
            print("No relevant documents found.")
        else:
            print(f"\nTop {top_k} results using '{method}' retrieval:")
            for rank, (doc_name, score) in enumerate(results, 1):
                print(f"{rank}. {doc_name} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
