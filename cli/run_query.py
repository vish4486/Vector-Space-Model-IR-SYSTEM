import os
import sys
import time

# Add project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import main search dispatcher from src
from src import search

def main():
    print("=== Vector Space Model: Search Interface ===")

    # === Load index files and measure disk read time ===
    start_disk = time.time()
    from src import search  # Triggers index file loading in src/search.py
    end_disk = time.time()
    disk_load_time = round(end_disk - start_disk, 4)
    print(f"[INFO] Time to load index from disk: {disk_load_time} seconds\n")

     # Friendly input mapped to internal method names
    method_mapping = {
        "basic": "basic",
        "champion": "champion",
        "cluster": "cluster",
        "static": "static",
        "impact": "impact",
        "relevance feedback": "feedback",
        "pseudo feedback": "pseudo"
    }

    # === Choose retrieval method ===
    method_input = input("Choose retrieval method [basic/champion/cluster/static/impact/relevance feedback/pseudo feedback]: ").strip().lower()
    method = method_mapping.get(method_input)

    if method is None:
        print(f"[Error] Invalid method '{method_input}'. Using default: basic")
        method = "basic"

    # === Query Loop ===
    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if not query:
            print("[Warning] Empty query. Please type something meaningful.")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting search.")
            break

        top_k = 5  # we can make this user-configurable

        try:
            start_query = time.time()
            results = search.search(query, top_k=top_k, method=method)
            end_query = time.time()
            query_exec_time = round(end_query - start_query, 4)
        except NotImplementedError as e:
            print(f"[Error] {e}")
            continue
        except Exception as e:
            print(f"[Unexpected Error] {e}")
            continue
        #====DISPLAY RESULTS=======
        if not results or all(score == 0 for _, score in results):
            print("No relevant documents found.")
        else:
            #print(f"\nTop {top_k} results using '{method}' retrieval:")
            print(f"\nTop {len(results)} results using '{method}' retrieval:")
            for rank, (doc_name, score) in enumerate(results, 1):
                print(f"{rank}. {doc_name} (Score: {score:.4f})")
            print(f"\n[INFO] Query execution time: {query_exec_time} seconds")

if __name__ == "__main__":
    main()
