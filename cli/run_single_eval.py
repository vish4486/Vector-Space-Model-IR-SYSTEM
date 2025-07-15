import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.search import search

# Query and methods
query = "aerodynamic heat transfer conical laminar turbulent"
methods = ["basic", "champion", "cluster", "static", "impact", "pseudo"]
top_k = 5

print(f"\n[Query]: {query}\n")
# Store timing results
timings = {}

# Execute each method and record timing
for method in methods:
    print(f"Running: {method}")
    start = time.time()
    _ = search(query, top_k=top_k, method=method)
    end = time.time()
    timings[method] = round(end - start, 4)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(timings.keys(), timings.values(), color="lightgreen")
plt.xlabel("Retrieval Method")
plt.ylabel("Time (seconds)")
plt.title("Query Execution Time per Retrieval Method")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save
plt.savefig("plots/query_time_comparison.png")
print("[✓] Saved time comparison plot to plots/query_time_comparison.png")

