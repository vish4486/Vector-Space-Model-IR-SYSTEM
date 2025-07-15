import os
import random

# Ensure the target directory exists
os.makedirs("data/raw_docs", exist_ok=True)

# Define fake topic areas and phrases
topics = [
    "climate change", "quantum computing", "renewable energy", "AI in healthcare",
    "space exploration", "cybersecurity", "blockchain", "data science",
    "financial technology", "biotechnology"
]

phrases = [
    "This paper explores", "An in-depth look at", "The challenges of",
    "Emerging trends in", "Potential risks of", "Future directions for",
    "Applications of", "The societal impact of", "Case studies on", "Historical development of"
]

# Generate 100 documents
for i in range(1, 101):
    filename = f"data/raw_docs/doc{i}.txt"
    with open(filename, "w") as f:
        for _ in range(8):  # 8 lines per document
            topic = random.choice(topics)
            phrase = random.choice(phrases)
            sentence = f"{phrase} {topic} and its implications in modern society.\n"
            f.write(sentence)

print("100 fake documents generated in data/raw_docs/")
