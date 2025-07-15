import os
import json
import math
from collections import defaultdict
from src.preprocessing import preprocess_text
from src.utils import cosine_similarity


def read_documents(folder="data/raw_docs/"):
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                documents[filename] = preprocess_text(file.read())
    return documents

def compute_tf(documents):
    tf = {}
    for doc_id, tokens in documents.items():
        tf_doc = defaultdict(int)
        for token in tokens:
            tf_doc[token] += 1
        tf[doc_id] = dict(tf_doc)
    return tf

def compute_df(tf):
    df = defaultdict(int)
    for tf_doc in tf.values():
        for term in tf_doc:
            df[term] += 1
    return dict(df)

def compute_idf(df, N):
    return {term: math.log(N / df[term]) for term in df}

def compute_tfidf(tf, idf):
    tfidf = {}
    for doc_id, tf_doc in tf.items():
        tfidf_doc = {}
        for term, freq in tf_doc.items():
            tfidf_doc[term] = freq * idf.get(term, 0)
        tfidf[doc_id] = tfidf_doc
    return tfidf

def build_inverted_index(tfidf):
    index = defaultdict(list)
    for doc_id, terms in tfidf.items():
        for term, weight in terms.items():
            index[term].append((doc_id, weight))
    return dict(index)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
