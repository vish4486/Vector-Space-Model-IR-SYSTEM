import streamlit as st
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.search import search, compute_query_vector, DOC_VECTORS, rank_documents, search_with_feedback
import json

# Load IDF for query vectorization
with open("index/idf.json", "r") as f:
    IDF = json.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Vector Space IR System", layout="wide")
st.title("🔍 Vector Space Model IR System")

query = st.text_input("Enter your query")
method = st.selectbox(
    "Select Retrieval Method",
    ["basic", "champion", "cluster", "static", "impact", "pseudo", "feedback"]
)
top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

if 'feedback_stage' not in st.session_state:
    st.session_state.feedback_stage = 0
    st.session_state.initial_results = []
    st.session_state.selected_docs = []

if st.button("Search"):
    st.session_state.feedback_stage = 0
    st.session_state.initial_results = []
    st.session_state.selected_docs = []

    if not query.strip():
        st.warning("Please enter a query.")
    else:
        query_vector = compute_query_vector(query.split(), IDF)  # basic tokenization

        if method == "feedback":
            st.session_state.initial_results = rank_documents(query_vector, DOC_VECTORS, top_k)
            st.session_state.feedback_stage = 1
        else:
            start_time = time.time()
            results = search(query, top_k=top_k, method=method)
            end_time = time.time()

            st.success(f"Retrieved in {end_time - start_time:.4f} seconds")
            st.write("### Top Results:")
            for i, (doc, score) in enumerate(results, 1):
                st.write(f"**{i}.** `{doc}` — Score: {score:.4f}")

# === FEEDBACK STAGE 2 ===
if st.session_state.feedback_stage == 1:
    st.write("### Initial Results — Select Relevant Documents")
    st.session_state.selected_docs = st.multiselect(
        "Mark documents as relevant:",
        [doc for doc, _ in st.session_state.initial_results]
    )
    if st.button("Refine using Feedback"):
        if st.session_state.selected_docs:
            st.write("### Re-ranked with Relevance Feedback")
            results = search_with_feedback(
                compute_query_vector(query.split(), IDF),
                st.session_state.selected_docs,
                top_k
            )
            for i, (doc, score) in enumerate(results, 1):
                st.write(f"**{i}.** `{doc}` — Score: {score:.4f}")
        else:
            st.warning("No relevant documents selected. Showing initial results.")
            for i, (doc, score) in enumerate(st.session_state.initial_results, 1):
                st.write(f"**{i}.** `{doc}` — Score: {score:.4f}")

        st.session_state.feedback_stage = 0
