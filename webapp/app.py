import streamlit as st
import time
import sys
import os
# Add project root to sys.path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions and data structures from the search module
from src.search import (
    search,
    compute_query_vector,
    DOC_VECTORS,
    rank_documents,
    search_with_feedback
)

import json

# === Load IDF dictionary for computing TF-IDF query vector ===
with open("index/idf.json", "r") as f:
    IDF = json.load(f)

# === Streamlit UI setup===
st.set_page_config(page_title="Vector Space IR System", layout="wide")
st.title("🔍 Vector Space Model IR System")

# === User Inputs ===
query = st.text_input("Enter your query")
method = st.selectbox(
    "Select Retrieval Method",
    ["basic", "champion", "cluster", "static", "impact", "pseudo", "feedback"]
)
top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)


# === State Variables for Feedback Flow ===
if 'feedback_stage' not in st.session_state:
    st.session_state.feedback_stage = 0
    st.session_state.initial_results = []
    st.session_state.selected_docs = []


# === When "Search" button is pressed ===
if st.button("Search"):
    # Reset feedback state
    st.session_state.feedback_stage = 0
    st.session_state.initial_results = []
    st.session_state.selected_docs = []

    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Convert raw query to TF-IDF vector
        query_vector = compute_query_vector(query.split(), IDF)  # basic tokenization

        if method == "feedback":
             # Save top-k results for user selection in feedback stage
            st.session_state.initial_results = rank_documents(query_vector, DOC_VECTORS, top_k)
            st.session_state.feedback_stage = 1
        else:
            # Perform regular search and show results
            start_time = time.time()
            results = search(query, top_k=top_k, method=method)
            end_time = time.time()

            st.success(f"Retrieved in {end_time - start_time:.4f} seconds")
            st.write("### Top Results:")
            for i, (doc, score) in enumerate(results, 1):
                st.write(f"**{i}.** `{doc}` — Score: {score:.4f}")


# === FEEDBACK STAGE ===
# Show top documents and let user select relevant ones
if st.session_state.feedback_stage == 1:
    st.write("### Initial Results — Select Relevant Documents")
    st.session_state.selected_docs = st.multiselect(
        "Mark documents as relevant:",
        [doc for doc, _ in st.session_state.initial_results]
    )
    # Refine using relevance feedback when user clicks the button
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
        # Reset feedback stage to avoid re-showing interface
        st.session_state.feedback_stage = 0
