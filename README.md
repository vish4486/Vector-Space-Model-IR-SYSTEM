# 🔍 Vector Space Model – Information Retrieval System

This is a fully functional Information Retrieval (IR) system based on the **Vector Space Model** with multiple retrieval strategies. It supports interactive search (CLI and Streamlit UI), relevance feedback, and evaluation metrics using the classic **Cranfield dataset**.

---

## 📂 Project Structure

Vector-Space-Model-IR-SYSTEM/
├── data/
│ └── raw_docs/ # 1400 documents (Cranfield corpus)
├── queries/ # 225 user queries
├── results/ # Relevance judgments for each query
├── index/ # All index files (TF-IDF, Champion Lists, etc.)
├── src/
│ ├── search.py # Core search logic and ranking functions
│ ├── preprocessing.py # Text cleaning and tokenization
│ ├── utils.py # Cosine similarity function
│ ├── spell_correction.py # Spelling correction module
│ └── relevance_feedback.py # Rocchio feedback for user and pseudo relevance
├── cli/
│ ├── run_indexer.py # Builds index from documents
│ ├── run_query.py # CLI search interface
│ ├── run_evaluation.py # Batch evaluation with metrics
│ ├── run_single_eval.py # One-query performance and timing comparison
│ └── plot_eval.py # Timing comparison bar chart
├── webapp/
│ └── app.py # Streamlit search UI
├── notebooks/
│ └── prepare_cranfield.py # Converts original Cranfield files to usable format
├── plots/
│ └── timing_comparison.png # Saved matplotlib chart comparing methods
└── README.md


---

## 💡 Features

- TF-IDF vectorization with cosine similarity
- Multiple retrieval strategies:
  - Basic (full-scan)
  - Champion Lists
  - Cluster Pruning
  - Static Quality Scores
  - Impact-Ordered Index
  - Relevance Feedback (Rocchio)
  - Pseudo-Relevance Feedback
- Spelling correction module
- Performance Evaluation (Precision, Recall, F1 @5)
- Query Time Plotting
- Streamlit Web UI for demo

---

## 📊 Indexing the Corpus

Before searching, you must build all necessary index files:

```bash
python cli/run_indexer.py
```
This will generate TF-IDF vectors, inverted index, champion lists, cluster leaders, static scores, and impact index into the index/ folder.

---

## 🔎 Running CLI Search
Interactive search with choice of method:
```
python cli/run_query.py
```
---

## 📈 Evaluate IR Methods (Optional)
Evaluate all methods on Cranfield relevance data:

```
python cli/run_evaluation.py
```
Or evaluate a single query and compare all methods and see the timing plots
Saved as: plots/query_time_comparison.png

```
python cli/run_single_eval.py
```
---

## 🌐 Launch Streamlit UI

```
streamlit run webapp/app.py
```

Features:

--Interactive query box

--Method selector

--Top-k results with scores

--Manual relevance marking (for feedback)

---

## 🧰 Dependencies
Install all dependencies (preferably in a virtual environment):

```
pip install -r requirements.txt
```

Main packages:

streamlit
matplotlib
numpy
nltk

## 📚 Dataset Info
This project uses the Cranfield Dataset containing:

1400 documents on aerodynamics
365 queries
225 relevance judgments

Converted using notebooks/prepare_cranfield.py.


## 📌 Credits
Developed by Vishal Nigam as part of academic coursework on Information Retrieval & Data Visualization at Trieste University.

## 📃 License
This project is for academic and educational purposes only. Not licensed for commercial use.






