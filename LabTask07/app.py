
import streamlit as st
import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

""

# Load sentence embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data
df = pd.read_csv("dataset_july_20.csv")

# Helper functions
@st.cache_data(show_spinner="Embedding submissions...")
def embed_texts(texts: list, _model) -> np.array:
    doc_embeddings = _model.encode(texts, show_progress_bar=False)
    return doc_embeddings

embeddings = embed_texts(texts=df["activity"].to_list(), _model=model)

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T

tab1, tab2 = st.tabs(["Semantic Similarity", "Zero-shot with Gemma3"])

with tab1:
    query = st.text_input("Type your query: ")

    if query:
        query_embedding = model.encode(query)
        sims = cosine_similarity_matrix(query_embedding.reshape(1, -1), embeddings)[0]

        results = df.copy()
        results["similarity"] = sims
        results = results.sort_values("similarity", ascending=False).head(5)

        st.subheader("Most similar activities")
        for i, row in results.iterrows():
            st.write(row["activity"])

with tab2:
    st.write("Using Gemma3: ")
    labels_text = st.text_area("Categories", height=160)
    st.write(labels_text)
    # category_names, category_descriptions = [], []

    # for line in labels_text.strip().split("\n"):
    #     if ":" not in line:
    #         continue
    #     name, desc = line.split(":", 1)
    #     category_names.append(name.strip())
    #     category_descriptions.append(desc.strip())
    #     st.write(category_names)
    #     st.write(category_descriptions)
    texts = df["activity"].to_list()[:5]
    texts_to_inject = "\n".join(texts)
    prompt = f"""
Your task is to classify any given document based on the following categories:
{labels_text}

You must classify the following:
{texts_to_inject}

Your output MUST be an ARRAY OF JSON objects with the following schema:
{{"the original text": "category"}}
"""
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    st.write(response["message"]["content"])