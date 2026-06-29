import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("Taelor Knowledge Search Demo")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = st.text_input("Ask a question")

if question:
    docs = db.similarity_search(question, k=3)

    st.subheader("Most Relevant Results")

    for i, doc in enumerate(docs, start=1):
        st.markdown(f"### Result {i}")
        st.write(doc.page_content)