# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# #from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS


# from langchain_community.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# load_dotenv()

# with open("data/taelor.style_pages_onboarding.2026-06-18T05_36_02.870Z.md", "r", encoding="utf-8") as f:
#     text = f.read()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# docs = splitter.create_documents([text])

# #embeddings = OpenAIEmbeddings()

# db = FAISS.from_documents(
#     docs,
#     embeddings
# )

# db.save_local("faiss_index")

# print("Index created successfully!")

import os
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

st.title("Taelor AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    docs = db.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    history = ""

    for msg in st.session_state.messages[-6:]:
        history += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a helpful Taelor website assistant.

Use the retrieved website information below to answer questions.

Retrieved Context:
{context}

Recent Chat History:
{history}

User Question:
{question}

If the answer is not in the retrieved context, say you do not know.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()