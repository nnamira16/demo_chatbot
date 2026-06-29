import os

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

while True:
    question = input("\nAsk a question (or type quit): ")

    if question.lower() == "quit":
        break

    docs = db.similarity_search(question, k=4)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful Taelor website assistant.

Answer ONLY using the information below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you do not know.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:")
    print(response.text)