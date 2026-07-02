# api/index.py

import os
import pickle
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug
print("GEMINI KEY FOUND:", bool(os.getenv("GEMINI_API_KEY")))

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load embeddings
BASE_DIR = os.path.dirname(__file__)
EMBED_PATH = os.path.join(BASE_DIR, "..", "embeddings.pkl")

print("Loading:", EMBED_PATH)

with open(EMBED_PATH, "rb") as f:
    knowledge_base = pickle.load(f)

print(f"Loaded {len(knowledge_base)} chunks")


class ChatRequest(BaseModel):
    question: str
    history: str = ""


@app.get("/")
def root():
    return {"status": "Taelor chatbot running"}


@app.post("/api/chat")
@app.post("/chat")
def chat(data: ChatRequest):

    # Create embedding for question
    q_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=data.question
    )

    query_vector = np.array(
        q_response.embeddings[0].values
    )

    scored_chunks = []

    for item in knowledge_base:
        doc_vector = np.array(item["embedding"])

        score = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector)
            * np.linalg.norm(doc_vector)
        )

        scored_chunks.append(
            (score, item["text"])
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top_docs = [
        chunk[1]
        for chunk in scored_chunks[:4]
    ]

    # IMPORTANT DEBUGGING
    print("\nQUESTION:")
    print(data.question)

    print("\nTOP RETRIEVED CHUNK:")
    print(top_docs[0][:1000])

    context = "\n\n".join(top_docs)

    prompt = f"""
You are Taelor's friendly virtual assistant.

Answer using ONLY the information provided below.

If a useful link appears in the context,
include it in your answer.

Retrieved Context:
{context}

Chat History:
{data.history}

Question:
{data.question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "answer": response.text
    }