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

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load the lightweight pre-calculated vectors
with open("embeddings.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

class ChatRequest(BaseModel):
    question: str
    history: str = ""

@app.post("/api/chat")
def chat(data: ChatRequest):
    # 1. Embed the incoming user question via Gemini's API
    q_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=data.question
    )
    query_vector = np.array(q_response.embeddings[0].values)

    # 2. Compute similarity manually using NumPy (Fast & Lightweight!)
    scored_chunks = []
    for item in knowledge_base:
        doc_vector = np.array(item["embedding"])
        # Cosine similarity calculation
        score = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
        scored_chunks.append((score, item["text"]))

    # Sort and pick the top 4 matches
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_docs = [chunk[1] for score, chunk in scored_chunks[:4]]
    context = "\n\n".join(top_docs)

    # 3. Formulate the prompt
    prompt = f"""
You are a helpful Taelor website assistant.
Use the retrieved website information below to answer questions.

Retrieved Context:
{context}

Recent Chat History:
{data.history}

User Question:
{data.question}

If the answer is not in the retrieved context, say you do not know.
"""

    # 4. Generate response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"answer": response.text}