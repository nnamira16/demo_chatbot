# api/index.py

import os
import pickle
import time
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

print("Starts with:", key[:10] if key else None)
print("Length:", len(key) if key else None)


# Debug
print("OPENAI KEY FOUND:", bool(os.getenv("OPENAI_API_KEY")))

# client = genai.Client(
#     api_key=os.getenv("GEMINI_API_KEY")
# )

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
    total_start = time.time()

    print("\n==============================")
    print("NEW CHAT REQUEST")
    print("==============================")

    # Create embedding for question
    embedding_start = time.time()

    # Create embedding for question
    q_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=data.question
    )

    embedding_time = time.time() - embedding_start
    print(f"Embedding time: {embedding_time:.2f} seconds")

    query_vector = np.array(
        q_response.data[0].embedding
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
    retrieval_time = time.time() - embedding_start

    print(f"Embedding + retrieval time: {retrieval_time:.2f} seconds")
    print(f"Knowledge base size: {len(knowledge_base)} chunks")

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

Do not inclue Skip to Conent. Or any other unrelated text. 

Do not make assumptions about a customer's gender, idenity, or ability to use TAELOR.
If a customer asks whether they can use Taelor based on gender, explain that Taelor specializes in menswear but anyone interested in the service is welcome to explore it.
If information is unclear, acknowledge the limitation and suggest contacting Taelor support for more guidance. 
Format guidance as follows:
- Separate ideas with a blank line.
- Use numbered steps for instructions.
- Use dot bullet points only when listing options.
- Never write one long paragraph.
- End with a friendly follow-up question when appropriate.

Make sure you answer word from word to the documents provided. 
Answer using ONLY the retrieved context.

Retrieved Context:
{context}

Chat History:
{data.history}

Question:
{data.question}
"""

    generation_start = time.time()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    generation_time = time.time() - generation_start

    print(f"OpenAI generation time: {generation_time:.2f} seconds")

    total_time = time.time() - total_start

    print(f"TOTAL REQUEST TIME: {total_time:.2f} seconds")

    return {
        "answer": response.output_text
    }