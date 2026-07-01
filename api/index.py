# api/index.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI()

# Allows your Vercel HTML frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

class ChatRequest(BaseModel):
    question: str
    history: str = ""

@app.post("/api/chat")
def chat(data: ChatRequest):
    # 1. Search vector DB
    docs = db.similarity_search(data.question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Build prompt
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
    # 3. Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return {"answer": response.text}