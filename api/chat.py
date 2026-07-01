import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

index = faiss.read_index(
    os.path.join(BASE_DIR, "faiss_index", "index.faiss")
)

with open(os.path.join(BASE_DIR, "faiss_index", "chunks.json"), "r") as f:
    chunks = json.load(f)


def handler(request):
    try:
        
        body = json.loads(request.body)
        query = body.get("message", "")

        # embed
        q_emb = model.encode([query])
        q_emb = np.array(q_emb).astype("float32")

        # search
        _, I = index.search(q_emb, k=3)

        results = [
            chunks[i] for i in I[0] if i < len(chunks)
        ]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "response": "\n\n".join(results) if results else "No results found."
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }