from pathlib import Path
import pickle
import re
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

knowledge_base = []

for file in Path("data").glob("*.md"):

    text = file.read_text(
        encoding="utf-8"
    )

    # clean markdown
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # simple chunking
    chunks = [
        text[i:i+500]
        for i in range(
            0,
            len(text),
            400
        )
    ]

    for chunk in chunks:

        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk
        )

        knowledge_base.append({
            "text": chunk,
            "embedding": response.embeddings[0].values
        })

with open(
    "embeddings.pkl",
    "wb"
) as f:
    pickle.dump(
        knowledge_base,
        f
    )

with open("embeddings.pkl","rb") as f:
    data = pickle.load(f)

print(len(data))