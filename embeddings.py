from pathlib import Path
import pickle
import re
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")


print("Starts with:", key[:10] if key else None)
print("Length:", len(key) if key else None)

client = OpenAI(api_key=key)

knowledge_base = []

for file in Path("data").glob("*.md"):

    print(f"\nReading: {file}")

    text = file.read_text(
        encoding="utf-8"
    )

    text = re.sub(
    r"\[([^\]]+)\]\(([^)]+)\)",
    r"\1\nURL: \2",
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
    print(f"Found {len(chunks)} chunks")

    # for chunk in chunks:

    #     response = client.models.embed_content(
    #         model="gemini-embedding-2",
    #         contents=chunk
    #     )

    #     knowledge_base.append({
    #         "text": chunk,
    #         "embedding": response.embeddings[0].values
    #     })

    for i, chunk in enumerate(chunks):

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embedding = response.data[0].embedding

        knowledge_base.append({
            "text": chunk,
            "embedding": embedding
        })

        print(
            f"   Processed chunk {i+1}/{len(chunks)}"
        )

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