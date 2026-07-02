# convert_index.py
import os
import pickle
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- THE HACK: Trick pickle into ignoring missing LangChain modules ---
class DummyModule:
    def __getattr__(self, name):
        return DummyClass

class DummyClass:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)

# Fake the langchain modules so pickle doesn't crash
sys.modules['langchain_community'] = DummyModule()
sys.modules['langchain_community.docstore'] = DummyModule()
sys.modules['langchain_community.docstore.in_memory'] = DummyModule()
sys.modules['langchain_core'] = DummyModule()
sys.modules['langchain_core.documents'] = DummyModule()
sys.modules['langchain_core.documents.base'] = DummyModule()
# ----------------------------------------------------------------------

print("🔄 Initializing Gemini client...")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

langchain_pkl_path = os.path.join("faiss_index", "index.pkl")

if not os.path.exists(langchain_pkl_path):
    print(f"❌ Error: Could not find {langchain_pkl_path}.")
    exit()

# print("📥 Unpacking LangChain's raw text data...")
# with open(langchain_pkl_path, "rb") as f:
#     faiss_data = pickle.load(f)
#     docstore = faiss_data[0]
print("📥 Unpacking LangChain's raw text data...")

with open(langchain_pkl_path, "rb") as f:
    faiss_data = pickle.load(f)
    docstore = faiss_data[0]

raw_documents = list(docstore._dict.values())

print("Document count:", len(raw_documents))

if len(raw_documents) > 0:
    print("First document type:", type(raw_documents[0]))
    print("First document dict:", raw_documents[0].__dict__)

# #debug
# print("Document count:", len(list(raw_documents)))

# raw_documents = list(raw_documents)

# print("First document type:", type(raw_documents[0]))
# print("First document dict:", raw_documents[0].__dict__) 


# # Extract the dictionary values safely
# raw_documents = docstore._dict.values()

knowledge_base = []

print(f"🧠 Generating Gemini cloud embeddings for {len(raw_documents)} chunks...")
for i, doc in enumerate(raw_documents):
    # Fallback in case the object structure changed slightly during unpickling
    #  NEW EXTRACTION LOGIC (Digs into the nested __dict__)
    doc_dict = getattr(doc, '__dict__', {})
    
    # Check if our dummy class double-nested the state
    if '__dict__' in doc_dict:
        text_content = doc_dict['__dict__'].get('page_content', None)
    else:
        text_content = doc_dict.get('page_content', None)
        
    # Standard fallback just in case
    if not text_content and hasattr(doc, 'page_content'):
        text_content = doc.page_content

    if not text_content:
        print(f"Warning: Could not find page_content for chunk {i+1}. Skipping.")
        continue

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text_content
    )
    vector = response.embeddings[0].values
    
    knowledge_base.append({
        "text": text_content,
        "embedding": vector
    })
    print(f"   Processed chunk {i+1}/{len(raw_documents)}")

output_file = "embeddings.pkl"
with open(output_file, "wb") as f:
    pickle.dump(knowledge_base, f)

print(f"\n✅ Success! '{output_file}' has been created.")