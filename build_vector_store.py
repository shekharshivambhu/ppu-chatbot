import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === Configuration ===
NOTICE_DIR = "notices"
VECTOR_STORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")
DATA_FILE = os.path.join(VECTOR_STORE_DIR, "vector_data.json")

# === Load Embedding Model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Read .txt Files ===
print("📄 Reading all .txt files from 'notices/'...")
documents = []
for filename in os.listdir(NOTICE_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(NOTICE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                print(f"⚠️ Skipping empty file: {filename}")
                continue
            word_count = len(text.split())
            print(f"✅ Loaded {filename} ({word_count} words)")
            documents.append({"source": filename, "text": text})

# === Chunk Documents ===
print("\n✂️  Chunking documents by paragraph...\n")
chunks = []
for doc in documents:
    paragraphs = doc["text"].split("\n\n")  # Split by empty lines
    doc_chunks = 0
    for block in paragraphs:
        block = block.strip().replace("\n", " ")  # Normalize line breaks
        if len(block.split()) >= 10:  # Filter out very short lines
            chunks.append({
                "text": block,
                "source": doc["source"]
            })
            doc_chunks += 1
    print(f"📚 {doc['source']} → {doc_chunks} chunks")

# === Generate Embeddings ===
print("\n🧠 Generating embeddings...")
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts).astype("float32")

# === Build and Save FAISS Index ===
print("📦 Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Save Vector Store ===
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
faiss.write_index(index, INDEX_FILE)
with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# === Final Summary ===
print(f"\n✅ Vector store saved: {INDEX_FILE}, {DATA_FILE}")
print(f"🔢 Total embedded chunks: {len(chunks)}")

if chunks:
    print("📌 Example entry:\n")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
else:
    print("⚠️ No chunks were created. Check if the input files are too short or empty.")
