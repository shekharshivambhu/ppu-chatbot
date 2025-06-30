# update_vector_store.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

notices_dir = "new_notices"
faiss_index_path = "vectorstore/index.faiss"
metadata_path = "vectorstore/vector_data.json"
embed_model_name = "all-MiniLM-L6-v2"

print(f"📄 Reading all .txt files from '{notices_dir}'...")

documents = []
file_names = []

for fname in os.listdir(notices_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(notices_dir, fname), encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                documents.append(text)
                file_names.append(fname)
                print(f"✅ Loaded {fname} ({len(text.split())} words)")

if not documents:
    raise ValueError(f"❌ No .txt files found or files are empty in '{notices_dir}'!")

# Embedding
print(f"🧠 Generating embeddings...")
embedder = SentenceTransformer(embed_model_name)
embeddings = embedder.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS Index
print(f"📦 Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, faiss_index_path)

# Save metadata
metadata = [{"text": text, "title": fname} for text, fname in zip(documents, file_names)]
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ Vector store saved: {faiss_index_path}, {metadata_path}")
print(f"🔢 Total embedded chunks: {len(metadata)}")
print(f"📌 Example:\n{json.dumps(metadata[0], indent=2)}")
