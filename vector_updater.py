import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def update_vector_store():
    notices_dir = "notices"
    data_path = "data/vector_data.json"
    index_path = "data/faiss.index"

    # Make sure 'data/' folder exists
    os.makedirs("data", exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []

    for filename in os.listdir(notices_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(notices_dir, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                texts.append({"filename": filename, "text": content})

    # Save text data
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)

    # Compute and save vector index
    corpus = [x["text"] for x in texts]
    embeddings = model.encode(corpus).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    print("✅ Vector store updated.")

