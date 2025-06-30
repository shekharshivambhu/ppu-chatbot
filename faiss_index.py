import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "faiss_index.idx"
DATA_FOLDER = "notices"

def load_documents():
    documents = []
    filenames = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, file), 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                filenames.append(file)
    return documents, filenames

def create_or_load_index():
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open("docs.npy", "rb") as f:
            docs = np.load(f, allow_pickle=True)
        return index, docs.tolist()

    documents, _ = load_documents()
    embeddings = MODEL.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    with open("docs.npy", "wb") as f:
        np.save(f, documents)
    return index, documents

def search(query, k=1):
    index, documents = create_or_load_index()
    query_vector = MODEL.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    results = [documents[i] for i in indices[0]]
    return results
