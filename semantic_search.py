import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load vector data (notices) and FAISS index
with open("vector_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} notices into memory.")

index = faiss.read_index("faiss.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fallback response
fallback_info = """
❓ Sorry, I couldn’t find any official answer in the notices.

📞 Contact PPU:
- Helpline: +91-612-2350132
- Email: info@ppup.ac.in
- Website: https://www.ppup.ac.in
"""

def get_most_relevant_answer(query, top_k=3, threshold=0.65):
    # Embed the query
    query_vector = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_vector]), top_k)


    for dist, i in zip(D[0], I[0]):
        if i < len(data):
            print(f"• Rank: {i}, Distance: {dist:.4f}")
            print(f"  Text: {data[i].get('text', '')[:100]}...")

    for dist, i in zip(D[0], I[0]):
        if i < len(data) and dist <= threshold:
            return data[i].get("text", "")

    return fallback_info

