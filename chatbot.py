import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# Load notices and FAISS index from the correct path
with open("vector_store/vector_data.json", "r", encoding="utf-8") as f:  # Updated path
    data = json.load(f)

index = faiss.read_index("vector_store/faiss.index")  # Updated path
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"✅ Loaded {len(data)} notices into memory.\n")
print("👋 Welcome to Patliputra University Chatbot!")
print("Type 'exit' to quit. Ask your question in Hindi, English, or Hinglish.\n")

# Fallback if no match is confident
fallback_info = """
❓ Sorry, I couldn’t find any official answer in the notices.

📞 Contact PPU:
- Helpline: +91-612-2350132
- Email: info@ppup.ac.in
- Website: https://www.ppup.ac.in
"""

def search_notices(query, top_k=5, threshold=0.7):  # Increased top_k and adjusted threshold
    query_vec = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)

    matches = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(data) and dist < (1 - threshold):  # 1 - distance for similarity
            matches.append(data[idx]["text"])
    return matches if matches else None  # Return None if no matches to handle separately

def ask_mistral(query, context):
    if not context:
        return "I’m sorry, I couldn’t find an official answer in the notices."
    
    prompt = f"""
You are a chatbot for Patliputra University.

ONLY use the information from the provided notices below. DO NOT use your own knowledge. 
If the answer is not present, reply with:
"I’m sorry, I couldn’t find an official answer in the notices."

### Notices:
{context}

### Question:
{query}

### Answer (based ONLY on the above notices):
"""
    response = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()

# Main loop
while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() == "exit":
        print("\n🤖 Chatbot: Thank you! Have a great day.")
        break

    matches = search_notices(user_input)
    if not matches:
        print("🤖 Chatbot:", fallback_info)
        continue

    context = "\n\n".join(matches)
    reply = ask_mistral(user_input, context)
    print("🤖 Chatbot:", reply)
