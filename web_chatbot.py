import gradio as gr
import requests
import json
import faiss
import numpy as np
import traceback
from sentence_transformers import SentenceTransformer

# ====== Configuration ======
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:7b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/vector_data.json"

# ====== Load Metadata and FAISS Index ======
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print("📁 Loaded metadata files:", [m["title"] for m in metadata])

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

# ====== RAG Logic ======
def chatbot_rag(user_input):
    try:
        embedding = embed_model.encode([user_input]).astype("float32")
        D, I = index.search(embedding, k=3)

        valid_indices = [i for i in I[0] if 0 <= i < len(metadata)]
        if not valid_indices:
            return "⚠️ No matching notices found. Please update vector store."

        top_chunks = [metadata[i] for i in valid_indices]
        context = "\n".join([chunk["text"] for chunk in top_chunks])

        prompt = f"""You are a helpful assistant for Patliputra University.
Use the following official notices to answer the question clearly.

Notices:
{context}

Question:
{user_input}

Answer:"""

        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })

        if response.status_code != 200:
            return f"⚠️ Ollama error: {response.status_code} - {response.text}"

        data = response.json()
        return data.get("response", "⚠️ No response from model").strip()

    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"

# ====== Gradio Chat Interface ======
def gradio_chat(message, history):
    response = chatbot_rag(message)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history

chat_interface = gr.ChatInterface(
    fn=gradio_chat,
    title="📘 Patliputra University Chatbot (Gemma + RAG)",
    chatbot=gr.Chatbot(label="Patliputra University Bot", type="messages"),
    theme="soft",
    examples=["When is UG admission 2024 deadline?", "What is the holiday schedule?"],
    type="messages"
)

# ====== LAUNCH with Public + Local Links ======
print("🚀 Launching chatbot... Please wait for Gradio links.")
chat_interface.launch(share=True)
