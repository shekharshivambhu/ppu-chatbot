# PPU Chatbot - Version 2 (Multilingual + Semantic Search)
import json
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
import torch

# Load vector store
with open("vector_store_updated.json", "r", encoding="utf-8") as f:
    vector_store = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

print("\U0001F44B Welcome to Patliputra University Chatbot!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("\U0001F9D1 You: ")

    if user_input.lower() == "exit":
        print("\U0001F916 Chatbot: Thank you! Have a great day.")
        break

    # Detect language
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    # Translate input to English for matching
    if lang != "en":
        translated_input = translator.translate(user_input, dest="en").text
    else:
        translated_input = user_input

    # Semantic search
    query_embedding = model.encode(translated_input, convert_to_tensor=True)
    best_score = 0.6
    best_answer = None

    for item in vector_store:
        text_tensor = torch.tensor(item["embedding"])
        score = util.cos_sim(query_embedding, text_tensor)[0][0].item()
        if score > best_score:
            best_score = score
            best_answer = item["text"]

    # Translate answer if necessary
    if best_answer:
        if lang != "en":
            final_answer = translator.translate(best_answer, dest=lang).text
        else:
            final_answer = best_answer
        print("\U0001F916 Chatbot:", final_answer)
    else:
        fallback = ("\u2753 Sorry, I couldn't find official information for your question.\n"
                    "\ud83d\udce7 Email: info@ppup.ac.in\n"
                    "\ud83d\udcf1 Helpline: 0612-2366003")
        if lang != "en":
            fallback = translator.translate(fallback, dest=lang).text
        print("\U0001F916 Chatbot:", fallback)
