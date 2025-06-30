import json

# Load your old vector store
with open("vector_store.json", "r", encoding="utf-8") as file:
    old_data = json.load(file)

new_data = []

for item in old_data:
    new_item = {
        "english": item.get("query", ""),
        "hinglish": item.get("query", ""),  # Duplicate the English for now
        "answer": item.get("answer", "")
    }
    new_data.append(new_item)

# Save to new file
with open("vector_store_updated.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4, ensure_ascii=False)

print("✅ Conversion complete! Saved as vector_store_updated.json")
