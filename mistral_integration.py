import subprocess

def generate_answer_from_notices(query, notices):
    # Combine all chunks into a single context
    context = "\n".join([f"{i+1}. {n['text']}" for i, n in enumerate(notices)])

    # Create the prompt for Mistral
    prompt = f"""You are a helpful assistant for Patliputra University.

Use the following official university notices to answer the question. Be short, direct, and answer only from the notices.

Notices:
{context}

Question: {query}
Answer (short):"""

    # Run the Mistral model via Ollama CLI
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Decode and return the model's output
    output = result.stdout.decode("utf-8", errors="ignore")
    return output.split("Answer (short):")[-1].strip()
