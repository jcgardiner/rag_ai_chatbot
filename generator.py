import ollama
from config import LANGUAGE_MODEL

def generate_response(query: str, context_chunks: list[str]) -> str:
    prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. Don't make up any new information:\n\n"
        + "\n".join(f" - {chunk}" for chunk in context_chunks)
    )

    response = ""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk['message'].get('content')
        # print(f"Chunk content: {content}")  # Debug log to console
        if content:
            response += content

    if not response:
        print("⚠️ No response generated from the model.")
        return "Sorry, I could not generate a response."

    return response
