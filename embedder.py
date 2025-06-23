import ollama
from config import EMBEDDING_MODEL

def embed_text(text: str) -> list[float]:
    return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]

def build_vector_db(chunks: list[str]) -> list[tuple[str, list[float]]]:
    vector_db = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vector_db.append((chunk, embedding))
        # print(f"📦 Embedded chunk {i+1}/{len(chunks)}")
    return vector_db

