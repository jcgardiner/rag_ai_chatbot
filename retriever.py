from typing import List, Tuple
from embedder import embed_text

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

def retrieve_relevant_chunks(query: str, vector_db: List[Tuple[str, List[float]]], top_n: int = 3) -> List[Tuple[str, float]]:
    query_embedding = embed_text(query)
    similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in vector_db]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]