import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
MEMORY_FILE = "memories.json"


def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memories(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)


def add_memory(text):
    memories = load_memories()

    embedding = model.encode(text).tolist()

    memory = {
        "id": len(memories) + 1,
        "text": text,
        "embedding": embedding
    }

    memories.append(memory)
    save_memories(memories)


def search_memory(query, top_k=2):
    memories = load_memories()
    if not memories:
        return []

    query_embedding = model.encode(query)
    embeddings = np.array([m["embedding"] for m in memories])

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [memories[i] for i in top_indices]