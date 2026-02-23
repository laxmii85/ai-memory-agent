# debug_retrieval.py

from memory_store import search_memory, load_memories

print("All memories:")
print(load_memories())

print("\nSearch result for 'Italian':")
print(search_memory("Italian"))

print("\nSearch result for 'Tell me about myself':")
print(search_memory("Tell me about myself"))