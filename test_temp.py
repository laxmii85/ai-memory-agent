from memory_store import add_memory, search_memory, save_memories

print("Resetting memory...")
save_memories([])

print("Adding memories...")
add_memory("User loves Italian food")
add_memory("User allergic to peanuts")
add_memory("User lives in Mumbai")

print("\nSearching for relevant memory...\n")

results = search_memory("What should I eat for dinner?")

for r in results:
    print("Retrieved:", r["text"])