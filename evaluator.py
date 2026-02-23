import json
from memory_store import add_memory, search_memory, save_memories


def run_evaluation():
    print("Resetting memory...")
    save_memories([])

    with open("test_data.json", "r") as f:
        data = json.load(f)

    print("Adding test memories...")
    for memory in data["memories"]:
        add_memory(memory)

    correct_top1 = 0
    correct_top3 = 0
    total = len(data["queries"])

    print("\nRunning evaluation...\n")

    for item in data["queries"]:
        query = item["query"]
        expected = item["expected"]

        results = search_memory(query, top_k=3)
        retrieved_texts = [r["text"] for r in results]

        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Retrieved Top-3: {retrieved_texts}")

        # Top-1 check
        if retrieved_texts and retrieved_texts[0] == expected:
            correct_top1 += 1

        # Top-3 check
        if expected in retrieved_texts:
            correct_top3 += 1

        print("-" * 40)

    top1_accuracy = correct_top1 / total
    top3_hit_rate = correct_top3 / total

    print("\nFinal Results:")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}")
    print(f"Top-3 Hit Rate: {top3_hit_rate:.2f}")


if __name__ == "__main__":
    run_evaluation()