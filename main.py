from memory_store import add_memory, search_memory
from extractor import extract_memories
import os
from openai import OpenAI
from dotenv import load_dotenv
from memory_store import save_memories
save_memories([])


load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_response(user_input):
    memories = search_memory(user_input)
    memory_text = "\n".join([m["text"] for m in memories])

    prompt = f"""
User query: {user_input}

Relevant memories:
{memory_text}

Generate a helpful and personalized response.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def run_chat():
    print("AI Memory Chatbot Started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Extract and store memories
        facts = extract_memories(user_input)

        for fact in facts:
            add_memory(fact)

        response = generate_response(user_input)
        print("AI:", response)


if __name__ == "__main__":
    run_chat()