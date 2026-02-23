import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_memories(user_input):

    prompt = f"""
Extract ONLY long-term personal facts about the user from the text below.

Rules:
- Return each fact on a separate line.
- Each fact must start with "User".
- Do NOT explain anything.
- If no personal facts exist, return NOTHING.

Text:
{user_input}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()

    if not text:
        return []

    facts = text.split("\n")

    # Filter strictly
    cleaned = []
    for fact in facts:
        fact = fact.strip()
        if fact.startswith("User"):
            cleaned.append(fact)

    return cleaned