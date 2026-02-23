from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from memory_store import add_memory, search_memory
from extractor import extract_memories
import os
from openai import OpenAI
from dotenv import load_dotenv


# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI(title="AI Memory Agent")

# -----------------------------
# Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static + Templates
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Models
# -----------------------------
class ChatInput(BaseModel):
    message: str


# -----------------------------
# Chat UI Route
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
def chat(chat_input: ChatInput):
    user_message = chat_input.message

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 1: Extract long-term memories
    facts = extract_memories(user_message)
    for fact in facts:
        add_memory(fact)

    # Step 2: Retrieve relevant memories
    memories = search_memory(user_message)
    memory_text = "\n".join([m["text"] for m in memories])

    if not memory_text:
        memory_text = "No stored memories available."

    # Step 3: Improved Prompt
    prompt = f"""
You are an AI assistant with long-term memory about the user.

Use stored memories when relevant.

If the user asks about themselves, summarize all known information clearly.

If the user asks for advice, personalize the answer using stored memories.

Stored Memories:
{memory_text}

User Query:
{user_message}

Respond naturally and intelligently.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"reply": response.choices[0].message.content}