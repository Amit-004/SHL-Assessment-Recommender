from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


from app.agent import handle_chat


app = FastAPI()


# -----------------------------
# Request Schema
# -----------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    return handle_chat(request.messages)