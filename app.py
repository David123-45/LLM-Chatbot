import sqlite3
import json
import re
import time
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- App ---

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = ChatOllama(model="llama3.2")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are Aldric the Ancient, a wise and mystical wizard who has lived for thousands of years.
You speak in an old, eloquent manner using archaic words like 'thee', 'thou', 'hath', 'dost' and 'verily'.
You relate all answers to ancient wisdom, magic, and the arcane arts, yet your answers are always accurate and helpful.
You occasionally reference your long life experiences and mystical adventures."""

DB_PATH = "chat.db"

# --- DB setup ---

def get_db(path: str = None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(path: str = None):
    conn = get_db(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            category TEXT NOT NULL,
            fact TEXT NOT NULL,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Migrate existing tables
    for table, col in [("facts", "embedding"), ("memories", "embedding")]:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} BLOB")
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", path or DB_PATH)

init_db()

# --- DB helpers ---

def save_message(conversation_id: str, role: str, text: str, db: str = None):
    conn = get_db(db)
    conn.execute(
        "INSERT INTO messages (conversation_id, role, text) VALUES (?, ?, ?)",
        (conversation_id, role, text)
    )
    conn.commit()
    conn.close()

def load_messages(conversation_id: str, db: str = None):
    conn = get_db(db)
    rows = conn.execute(
        "SELECT role, text FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [{"role": row["role"], "text": row["text"]} for row in rows]

def save_conversation(conversation_id: str, title: str, db: str = None):
    conn = get_db(db)
    conn.execute("INSERT OR IGNORE INTO conversations (id, title) VALUES (?, ?)", (conversation_id, title))
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    conn.commit()
    conn.close()

# --- Embedding helpers ---

def to_blob(vec) -> bytes:
    return vec.astype(np.float32).tobytes()

def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def semantic_search(query: str, rows, top_k: int = 3):
    if not rows:
        return []
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    scored = [
        (cosine_similarity(query_vec, from_blob(row["embedding"])), row)
        for row in rows if row["embedding"]
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]

# --- PDF chunking ---

CHUNK_SIZE = 500

def chunk_text(text: str):
    words = text.split()
    return [
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0, len(words), CHUNK_SIZE)
    ]

# --- Fact extraction ---

FACT_CATEGORIES = ["goal", "preference", "project"]

def extract_facts(conversation_id: str, transcript: str, db: str = None):
    prompt = (
        "Read this conversation and extract facts about the user. "
        "Return ONLY a JSON array, no explanation, no markdown. Each item must have 'category' and 'fact'. "
        "category must be one of: goal, preference, project. "
        "Only include facts clearly stated by the user. If none found, return [].\n\n"
        "Example: [{\"category\": \"goal\", \"fact\": \"User wants to learn Python\"}, {\"category\": \"preference\", \"fact\": \"User prefers short answers\"}]\n\n"
        + transcript
    )
    response = model.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        facts = json.loads(match.group()) if match else []
    except Exception:
        facts = []

    conn = get_db(db)
    for item in facts:
        if item.get("category") in FACT_CATEGORIES and item.get("fact"):
            vec = embed_model.encode(item["fact"], convert_to_numpy=True)
            conn.execute(
                "INSERT INTO facts (conversation_id, category, fact, embedding) VALUES (?, ?, ?, ?)",
                (conversation_id, item["category"], item["fact"], to_blob(vec))
            )
    conn.commit()
    conn.close()
    logger.info("Extracted %d facts for conversation %s", len(facts), conversation_id)

# --- Models ---

class ChatRequest(BaseModel):
    conversationId: str
    message: str

class ConversationRequest(BaseModel):
    conversationId: str
    title: str

class SummarizeRequest(BaseModel):
    conversationId: str

class FactUpdateRequest(BaseModel):
    fact: str

# --- Endpoints ---

@app.get("/conversations")
def get_conversations():
    conn = get_db()
    rows = conn.execute("SELECT id, title FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": row["id"], "title": row["title"]} for row in rows]

@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str):
    return load_messages(conversation_id)

@app.post("/conversations")
def create_conversation(req: ConversationRequest):
    save_conversation(req.conversationId, req.title)
    return {"ok": True}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    conn = get_db()
    for table in ["messages", "conversations", "memories", "facts", "documents"]:
        col = "id" if table == "conversations" else "conversation_id"
        conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (conversation_id,))
    conn.commit()
    conn.close()
    logger.info("Deleted conversation %s", conversation_id)
    return {"ok": True}

@app.get("/facts")
def get_facts():
    conn = get_db()
    rows = conn.execute("SELECT id, category, fact FROM facts ORDER BY category, created_at DESC").fetchall()
    conn.close()
    return [{"id": row["id"], "category": row["category"], "fact": row["fact"]} for row in rows]

@app.patch("/facts/{fact_id}")
def update_fact(fact_id: int, req: FactUpdateRequest):
    conn = get_db()
    conn.execute("UPDATE facts SET fact = ? WHERE id = ?", (req.fact, fact_id))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/facts/{fact_id}")
def delete_fact(fact_id: int):
    conn = get_db()
    conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/memories")
def get_memories():
    conn = get_db()
    rows = conn.execute("SELECT id, summary FROM memories ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": row["id"], "summary": row["summary"]} for row in rows]

@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: int):
    conn = get_db()
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/conversations/{conversation_id}/upload")
async def upload_document(conversation_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    reader = PdfReader(BytesIO(contents))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    chunks = chunk_text(full_text)
    conn = get_db()
    for i, chunk in enumerate(chunks):
        vec = embed_model.encode(chunk, convert_to_numpy=True)
        conn.execute(
            "INSERT INTO documents (conversation_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, file.filename, i, chunk, to_blob(vec))
        )
    conn.commit()
    conn.close()
    logger.info("Uploaded %s (%d chunks) for conversation %s", file.filename, len(chunks), conversation_id)
    return {"ok": True, "filename": file.filename, "chunks": len(chunks)}

@app.get("/conversations/{conversation_id}/documents")
def get_documents(conversation_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT filename FROM documents WHERE conversation_id = ?",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [{"filename": row["filename"]} for row in rows]

@app.delete("/conversations/{conversation_id}/documents/{filename}")
def delete_document(conversation_id: str, filename: str):
    conn = get_db()
    conn.execute(
        "DELETE FROM documents WHERE conversation_id = ? AND filename = ?",
        (conversation_id, filename)
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/summarize")
def summarize_conversation(req: SummarizeRequest):
    history = load_messages(req.conversationId)
    if len(history) < 2:
        return {"ok": False, "reason": "Not enough messages to summarize"}

    conn = get_db()
    already = conn.execute("SELECT id FROM memories WHERE conversation_id = ?", (req.conversationId,)).fetchone()
    conn.close()
    if already:
        return {"ok": False, "reason": "Already summarized"}

    transcript = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in history)
    prompt = (
        "Read this conversation and extract a concise bullet-point memory summary. "
        "Focus on facts about the user: their name, interests, preferences, topics they care about, and any important context. "
        "Be brief. Output only the bullet points, nothing else.\n\n"
        + transcript
    )
    response = model.invoke([HumanMessage(content=prompt)])
    summary = response.content.strip()

    vec = embed_model.encode(summary, convert_to_numpy=True)
    conn = get_db()
    conn.execute(
        "INSERT INTO memories (conversation_id, summary, embedding) VALUES (?, ?, ?)",
        (req.conversationId, summary, to_blob(vec))
    )
    conn.commit()
    conn.close()

    extract_facts(req.conversationId, transcript)
    logger.info("Summarized conversation %s", req.conversationId)
    return {"ok": True, "summary": summary}

@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    save_message(req.conversationId, "user", req.message)
    history = load_messages(req.conversationId)

    conn = get_db()
    memory_rows = conn.execute("SELECT id, summary, embedding FROM memories ORDER BY created_at DESC").fetchall()
    fact_rows = conn.execute("SELECT id, category, fact, embedding FROM facts ORDER BY category, created_at DESC").fetchall()
    doc_rows = conn.execute(
        "SELECT chunk_text, embedding FROM documents WHERE conversation_id = ?",
        (req.conversationId,)
    ).fetchall()
    conn.close()

    relevant_memories = semantic_search(req.message, memory_rows, top_k=2)
    relevant_facts = semantic_search(req.message, fact_rows, top_k=4)
    relevant_chunks = semantic_search(req.message, doc_rows, top_k=3)

    system_prompt = SYSTEM_PROMPT
    if relevant_memories:
        memories_text = "\n\n".join(row["summary"] for row in relevant_memories)
        system_prompt += f"\n\nWhat thou rememberest about this user from past conversations:\n{memories_text}"
    if relevant_facts:
        grouped = {}
        for row in relevant_facts:
            grouped.setdefault(row["category"], []).append(row["fact"])
        facts_text = "\n".join(
            f"{cat.upper()}S:\n" + "\n".join(f"  - {f}" for f in items)
            for cat, items in grouped.items()
        )
        system_prompt += f"\n\nKnown facts about this user:\n{facts_text}"
    if relevant_chunks:
        doc_context = "\n\n".join(row["chunk_text"] for row in relevant_chunks)
        system_prompt += f"\n\nRelevant context from uploaded documents:\n{doc_context}"

    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["text"]))
        elif msg["role"] == "bot":
            lc_messages.append(AIMessage(content=msg["text"]))

    response = model.invoke(lc_messages)
    reply = response.content

    save_message(req.conversationId, "bot", reply)
    save_conversation(req.conversationId, req.message[:30])

    elapsed = round(time.time() - start, 2)
    logger.info("Chat [conv=%s] responded in %ss | facts=%d memories=%d doc_chunks=%d",
                req.conversationId, elapsed, len(relevant_facts), len(relevant_memories), len(relevant_chunks))

    return {"reply": reply}
