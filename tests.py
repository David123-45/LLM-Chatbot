import os
import pytest
from fastapi.testclient import TestClient

TEST_DB = "test.db"
os.environ["TEST_DB"] = TEST_DB

import app as application
application.DB_PATH = TEST_DB
application.init_db(TEST_DB)

from app import app as fastapi_app

client = TestClient(fastapi_app)

CONV_ID = "test-conv-001"

@pytest.fixture(autouse=True)
def clean_db():
    yield
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    for table in ["messages", "conversations", "memories", "facts", "documents"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()

# --- Conversations ---

def test_create_and_list_conversation():
    r = client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test Chat"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/conversations")
    ids = [c["id"] for c in r.json()]
    assert CONV_ID in ids

def test_delete_conversation():
    client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test Chat"})
    r = client.delete(f"/conversations/{CONV_ID}")
    assert r.json()["ok"] is True

    r = client.get("/conversations")
    assert all(c["id"] != CONV_ID for c in r.json())

def test_get_messages_empty():
    client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test Chat"})
    r = client.get(f"/conversations/{CONV_ID}/messages")
    assert r.status_code == 200
    assert r.json() == []

# --- Facts ---

def test_facts_crud():
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    conn.execute(
        "INSERT INTO facts (conversation_id, category, fact) VALUES (?, ?, ?)",
        (CONV_ID, "goal", "User wants to build a chatbot")
    )
    conn.commit()
    conn.close()

    r = client.get("/facts")
    assert r.status_code == 200
    facts = r.json()
    assert len(facts) == 1
    fact_id = facts[0]["id"]

    r = client.patch(f"/facts/{fact_id}", json={"fact": "User wants to deploy a chatbot"})
    assert r.json()["ok"] is True

    r = client.get("/facts")
    assert r.json()[0]["fact"] == "User wants to deploy a chatbot"

    r = client.delete(f"/facts/{fact_id}")
    assert r.json()["ok"] is True

    r = client.get("/facts")
    assert r.json() == []

# --- Memories ---

def test_memories_crud():
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    conn.execute(
        "INSERT INTO memories (conversation_id, summary) VALUES (?, ?)",
        (CONV_ID, "User is a developer building an LLM chatbot")
    )
    conn.commit()
    conn.close()

    r = client.get("/memories")
    assert r.status_code == 200
    memories = r.json()
    assert len(memories) == 1
    mem_id = memories[0]["id"]

    r = client.delete(f"/memories/{mem_id}")
    assert r.json()["ok"] is True

    r = client.get("/memories")
    assert r.json() == []

# --- Summarize ---

def test_summarize_too_short():
    client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test"})
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    conn.execute("INSERT INTO messages (conversation_id, role, text) VALUES (?, ?, ?)", (CONV_ID, "user", "Hi"))
    conn.commit()
    conn.close()

    r = client.post("/summarize", json={"conversationId": CONV_ID})
    assert r.json()["ok"] is False
    assert r.json()["reason"] == "Not enough messages to summarize"

# --- Documents ---

def test_upload_non_pdf():
    client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test"})
    r = client.post(
        f"/conversations/{CONV_ID}/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert r.status_code == 400

def test_get_documents_empty():
    client.post("/conversations", json={"conversationId": CONV_ID, "title": "Test"})
    r = client.get(f"/conversations/{CONV_ID}/documents")
    assert r.status_code == 200
    assert r.json() == []

def test_delete_document():
    import sqlite3, numpy as np
    conn = sqlite3.connect(TEST_DB)
    dummy_vec = np.zeros(384, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO documents (conversation_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
        (CONV_ID, "test.pdf", 0, "some text", dummy_vec)
    )
    conn.commit()
    conn.close()

    r = client.delete(f"/conversations/{CONV_ID}/documents/test.pdf")
    assert r.json()["ok"] is True

    r = client.get(f"/conversations/{CONV_ID}/documents")
    assert r.json() == []
