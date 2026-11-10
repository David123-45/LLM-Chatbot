# Aldric the Ancient — LLM Chatbot

A locally-running AI chatbot with a persistent memory system and document-aware RAG (Retrieval-Augmented Generation). The assistant persona is **Aldric the Ancient**, a wise wizard who speaks in archaic English and draws on ancient wisdom to answer questions.

---

## Features

- **Conversational AI** powered by `llama3.2` running locally via Ollama
- **Multi-source RAG** — answers are augmented with context from 3 sources: uploaded documents, past conversation summaries, and extracted user facts
- **PDF document upload** — attach PDFs to a conversation and ask questions about their content
- **Persistent memory** — conversations are summarized and stored; relevant memories are retrieved on future chats
- **User fact extraction** — the LLM automatically extracts goals, preferences, and projects from conversations
- **Memory panel** — view, edit, and delete stored facts and summaries from the UI
- **Multi-conversation** — sidebar with full conversation history, create and delete chats freely
- **Fully local** — no external API calls, everything runs on your machine

---

## Tech Stack

### Backend
| Library | Purpose |
|---|---|
| `FastAPI` | REST API framework |
| `uvicorn` | ASGI server |
| `langchain-ollama` | Interface to local Ollama LLM |
| `sentence-transformers` | Text embedding model (`all-MiniLM-L6-v2`) |
| `pypdf` | PDF text extraction |
| `numpy` | Cosine similarity computation |
| `sqlite3` | Embedded database (built-in) |

### Frontend
| Library | Purpose |
|---|---|
| `React 19` | UI framework |
| `Vite` | Build tool and dev server |
| `react-markdown` | Render markdown in bot responses |

---

## Project Structure

```
LLM - Chatbot/
├── app.py              # FastAPI backend — all API endpoints, RAG logic, DB
├── requirements.txt    # Python dependencies
├── tests.py            # Pytest test suite
├── chat.db             # SQLite database (auto-created on first run)
├── app.log             # Application log file (auto-created)
└── frontend/
    ├── src/
    │   ├── App.jsx     # Main React component
    │   ├── App.css     # Styles
    │   └── main.jsx    # Entry point
    ├── package.json
    └── vite.config.js
```

---

## Prerequisites

- [Python 3.10+](https://www.python.org/)
- [Node.js 18+](https://nodejs.org/)
- [Ollama](https://ollama.com/) installed and running locally with the `llama3.2` model pulled

```bash
ollama pull llama3.2
```

---

## Setup & Running

### 1. Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --reload
```

The backend will be available at `http://localhost:8000`.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## How RAG Works in This Project

The project implements a **multi-source RAG pipeline** with no external vector database — embeddings are stored directly in SQLite as binary blobs.

### Embedding Model

`all-MiniLM-L6-v2` from `sentence-transformers` is used to produce 384-dimensional float32 vectors for all stored text.

### Three Knowledge Sources

#### 1. Documents (`documents` table)
- User uploads a PDF via the UI
- The PDF text is extracted and split into **500-word chunks**
- Each chunk is embedded and stored, scoped to the active conversation
- On chat, the top 3 most relevant chunks are retrieved

#### 2. Memories (`memories` table)
- When the user starts a new chat, the current conversation is sent to the LLM for summarization
- The summary is embedded and stored globally (available across all conversations)
- On chat, the top 2 most relevant summaries are retrieved

#### 3. Facts (`facts` table)
- During summarization, the LLM also extracts structured facts about the user
- Facts are categorized as `goal`, `preference`, or `project`
- Each fact is individually embedded and stored globally
- On chat, the top 4 most relevant facts are retrieved

### Retrieval at Chat Time

Every `/chat` request runs semantic search against all 3 stores using the user's message as the query:

```
cosine_similarity(embed(user_message), stored_embedding)
```

The top results are injected into the system prompt before the LLM call:

```
[Base persona prompt]
+ Relevant memories from past conversations
+ Known facts about the user
+ Relevant context from uploaded documents
```

The LLM then generates a response with full awareness of all retrieved context.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/conversations` | List all conversations |
| `POST` | `/conversations` | Create a conversation |
| `DELETE` | `/conversations/{id}` | Delete a conversation and all its data |
| `GET` | `/conversations/{id}/messages` | Get all messages in a conversation |
| `POST` | `/chat` | Send a message and get a reply |
| `POST` | `/summarize` | Summarize a conversation and extract facts |
| `POST` | `/conversations/{id}/upload` | Upload a PDF document |
| `GET` | `/conversations/{id}/documents` | List uploaded documents |
| `DELETE` | `/conversations/{id}/documents/{filename}` | Delete a document |
| `GET` | `/facts` | List all extracted facts |
| `PATCH` | `/facts/{id}` | Edit a fact |
| `DELETE` | `/facts/{id}` | Delete a fact |
| `GET` | `/memories` | List all summaries |
| `DELETE` | `/memories/{id}` | Delete a summary |

---

## Database Schema

```sql
messages       (id, conversation_id, role, text, timestamp)
conversations  (id, title, created_at)
memories       (id, conversation_id, summary, embedding, created_at)
facts          (id, conversation_id, category, fact, embedding, created_at)
documents      (id, conversation_id, filename, chunk_index, chunk_text, embedding, created_at)
```

---

## Running Tests

```bash
pytest tests.py -v
```

Tests cover conversation CRUD, facts CRUD, memory CRUD, summarization edge cases, and document upload/delete. All tests use an isolated `test.db` database and clean up after each test.

---

## Notes

- Memories and facts are **global** — they persist across conversations and are retrieved based on semantic relevance, not conversation scope
- Document chunks are **conversation-scoped** — only chunks from the active conversation are searched
- Summarization is triggered automatically when starting a new chat from the UI
- The LLM and embedding model both run fully locally — no data leaves your machine
