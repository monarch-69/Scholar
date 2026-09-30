# Scholar

A web app for uploading research papers (PDFs) and asking questions about their content using an agentic AI loop. The agent retrieves passages from your indexed papers using specialized tools and streams its reasoning and answer to the UI in real time.

## What it does

- **Upload PDFs** - accepted immediately and indexed in the background; you can keep using the app while it processes
- **Track indexing** - each paper's indexing status (processing / done / failed) is stored in PostgreSQL
- **ReAct agent loop** - instead of a single retrieval pass, a LangGraph agent decides which tool to call, sees the results, and decides whether to search again or answer. This loop repeats until the agent has enough information, capped at 5 tool calls
- **Four retrieval tools the agent can choose from:**
  - `search_papers` - general semantic search across all selected papers
  - `compare_papers` - searches each paper individually for a specific aspect so the agent can compare them side by side
  - `extract_structured` - targeted retrieval for specific fields: methodology, datasets, results, limitations, contributions, or future work
  - `find_contradictions` - retrieves per paper on a topic so the agent can identify conflicting claims
- **Streaming** - tool calls and the final answer are streamed to the frontend via SSE; the UI shows each tool invocation as it happens, then streams the answer token by token
- **Source attribution** - after the agent finishes, every document it retrieved is collected and shown as sources under the answer
- **Explore papers** - browse all indexed papers; ask questions scoped to a single paper or across your whole library
- **Semantic cache** - questions with cosine similarity ≥ 0.92 against a previously answered question (for the same paper set) return the cached answer without touching the agent, tools, or LLM

## Stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · psycopg3 |
| Agent framework | LangGraph (`create_react_agent`) |
| LLM / tool orchestration | LangChain · LangChain Google GenAI |
| Vector store | ChromaDB (persisted to disk) |
| Embeddings | Ollama - `mxbai-embed-large` |
| LLM | Google Gemini `gemini-3.6-flash` |
| Metadata DB | PostgreSQL |
| Frontend | React · TypeScript · Vite |

## Project structure

```
backend/            FastAPI server, agent tools, indexing logic
frontend/scholar/   React + TypeScript frontend (Vite)
```

## Prerequisites

- Python 3.11+
- Node.js and npm
- [Ollama](https://ollama.com) running locally with `mxbai-embed-large` pulled
- PostgreSQL running locally
- A Google AI API key from [aistudio.google.com](https://aistudio.google.com)

## Setup

### 1 - PostgreSQL

Create the database and table:

```sql
CREATE DATABASE research_rag;

\c research_rag

CREATE TABLE paper_store (
    paper_id             UUID PRIMARY KEY,
    paper_title          TEXT,
    paper_summary        TEXT,
    paper_authors        TEXT[],
    paper_size           NUMERIC,
    paper_chunks_created INTEGER,
    indexed              BOOLEAN DEFAULT FALSE,
    status               TEXT,
    error                TEXT
);
```

The connection string is hardcoded in `backend/app.py`:

```
postgres://<user>:<password>@localhost/research_rag
```

Update it to match your PostgreSQL user, password, and database name.

### 2 - Ollama

```bash
ollama pull mxbai-embed-large
```

Ollama must be running (`ollama serve`) before the backend starts.

### 3 - Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GOOGLE_API_KEY=your_key_here
CHROMA_COLLECTION_NAME=research_papers
CHROMA_PERSISTANT_DIR=rag-store/chroma
UPLOAD_FILE_LIMIT=100
META_DB_NAME=research_rag
PAPER_STORE_DIR=user-papers/store
```

Start the server:

```bash
uvicorn app:app --reload
```

API runs at `http://localhost:8000`.

### 4 - Frontend

```bash
cd frontend/scholar
npm install
npm run dev
```

App opens at `http://localhost:5173`.

If your backend is on a different port, copy `.env.example` to `.env` and set `VITE_API_BASE`.

## Known limitations

- The semantic cache is in-memory and resets on every server restart.
- The agent always calls at least one tool before answering, which adds latency compared to a direct LLM call.
- Grounding is enforced through prompting; the LLM may still occasionally draw on its own training knowledge.
- There is no persistent memory across sessions - the agent starts fresh for every question.
