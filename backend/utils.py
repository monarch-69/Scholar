from dataclasses import dataclass
from typing import List, Tuple
from fastapi import Request
from langchain_core.documents import Document
from langchain_core.tools import tool as lc_tool
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
import asyncio
import logging
import numpy as np
import os

from vector_store import build_index

logger = logging.getLogger(__name__)


class Config(BaseModel):
    chroma_collection_name: str
    chroma_persistant_dir: str
    upload_file_limit: int
    meta_db_name: str
    paper_store_dir: str


class AskRequest(BaseModel):
    question: str
    paper_ids: List[str]


# ── Agent system prompt ────────────────────────────────────────────────────────

_ASK_SYSTEM = (
    "You are a research assistant with tools for searching indexed research papers.\n\n"
    "Available tools:\n"
    "- search_papers: general semantic search — use for most questions\n"
    "- compare_papers: searches each paper individually for an aspect — use when asked to compare across papers\n"
    "- extract_structured: pull specific structured info (methodology / datasets / results / limitations / contributions / future_work)\n"
    "- find_contradictions: find conflicting or contrasting claims across papers on a topic\n\n"
    "Rules:\n"
    "1. Always call at least one tool before answering — never answer from memory or prior knowledge\n"
    "2. If the first search is insufficient, search again with a different query or tool\n"
    "3. Answer ONLY from the retrieved passages; if they don't contain the answer, say so clearly\n"
    "4. Be concise and accurate"
)

# Labels shown in the UI for each tool call
_TOOL_LABELS: dict = {
    "search_papers": "Searching",
    "compare_papers": "Comparing across papers",
    "extract_structured": "Extracting",
    "find_contradictions": "Scanning for contradictions",
}


# ── Agent tool factory ─────────────────────────────────────────────────────────

def make_agent_tools(vector_store, paper_ids: List[str]) -> Tuple[list, List[Document]]:
    """Build the four agent tools scoped to `paper_ids`.

    Returns (tools, retrieved) where `retrieved` is a shared list that every
    tool appends to, giving the caller a full record of every doc the agent
    touched.
    """
    retrieved: List[Document] = []

    chroma_filter = (
        {"id": paper_ids[0]}
        if len(paper_ids) == 1
        else {"id": {"$in": paper_ids}}
    )

    @lc_tool
    def search_papers(query: str) -> str:
        """Search indexed research papers for passages relevant to a query. Use for general questions."""
        docs: List[Document] = vector_store.similarity_search(query, k=5, filter=chroma_filter)
        retrieved.extend(docs)
        if not docs:
            return "No relevant passages found in the selected papers."
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('name', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    @lc_tool
    def compare_papers(aspect: str) -> str:
        """Compare how different papers approach a specific aspect (e.g. methodology, evaluation, results). Searches each paper individually."""
        per_paper: dict = {}
        for pid in paper_ids:
            docs: List[Document] = vector_store.similarity_search(aspect, k=2, filter={"id": pid})
            if docs:
                retrieved.extend(docs)
                name = docs[0].metadata.get("name", pid)
                per_paper[name] = "\n".join(d.page_content for d in docs)

        if not per_paper:
            return "No relevant passages found to compare."

        return "\n\n═══\n\n".join(
            f"[{name}]\n{content}" for name, content in per_paper.items()
        )

    @lc_tool
    def extract_structured(field: str) -> str:
        """Extract specific structured information from the papers. field must be one of: methodology, datasets, results, limitations, contributions, future_work."""
        field_queries: dict = {
            "methodology":    "research methodology approach method technique algorithm",
            "datasets":       "dataset data benchmark evaluation experiments training",
            "results":        "results performance accuracy metrics evaluation scores",
            "limitations":    "limitations drawbacks constraints weaknesses",
            "contributions":  "contributions novelty proposed method innovation key insight",
            "future_work":    "future work open problems next steps conclusion",
        }
        query = field_queries.get(field.lower().strip(), field)
        docs: List[Document] = vector_store.similarity_search(query, k=6, filter=chroma_filter)
        retrieved.extend(docs)
        if not docs:
            return f"No passages found for '{field}'."
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('name', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    @lc_tool
    def find_contradictions(topic: str) -> str:
        """Find potentially conflicting or contradicting claims across papers on a specific topic."""
        parts: List[str] = []
        for pid in paper_ids:
            docs: List[Document] = vector_store.similarity_search(topic, k=3, filter={"id": pid})
            if docs:
                retrieved.extend(docs)
                name = docs[0].metadata.get("name", pid)
                parts.append(f"[{name}]\n" + "\n".join(d.page_content for d in docs))

        if not parts:
            return "No relevant passages found."
        return "\n\n═══\n\n".join(parts)

    return [search_papers, compare_papers, extract_structured, find_contradictions], retrieved


# ── Semantic cache ─────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    embedding: list
    paper_ids: frozenset
    answer: str
    sources: list


class SemanticCache:
    """In-memory semantic cache keyed on (question embedding, paper_ids set).

    Questions with cosine similarity ≥ threshold against a cached question
    (for the same paper set) return the stored answer without touching the
    agent, tools, or LLM.
    """

    def __init__(self, embedding_model, threshold: float = 0.92):
        self._model = embedding_model
        self._threshold = threshold
        self._entries: List[_CacheEntry] = []

    def _cosine(self, a: list, b: list) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0

    def lookup(self, question: str, paper_ids: frozenset) -> Tuple:
        """Return (answer, sources, embedding).

        On a hit: answer and sources are the cached values.
        On a miss: answer and sources are None; embedding is always returned
        so the caller can pass it to store() without re-embedding.
        """
        emb = self._model.embed_query(question)
        for entry in self._entries:
            if entry.paper_ids != paper_ids:
                continue
            sim = self._cosine(emb, entry.embedding)
            if sim >= self._threshold:
                logger.info("Semantic cache HIT (%.3f) for %r", sim, question[:60])
                return entry.answer, entry.sources, emb
        return None, None, emb

    def store(self, embedding: list, paper_ids: frozenset, answer: str, sources: list) -> None:
        self._entries.append(_CacheEntry(
            embedding=embedding,
            paper_ids=paper_ids,
            answer=answer,
            sources=sources,
        ))


# ── LLM token extraction ───────────────────────────────────────────────────────

def _extract_token(chunk) -> str:
    """Extract plain text from an LLM chunk or message (handles Gemini content blocks)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# ── Config + DB helpers ────────────────────────────────────────────────────────

def load_config(
    collection_name: str | None = None,
    persistant_dir: str | None = None,
    upload_file_limit: int | None = None,
    meta_db_name: str | None = None,
    paper_store_dir: str | None = None,
) -> Config:
    return Config(
        chroma_collection_name=collection_name or os.getenv("CHROMA_COLLECTION_NAME") or "research_papers",
        chroma_persistant_dir=persistant_dir or os.getenv("CHROMA_PERSISTANT_DIR") or "rag-store/chroma",
        upload_file_limit=upload_file_limit or int(os.getenv("UPLOAD_FILE_LIMIT") or 100),
        meta_db_name=meta_db_name or os.getenv("META_DB_NAME") or "research_rag",
        paper_store_dir=paper_store_dir or os.getenv("PAPER_STORE_DIR") or "user-papers/store",
    )


async def get_meta_db(request: Request):
    """FastAPI dependency: yields a checked-out pool connection."""
    _conn_pool: AsyncConnectionPool = request.app.state.meta_db
    async with _conn_pool.connection() as _connection:
        yield _connection


async def indexer_bg_worker(
    request: Request,
    paper_id: str,
    paper_title: str,
    content: bytes,
) -> None:
    """Background task: index a PDF and update its status in the DB."""
    try:
        chunks_created: int = await asyncio.to_thread(
            build_index, request.app.state.vector_store, paper_id, paper_title, pdf_content=content
        )
        async with request.app.state.meta_db.connection() as connection:
            await connection.execute(
                "UPDATE paper_store SET status = 'done', paper_chunks_created = %s WHERE paper_id = %s",
                (chunks_created, paper_id),
            )
    except Exception:
        logger.exception("Failed to index paper %s", paper_id)
        try:
            async with request.app.state.meta_db.connection() as connection:
                await connection.execute(
                    "UPDATE paper_store SET status = 'failed' WHERE paper_id = %s",
                    (paper_id,),
                )
        except Exception:
            logger.exception("Could not mark paper %s as failed", paper_id)
