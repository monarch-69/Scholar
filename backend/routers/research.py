from typing import List
from math import ceil
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from fastapi.background import BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from uuid_utils import uuid7
import asyncio
import json
import time

from utils import (
    SemanticCache,
    make_agent_tools,
    logger,
    get_meta_db,
    indexer_bg_worker,
    AskRequest,
    _ASK_SYSTEM,
    _TOOL_LABELS,
    _extract_token,
)

router: APIRouter = APIRouter(prefix="/research")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

# Hard ceiling on how long one agent run can take (seconds).
_AGENT_TIMEOUT = 120.0


# ── Upload & index ─────────────────────────────────────────────────────────

@router.post("/index", status_code=202)
async def index_research_paper(
    request: Request,
    bg_worker: BackgroundTasks,
    paper_title: str = Form(...),
    paper_summary: str | None = Form(None),
    paper_authors: str | None = Form(None),
    file: UploadFile = File(...),
    connection=Depends(get_meta_db),
):
    size_in_mb: int = ceil((file.size or 0) / 1024 / 1024)

    if size_in_mb <= 0 or size_in_mb > 101:
        raise HTTPException(
            status_code=422,
            detail=f"File size must be between 1 and 100 MB (got {size_in_mb} MB).",
        )

    file_content: bytes = await file.read()
    paper_id: str = str(uuid7())
    _paper_authors: List[str] = [a.strip() for a in (paper_authors or "").split(",")]

    async with connection.cursor() as cursor:
        try:
            await cursor.execute(
                """
                INSERT INTO paper_store (
                    paper_id, paper_title, paper_summary, paper_authors, paper_size, status
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (paper_id, paper_title, paper_summary, _paper_authors, size_in_mb, "processing"),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=500, detail="DB insert returned 0 rows.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    await connection.commit()
    bg_worker.add_task(indexer_bg_worker, request, paper_id, paper_title, file_content)
    return {"paper_id": paper_id, "status": "processing"}


# ── List all papers ─────────────────────────────────────────────────────────

@router.get("/papers", status_code=200)
async def list_papers(connection=Depends(get_meta_db)):
    async with connection.cursor() as cursor:
        await cursor.execute(
            """
            SELECT paper_id, paper_title, paper_summary, paper_authors, status, paper_chunks_created
            FROM paper_store
            """
        )
        rows = await cursor.fetchall()

    return [
        {
            "paper_id": row[0],
            "paper_title": row[1],
            "paper_summary": row[2],
            "paper_authors": ", ".join(row[3]) if row[3] else None,
            "status": row[4],
            "chunks": row[5],
        }
        for row in rows
    ]


# ── Per-paper status ────────────────────────────────────────────────────────

@router.get("/{paper_id}/status", status_code=200)
async def get_paper_status(paper_id: str, connection=Depends(get_meta_db)):
    async with connection.cursor() as cursor:
        await cursor.execute(
            "SELECT status, paper_chunks_created FROM paper_store WHERE paper_id = %s",
            (paper_id,),
        )
        res = await cursor.fetchone()

    if res is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    return {"paper_id": paper_id, "status": res[0], "chunks": res[1], "error": None}


# ── Ask (ReAct agent + SSE streaming + semantic cache) ──────────────────────

@router.post("/ask")
async def ask_papers(request: Request, body: AskRequest):
    if not body.paper_ids:
        raise HTTPException(status_code=422, detail="paper_ids must not be empty.")
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")

    llm            = request.app.state.llm
    vector_store   = request.app.state.vector_store
    cache: SemanticCache = request.app.state.semantic_cache
    paper_ids_set  = frozenset(body.paper_ids)

    # ── Semantic cache check ─────────────────────────────────────────────────
    cached_answer, cached_sources, question_emb = await asyncio.to_thread(
        cache.lookup, body.question, paper_ids_set
    )

    if cached_answer is not None:
        async def _cached_stream():
            yield f"event: token\ndata: {json.dumps(cached_answer)}\n\n"
            yield f"event: done\ndata: {json.dumps({'sources': cached_sources})}\n\n"

        return StreamingResponse(_cached_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    # ── Build per-request tools + agent ─────────────────────────────────────
    tools, retrieved_docs = make_agent_tools(vector_store, body.paper_ids)
    agent = create_react_agent(llm, tools)

    async def _live_stream():
        sources_to_send: list = []
        full_answer = ""
        deadline = time.monotonic() + _AGENT_TIMEOUT

        try:
            yield f"event: step\ndata: {json.dumps({'type': 'planning', 'text': 'Analyzing question…'})}\n\n"

            messages = [
                SystemMessage(content=_ASK_SYSTEM),
                HumanMessage(content=body.question),
            ]

            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": 12},
            ):
                # Hard timeout — kills the loop if the agent runs too long
                if time.monotonic() > deadline:
                    yield f"event: error\ndata: {json.dumps('Agent timed out — please try again.')}\n\n"
                    break

                kind = event["event"]

                # ── Tool started → show a step in the UI ────────────────────
                if kind == "on_tool_start":
                    tool_name  = event.get("name", "")
                    inp        = event["data"].get("input", {})
                    query_text = (
                        inp.get("query") or inp.get("aspect") or
                        inp.get("field") or inp.get("topic") or ""
                    )
                    label = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title())
                    step_text = f"{label}: {query_text}" if query_text else label
                    yield f"event: step\ndata: {json.dumps({'type': 'searching', 'text': step_text})}\n\n"

                # ── LLM token from the agent node → stream to client ─────────
                elif kind == "on_chat_model_stream":
                    if event["metadata"].get("langgraph_node") != "agent":
                        continue
                    chunk = event["data"]["chunk"]
                    # Skip chunks that are part of a tool call (JSON arguments)
                    if getattr(chunk, "tool_call_chunks", None):
                        continue
                    token = _extract_token(chunk)
                    if token:
                        full_answer += token
                        yield f"event: token\ndata: {json.dumps(token)}\n\n"

            # ── Build sources: one entry per unique paper retrieved ───────────
            seen_papers: dict = {}
            for doc in retrieved_docs:
                pid = doc.metadata.get("id", "")
                if pid and pid not in seen_papers:
                    seen_papers[pid] = {
                        "paper_id": pid,
                        "paper_name": doc.metadata.get("name", "Unknown"),
                        "excerpt": doc.page_content[:500],
                        "section": None,
                    }
            sources_to_send = list(seen_papers.values())

            if full_answer:
                cache.store(question_emb, paper_ids_set, full_answer, sources_to_send)

        except Exception:
            logger.exception("Agent failed for %r", body.question)
            yield f"event: error\ndata: {json.dumps('Agent request failed — please try again.')}\n\n"

        # Always send done so the client clears its busy state
        yield f"event: done\ndata: {json.dumps({'sources': sources_to_send})}\n\n"

    return StreamingResponse(_live_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
