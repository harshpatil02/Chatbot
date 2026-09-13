import asyncio
import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents import build_graph
from app.ingestion import ingest_file

app = FastAPI(title="Guardrail RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a simple static UI at the app root
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("app/static/index.html")


class QueryRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    file_path: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: QueryRequest) -> Dict[str, Any]:
    graph = build_graph()
    if graph is None:
        return {
            "guardrail_status": "error",
            "blocked": False,
            "reason": "LangGraph is not available in the current environment.",
            "answer": "LangGraph is not available in the current environment.",
            "documents": [],
        }

    result = graph.invoke({"query": request.query, "messages": [request.query]})
    return {
        "guardrail_status": "blocked" if result.get("blocked") else "passed",
        "blocked": bool(result.get("blocked", False)),
        "reason": result.get("reason", ""),
        "answer": result.get("answer", ""),
        "documents": result.get("documents", []),
        "citations": result.get("citations", []),
    }


@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest) -> StreamingResponse:
    async def event_stream() -> Any:
        graph = build_graph()
        if graph is None:
            yield f"data: {json.dumps({'answer': 'LangGraph is not available in the current environment.'})}\n\n"
            return

        async for event in graph.astream({"query": request.query, "messages": [request.query]}, stream_mode="values"):
            state = event.get("auditor") or event.get("retrieve") or event.get("guardrail") or event.get("error")
            if isinstance(state, dict):
                answer = state.get("answer", "")
                if answer:
                    for token in answer.split():
                        yield f"data: {json.dumps({'token': token + ' '})}\n\n"
                        await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'answer': answer})}\n\n"
                    break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/ingest")
def ingest(request: IngestRequest) -> Dict[str, Any]:
    try:
        count = ingest_file(request.file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "chunks": count}


@app.exception_handler(Exception)
async def handle_exception(_: Any, exc: Exception) -> Any:
    return HTTPException(status_code=500, detail=str(exc))
