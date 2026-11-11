from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphrag.cli import build_graph, build_fts, ingest, query
from graphrag.config import get_settings

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    hops: int = Field(default=2, ge=0, le=3)
    json_output: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    confidence: float
    retrieval: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GraphRAG server starting up")
    yield
    logger.info("GraphRAG server shutting down")


app = FastAPI(
    title="GraphRAG API",
    description="Graph Retrieval-Augmented Generation API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        result = query(
            question=request.question,
            top_k=request.top_k,
            hops=request.hops,
            json_output=request.json_output,
        )
        if isinstance(result, str):
            import json

            result = json.loads(result)
        return QueryResponse(**result)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_endpoint():
    try:
        result = ingest()
        return {"message": result}
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build-graph")
async def build_graph_endpoint():
    try:
        result = build_graph()
        return {"message": result}
    except Exception as e:
        logger.exception("Build graph failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build-fts")
async def build_fts_endpoint():
    try:
        result = build_fts()
        return {"message": result}
    except Exception as e:
        logger.exception("Build FTS failed")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
