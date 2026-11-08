from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from graphrag.graph import GraphStore, NodeType


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float


@dataclass(slots=True)
class RetrievedEntity:
    entity_id: str
    name: str
    entity_type: str
    canonical_name: str
    score: float


FTS_DB_NAME = "fts_index.sqlite"


def _get_fts_db_path(processed_dir: Path) -> Path:
    return processed_dir / FTS_DB_NAME


def build_fts_index(store: GraphStore, processed_dir: Path) -> None:
    """Build SQLite FTS5 index from chunk nodes."""
    db_path = _get_fts_db_path(processed_dir)
    conn = sqlite3.connect(db_path)
    try:
        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                text,
                tokenize='porter unicode61'
            )
        """)
        # Clear existing
        conn.execute("DELETE FROM chunks_fts")

        # Insert all chunks
        for node in store.iter_nodes(NodeType.CHUNK):
            chunk_id = node.id
            doc_id = node.properties.get("doc_id", "")
            text = node.properties.get("text", "")
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                (chunk_id, doc_id, text),
            )
        conn.commit()
    finally:
        conn.close()


def fts_search_chunks(
    store: GraphStore, processed_dir: Path, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """Search chunks using SQLite FTS5."""
    db_path = _get_fts_db_path(processed_dir)
    if not db_path.exists():
        # Fallback to keyword search if index doesn't exist
        return keyword_search_chunks(store, query, top_k)

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        # FTS5 MATCH query - sanitize and use OR between terms for better recall
        import re

        # Split into terms, filter short/stop words, join with OR
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        if not terms:
            return keyword_search_chunks(store, query, top_k)
        # Use OR between terms for better recall (AND is too strict with stop words)
        fts_query = " OR ".join(terms)

        cursor = conn.execute(
            """
            SELECT chunk_id, doc_id, text, bm25(chunks_fts) as rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, top_k),
        )
        results = []
        for row in cursor:
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    score=1.0 / (1.0 + row["rank"]),  # Convert BM25 to similarity-ish
                )
            )
        return results
    finally:
        conn.close()


def keyword_search_chunks(
    store: GraphStore, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """Simple keyword/token overlap search over chunks (fallback)."""
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return []

    scored = []
    for node in store.iter_nodes(NodeType.CHUNK):
        text = node.properties.get("text", "")
        text_tokens = set(text.lower().split())
        overlap = len(query_tokens & text_tokens)
        if overlap > 0:
            score = overlap / len(query_tokens)
            scored.append(
                RetrievedChunk(
                    chunk_id=node.id,
                    doc_id=node.properties.get("doc_id", ""),
                    text=text,
                    score=score,
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def find_entities_by_name(store: GraphStore, query: str) -> list[RetrievedEntity]:
    """Find entities mentioned in query by name matching."""
    query_lower = query.lower()
    results = []
    for node in store.iter_nodes(NodeType.ENTITY):
        name = node.properties.get("name", "").lower()
        canonical = node.properties.get("canonical_name", "").lower()
        if name in query_lower or canonical in query_lower:
            results.append(
                RetrievedEntity(
                    entity_id=node.id,
                    name=node.properties.get("name", ""),
                    entity_type=node.properties.get("entity_type", ""),
                    canonical_name=node.properties.get("canonical_name", ""),
                    score=1.0,
                )
            )
    return results


def expand_entities_neighbors(
    store: GraphStore, entity_ids: list[str], hops: int = 1
) -> tuple[list[str], list[str]]:
    """Expand from entity IDs to related entities and their chunks.
    Returns (expanded_entity_ids, source_chunk_ids).
    """
    seen_entities = set(entity_ids)
    source_chunks = set()
    frontier = list(entity_ids)

    for _ in range(hops):
        next_frontier = []
        for eid in frontier:
            # Outgoing RELATED_TO edges
            for edge in store.get_outgoing_edges(eid):
                if edge.target not in seen_entities:
                    seen_entities.add(edge.target)
                    next_frontier.append(edge.target)
                chunk_id = edge.properties.get("source_chunk_id")
                if chunk_id:
                    source_chunks.add(chunk_id)

            # Incoming RELATED_TO edges
            for edge in store.get_incoming_edges(eid):
                if edge.source not in seen_entities:
                    seen_entities.add(edge.source)
                    next_frontier.append(edge.source)
                chunk_id = edge.properties.get("source_chunk_id")
                if chunk_id:
                    source_chunks.add(chunk_id)

            # MENTIONS edges from chunks to this entity
            for edge in store.get_incoming_edges(eid):
                chunk_id = edge.source
                if chunk_id:
                    source_chunks.add(chunk_id)

        frontier = next_frontier
        if not frontier:
            break

    return list(seen_entities), list(source_chunks)


def retrieve_context(
    store: GraphStore,
    processed_dir: Path,
    query: str,
    top_k_chunks: int = 5,
    graph_hops: int = 1,
) -> dict:
    """Full hybrid retrieval: FTS5 chunks + entity graph expansion."""
    # 1. FTS5 search (with fallback to keyword)
    keyword_chunks = fts_search_chunks(store, processed_dir, query, top_k_chunks)

    # 2. Find entities in query
    query_entities = find_entities_by_name(store, query)
    entity_ids = [e.entity_id for e in query_entities]

    # 3. Graph expansion
    expanded_entities, graph_chunks = expand_entities_neighbors(
        store, entity_ids, graph_hops
    )

    # 4. Combine chunk sources
    all_chunk_ids = {c.chunk_id for c in keyword_chunks}
    all_chunk_ids.update(graph_chunks)

    # Get full chunk objects
    all_chunks = []
    for cid in all_chunk_ids:
        node = store.get_node(cid)
        if node and node.type == NodeType.CHUNK:
            score = next((c.score for c in keyword_chunks if c.chunk_id == cid), 0.1)
            all_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=node.properties.get("doc_id", ""),
                    text=node.properties.get("text", ""),
                    score=score,
                )
            )

    all_chunks.sort(key=lambda x: x.score, reverse=True)

    return {
        "keyword_chunks": keyword_chunks,
        "query_entities": query_entities,
        "expanded_entities": expanded_entities,
        "all_chunks": all_chunks[: top_k_chunks * 2],
        "entity_count": len(expanded_entities),
    }
