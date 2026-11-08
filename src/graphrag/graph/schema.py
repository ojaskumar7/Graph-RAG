from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    ENTITY = "Entity"


class EdgeType(str, Enum):
    HAS_CHUNK = "HAS_CHUNK"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"


@dataclass(slots=True)
class Node:
    id: str
    type: NodeType
    properties: dict


@dataclass(slots=True)
class Edge:
    id: str
    type: EdgeType
    source: str
    target: str
    properties: dict


def make_document_node(doc_id: str, path: str) -> Node:
    return Node(id=doc_id, type=NodeType.DOCUMENT, properties={"path": path})


def make_chunk_node(
    chunk_id: str, doc_id: str, text: str, start: int, end: int
) -> Node:
    return Node(
        id=chunk_id,
        type=NodeType.CHUNK,
        properties={"doc_id": doc_id, "text": text, "start": start, "end": end},
    )


def make_entity_node(
    entity_id: str, name: str, entity_type: str, canonical_name: str | None = None
) -> Node:
    return Node(
        id=entity_id,
        type=NodeType.ENTITY,
        properties={
            "name": name,
            "entity_type": entity_type,
            "canonical_name": canonical_name or name,
        },
    )


def make_has_chunk_edge(edge_id: str, doc_id: str, chunk_id: str) -> Edge:
    return Edge(
        id=edge_id,
        type=EdgeType.HAS_CHUNK,
        source=doc_id,
        target=chunk_id,
        properties={},
    )


def make_mentions_edge(edge_id: str, chunk_id: str, entity_id: str) -> Edge:
    return Edge(
        id=edge_id,
        type=EdgeType.MENTIONS,
        source=chunk_id,
        target=entity_id,
        properties={},
    )


def make_related_to_edge(
    edge_id: str,
    source_entity: str,
    target_entity: str,
    predicate: str,
    confidence: float,
    source_chunk_id: str,
) -> Edge:
    return Edge(
        id=edge_id,
        type=EdgeType.RELATED_TO,
        source=source_entity,
        target=target_entity,
        properties={
            "predicate": predicate,
            "confidence": confidence,
            "source_chunk_id": source_chunk_id,
        },
    )
