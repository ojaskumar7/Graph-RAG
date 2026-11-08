from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterator

from .schema import Edge, EdgeType, Node, NodeType


NODES_FILE = "nodes.jsonl"
EDGES_FILE = "edges.jsonl"


def _deserialize_node(data: dict) -> Node:
    data = data.copy()
    data["type"] = NodeType(data["type"])
    return Node(**data)


def _deserialize_edge(data: dict) -> Edge:
    data = data.copy()
    data["type"] = EdgeType(data["type"])
    return Edge(**data)


class GraphStore:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._load()

    def _load(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        nodes_path = self.processed_dir / NODES_FILE
        edges_path = self.processed_dir / EDGES_FILE
        if nodes_path.exists():
            with nodes_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.nodes[data["id"]] = _deserialize_node(data)
        if edges_path.exists():
            with edges_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.edges[data["id"]] = _deserialize_edge(data)

    def save(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = self.processed_dir / NODES_FILE
        edges_path = self.processed_dir / EDGES_FILE
        with nodes_path.open("w", encoding="utf-8") as f:
            for node in self.nodes.values():
                f.write(json.dumps(dataclasses.asdict(node)) + "\n")
        with edges_path.open("w", encoding="utf-8") as f:
            for edge in self.edges.values():
                f.write(json.dumps(dataclasses.asdict(edge)) + "\n")

    def upsert_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def upsert_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Edge | None:
        return self.edges.get(edge_id)

    def iter_nodes(self, node_type: NodeType | None = None) -> Iterator[Node]:
        for n in self.nodes.values():
            if node_type is None or n.type == node_type:
                yield n

    def iter_edges(self, edge_type: EdgeType | None = None) -> Iterator[Edge]:
        for e in self.edges.values():
            if edge_type is None or e.type == edge_type:
                yield e

    def get_chunk_text(self, chunk_id: str) -> str | None:
        node = self.nodes.get(chunk_id)
        if node and node.type == NodeType.CHUNK:
            return node.properties.get("text")
        return None

    def get_entity_by_name(self, name: str) -> Node | None:
        for n in self.iter_nodes(NodeType.ENTITY):
            if n.properties.get("canonical_name", "").lower() == name.lower():
                return n
            if n.properties.get("name", "").lower() == name.lower():
                return n
        return None

    def get_outgoing_edges(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[Edge]:
        return [
            e
            for e in self.edges.values()
            if e.source == node_id and (edge_type is None or e.type == edge_type)
        ]

    def get_incoming_edges(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[Edge]:
        return [
            e
            for e in self.edges.values()
            if e.target == node_id and (edge_type is None or e.type == edge_type)
        ]

    def stats(self) -> dict:
        node_counts = {}
        for n in self.nodes.values():
            node_counts[n.type.value] = node_counts.get(n.type.value, 0) + 1
        edge_counts = {}
        for e in self.edges.values():
            edge_counts[e.type.value] = edge_counts.get(e.type.value, 0) + 1
        return {"nodes": node_counts, "edges": edge_counts}
