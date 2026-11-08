from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from graphrag.graph import GraphStore, Node, Edge, NodeType, EdgeType


def export_nodes_json(store: GraphStore, output_path: Path) -> None:
    """Export all nodes to JSON."""
    data = []
    for node in store.iter_nodes():
        data.append(
            {
                "id": node.id,
                "type": node.type.value,
                "properties": node.properties,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def export_edges_json(store: GraphStore, output_path: Path) -> None:
    """Export all edges to JSON."""
    data = []
    for edge in store.iter_edges():
        data.append(
            {
                "id": edge.id,
                "type": edge.type.value,
                "source": edge.source,
                "target": edge.target,
                "properties": edge.properties,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def export_graph_json(store: GraphStore, output_path: Path) -> None:
    """Export full graph to JSON."""
    nodes = []
    for node in store.iter_nodes():
        nodes.append(
            {
                "id": node.id,
                "type": node.type.value,
                "properties": node.properties,
            }
        )
    edges = []
    for edge in store.iter_edges():
        edges.append(
            {
                "id": edge.id,
                "type": edge.type.value,
                "source": edge.source,
                "target": edge.target,
                "properties": edge.properties,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))


def export_subgraph_json(
    store: GraphStore,
    chunk_ids: list[str],
    output_path: Path,
    hops: int = 1,
) -> None:
    """Export subgraph reachable from given chunk IDs."""
    # Collect nodes and edges in subgraph
    node_ids = set(chunk_ids)
    edge_ids = set()

    # Expand through graph
    for _ in range(hops + 1):
        new_node_ids = set(node_ids)
        for node_id in node_ids:
            # Outgoing edges
            for edge in store.get_outgoing_edges(node_id):
                edge_ids.add(edge.id)
                new_node_ids.add(edge.target)
            # Incoming edges
            for edge in store.get_incoming_edges(node_id):
                edge_ids.add(edge.id)
                new_node_ids.add(edge.source)
        node_ids = new_node_ids

    # Build subgraph data
    nodes = []
    for node_id in node_ids:
        node = store.get_node(node_id)
        if node:
            nodes.append(
                {
                    "id": node.id,
                    "type": node.type.value,
                    "properties": node.properties,
                }
            )

    edges = []
    for edge_id in edge_ids:
        edge = store.get_edge(edge_id)
        if edge:
            edges.append(
                {
                    "id": edge.id,
                    "type": edge.type.value,
                    "source": edge.source,
                    "target": edge.target,
                    "properties": edge.properties,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))


def export_graph_dot(
    store: GraphStore, output_path: Path, max_nodes: int = 200
) -> None:
    """Export graph to DOT format for visualization."""
    lines = [
        "digraph graphrag {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]

    node_count = 0
    node_labels = {}
    for node in store.iter_nodes():
        if node_count >= max_nodes:
            break
        label = node.id[:12]
        if node.type == NodeType.ENTITY:
            label = node.properties.get("canonical_name", node.id[:12])
        elif node.type == NodeType.CHUNK:
            label = f"Chunk:{node.id[:8]}"
        elif node.type == NodeType.DOCUMENT:
            label = f"Doc:{node.properties.get('path', node.id[:12])}"
        # Escape quotes
        label = label.replace('"', '\\"')
        lines.append(f'  "{node.id}" [label="{label}", type="{node.type.value}"];')
        node_labels[node.id] = label
        node_count += 1

    edge_count = 0
    for edge in store.iter_edges():
        if edge_count >= max_nodes * 2:
            break
        if edge.source in node_labels and edge.target in node_labels:
            label = edge.type.value
            if edge.type == EdgeType.RELATED_TO:
                label = edge.properties.get("predicate", "related_to")
            label = label.replace('"', '\\"')
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')
            edge_count += 1

    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def export_subgraph_dot(
    store: GraphStore,
    chunk_ids: list[str],
    output_path: Path,
    hops: int = 1,
    max_nodes: int = 100,
) -> None:
    """Export subgraph to DOT format."""
    # Collect nodes and edges in subgraph
    node_ids = set(chunk_ids)
    edge_ids = set()

    for _ in range(hops + 1):
        new_node_ids = set(node_ids)
        for node_id in node_ids:
            for edge in store.get_outgoing_edges(node_id):
                edge_ids.add(edge.id)
                new_node_ids.add(edge.target)
            for edge in store.get_incoming_edges(node_id):
                edge_ids.add(edge.id)
                new_node_ids.add(edge.source)
        node_ids = new_node_ids

    lines = [
        "digraph graphrag_subgraph {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]

    node_labels = {}
    for node_id in node_ids:
        node = store.get_node(node_id)
        if not node:
            continue
        label = node.id[:12]
        if node.type == NodeType.ENTITY:
            label = node.properties.get("canonical_name", node.id[:12])
        elif node.type == NodeType.CHUNK:
            label = f"Chunk:{node.id[:8]}"
        elif node.type == NodeType.DOCUMENT:
            label = f"Doc:{node.properties.get('path', node.id[:12])}"
        label = label.replace('"', '\\"')
        lines.append(f'  "{node.id}" [label="{label}", type="{node.type.value}"];')
        node_labels[node.id] = label

    for edge_id in edge_ids:
        edge = store.get_edge(edge_id)
        if not edge:
            continue
        if edge.source in node_labels and edge.target in node_labels:
            label = edge.type.value
            if edge.type == EdgeType.RELATED_TO:
                label = edge.properties.get("predicate", "related_to")
            label = label.replace('"', '\\"')
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')

    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
