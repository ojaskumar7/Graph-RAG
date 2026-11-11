from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from graphrag.config import get_settings, Settings
from graphrag.ingest import (
    load_documents,
    chunk_document,
    write_manifest,
    ChunkRecord,
    load_manifest,
)
from graphrag.graph import (
    GraphStore,
    make_document_node,
    make_chunk_node,
    make_has_chunk_edge,
    make_entity_node,
    make_mentions_edge,
    make_related_to_edge,
    NodeType,
    EdgeType,
    export_graph_json,
    export_subgraph_json,
    export_graph_dot,
    export_subgraph_dot,
)
from graphrag.llm.client import OpenAICompatibleClient, LLMError
from graphrag.retrieve import (
    retrieve_context,
    build_fts_index as build_fts_index_fn,
    fts_search_chunks,
    keyword_search_chunks,
)
from graphrag.generate import AnswerResult, generate_answer
from graphrag.eval import run_eval
from graphrag.extract import ExtractionPipeline, EntityNormalizer


def _get_settings(env_file: str | None = None) -> Settings:
    return get_settings(Path(env_file) if env_file else None)


def _load_chunk_text(s: Settings, rec: ChunkRecord) -> str:
    path = s.raw_dir / rec.path
    text = path.read_text(encoding="utf-8")
    return text[rec.start : rec.end]


def cmd_info(args: argparse.Namespace) -> int:
    s = _get_settings(args.env_file)
    print(
        json.dumps(
            {
                "project": "graphrag-lite",
                "llm_base_url": s.llm_base_url,
                "llm_model": s.llm_model,
                "llm_api_key_set": bool(s.llm_api_key),
                "data_dir": str(s.data_dir),
                "dataset": "HotpotQA subset (planned) - data/raw, data/eval/golden.jsonl",
                "deps": "stdlib-only (0 MB), no PyTorch/CUDA, no venv required",
            },
            indent=2,
        )
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    result = ingest(args.env_file)
    print(result)
    return 0


def load_chunk_text(s: Settings, rec: ChunkRecord) -> str:
    path = s.raw_dir / rec.path
    text = path.read_text(encoding="utf-8")
    return text[rec.start : rec.end]


def cmd_build_graph(args: argparse.Namespace) -> int:
    result = build_graph(args.env_file)
    print(result)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    try:
        output = query(
            question=args.question,
            top_k=args.top_k,
            hops=args.hops,
            json_output=args.json,
            env_file=args.env_file,
        )
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"\nQuestion: {output['question']}")
            print(f"\nAnswer: {output['answer']}")
            print(
                f"\nCitations: {', '.join(output['citations']) if output['citations'] else 'none'}"
            )
            print(f"Confidence: {output['confidence']:.2f}")
            print(f"\nRetrieval: {output['retrieval']}")
        return 0
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        return 1


def cmd_build_fts(args: argparse.Namespace) -> int:
    result = build_fts(args.env_file)
    print(result)
    return 0


def cmd_build_vector(args: argparse.Namespace) -> int:
    result = build_vector(args.env_file)
    print(result)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    s = get_settings(Path(args.env_file) if args.env_file else None)
    store = GraphStore(s.processed_dir)
    stats = store.stats()
    print(json.dumps(stats, indent=2))

    if args.stats:
        return 0

    # Subgraph export takes precedence
    if args.subgraph_chunks:
        chunk_ids = [c.strip() for c in args.subgraph_chunks.split(",")]
        if args.export_json:
            export_subgraph_json(
                store, chunk_ids, Path(args.export_json), hops=args.hops
            )
            print(f"Exported subgraph to {args.export_json}")
        if args.export_dot:
            export_subgraph_dot(
                store,
                chunk_ids,
                Path(args.export_dot),
                hops=args.hops,
                max_nodes=args.max_nodes,
            )
            print(f"Exported subgraph DOT to {args.export_dot}")
        if not args.export_json and not args.export_dot:
            export_subgraph_json(
                store, chunk_ids, Path("subgraph.json"), hops=args.hops
            )
            print("Exported subgraph to subgraph.json")
        return 0

    # Full graph export
    if args.export_json:
        export_graph_json(store, Path(args.export_json))
        print(f"Exported full graph to {args.export_json}")
        return 0

    if args.export_dot:
        export_graph_dot(store, Path(args.export_dot), max_nodes=args.max_nodes)
        print(f"Exported full graph DOT to {args.export_dot}")
        return 0

    print("\nSample nodes:")
    for i, n in enumerate(store.iter_nodes()):
        if i >= 10:
            break
        print(f"  {n.id} ({n.type.value}): {n.properties}")
    print("\nSample edges:")
    for i, e in enumerate(store.iter_edges()):
        if i >= 10:
            break
        print(f"  {e.id} ({e.type.value}): {e.source} -> {e.target} {e.properties}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    s = _get_settings(args.env_file)

    try:
        report = run_eval(
            s,
            top_k=args.top_k,
            hops=args.hops,
            output_path=Path(args.output) if args.output else None,
            use_mock_llm=args.mock_llm,
            full_pipeline=args.full_pipeline,
        )
    except RuntimeError as e:
        print(f"Eval failed: {e}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    print(
        f"\nHit rate: {report['hit_rate']:.2%} ({report['hit_at_k']}/{report['total_questions']})"
    )
    print(f"Avg recall@k: {report['avg_recall_at_k']:.2%}")
    if args.full_pipeline:
        print(
            f"Citation valid rate: {report['citation_valid_rate']:.2%} ({report['citations_valid_count']}/{report['total_questions']})"
        )
    return 0


# --- FastAPI wrapper functions ---


def ingest(env_file: str | None = None) -> str:
    """Ingest documents from data/raw."""
    s = _get_settings(env_file)
    docs = load_documents(s.raw_dir)
    if not docs:
        return "No .md/.txt files found in data/raw"

    all_records: list[ChunkRecord] = []
    for doc in docs:
        chunks = chunk_document(doc.doc_id, doc.text)
        for c in chunks:
            all_records.append(
                ChunkRecord(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    path=str(doc.path.relative_to(s.raw_dir)),
                    start=c.start,
                    end=c.end,
                    text_hash=hashlib.sha256(c.text.encode()).hexdigest(),
                )
            )

    write_manifest(s.processed_dir, all_records)
    return f"Ingested {len(docs)} docs -> {len(all_records)} chunks (manifest written)"


def build_graph(env_file: str | None = None) -> str:
    """Build knowledge graph with LLM extraction."""
    s = _get_settings(env_file)

    manifest = load_manifest(s.processed_dir)
    if not manifest:
        return "No manifest found. Run 'ingest' first."

    store = GraphStore(s.processed_dir)

    for chunk_id, rec in manifest.items():
        if rec.doc_id not in store.nodes:
            store.upsert_node(make_document_node(rec.doc_id, rec.path))
        if chunk_id not in store.nodes:
            chunk_text = _load_chunk_text(s, rec)
            store.upsert_node(
                make_chunk_node(chunk_id, rec.doc_id, chunk_text, rec.start, rec.end)
            )
        if f"edge_{rec.doc_id}_{chunk_id}" not in store.edges:
            store.upsert_edge(
                make_has_chunk_edge(
                    f"edge_{rec.doc_id}_{chunk_id}", rec.doc_id, chunk_id
                )
            )

    if not s.llm_api_key:
        print(
            "LLM API key not set (GRAPHRAG_LLM_API_KEY). Extraction skipped.",
            file=sys.stderr,
        )
        print("Graph structure (documents/chunks) saved.")
        store.save()
        return "Graph structure (documents/chunks) saved."

    client = OpenAICompatibleClient(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        timeout=s.request_timeout,
        max_retries=s.max_retries,
    )

    pipeline = ExtractionPipeline(client, max_retries=s.max_retries)
    normalizer = EntityNormalizer(store)

    for chunk_id, rec in manifest.items():
        chunk_text = _load_chunk_text(s, rec)
        try:
            result = pipeline.extract(chunk_text)
        except Exception as e:
            print(f"Extraction failed for chunk {chunk_id}: {e}", file=sys.stderr)
            continue

        for ent in result.entities:
            record = normalizer.normalize(ent.name, ent.entity_type, ent.canonical_name)
            entity_id = record.entity_id

            edge_id = f"mention_{chunk_id}_{entity_id}"
            if edge_id not in store.edges:
                store.upsert_edge(make_mentions_edge(edge_id, chunk_id, entity_id))

        for rel in result.relations:
            subj_id = normalizer.resolve_entity_id(rel.subject)
            obj_id = normalizer.resolve_entity_id(rel.object)
            if not subj_id or not obj_id:
                continue
            edge_id = f"rel_{subj_id}_{rel.predicate}_{obj_id}_{chunk_id}"
            if edge_id not in store.edges:
                store.upsert_edge(
                    make_related_to_edge(
                        edge_id,
                        subj_id,
                        obj_id,
                        rel.predicate,
                        rel.confidence,
                        chunk_id,
                    )
                )

    store.save()
    stats = store.stats()
    return f"Graph built: {stats['nodes']} nodes, {stats['edges']} edges"


def build_fts(env_file: str | None = None) -> str:
    """Build SQLite FTS5 index."""
    s = _get_settings(env_file)

    store = GraphStore(s.processed_dir)
    if not store.nodes:
        return "Graph is empty. Run 'build-graph' first."

    print("Building FTS5 index...")
    build_fts_index_fn(store, s.processed_dir)
    return "FTS5 index built successfully."


def build_vector(env_file: str | None = None) -> str:
    """Build vector index for semantic search."""
    s = _get_settings(env_file)

    store = GraphStore(s.processed_dir)
    if not store.nodes:
        return "Graph is empty. Run 'build-graph' first."

    print("Building vector index...")
    try:
        from graphrag.embeddings import LocalEmbedder
        from graphrag.retrieve.search import build_vector_index

        embedder = LocalEmbedder()
        build_vector_index(store, s.processed_dir, embedder)
        return "Vector index built successfully."
    except RuntimeError as e:
        return f"Vector index failed: {e}. Install with: pip install fastembed"
    except Exception as e:
        return f"Vector index failed: {e}"


def query(
    question: str,
    top_k: int = 5,
    hops: int = 2,
    json_output: bool = False,
    env_file: str | None = None,
) -> dict[str, Any]:
    """Query the graph and generate an answer."""
    s = _get_settings(env_file)

    store = GraphStore(s.processed_dir)
    if not store.nodes:
        raise RuntimeError("Graph is empty. Run 'build-graph' first.")

    # Retrieve context
    retrieval = retrieve_context(
        store,
        s.processed_dir,
        question,
        top_k_chunks=top_k,
        graph_hops=hops,
    )

    if not retrieval["all_chunks"]:
        raise RuntimeError("No relevant chunks found.")

    # Prepare evidence
    evidence = [(c.chunk_id, c.text) for c in retrieval["all_chunks"]]

    # If no API key, return retrieval only
    if not s.llm_api_key:
        output = {
            "question": question,
            "answer": "[LLM API key not set - retrieval only]",
            "citations": [],
            "confidence": 0.0,
            "retrieval": {
                "keyword_chunks": len(retrieval["keyword_chunks"]),
                "query_entities": [e.name for e in retrieval["query_entities"]],
                "expanded_entity_count": retrieval["entity_count"],
                "total_chunks_retrieved": len(retrieval["all_chunks"]),
                "retrieved_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "score": c.score,
                        "preview": c.text[:200],
                    }
                    for c in retrieval["all_chunks"]
                ],
            },
        }
        return output

    client = OpenAICompatibleClient(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        timeout=s.request_timeout,
        max_retries=s.max_retries,
    )

    # Generate answer
    result: AnswerResult = generate_answer(client, question, evidence)

    output = {
        "question": question,
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
        "retrieval": {
            "keyword_chunks": len(retrieval["keyword_chunks"]),
            "query_entities": [e.name for e in retrieval["query_entities"]],
            "expanded_entity_count": retrieval["entity_count"],
            "total_chunks_retrieved": len(retrieval["all_chunks"]),
        },
    }

    return output


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"'{args.command}' is scaffolded but not implemented yet.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphrag", description="Lightweight GraphRAG MVP")
    p.add_argument("--env-file", default=".env", help="Path to .env file")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Show configuration and project status")
    info.set_defaults(func=cmd_info)

    ingest = sub.add_parser("ingest", help="Load/chunk .md/.txt from data/raw")
    ingest.set_defaults(func=cmd_ingest)

    build_graph = sub.add_parser(
        "build-graph", help="Build graph structure + LLM extraction"
    )
    build_graph.set_defaults(func=cmd_build_graph)

    build_fts = sub.add_parser(
        "build-fts", help="Build SQLite FTS5 index for full-text search"
    )
    build_fts.set_defaults(func=cmd_build_fts)

    build_vector = sub.add_parser(
        "build-vector",
        help="Build vector index for semantic search (requires fastembed)",
    )
    build_vector.set_defaults(func=cmd_build_vector)

    query = sub.add_parser(
        "query", help="Retrieve evidence + graph context and answer with citations"
    )
    query.add_argument("question", help="Question to answer")
    query.add_argument("--top-k", type=int, default=5, help="Top K chunks to retrieve")
    query.add_argument("--hops", type=int, default=1, help="Graph expansion hops")
    query.add_argument("--json", action="store_true", help="Output JSON")
    query.set_defaults(func=cmd_query)

    inspect = sub.add_parser(
        "inspect", help="Show graph stats or export query subgraph"
    )
    inspect.add_argument("--stats", action="store_true", help="Show only stats")
    inspect.add_argument(
        "--export-json", help="Export full graph or subgraph to JSON file"
    )
    inspect.add_argument(
        "--export-dot", help="Export full graph or subgraph to DOT file"
    )
    inspect.add_argument(
        "--subgraph-chunks", help="Comma-separated chunk IDs for subgraph export"
    )
    inspect.add_argument(
        "--hops", type=int, default=1, help="Graph expansion hops for subgraph"
    )
    inspect.add_argument(
        "--max-nodes", type=int, default=200, help="Max nodes for DOT export"
    )
    inspect.set_defaults(func=cmd_inspect)

    eval_parser = sub.add_parser(
        "eval", help="Run evaluation on data/eval/golden.jsonl"
    )
    eval_parser.add_argument(
        "--top-k", type=int, default=5, help="Top K chunks to retrieve"
    )
    eval_parser.add_argument("--hops", type=int, default=1, help="Graph expansion hops")
    eval_parser.add_argument("--output", help="Output report JSON path")
    eval_parser.add_argument(
        "--mock-llm", action="store_true", help="Use mock LLM for testing"
    )
    eval_parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline with answer generation and citation validation",
    )
    eval_parser.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
