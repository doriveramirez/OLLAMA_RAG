from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CLEAN_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_VECTORSTORE_DIR,
    build_documents,
    ensure_safe_output_dir,
    load_clean_records,
    resolve_default_embedding_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local Chroma vector store for the Khan Academy RAG project."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Clean dataset not found at {args.input.resolve()}. Run clean_json.py first."
        )

    records = load_clean_records(args.input)
    documents = build_documents(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    embedding_model = args.embedding_model or resolve_default_embedding_model()

    ensure_safe_output_dir(args.persist_dir, Path.cwd())
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=args.ollama_base_url,
    )
    vectorstore = Chroma(
        collection_name=args.collection_name,
        persist_directory=str(args.persist_dir),
        embedding_function=embeddings,
    )

    try:
        for start in range(0, len(documents), 64):
            batch = documents[start : start + 64]
            vectorstore.add_documents(batch, ids=[doc.id for doc in batch])
    except Exception as exc:
        raise SystemExit(
            "Could not create the vector index. Verify that Ollama is running and "
            "that the embedding model is available. Suggested command: "
            "`ollama pull nomic-embed-text:latest`"
        ) from exc

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection_name": args.collection_name,
        "dataset_path": str(args.input.resolve()),
        "record_count": len(records),
        "chunk_count": len(documents),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": embedding_model,
        "ollama_base_url": args.ollama_base_url,
        "topics": sorted({record["topic"] for record in records}),
    }
    write_json(args.persist_dir / DEFAULT_MANIFEST_NAME, manifest)

    print(f"Clean records indexed: {len(records)}")
    print(f"Chunks stored in Chroma: {len(documents)}")
    print(f"Embedding model: {embedding_model}")
    print(f"Vector store directory: {args.persist_dir.resolve()}")


if __name__ == "__main__":
    main()

