from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DATASET_NAME = "iblai/ibl-khanacademy-transcripts"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_MIN_WORDS = 50
DEFAULT_MAX_RECORDS = 120
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_COLLECTION_NAME = "khanacademy_rag"
DEFAULT_CLEAN_PATH = Path("train_clean.json")
DEFAULT_VECTORSTORE_DIR = Path("vectorstore")
DEFAULT_MANIFEST_NAME = "index_manifest.json"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
TOPIC_ALL_LABEL = "Todos"

TIMESTAMP_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*$"
)
CUE_INDEX_PATTERN = re.compile(r"^\d+$")
TAG_PATTERN = re.compile(r"\[[^\]]+\]")
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

TOPIC_KEYWORDS = {
    "Matematicas": [
        "algebra",
        "geometry",
        "trigonometry",
        "calculus",
        "equation",
        "inequality",
        "fraction",
        "probability",
        "statistics",
        "ratio",
        "exponent",
        "linear",
        "quadratic",
        "function",
        "derivative",
        "integral",
        "triangle",
        "circle",
        "polynomial",
        "graph",
        "slope",
        "decimal",
        "percent",
    ],
    "Ciencia": [
        "biology",
        "chemistry",
        "physics",
        "energy",
        "fuel",
        "fossil",
        "cell",
        "atom",
        "molecule",
        "photosynthesis",
        "electricity",
        "force",
        "motion",
        "wave",
        "climate",
        "planet",
        "earth",
        "ecosystem",
        "reaction",
        "matter",
    ],
    "Economia y finanzas": [
        "economics",
        "finance",
        "inflation",
        "gdp",
        "market",
        "supply",
        "demand",
        "trade",
        "money",
        "bank",
        "interest",
        "revenue",
        "cost",
        "profit",
        "tax",
        "budget",
        "investment",
        "capital",
        "consumer",
        "producer",
    ],
    "Computacion": [
        "computer",
        "programming",
        "algorithm",
        "python",
        "javascript",
        "html",
        "css",
        "sql",
        "binary",
        "internet",
        "database",
        "debug",
        "coding",
        "code",
    ],
    "Lengua y vocabulario": [
        "grammar",
        "vocabulary",
        "sentence",
        "verb",
        "noun",
        "adjective",
        "pronoun",
        "punctuation",
        "reading",
        "writing",
        "essay",
        "poem",
        "synonym",
        "antonym",
        "prefix",
        "suffix",
        "wordsmith",
        "connotation",
    ],
    "Historia y civismo": [
        "history",
        "government",
        "constitution",
        "war",
        "revolution",
        "empire",
        "president",
        "democracy",
        "civil rights",
        "colonial",
        "ancient",
        "medieval",
        "state",
        "federal",
        "citizen",
        "geography",
    ],
    "Preparacion de examenes": [
        "sat",
        "lsat",
        "gmat",
        "test prep",
        "exam",
        "passage",
        "reading comprehension",
        "practice question",
    ],
    "Arte y humanidades": [
        "art",
        "music",
        "painting",
        "philosophy",
        "theater",
        "humanities",
        "story",
        "narrative",
    ],
}


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", (text or "").strip())


def stable_text_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def dedupe_adjacent_sentences(text: str) -> str:
    sentences = []
    previous = ""
    for sentence in SENTENCE_SPLIT_PATTERN.split(text):
        cleaned = normalize_whitespace(sentence)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered == previous:
            continue
        sentences.append(cleaned)
        previous = lowered
    return " ".join(sentences)


def clean_transcript(raw_text: str) -> str:
    lines = []
    previous_line = ""

    for raw_line in (raw_text or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "WEBVTT":
            continue
        if line.lower().startswith("kind:"):
            continue
        if line.lower().startswith("language:"):
            continue
        if line.startswith("NOTE"):
            continue
        if TIMESTAMP_PATTERN.fullmatch(line):
            continue
        if CUE_INDEX_PATTERN.fullmatch(line):
            continue
        normalized = normalize_whitespace(line)
        if normalized.lower() == previous_line:
            continue
        lines.append(normalized)
        previous_line = normalized.lower()

    text = " ".join(lines)
    text = HTML_PATTERN.sub(" ", text)
    text = TAG_PATTERN.sub(" ", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"(^|\s)-\s+", " ", text)
    text = normalize_whitespace(text)
    text = dedupe_adjacent_sentences(text)
    return normalize_whitespace(text)


def infer_topic(title: str, content: str) -> str:
    title_lower = (title or "").lower()
    preview = (content or "")[:1800].lower()
    blob = f"{title_lower} {preview}"

    scores = {
        topic: sum(1 for keyword in keywords if keyword in blob)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic = max(scores, key=scores.get)
    if scores[best_topic] > 0:
        return best_topic

    short_title = len(title.split()) <= 3
    language_markers = ["word", "noun", "verb", "adjective", "synonym", "meaning"]
    if short_title and any(marker in blob for marker in language_markers):
        return "Lengua y vocabulario"

    return "General"


def normalize_record(raw_record: dict[str, Any], min_words: int = DEFAULT_MIN_WORDS) -> dict[str, Any] | None:
    title = normalize_whitespace(str(raw_record.get("title", "")))
    content = clean_transcript(str(raw_record.get("content", "")))
    url = normalize_whitespace(
        str(raw_record.get("video_url") or raw_record.get("url") or "")
    )
    subtitle_url = normalize_whitespace(str(raw_record.get("url", "")))

    if not title or not content:
        return None

    word_count = len(content.split())
    if word_count < min_words:
        return None

    record_id = stable_text_id(title, url or content[:200])
    return {
        "id": record_id,
        "title": title,
        "content": content,
        "url": url,
        "subtitle_url": subtitle_url,
        "language": normalize_whitespace(str(raw_record.get("language", "en"))) or "en",
        "topic": infer_topic(title, content),
        "word_count": word_count,
    }


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()

    for record in records:
        fingerprint = record["url"] or stable_text_id(
            record["title"], record["content"][:400]
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)

    return deduped


def choose_balanced_subset(
    records: list[dict[str, Any]],
    max_records: int | None = DEFAULT_MAX_RECORDS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not max_records or max_records <= 0 or len(records) <= max_records:
        return records

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["topic"]].append(record)

    rng = random.Random(seed)
    for topic_records in grouped.values():
        rng.shuffle(topic_records)

    ordered_topics = sorted(grouped)
    selected = []

    while len(selected) < max_records and any(grouped.values()):
        for topic in ordered_topics:
            if not grouped[topic]:
                continue
            selected.append(grouped[topic].pop())
            if len(selected) >= max_records:
                break

    return selected


def build_documents(
    records: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for record in records:
        chunks = splitter.split_text(record["content"])
        for index, chunk in enumerate(chunks, start=1):
            documents.append(
                Document(
                    id=f"{record['id']}-{index:03d}",
                    page_content=chunk,
                    metadata={
                        "record_id": record["id"],
                        "title": record["title"],
                        "url": record["url"],
                        "source": record["url"],
                        "subtitle_url": record["subtitle_url"],
                        "topic": record["topic"],
                        "language": record["language"],
                        "word_count": record["word_count"],
                        "chunk_index": index,
                        "chunk_total": len(chunks),
                    },
                )
            )
    return documents


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_clean_records(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_index_manifest(vectorstore_dir: Path) -> dict[str, Any] | None:
    manifest_path = vectorstore_dir / DEFAULT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    return read_json(manifest_path)


def ensure_safe_output_dir(path: Path, project_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Unsafe output directory: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def list_installed_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    models = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def pick_ollama_model(preferred_names: Sequence[str], fallback: str) -> str:
    installed = list_installed_ollama_models()
    installed_lookup = {name.lower(): name for name in installed}

    for candidate in preferred_names:
        if candidate.lower() in installed_lookup:
            return installed_lookup[candidate.lower()]

    for candidate in preferred_names:
        root_name = candidate.replace(":latest", "").lower()
        for installed_name in installed:
            if installed_name.lower().startswith(root_name):
                return installed_name

    return fallback


def resolve_default_llm_model() -> str:
    env_model = os.getenv("OLLAMA_LLM_MODEL")
    if env_model:
        return env_model
    return pick_ollama_model(
        ["llama3.2:latest", "mistral:latest", "qwen2.5:latest"],
        "qwen2.5:latest",
    )


def resolve_default_embedding_model(manifest_model: str | None = None) -> str:
    env_model = os.getenv("OLLAMA_EMBED_MODEL")
    if env_model:
        return env_model
    if manifest_model:
        return manifest_model
    return pick_ollama_model(
        ["nomic-embed-text:latest", "nomic-embed-text"],
        "nomic-embed-text:latest",
    )


def build_context_block(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for index, (doc, score) in enumerate(results, start=1):
        clipped_score = max(0.0, min(float(score), 1.0))
        blocks.append(
            "\n".join(
                [
                    f"[Fuente {index}]",
                    f"Titulo: {doc.metadata.get('title', 'Sin titulo')}",
                    f"Tema: {doc.metadata.get('topic', 'General')}",
                    f"Relevancia: {clipped_score:.3f}",
                    f"URL: {doc.metadata.get('url', '')}",
                    "Contenido:",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def format_sources_markdown(results: list[tuple[Document, float]]) -> str:
    if not results:
        return "No se recuperaron fragmentos con los parametros actuales."

    entries = []
    for index, (doc, score) in enumerate(results, start=1):
        snippet = normalize_whitespace(doc.page_content)[:320]
        clipped_score = max(0.0, min(float(score), 1.0))
        entries.append(
            "\n".join(
                [
                    f"### Fuente {index}",
                    f"**Titulo:** {doc.metadata.get('title', 'Sin titulo')}",
                    f"**Tema:** {doc.metadata.get('topic', 'General')}",
                    f"**Score:** {clipped_score:.3f}",
                    f"**URL:** {doc.metadata.get('url', '')}",
                    f"**Extracto:** {snippet}...",
                ]
            )
        )
    return "\n\n".join(entries)

