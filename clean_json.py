from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from rag_utils import (
    DEFAULT_CLEAN_PATH,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_MAX_RECORDS,
    DEFAULT_MIN_WORDS,
    choose_balanced_subset,
    deduplicate_records,
    load_clean_records,
    normalize_record,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Khan Academy transcripts for the Ollama + LangChain RAG project."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional local JSON file instead of downloading from Hugging Face.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source_records(args: argparse.Namespace) -> list[dict]:
    if args.input_json:
        payload = read_json(args.input_json)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("train"), list):
            return payload["train"]
        raise ValueError(f"Unsupported JSON structure in {args.input_json}")

    dataset = load_dataset(args.dataset_name, split=args.split)
    return list(dataset)


def main() -> None:
    args = parse_args()
    raw_records = load_source_records(args)

    cleaned_records = []
    for raw_record in raw_records:
        normalized = normalize_record(raw_record, min_words=args.min_words)
        if normalized is not None:
            cleaned_records.append(normalized)

    deduped_records = deduplicate_records(cleaned_records)
    selected_records = choose_balanced_subset(
        deduped_records,
        max_records=args.max_records,
        seed=args.seed,
    )
    write_json(args.output, selected_records)

    topic_counts = Counter(record["topic"] for record in selected_records)

    print(f"Raw records loaded: {len(raw_records)}")
    print(f"Records after cleaning: {len(cleaned_records)}")
    print(f"Records after deduplication: {len(deduped_records)}")
    print(f"Records written to {args.output.resolve()}: {len(selected_records)}")
    print("Topic distribution:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  - {topic}: {count}")


if __name__ == "__main__":
    main()

