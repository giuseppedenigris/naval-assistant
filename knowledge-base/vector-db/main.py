"""CLI: build one embedding model's collection from a parsed knowledge base."""

import argparse
from pathlib import Path

from ingest import MODELS, build_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="folder produced by the parsing pipeline")
    parser.add_argument("db_dir", type=Path, help="Chroma database folder")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--text-only", action="store_true", help="index text chunks only")
    parser.add_argument("--device", default="cpu", help="device for document embeddings, e.g. cuda")
    args = parser.parse_args()
    build_db(args.kb_dir, args.db_dir, args.model, args.text_only, args.device)


if __name__ == "__main__":
    main()
