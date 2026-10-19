"""The document catalog: the title and products of each manual, for the excerpt labels."""

import json
from pathlib import Path

CATALOG_FILE = (Path(__file__).parent / "../../knowledge-base/catalog/data/catalog.json").resolve()


def title(doc_id: str) -> str:
    entry = by_id[doc_id]
    return entry["card"]["title"]["value"] if entry["card"]["title"] else entry["source_file"]


def describe(doc_id: str) -> str:
    """Title of a manual and the products it covers, for the excerpt labels."""
    products = by_id[doc_id]["card"]["products"]
    return f"{title(doc_id)} ({', '.join(p['name'] for p in products)})" if products else title(doc_id)


by_id = {e["doc_id"]: e for e in json.loads(CATALOG_FILE.read_text(encoding="utf-8"))}
