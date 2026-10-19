"""Build the vector store from a parsed knowledge base: text chunks, images and tables."""

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

from chunking import chunk

ANCHOR = re.compile(r"\[\[(?:IMG|TABLE):(\w+)\]\]")
KINDS = {"image": ("picture_ids", "images"), "table": ("table_ids", "tables")}  # type -> (doc metadata key, folder)

MODELS = {  # collection name -> (HF id, document prompt name, SentenceTransformer kwargs)
    "bge-m3": ("BAAI/bge-m3", None, {"model_kwargs": {"dtype": "float32"}}),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", None, {"default_prompt_name": "query", "model_kwargs": {"dtype": "float32"}}),
    "voyage-4-nano": ("voyageai/voyage-4-nano", "document", {"default_prompt_name": "query", "trust_remote_code": True, "model_kwargs": {"dtype": "float32"}}),
    "harrier-0.6b": ("microsoft/harrier-oss-v1-0.6b", None, {"default_prompt_name": "web_search_query", "model_kwargs": {"dtype": "float32"}}),
}


def strip_anchors(text: str) -> str:
    return ANCHOR.sub("", text)


def count_tokens(text: str) -> int:
    """Word count with anchors weighing zero; placeholder until the embedder tokenizer is chosen."""
    return len(strip_anchors(text).split())


def load_items(kb_dir: Path, doc_meta: dict) -> dict[str, dict]:
    """Images and tables of a document that have a retrieval text, by id."""
    items = {}
    for kind, (key, folder) in KINDS.items():
        for item_id in doc_meta[key]:
            meta = json.loads((kb_dir / folder / "metadata" / f"{item_id}.json").read_text(encoding="utf-8"))
            if meta["retrieval_text"]:
                items[item_id] = {"type": kind, "text": meta["retrieval_text"], "path": f"{folder}/{meta['content_file']}"}
    return items


def clean(md: str, items: dict) -> str:
    """Remove the anchors of items that were discarded."""
    return ANCHOR.sub(lambda m: m.group(0) if m.group(1) in items else "", md)


def records(kb_dir: Path, doc_meta: dict) -> list[tuple[str, str, str, dict]]:
    """(id, text to embed, stored value, metadata) for every chunk, image and table of a document."""
    doc_id = doc_meta["doc_id"]
    items = load_items(kb_dir, doc_meta)
    md = clean((kb_dir / doc_meta["md_file"]).read_text(encoding="utf-8"), items)

    rows = [
        (
            f"{doc_id}:{i}",
            strip_anchors(c.text),
            c.text,
            {"type": "text", "doc_id": doc_id, "source_file": doc_meta["source_file"], "sections": " · ".join(c.sections)},
        )
        for i, c in enumerate(chunk(md, count_tokens))
    ]
    rows += [
        (item_id, item["text"], item["path"], {"type": item["type"], "doc_id": doc_id})
        for item_id, item in items.items()
    ]
    return rows


def build_db(kb_dir: Path, db_dir: Path, model: str, text_only: bool = False, device: str = "cpu") -> None:
    """Index kb_dir into the collection named after model; the stored embedding function embeds queries."""
    hf_id, doc_prompt, kwargs = MODELS[model]
    # the stored function stays on cpu so the database can be queried anywhere; documents use the given device
    ef = SentenceTransformerEmbeddingFunction(hf_id, normalize_embeddings=True, **kwargs)
    encoder = ef._model if device == "cpu" else SentenceTransformer(hf_id, device=device, **kwargs).half()
    prompt = encoder.prompts[doc_prompt] if doc_prompt else ""  # "" overrides the default query prompt
    collection = chromadb.PersistentClient(path=str(db_dir)).get_or_create_collection(
        model, embedding_function=ef, configuration={"hnsw": {"space": "cosine"}}
    )
    for path in sorted((kb_dir / "metadata").glob("*.json")):
        doc_meta = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for r in records(kb_dir, doc_meta) if not text_only or r[3]["type"] == "text"]
        ids, keys, values, metas = map(list, zip(*rows))
        embeddings = encoder.encode(keys, prompt=prompt, normalize_embeddings=True, batch_size=8)
        collection.upsert(ids=ids, embeddings=embeddings, documents=values, metadatas=metas)
        print(f"{model} | {doc_meta['md_file']}: {len(ids)} records")
