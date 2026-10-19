"""Build a LightRAG knowledge graph from parsed .md documents."""

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, TokenTracker
from openai import AsyncOpenAI

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDER = "microsoft/harrier-oss-v1-0.6b"
EMBEDDING_DIM = 1024
QUERY_PROMPT = "web_search_query"
EMBEDDING_TIMEOUT_S = 600

ENTITY_TYPES = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Product: a piece of equipment, model or product family made or sold by a manufacturer
- Component: a physical part of a product
- Manufacturer: a company that makes or sells products
- System: a kind of on-board system of a boat
- Procedure: a way to install, configure, operate, test or maintain equipment
- Setting: a configurable parameter or menu option
- Alarm: an alarm, warning, fault or error message or code
- Specification: a technical value, rating or requirement"""


def build_llm(spec: str, tracker: TokenTracker):
    """LightRAG completion function from a "<model>@<url>" spec: url "google_api" is Gemini, anything else an OpenAI-compatible endpoint."""
    model, _, url = spec.partition("@")
    base_url, api_key = (GEMINI_OPENAI_BASE_URL, os.environ["GOOGLE_API_KEY"]) if url == "google_api" else (url, "not-needed")

    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(model, prompt, system_prompt=system_prompt, history_messages=history_messages or [], base_url=base_url, api_key=api_key, token_tracker=tracker, **kwargs)

    return complete, model


def build_embedder(url: str) -> EmbeddingFunc:
    """Harrier, the embedder of the vector store, served over HTTP: queries get its query prompt, documents none."""
    client = AsyncOpenAI(base_url=url, api_key="not-needed")

    async def embed(texts, context="document"):
        extra = {"prompt_name": QUERY_PROMPT} if context == "query" else {}
        response = await client.embeddings.create(model=EMBEDDER, input=list(texts), encoding_format="float", extra_body=extra)
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    return EmbeddingFunc(embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=embed, model_name=EMBEDDER, supports_asymmetric=True)


async def build(kb_dir: Path, work_dir: Path, llm_spec: str, embedder_url: str, docs: list[str]) -> None:
    tracker = TokenTracker()
    llm, model = build_llm(llm_spec, tracker)
    rag = LightRAG(working_dir=str(work_dir), llm_model_func=llm, llm_model_name=model, embedding_func=build_embedder(embedder_url), embedding_func_max_async=1, default_embedding_timeout=EMBEDDING_TIMEOUT_S, addon_params={"entity_types_guidance": ENTITY_TYPES})
    await rag.initialize_storages()
    for name in docs:
        await rag.ainsert((kb_dir / name).read_text(encoding="utf-8"), ids=name, file_paths=name)
        print(f"{name}: done | tokens so far {tracker.get_usage()}", flush=True)
    await rag.finalize_storages()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="folder of the parsed .md documents")
    parser.add_argument("work_dir", type=Path, help="folder of the LightRAG storage")
    parser.add_argument("--llm-model", required=True, help='"<model>@<url>": url "google_api" means Gemini cloud, anything else an OpenAI-compatible endpoint')
    parser.add_argument("--embedder-url", required=True, help="OpenAI-compatible embedding server running Harrier, e.g. http://192.168.0.103:8080/v1")
    parser.add_argument("--docs", nargs="+", required=True, help=".md files to insert")
    args = parser.parse_args()

    load_dotenv()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(build(args.kb_dir, args.work_dir, args.llm_model, args.embedder_url, args.docs))


if __name__ == "__main__":
    main()
