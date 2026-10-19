"""Serve the traditional RAG graph with the LangGraph dev server (HTTP API, Studio, LangSmith tracing)."""

import argparse
import os
import subprocess
from pathlib import Path

NAVAL_DB = f"harrier-0.6b:{(Path(__file__).parent / '../../knowledge-base/vector-db/data/chroma').resolve()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", required=True, help='"<model>@<url>": url "google_api" means Gemini cloud, anything else an OpenAI-compatible endpoint such as a local Ollama (e.g. http://localhost:11434/v1)')
    parser.add_argument("--nothink", action="store_true", help="disable the model's thinking (reasoning_effort=none)")
    parser.add_argument("--embedder-host", required=True, help="host:port of the OpenAI-compatible embedding server running Harrier, e.g. localhost:8080")
    parser.add_argument("--database", default=NAVAL_DB, help='"<collection>:<chroma directory>"')
    parser.add_argument("--port", type=int, default=2024)
    args = parser.parse_args()
    if "@" not in args.llm_model:
        parser.error(f"--llm-model must be '<model>@<url>', got: {args.llm_model!r}")
    if ":" not in args.database:
        parser.error(f"--database must be '<collection>:<chroma directory>', got: {args.database!r}")

    # langgraph dev loads rag.py itself, so the settings reach it through the environment
    os.environ["LLM_MODEL"] = args.llm_model
    os.environ["LLM_NOTHINK"] = "1" if args.nothink else "0"
    os.environ["EMBEDDER_HOST"] = args.embedder_host
    os.environ["DATABASE"] = args.database
    raise SystemExit(subprocess.call(["langgraph", "dev", "--no-reload", "--port", str(args.port)]))


if __name__ == "__main__":
    main()
