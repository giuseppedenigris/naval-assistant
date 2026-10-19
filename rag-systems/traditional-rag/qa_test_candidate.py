"""naval-qa-test candidate for this RAG, symlinked as naval-qa-test/candidates/traditional_rag.py.

It runs inside the evaluator's process (stdlib only) and calls this RAG over HTTP,
so start the server first: uv run main.py --llm-model <model>@<url> --embedder-host <host:port>
"""

from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:2024/runs/wait"


def answer(input: dict) -> dict:
    """input is the graph input as is: {"messages": [{"role": "user" | "assistant", "content": "..."}, ...]}."""
    body = json.dumps({"assistant_id": "rag", "input": input}).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        state = json.load(response)
    latency_s = time.perf_counter() - start

    return {
        "response": state["messages"][-1]["content"],
        "contexts": state["contexts"],
        "latency_s": latency_s,
        "tokens": state["tokens"],
    }
