"""Locate retrieval losses on a testset: recall of the plain vector top-k, of the RRF top-n for several cutoffs and of all fused candidates.

Recall is the share of reference characters covered by the chunks, as in naval-qa-test's eval-retrieval.
"""

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

HOP_PREFIX = re.compile(r"^<\d+-hop>\s*")
RRF_CUTOFFS = (5, 10, 15)


def recall(references: list[str], chunks: list[str]) -> float:
    covered_total = 0
    length_total = 0
    for reference in references:
        reference = HOP_PREFIX.sub("", reference)
        covered = [False] * len(reference)
        for chunk in chunks:
            match = SequenceMatcher(None, reference, chunk, autojunk=False).find_longest_match(0, len(reference), 0, len(chunk))
            for i in range(match.a, match.a + match.size):
                covered[i] = True
        covered_total += sum(covered)
        length_total += len(reference)
    return covered_total / length_total if length_total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testset", type=Path)
    parser.add_argument("--llm-model", required=True, help='"<model>@<url>", as in main.py')
    parser.add_argument("--nothink", action="store_true", help="disable the model's thinking, as in main.py")
    parser.add_argument("--embedder-host", required=True, help="host:port of the embedding server, as in main.py")
    parser.add_argument("--database", default=f"harrier-0.6b:{(Path(__file__).parent / '../../knowledge-base/vector-db/data/chroma').resolve()}", help='"<collection>:<chroma directory>", as in main.py')
    args = parser.parse_args()

    load_dotenv()
    os.environ["LANGSMITH_PROJECT"] = "advanced-rag-diagnose"
    os.environ["LLM_MODEL"] = args.llm_model
    os.environ["LLM_NOTHINK"] = "1" if args.nothink else "0"
    os.environ["EMBEDDER_HOST"] = args.embedder_host
    os.environ["DATABASE"] = args.database
    import rag

    rows = [json.loads(line) for line in args.testset.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = []
    for n, row in enumerate(rows, 1):
        question, references = row["user_input"], row["reference_contexts"]
        state = {"messages": [rag.HumanMessage(question)]}
        state |= rag.condense(state)
        state |= rag.expand(state)
        state |= rag.retrieve(state)
        state |= rag.fuse(state)

        candidates = [rag.documents[i] for i in state["candidates"]]
        vector_top = rag.collection.query(query_embeddings=rag.embed([question]),n_results=rag.TOP_K, where={"type": "text"})["documents"][0]
        chunk_sets = {f"vector@{rag.TOP_K}": vector_top, **{f"rrf@{k}": candidates[:k] for k in RRF_CUTOFFS}, "candidates": candidates}
        row_scores = {name: recall(references, chunks) for name, chunks in chunk_sets.items()}
        scores.append(row_scores)
        print(f"{n}/{len(rows)}  " + "  ".join(f"{name}={score:.2f}" for name, score in row_scores.items()) + f"  n_candidates={len(candidates)}  {question[:60]}", flush=True)

    print("\nrecall  " + "  |  ".join(f"{name} = {mean(s[name] for s in scores):.3f}" for name in scores[0]))


if __name__ == "__main__":
    main()
