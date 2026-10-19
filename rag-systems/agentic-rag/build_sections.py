"""Embeds the sections missing from retrieval.py's vector cache, so that no search has to embed them on the fly.
Resumable: the cache is saved every few minutes and the sections already in it are skipped."""

import argparse
import os
import time

from main import NAVAL_DB

BATCH_CHARS = 40_000
BATCH_SIZE = 64
SAVE_EVERY = 300  # seconds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--embedder-host", required=True, help="host:port of the embedding server, as in main.py")
parser.add_argument("--database", default=NAVAL_DB, help='"<collection>:<chroma directory>", as in main.py')
args = parser.parse_args()
if ":" not in args.database:
    parser.error(f"--database must be '<collection>:<chroma directory>', got: {args.database!r}")
os.environ["EMBEDDER_HOST"] = args.embedder_host
os.environ["DATABASE"] = args.database

import retrieval  # noqa: E402  reads the settings from the environment


def batches(section_ids: list[str]) -> list[list[str]]:
    """Consecutive groups of sections, each up to BATCH_SIZE sections or BATCH_CHARS characters."""
    groups, chars = [], 0
    for s in section_ids:
        if not groups or chars + len(retrieval.texts[s]) > BATCH_CHARS or len(groups[-1]) == BATCH_SIZE:
            groups.append([])
            chars = 0
        groups[-1].append(s)
        chars += len(retrieval.texts[s])
    return groups


missing = {retrieval.section_key(s): s for s in retrieval.section_ids if retrieval.section_key(s) not in retrieval.vectors}  # one section per text
todo = sorted(missing.values(), key=lambda s: len(retrieval.texts[s]))  # similar lengths batch with less padding
total_chars = sum(len(retrieval.texts[s]) for s in todo)
print(f"{len(retrieval.vectors)} vectors cached, {len(todo)} sections to embed", flush=True)

start = saved = time.perf_counter()
done = chars = 0
for batch in batches(todo):
    retrieval.embed_sections(batch)
    done += len(batch)
    chars += sum(len(retrieval.texts[s]) for s in batch)
    if time.perf_counter() - saved > SAVE_EVERY or batch[-1] == todo[-1]:
        retrieval.save_vectors()
        saved = time.perf_counter()
        rate = chars / (saved - start)
        print(f"{done}/{len(todo)} sections | {(saved - start) / 60:.0f} min | about {(total_chars - chars) / rate / 60:.0f} min left", flush=True)
