"""Search of the manuals: hybrid retrieval of text chunks (vector + BM25, fused with RRF) in the whole library, condensed to the sections of those chunks that best match the query (cached section vectors + BM25, fused with RRF)."""

import hashlib
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import bm25s
import chromadb
import numpy as np
import Stemmer
from openai import OpenAI

VECTORS = Path(__file__).parent / "data/section_vectors.npz"  # cache of section vectors by hash of model and text
K_PER_RETRIEVER = 15
TOP_K = 15
RRF_K = 60
WORD_BUDGET = 1500

ANCHOR = re.compile(r"\[\[(?:IMG|TABLE):\w+\]\]")
BLANK_LINES = re.compile(r"\n{3,}")  # left behind by the removed anchors
HEADER = re.compile(r"^#{1,6}\s+(.*)")


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="english", stemmer=stemmer, return_ids=False, show_progress=False)


@cache
def embed_query(query: str) -> np.ndarray:
    """Query embedding from the remote server running the model the collection was built with."""
    response = embedder.embeddings.create(model=embedding["model_name"], input=[query], encoding_format="float", extra_body={"prompt_name": query_prompt} if query_prompt else {"prompt": ""})
    return np.array(response.data[0].embedding, dtype=np.float32)


def section_key(section_id: str) -> str:
    return hashlib.sha1(f"{embedding['model_name']}\n{texts[section_id]}".encode()).hexdigest()


def embed_sections(section_ids: list[str]) -> None:
    """Adds the section vectors to the cache, embedded with no prompt as the collection embedded the chunks."""
    response = embedder.embeddings.create(model=embedding["model_name"], input=[texts[s] for s in section_ids], encoding_format="float", extra_body={"prompt": ""})
    vectors.update((section_key(s), np.array(d.embedding, dtype=np.float16)) for s, d in zip(section_ids, response.data))


def save_vectors() -> None:
    """Writes a partial file and then moves it into place, so an interrupted save keeps the previous one."""
    partial = VECTORS.with_suffix(".partial.npz")
    VECTORS.parent.mkdir(exist_ok=True)
    np.savez(partial, keys=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))
    partial.replace(VECTORS)


def fuse(rankings: list[list[str]]) -> list[str]:
    """The ids of the rankings ordered by reciprocal rank fusion."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] += 1 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def split_sections(text: str, title: str) -> list[tuple[str, str]]:
    """(title, text) of the sections of a chunk: a header with no text below joins the next one, text before the first header continues the given title."""
    parts, titles, lines = [], [title], []
    for line in text.splitlines():
        header = HEADER.match(line)
        if header and any(l.strip() and not HEADER.match(l) for l in lines):
            parts.append((" · ".join(titles), "\n".join(lines).strip()))
            titles, lines = [], []
        if header:
            titles = titles if lines else []
            titles.append(header.group(1).strip())
        lines.append(line)
    return parts + [(" · ".join(titles), "\n".join(lines).strip())] if lines else parts


def rankings(query: str) -> list[list[str]]:
    """The ids of the chunks found by vector search and by BM25, best first."""
    vector_ids = collection.query(query_embeddings=[embed_query(query)], n_results=K_PER_RETRIEVER, where={"type": "text"})["ids"][0]
    bm25_ids, bm25_scores = bm25.retrieve(tokenize([query]), corpus=chunk_ids, k=K_PER_RETRIEVER, show_progress=False)
    return [vector_ids, [i for i, s in zip(bm25_ids[0].tolist(), bm25_scores[0].tolist()) if s > 0]]


def condense(query: str, chunk_ids: list[str]) -> list[str]:
    """The sections of the chunks that best match the query by vector similarity and BM25, fused with RRF, each with the
    section that follows it in the manual, up to WORD_BUDGET words: manuals in the order of their best section, each in reading order."""
    candidates = [s for c in chunk_ids for s in chunk_sections[c]]
    if not candidates:
        return []
    missing =[s for s in candidates if section_key(s) not in vectors]
    if missing:
        embed_sections(missing)
        save_vectors()
    similarity = np.stack([vectors[section_key(s)] for s in candidates]).astype(np.float32) @ embed_query(query)
    tokens = tokenize([query])[0]
    keyword = bm25s.BM25()
    keyword.index(tokenize([texts[s] for s in candidates]), show_progress=False)
    matches = keyword.get_scores(tokens) if tokens else np.zeros(len(candidates))
    by_similarity = [candidates[n] for n in np.argsort(-similarity)]
    by_keyword = [candidates[n] for n in np.argsort(-matches) if matches[n] > 0]
    picked, words = [], 0
    for section_id in fuse([by_similarity, by_keyword]):
        for s in (section_id, following.get(section_id)):
            if s and s not in picked:
                picked.append(s)
                words += len(texts[s].split())
        if words >= WORD_BUDGET:
            break
    manual_rank = {}
    for s in picked:
        manual_rank.setdefault(doc_of[s], len(manual_rank))
    return sorted(picked, key=lambda s: (manual_rank[doc_of[s]], section_row[s]))


def search(query: str) -> list[str]:
    """Ids of the sections that best match the query, from the TOP_K chunks found by vector search and BM25, fused with RRF."""
    return condense(query, fuse(rankings(query))[:TOP_K])


collection_name, db_dir = os.environ["DATABASE"].split(":", 1)
collection = chromadb.PersistentClient(path=db_dir).get_collection(collection_name)
embedding = collection.configuration_json["embedding_function"]["config"]
query_prompt = embedding["kwargs"].get("default_prompt_name")
embedder = OpenAI(base_url=f"http://{os.environ['EMBEDDER_HOST']}/v1", api_key="not-needed")
chunks = collection.get(where={"type": "text"}, include=["documents", "metadatas"])
chunk_ids = chunks["ids"]
chunk_doc = dict(zip(chunk_ids, (m["doc_id"] for m in chunks["metadatas"])))
chunk_texts = dict(zip(chunk_ids, (BLANK_LINES.sub("\n\n", ANCHOR.sub("", d)).strip() for d in chunks["documents"])))

texts, doc_of, chunk_sections, following = {}, {}, defaultdict(list), {}  # section id -> text, manual, next section
previous, title = None, ""
for chunk_id in sorted(chunk_ids, key=lambda i: (i.rpartition(":")[0], int(i.rpartition(":")[2]))):
    if previous and doc_of[previous] != chunk_doc[chunk_id]:
        previous, title = None, ""
    for k, (title, text) in enumerate(split_sections(chunk_texts[chunk_id], title)):
        section_id = f"{chunk_id}/{k}"
        texts[section_id] = text if HEADER.match(text) or not title else f"## {title}\n\n{text}"
        doc_of[section_id] = chunk_doc[chunk_id]
        chunk_sections[chunk_id].append(section_id)
        if previous:
            following[previous] = section_id
        previous = section_id
section_ids = list(texts)  # reading order
section_row = {section_id: n for n, section_id in enumerate(section_ids)}

vectors = {}  # section key -> vector
if VECTORS.exists():
    with np.load(VECTORS) as stored:
        vectors = dict(zip(stored["keys"].tolist(), stored["vectors"]))

stemmer = Stemmer.Stemmer("english")
bm25 = bm25s.BM25()
bm25.index(tokenize([chunk_texts[i] for i in chunk_ids]), show_progress=False)
