"""Advanced RAG over a chat: one retrieval query from the conversation, hybrid retrieval (vector + BM25) fused with RRF, one LLM answer."""

import os
import re
from collections import defaultdict

import bm25s
import chromadb
import Stemmer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import add_usage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import OpenAI

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
K_PER_RETRIEVER = 5
TOP_K = 5
RRF_K = 60

ANCHOR = re.compile(r"\[\[(?:IMG|TABLE):\w+\]\]")

SYSTEM_PROMPT = """You are a technical assistant for boat owners and crews.
Answer from the manual excerpts in the last message, when present, and from what the user has told you.
Use the excerpts about the products the user asks about.
When the excerpts give only part of the answer, give that part and name what is missing.
Answer every request, short ones included, with the facts from the excerpts and the practical steps, briefly.
Answer in the user's language."""

ANSWER_PROMPT = """Context:
{context}

Question: {question}"""

CONDENSE_PROMPT = """Write the user's latest request as one self-contained question in English, with the details from the conversation. Return only the question.

Conversation:
{conversation}"""


class State(MessagesState):
    query: str
    rankings: list[list[str]]
    candidates: list[str]
    contexts: list[str]
    sources: list[str]
    tokens: dict


def build_llm(spec: str, nothink: bool) -> ChatOpenAI:
    """LLM from a "<model>@<url>" spec: url "google_api" is Gemini, anything else an OpenAI-compatible endpoint."""
    model, _, url = spec.partition("@")
    reasoning_effort = "none" if nothink else None
    if url == "google_api":
        return ChatOpenAI(model=model, base_url=GEMINI_OPENAI_BASE_URL, api_key=os.environ["GOOGLE_API_KEY"], stream_usage=True, reasoning_effort=reasoning_effort)
    return ChatOpenAI(model=model, base_url=url, api_key="not-needed", stream_usage=True, reasoning_effort=reasoning_effort)


def strip_anchors(text: str) -> str:
    return ANCHOR.sub("", text)


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize([strip_anchors(t) for t in texts], stopwords="english", stemmer=stemmer, return_ids=False, show_progress=False)


def embed(queries: list[str]) -> list[list[float]]:
    """Query embeddings from the remote server running the model the collection was built with."""
    response = embedder.embeddings.create(model=embedding["model_name"], input=queries, encoding_format="float", extra_body={"prompt_name": query_prompt} if query_prompt else {"prompt": ""})
    return [d.embedding for d in response.data]


collection_name, db_dir = os.environ["DATABASE"].split(":", 1)
collection = chromadb.PersistentClient(path=db_dir).get_collection(collection_name)
embedding = collection.configuration_json["embedding_function"]["config"]
query_prompt = embedding["kwargs"].get("default_prompt_name")
embedder = OpenAI(base_url=f"http://{os.environ['EMBEDDER_HOST']}/v1", api_key="not-needed")
chunks = collection.get(where={"type": "text"}, include=["documents", "metadatas"])
ids = chunks["ids"]
documents = dict(zip(ids, chunks["documents"]))
source_files = {i: m["source_file"] for i, m in zip(ids, chunks["metadatas"])}

stemmer = Stemmer.Stemmer("english")
bm25 = bm25s.BM25()
bm25.index(tokenize(chunks["documents"]), show_progress=False)

llm = build_llm(os.environ["LLM_MODEL"], os.environ.get("LLM_NOTHINK") == "1")


def condense(state: State) -> dict:
    """Retrieval query: the user message itself in a single-turn chat, otherwise rewritten as a standalone question from the history."""
    messages = state["messages"]
    if sum(isinstance(m, HumanMessage) for m in messages) == 1:
        return {"query": messages[-1].text}
    conversation = "\n".join(f"{m.type}: {m.text}" for m in messages)
    message = llm.invoke(CONDENSE_PROMPT.format(conversation=conversation))
    return {"query": message.text.strip(), "tokens": message.usage_metadata}


def retrieve(state: State) -> dict:
    vector_rankings = collection.query(query_embeddings=embed([state["query"]]), n_results=K_PER_RETRIEVER, where={"type": "text"})["ids"]
    bm25_ids, bm25_scores = bm25.retrieve(tokenize([state["query"]]), corpus=ids, k=K_PER_RETRIEVER, show_progress=False)
    bm25_rankings = [[i for i, score in zip(row_ids, row_scores) if score > 0] for row_ids, row_scores in zip(bm25_ids.tolist(), bm25_scores.tolist())]
    return {"rankings": vector_rankings + bm25_rankings}


def fuse(state: State) -> dict:
    scores = defaultdict(float)
    for ranking in state["rankings"]:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1 / (RRF_K + rank)
    candidates = sorted(scores, key=scores.get, reverse=True)
    top = candidates[:TOP_K]
    return {"candidates": candidates, "contexts": [documents[i] for i in top], "sources": [source_files[i] for i in top]}


def generate(state: State) -> dict:
    *history, last = state["messages"]
    context = "\n\n---\n\n".join(f"Manual: {source}\n{text}" for source, text in zip(state["sources"], state["contexts"]))
    message = llm.invoke([SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(ANSWER_PROMPT.format(context=context, question=last.text))])
    return {"messages": [message], "tokens": add_usage(state.get("tokens"), message.usage_metadata)}


graph = (
    StateGraph(State, input_schema=MessagesState)
    .add_node(condense)
    .add_node(retrieve)
    .add_node(fuse)
    .add_node(generate)
    .add_edge(START, "condense")
    .add_edge("condense", "retrieve")
    .add_edge("retrieve", "fuse")
    .add_edge("fuse", "generate")
    .add_edge("generate", END)
    .compile()
)
