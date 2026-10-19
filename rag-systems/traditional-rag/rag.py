"""Traditional RAG over a chat: one retrieval query from the conversation, similarity search on the vector store, then a single LLM call."""

import os

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import add_usage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import OpenAI

TOP_K = 5
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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


def embed(queries: list[str]) -> list[list[float]]:
    """Query embeddings from the remote server running the model the collection was built with."""
    response = embedder.embeddings.create(model=embedding["model_name"], input=queries, encoding_format="float", extra_body={"prompt_name": query_prompt} if query_prompt else {"prompt": ""})
    return [d.embedding for d in response.data]


collection_name, db_dir = os.environ["DATABASE"].split(":", 1)
collection = chromadb.PersistentClient(path=db_dir).get_collection(collection_name)
embedding = collection.configuration_json["embedding_function"]["config"]
query_prompt = embedding["kwargs"].get("default_prompt_name")
embedder = OpenAI(base_url=f"http://{os.environ['EMBEDDER_HOST']}/v1", api_key="not-needed")
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
    result = collection.query(query_embeddings=embed([state["query"]]), n_results=TOP_K, where={"type": "text"})
    return {"contexts": result["documents"][0], "sources": [m["source_file"] for m in result["metadatas"][0]]}


def generate(state: State) -> dict:
    *history, last = state["messages"]
    context = "\n\n---\n\n".join(f"Manual: {source}\n{text}" for source, text in zip(state["sources"], state["contexts"]))
    message = llm.invoke([SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(ANSWER_PROMPT.format(context=context, question=last.text))])
    return {"messages": [message], "tokens": add_usage(state.get("tokens"), message.usage_metadata)}


graph = (
    StateGraph(State, input_schema=MessagesState)
    .add_node(condense)
    .add_node(retrieve)
    .add_node(generate)
    .add_edge(START, "condense")
    .add_edge("condense", "retrieve")
    .add_edge("retrieve", "generate")
    .add_edge("generate", END)
    .compile()
)
