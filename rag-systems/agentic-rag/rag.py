"""Agentic RAG over a chat: small LLM steps tell whether to search and rewrite the request; the code searches the whole
library; the excerpts are judged and searched once more if needed; the answer comes from the excerpts."""

import json
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import add_usage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

import catalog
import retrieval

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

ROUTE_PROMPT = """You read a conversation between a boat owner and a technical assistant.
Decide whether to search the manuals for the user's latest message.

Answer with a JSON object following this schema:
{schema}"""

REWRITE_PROMPT = """You read a conversation between a boat owner and a technical assistant.
Rewrite the user's latest request so that it reads on its own.

Answer with a JSON object following this schema:
{schema}"""

GRADE_PROMPT = """Question: {question}

Excerpts from the manuals:
{context}

Judge whether the excerpts answer the question. When information is missing, write a search query for it in the words a manual uses.

Answer with a JSON object following this schema:
{schema}"""

SYSTEM_PROMPT = """You are a technical assistant for boat owners and crews.
Answer from the manual excerpts in the last message, when present, and from what the user has told you.
Use the excerpts about the products the user asks about.
When the excerpts give only part of the answer, give that part and name what is missing.
Answer every request, short ones included, with the facts from the excerpts and the practical steps, briefly.
Answer in the user's language."""

ANSWER_PROMPT = """Excerpts:
{context}

Request: {question}"""


class Route(BaseModel):
    search: bool = Field(description="false for greetings, thanks and comments that ask nothing; true for every question or request")


class Rewrite(BaseModel):
    question: str = Field(description="the request as a self-contained question in English, with the details from the conversation")


class Grade(BaseModel):
    enough: bool = Field(description="true when the excerpts answer the question")
    query: str | None = Field(description="the search query for the missing information, when enough is false")


class State(MessagesState):
    search: bool
    question: str
    queries: list[str]
    sections: list[str]
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


def schema_text(schema: type[BaseModel]) -> str:
    """The JSON schema of a structured answer, written into the prompt too: llama.cpp and Ollama enforce it while decoding but never show it to the model."""
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)


def structured(schema: type[BaseModel], prompt: str, messages: list, tokens: dict | None) -> tuple[BaseModel, dict]:
    """One structured LLM call: the prompt as system message before the conversation, or as the user message when there is none (chat templates such as Qwen's require one)."""
    prompt = prompt.replace("{schema}", schema_text(schema))
    result = llm.with_structured_output(schema, method="json_schema", include_raw=True).invoke([SystemMessage(prompt), *messages] if messages else [HumanMessage(prompt)])
    return result["parsed"], add_usage(tokens, result["raw"].usage_metadata)


def pack(section_ids: list[str]) -> str:
    """The sections in the given order, under the label of their manual, a new label whenever the manual changes."""
    lines, manual = [], None
    for section_id in section_ids:
        if retrieval.doc_of[section_id] != manual:
            manual = retrieval.doc_of[section_id]
            lines.append(f"Manual: {catalog.describe(manual)}")
        lines.append(retrieval.texts[section_id])
    return "\n\n".join(lines)


llm = build_llm(os.environ["LLM_MODEL"], os.environ.get("LLM_NOTHINK") == "1")


def route(state: State) -> dict:
    """Whether the latest message needs the manuals; resets what the previous turn left in the state."""
    found, tokens = structured(Route, ROUTE_PROMPT, state["messages"], None)
    return {"search": found.search, "queries": [], "sections": [], "contexts": [], "sources": [], "tokens": tokens}


def rewrite(state: State) -> dict:
    found, tokens = structured(Rewrite, REWRITE_PROMPT, state["messages"], state["tokens"])
    return {"question": found.question, "queries": [found.question], "tokens": tokens}


def retrieve(state: State) -> dict:
    found = retrieval.search(state["queries"][-1])
    return {"sections": state["sections"] + [s for s in found if s not in state["sections"]]}


def grade(state: State) -> dict:
    prompt = GRADE_PROMPT.replace("{question}", state["question"]).replace("{context}", pack(state["sections"]) or "none")
    found, tokens = structured(Grade, prompt, [], state["tokens"])
    return {"queries": state["queries"] + ([found.query] if not found.enough and found.query else []), "tokens": tokens}


def answer(state: State) -> dict:
    *history, last = state["messages"]
    if not state["search"]:
        message = llm.invoke([SystemMessage(SYSTEM_PROMPT), *history, last])
        return {"messages": [message], "tokens": add_usage(state["tokens"], message.usage_metadata)}
    request = ANSWER_PROMPT.format(context=pack(state["sections"]) or "none", question=last.text)
    message = llm.invoke([SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(request)])
    sources = list(dict.fromkeys(catalog.describe(retrieval.doc_of[s]) for s in state["sections"]))
    return {"messages": [message], "contexts": [retrieval.texts[s] for s in state["sections"]], "sources": sources, "tokens": add_usage(state["tokens"], message.usage_metadata)}


def after_route(state: State) -> Literal["rewrite", "answer"]:
    return "rewrite" if state["search"] else "answer"


def after_retrieve(state: State) -> Literal["grade", "answer"]:
    return "grade" if len(state["queries"]) == 1 else "answer"


def after_grade(state: State) -> Literal["retrieve", "answer"]:
    return "retrieve" if len(state["queries"]) == 2 else "answer"


graph = (
    StateGraph(State, input_schema=MessagesState)
    .add_node(route)
    .add_node(rewrite)
    .add_node(retrieve)
    .add_node(grade)
    .add_node(answer)
    .add_edge(START, "route")
    .add_conditional_edges("route", after_route)
    .add_edge("rewrite", "retrieve")
    .add_conditional_edges("retrieve", after_retrieve)
    .add_conditional_edges("grade", after_grade)
    .add_edge("answer", END)
    .compile()
)
