"""Build the catalog of the manuals: one card per document with its type, scope, manufacturer, products and models, system and boat.

The LLM reads each whole document, tables and image descriptions included. Every identity value comes with the text it was taken from,
and values whose text is not in the document are dropped and flagged. A report lists the flags and a random sample of cards to review.
"""

import argparse
import html
import json
import os
import random
import re
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage
from pydantic import BaseModel, Field

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CATEGORIES = tuple(s for s in json.loads((Path(__file__).parent / "../wiki/systems.json").read_text(encoding="utf-8")) if s != "yacht") + ("battery", "lighting", "safety equipment", "satellite communication", "whole boat", "unknown")
ANCHOR = re.compile(r"\[\[(IMG|TABLE):(\w+)\]\]")
MAX_OUTPUT_TOKENS = 65_536
SAMPLE_SIZE = 30

PROMPT = """You catalogue technical documents for a library that answers the questions of boat owners and crews about the equipment on board.
Read the whole document and fill in its card.

- products: the products the document is about. Leave out products that are only mentioned, such as compatible accessories. For the manual of a whole boat, the only product is the boat.
- models: the names or codes that identify each version of the products the document is about. Write each one in full, as it appears on the product or its nameplate. Codes of spare parts, accessories and consumables are not models. When the document shortens or combines several models into one expression, list each model it stands for.
- manufacturer: the company that makes the products, or that built the boat for the manual of a boat.
- boat: only for documents about a whole boat, its builder and model.
- evidence: for every value that has one, copy the passage of the document it comes from. The passage must be one continuous piece of the text, copied exactly as written: no ellipses, no joining of separate parts, no corrections.
- category: the kind of equipment a product is, or unknown when no value fits.
- summary: one sentence on what the document covers and what it is for."""


class Evidenced(BaseModel):
    value: str
    evidence: str = Field(description="one continuous passage of the document the value comes from, copied exactly as written")


class Product(BaseModel):
    name: str = Field(description="what the product is, with its brand and series")
    category: Literal[CATEGORIES] = Field(description="the kind of equipment the product is")
    models: list[Evidenced] = Field(description="each model of the product covered by the document, written in full")


class Card(BaseModel):
    title: Evidenced = Field(description="title of the document as printed")
    document_type: Literal["user manual", "installation manual", "service manual", "quick guide", "datasheet", "catalog or brochure", "boat owner's manual", "guide or regulation", "other"] = Field(description="user manual: how to operate and maintain the products; installation manual: how to install them; service manual: repair and overhaul for technicians; quick guide: a short reference; datasheet: specifications; catalog or brochure: a range of products on offer; boat owner's manual: the manual of a whole boat, from its builder; guide or regulation: general knowledge, rules or standards not tied to one product; other: anything else")
    scope: Literal["one product", "one product series", "several product lines", "whole boat", "general topic"] = Field(description="one product: a single model; one product series: the models of one series; several product lines: different kinds of products; whole boat: a boat and its systems; general topic: knowledge not tied to specific products")
    manufacturer: Evidenced | None
    products: list[Product]
    boat: Evidenced | None
    language: str = Field(description="ISO 639-1 code of the main language")
    summary: str


def build_client(spec: str) -> tuple[OpenAI, str]:
    """Client and model from a "<model>@<url>" spec: url "google_api" is Gemini, anything else an OpenAI-compatible endpoint."""
    model, _, url = spec.partition("@")
    if url == "google_api":
        return OpenAI(base_url=GEMINI_OPENAI_BASE_URL, api_key=os.environ["GOOGLE_API_KEY"], max_retries=5), model
    return OpenAI(base_url=url, api_key="not-needed", max_retries=5), model


def table_text(path: Path) -> str:
    rows = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").find_all("tr")
    return "\n".join(" | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])) for row in rows)


def expand(kb_dir: Path, match: re.Match) -> str:
    """The text of an anchor: the table itself or the image description; empty for items filtered out at ingestion."""
    kind, item_id = match.groups()
    folder = "tables" if kind == "TABLE" else "images"
    meta = json.loads((kb_dir / folder / "metadata" / f"{item_id}.json").read_text(encoding="utf-8"))
    if meta["filtered_out"]:
        return ""
    if kind == "TABLE":
        return table_text(kb_dir / folder / meta["content_file"])
    return f"[Image: {meta['retrieval_text']}]" if meta["retrieval_text"] else ""


def full_text(kb_dir: Path, doc_meta: dict) -> str:
    return ANCHOR.sub(lambda m: expand(kb_dir, m), (kb_dir / doc_meta["md_file"]).read_text(encoding="utf-8"))


def compact(text: str) -> str:
    """Letters and digits only, lowercase: text compared regardless of spacing, punctuation and markdown."""
    return re.sub(r"[\W_]+", "", html.unescape(text)).casefold()


def grounded(evidence: str, document: str) -> bool:
    return bool(compact(evidence)) and compact(evidence) in document


def verify(card: dict, text: str) -> tuple[list[str], list[dict]]:
    """Keeps a value when its evidence or the value itself is in the document; returns what to review and the values dropped."""
    document = compact(text)
    flags, rejected = [], []

    def keep(field: str, item: dict, product: str | None) -> bool:
        if grounded(item["evidence"], document):
            if field == "model" and compact(item["value"]) not in compact(item["evidence"]):
                flags.append(f"model {item['value']!r} differs from its evidence {item['evidence']!r}")
            return True
        if grounded(item["value"], document):
            flags.append(f"{field} {item['value']!r}: evidence not literal, value found in the document")
            return True
        flags.append(f"{field} {item['value']!r}: neither evidence nor value in the document")
        rejected.append({"field": field, "product": product, **item})
        return False

    for field in ("title", "manufacturer", "boat"):
        if card[field] and not keep(field, card[field], None):
            card[field] = None
    for product in card["products"]:
        product["models"] = [m for m in product["models"] if keep("model", m, product["name"])]
    if card["scope"] in ("one product", "one product series") and not any(p["models"] for p in card["products"]):
        flags.append("no model")
    if card["document_type"] == "other":
        flags.append("document type other")
    return flags, rejected


def extract(client: OpenAI, model: str, text: str) -> tuple[Card, CompletionUsage]:
    completion = client.chat.completions.parse(model=model, messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}], response_format=Card, max_completion_tokens=MAX_OUTPUT_TOKENS, reasoning_effort="low", temperature=0)
    return completion.choices[0].message.parsed, completion.usage


def with_evidence(field: dict | None) -> str:
    return f"{field['value']!r} <- {field['evidence']!r}" if field else "-"


def describe(entry: dict) -> str:
    card = entry["card"]
    lines = [f"## {entry['md_file']}", f"title: {with_evidence(card['title'])}", f"{card['document_type']} | {card['scope']} | {card['language']}", f"manufacturer: {with_evidence(card['manufacturer'])}", f"boat: {with_evidence(card['boat'])}"]
    lines += [f"product: {p['name']} ({p['category']}) | models: " + ", ".join(with_evidence(m) for m in p["models"]) for p in card["products"]]
    return "\n".join(lines + [f"summary: {card['summary']}", f"flags: {entry['flags']}"])


def report(entries: list[dict]) -> str:
    flagged = [e for e in entries if e["flags"]]
    sample = random.Random(0).sample(entries, min(SAMPLE_SIZE, len(entries)))
    lines = [f"cards: {len(entries)} | flagged: {len(flagged)}", "", "# Flagged", *(f"{e['md_file']}: " + "; ".join(e["flags"]) for e in flagged), "", "# Sample"]
    return "\n".join(lines + [describe(e) + "\n" for e in sample])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="folder of the parsed documents (.md, metadata/, tables/, images/)")
    parser.add_argument("out_dir", type=Path, help="folder for the cards and report.md; finalize.py turns the cards into catalog.json")
    parser.add_argument("--llm-model", required=True, help='"<model>@<url>": url "google_api" means Gemini cloud, anything else an OpenAI-compatible endpoint')
    parser.add_argument("--docs", nargs="*", help="only these .md files")
    args = parser.parse_args()

    load_dotenv()
    client, model = build_client(args.llm_model)
    cards_dir = args.out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((args.kb_dir / "metadata").glob("*.json"))]
    selected = [m for m in metas if not args.docs or m["md_file"] in args.docs]

    for n, meta in enumerate(selected, 1):
        path = cards_dir / f"{Path(meta['md_file']).stem}.json"
        if path.exists():
            continue
        text = full_text(args.kb_dir, meta)
        try:
            card, usage = extract(client, model, text)
        except Exception as error:
            print(f"{n}/{len(selected)} {meta['md_file']}: failed, {error}", flush=True)
            continue
        card = card.model_dump()
        flags, rejected = verify(card, text)
        path.write_text(json.dumps({"doc_id": meta["doc_id"], "md_file": meta["md_file"], "source_file": meta["source_file"], "llm_model": model, "card": card, "flags": flags, "rejected": rejected}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{n}/{len(selected)} {meta['md_file']}: {card['document_type']}, {sum(len(p['models']) for p in card['products'])} models, {len(flags)} flags | tokens {usage.prompt_tokens} in, {usage.completion_tokens} out", flush=True)

    entries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cards_dir.glob("*.json"))]
    (args.out_dir / "report.md").write_text(report(entries), encoding="utf-8")
    print(f"catalog: {len(entries)} cards, {sum(bool(e['flags']) for e in entries)} flagged -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
