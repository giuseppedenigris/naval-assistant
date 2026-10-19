"""Wiki build, step 1: read each document once and turn it into a card of short facts, each tied to the section it comes from."""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from sections import ANCHOR, label_sections, split_sections

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SYSTEMS_FILE = Path(__file__).parent / "systems.json"
WINDOW_CHARS = 20_000
MAX_OUTPUT_TOKENS = 32_768

PROMPT = """You turn a technical document about boat equipment into notes for a wiki. The readers of the wiki are boat owners and crews who install, configure, operate, troubleshoot and maintain the equipment.

The document is split into sections, each starting with a label such as "## [§42] Title". The user message contains part of the document.

Return:
- title and document_type of the whole document.
- system: the system the document belongs to, one of: {systems}. The search that found the document suggests "{default_system}".
- entities: the concrete things the text is about: products, product families, their physical components and manufacturers. Give the brand, the product a component is part of (empty for the others) and the other names the text uses for them.
- facts: short notes, each with:
  - product: the entity the note applies to, written exactly as in your entity list, or empty when the note does not apply to a specific one;
  - topic: what the note is about, in a few words;
  - aspect: the kind of information;
  - statement: one piece of information in your own words, keeping values, units, part numbers, menu paths and procedure steps exactly as the text gives them. A procedure is one fact with all its steps;
  - section: the label number of the one section the note comes from;
  - anchors: the [[IMG:...]] and [[TABLE:...]] anchors of that section that illustrate the note, copied verbatim, or none.

Write only what the text says. Skip legal notices, copyright and marketing claims."""


class Entity(BaseModel):
    name: str
    kind: Literal["product", "product family", "component", "manufacturer"]
    brand: str
    part_of: str
    aliases: list[str]


class Fact(BaseModel):
    product: str
    topic: str
    aspect: Literal["overview", "specifications", "installation", "configuration", "operation", "troubleshooting", "maintenance", "warnings"]
    statement: str
    section: int
    anchors: list[str]


class Extraction(BaseModel):
    title: str
    document_type: Literal["user manual", "installation manual", "service manual", "datasheet", "catalog", "brochure", "boat owner's manual", "other"]
    system: str
    entities: list[Entity]
    facts: list[Fact]


def build_client(spec: str) -> tuple[OpenAI, str]:
    """Client and model from a "<model>@<url>" spec: url "google_api" is Gemini, anything else an OpenAI-compatible endpoint."""
    model, _, url = spec.partition("@")
    if url == "google_api":
        return OpenAI(base_url=GEMINI_OPENAI_BASE_URL, api_key=os.environ["GOOGLE_API_KEY"]), model
    return OpenAI(base_url=url, api_key="not-needed"), model


def key(name: str) -> str:
    """Name for matching: lowercase, letters and digits only."""
    return re.sub(r"[\W_]+", "", name).lower()


def windows(sections: list[tuple[int, str, str]]) -> list[list[str]]:
    """Consecutive section texts grouped into windows of about WINDOW_CHARS characters."""
    groups, current = [], []
    for _, _, text in sections:
        if current and sum(map(len, current)) + len(text) > WINDOW_CHARS:
            groups.append(current)
            current = []
        current.append(text)
    return groups + [current] if current else groups


def extract_window(client: OpenAI, model: str, window: list[str], systems: list[str], default_system: str) -> list[Extraction]:
    """Extraction of a window; a window whose output hits the length limit is split in two halves."""
    prompt = PROMPT.format(systems=", ".join(systems), default_system=default_system)
    try:
        completion = client.chat.completions.parse(model=model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "".join(window)}], response_format=Extraction, max_completion_tokens=MAX_OUTPUT_TOKENS)
        return [completion.choices[0].message.parsed]
    except LengthFinishReasonError:
        if len(window) == 1:
            raise
        half = len(window) // 2
        return extract_window(client, model, window[:half], systems, default_system) + extract_window(client, model, window[half:], systems, default_system)


def extract_document(client: OpenAI, model: str, path: Path, systems: list[str], default_system: str) -> dict:
    sections = split_sections(label_sections(path.read_text(encoding="utf-8")))
    texts = {n: text for n, _, text in sections}
    parts = [part for window in windows(sections) for part in extract_window(client, model, window, systems, default_system)]

    entities = {key(e.name): e.model_dump() for part in parts for e in part.entities}
    names = {key(a): e["name"] for e in entities.values() for a in [e["name"], *e["aliases"]]}

    facts, dropped, unlisted = [], Counter(), 0
    for fact in (f for part in parts for f in part.facts):
        if fact.section not in texts:
            dropped["unknown section"] += 1
            continue
        if fact.product and key(fact.product) not in names:
            entities[key(fact.product)] = {"name": fact.product, "kind": "product", "brand": "", "part_of": "", "aliases": []}
            names[key(fact.product)] = fact.product
            unlisted += 1
        fact.product = names.get(key(fact.product), "")
        fact.anchors = [a for a in fact.anchors if ANCHOR.fullmatch(a) and a in texts[fact.section]]
        facts.append(fact.model_dump() | {"component": ""})

    roll_up_components(facts, entities, names)
    used = {name for f in facts for name in (f["product"], f["component"])}
    entities = [e for e in entities.values() if e["name"] in used]

    system = parts[0].system if parts[0].system in systems else default_system
    return {"file": path.name, "title": parts[0].title, "document_type": parts[0].document_type, "system": system, "entities": entities, "facts": facts, "unlisted_products": unlisted, "dropped": dict(dropped)}


def roll_up_components(facts: list[dict], entities: dict[str, dict], names: dict[str, str]) -> None:
    """Facts about a component move to the product it is part of, or to the document's main product, keeping the component name."""
    by_name = {e["name"]: e for e in entities.values()}
    products = {name for name, e in by_name.items() if e["kind"] != "component"}
    counts = Counter(f["product"] for f in facts)
    main = max(products, key=lambda name: counts[name], default="")
    for fact in facts:
        entity = by_name.get(fact["product"])
        if entity and entity["kind"] == "component":
            parent = names.get(key(entity["part_of"]), "")
            fact["component"] = fact["product"]
            fact["product"] = parent if parent in products else main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="folder of the parsed .md documents")
    parser.add_argument("out_dir", type=Path, help="folder for the cards")
    parser.add_argument("--llm-model", required=True, help='"<model>@<url>": url "google_api" means Gemini cloud, anything else an OpenAI-compatible endpoint')
    parser.add_argument("--docs", nargs="*", help="only these .md files")
    args = parser.parse_args()

    load_dotenv()
    client, model = build_client(args.llm_model)
    by_system = json.loads(SYSTEMS_FILE.read_text(encoding="utf-8"))
    systems = [s.lower().replace(" ", "-") for s in by_system]
    default_systems = {doc: s.lower().replace(" ", "-") for s, docs in by_system.items() for doc in docs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.docs or sorted(default_systems):
        card = extract_document(client, model, args.kb_dir / name, systems, default_systems[name])
        (args.out_dir / f"{Path(name).stem}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        with_product = sum(1 for f in card["facts"] if f["product"])
        print(f"{name}: {len(card['entities'])} entities, {len(card['facts'])} facts ({with_product} with a product), {len({f['topic'] for f in card['facts']})} topics, {card['unlisted_products']} unlisted products, dropped {card['dropped']}", flush=True)


if __name__ == "__main__":
    main()
