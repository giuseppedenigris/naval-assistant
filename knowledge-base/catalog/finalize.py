"""Final catalog: the cards with their canonical brands and the manual corrections applied.

Brands come from brands.json (manufacturer as printed -> canonical brands). Corrections come from overrides.json
(md file -> operations, each with its reason); every value they add must be backed by the document text, unless the
operation states why it cannot be ("unchecked", e.g. a model name garbled by OCR and confirmed by the file name).
A document whose text repeats another one is marked with "duplicate_of".
Writes catalog.json and changes.md, the log of every correction.
"""

import argparse
import json
from pathlib import Path

import build


def product(card: dict, op: dict) -> dict:
    """The product an operation refers to, by name, and by category or by one of its models when the name is repeated ("where_category", "where_model")."""
    matches = [p for p in card["products"] if p["name"] == op["product"] and op.get("where_category", p["category"]) == p["category"] and ("where_model" not in op or op["where_model"] in [m["value"] for m in p["models"]])]
    if len(matches) != 1:
        raise ValueError(f"product {op['product']!r} not found exactly once")
    return matches[0]


def checked(item: dict, document: str, op: dict) -> dict:
    """The item itself, if its evidence or value is in the document or the operation says why it cannot be ("unchecked")."""
    if not (op.get("unchecked") or build.grounded(item["evidence"], document) or build.grounded(item["value"], document)):
        raise ValueError(f"{item['value']!r}: neither evidence nor value in the document")
    return item


def apply(card: dict, op: dict, document: str) -> None:
    """One correction: set a card field, set a product's category, rename, add or remove a product or a model, replace a product's models."""
    kind = op["op"]
    if kind == "set":
        card[op["field"]] = checked(op["value"], document, op) if isinstance(op["value"], dict) else op["value"]
    elif kind == "set_category":
        if op["value"] not in build.CATEGORIES:
            raise ValueError(f"unknown category {op['value']!r}")
        product(card, op)["category"] = op["value"]
    elif kind == "rename_product":
        product(card, op)["name"] = op["value"]
    elif kind == "remove_product":
        card["products"].remove(product(card, op))
    elif kind == "add_product":
        card["products"].append({**op["value"], "models": [checked(m, document, op) for m in op["value"]["models"]]})
    elif kind == "remove_model":
        models = product(card, op)["models"]
        models.remove(next(m for m in models if m["value"] == op["value"]))
    elif kind == "add_model":
        product(card, op)["models"].append(checked(op["value"], document, op))
    elif kind == "set_models":
        product(card, op)["models"] = [checked(m, document, op) for m in op["value"]]
    else:
        raise ValueError(f"unknown operation {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="folder of the parsed documents")
    parser.add_argument("data_dir", type=Path, help="folder with cards/, brands.json and overrides.json")
    args = parser.parse_args()

    brands = json.loads((args.data_dir / "brands.json").read_text(encoding="utf-8"))
    overrides = json.loads((args.data_dir / "overrides.json").read_text(encoding="utf-8"))
    metas = {}
    for path in (args.kb_dir / "metadata").glob("*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        metas[meta["md_file"]] = meta

    catalog, log = [], []
    for path in sorted((args.data_dir / "cards").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        card = entry["card"]
        ops = overrides.get(entry["md_file"], [])
        explicit_brands, duplicate_of = None, None
        if ops:
            document = build.compact(build.full_text(args.kb_dir, metas[entry["md_file"]]))
            for op in ops:
                if op["op"] == "set_brands":
                    explicit_brands = op["value"]
                elif op["op"] == "duplicate_of":
                    duplicate_of = op["value"]
                else:
                    apply(card, op, document)
                unchecked = f" | unchecked: {op['unchecked']}" if op.get("unchecked") else ""
                log.append(f"{entry['md_file']}: {op['op']} {op.get('product', op.get('field', ''))} -> {json.dumps(op.get('value'), ensure_ascii=False)} | {op['reason']}{unchecked}")
        manufacturer = card["manufacturer"]["value"] if card["manufacturer"] else None
        entry_brands = explicit_brands if explicit_brands is not None else brands[manufacturer] if manufacturer else []
        catalog.append({"doc_id": entry["doc_id"], "md_file": entry["md_file"], "source_file": entry["source_file"], "llm_model": entry["llm_model"], "brands": entry_brands, "duplicate_of": duplicate_of, "card": card, "corrections": len(ops)})

    md_files = {e["md_file"] for e in catalog}
    missing = [e["duplicate_of"] for e in catalog if e["duplicate_of"] and e["duplicate_of"] not in md_files]
    if missing:
        raise ValueError(f"duplicate_of points to unknown documents: {missing}")

    (args.data_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.data_dir / "changes.md").write_text("\n".join(log), encoding="utf-8")
    print(f"catalog: {len(catalog)} documents, {len(log)} corrections in {sum(bool(e['corrections']) for e in catalog)} documents")


if __name__ == "__main__":
    main()
