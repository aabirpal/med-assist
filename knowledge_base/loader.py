"""
knowledge_base/loader.py
Reads manifest.json and loads each document's text from its .txt file.

This is the only file that knows where documents live on disk.
Everything else imports DOCUMENTS from here.

TO ADD A NEW DOCUMENT:
  1. Create a new .txt file in knowledge_base/docs/
  2. Add an entry to manifest.json
  3. Delete ./chroma_db (triggers rebuild on next startup)

TO UPDATE AN EXISTING DOCUMENT:
  1. Edit the relevant .txt file in knowledge_base/docs/
  2. Delete ./chroma_db (fingerprint will detect the change automatically)
"""

import json
import os

# Paths
_HERE         = os.path.dirname(os.path.abspath(__file__))
_MANIFEST     = os.path.join(_HERE, "manifest.json")
_DOCS_DIR     = os.path.join(_HERE, "docs")


def load_documents() -> list[dict]:
    """
    Read manifest.json and load text for each entry from its .txt file.

    Returns:
        List of dicts with keys: id, topic, text.
        Everything downstream is unchanged.

    Raises:
        FileNotFoundError: if manifest.json or a referenced .txt file is missing.
        ValueError: if a .txt file exists but is empty.
    """
    with open(_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    documents = []
    missing   = []
    empty     = []

    for entry in manifest:
        txt_path = os.path.join(_DOCS_DIR, entry["file"])

        if not os.path.exists(txt_path):
            missing.append(entry["file"])
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            empty.append(entry["file"])
            continue

        documents.append({
            "id":    entry["id"],
            "topic": entry["topic"],
            "text":  text,
        })

    if missing:
        raise FileNotFoundError(
            f"The following document files listed in manifest.json were not found:\n"
            + "\n".join(f"  {_DOCS_DIR}/{f}" for f in missing)
        )

    if empty:
        raise ValueError(
            f"The following document files are empty — paste content before running:\n"
            + "\n".join(f"  {_DOCS_DIR}/{f}" for f in empty)
        )

    return documents


# Load once on import
DOCUMENTS = load_documents()
