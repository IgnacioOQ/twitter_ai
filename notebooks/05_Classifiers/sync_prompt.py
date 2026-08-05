#!/usr/bin/env python3
"""Paste llm_bootstrap_prompt.md into notebook 01's `c-prompt` cell.

`llm_bootstrap_prompt.md` is the single authored copy of the prompt. The notebook cannot
read it at runtime — `01_llm_bootstrap_labelling.ipynb` mounts Drive but does not clone
this repo on Colab — so the file has to be embedded in the notebook as a literal. This
script does that embedding, and is the only thing that should:

    python3 notebooks/05_Classifiers/sync_prompt.py            # write
    python3 notebooks/05_Classifiers/sync_prompt.py --check    # verify only, exit 1 on drift

`--check` is the form to put in CI or a pre-commit hook. Without one of these, "re-paste
the criteria into the notebook" is a manual step performed by a human under time pressure,
which is how the notebook and `categories.md` came to hold two copies of the same text in
the first place.

Notebook edits preserve the file's own JSON serialization (indent=1, ensure_ascii=True, no
trailing newline) so the diff is confined to the one cell — see NOTEBOOK_WRITING_SKILL.md
§8. Colab-flavored ids stay mirrored at `metadata.id` (§1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_MD = HERE / "llm_bootstrap_prompt.md"
NOTEBOOK = HERE / "01_llm_bootstrap_labelling.ipynb"
CELL_ID = "c-prompt"

INDENT, ENSURE_ASCII = 1, True

# Sentinels bounding the generated region, so the surrounding cell (imports, helpers,
# assertions) can be edited by hand without this script clobbering it.
BEGIN = "# ── BEGIN GENERATED — llm_bootstrap_prompt.md ──────────────────────────"
END = "# ── END GENERATED ─────────────────────────────────────────────────────"


def literal_block(doc: str) -> str:
    """Render the markdown as a Python literal that reproduces it byte for byte.

    Delimited with ''' because the prompt itself contains \"\"\" around the tweet. The
    backslash after the opening quotes swallows the newline that would otherwise be
    prepended, so PROMPT_DOC == the file on disk with no strip() to fudge it.
    """
    if "'''" in doc:
        sys.exit("ERROR: the markdown contains ''' which would terminate the literal.")
    if "\\" in doc:
        sys.exit("ERROR: the markdown contains a backslash; it would be interpreted as an "
                 "escape inside the literal. Remove it or switch this script to a raw "
                 "string (r'''...'''), which then forbids a trailing backslash instead.")
    if not doc.endswith("\n"):
        sys.exit("ERROR: the markdown must end with exactly one trailing newline.")
    return f"PROMPT_DOC: str = '''\\\n{doc}'''"


def render_cell(doc: str) -> str:
    """The full source of the c-prompt cell."""
    fp = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
    return f"""%%time
# The prompt lives in llm_bootstrap_prompt.md, beside this notebook in the repo. It is
# embedded here verbatim because Colab does not clone the repo at runtime, and it is
# written back out unchanged in the Save section, so the copy that lands next to the
# labels is byte-identical to the repo's.
#
# DO NOT EDIT THE LITERAL BY HAND. Edit llm_bootstrap_prompt.md, then run:
#     python3 notebooks/05_Classifiers/sync_prompt.py
# Hand edits here are silently reverted by the next sync, and `sync_prompt.py --check`
# fails while they are present.
{BEGIN}
{literal_block(doc)}
PROMPT_DOC_SHA256 = '{fp}'   # of llm_bootstrap_prompt.md at sync time
{END}

# Everything the model receives is inside the single ```` fence; the prose around it is
# for humans and never leaves this notebook. Splitting on the fence rather than on a
# line number means reordering the document cannot silently change the prompt.
_parts = PROMPT_DOC.split('````')
assert len(_parts) == 3, (
    f'expected exactly one four-backtick fence in llm_bootstrap_prompt.md, '
    f'found {{max(len(_parts) - 1, 0) // 2}}')
PROMPT_TEMPLATE: str = _parts[1].split('\\n', 1)[1].rstrip('\\n')   # drop the 'text' info string

# The tweet is the only variable part of the prompt, and it goes last so that everything
# before it is byte-identical on every call and stays eligible for the implicit cache.
assert PROMPT_TEMPLATE.count('{{{{TWEET}}}}') == 1, (
    'llm_bootstrap_prompt.md must contain exactly one {{{{TWEET}}}} placeholder inside the fence')

# CATEGORIES drives the response-schema enum in code; the fence states the label set in
# prose. Neither is derived from the other, so they are checked against each other here —
# a category added to one and not the other is the drift that would otherwise show up as
# the model confidently using a label the schema rejects.
assert ', '.join(CATEGORIES) in PROMPT_TEMPLATE, (
    f'the prompt does not list the label set as CATEGORIES has it: {{", ".join(CATEGORIES)!r}}')


def build_prompt(tweet_text: str) -> str:
    return PROMPT_TEMPLATE.replace('{{{{TWEET}}}}', tweet_text)


print(f'prompt: {{len(PROMPT_TEMPLATE):,}} chars '
      f'(~{{len(PROMPT_TEMPLATE) // 4:,}} tokens, re-sent on every call), '
      f'sha256:{{PROMPT_DOC_SHA256}}')
"""


def make_source(text: str) -> list[str]:
    """Multi-line string -> the list-of-strings .ipynb requires."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the notebook matches the markdown; do not write")
    args = ap.parse_args()

    doc = PROMPT_MD.read_text(encoding="utf-8")
    wanted = render_cell(doc)

    raw = NOTEBOOK.read_text(encoding="utf-8")
    nb = json.loads(raw)
    if json.dumps(nb, indent=INDENT, ensure_ascii=ENSURE_ASCII) != raw:
        sys.exit(f"ERROR: {NOTEBOOK.name} is not serialized as indent={INDENT}, "
                 f"ensure_ascii={ENSURE_ASCII}; refusing to rewrite it wholesale.")

    cell = next((c for c in nb["cells"] if c.get("id") == CELL_ID), None)
    if cell is None:
        sys.exit(f"ERROR: no cell with id {CELL_ID!r} in {NOTEBOOK.name}")

    # Compare in the notebook's own representation. make_source drops the trailing
    # newline the rendered cell ends with, so comparing raw strings never matches and
    # --check would report drift forever.
    wanted_source = make_source(wanted)
    if cell["source"] == wanted_source:
        print(f"in sync — {PROMPT_MD.name} matches {NOTEBOOK.name} cell {CELL_ID!r}")
        return 0

    if args.check:
        print(f"DRIFT: {NOTEBOOK.name} cell {CELL_ID!r} does not match {PROMPT_MD.name}.\n"
              f"       Run: python3 {Path(__file__).name}", file=sys.stderr)
        return 1

    cell["source"] = wanted_source
    cell.setdefault("metadata", {})["id"] = CELL_ID

    out = json.dumps(nb, indent=INDENT, ensure_ascii=ENSURE_ASCII)
    NOTEBOOK.write_text(out, encoding="utf-8")
    json.loads(NOTEBOOK.read_text(encoding="utf-8"))   # round-trip check

    fp = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
    print(f"synced {PROMPT_MD.name} ({len(doc):,} chars, sha256:{fp}) "
          f"-> {NOTEBOOK.name} cell {CELL_ID!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
