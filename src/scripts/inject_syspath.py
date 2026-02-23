#!/usr/bin/env python3
"""
One-shot migration script: injects sys.path manipulation into every notebook's
env_switch_setup cell so that `from src.*` imports resolve when running locally.

Usage:
    python src/scripts/inject_syspath.py
"""

import json
import os
from pathlib import Path

# Lines to inject (as notebook source strings, each ending with \n)
INJECT_LINES = [
    "import sys\n",
    "\n",
    "# --- REPO ROOT ON sys.path (so `from src.*` works locally) ---\n",
    "_REPO_ROOT = str(Path(os.getcwd()).resolve().parents[1])\n",
    "if _REPO_ROOT not in sys.path:\n",
    "    sys.path.insert(0, _REPO_ROOT)\n",
    "\n",
]

MARKER = "sys.path.insert"  # used to detect if already injected


def find_notebooks(root: Path):
    """Yield every .ipynb under root/notebooks/."""
    notebooks_dir = root / "notebooks"
    for dirpath, _, filenames in os.walk(notebooks_dir):
        for fname in sorted(filenames):
            if fname.endswith(".ipynb"):
                yield Path(dirpath) / fname


def inject_into_notebook(nb_path: Path) -> bool:
    """
    Open a notebook, find the env_switch_setup cell, inject the sys.path
    lines right after `from pathlib import Path\n`.
    Returns True if the file was modified.
    """
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        cell_id = cell.get("metadata", {}).get("id", "")
        if cell_id != "env_switch_setup":
            continue

        source = cell.get("source", [])

        # Already injected?
        if any(MARKER in line for line in source):
            print(f"  SKIP (already has sys.path): {nb_path.name}")
            return False

        # Find the anchor line: `from pathlib import Path\n`
        anchor_idx = None
        for i, line in enumerate(source):
            if line.strip().startswith("from pathlib import Path"):
                anchor_idx = i
                break

        if anchor_idx is None:
            print(f"  WARN: no 'from pathlib import Path' found in {nb_path.name}")
            return False

        # Insert right after the anchor
        insert_at = anchor_idx + 1
        for j, new_line in enumerate(INJECT_LINES):
            source.insert(insert_at + j, new_line)

        cell["source"] = source

        # Write back
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  OK: {nb_path.name}")
        return True

    print(f"  WARN: no env_switch_setup cell in {nb_path.name}")
    return False


def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Repo root: {repo_root}")
    print()

    modified = 0
    skipped = 0
    for nb_path in find_notebooks(repo_root):
        result = inject_into_notebook(nb_path)
        if result:
            modified += 1
        else:
            skipped += 1

    print()
    print(f"Done. Modified: {modified}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
