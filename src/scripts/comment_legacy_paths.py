#!/usr/bin/env python3
"""
One-shot migration script: comments out legacy path redefinitions outside of env_switch_setup.
Specifically targets lines defining `BASE = `, `twits_folder = `, `test_folder = `,
`datasets_folder = `, `cleanedds_folder = `, `networks_folder = `, `block_folder = `,
and `literature_folder = ` to prevent them from overwriting the correct `BASE_PATH` logic.

Usage:
    python src/scripts/comment_legacy_paths.py
"""

import json
import os
from pathlib import Path

TARGET_PREFIXES = (
    "BASE = ",
    "twits_folder = ",
    "test_folder = ",
    "datasets_folder = ",
    "cleanedds_folder = ",
    "networks_folder = ",
    "block_folder = ",
    "literature_folder = ",
    "topic_models_folder = ",
    "print(\"Current Directory:\""
)

def find_notebooks(root: Path):
    notebooks_dir = root / "notebooks"
    for dirpath, _, filenames in os.walk(notebooks_dir):
        for fname in sorted(filenames):
            if fname.endswith(".ipynb"):
                yield Path(dirpath) / fname

def process_notebook(nb_path: Path) -> bool:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified = False

    for cell in nb.get("cells", []):
        cell_id = cell.get("metadata", {}).get("id", "")
        # Skip the properly guarded setup cell
        if cell_id == "env_switch_setup":
            continue

        source = cell.get("source", [])
        new_source = []
        cell_modified = False

        for line in source:
            stripped = line.lstrip()
            
            # If the line starts with any of our targets and is NOT already a comment
            if stripped.startswith(TARGET_PREFIXES) and not stripped.startswith("#"):
                new_line = "# " + line
                new_source.append(new_line)
                cell_modified = True
            else:
                new_source.append(line)

        if cell_modified:
            cell["source"] = new_source
            modified = True

    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  MODIFIED: {nb_path.name}")
        return True
    
    return False

def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Repo root: {repo_root}")
    print()

    modified_count = 0
    skipped_count = 0
    for nb_path in find_notebooks(repo_root):
        if process_notebook(nb_path):
            modified_count += 1
        else:
            skipped_count += 1

    print()
    print(f"Done. Modified: {modified_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
