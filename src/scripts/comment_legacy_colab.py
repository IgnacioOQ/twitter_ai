#!/usr/bin/env python3
"""
One-shot migration script: comments out legacy `from google.colab import drive`
and `drive.mount('/content/drive')` in any notebook cell EXCEPT the main
`env_switch_setup` cell.

Usage:
    python src/scripts/comment_legacy_colab.py
"""

import json
import os
from pathlib import Path

def find_notebooks(root: Path):
    """Yield every .ipynb under root/notebooks/."""
    notebooks_dir = root / "notebooks"
    for dirpath, _, filenames in os.walk(notebooks_dir):
        for fname in sorted(filenames):
            if fname.endswith(".ipynb"):
                yield Path(dirpath) / fname

def process_notebook(nb_path: Path) -> bool:
    """
    Open a notebook, find cells that are NOT env_switch_setup, and comment out
    lines containing `google.colab` or `drive.mount`.
    Returns True if the file was modified.
    """
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
            # We want to comment out `from google.colab import drive` and `drive.mount(...)`
            # and any related forced remount hints without touching `BASE =` definitions.
            if ("import google.colab" in line or 
                "from google.colab" in line or 
                "drive.mount" in line):
                
                # Check if it's already commented out
                if not line.lstrip().startswith("#"):
                    new_line = "# " + line
                    new_source.append(new_line)
                    cell_modified = True
                    continue
                    
            new_source.append(line)

        if cell_modified:
            cell["source"] = new_source
            modified = True

    if modified:
        # Write back
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  MODIFIED: {nb_path.name}")
        return True
    
    # print(f"  SKIPPED: {nb_path.name}")
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
