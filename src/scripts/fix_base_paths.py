#!/usr/bin/env python3
"""
Migration script to correct the BASE_PATH strings in the `env_switch_setup` block.
The original setup incorrectly used `/content/drive/MyDrive/AI Public Trust`,
but it should be `/content/drive/My Drive/Colab Projects/AI Public Trust`.

Usage:
    python src/scripts/fix_base_paths.py
"""

import json
import os
from pathlib import Path

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
        # Only target the setup cell
        if cell_id == "env_switch_setup":
            source = cell.get("source", [])
            new_source = []
            cell_modified = False

            for line in source:
                original_line = line
                # Replace the old path strings with the new correct ones
                if "'/Volumes/GoogleDrive/MyDrive/AI Public Trust'" in line:
                    line = line.replace("'/Volumes/GoogleDrive/MyDrive/AI Public Trust'", "'/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust'")
                
                if "'/content/drive/MyDrive/AI Public Trust'" in line:
                    line = line.replace("'/content/drive/MyDrive/AI Public Trust'", "'/content/drive/My Drive/Colab Projects/AI Public Trust'")
                
                if line != original_line:
                    cell_modified = True
                
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
