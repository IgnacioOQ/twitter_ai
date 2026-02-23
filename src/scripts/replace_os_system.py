#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

def find_notebooks(root: Path):
    notebooks_dir = root / "notebooks"
    for dirpath, _, filenames in os.walk(notebooks_dir):
        for fname in sorted(filenames):
            if fname.endswith(".ipynb"):
                yield Path(dirpath) / fname

def process_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        source = cell.get("source", [])
        new_source = []
        cell_modified = False

        for line in source:
            original_line = line
            
            # Direct string replacements matching grep results
            if "os.system('git clone https://github.com/IgnacioOQ/twitter-ai')" in line:
                line = line.replace("os.system('git clone https://github.com/IgnacioOQ/twitter-ai')", "!git clone https://github.com/IgnacioOQ/twitter_ai.git")
                cell_modified = True
            elif "os.system('git clone https://github.com/IgnacioOQ/twitter_ai')" in line:
                line = line.replace("os.system('git clone https://github.com/IgnacioOQ/twitter_ai')", "!git clone https://github.com/IgnacioOQ/twitter_ai.git")
                cell_modified = True
                
            # Replace other pip installs
            elif "os.system('pip " in line:
                line = re.sub(r"os\.system\('pip (.*?)'\)", r"!pip \1", line)
                cell_modified = True
                
            elif "os.system('git clone " in line:
                line = re.sub(r"os\.system\('git clone (.*?)'\)", r"!git clone \1", line)
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
    for nb_path in find_notebooks(repo_root):
        if process_notebook(nb_path):
            modified_count += 1

    print(f"Done. Modified: {modified_count}")

if __name__ == "__main__":
    main()
