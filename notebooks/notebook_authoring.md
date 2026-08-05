# Notebook Authoring Guide
- status: active
- type: guideline
- id: notebook_authoring_guide
- last_checked: 2026-04-10
<!-- content -->

This document complements `notebook_setup.md` by explaining **how to correctly create new Jupyter notebooks** in this project. It exists because `.ipynb` files are plain JSON, and hand-writing or templating that JSON is error-prone and has caused parse failures in the past.

---

## The Golden Rule

> **Never write or edit `.ipynb` files as raw text or JSON by hand.**  
> Always use the `create_notebook.py` helper script (see below) or JupyterLab/VS Code's notebook UI.

---

## Why `.ipynb` Files Break

A `.ipynb` file is a JSON document. The following are the most common causes of corruption:

| Mistake | Symptom |
| :--- | :--- |
| Embedding literal newlines inside a JSON string | `Expected property name or '}'` parse error |
| Using `\n` escape sequences incorrectly when building cell source by hand | Notebook opens as raw JSON |
| Mixing escaped and unescaped quotes inside source strings | JSON syntax error at position N |
| Writing the `metadata` block as a string literal instead of a JSON object | Kernel info missing; file may not open |
| Forgetting that `source` must be a **list of strings** (one per line) | Notebook may render incorrectly |

The safe solution is to let **Python's `json` module** serialise the file — it handles all escaping automatically.

---

## The Correct Structure of a `.ipynb` Cell

Every cell must follow this schema:

**Markdown cell:**
```json
{
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "# My heading\n",
    "\n",
    "Some text."
  ]
}
```

**Code cell:**
```json
{
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
    "import os\n",
    "print('hello world')"
  ]
}
```

Key rules:
- `source` is a **list of strings**, one element per line.
- Every line except the **last** must end with `\n`.
- The last line must **not** end with `\n`.
- `outputs` must be an empty list `[]` for new cells.
- `execution_count` must be `null` (not a number) for new cells.

---

## The Helper Script: `create_notebook.py`

Use `notebooks/create_notebook.py` to create new notebooks programmatically. It guarantees valid JSON output and handles all line-splitting automatically.

### Usage

```bash
# From the repo root
python3 notebooks/create_notebook.py
```

Edit the script to define your cells, then run it. A validated `.ipynb` file will be written to the target path.

### How the script works

```python
import json

def make_source(text: str) -> list[str]:
    """Split a multiline string into a properly formatted source list."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": make_source(text)}

def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": make_source(text)}

def write_notebook(path: str, cells: list) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    # Validate round-trip
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)
    print(f"✅ Written and validated: {path}")
```

Then define your cells as plain Python strings:

```python
cells = [
    md("# My Notebook\nA short description."),
    code("""import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
"""),
]

write_notebook("notebooks/05_Classifiers/my_notebook.ipynb", cells)
```

---

## Checklist Before Creating a New Notebook

- [ ] Will I use `create_notebook.py` or the VS Code / JupyterLab UI? If neither, stop.
- [ ] Does the notebook follow the 5-cell setup structure defined in `notebook_setup.md`?
- [ ] Does Cell 1 include `RUNNING_LOCALLY`, `BASE_PATH`, and all required path variables?
- [ ] Are pip installs guarded by `if not RUNNING_LOCALLY`?
- [ ] Are all imports explicit (no wildcard `import *` from external libraries)?
- [ ] Are sections written as `#` (not `##`), each given a stable cell id, and listed in
      `metadata.colab.collapsed_sections` + `jp-MarkdownHeadingCollapsed`? See
      `notebook_setup.md` § Collapsible Sections.
- [ ] After creation, does `python3 -c "import json; json.load(open('my_notebook.ipynb'))"` succeed without error?

---

## Validating an Existing Notebook

To check whether an existing notebook has valid JSON:

```bash
python3 -c "import json; json.load(open('notebooks/05_Classifiers/my_notebook.ipynb'))"
```

If this command exits silently, the file is valid. Any error message indicates corruption.

To pretty-print and inspect the raw structure:

```bash
python3 -m json.tool notebooks/05_Classifiers/my_notebook.ipynb | head -60
```
