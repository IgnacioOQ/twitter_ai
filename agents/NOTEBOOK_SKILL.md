# Notebook Management Skill
- status: active
- type: agent_skill
- id: notebook_skill
- label: [jupyter, skill, agent_capability]
<!-- content -->
This document outlines the protocols, methodologies, and best practices conferring the ability to effectively read, edit, manage, and debug Jupyter Notebooks (`.ipynb` files) within this repository. 

As an AI agent, modifying Jupyter notebooks requires special care due to their underlying JSON document structure, which couples code, markdown, and execution output together.

## Core Rules for Notebook Manipulation
- status: active
- id: notebook_skill.core_rules
<!-- content -->

### 1. Structural Integrity (JSON Saftey)
- status: active
- id: notebook_skill.core_rules.json_safety
<!-- content -->
Jupyter notebooks are strictly formatted JSON arrays. Direct string replacement and patching can easily corrupt the file by misaligning brackets, commas, or execution counts.
- **Prefer Whole-Cell Replacements**: When modifying code within an `.ipynb` file natively, aim to overwrite the full `source` array of a cell rather than making fragile substring edits inside the JSON text.
- **Avoid Output Churn**: Do not manually modify or forge execution outputs (`outputs` array) unless explicitly instructed. Allow the notebook runtime to regenerate outputs.
- **Validate Syntax**: Always ensure that any edit respects standard JSON serialization constraints.

### 2. Execution & State Awareness
- status: active
- id: notebook_skill.core_rules.state_awareness
<!-- content -->
The linear file syntax of a notebook does not always reflect the temporal execution state of the kernel.
- **Assume Linear Dataflow**: When analyzing a notebook, trace the dataflow strictly from the top cell downwards. Assume this is the intended execution order.
- **Headless Execution**: Use headless CLI tools like `jupyter nbconvert --execute` or `papermill` when you are requested to run the notebook and generate outputs, rather than manually simulating kernel commands.

## Architecture and Refactoring Strategy
- status: active
- id: notebook_skill.architecture
<!-- content -->

### Keep Notebooks Lightweight
- status: active
- id: notebook_skill.architecture.lightweight
<!-- content -->
Long, complex notebooks with thousands of lines are difficult for LLMs to patch reliably via JSON edits and are hostile to Git version control.
- **Extract Logic to Modules**: Proactively recommend and execute the extraction of large, reusable Python functions and classes out of the notebook and into sibling Python modules (`.py` files).
- **Import Driven Design**: Import these abstracted helper files at the top of the notebook context. This keeps the notebook focused on high-level data orchestration, visualization, and storytelling.

### Cell Organization
- status: active
- id: notebook_skill.architecture.cell_org
<!-- content -->
- **Markdown Documentation**: Use markdown cells generously to document what the subsequent code blocks are doing. This serves as mental breadcrumbs for both the user and the agent.
- **Idempotency**: Encourage notebook cells to be idempotent. Cells should handle scenarios where they are run multiple times (e.g., checking if a directory exists before creating it, or if a variable is already instantiated).

## Advanced Tooling: Jupytext integration
- status: active
- id: notebook_skill.tooling.jupytext
<!-- content -->
If heavy refactoring of an existing notebook is required, it is highly recommended to bridge the `ipynb` gap by utilizing tools like `jupytext`.
1. Convert the JSON-based notebook to a Python percent-script (`.py` with `# %%` cell markers).
2. Perform extensive multi-line code modifications, linting, and refactoring on the Python script.
3. Sync the `.py` improvements back onto the `.ipynb` file.
