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

## Google Drive & Shared Folder Workflows
- status: active
- id: notebook_skill.gdrive_workflows
<!-- content -->
This repository heavily relies on Google Drive and Google Colab for storage and compute. AI agents modifying or generating code must be aware of the following filesystem quirks and connection protocols.

### 1. Colab `drive.mount` Protocol
- status: active
- id: notebook_skill.gdrive_workflows.mount_protocol
<!-- content -->
- **Mounting Mechanism**: Notebooks interacting with datasets must import `drive` from `google.colab` and mount it:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- **Local vs Cloud Execution**: Be aware that this syntax is exclusively for Google Colab environments. If testing code locally, the user will need Google Drive for Desktop installed, mapping the drive to a local path (e.g., `/Volumes/GoogleDrive/MyDrive`). 
- **Avoid Hardcoding Local Paths**: When writing path strings, parameterize the root storage directory so it can be easily toggled between Colab paths and Local paths.

### 2. Shared Folder Shortcut Resolution
- status: active
- id: notebook_skill.gdrive_workflows.shortcuts
<!-- content -->
Collaborating on Google Drive requires all participants to add a **Shortcut** to the shared folder (e.g., `AI Public Trust`) into their personal `MyDrive`.
- **Target Path**: The universal target path for accessing the shared data within Colab is typically:
  `SHARED_FOLDER_PATH = '/content/drive/MyDrive/AI Public Trust'`
- **Folder Verification**: Before executing data-heavy operations, agents should instruct notebooks to verify the existence of this target path using `os.path.exists(SHARED_FOLDER_PATH)`.
- **Warning on Shortcuts**: If a collaborator names their shortcut differently, the `SHARED_FOLDER_PATH` variable is the only place that needs to be updated. Agents should keep this path abstracted into a top-level constant rather than scattering `/content/drive/...` strings throughout the code.

### 3. Notebook Environment Switch & Path Management
- status: active
- id: notebook_skill.gdrive_workflows.env_switch
<!-- content -->
Because notebooks in this project are executed both in the cloud (Google Colab) and locally (VS Code / Jupyter), an `env_switch_setup` cell pattern must be strictly adhered to at the top of every notebook. 

- **The `RUNNING_LOCALLY` flag**: This boolean controls the flow of pathing and imports. 
- **Guarding Colab Imports**: Never import `google.colab` or call `drive.mount('/content/drive')` unconditionally. It will cause a `ModuleNotFoundError` when run locally. It must be guarded by `if not RUNNING_LOCALLY:`.
- **Project Root Resolution**: Local execution of notebooks often happens from subdirectories (e.g. `notebooks/04_Network_Analysis/`), which breaks Python's ability to find the `src` module folder at the root of the project. A standardized setup block must dynamically append the repository root to `sys.path` *when running locally*.
- **Single Source of Truth**: Define variables like `BASE_PATH`, `networks_folder`, `datasets_folder`, etc. **once** in the `env_switch_setup` cell. Never manually redefine them (e.g., `BASE_PATH = Path('/content/drive...')`) further down in the notebook, as this overwrites the correct local paths established by the environment switch block.

#### Standardized Setup Cell Template

```python
import os
from pathlib import Path
import sys

# --- ENVIRONMENT SWITCH ---
# Set to True if running on local machine with Google Drive Desktop mounted
# Set to False if running in Google Colab cloud
RUNNING_LOCALLY = False

if RUNNING_LOCALLY:
  # --- REPO ROOT ON sys.path (so `from src.*` works locally) ---
    _REPO_ROOT = str(Path(os.getcwd()).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    # Standard macOS path for Google Drive Desktop
    BASE_PATH = Path('/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust')
    
else:
    # Google Colab cloud path
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/My Drive/Colab Projects/AI Public Trust')

# Pre-compute critical paths used across notebooks
twits_folder = BASE_PATH / 'Raw Data/Twits/'
test_folder = BASE_PATH / 'Raw Data/'
datasets_folder = BASE_PATH / 'Data Sets'
cleanedds_folder = BASE_PATH / 'Data Sets/Cleaned Data'
networks_folder = BASE_PATH / 'Data Sets/Networks/'
literature_folder = BASE_PATH / 'Literature/'
topic_models_folder = BASE_PATH / 'Models/Topic Modeling/'
```
