# Knowledge Base Repository
- status: active
- type: guideline
<!-- content -->

## Production ready app

https://knowledge-base-app-216559257034.us-central1.run.app/

## Overview
This repository serves as the central nervous system for **Prompt Context Management**. It tracks, organizes, and serves Markdown-based knowledge modules (Agents, Plans, Guidelines, etc.) that are used to "inject" context into Large Language Model (LLM) prompts.

## Key Features

### 1. Context Tracking & Storage
- type: documentation
- **Centralized Knowledge**: Stores critical project documentation in `content/`, organized by type (Agents, Guidelines, Protocols).
- **Dependency Resolution**: Implements a strict **Depth-First Dependency Resolution** protocol. If you need `MC_AGENT.md`, the system automatically ensures you also get `AGENTS.md` and `MD_CONVENTIONS.md` in the correct order.
- **Dependency Registry**: Maintains a `dependency_registry.json` that maps every file to its required dependencies, ensuring no context is missing.
<!-- content -->

### 2. Prompt Injection Logic
- type: documentation
- **Automated assembly**: Logic in `src/dependency_manager.py` resolves the dependency graph for any given file and generates a linear reading list.
- **Protocol Compliance**: Ensures that the resulting context block follows the project's strict Markdown-JSON Hybrid Schema.
<!-- content -->

### 3. The Knowledge Base Injector App
A built-in Streamlit application allows users to visually browse and assemble prompts.

🚀 **[Launch Active App](https://knowledgebases-eikasia.streamlit.app/)**
- **Browse**: Filter knowledge by category (Skills, Logs, Plans).
- **Build**: Select the modules you need.
- **Inject**: Download a ZIP bundle (`context_bundle.zip`) containing all required markdown files, ready for upload to your LLM workspace.

## Directory Structure
- type: documentation
- `src/`: Application source code (`app.py`, `dependency_manager.py`, `md_parser.py`, `mcp_server.py`, `index_builder.py`).
- `content/`: The actual knowledge base files.
- `agents/`: AI Agent definitions.
- `plans/`: Project plans and roadmaps.
- `guidelines/`: Additional guidelines and templates.
- `logs/`: Operational logs and artifacts.
- `manager/`: Maintenance and management tools.
- `cleaner/`: Pipeline for ingesting and cleaning external repositories.
- `language/`: Tools for Markdown parsing and schema enforcement.
- `util/`: Shared utilities.
- `bin/`: Executable scripts and binaries.
- `tests/`: Test suite.
- `AGENTS.md`: Core agent protocols and workflow.
- `AGENTS_LOG.md`: Log of agent interventions and changes.
- `HOUSEKEEPING.md`: Routine maintenance protocols.
- `INFRASTRUCTURE.md`: Deployment and cloud resource documentation.
- `MD_CONVENTIONS.md`: The Markdown-JSON Hybrid Schema specification.
- `META_MCP_AGENT.md`: Meta-Agent documentation and MCP server specifications.
- `dependency_registry.json`: The source of truth for file relationships.
<!-- content -->

## Maintenance & Cleaning Protocol
To update the knowledge base with content from external repositories, follow this strict protocol:

1.  **Update Repository List**: Add the target repository URLs to `manager/cleaner/toclean_repolist.txt`.
2.  **Run Pipeline**: Execute the cleaning pipeline:
    ```bash
    python3 manager/cleaner/pipeline.py
    ```
    This script will clone the repositories, migrate Markdown files to the correct schema, and save them to `manager/cleaner/temprepo_cleaning/`.
3.  **Integrate Content**:
    -   Review files in `manager/cleaner/temprepo_cleaning/`.
    -   **Action**: Move *only* new files into the appropriate `content/` subdirectories (`agents/`, `plans/`, `core/`, etc.).
    -   **Constraint**: Do NOT overwrite existing files in `content/` unless explicitly instructed. Current working files must be preserved.

## Cloud Implementation

The application is deployed as a serverless container on **Google Cloud Run**, utilizing **Artifact Registry** for image storage and **Identity-Aware Proxy (IAP)** for security.

### Deployment Methods

#### 1. Manual Deployment (`deploy.sh`)
For local development to production pushes:
```bash
./deploy.sh
```
This script performs a straightforward workflow:
1.  Builds the Docker image locally.
2.  Pushes it to Google Artifact Registry.
3.  Deploys the new revision to Cloud Run.

*Note: This requires a fast internet connection as the full image is pushed from your machine.*

#### 2. Automated Pipeline (`cloudbuild.yaml`)
For CI/CD (triggered via `gcloud builds submit` or GitHub triggers):
-   Uses **Google Cloud Build**.
-   **Caching Strategy**: Pulls the previous `builder` image to skip re-installing Python dependencies (`requirements.txt`) if they haven't changed.
-   **Efficiency**: The heavy final image push happens entirely within Google's internal network, significantly speeding up deployment compared to local pushes.

### Docker Configuration
The `Dockerfile` uses a **multi-stage build** definition to keep the final image approximately ~200MB (uncompressed):
1.  **Builder Stage**: Installs system compilers (`gcc`) and python headers to wheel binary dependencies.
2.  **Final Stage**: Copies only the pre-compiled wheels and source code (`src/`) into a slim `python:3.11-slim` runtime image. Includes `git` for repository synchronization features.

### Infrastructure & Security
For a deep dive into the architecture, network flow, and security layers, refer to [INFRASTRUCTURE.md](INFRASTRUCTURE.md).
-   **Security**: Access is restricted via IAP; only authorized Organization users can access the web interface.
-   **Secrets**: GitHub tokens are managed via **Secret Manager** and injected into the container at runtime.

## Usage
To run the Injector App locally:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Streamlit app

The main command for running a Streamlit app is `streamlit run`. Streamlit also offers several other command-line interface (CLI) tools for configuration, documentation, and example apps. 

`streamlit run your_app.py` starts a local web server and opens your app in a new tab in your default web browser, typically at http://localhost:8501. 
You can also pass a URL to streamlit run if your script is hosted remotely, such as on GitHub: 
streamlit run https://raw.githubusercontent.com 


### Running the Meta MCP Server
To enable dynamic context discovery in your IDE:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements-mcp.txt
    ```
2.  **Build the Index** (First run only):
    ```bash
    python src/index_builder.py .
    ```
3.  **Configure your IDE** (e.g., `claude_desktop_config.json`):
    ```json
    {
      "mcpServers": {
        "meta-knowledge": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/knowledge_bases"
        }
      }
    }
    ```

## Deploying and Cloud Resources

refer to `INFRASTRUCTURE.md`

### Access without Identity-Aware Proxy (IAP), by a proxy

With just this we would see the 403 Forbidden error because Cloud Run private services (which this is, due to your Org Policy) require an Authorization header with a valid identity token in every request. A standard browser visit doesn't send this token automatically, even if you have the right permissions.

To solve this we use the Cloud Run Proxy. This creates a local bridge that handles the authentication for you.

This is the command:

```
gcloud run services proxy knowledge-base-app --region=us-central1 --project=eikasia-ops
```

gcloud run services add-iam-policy-binding

