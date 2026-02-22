# MCP Protocol & Data Tools Skill
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "server": "src/mcp/server.py", "tools": "src/mcp/tools.py"}
<!-- content -->
This document explains the **Model Context Protocol (MCP)** implementation within the MCMP Chatbot and provides guidelines for extending it.

## 1. Architecture Overview
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
The project implements a **Lightweight In-Process MCP Server** rather than a separate subprocess. This reduces latency and deployment complexity for the Streamlit app.

### Core Components
- id: mcp_protocol_data_tools_skill.1_architecture_overview.core_components
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
1.  **Server (`src/mcp/server.py`)**:
    -   Defines the `MCPServer` class.
    -   Host the tool registry.
    -   Exposes `list_tools()` (returns OpenAI/Gemini compatible schemas) and `call_tool()` (executes Python functions).
2.  **Tools (`src/mcp/tools.py`)**:
    -   Contains the actual Python logic for tools.
    -   Loads data from JSON files in `data/` (`raw_events.json`, `people.json`, `research.json`).
3.  **Integration (`src/core/engine.py`)**:
    -   The `RAGEngine` initializes the `MCPServer`.
    -   It passes tools to the LLM (Gemini/OpenAI) during chat generation.
    -   It handles the tool call loop (LLM requests tool -> Engine executes tool -> Engine feeds result back to LLM).

## 2. The "JSON Database" Pattern
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
We use a **JSON-as-Database** pattern exposed via MCP tools. This acts as a bridge between unstructured LLM queries and structured data.

### How it works
- id: mcp_protocol_data_tools_skill.2_the_json_database_pattern.how_it_works
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
1.  **Raw Data**: We scrape the website and store data in `data/*.json` files. This is our "source of truth".
2.  **Tool Abstraction**: We write Python functions (`search_people`, `get_events`) that load these JSONs and perform filtering/searching in memory.
3.  **LLM Interface**: We expose these functions to the LLM with strict schemas.

**Example Flow:**
> User: "Who works on Logic?"
> 1. LLM sees tool `search_people(query: str, role_filter: str)`.
> 2. LLM calls `search_people(query="Logic")`.
> 3. Python tool loads `people.json`, filters by "Logic" in description/interests, and returns a JSON list of matches.
> 4. LLM uses this JSON list to answer the user.

### Strengths
- id: mcp_protocol_data_tools_skill.2_the_json_database_pattern.strengths
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
-   **Simplicity**: No external database (SQL/NoSQL) required.
-   **Flexibility**: JSON schemas can evolve easily.
-   **Deterministic**: Critical data retrieval (dates, contact info) is handled by code, not by the LLM guessing.

### Weaknesses & Improvements
- id: mcp_protocol_data_tools_skill.2_the_json_database_pattern.weaknesses_improvements
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
-   **Performance**: Loading large JSON files (e.g., `raw_events.json`) from disk on *every* tool call is inefficient.
    -   *Improvement*: Cache the loaded data in memory using `@functools.lru_cache` or a singleton `DataManager`.
-   **Scalability**: In-memory searching (linear scan) is slow for datasets >10k items.
    -   *Improvement*: For larger datasets, migrate the backend of the tool to **SQLite** (included in Python) or **DuckDB**. The MCP interface remains the same, but `src/mcp/tools.py` would query a DB file instead of parsing JSON.
-   **Schema Validation**: Currently we manually parse arguments.
    -   *Improvement*: Use **Pydantic** models to define tool inputs/outputs. This enforces types automatically and generates the JSON schema for the LLM.

## 3. Workflow for Adding New Tools
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
To add a new capability:

1.  **Define the Logic**:
    -   Add a Python function in `src/mcp/tools.py`.
    -   Ensure it returns JSON-serializable data (dict or list).
2.  **Register the Tool**:
    -   Import the function in `src/mcp/server.py`.
    -   Add it to `self.tools` dict in `__init__`.
    -   Add its JSON schema definition to `list_tools()`.
3.  **Test**:
    -   Create a unit test in `tests/` to verify the tool handles parameters correctly.
    -   Verify the LLM picks it up (check logs/debug).

## 4. Best Practices
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
-   **Tool Descriptions**: The LLM relies *entirely* on the `description` field in `list_tools` to decide when to use a tool. Be verbose and specific (e.g., "Use this for X, but not for Y").
-   **Parameter Descriptions**: Explain the format (e.g., "YYYY-MM-DD") and examples.
-   **Return Concise Data**: The LLM has a context window. Do not return the entire database. limit results (e.g., `[:10]`) or summarize fields in the tool before returning.

## 5. Prompt Engineering for Tools
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Simply defining a tool is often insufficient; the LLM must be "coached" to use it correctly, especially in a hybrid RAG system.

### A. Dynamic Injection
- id: mcp_protocol_data_tools_skill.5_prompt_engineering_for_tools.a_dynamic_injection
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
Do not rely on the API's implicit tool support alone. **Explicitly inject the list of tools** into the System Prompt.
- **Why**: It improves tool awareness for smaller or less-tuned models.
- **How**: In `src/core/engine.py`, we iterate over `mcp_server.list_tools()` and append a "### AVAILABLE DATA TOOLS" section to the system instruction.

### B. The "Force Usage" Pattern
- id: mcp_protocol_data_tools_skill.5_prompt_engineering_for_tools.b_the_force_usage_pattern
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
LLMs are trained to be polite and helpful, often asking "Would you like me to check the database?". This breaks the seamless RAG experience.
- **Fix**: Use **Imperative Instructions** in the system prompt.
- **Example**: *"IMPORTANT: You have permission to use these tools. Do NOT ask the user if they want you to check. Just check."*

### C. The "Data Enrichment" Pattern (RAG vs Tool Conflict)
- id: mcp_protocol_data_tools_skill.5_prompt_engineering_for_tools.c_the_data_enrichment_pattern_rag_vs_tool_conflict
- status: active
- type: documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->
A common failure mode is "Partial Satisfaction": The LLM finds an event title in the Vector Store (RAG) and thinks it's done, ignoring the Tool that has the full abstract/time.
- **Fix**: Explicitly instruct the LLM to use tools for **Enrichment**.
- **Rule**: *"If the text context provides only partial information (like a title without an abstract), you MUST call the tool to get the full details."*
