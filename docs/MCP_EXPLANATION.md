# Model Context Protocol (MCP) — Explanation
- status: active
- type: guideline
- id: mcp-explanation
- last_checked: 2026-02-04
<!-- content -->
This document provides a comprehensive explanation of the **Model Context Protocol (MCP)** as implemented in this project. It covers the fundamental concepts, architecture, information flow, and how to extend the pattern from JSON data sources to relational databases.

## 1. What is MCP?
- status: active
- type: documentation
- id: mcp-explanation.definition
<!-- content -->
The **Model Context Protocol (MCP)** is an abstraction layer that enables Large Language Models (LLMs) to interact with external data sources and services through a standardized interface. Rather than relying solely on the LLM's training data or raw text retrieval, MCP exposes **tools** — discrete, well-defined functions that the LLM can invoke to perform specific operations.

MCP solves a fundamental problem: LLMs are excellent at understanding language and reasoning, but they cannot directly query databases, call APIs, or perform deterministic computations. MCP bridges this gap by providing a structured way for the LLM to request operations and receive structured results.

### Key Characteristics
- id: mcp-explanation.definition.characteristics
- status: active
- type: documentation
<!-- content -->
| Characteristic | Description |
|:---------------|:------------|
| **Declarative** | Tools are defined by their schema (name, parameters, description), not their implementation |
| **Deterministic** | Tool execution is handled by code, producing reliable, reproducible results |
| **Bidirectional** | The LLM requests tool execution; the system returns structured data |
| **Model-Agnostic** | The same tool definitions work across different LLM providers (OpenAI, Gemini, Claude) |

## 2. Architecture Overview
- status: active
- type: documentation
- id: mcp-explanation.architecture
<!-- content -->
The MCP implementation consists of four core components that work together to enable LLM-tool interaction. Each component has a distinct responsibility, creating a clean separation of concerns.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
│                    "Who works on Logic at MCMP?"                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENGINE                                         │
│  (src/core/engine.py)                                                       │
│                                                                             │
│  • Receives user query                                                      │
│  • Initializes MCP Server                                                   │
│  • Passes tool schemas to LLM                                               │
│  • Orchestrates the tool call loop                                          │
│  • Assembles final context for response generation                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│           MCP SERVER              │   │              LLM                  │
│  (src/mcp/server.py)              │   │     (Gemini/OpenAI/Claude)        │
│                                   │   │                                   │
│  • Hosts the tool registry        │   │  • Receives query + tool schemas  │
│  • Exposes list_tools()           │   │  • Decides which tools to call    │
│  • Exposes call_tool()            │   │  • Generates tool call requests   │
│  • Routes calls to tool functions │   │  • Synthesizes final response     │
└───────────────────┬───────────────┘   └───────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TOOLS                                          │
│  (src/mcp/tools.py)                                                         │
│                                                                             │
│  • Python functions implementing actual logic                               │
│  • Load and query data sources (JSON files, databases)                      │
│  • Return JSON-serializable results                                         │
│                                                                             │
│  Examples: search_people(), get_events(), query_research()                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Component: Engine
- id: mcp-explanation.architecture.engine
- status: active
- type: documentation
<!-- content -->
The **Engine** (`src/core/engine.py`) is the central orchestrator. It coordinates the interaction between the user, the LLM, and the MCP Server. The engine is responsible for:

1. **Initialization**: Creating an instance of the MCP Server when the engine starts.
2. **Schema Injection**: Retrieving tool schemas from `mcp_server.list_tools()` and passing them to the LLM alongside the user's query.
3. **Tool Call Loop**: When the LLM responds with a tool call request, the engine executes it via `mcp_server.call_tool()`, then feeds the result back to the LLM for further processing.
4. **Context Assembly**: Combining tool results with other context sources (e.g., RAG chunks) before generating the final response.

```python
# Simplified engine flow (pseudocode)
class Engine:
    def __init__(self):
        # Initialize the MCP Server (in-process, not subprocess)
        self.mcp_server = MCPServer()
    
    def process_query(self, user_query: str) -> str:
        # Step 1: Get available tools
        tools = self.mcp_server.list_tools()
        
        # Step 2: Send query + tools to LLM
        response = self.llm.generate(
            messages=[{"role": "user", "content": user_query}],
            tools=tools
        )
        
        # Step 3: Handle tool calls (loop until LLM stops requesting tools)
        while response.has_tool_calls():
            for tool_call in response.tool_calls:
                # Execute the tool via MCP Server
                result = self.mcp_server.call_tool(
                    name=tool_call.name,
                    arguments=tool_call.arguments
                )
                # Feed result back to LLM
                response = self.llm.continue_with_tool_result(result)
        
        # Step 4: Return final text response
        return response.text
```

### 2.2 Component: MCP Server
- id: mcp-explanation.architecture.server
- status: active
- type: documentation
<!-- content -->
The **MCP Server** (`src/mcp/server.py`) acts as a registry and dispatcher for tools. It provides two primary methods:

| Method | Purpose | Returns |
|:-------|:--------|:--------|
| `list_tools()` | Returns schemas for all registered tools | List of tool definitions (OpenAI/Gemini compatible) |
| `call_tool(name, arguments)` | Executes a tool by name with given arguments | JSON-serializable result from the tool function |

The server maintains an internal registry mapping tool names to their Python implementations:

```python
class MCPServer:
    def __init__(self):
        # Registry: tool name -> callable function
        self.tools = {
            "search_people": search_people,
            "get_events": get_events,
            "search_research": search_research
        }
    
    def list_tools(self) -> list[dict]:
        """
        Return tool schemas in a format compatible with LLM APIs.
        
        Each schema includes:
        - name: The tool identifier
        - description: What the tool does (critical for LLM decision-making)
        - parameters: JSON Schema defining expected arguments
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_people",
                    "description": "Search for researchers by name, role, or research interests. Use this to find people working on specific topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search term (name, topic, or research area)"
                            },
                            "role_filter": {
                                "type": "string",
                                "enum": ["all", "faculty", "postdoc", "phd"],
                                "description": "Filter by academic role"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            # ... additional tool schemas
        ]
    
    def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Execute a tool by name and return its result.
        
        Args:
            name: The tool identifier
            arguments: Dictionary of parameters to pass to the tool
        
        Returns:
            JSON-serializable result from the tool function
        
        Raises:
            KeyError: If the tool name is not registered
        """
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        
        # Invoke the actual Python function
        return self.tools[name](**arguments)
```

### 2.3 Component: Tools
- id: mcp-explanation.architecture.tools
- status: active
- type: documentation
<!-- content -->
**Tools** (`src/mcp/tools.py`) are the Python functions that implement actual business logic. Each tool:

1. Receives typed parameters from the MCP Server.
2. Interacts with data sources (JSON files, databases, APIs).
3. Returns a JSON-serializable result (dict or list).

```python
# Example tool implementation
import json
from pathlib import Path

# Data path constant
DATA_DIR = Path("data")

def search_people(query: str, role_filter: str = "all") -> list[dict]:
    """
    Search for researchers by name or research interests.
    
    Args:
        query: Search term to match against names, titles, or interests
        role_filter: Filter results by academic role
    
    Returns:
        List of matching researcher profiles (limited to 10 results)
    """
    # Load data from JSON file
    with open(DATA_DIR / "people.json", "r") as f:
        people = json.load(f)
    
    # Filter by query (case-insensitive substring matching)
    query_lower = query.lower()
    matches = [
        person for person in people
        if query_lower in person.get("name", "").lower()
        or query_lower in person.get("interests", "").lower()
        or query_lower in person.get("description", "").lower()
    ]
    
    # Apply role filter
    if role_filter != "all":
        matches = [p for p in matches if p.get("role") == role_filter]
    
    # Limit results to avoid context overflow
    return matches[:10]
```

### 2.4 Component: LLM
- id: mcp-explanation.architecture.llm
- status: active
- type: documentation
<!-- content -->
The **LLM** (Gemini, OpenAI, Claude) is the reasoning component. It receives:

1. The user's natural language query.
2. The list of available tool schemas from `list_tools()`.
3. System instructions guiding when and how to use tools.

Based on this information, the LLM decides whether to:
- Answer directly from its knowledge.
- Request one or more tool calls to gather information.
- Synthesize a response from tool results.

The LLM does not execute tools directly — it generates a structured request specifying which tool to call and with what arguments. The engine then handles execution and feeds results back.

## 3. Information Flow
- status: active
- type: documentation
- id: mcp-explanation.flow
<!-- content -->
Understanding the complete information flow is essential for debugging and extending the system. Here is the step-by-step sequence for a typical query:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: User Query Arrives                                                   │
│ User: "Who works on Logic at MCMP?"                                          │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Engine Prepares LLM Request                                          │
│                                                                              │
│ • Retrieves tool schemas via mcp_server.list_tools()                         │
│ • Constructs messages array with user query                                  │
│ • Includes system prompt with tool usage instructions                        │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: LLM Analyzes Query + Tools                                           │
│                                                                              │
│ LLM sees:                                                                    │
│ - Query: "Who works on Logic at MCMP?"                                       │
│ - Tool: search_people(query, role_filter) - "Search for researchers..."      │
│                                                                              │
│ LLM decides: "I need structured data about people. I'll call search_people." │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Generates Tool Call Request                                      │
│                                                                              │
│ Response: {                                                                  │
│   "tool_calls": [{                                                           │
│     "name": "search_people",                                                 │
│     "arguments": {"query": "Logic", "role_filter": "all"}                    │
│   }]                                                                         │
│ }                                                                            │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Engine Executes Tool via MCP Server                                  │
│                                                                              │
│ engine calls: mcp_server.call_tool("search_people", {"query": "Logic"})      │
│                                                                              │
│ MCP Server:                                                                  │
│ 1. Looks up "search_people" in registry                                      │
│ 2. Calls search_people(query="Logic", role_filter="all")                     │
│ 3. Returns result to engine                                                  │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Tool Executes and Returns Data                                       │
│                                                                              │
│ search_people() function:                                                    │
│ 1. Loads people.json from disk                                               │
│ 2. Filters entries where "Logic" appears in interests/description            │
│ 3. Returns: [                                                                │
│      {"name": "Dr. Alice Smith", "role": "faculty", "interests": "Logic..."},│
│      {"name": "Bob Jones", "role": "phd", "interests": "Modal Logic..."}     │
│    ]                                                                         │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Engine Feeds Result Back to LLM                                      │
│                                                                              │
│ Engine sends new message:                                                    │
│ {                                                                            │
│   "role": "tool",                                                            │
│   "tool_call_id": "...",                                                     │
│   "content": "[{\"name\": \"Dr. Alice Smith\", ...}, ...]"                   │
│ }                                                                            │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: LLM Synthesizes Final Response                                       │
│                                                                              │
│ LLM receives structured data and generates human-readable response:          │
│                                                                              │
│ "Based on the MCMP directory, several researchers work on Logic:             │
│  - Dr. Alice Smith (Faculty) - specializes in Mathematical Logic             │
│  - Bob Jones (PhD Student) - focuses on Modal Logic                          │
│  ..."                                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Observations
- id: mcp-explanation.flow.observations
- status: active
- type: documentation
<!-- content -->
1. **The LLM never directly accesses data.** It can only request tool execution through the engine.
2. **Tool results are JSON strings.** The engine serializes tool output and passes it as a message to the LLM.
3. **Multiple iterations are possible.** If the LLM needs more information, it can request additional tool calls before generating the final response.
4. **The engine controls the loop.** It decides when to stop (typically when the LLM returns text without tool calls).

## 4. Generalization: The RAGEngine
- status: active
- type: documentation
- id: mcp-explanation.rag-engine
<!-- content -->
The basic `Engine` described above handles only MCP tool calls. In practice, many applications require **both** structured data retrieval (via MCP tools) and semantic search over unstructured documents. The `RAGEngine` is a generalization that combines these approaches.

### RAG + MCP Architecture
- id: mcp-explanation.rag-engine.architecture
- status: active
- type: documentation
<!-- content -->
RAG (Retrieval-Augmented Generation) uses vector embeddings to find semantically similar text chunks from a document store. The `RAGEngine` integrates this with MCP tools, routing queries to the appropriate retrieval path.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY ROUTER                                      │
│                                                                             │
│  Classifies query as: structured / unstructured / hybrid                    │
└───────────────┬─────────────────────────────────────────┬───────────────────┘
                │                                         │
                ▼                                         ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│       VECTOR STORE (RAG)          │   │         MCP TOOLS                 │
│       ChromaDB                    │   │    (Structured Queries)           │
│                                   │   │                                   │
│  • Embeddings of documents        │   │  • search_people()                │
│  • Semantic similarity search     │   │  • get_events()                   │
│  • Returns text chunks            │   │  • Returns JSON data              │
└───────────────┬───────────────────┘   └───────────────┬───────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT ASSEMBLY                                     │
│                                                                             │
│  Combines: RAG chunks + Tool results → Unified context for LLM             │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM                                            │
│                    Generates final response                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When to Use Each Path
- id: mcp-explanation.rag-engine.routing
- status: active
- type: documentation
<!-- content -->
| Query Type | Example | Retrieval Path |
|:-----------|:--------|:---------------|
| **Structured** | "Who works on Logic?" | MCP Tools |
| **Unstructured** | "What's our vacation policy?" | Vector Store (RAG) |
| **Hybrid** | "Find researchers who published on topics related to the Gödel paper" | Both |

### RAGEngine Pseudocode
- id: mcp-explanation.rag-engine.code
- status: active
- type: documentation
<!-- content -->
```python
class RAGEngine:
    """
    Generalized engine combining RAG and MCP for hybrid retrieval.
    
    Extends the basic Engine with vector store integration,
    enabling both semantic search and structured tool calls.
    """
    
    def __init__(self):
        # MCP component (same as basic Engine)
        self.mcp_server = MCPServer()
        
        # RAG component (vector store for embeddings)
        self.vector_store = ChromaDB(persist_directory="data/vectordb")
    
    def process_query(self, user_query: str) -> str:
        # Step 1: Classify the query
        query_type = self.router.classify(user_query)
        
        # Step 2: Retrieve context based on query type
        context_parts = []
        
        if query_type in ["unstructured", "hybrid"]:
            # RAG retrieval: semantic search over documents
            chunks = self.vector_store.search(user_query, top_k=5)
            context_parts.append(self._format_chunks(chunks))
        
        if query_type in ["structured", "hybrid"]:
            # MCP retrieval: tool calls for structured data
            tools = self.mcp_server.list_tools()
            tool_results = self._execute_tool_loop(user_query, tools)
            context_parts.append(self._format_tool_results(tool_results))
        
        # Step 3: Assemble unified context
        combined_context = "\n\n".join(context_parts)
        
        # Step 4: Generate response with all context
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": f"Use this context:\n{combined_context}"},
                {"role": "user", "content": user_query}
            ]
        )
        
        return response.text
```

The key insight is that MCP tools and RAG are **complementary**: MCP excels at precise, structured queries (dates, counts, specific records), while RAG excels at semantic understanding (concepts, policies, explanations). The `RAGEngine` leverages both.

## 5. The JSON Database Pattern
- status: active
- type: documentation
- id: mcp-explanation.json-pattern
<!-- content -->
This project uses a **JSON-as-Database** pattern where structured data is stored in JSON files and exposed through MCP tools. This pattern is simple and effective for small to medium datasets.

### How It Works
- id: mcp-explanation.json-pattern.mechanism
- status: active
- type: documentation
<!-- content -->
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│                                                                             │
│   data/                                                                     │
│   ├── people.json         # Researcher profiles                             │
│   ├── raw_events.json     # Events and seminars                             │
│   └── research.json       # Research projects and publications              │
│                                                                             │
│   Each file contains an array of objects with consistent schemas.           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOL LAYER                                        │
│                                                                             │
│   src/mcp/tools.py                                                          │
│                                                                             │
│   def search_people(query, role_filter):                                    │
│       people = load_json("people.json")  # Load from disk                   │
│       return filter(people, query)        # In-memory filtering             │
│                                                                             │
│   def get_events(date_from, date_to):                                       │
│       events = load_json("raw_events.json")                                 │
│       return filter_by_date(events, date_from, date_to)                     │
│                                                                             │
│   Tools abstract the storage format from the LLM.                           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM INTERFACE                                     │
│                                                                             │
│   The LLM sees only:                                                        │
│   - Tool name: "search_people"                                              │
│   - Description: "Search for researchers by name or interests"              │
│   - Parameters: query (string), role_filter (enum)                          │
│                                                                             │
│   The LLM does NOT know:                                                    │
│   - Data is stored in JSON files                                            │
│   - How filtering is implemented                                            │
│   - File paths or internal structure                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example: Complete Tool Implementation
- id: mcp-explanation.json-pattern.example
- status: active
- type: documentation
<!-- content -->
```python
# src/mcp/tools.py

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configuration
DATA_DIR = Path("data")

def load_json(filename: str) -> list[dict]:
    """
    Load a JSON file from the data directory.
    
    Args:
        filename: Name of the JSON file (e.g., "people.json")
    
    Returns:
        Parsed JSON content (typically a list of dictionaries)
    """
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def search_people(query: str, role_filter: str = "all") -> list[dict]:
    """
    Search for researchers by name, role, or research interests.
    
    This tool queries the MCMP people directory and returns matching
    researcher profiles. Use it when users ask about specific people,
    or want to find researchers working on particular topics.
    
    Args:
        query: Search term to match against names, titles, or interests.
               Case-insensitive substring matching is used.
        role_filter: Restrict results to a specific academic role.
                     Options: "all", "faculty", "postdoc", "phd"
    
    Returns:
        List of researcher profiles (max 10). Each profile contains:
        - name: Full name
        - role: Academic position
        - email: Contact email
        - interests: Research interests
        - description: Brief bio
    
    Example:
        search_people("Logic", "faculty") -> Returns faculty working on Logic
    """
    # Load data from JSON file
    people = load_json("people.json")
    
    # Normalize query for case-insensitive search
    query_lower = query.lower()
    
    # Filter by query (check multiple fields)
    matches = [
        person for person in people
        if query_lower in person.get("name", "").lower()
        or query_lower in person.get("interests", "").lower()
        or query_lower in person.get("description", "").lower()
        or query_lower in person.get("title", "").lower()
    ]
    
    # Apply role filter if specified
    if role_filter != "all":
        matches = [p for p in matches if p.get("role") == role_filter]
    
    # Limit results to prevent context overflow
    return matches[:10]


def get_events(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    event_type: str = "all"
) -> list[dict]:
    """
    Retrieve events and seminars from the MCMP calendar.
    
    Use this tool when users ask about upcoming talks, seminars,
    conferences, or want to know what events are scheduled.
    
    Args:
        date_from: Start date filter (ISO format: YYYY-MM-DD).
                   If not provided, defaults to today.
        date_to: End date filter (ISO format: YYYY-MM-DD).
                 If not provided, returns all future events.
        event_type: Filter by event category.
                    Options: "all", "seminar", "conference", "workshop"
    
    Returns:
        List of events (max 20). Each event contains:
        - title: Event name
        - date: Event date (YYYY-MM-DD)
        - time: Start time
        - speaker: Presenter name (if applicable)
        - abstract: Event description
        - location: Venue
    """
    events = load_json("raw_events.json")
    
    # Parse date filters
    if date_from:
        from_date = datetime.strptime(date_from, "%Y-%m-%d")
        events = [e for e in events if datetime.strptime(e["date"], "%Y-%m-%d") >= from_date]
    
    if date_to:
        to_date = datetime.strptime(date_to, "%Y-%m-%d")
        events = [e for e in events if datetime.strptime(e["date"], "%Y-%m-%d") <= to_date]
    
    # Apply event type filter
    if event_type != "all":
        events = [e for e in events if e.get("type") == event_type]
    
    # Sort by date and limit results
    events.sort(key=lambda x: x["date"])
    return events[:20]
```

### Characteristics of the JSON Pattern
- id: mcp-explanation.json-pattern.characteristics
- status: active
- type: documentation
<!-- content -->
| Aspect | Description |
|:-------|:------------|
| **Data Storage** | Plain JSON files in `data/` directory |
| **Query Method** | In-memory filtering with Python list comprehensions |
| **Schema** | Implicit (defined by JSON structure) |
| **Transactions** | None (read-only in typical usage) |
| **Performance** | Loads entire file per query; suitable for <10K records |
| **Concurrency** | File-based; no built-in locking |

## 6. Extending to Relational Databases
- status: active
- type: documentation
- id: mcp-explanation.database
<!-- content -->
The JSON pattern works well for small datasets, but as data grows or query complexity increases, a relational database becomes necessary. The key insight is that **the MCP interface remains unchanged** — only the tool implementation changes.

### The Text-to-SQL Pattern
- id: mcp-explanation.database.text-to-sql
- status: active
- type: documentation
<!-- content -->
When extending to relational databases, there are two architectural approaches:

**Approach A: Fixed Query Tools**
Tools execute predefined SQL queries with parameterized inputs.

**Approach B: Dynamic Text-to-SQL**
The LLM generates SQL queries based on natural language, which are then executed.

This section covers **Approach A** (recommended for most cases) as it provides better security and predictability.

### Architecture with DuckDB
- id: mcp-explanation.database.architecture
- status: active
- type: documentation
<!-- content -->
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│                                                                             │
│   data/nexus.duckdb                                                         │
│   ├── people         (TABLE: id, name, role, email, interests, bio)         │
│   ├── events         (TABLE: id, title, date, time, speaker_id, abstract)   │
│   ├── publications   (TABLE: id, title, authors, year, venue, doi)          │
│   └── topics         (TABLE: id, name, parent_id)                           │
│                                                                             │
│   Relational schema with foreign keys and indexes.                          │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABASE CONNECTION                                  │
│                                                                             │
│   src/core/database.py                                                      │
│                                                                             │
│   class Database:                                                           │
│       def __init__(self):                                                   │
│           self.conn = duckdb.connect("data/nexus.duckdb")                   │
│                                                                             │
│       def execute(self, query: str, params: tuple) -> list[dict]:           │
│           result = self.conn.execute(query, params)                         │
│           return result.fetchdf().to_dict('records')                        │
│                                                                             │
│   Singleton connection with parameterized query execution.                  │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOL LAYER                                        │
│                                                                             │
│   src/mcp/tools.py                                                          │
│                                                                             │
│   def search_people(query, role_filter):                                    │
│       sql = "SELECT * FROM people WHERE ..."                                │
│       return db.execute(sql, (query,))                                      │
│                                                                             │
│   def get_events_with_speakers(date_from, date_to):                         │
│       sql = '''                                                             │
│           SELECT e.*, p.name as speaker_name                                │
│           FROM events e                                                     │
│           JOIN people p ON e.speaker_id = p.id                              │
│           WHERE e.date BETWEEN ? AND ?                                      │
│       '''                                                                   │
│       return db.execute(sql, (date_from, date_to))                          │
│                                                                             │
│   Tools now use SQL queries instead of JSON filtering.                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example: Database-Backed Tool Implementation
- id: mcp-explanation.database.example
- status: active
- type: documentation
<!-- content -->
```python
# src/core/database.py

import duckdb
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class Database:
    """
    Database connection manager for DuckDB.
    
    Provides a singleton connection with parameterized query execution.
    Uses DuckDB for its simplicity (single file, no server required)
    and excellent analytical query performance.
    """
    
    _instance: Optional['Database'] = None
    
    def __new__(cls, db_path: str = "data/nexus.duckdb"):
        """
        Singleton pattern ensures one database connection per process.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(db_path)
        return cls._instance
    
    def _initialize(self, db_path: str):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the DuckDB file
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
    
    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """
        Execute a parameterized SQL query and return results as dictionaries.
        
        Args:
            query: SQL query with ? placeholders for parameters
            params: Tuple of parameter values
        
        Returns:
            List of dictionaries, one per row
        
        Example:
            db.execute("SELECT * FROM people WHERE role = ?", ("faculty",))
        """
        result = self.conn.execute(query, params)
        
        # Convert to list of dictionaries for JSON serialization
        df = result.fetchdf()
        return df.to_dict('records')
    
    def execute_scalar(self, query: str, params: tuple = ()):
        """
        Execute a query that returns a single value.
        
        Args:
            query: SQL query expected to return one row, one column
            params: Tuple of parameter values
        
        Returns:
            The scalar value, or None if no results
        """
        result = self.conn.execute(query, params).fetchone()
        return result[0] if result else None


# Create global database instance
db = Database()
```

```python
# src/mcp/tools.py (database version)

from src.core.database import db
from typing import Optional


def search_people(query: str, role_filter: str = "all") -> list[dict]:
    """
    Search for researchers by name, role, or research interests.
    
    Uses full-text search capabilities of DuckDB for efficient matching
    across multiple columns.
    
    Args:
        query: Search term to match against names, titles, or interests.
        role_filter: Restrict results to a specific academic role.
                     Options: "all", "faculty", "postdoc", "phd"
    
    Returns:
        List of researcher profiles (max 10).
    """
    # Build query with optional role filter
    # Using ILIKE for case-insensitive pattern matching
    search_pattern = f"%{query}%"
    
    if role_filter == "all":
        sql = '''
            SELECT id, name, role, email, interests, bio
            FROM people
            WHERE name ILIKE ?
               OR interests ILIKE ?
               OR bio ILIKE ?
            ORDER BY 
                CASE WHEN name ILIKE ? THEN 0 ELSE 1 END,  -- Prioritize name matches
                name
            LIMIT 10
        '''
        params = (search_pattern, search_pattern, search_pattern, search_pattern)
    else:
        sql = '''
            SELECT id, name, role, email, interests, bio
            FROM people
            WHERE role = ?
              AND (name ILIKE ? OR interests ILIKE ? OR bio ILIKE ?)
            ORDER BY name
            LIMIT 10
        '''
        params = (role_filter, search_pattern, search_pattern, search_pattern)
    
    return db.execute(sql, params)


def get_events(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    event_type: str = "all"
) -> list[dict]:
    """
    Retrieve events with speaker information via JOIN.
    
    Demonstrates the power of relational queries — the JSON version
    would require loading two files and manually joining.
    
    Args:
        date_from: Start date filter (ISO format: YYYY-MM-DD)
        date_to: End date filter (ISO format: YYYY-MM-DD)
        event_type: Filter by event category
    
    Returns:
        List of events with embedded speaker details.
    """
    # Build dynamic WHERE clause
    conditions = []
    params = []
    
    if date_from:
        conditions.append("e.date >= ?")
        params.append(date_from)
    
    if date_to:
        conditions.append("e.date <= ?")
        params.append(date_to)
    
    if event_type != "all":
        conditions.append("e.event_type = ?")
        params.append(event_type)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    sql = f'''
        SELECT 
            e.id,
            e.title,
            e.date,
            e.time,
            e.abstract,
            e.location,
            e.event_type,
            p.name AS speaker_name,
            p.email AS speaker_email,
            p.interests AS speaker_interests
        FROM events e
        LEFT JOIN people p ON e.speaker_id = p.id
        WHERE {where_clause}
        ORDER BY e.date ASC
        LIMIT 20
    '''
    
    return db.execute(sql, tuple(params))


def get_publication_stats(author_id: Optional[int] = None) -> dict:
    """
    Get publication statistics — a query that would be complex with JSON.
    
    Demonstrates aggregation capabilities of SQL.
    
    Args:
        author_id: If provided, stats for specific author. Otherwise, global stats.
    
    Returns:
        Dictionary with publication statistics.
    """
    if author_id:
        sql = '''
            SELECT 
                COUNT(*) as total_publications,
                COUNT(DISTINCT venue) as unique_venues,
                MIN(year) as first_year,
                MAX(year) as latest_year,
                AVG(citation_count) as avg_citations
            FROM publications
            WHERE author_id = ?
        '''
        params = (author_id,)
    else:
        sql = '''
            SELECT 
                COUNT(*) as total_publications,
                COUNT(DISTINCT author_id) as total_authors,
                COUNT(DISTINCT venue) as unique_venues,
                MIN(year) as first_year,
                MAX(year) as latest_year
            FROM publications
        '''
        params = ()
    
    results = db.execute(sql, params)
    return results[0] if results else {}
```

### Comparison: JSON vs Database Implementation
- id: mcp-explanation.database.comparison
- status: active
- type: documentation
<!-- content -->
| Aspect | JSON Pattern | Database Pattern |
|:-------|:-------------|:-----------------|
| **Setup Complexity** | Minimal (just files) | Requires schema definition |
| **Query Performance** | O(n) linear scan | O(log n) with indexes |
| **Joins** | Manual (load multiple files) | Native SQL JOIN |
| **Aggregations** | Python code (sum, count, etc.) | Native SQL (SUM, COUNT, GROUP BY) |
| **Data Integrity** | None (schema-less) | Foreign keys, constraints |
| **Concurrent Access** | File locking issues | ACID transactions |
| **Scalability** | <10K records | Millions of records |
| **Tool Interface** | Same | Same |

### MCP Server: No Changes Required
- id: mcp-explanation.database.server
- status: active
- type: documentation
<!-- content -->
The MCP Server implementation remains identical regardless of backend:

```python
# src/mcp/server.py — unchanged

class MCPServer:
    def __init__(self):
        # Import tools (whether JSON or database-backed)
        from src.mcp.tools import search_people, get_events, get_publication_stats
        
        self.tools = {
            "search_people": search_people,
            "get_events": get_events,
            "get_publication_stats": get_publication_stats
        }
    
    def list_tools(self) -> list[dict]:
        # Same schema definitions — LLM sees no difference
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_people",
                    "description": "Search for researchers by name, role, or research interests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "role_filter": {"type": "string", "enum": ["all", "faculty", "postdoc", "phd"]}
                        },
                        "required": ["query"]
                    }
                }
            },
            # ... other tools
        ]
    
    def call_tool(self, name: str, arguments: dict) -> dict:
        # Same dispatch logic
        return self.tools[name](**arguments)
```

### Migration Path: JSON to Database
- id: mcp-explanation.database.migration
- status: active
- type: documentation
<!-- content -->
To migrate from JSON to a relational database:

1. **Design the schema**: Create tables that match your JSON structure, adding foreign keys where relationships exist.

2. **Write migration script**: Load JSON files and insert into database tables.

3. **Update tool implementations**: Replace `load_json()` calls with SQL queries.

4. **Test**: Verify that tool outputs remain compatible (same structure, types).

5. **Deploy**: No changes needed to MCP Server, Engine, or LLM configuration.

```python
# Example migration script
import json
import duckdb

def migrate_json_to_duckdb():
    """
    Migrate JSON files to DuckDB database.
    """
    conn = duckdb.connect("data/nexus.duckdb")
    
    # Create schema
    conn.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            role VARCHAR,
            email VARCHAR,
            interests TEXT,
            bio TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            title VARCHAR NOT NULL,
            date DATE,
            time VARCHAR,
            speaker_id INTEGER REFERENCES people(id),
            abstract TEXT,
            location VARCHAR,
            event_type VARCHAR
        )
    ''')
    
    # Load and insert people
    with open("data/people.json", "r") as f:
        people = json.load(f)
    
    for i, person in enumerate(people):
        conn.execute(
            "INSERT INTO people VALUES (?, ?, ?, ?, ?, ?)",
            (i, person["name"], person.get("role"), person.get("email"),
             person.get("interests"), person.get("description"))
        )
    
    # Load and insert events (with speaker_id lookup)
    with open("data/raw_events.json", "r") as f:
        events = json.load(f)
    
    for i, event in enumerate(events):
        # Look up speaker_id by name
        speaker_name = event.get("speaker")
        speaker_id = None
        if speaker_name:
            result = conn.execute(
                "SELECT id FROM people WHERE name = ?",
                (speaker_name,)
            ).fetchone()
            speaker_id = result[0] if result else None
        
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, event["title"], event.get("date"), event.get("time"),
             speaker_id, event.get("abstract"), event.get("location"),
             event.get("type"))
        )
    
    # Create indexes for common queries
    conn.execute("CREATE INDEX idx_people_role ON people(role)")
    conn.execute("CREATE INDEX idx_events_date ON events(date)")
    
    conn.close()
    print("Migration complete!")
```

## 7. Summary: Key Takeaways
- status: active
- type: documentation
- id: mcp-explanation.summary
<!-- content -->
| Concept | Description |
|:--------|:------------|
| **MCP** | Protocol enabling LLMs to invoke external tools through a standardized interface |
| **Engine** | Basic orchestrator that manages the LLM-tool interaction loop |
| **RAGEngine** | Generalized engine combining MCP tools with vector-based semantic search |
| **MCP Server** | Registry and dispatcher for tool functions |
| **Tools** | Python functions that perform actual data operations |
| **Tool Schema** | JSON description of tool name, purpose, and parameters |
| **JSON Pattern** | Simple approach using JSON files as a data source |
| **Database Pattern** | Scalable approach using SQL queries for complex data needs |
| **Abstraction** | The LLM sees only the tool interface, not the implementation details |

The power of MCP lies in its **abstraction**: the LLM interacts with tools through a stable interface, while the underlying implementation can evolve from simple JSON files to sophisticated database systems without changing the LLM's experience or the prompt engineering required.
