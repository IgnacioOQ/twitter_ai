# MCP Protocol & Data Tools Skill - Extended
- status: active
- type: agent_skill
- id: mcp_protocol_extended
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

This document provides a comprehensive explanation of the **Model Context Protocol (MCP)** implementation within the MCMP Chatbot, including the fundamental architecture, information flow, and how LLMs interact with tools at multiple levels of abstraction.

## 1. Understanding MCP: The Three-Layer Architecture
- status: active
- type: documentation
- id: mcp_protocol_extended.three_layer_architecture
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

MCP is fundamentally a **protocol for structured communication** between three distinct layers. Understanding these layers is essential to grasping how LLMs "use tools".

### Layer 1: The LLM (Neural Network)
- status: active
- type: documentation
- id: mcp_protocol_extended.three_layer_architecture.layer_1_llm
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**What it does**: Generates text tokens probabilistically based on input context.

**What it CANNOT do**:
- Execute code
- Make HTTP requests
- Access databases
- Perform actual function calls

**What it CAN do**:
- Recognize patterns in its training data
- Output structured text in specific formats (XML, JSON)
- "Signal" that it wants to use a tool by generating specific text patterns

**Key Insight**: The LLM is trained (via fine-tuning or in-context learning) to output special text patterns when it determines a tool should be used. These patterns are just strings—the LLM has no concept of "calling" anything.

**Example LLM Output**:
```xml
<function_call>
  <tool_name>search_people</tool_name>
  <parameters>
    <query>Logic</query>
    <role_filter>faculty</role_filter>
  </parameters>
</function_call>
```

This is pure text. The LLM generated these tokens the same way it generates essay paragraphs—by predicting the next most likely token.

### Layer 2: The MCP Client (Parser & Orchestrator)
- status: active
- type: documentation
- id: mcp_protocol_extended.three_layer_architecture.layer_2_client
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**What it does**: Acts as the "interpreter" between the LLM's text output and actual executable code.

**Responsibilities**:
1. **Parsing**: Scans the LLM's text output for tool call patterns (XML tags, JSON blocks, etc.)
2. **Validation**: Checks if the requested tool exists and if parameters match the expected schema
3. **Routing**: Forwards validated tool calls to the MCP Server
4. **Loop Management**: Coordinates the multi-turn conversation:
   - User message → LLM → Tool call detected → Execute tool → Inject result → LLM continues
5. **Error Handling**: Manages cases where the LLM outputs invalid tool calls

**In the MCMP Chatbot**: The `RAGEngine` class in `src/core/engine.py` serves as the MCP Client.

**Key Functions**:
```python
# Pseudo-code representation
class MCPClient:
    def parse_llm_output(self, text: str) -> Optional[ToolCall]:
        """
        Scans the text for patterns like:
        <function_call>...</function_call>
        or
        {"tool_use": "...", "arguments": {...}}
        
        Returns a structured ToolCall object or None.
        """
        pass
    
    def execute_conversation_loop(self, user_message: str) -> str:
        """
        Manages the full conversation cycle:
        1. Send message + tools to LLM
        2. Parse output for tool calls
        3. Execute tools via MCP Server
        4. Inject results and continue
        5. Return final answer
        """
        pass
```

### Layer 3: The MCP Server (Tool Provider)
- status: active
- type: documentation
- id: mcp_protocol_extended.three_layer_architecture.layer_3_server
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**What it does**: Hosts and executes the actual tool implementations.

**Responsibilities**:
1. **Tool Registry**: Maintains a catalog of available tools with their schemas
2. **Schema Generation**: Provides JSON schemas that describe each tool's signature (name, parameters, return type)
3. **Execution**: Runs the actual Python functions when the client requests a tool call
4. **Data Access**: Loads data from JSON files, databases, APIs, or any other source
5. **Result Serialization**: Converts Python objects to JSON-serializable formats for the LLM

**In the MCMP Chatbot**: The `MCPServer` class in `src/mcp/server.py` serves as the MCP Server.

**Key Functions**:
```python
# Actual implementation from src/mcp/server.py
class MCPServer:
    def __init__(self):
        # Registry of available tools
        self.tools = {
            "search_people": search_people,
            "get_events": get_events,
            "search_research": search_research
        }
    
    def list_tools(self) -> List[Dict]:
        """
        Returns a list of tool schemas that can be injected
        into the LLM's context or sent via API.
        
        Example output:
        [
            {
                "name": "search_people",
                "description": "Search for people by name, role, or research interests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                        "role_filter": {"type": "string", "enum": ["faculty", "postdoc", "phd"]}
                    },
                    "required": ["query"]
                }
            }
        ]
        """
        pass
    
    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Executes the requested tool with the provided arguments.
        
        Returns the result as a JSON-serializable object.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Execute the actual Python function
        return self.tools[tool_name](**arguments)
```

### Architectural Diagram
- status: active
- type: documentation
- id: mcp_protocol_extended.three_layer_architecture.diagram
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER / APPLICATION                          │
│                  (Streamlit Interface)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ user_message = "Who works on Logic?"
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MCP CLIENT (RAGEngine)                        │
│                                                                 │
│  1. Inject tools into system prompt                            │
│  2. Call LLM API with tools + user message                     │
│  3. Receive LLM text output                                    │
│  4. Parse output for tool call patterns                        │
│  5. Validate tool call                                         │
│  6. Execute tool via MCP Server                                │
│  7. Inject result back into conversation                       │
│  8. Call LLM again for final answer                            │
└────────────────────────────┬───────────────────────────────────┘
                             │
                  ┌──────────┼───────────┐
                  │          │           │
                  ▼          ▼           ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  LLM (Gemini)   │  │  MCP SERVER     │
    │                 │  │                 │
    │ Input:          │  │ Tool Registry:  │
    │ - System prompt │  │ - search_people │
    │ - Tools list    │  │ - get_events    │
    │ - Conversation  │  │ - search_research│
    │                 │  │                 │
    │ Output:         │  │ Executes:       │
    │ Text with       │  │ Python functions│
    │ <tool_call>     │  │ Returns:        │
    │ patterns        │  │ JSON data       │
    └─────────────────┘  └────────┬────────┘
                                  │
                                  │ Loads data from
                                  ▼
                         ┌─────────────────┐
                         │  DATA SOURCES   │
                         │                 │
                         │ - people.json   │
                         │ - events.json   │
                         │ - research.json │
                         └─────────────────┘
```

## 2. Information Flow: A Complete Trace
- status: active
- type: documentation
- id: mcp_protocol_extended.information_flow
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Let's trace a complete request through the system to understand how information flows at each layer.

### User Query: "Who works on Logic?"
- status: active
- type: documentation
- id: mcp_protocol_extended.information_flow.example_trace
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Step 1: Client Preparation (RAGEngine)**
```python
# In src/core/engine.py
def chat(self, user_message: str) -> str:
    # Get tool schemas from MCP Server
    tools = self.mcp_server.list_tools()
    # tools = [
    #     {"name": "search_people", "description": "...", "parameters": {...}},
    #     {"name": "get_events", "description": "...", "parameters": {...}}
    # ]
    
    # Build system prompt with tools injected
    system_prompt = self._build_system_prompt_with_tools(tools)
    # system_prompt = """
    # You are an assistant for MCMP.
    # You have access to the following tools:
    # 
    # 1. search_people(query: str, role_filter: Optional[str])
    #    Description: Search for people by name, role, or research interests.
    #    When to use: User asks about faculty, researchers, or specific people.
    # 
    # 2. get_events(date_filter: Optional[str])
    #    Description: Retrieve upcoming or past events.
    #    When to use: User asks about talks, conferences, or schedule.
    # 
    # IMPORTANT: You have permission to use these tools. Do NOT ask the user
    # if they want you to check. Just use the tool directly.
    # """
```

**Step 2: LLM Invocation (Client → LLM)**
```python
    # Call the LLM API (Gemini, OpenAI, etc.)
    response = self.llm.generate_content(
        prompt=user_message,  # "Who works on Logic?"
        system_instruction=system_prompt,
        tools=tools  # Some APIs accept tools as a parameter
    )
```

**Step 3: LLM Processing (Neural Network Layer)**

The LLM receives this context:
```
System: You have access to search_people(query, role_filter)...
User: Who works on Logic?
```

The neural network:
1. Recognizes the pattern: "Who works on" → Person search
2. Has been trained to output structured tool calls
3. Generates tokens probabilistically:
   - Next token: `<` (starts XML tag)
   - Next token: `function` 
   - Next token: `_call>`
   - ... and so on

**LLM Output (Pure Text)**:
```xml
<function_call>
  <tool_name>search_people</tool_name>
  <parameters>
    <query>Logic</query>
    <role_filter>faculty</role_filter>
  </parameters>
</function_call>

I'll search for people working on Logic.
```

**Step 4: Client Parsing (RAGEngine)**
```python
    # Parse the LLM's text output
    raw_text = response.text
    # raw_text = "<function_call>...</function_call>\n\nI'll search for people..."
    
    tool_call = self._parse_tool_call(raw_text)
    # tool_call = {
    #     "name": "search_people",
    #     "arguments": {"query": "Logic", "role_filter": "faculty"}
    # }
    
    if tool_call:
        # Validate the tool exists
        if tool_call["name"] not in self.mcp_server.tools:
            return "Error: Invalid tool requested"
```

**Step 5: Server Execution (MCPServer → Tool Function)**
```python
        # Execute the tool via MCP Server
        result = self.mcp_server.call_tool(
            tool_name=tool_call["name"],     # "search_people"
            arguments=tool_call["arguments"]  # {"query": "Logic", "role_filter": "faculty"}
        )
```

**Inside the MCP Server**:
```python
# In src/mcp/server.py
def call_tool(self, tool_name: str, arguments: Dict) -> Any:
    # Look up the function in the registry
    tool_function = self.tools[tool_name]  # tool_function = search_people
    
    # Execute it (actual Python function call)
    return tool_function(**arguments)  # search_people(query="Logic", role_filter="faculty")
```

**Inside the Tool Function**:
```python
# In src/mcp/tools.py
def search_people(query: str, role_filter: Optional[str] = None) -> List[Dict]:
    # Load the JSON data
    with open("data/people.json", "r") as f:
        people_data = json.load(f)
    
    # Filter by query
    results = []
    for person in people_data:
        if query.lower() in person["name"].lower() or \
           query.lower() in person.get("research_interests", "").lower():
            if role_filter is None or person["role"] == role_filter:
                results.append(person)
    
    return results
    # Returns: [
    #     {"name": "Prof. X", "role": "faculty", "research_interests": "Modal Logic, ..."},
    #     {"name": "Prof. Y", "role": "faculty", "research_interests": "Mathematical Logic, ..."}
    # ]
```

**Step 6: Result Injection (Client)**
```python
        # result = [{"name": "Prof. X", ...}, {"name": "Prof. Y", ...}]
        
        # Inject the result back into the conversation
        tool_result_message = f"""
<tool_result>
<tool_name>search_people</tool_name>
<result>
{json.dumps(result, indent=2)}
</result>
</tool_result>
"""
        
        # Call LLM again with the tool result
        final_response = self.llm.generate_content(
            prompt=user_message,
            system_instruction=system_prompt,
            history=[
                {"role": "assistant", "content": raw_text},  # Original tool call
                {"role": "user", "content": tool_result_message}  # Tool result
            ]
        )
```

**Step 7: LLM Final Answer**

The LLM now sees:
```
System: You have access to tools...
User: Who works on Logic?
Assistant: <function_call>search_people...</function_call>
User: <tool_result>[{"name": "Prof. X", ...}, ...]</tool_result>
```

The LLM generates:
```
Based on the search, the following faculty members work on Logic:

1. **Prof. X** - Specializes in Modal Logic and Epistemic Logic. 
   Contact: x@mcmp.de

2. **Prof. Y** - Focuses on Mathematical Logic and Set Theory.
   Contact: y@mcmp.de

Would you like more information about their specific research projects?
```

**Step 8: Return to User**
```python
        return final_response.text
```

### Key Observations from the Trace
- status: active
- type: documentation
- id: mcp_protocol_extended.information_flow.observations
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **The LLM never executes anything**: It only generates text patterns that signal intent.
2. **The Client is the "brain"**: It orchestrates the entire flow, making actual function calls.
3. **The Server is stateless**: It just executes functions and returns results. No memory between calls.
4. **Tools are injected on every request**: The LLM doesn't "remember" tools. They must be in the context every time.
5. **Multi-turn loop**: Tool use often requires 2+ LLM calls: (1) to decide to use a tool, (2) to generate the final answer with the result.

## 3. How the LLM "Recognizes" When to Use Tools
- status: active
- type: documentation
- id: mcp_protocol_extended.tool_recognition
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

This is a common point of confusion: "How does the LLM know when to use a tool?"

### The Answer: Pattern Matching in High-Dimensional Space
- status: active
- type: documentation
- id: mcp_protocol_extended.tool_recognition.pattern_matching
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

The LLM doesn't "reason" about tools in a symbolic way. Instead:

1. **Training**: During fine-tuning, the model is shown many examples like:
   ```
   System: You have access to search_people(query, role)...
   User: Who works on X?
   Assistant: <function_call><tool_name>search_people</tool_name>...
   ```
   
   The model learns the statistical correlation: 
   - Query pattern "Who works on X" + Available tool "search_people" → Output tool call pattern

2. **Context Window**: On inference, the tool schemas are injected into the context:
   ```
   System: Available tools:
   1. search_people(query, role_filter)
      Description: Search for people by name, role, or research interests.
      When to use: User asks about faculty, researchers, or who works on a topic.
   ```

3. **Attention Mechanism**: The transformer's attention heads identify the relationship:
   - User query tokens: ["Who", "works", "on", "Logic"]
   - Tool description tokens: ["search", "people", "who", "works", "on", "a", "topic"]
   - High attention scores → Model assigns high probability to generating `<function_call>` tokens

4. **Next Token Prediction**: The model predicts the next most likely token given the context. If the context strongly suggests "this is a tool use scenario," it generates the tool call pattern.

### Why Explicit Tool Injection Helps
- status: active
- type: documentation
- id: mcp_protocol_extended.tool_recognition.why_injection_helps
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Some LLM APIs (like OpenAI's) support implicit tool injection—you just pass a `tools` parameter and the API handles it internally. However, **explicit injection into the system prompt is often superior** for several reasons:

1. **Model Awareness**: Some models (especially smaller or open-source ones) aren't fine-tuned for implicit tool use. Explicit descriptions in natural language increase tool use accuracy.

2. **Control Over Triggering**: You can use imperative language to override the LLM's default "polite" behavior:
   ```
   CRITICAL: Do NOT ask the user "Would you like me to search?". 
   ALWAYS use the tool directly when relevant.
   ```

3. **Debugging**: If the LLM isn't using a tool, you can read the system prompt and identify ambiguous descriptions.

4. **Hybrid RAG Systems**: In systems like MCMP Chatbot where you have both vector search (RAG) AND tools, explicit instructions help the LLM decide when to use which:
   ```
   - Use RAG (vector search) for: General questions, conceptual explanations
   - Use search_people tool for: Specific person lookup, contact info, research interests
   - Use BOTH when: RAG provides a partial answer (e.g., mentions a name), 
     then use search_people to enrich with full details
   ```

### Example: The "Data Enrichment" Pattern
- status: active
- type: documentation
- id: mcp_protocol_extended.tool_recognition.data_enrichment
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

A common failure case in hybrid systems:

**User**: "Tell me about the upcoming Logic seminar."

**RAG retrieves**: 
```
Document: "Prof. X is giving a talk on Modal Logic next Tuesday."
```

**LLM response** (without proper prompting):
```
Prof. X is giving a talk on Modal Logic next Tuesday.
```

**Problem**: The LLM is satisfied with the RAG result and doesn't use the `get_events` tool to retrieve the full abstract, time, and location.

**Solution**: Add to system prompt:
```
ENRICHMENT RULE:
If the retrieved context provides ONLY partial information (like a title or mention 
without an abstract, time, or full details), you MUST call the relevant tool to 
get complete information.

Examples of partial information:
- Event title without time/location
- Person name without research interests/contact info
- Paper title without abstract/authors
```

Now the LLM recognizes the pattern:
- RAG result: ✓ Has title
- RAG result: ✗ Missing time, location, abstract
- Pattern match: "Partial information" → Use tool

**LLM response** (with enrichment prompt):
```
<function_call>
<tool_name>get_events</tool_name>
<parameters><date_filter>next Tuesday</date_filter></parameters>
</function_call>

Prof. X is giving a talk on Modal Logic next Tuesday at 4 PM in Room 101.
Abstract: "In this talk, I will explore..."
```

## 4. The "JSON Database" Pattern
- status: active
- type: documentation
- id: mcp_protocol_extended.json_database_pattern
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

The MCMP Chatbot uses a **JSON-as-Database** pattern for tool data sources. This is a deliberate architectural choice with specific trade-offs.

### Architecture
- status: active
- type: documentation
- id: mcp_protocol_extended.json_database_pattern.architecture
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

```
┌─────────────────────────────────────────┐
│     Web Scraping / Data Pipeline        │
│  (Scrapes MCMP website, PDFs, etc.)    │
└─────────────────┬───────────────────────┘
                  │ Outputs
                  ▼
┌─────────────────────────────────────────┐
│        JSON Files (data/*.json)         │
│                                         │
│  - people.json (faculty, PhDs, etc.)   │
│  - events.json (talks, conferences)     │
│  - research.json (papers, projects)     │
│                                         │
│  Structure: List of dictionaries        │
│  [                                      │
│    {"name": "...", "role": "...", ...}, │
│    {"name": "...", "role": "...", ...}  │
│  ]                                      │
└─────────────────┬───────────────────────┘
                  │ Loaded by
                  ▼
┌─────────────────────────────────────────┐
│     MCP Tools (src/mcp/tools.py)        │
│                                         │
│  def search_people(query, role_filter): │
│      data = json.load("people.json")    │
│      # Filter logic                     │
│      return matches                     │
└─────────────────┬───────────────────────┘
                  │ Exposed via
                  ▼
┌─────────────────────────────────────────┐
│     MCP Server (src/mcp/server.py)      │
│                                         │
│  tools = {                              │
│      "search_people": search_people,    │
│      "get_events": get_events           │
│  }                                      │
└─────────────────────────────────────────┘
```

### Why JSON Instead of a Real Database?
- status: active
- type: documentation
- id: mcp_protocol_extended.json_database_pattern.why_json
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Advantages**:
1. **Zero Dependencies**: No need to run PostgreSQL, MongoDB, or any database server.
2. **Simple Deployment**: JSON files can be committed to Git and deployed with the app.
3. **Human-Readable**: Easy to inspect, debug, and manually edit.
4. **Version Control**: Changes to data are tracked via Git diffs.
5. **Fast for Small Datasets**: For datasets with <10,000 items, in-memory filtering is fast enough.

**Disadvantages**:
1. **I/O Overhead**: Loading the entire JSON file on every tool call is inefficient.
2. **No Indexing**: Linear search (O(n)) for every query.
3. **No Relational Queries**: Can't easily express "Find all events by people who work on Logic."
4. **Scalability Ceiling**: At 100k+ items, in-memory filtering becomes slow.

### When to Migrate to a Real Database
- status: active
- type: documentation
- id: mcp_protocol_extended.json_database_pattern.when_to_migrate
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Migrate when:
- Dataset grows beyond 10,000 items
- Query latency exceeds 500ms
- You need relational queries (joins across entities)
- You need full-text search with ranking

**Recommended Path**:
1. **First upgrade**: SQLite (embedded, file-based, zero dependencies)
   - Replace `json.load()` with SQL queries
   - Keep the MCP interface identical (tool signatures unchanged)
   - Gains: Indexing, query optimization, relational queries

2. **Second upgrade** (if needed): PostgreSQL or DuckDB
   - For multi-user write access, complex analytics, or >1M items
   - Still maintain the same MCP tool interface

**Key Principle**: The MCP abstraction layer means you can change the backend without touching the LLM integration. The tool signature `search_people(query, role_filter) -> List[Dict]` remains the same whether it's loading JSON, querying SQLite, or calling a REST API.

## 5. Performance Optimization Strategies
- status: active
- type: documentation
- id: mcp_protocol_extended.performance_optimization
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

### Current Bottleneck: Repeated File I/O
- status: active
- type: documentation
- id: mcp_protocol_extended.performance_optimization.bottleneck
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: Every tool call executes:
```python
def search_people(query: str, role_filter: Optional[str] = None):
    with open("data/people.json", "r") as f:  # ← Disk I/O every call
        people_data = json.load(f)
    # Filter and return
```

For a single user session with 5 tool calls, this loads the same file 5 times from disk.

### Optimization 1: In-Memory Caching
- status: active
- type: documentation
- id: mcp_protocol_extended.performance_optimization.caching
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Solution**: Use `functools.lru_cache` to cache loaded data.

```python
# In src/mcp/tools.py
import json
from functools import lru_cache
from typing import List, Dict, Optional

@lru_cache(maxsize=1)
def _load_people_data() -> List[Dict]:
    """
    Loads people.json and caches it in memory.
    The cache is cleared when the process restarts.
    """
    with open("data/people.json", "r") as f:
        return json.load(f)

def search_people(query: str, role_filter: Optional[str] = None) -> List[Dict]:
    """
    Search for people by name, role, or research interests.
    
    Args:
        query: Search term (name, research interest keyword)
        role_filter: Optional filter by role ("faculty", "postdoc", "phd")
    
    Returns:
        List of matching person dictionaries
    """
    # Load from cache (only hits disk once)
    people_data = _load_people_data()
    
    # Filter logic
    results = []
    for person in people_data:
        # Check if query matches name or interests
        if query.lower() in person["name"].lower() or \
           query.lower() in person.get("research_interests", "").lower():
            # Apply role filter if specified
            if role_filter is None or person.get("role") == role_filter:
                results.append(person)
    
    return results
```

**Impact**: 
- First call: Loads from disk (~10ms)
- Subsequent calls: Loads from memory (~0.1ms)
- 100x speedup for repeated queries

### Optimization 2: Lazy Loading with Singleton Pattern
- status: active
- type: documentation
- id: mcp_protocol_extended.performance_optimization.singleton
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

For more control, use a singleton `DataManager`:

```python
# In src/mcp/tools.py
class DataManager:
    """
    Singleton that loads and caches all JSON data sources.
    Provides lazy loading: data is only loaded when first accessed.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._people = None
            cls._instance._events = None
            cls._instance._research = None
        return cls._instance
    
    @property
    def people(self) -> List[Dict]:
        """Lazy-loads people data on first access."""
        if self._people is None:
            with open("data/people.json", "r") as f:
                self._people = json.load(f)
        return self._people
    
    @property
    def events(self) -> List[Dict]:
        """Lazy-loads events data on first access."""
        if self._events is None:
            with open("data/events.json", "r") as f:
                self._events = json.load(f)
        return self._events
    
    def reload(self):
        """Force reload all data (useful for development/testing)."""
        self._people = None
        self._events = None
        self._research = None

# Global instance
data_manager = DataManager()

def search_people(query: str, role_filter: Optional[str] = None) -> List[Dict]:
    """Search for people using the cached data manager."""
    people_data = data_manager.people  # Loads once, caches forever
    
    results = []
    for person in people_data:
        if query.lower() in person["name"].lower() or \
           query.lower() in person.get("research_interests", "").lower():
            if role_filter is None or person.get("role") == role_filter:
                results.append(person)
    
    return results
```

**Advantages**:
- Explicit control over caching lifecycle
- Can manually reload data without restarting the app
- Easy to add cache invalidation logic (e.g., reload every 24 hours)

### Optimization 3: Result Truncation
- status: active
- type: documentation
- id: mcp_protocol_extended.performance_optimization.truncation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: Returning all 50 matching people to the LLM wastes tokens and may exceed context limits.

**Solution**: Truncate results in the tool before returning:

```python
def search_people(query: str, role_filter: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Search for people with a configurable result limit.
    
    Args:
        query: Search term
        role_filter: Optional role filter
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        List of matching person dictionaries (up to `limit` results)
    """
    people_data = data_manager.people
    
    results = []
    for person in people_data:
        if query.lower() in person["name"].lower() or \
           query.lower() in person.get("research_interests", "").lower():
            if role_filter is None or person.get("role") == role_filter:
                results.append(person)
                # Early termination once we have enough results
                if len(results) >= limit:
                    break
    
    return results
```

**Impact**: 
- Reduces token usage in LLM context
- Faster JSON serialization
- Forces the tool to return the "best" matches first (if you add relevance ranking)

## 6. Workflow for Adding New Tools
- status: active
- type: guideline
- id: mcp_protocol_extended.adding_new_tools
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Follow this checklist when adding a new tool:

### Step 1: Define the Tool Logic
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.define_logic
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Create the Python function in `src/mcp/tools.py`:

```python
def get_publications(author_name: Optional[str] = None, 
                     year_filter: Optional[int] = None,
                     limit: int = 20) -> List[Dict]:
    """
    Retrieve publications from the MCMP research database.
    
    Args:
        author_name: Filter by author name (partial match)
        year_filter: Filter by publication year
        limit: Maximum number of results (default: 20)
    
    Returns:
        List of publication dictionaries with keys:
        - title: str
        - authors: List[str]
        - year: int
        - abstract: str
        - url: str
    """
    # Load data (using cached loader)
    publications = data_manager.publications
    
    # Apply filters
    results = []
    for pub in publications:
        # Filter by author
        if author_name and not any(author_name.lower() in author.lower() 
                                   for author in pub["authors"]):
            continue
        
        # Filter by year
        if year_filter and pub["year"] != year_filter:
            continue
        
        results.append(pub)
        
        # Respect limit
        if len(results) >= limit:
            break
    
    return results
```

### Step 2: Register the Tool in the MCP Server
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.register
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Update `src/mcp/server.py`:

```python
# In MCPServer.__init__()
from src.mcp.tools import search_people, get_events, get_publications  # Import the new tool

class MCPServer:
    def __init__(self):
        self.tools = {
            "search_people": search_people,
            "get_events": get_events,
            "get_publications": get_publications  # Register here
        }
```

### Step 3: Define the Tool Schema
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.schema
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Add the JSON schema to `list_tools()` in `src/mcp/server.py`:

```python
def list_tools(self) -> List[Dict]:
    return [
        # ... existing tools ...
        {
            "name": "get_publications",
            "description": """
Retrieve publications from the MCMP research database.

**When to use**:
- User asks about papers, publications, or research output
- User asks "What has X published?"
- User asks for publications from a specific year

**When NOT to use**:
- User asks about general research interests (use search_people instead)
- User asks about ongoing projects (use search_research instead)
            """.strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    "author_name": {
                        "type": "string",
                        "description": "Filter by author name (partial match, case-insensitive). Example: 'Smith'"
                    },
                    "year_filter": {
                        "type": "integer",
                        "description": "Filter by publication year. Example: 2023"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []  # All parameters are optional
            }
        }
    ]
```

**Schema Best Practices**:
- **Description**: Be verbose. The LLM reads this to decide when to use the tool.
- **"When to use" vs "When NOT to use"**: Prevents tool misuse and overlap.
- **Parameter descriptions**: Include examples and format requirements (e.g., "YYYY-MM-DD", "case-insensitive").
- **Defaults**: Specify default values so the LLM can call with minimal arguments.

### Step 4: Write Tests
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.tests
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Create `tests/test_publications_tool.py`:

```python
import pytest
from src.mcp.tools import get_publications

def test_get_publications_no_filter():
    """Test that get_publications returns results without filters."""
    results = get_publications()
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]
    assert "authors" in results[0]

def test_get_publications_by_author():
    """Test filtering by author name."""
    results = get_publications(author_name="Smith")
    for pub in results:
        assert any("smith" in author.lower() for author in pub["authors"])

def test_get_publications_by_year():
    """Test filtering by year."""
    results = get_publications(year_filter=2023)
    for pub in results:
        assert pub["year"] == 2023

def test_get_publications_limit():
    """Test that limit parameter works."""
    results = get_publications(limit=5)
    assert len(results) <= 5
```

Run tests:
```bash
pytest tests/test_publications_tool.py -v
```

### Step 5: Update Documentation
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.documentation
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Add an entry to this document (`MCP_AGENT.md`) under a "Available Tools" section:

```markdown
### get_publications
- type: agent_skill
- **Purpose**: Retrieve academic publications from the MCMP database
- **Parameters**: 
- `author_name` (optional): Filter by author
- `year_filter` (optional): Filter by year
- `limit` (optional): Max results (default: 20)
- **Use Cases**: "What has Prof. X published?", "Show me papers from 2023"
- **Data Source**: `data/publications.json`
- label: ['agent']
<!-- content -->
```

### Step 6: Test with the LLM
- status: active
- type: task
- id: mcp_protocol_extended.adding_new_tools.llm_test
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Start the Streamlit app and test queries that should trigger the new tool:

```
User: "What has Professor Smith published?"
Expected: LLM calls get_publications(author_name="Smith")

User: "Show me papers from 2023."
Expected: LLM calls get_publications(year_filter=2023)

User: "What are Prof. Smith's research interests?"
Expected: LLM calls search_people(query="Smith"), NOT get_publications
```

**Debugging**: If the LLM doesn't use the tool:
1. Check the logs to see what tools were injected in the system prompt
2. Verify the tool description is clear about "When to use"
3. Add explicit guidance in the system prompt: "For publication queries, ALWAYS use get_publications tool"

## 7. Advanced Patterns: Prompt Engineering for Tools
- status: active
- type: guideline
- id: mcp_protocol_extended.prompt_engineering
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Simply defining a tool is insufficient. LLMs require explicit prompting strategies to use tools correctly, especially in hybrid RAG systems.

### Pattern 1: Dynamic Tool Injection
- status: active
- type: guideline
- id: mcp_protocol_extended.prompt_engineering.dynamic_injection
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: Implicit tool support (passing `tools` to API) works for GPT-4, but smaller models often miss tools.

**Solution**: Inject tool descriptions into the system prompt as natural language.

```python
# In src/core/engine.py
def _build_system_prompt_with_tools(self, tools: List[Dict]) -> str:
    """
    Constructs a system prompt with explicit tool descriptions.
    """
    base_prompt = """
You are an intelligent assistant for the Munich Center for Mathematical Philosophy (MCMP).
You have access to specialized data retrieval tools.
"""
    
    # Dynamically generate tool descriptions
    tools_section = "\n\n### AVAILABLE DATA TOOLS\n\n"
    for tool in tools:
        tools_section += f"**{tool['name']}**\n"
        tools_section += f"{tool['description']}\n\n"
        tools_section += "Parameters:\n"
        for param_name, param_info in tool['parameters']['properties'].items():
            required = "REQUIRED" if param_name in tool['parameters'].get('required', []) else "optional"
            tools_section += f"  - {param_name} ({required}): {param_info['description']}\n"
        tools_section += "\n"
    
    return base_prompt + tools_section
```

**Why it works**: 
- LLM sees tools in natural language, increasing awareness
- Works for models without native tool support
- Allows custom formatting and emphasis

### Pattern 2: The "Force Usage" Pattern
- status: active
- type: guideline
- id: mcp_protocol_extended.prompt_engineering.force_usage
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: LLMs are trained to be polite. They often ask permission before using tools:
```
"Would you like me to search the database for events?"
```

This breaks the seamless experience. Users expect the assistant to just retrieve the information.

**Solution**: Use imperative instructions in the system prompt:

```python
tools_usage_rules = """

### CRITICAL TOOL USAGE RULES

1. **PERMISSION**: You have explicit permission to use these tools. 
   DO NOT ask the user "Would you like me to check?" or "Should I search?".
   ALWAYS use the tool directly when relevant.

2. **AUTONOMY**: If the user's query requires data from a tool, retrieve it immediately.
   Do NOT explain what you're about to do. Just do it and present the results.

3. **TRANSPARENCY**: After using a tool, you MAY briefly mention what you found
   (e.g., "I found 3 matching events"), but keep it concise.

4. **ERROR HANDLING**: If a tool returns no results, inform the user and suggest
   alternative queries or broader search terms.
"""

system_prompt = base_prompt + tools_section + tools_usage_rules
```

**Impact**: 
- Reduces unnecessary back-and-forth
- Creates a more confident, proactive assistant
- Aligns with user expectations (they assume the assistant has database access)

### Pattern 3: The "Data Enrichment" Pattern
- status: active
- type: guideline
- id: mcp_protocol_extended.prompt_engineering.data_enrichment
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: In hybrid RAG systems, the LLM might find partial information in the vector store and stop, ignoring tools that could provide complete data.

**Example**:
```
User: "Tell me about the Logic seminar next week."
RAG retrieves: "Prof. Smith is giving a talk on Modal Logic."
LLM response: "Prof. Smith is giving a talk on Modal Logic next week."
```

**Missing**: Time, location, abstract (all available via `get_events` tool).

**Solution**: Add an enrichment rule to the system prompt:

```python
enrichment_rule = """

### DATA ENRICHMENT RULE

If retrieved context (from RAG) provides ONLY partial information, you MUST call
the relevant tool to obtain complete details.

**Examples of PARTIAL information that require enrichment**:
- Event title WITHOUT time, location, or abstract → Use get_events
- Person name WITHOUT research interests, contact info, or role → Use search_people
- Paper title WITHOUT authors, year, or abstract → Use get_publications

**How to recognize partial information**:
- Ask yourself: "Does this answer provide specific, actionable details?"
- If the answer is "It mentions X but doesn't tell me HOW/WHEN/WHERE/WHO", 
  then it's partial and requires enrichment.

**Workflow**:
1. RAG retrieves context
2. Check: Is the information complete?
3. If NO, call the tool to enrich
4. Synthesize RAG context + tool data into a comprehensive answer
"""

system_prompt = base_prompt + tools_section + tools_usage_rules + enrichment_rule
```

**Implementation Example**:

```python
# Inside the RAGEngine's chat method
def chat(self, user_message: str) -> str:
    # Step 1: RAG retrieval
    rag_context = self.vector_store.retrieve(user_message)
    # rag_context = "Prof. Smith is giving a talk on Modal Logic."
    
    # Step 2: Build prompt with RAG + tools
    system_prompt = self._build_system_prompt_with_tools(tools)
    user_prompt = f"""
Context from knowledge base:
{rag_context}

User question: {user_message}

Remember the DATA ENRICHMENT RULE: If the context is partial, use tools to complete it.
"""
    
    # Step 3: LLM call (may use tools)
    response = self.llm.generate_content(
        prompt=user_prompt,
        system_instruction=system_prompt
    )
    
    # Step 4: Handle tool calls
    if self._detect_tool_call(response.text):
        # ... execute tool, inject result, call LLM again ...
```

**Result**:
```
User: "Tell me about the Logic seminar next week."
LLM: <tool_call>get_events(date_filter="next week")</tool_call>
Tool: [{"title": "Modal Logic Seminar", "speaker": "Prof. Smith", 
        "time": "Tuesday 4 PM", "location": "Room 101", 
        "abstract": "We explore..."}]
LLM: "Prof. Smith is giving a talk on Modal Logic next Tuesday at 4 PM in Room 101. 
      In this talk, he will explore..."
```

### Pattern 4: The "Conflict Resolution" Pattern
- status: active
- type: guideline
- id: mcp_protocol_extended.prompt_engineering.conflict_resolution
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Problem**: RAG and tools might return contradictory information.

**Example**:
```
RAG: "Prof. Smith works on Epistemology."
Tool (search_people): {"name": "Prof. Smith", "research_interests": "Modal Logic, Metaphysics"}
```

**Solution**: Establish a priority hierarchy in the system prompt:

```python
conflict_resolution_rule = """

### INFORMATION SOURCE PRIORITY

When multiple sources provide conflicting information:

1. **PRIORITY ORDER**:
   a) Structured data from tools (highest priority - this is the source of truth)
   b) RAG context from recent documents
   c) Your general knowledge (lowest priority)

2. **RATIONALE**:
   - Tools query the live database (most up-to-date)
   - RAG may have outdated or cached information
   - Your training data has a knowledge cutoff

3. **CONFLICT HANDLING**:
   - If tool data contradicts RAG, trust the tool
   - If you notice a conflict, you MAY mention: "According to the latest database..."
   - Update your response based on the authoritative source

4. **EXAMPLE**:
   User: "What does Prof. Smith work on?"
   RAG: "Epistemology"
   Tool: "Modal Logic, Metaphysics"
   → Correct answer: "Prof. Smith's current research focuses on Modal Logic and Metaphysics."
"""

system_prompt = base_prompt + tools_section + tools_usage_rules + enrichment_rule + conflict_resolution_rule
```

**Why this matters**:
- Prevents outdated RAG documents from misleading users
- Ensures tools serve as the authoritative source for structured data
- Reduces user trust issues ("But the website says something different!")

## 8. Debugging Tool Usage
- status: active
- type: guideline
- id: mcp_protocol_extended.debugging
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

When the LLM isn't using tools correctly, follow this diagnostic checklist:

### Diagnostic Checklist
- status: active
- type: guideline
- id: mcp_protocol_extended.debugging.checklist
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **Are tools being injected?**
   - Add logging: `print(f"Tools injected: {[t['name'] for t in tools]}")`
   - Check: Does the LLM see the tools in its context?

2. **Is the tool description clear?**
   - Read the `description` field from a user's perspective
   - Ask: "Would I know when to use this tool based on this description?"
   - Common issues: Vague wording, missing examples, ambiguous "When to use"

3. **Is there tool overlap?**
   - Check if multiple tools could answer the same query
   - Example: `search_people` vs `get_publications` both can find author info
   - Fix: Add "When NOT to use" sections to create clear boundaries

4. **Is the LLM generating tool calls?**
   - Add logging: `print(f"LLM output: {response.text}")`
   - Search for tool call patterns (`<function_call>`, `<tool_use>`, etc.)
   - If absent: LLM doesn't recognize the need for a tool

5. **Is the parser detecting tool calls?**
   - Add logging in the parsing function
   - Check: Are you using the correct regex/XML parser for your LLM's output format?
   - Different LLMs use different formats (OpenAI vs Anthropic vs Gemini)

6. **Is the tool execution succeeding?**
   - Wrap tool execution in try-except with detailed logging
   - Common errors: Missing parameters, wrong parameter types, file not found

7. **Is the result being injected back?**
   - Log the tool result before sending it back to the LLM
   - Check: Is it properly formatted (JSON, XML, etc.)?
   - Too much data? Consider truncating large results

### Common Failure Modes and Fixes
- status: active
- type: documentation
- id: mcp_protocol_extended.debugging.failure_modes
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| LLM never calls tools | Tool descriptions too vague | Add explicit "When to use" section |
| LLM asks permission to use tools | Default polite behavior | Add "ALWAYS use tools directly" instruction |
| LLM uses wrong tool | Tool overlap or ambiguous descriptions | Add "When NOT to use" sections, clarify boundaries |
| Tool returns no results | Data missing or filtering too strict | Add logging, check data files, relax filters |
| LLM ignores tool results | Result not injected or wrong format | Check conversation history structure |
| LLM uses RAG instead of tool | Enrichment pattern not applied | Add "partial information → use tool" rule |
| Tool call fails with error | Missing parameter or wrong type | Add parameter validation, improve schema |

### Logging Best Practices
- status: active
- type: guideline
- id: mcp_protocol_extended.debugging.logging
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

Add structured logging to trace the full execution flow:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def chat(self, user_message: str) -> str:
    logger.info(f"User message: {user_message}")
    
    # Get tools
    tools = self.mcp_server.list_tools()
    logger.debug(f"Tools injected: {[t['name'] for t in tools]}")
    
    # Build prompt
    system_prompt = self._build_system_prompt_with_tools(tools)
    logger.debug(f"System prompt length: {len(system_prompt)} chars")
    
    # LLM call
    response = self.llm.generate_content(prompt=user_message, system_instruction=system_prompt)
    logger.info(f"LLM response (first 200 chars): {response.text[:200]}")
    
    # Detect tool call
    tool_call = self._parse_tool_call(response.text)
    if tool_call:
        logger.info(f"Tool call detected: {tool_call['name']}({tool_call['arguments']})")
        
        # Execute tool
        try:
            result = self.mcp_server.call_tool(tool_call['name'], tool_call['arguments'])
            logger.info(f"Tool result: {result[:200] if isinstance(result, str) else result}")
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"
    else:
        logger.info("No tool call detected")
    
    return response.text
```

This logging helps you trace exactly where the flow breaks down.

## 9. Future Extensions and Scalability
- status: active
- type: documentation
- id: mcp_protocol_extended.future_extensions
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

### Extension 1: Multi-Turn Tool Orchestration
- status: active
- type: documentation
- id: mcp_protocol_extended.future_extensions.multi_turn
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Current limitation**: Each tool call is independent. The LLM can't easily chain tools (e.g., "Find Prof. Smith, then get his publications, then summarize them").

**Solution**: Implement a "workflow" abstraction where the LLM can specify a sequence of tool calls:

```python
# Example LLM output for multi-turn orchestration
<workflow>
  <step id="1">
    <tool>search_people</tool>
    <arguments>{"query": "Smith"}</arguments>
  </step>
  <step id="2" depends_on="1">
    <tool>get_publications</tool>
    <arguments>{"author_name": "$1.name"}</arguments>  <!-- Reference previous result -->
  </step>
  <step id="3" depends_on="2">
    <tool>summarize_text</tool>
    <arguments>{"text": "$2[*].abstract"}</arguments>  <!-- Aggregate previous results -->
  </step>
</workflow>
```

The MCP Client would parse this, execute steps sequentially, and pass results between steps.

### Extension 2: Tool Composition
- status: active
- type: documentation
- id: mcp_protocol_extended.future_extensions.composition
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Concept**: Define high-level tools that combine multiple low-level tools.

**Example**: `get_researcher_profile` tool that internally calls:
1. `search_people` to find the person
2. `get_publications` to get their papers
3. `get_events` to find their upcoming talks

```python
def get_researcher_profile(name: str) -> Dict:
    """
    High-level tool that aggregates all information about a researcher.
    """
    # Internal tool calls
    person = search_people(query=name, limit=1)[0]
    publications = get_publications(author_name=name, limit=5)
    events = get_events(speaker_filter=name)
    
    # Aggregate results
    return {
        "name": person["name"],
        "role": person["role"],
        "research_interests": person["research_interests"],
        "contact": person["email"],
        "recent_publications": publications,
        "upcoming_talks": events
    }
```

**Benefit**: Reduces the number of tool calls the LLM needs to make, simplifying prompt engineering.

### Extension 3: Streaming Results
- status: active
- type: documentation
- id: mcp_protocol_extended.future_extensions.streaming
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Current limitation**: Tools return all results at once, which can be slow for large datasets.

**Solution**: Implement streaming tools that yield results incrementally:

```python
def search_people_stream(query: str, role_filter: Optional[str] = None):
    """
    Generator that yields matching people one at a time.
    """
    people_data = data_manager.people
    
    for person in people_data:
        if query.lower() in person["name"].lower() or \
           query.lower() in person.get("research_interests", "").lower():
            if role_filter is None or person.get("role") == role_filter:
                yield person  # Yield immediately, don't wait for all results
```

The MCP Client can display results progressively in the UI: "Found Prof. Smith... Found Prof. Jones... Found Dr. Brown..."

### Extension 4: External API Integration
- status: active
- type: documentation
- id: mcp_protocol_extended.future_extensions.external_apis
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

**Beyond JSON files**: Tools can call external APIs (arXiv, Google Scholar, university databases).

**Example**: `search_arxiv` tool:

```python
import requests

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv for papers matching the query.
    
    Args:
        query: Search query (keywords, authors, etc.)
        max_results: Maximum number of results
    
    Returns:
        List of paper dictionaries
    """
    # Call arXiv API
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
    response = requests.get(url)
    
    # Parse XML response
    # ... (parsing logic) ...
    
    return papers
```

**Benefit**: The chatbot can access real-time, external data without manually scraping and storing it.

## 10. Summary: Key Takeaways
- status: active
- type: documentation
- id: mcp_protocol_extended.summary
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

### Architectural Principles
- status: active
- type: documentation
- id: mcp_protocol_extended.summary.principles
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **Three-Layer Separation**: 
   - LLM generates text signals
   - Client orchestrates and parses
   - Server executes tools

2. **LLMs Don't Execute Code**: 
   - They output structured text patterns
   - The client interprets these patterns as function calls

3. **Tools Must Be In Context Every Time**: 
   - LLMs have no persistent memory of tools
   - Schemas must be injected on every request

4. **Explicit > Implicit**: 
   - Explicit tool descriptions in system prompts often outperform implicit API-level tool support

### Prompt Engineering Essentials
- status: active
- type: documentation
- id: mcp_protocol_extended.summary.prompting
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **Force Usage Pattern**: Tell the LLM to use tools directly, don't ask permission
2. **Data Enrichment Rule**: Use tools to complete partial information from RAG
3. **Conflict Resolution**: Establish clear priority (tools > RAG > general knowledge)
4. **Clear Descriptions**: "When to use" and "When NOT to use" prevent confusion

### Performance Optimization
- status: active
- type: documentation
- id: mcp_protocol_extended.summary.performance
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **Cache Loaded Data**: Use `@lru_cache` or singleton patterns
2. **Truncate Results**: Limit tool output to avoid token waste
3. **Lazy Loading**: Only load data when first accessed
4. **Migrate to SQL**: When datasets exceed 10k items or queries are slow

### Debugging Strategy
- status: active
- type: documentation
- id: mcp_protocol_extended.summary.debugging
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

1. **Log Everything**: Inject, parse, execute, return
2. **Check Each Layer**: Is the problem in LLM output, parsing, or execution?
3. **Read Tool Descriptions Critically**: Would a human know when to use this?
4. **Test Edge Cases**: Missing data, no results, multiple matches

### The MCP Abstraction Advantage
- status: active
- type: documentation
- id: mcp_protocol_extended.summary.abstraction
- last_checked: 2026-02-02
- label: ['agent']
<!-- content -->

The key insight of MCP is **separation of concerns**:
- **LLM**: Decides *when* to use a tool (pattern recognition)
- **Client**: Decides *how* to execute it (orchestration)
- **Server**: Decides *what* data to return (implementation)

This separation means you can:
- Swap LLMs without changing tools
- Swap data sources (JSON → SQL → API) without changing the LLM integration
- Add new tools without modifying the client or LLM prompts (just register and describe)

This makes the system modular, testable, and maintainable at scale.
