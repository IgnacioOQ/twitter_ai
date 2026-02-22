# MCP Scikit-Learn Integration Plan
- status: active
- type: plan
- owner: antigravity
<!-- content -->
This document outlines the strategy for integrating **Scikit-Learn** into the Local Nexus via the **Model Context Protocol (MCP)**. This allows the chatbot to perform machine learning operations (training, prediction, data loading) in an isolated, standardized environment.

## Context & Objective
- status: active
- type: documentation
<!-- content -->
We aim to replicate the capabilities of `mcp-server-scikit-learn` to give our LLM tools to manipulate data.
*   **Goal**: Enable the Nexus Chatbot to classify/regress data stored in the Local Warehouse.
*   **Method**: Run a local MCP server that exposes Sklearn functions as **Tools**.
*   **Reference**: [shibuiwilliam/mcp-server-scikit-learn](https://github.com/shibuiwilliam/mcp-server-scikit-learn)

## Architecture
- status: active
- type: documentation
<!-- content -->
The system will follow a Client-Host-Server model.

1.  **Host (Local Nexus)**: The Streamlit app (`src/app.py`).
2.  **Client**: The Gemini LLM (via `google-generativeai`). *Note: Google's Native Client might not support MCP directly yet, so we may need a "Tooling Bridge" that translates Gemini Function Calls -> MCP Tool Calls.*
3.  **Server (Sklearn MCP)**: A standalone Python process running `fastmcp` or standard `mcp` SDK.

## Implementation Steps
- status: active
- type: plan
<!-- content -->

### Phase 1: Server Setup
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_1_server_setup
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
1.  **Dependencies**:
    *   `mcp`
    *   `scikit-learn`
    *   `pandas`
    *   `numpy`
2.  **Server Script** (`src/mcp_server/sklearn_server.py`):
    *   Initialize `FastMCP("sklearn")`.
    *   Expose tools: `train_model`, `predict`, `evaluate_model`.

### Phase 2: Tool Definitions
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_2_tool_definitions
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
Define the specific tools the LLM can call.

#### 1. Data Loading
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_2_tool_definitions.1_data_loading
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
*   `load_data(table_name)`: Fetch data from the local DuckDB warehouse.

#### 2. Training
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_2_tool_definitions.2_training
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
*   `train_model(model_type, target_column, hyperparameters)`:
    *   Supports: `RandomForestClassifier`, `LinearRegression`, `LogisticRegression`.
    *   Returns: A unique `model_id` and metrics (Accuracy/R2).

#### 3. Inference
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_2_tool_definitions.3_inference
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
*   `predict(model_id, input_data)`: Use a trained model to make predictions.

### Phase 3: Client Integration
- id: mcp_scikit_learn_integration_plan.implementation_steps.phase_3_client_integration
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
1.  **Bridge Layer** (`src/core/mcp_client.py`):
    *   Start the MCP server subprocess (`stdio`).
    *   Fetch tool definitions (`list_tools`).
    *   Convert MCP Tools -> Gemini `tools` format.
2.  **Chat Loop Update**:
    *   Pass tools to `model.generate_content`.
    *   Handle `function_call` responses by routing them to the MCP Client.

## Execution Checklist
- status: todo
- type: task
<!-- content -->

### Dependencies
- id: mcp_scikit_learn_integration_plan.execution_checklist.dependencies
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
- [ ] Add `mcp` to `requirements.txt`.
- [ ] Add `scikit-learn` to `requirements.txt`.

### Server Development
- id: mcp_scikit_learn_integration_plan.execution_checklist.server_development
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
- [ ] Create `src/mcp_server/` directory.
- [ ] Implement `sklearn_server.py` using `FastMCP`.
- [ ] Implement `train_model` tool.
- [ ] Implement `predict` tool.

### Client Integration
- id: mcp_scikit_learn_integration_plan.execution_checklist.client_integration
- status: active
- type: documentation
- last_checked: 2026-01-27
<!-- content -->
- [ ] Create `src/core/mcp_bridge.py`.
- [ ] Wire up Gemini Function Calling to MCP Bridge.
