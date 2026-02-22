# Linearize Agent Instructions
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
**Role:** You are the **Linearize Agent**, a specialist in numerical computing and optimization.
**Goal:** Drastically improve the performance of the simulation by "linearizing" or "vectorizing" the logic—replacing explicit Python loops (iterating over agent objects) with efficient NumPy matrix operations.

## Core Constraints (Strict)
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
1.  **Immutable Legacy Code:** You **MUST NOT** modify `model.py`, `agents.py`, or `simulation_functions.py`. These files are the "ground truth" reference implementation.
2.  **New Implementation:** You will create new files, likely `vectorized_model.py` (and `vectorized_agents.py` if necessary).
3.  **Equivalence:** The vectorized implementation must produce statistically equivalent results to the original model when given the same random seed (allowing for minor floating-point differences).

## Technical Strategy
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->

### 1. Data Structure Transformation
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
The current object-oriented approach stores state inside N `BetaAgent` objects. You must refactor this into centralized matrices managed by your new `VectorizedModel`.

*   **Current:** `agent.alphas_betas` (list of 2x2 arrays scattered in memory).
*   **Target:** `self.agent_states` -> A NumPy array of shape `(N_agents, 2, 2)` or distinct arrays for Alphas/Betas.
*   **Current:** `agent.credences` (list of arrays).
*   **Target:** `self.credences` -> A NumPy array of shape `(N_agents, 2)`.

### 2. Vectorizing the Graph (The "Linearize" Part)
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Instead of iterating `network.predecessors(agent.id)`, use the Adjacency Matrix.

*   Convert the `networkx` graph to a sparse matrix or NumPy array: $A$.
*   **Directionality Note:** Recall from `AGENTS.md` that if $A \to B$ ($A$ cites $B$), then $B$ observes $A$.
    *   You need to determine the correct matrix multiplication: $M \cdot V$ or $M^T \cdot V$ to aggregate observations from neighbors.
    *   If $A_{ij} = 1$ means $i \to j$, and $j$ observes $i$, you are summing over column $j$ (predecessors).

### 3. Vectorizing the Experiment Step
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
*   Replace:
    ```python
    for agent in agents:
        agent.experiment(...)
    ```
*   With:
    ```python
    # Batch generate choices
    choices = self.egreedy_choice_vectorized(...)
    # Batch generate outcomes
    successes = np.random.binomial(n, p, size=N)
    ```

### 4. Vectorizing the Update Step
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
*   Accumulate successes/failures from neighbors using matrix multiplication.
*   Update the state matrices (`Alphas`, `Betas`) in one operation.

### 5. Bayes Agent Implementation
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
*   **Support:** The vectorized model now supports `agent_type="bayes"`.
*   **State:** `self.credences` is a 1D array of shape `(N_agents,)`.
*   **Choice:** Vectorized check `credences > 0.5`.
*   **Experiments:** Vectorized `binomial` generation, masked to only include agents who chose theory 1.
*   **Update:** Vectorized Bayesian update formula applied using aggregated evidence for Theory 1.

## Verification Plan
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
1.  **Unit Test:** Create `tests/test_vectorization.py`.
    *   Initialize `Model` and `VectorizedModel` with the same `seed`.
    *   Run 1 step.
    *   Assert `Model.agents[i].credences` $\approx$ `VectorizedModel.credences[i]`.
2.  **Integration Test:** Run `basic_model_testing.ipynb` AND `run_simulations_test.ipynb` using your new class to ensure visual and statistical behavior matches the baseline.
3.  **Benchmark:** Prove the speedup! Log the time difference between the loop-based and vectorized approaches.

## Checklist
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
- [x] Read `AGENTS.md` to understand the graph direction logic perfectly.
- [x] Create `vectorized_model.py`.
- [x] Implement global state matrices.
- [x] Implement matrix-based update logic.
- [x] Verify against `model.py` with a shared seed.
- [x] Run visual verification notebooks (`basic_model_testing.ipynb`, `run_simulations_test.ipynb`).
- [x] Log results in `AGENTS_LOG.md`.
- [x] Implement and verify Bayes Agent vectorization.
