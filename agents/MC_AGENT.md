# Markov Chain Analysis Agent Instructions
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
**Role:** You are the **MC Agent** (Markov Chain Agent), a specialist in stochastic processes and long-term behavior analysis.

**Goal:** Ensure the simulation codebase properly tracks and exposes the fundamental Markov Chain properties of the learning dynamics, enabling rigorous analysis of convergence, equilibrium, and information flow.

## Background: The Simulation as a Markov Chain
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
The network epistemology simulation is fundamentally a **Markov Chain** where:

### State Space
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
- **Beta Agent:** The state is the collection of all agents' credence pairs: $S_t = \{(\alpha_i^{(0)}, \beta_i^{(0)}, \alpha_i^{(1)}, \beta_i^{(1)})\}_{i=1}^N$. Since these are continuous parameters updated via Bayesian learning, the state space is a subset of $\mathbb{R}^{4N}$.
- **Bayes Agent:** The state is the collection of all agents' scalar credences: $S_t = \{c_i\}_{i=1}^N \in [0,1]^N$.

### Transition Dynamics
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
At each time step:
1. **Experiment Phase:** Each agent $i$ chooses a theory based on current credences and runs $n$ experiments, getting outcomes drawn from $\text{Binomial}(n, p_{\text{theory}})$.
2. **Observation Phase:** Each agent $i$ observes the experimental outcomes of agents it "listens to" (predecessors in the directed graph).
3. **Update Phase:** Each agent updates its beliefs via Bayesian updating.

The randomness comes from:
- Initial belief states (random priors)
- Theory choice (epsilon-greedy randomization)
- Experimental outcomes (binomial draws)
- Optionally: sampling from posterior for credences

### Key Markov Properties to Track
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
1. **Absorbing States / Consensus States**
   - States where all agents have converged to extreme beliefs
   - For Bayes: $\forall i: c_i < 0.5$ (all reject truth) or $\forall i: c_i > 0.99$ (all accept truth)
   - For Beta: When credence changes per step fall below tolerance

2. **Transient vs. Recurrent States**
   - Early-stage belief distributions (transient)
   - Long-run behavior depends on root node influence

3. **Stationary Distribution (if exists)**
   - The long-run distribution of belief states
   - May not exist in classical sense due to absorbing states

4. **Mixing Time**
   - How quickly the chain approaches its limiting behavior
   - Related to convergence speed already tracked

5. **Root Node Influence**
   - Root nodes (in-degree = 0) act as "sources" of information
   - Their final beliefs determine the beliefs of all their descendants
   - This is already tracked via `compute_root_analysis`

## Core Responsibilities
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->

### 1. Verify Markov Property Preservation
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
The simulation MUST maintain the Markov property: the future state depends only on the current state, not on the history of how we got there.

**Checks:**
- [ ] Ensure no hidden state variables affect transitions
- [ ] Verify that given the same random seed and initial state, trajectories are reproducible
- [ ] Confirm that belief updates depend only on current beliefs + new evidence

### 2. State Space Tracking
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Ensure the codebase exposes sufficient information to reconstruct the full state at any time step.

**Current Capabilities (in `VectorizedModel`):**
- `self.alphas_betas`: Full Beta distribution parameters (N, 2, 2)
- `self.credences`: Current credence values
- `self.credences_history`: Optional per-agent credence trajectory

**Enhancements to Add:**
- [ ] **Transition Matrix Estimation**: For finite discretizations of the state space
- [ ] **State Fingerprinting**: Hash or summary of current state for comparison
- [ ] **Return Time Tracking**: Steps to return to similar states (if applicable)

### 3. Convergence Analysis Infrastructure
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Build on existing convergence tracking to provide deeper MC insights.

**Current Capabilities:**
- `belief_change_history`: Per-step mean belief change
- `belief_change_abs`, `belief_change_kl`: Final step metrics
- `compute_convergence` flag enables tracking

**Enhancements to Add:**
- [ ] **Total Variation Distance**: Track $\|P_t - P_\infty\|_{TV}$ if a limiting distribution can be estimated
- [ ] **Spectral Gap Estimation**: For linearized dynamics around equilibrium
- [ ] **Coupling Time Estimation**: When two chains started from different states merge

### 4. Root Influence as Markov Structure
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
The root node analysis reveals structural properties of the Markov Chain.

**Key Insight:** 
In DAGs with root nodes, the long-run behavior is determined by:
1. Which roots reach truth
2. The reachability structure of the network

**Current Implementation:**
- `proportion_reached_by_truth`: Collective reachability from truthful roots
- `root_analysis`: Detailed breakdown per root

**Enhancements to Add:**
- [ ] **Absorption Probability Calculator**: P(absorbing to truth | initial state)
- [ ] **Mean Hitting Time**: Expected steps to reach consensus
- [ ] **Root Influence Decomposition**: Per-root contribution to final outcome

## Implementation Plan
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->

### Phase 1: State Space Utilities (New File: `mc_analysis.py`)
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Create `src/net_epistemology/analysis/mc_analysis.py` with:

```python
class MarkovChainAnalyzer:
    """Tools for analyzing the Markov Chain properties of simulations."""
    
    def __init__(self, model):
        """Attach to a VectorizedModel instance."""
        self.model = model
        self.state_snapshots = []
        
    def snapshot_state(self):
        """Record current state for trajectory analysis."""
        pass
        
    def state_fingerprint(self):
        """Compute a hashable representation of current state."""
        pass
        
    def estimate_transition_kernel(self, n_samples=1000):
        """Estimate local transition probabilities via Monte Carlo."""
        pass
        
    def check_markov_property(self, n_tests=100):
        """Verify that transitions don't depend on history."""
        pass
```

### Phase 2: Convergence Diagnostics
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Add to `VectorizedModel` or create separate analyzer:

```python
def estimate_mixing_time(self, epsilon=0.01, n_chains=10):
    """
    Estimate mixing time by running parallel chains from different starts
    and measuring when total variation distance falls below epsilon.
    """
    pass

def compute_spectral_gap(self):
    """
    For the linearized belief dynamics around equilibrium,
    estimate the spectral gap of the transition matrix.
    Larger gap = faster mixing.
    """
    pass
```

### Phase 3: Absorption Analysis
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
```python
def estimate_absorption_probabilities(self, n_simulations=100):
    """
    Estimate probability of absorbing to truth vs. falsehood
    from current state via Monte Carlo simulation.
    """
    pass

def mean_hitting_time_to_consensus(self, n_simulations=100):
    """
    Estimate expected time to reach consensus state.
    """
    pass
```

## Verification Checklist
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
Before any MC analysis code is complete, verify:

- [ ] **Reproducibility Test**: Same seed → same trajectory
- [ ] **Markov Test**: P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t)
- [ ] **Convergence Test**: Long runs approach stable distributions
- [ ] **Root Influence Test**: Predictions from root analysis match long-run outcomes

## Related Files
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
- `notebooks/convergence_analysis/convergence_studies.py`: Existing convergence tracking
- `notebooks/convergence_analysis/root_influence_analysis.py`: Root node influence studies
- `src/net_epistemology/core/vectorized_model.py`: Main simulation engine
- `AI_AGENTS/LINEARIZE_AGENT.md`: Vectorization details

## Mathematical Notes
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->

### Belief Update as Markov Transition
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
For **Beta agents**, the transition is:
$$
(\alpha_i^{(t)}, \beta_i^{(t)}) \to (\alpha_i^{(t+1)}, \beta_i^{(t+1)}) = (\alpha_i^{(t)} + S_i, \beta_i^{(t)} + F_i)
$$
where $S_i$ = successes observed by agent $i$, $F_i$ = failures observed.

The randomness enters through:
1. Which theory each agent tests (choice)
2. The binomial outcome of experiments
3. The aggregation over the network

### Long-Run Behavior Theorem (Informal)
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
**Claim:** In a DAG with root nodes, if we run the simulation long enough:
- All descendants of roots that converge to truth will converge to truth
- All descendants of roots that converge to falsehood will converge to falsehood
- The gap between "proportion reached by truthful roots" and "actual truth believers" → 0

This is empirically verified in `root_influence_analysis.py`.

## Agent Log Entry Template
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
When implementing MC analysis features, log in `AGENTS_LOG.md`:

```markdown

### [DATE] - Markov Chain Analysis Implementation (MC Agent)
- status: active
- type: agent_skill
- label: ['agent']
<!-- content -->
*   **Task:** [Specific MC feature implemented]
*   **Actions:**
    *   [File created/modified]
    *   [Tests added]
*   **Verification:**
    *   [How correctness was verified]
```
