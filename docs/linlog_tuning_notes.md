# Linlog Layout Tuning Notes

*Active tuning log for the GPU FA2 `linlog` layout recipe. Logged so we can resume tuning without repeating failed experiments.*

## Goal
Achieve a soft, radial nebula layout (characteristic of LinLog's logarithmic attraction) that still separates color-coded communities distinctly, without letting nodes sprawl endlessly into the borders or squashing the graph into an unstructured blob.

## Experiments

### 1. Default Baseline (The "Diffuse Ball")
- **Parameters**: `scaling_ratio=2.0`, `strong_gravity_mode=False`, `gravity=1.0`
- **Result**: A diffuse, highly mixed ball. Communities were slightly visible but heavily overlapping. A massive halo of pendant/leaf nodes sprawled outwards, hitting the clipping boundaries and forming a hard ring of nodes at the extreme edges.
- **Verdict**: Community clustering was decent, but boundary artifacts were unacceptable.

### 2. High Scaling + Strong Gravity (The "Foam Disc")
- **Parameters**: `scaling_ratio=50.0`, `strong_gravity_mode=True`, `gravity=1.0`
- **Result**: Pushed into a perfectly uniform, densely packed circle (a "Petri dish" or "foam" disc). `strong_gravity_mode` scales gravity with distance, which violently pulled the boundary nodes inward. This solved the boundary sprawl, but the inward pressure was so high it completely crushed the communities together into a uniform texture.
- **Verdict**: Excellent circular footprint, but destroyed community separation. Nodes appeared disconnected from edges due to extreme compression.

### 3. Final Tuned Version (High Scaling + Elevated Gravity)
- **Parameters**: `scaling_ratio=5.0`, `strong_gravity_mode=False`, `gravity=2.0`
- **Result**: Increasing `scaling_ratio` pulled the communities apart, solving the lack of distinct clustering seen in the baseline. Simultaneously, keeping `gravity=2.0` reeled in the outer leaves just enough to maintain a pleasing, soft circular nebula shape without crossing the threshold into the squashed "foam disc".
- **Verdict**: Optimal setup for the soft radial look. It balances clear community structure with LinLog's signature circular boundary.

## Conclusion
The final LinLog recipe (`scaling_ratio=5.0`, `gravity=2.0`) successfully achieves color-clustered communities within a soft radial boundary. 
**Crucial Finding**: `strong_gravity_mode` must be avoided on graphs of this scale (millions of edges), as it applies overwhelming compression that crushes any modularity structure into a foam-like singularity.
