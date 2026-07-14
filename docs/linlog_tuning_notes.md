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

### 3. Current Version (Slightly Higher Gravity)
- **Parameters**: `scaling_ratio=2.0`, `strong_gravity_mode=False`, `gravity=2.0`
- **Result**: Reverts the extreme compression of `strong_gravity_mode` to allow communities to naturally push apart (restoring the clustering of Experiment 1). Doubles the standard `gravity` to gently reel the boundary nodes back into a natural circular shape without squashing the core.
- **Verdict**: (Pending review) Should offer the best balance of circular footprint and readable community clusters.

## Next Steps / Future Tuning
If Experiment 3 is still too tight or too loose:
- To **separate communities more**: Increase `scaling_ratio` (e.g., `5.0` or `10.0`) while keeping `strong_gravity_mode=False`.
- To **pull the boundary in tighter**: Increase `gravity` incrementally (e.g., `3.0` or `5.0`).
- **Never** combine high `scaling_ratio` with `strong_gravity_mode` for this graph, as it induces the "foam disc" failure state.
