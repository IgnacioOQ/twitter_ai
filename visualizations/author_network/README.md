# Author network viewer

Offline WebGL viewer for the supplied matched-author network. Author IDs, topic
names, community assignments and counts come from the selected data bundle.
Changing display controls does not recompute analytical profiles or communities.

The `layout` and `export` author-analysis stages produce `matched_nodes.parquet`,
`matched_edges.parquet` and a compact viewer bundle. Serve that bundle directly:

```sh
python -m src.network.visualization.serve --data-dir OUT/networks/RetweetedOnce/viewer
```

Open http://127.0.0.1:8000. All browser dependencies are included: regl2.1.0
(`vendor/regl-LICENSE.txt`) and EB Garamond (`EB-GARAMOND-OFL.txt`). Optional
`community_labels.json` and `representative_posts.json` files may be supplied for
this exact network; numeric communities and no examples are the default.

## Build the 3D layout

Install the Python requirements and Graphviz development libraries. Graphviz14.1.3
was tested. Compile the included helper against the same installed Graphviz
headers and libraries. For a system with pkg-config and a C compiler:

```sh
mkdir -p data_sets/tools
cc src/network/visualization/sfdp_coords.c -o data_sets/tools/sfdp_coords $(pkg-config --cflags --libs libgvc)
```

On Windows, compile with the Graphviz include directory and `gvc.lib`, `cgraph.lib`
and `cdt.lib`; put its DLL directory on PATH when running. The helper extracts
native double-precision coordinates before conversion to the browser format.
The binary is a locally built dependency and should not be committed.

```sh
python -m src.network.visualization.build_layout_3d --nodes OUT/networks/RetweetedOnce/layout/matched_nodes.parquet --edges OUT/networks/RetweetedOnce/layout/matched_edges.parquet --viewer-nodes OUT/networks/RetweetedOnce/viewer/nodes.json --sfdp-runner ./data_sets/tools/sfdp_coords --output-dir OUT/networks/RetweetedOnce/viewer
python -m src.network.visualization.validate_3d_layout --data-dir OUT/networks/RetweetedOnce/viewer
```

The builder generates its own component graph, native coordinates, render indices
and metadata. It selects the largest weak component from the supplied graph and
uses seed42 by default. Native `sfdp` uses unweighted undirected topology. The 2D
macro constraint uses log1p summed edge weights, Laplacian strength0.10 and target
RMS displacement ratio0.80; all depth comes from the native 3D fit. Parameters are
explicit flags. Topics and community labels do not enter either layout.

Only the largest component is shown in 3D. All matched authors remain in 2D.
Inputs must preserve exact node order; metadata hashes bind each generated binary
to its viewer data, with no fixed author counts or fixed input fingerprints.
Axes, orientation and absolute scale have no substantive meaning.

For descriptive graph/geometry measurements and equal-axis renders, use
`python -m src.network.visualization.evaluate_3d_layout --help` and
`python -m src.network.visualization.render_layout_3d --help`. Structural validation
checks file consistency; geometry measurements still need scientific interpretation.
New data or dependency versions can produce different layouts.
