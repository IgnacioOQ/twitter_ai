#!/usr/bin/env python3
"""Derive a bipartite producer/consumer graph from the project's notebooks.

Nodes: (a) notebooks, (b) data artifacts (JSON / PKL / GML / CSV / GZ files).
Edges: notebook -> artifact (writes), artifact -> notebook (reads).

Run with no args to (re)generate docs/pipeline_graph.{json,png} and
docs/pipeline_graph_notebooks.png from the live notebooks. The graph is
re-derived on every run; there is no hand-maintained manifest.
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import nbformat
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import yaml
except ImportError as e:
    sys.exit(
        f"missing dependency: {e.name}\n"
        f"install with: pip install nbformat networkx matplotlib pyyaml"
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"
OVERRIDES_PATH = DOCS_DIR / "pipeline_overrides.yaml"
OUT_JSON = DOCS_DIR / "pipeline_graph.json"
OUT_PNG_FULL = DOCS_DIR / "pipeline_graph.png"
OUT_PNG_NB = DOCS_DIR / "pipeline_graph_notebooks.png"

DRIVE_PREFIXES = ("/content/drive/", "/Volumes/GoogleDrive/")

WRITE_MODES = {"w", "wb", "wt", "a", "ab", "at"}
READ_MODES = {"r", "rb", "rt"}

NX_WRITE_FUNCS = {"write_gml", "write_graphml", "write_gpickle", "Write_GML", "Write_GraphML"}
NX_READ_FUNCS = {"read_gml", "read_graphml", "read_gpickle", "Read_GML", "Read_GraphML"}

PD_WRITE_METHODS = {"to_csv", "to_pickle", "to_json", "to_parquet"}
PD_READ_FUNCS = {"read_csv", "read_pickle", "read_json", "read_parquet"}

# numpy save/load take a path directly as arg 0
NP_WRITE_FUNCS = {"save", "savez", "savez_compressed"}
NP_READ_FUNCS = {"load"}

DUMP_FUNCS = {"dump"}
LOAD_FUNCS = {"load"}

STAGE_COLORS = {
    1: "#cfe7ff",
    2: "#cdebd0",
    3: "#ffd9b3",
    4: "#e0c3f0",
    5: "#f5b0b0",
    6: "#dcdcdc",
}


def _strip_drive_prefix(p: str) -> str:
    for prefix in DRIVE_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix) :]
    return p


def _eval(node, env):
    """Best-effort evaluation of a path-shaped AST expression to a string.

    Returns None when the expression cannot be resolved with the bindings in env.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _eval(node.value, env)
        if owner is None:
            return None
        return f"{owner}.{node.attr}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if left is None or right is None:
            return None
        return f"{left.rstrip('/')}/{right.lstrip('/')}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    if isinstance(node, ast.Call):
        func = _func_qualname(node.func)
        if func == "Path" and node.args:
            return _eval(node.args[0], env)
        if func == "str" and node.args:
            return _eval(node.args[0], env)
        if func.endswith(".joinpath") or func == "joinpath":
            base = _eval(node.func.value, env) if isinstance(node.func, ast.Attribute) else None
            if base is None:
                return None
            parts = [_eval(a, env) for a in node.args]
            if any(p is None for p in parts):
                return None
            return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)
        return None
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = _eval(v.value, env)
                if isinstance(inner, str) and not _looks_templated(inner):
                    out.append(inner)
                else:
                    name = ast.unparse(v.value)
                    out.append("{" + name + "}")
            else:
                return None
        return "".join(out)
    return None


def _looks_templated(s: str) -> bool:
    return "{" in s and "}" in s


def _func_qualname(node):
    """Render a Call.func node as a dotted qualified name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _func_qualname(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_path_like(s: str) -> bool:
    """Heuristic: a resolved string is a real artifact reference, not a stray literal."""
    if not s or "/" not in s:
        return False
    # an artifact has a recognizable extension or sits under one of our pipeline folders
    if re.search(r"\.(json|jsonl|pkl|pickle|gml|graphml|csv|gz|npz|npy|parquet|txt|tsv)(\.gz)?$", s, re.I):
        return True
    return False


def collect_path_env(setup_cell_src: str) -> tuple[dict, list[str]]:
    """Walk a notebook's setup cell and return the {var_name: resolved_path} env.

    For `if RUNNING_LOCALLY: ... else: ...`, the else (Colab) branch wins. Returns
    (env, warnings).
    """
    env: dict = {}
    warnings: list[str] = []
    try:
        tree = ast.parse(_strip_magics(setup_cell_src))
    except SyntaxError:
        return env, warnings

    def visit(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                # canonical Colab path lives in the `else` branch
                test_src = ast.unparse(stmt.test)
                if "RUNNING_LOCALLY" in test_src or "USE_LOCAL_TEST_DATA" in test_src:
                    visit(stmt.orelse)
                else:
                    visit(stmt.body)
                    visit(stmt.orelse)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                name = stmt.targets[0].id
                value = _eval(stmt.value, env)
                if isinstance(value, str):
                    env[name] = _strip_drive_prefix(value)

    visit(tree.body)
    return env, warnings


def _classify_open_call(call: ast.Call):
    """For an open()/gzip.open() call, return ('write'|'read'|None, path_arg_node)."""
    func = _func_qualname(call.func)
    if func not in {"open", "gzip.open"}:
        return None, None
    mode = "r"
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        mode = str(call.args[1].value)
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = str(kw.value.value)
    if any(c in mode for c in ("w", "a", "x")):
        return "write", call.args[0] if call.args else None
    if "r" in mode and "+" not in mode:
        return "read", call.args[0] if call.args else None
    return None, None


def _strip_magics(src: str) -> str:
    """Drop Jupyter cell-magic and line-magic lines so ast.parse succeeds."""
    lines = src.splitlines()
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_io_from_cell(src: str, env: dict, in_loop: bool = False):
    """Yield (kind, path_str, templated, glob) tuples for one code cell.

    The env dict is mutated in place with any new {name: resolved_path} bindings
    discovered in this cell, so later cells in the same notebook see them.
    """
    try:
        tree = ast.parse(_strip_magics(src))
    except SyntaxError:
        return

    yield_buffer: list = []

    def emit(kind, path, in_loop_ctx):
        if not isinstance(path, str):
            return
        path = _strip_drive_prefix(path)
        if not _is_path_like(path):
            return
        templated = _looks_templated(path)
        is_glob = in_loop_ctx and templated
        yield_buffer.append((kind, path, templated, is_glob))

    def walk(node, in_loop_ctx):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            child_ctx = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child_ctx = in_loop_ctx
        else:
            child_ctx = in_loop_ctx

        # capture local path bindings as we walk, so later statements see them
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = _eval(node.value, env)
            if isinstance(v, str):
                env[node.targets[0].id] = v

        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call):
                    kind, path_node = _classify_open_call(ctx)
                    if kind and path_node is not None:
                        emit(kind, _eval(path_node, env), in_loop_ctx)

        if isinstance(node, ast.Call):
            kind, path_node = _classify_open_call(node)
            if kind and path_node is not None:
                emit(kind, _eval(path_node, env), in_loop_ctx)

            func = _func_qualname(node.func)
            simple = func.split(".")[-1]
            prefix = func.split(".")[0] if "." in func else ""
            if simple in NX_WRITE_FUNCS and len(node.args) >= 1:
                idx = 1 if len(node.args) >= 2 else 0
                emit("write", _eval(node.args[idx], env), in_loop_ctx)
            elif simple in NX_READ_FUNCS and node.args:
                emit("read", _eval(node.args[0], env), in_loop_ctx)
            elif simple in PD_READ_FUNCS and node.args:
                emit("read", _eval(node.args[0], env), in_loop_ctx)
            elif simple in PD_WRITE_METHODS and node.args:
                emit("write", _eval(node.args[0], env), in_loop_ctx)
            elif simple in NP_WRITE_FUNCS and prefix in {"np", "numpy"} and node.args:
                emit("write", _eval(node.args[0], env), in_loop_ctx)
            elif simple in NP_READ_FUNCS and prefix in {"np", "numpy"} and node.args:
                emit("read", _eval(node.args[0], env), in_loop_ctx)
            elif simple in DUMP_FUNCS and len(node.args) >= 2:
                file_arg = node.args[1]
                if isinstance(file_arg, ast.Call):
                    kind, path_node = _classify_open_call(file_arg)
                    if kind == "write" and path_node is not None:
                        emit("write", _eval(path_node, env), in_loop_ctx)
            elif simple in LOAD_FUNCS and node.args:
                file_arg = node.args[0]
                if isinstance(file_arg, ast.Call):
                    kind, path_node = _classify_open_call(file_arg)
                    if kind == "read" and path_node is not None:
                        emit("read", _eval(path_node, env), in_loop_ctx)

        for child in ast.iter_child_nodes(node):
            walk(child, child_ctx)

    walk(tree, in_loop)
    yield from yield_buffer


def parse_notebook(path: Path):
    """Return {'reads': [...], 'writes': [...], 'env': {...}, 'unresolved': [...]}.

    Each read/write entry is dict(path=str, templated=bool, glob=bool).
    """
    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if not code_cells:
        return {"reads": [], "writes": [], "env": {}, "unresolved": []}

    env, _ = collect_path_env(code_cells[0].source)

    reads: dict = {}
    writes: dict = {}
    unresolved: list = []

    for cell in code_cells[1:]:
        for kind, p, templated, glob in extract_io_from_cell(cell.source, env):
            entry = {"path": p, "templated": templated, "glob": glob}
            target = writes if kind == "write" else reads
            target[p] = entry

    return {
        "reads": list(reads.values()),
        "writes": list(writes.values()),
        "env": env,
        "unresolved": unresolved,
    }


def stage_of(notebook_id: str) -> int:
    m = re.match(r"^(\d+)_", notebook_id)
    return int(m.group(1)) if m else 0


def discover_notebooks() -> list[Path]:
    out = []
    for p in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        if "archive" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        out.append(p)
    return out


def load_overrides() -> dict:
    if not OVERRIDES_PATH.exists():
        return {"alternatives": [], "glob_artifacts": [], "friendly_labels": {}}
    with open(OVERRIDES_PATH) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("alternatives", [])
    data.setdefault("glob_artifacts", [])
    data.setdefault("friendly_labels", {})
    return data


def _glob_matches(pattern: str, path: str) -> bool:
    rx = re.escape(pattern).replace(r"\*", ".*")
    return re.fullmatch(rx, path) is not None


def build_graph(notebooks: list[Path], overrides: dict) -> tuple[nx.DiGraph, list[str]]:
    G = nx.DiGraph()
    warnings: list[str] = []

    glob_patterns = overrides.get("glob_artifacts", [])
    canonical_for_glob: dict = {}

    parsed: dict = {}
    for nb_path in notebooks:
        nb_id = str(nb_path.relative_to(NOTEBOOKS_DIR)).replace(".ipynb", "")
        info = parse_notebook(nb_path)
        parsed[nb_id] = info
        G.add_node(
            nb_id,
            kind="notebook",
            stage=stage_of(nb_id),
            path=str(nb_path.relative_to(REPO_ROOT)),
            bipartite=0,
        )

    base_paths = {info["env"].get("BASE_PATH") for info in parsed.values() if info["env"].get("BASE_PATH")}
    if len(base_paths) > 1:
        warnings.append(f"BASE_PATH divergence across notebooks: {sorted(base_paths)}")

    def canonical_artifact_id(p: str) -> str:
        for pat in glob_patterns:
            if _glob_matches(pat, p):
                canonical_for_glob[pat] = pat
                return pat
        return p

    for nb_id, info in parsed.items():
        for entry in info["writes"]:
            art_id = canonical_artifact_id(entry["path"])
            if art_id not in G:
                G.add_node(
                    art_id,
                    kind="artifact",
                    drive_path=art_id,
                    glob=art_id in canonical_for_glob or entry["glob"],
                    templated=entry["templated"],
                    bipartite=1,
                )
            G.add_edge(nb_id, art_id, direction="write")
        for entry in info["reads"]:
            art_id = canonical_artifact_id(entry["path"])
            if art_id not in G:
                G.add_node(
                    art_id,
                    kind="artifact",
                    drive_path=art_id,
                    glob=art_id in canonical_for_glob or entry["glob"],
                    templated=entry["templated"],
                    bipartite=1,
                )
            G.add_edge(art_id, nb_id, direction="read")

    for pair in overrides.get("alternatives", []):
        if len(pair) == 2 and pair[0] in G and pair[1] in G:
            G.add_edge(pair[0], pair[1], direction="alternative", style="alternative")
            G.add_edge(pair[1], pair[0], direction="alternative", style="alternative")

    return G, warnings


def project_to_notebook_dag(G: nx.DiGraph) -> nx.DiGraph:
    P = nx.DiGraph()
    for n, d in G.nodes(data=True):
        if d.get("kind") == "notebook":
            P.add_node(n, **d)
    for art, d in G.nodes(data=True):
        if d.get("kind") != "artifact":
            continue
        producers = [u for u, _ in G.in_edges(art) if G.nodes[u].get("kind") == "notebook"]
        consumers = [v for _, v in G.out_edges(art) if G.nodes[v].get("kind") == "notebook"]
        for p in producers:
            for c in consumers:
                if p == c:
                    continue
                P.add_edge(p, c, via=art)
    return P


def dump_json(G: nx.DiGraph, path: Path):
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"id": n, **d})
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"src": u, "dst": v, **d})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)


def _layered_positions(G_nb: nx.DiGraph) -> dict:
    """y = -stage (1 at top, 6 at bottom); x spread by within-stage notebook index."""
    by_stage: dict = defaultdict(list)
    for n in G_nb.nodes:
        s = G_nb.nodes[n].get("stage", 0)
        by_stage[s].append(n)
    pos = {}
    for stage, nodes in by_stage.items():
        nodes_sorted = sorted(nodes)
        for i, n in enumerate(nodes_sorted):
            x = (i - (len(nodes_sorted) - 1) / 2) * 5.5
            pos[n] = (x, -stage * 3.0)
    return pos


def _short_notebook_label(nb_id: str) -> str:
    """Return e.g. '02/02 sanity_check' for '02_Processing/02_sanity_check_and_network_generation'."""
    parts = nb_id.split("/")
    if len(parts) != 2:
        return nb_id
    stage_m = re.match(r"^(\d+)_", parts[0])
    idx_m = re.match(r"^(\d+)_(.+)$", parts[1])
    if not (stage_m and idx_m):
        return nb_id
    return f"{stage_m.group(1)}/{idx_m.group(1)} {idx_m.group(2)[:24]}"


def _label_for_artifact(art_id: str, friendly: dict) -> str:
    if art_id in friendly:
        return friendly[art_id]
    return Path(art_id).name


def render_full(G: nx.DiGraph, friendly: dict, out_path: Path):
    G_nb = project_to_notebook_dag(G)
    nb_pos = _layered_positions(G_nb)

    pos: dict = dict(nb_pos)
    artifacts = [n for n, d in G.nodes(data=True) if d.get("kind") == "artifact"]
    for art in artifacts:
        producers = [u for u, _ in G.in_edges(art) if G.nodes[u].get("kind") == "notebook"]
        consumers = [v for _, v in G.out_edges(art) if G.nodes[v].get("kind") == "notebook"]
        nbrs = producers + consumers
        nbr_ys = [nb_pos[n][1] for n in nbrs if n in nb_pos]
        nbr_xs = [nb_pos[n][0] for n in nbrs if n in nb_pos]
        if nbr_ys:
            pos[art] = (sum(nbr_xs) / len(nbr_xs), sum(nbr_ys) / len(nbr_ys) - 0.6)
        else:
            pos[art] = (0, -10)

    # de-overlap artifacts at same y by jittering x within their cluster
    by_y: dict = defaultdict(list)
    for art in artifacts:
        by_y[round(pos[art][1], 1)].append(art)
    for y, arts in by_y.items():
        arts_sorted = sorted(arts, key=lambda a: pos[a][0])
        n = len(arts_sorted)
        if n <= 1:
            continue
        center_x = sum(pos[a][0] for a in arts_sorted) / n
        for i, a in enumerate(arts_sorted):
            pos[a] = (center_x + (i - (n - 1) / 2) * 2.6, y)

    fig, ax = plt.subplots(figsize=(34, 22))

    notebook_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "notebook"]
    nb_colors = [STAGE_COLORS.get(G.nodes[n].get("stage", 0), "#cccccc") for n in notebook_nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=notebook_nodes, node_shape="s",
        node_size=3000, node_color=nb_colors, edgecolors="#222", linewidths=1.4, ax=ax,
    )

    artifact_nodes = artifacts
    art_edge_colors = ["#444" for _ in artifact_nodes]
    art_widths = [1.8 if G.nodes[a].get("glob") else 0.8 for a in artifact_nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=artifact_nodes, node_shape="o",
        node_size=600, node_color="#fff5cc",
        edgecolors=art_edge_colors, linewidths=art_widths, ax=ax,
    )

    write_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "write"]
    read_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "read"]
    alt_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "alternative"]

    nx.draw_networkx_edges(
        G, pos, edgelist=write_edges, edge_color="#3b6", width=1.2,
        arrows=True, arrowsize=10, alpha=0.85, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=read_edges, edge_color="#48a", width=0.7,
        arrows=True, arrowsize=8, alpha=0.7, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=alt_edges, edge_color="#a44", width=1.2,
        arrows=False, style="dashed", alpha=0.7, ax=ax,
    )

    nb_labels = {n: _short_notebook_label(n) for n in notebook_nodes}
    nx.draw_networkx_labels(G, pos, labels=nb_labels, font_size=7, ax=ax)
    art_labels = {a: _label_for_artifact(a, friendly) for a in artifact_nodes}
    nx.draw_networkx_labels(G, pos, labels=art_labels, font_size=5, font_color="#333", ax=ax)

    legend_handles = [
        mpatches.Patch(color=STAGE_COLORS[s], label=f"Stage {s}") for s in sorted(STAGE_COLORS)
    ]
    legend_handles += [
        mpatches.Patch(color="#fff5cc", label="Artifact"),
        mpatches.Patch(color="#3b6", label="write"),
        mpatches.Patch(color="#48a", label="read"),
        mpatches.Patch(color="#a44", label="alternative-of"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, ncol=2)

    ax.set_axis_off()
    ax.set_title("twitter_ai data pipeline (notebooks ↔ artifacts)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_projection(G: nx.DiGraph, out_path: Path):
    P = project_to_notebook_dag(G)
    pos = _layered_positions(P)

    fig, ax = plt.subplots(figsize=(22, 14))
    nb_nodes = list(P.nodes)
    nb_colors = [STAGE_COLORS.get(P.nodes[n].get("stage", 0), "#cccccc") for n in nb_nodes]
    nx.draw_networkx_nodes(
        P, pos, nodelist=nb_nodes, node_shape="s",
        node_size=3200, node_color=nb_colors, edgecolors="#222", linewidths=1.2, ax=ax,
    )
    nx.draw_networkx_edges(P, pos, edge_color="#666", width=1.0, arrows=True, arrowsize=12, ax=ax)
    nx.draw_networkx_labels(P, pos, labels={n: _short_notebook_label(n) for n in nb_nodes}, font_size=8, ax=ax)

    legend_handles = [
        mpatches.Patch(color=STAGE_COLORS[s], label=f"Stage {s}") for s in sorted(STAGE_COLORS)
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)
    ax.set_axis_off()
    ax.set_title("twitter_ai pipeline — notebook DAG (artifacts folded into edges)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_DOC_TAG_RE = re.compile(
    r"\[(written by|used by)\s+(\d{2})_[A-Za-z_]+/(\d{2})\]"
    r"|\[(\d{2})/(\d{2})\]"
)


def parse_doc_tags():
    """Walk README.md and notebook_setup.md, return {filename: producer_id}.

    Only treats a tag as a write tag when it's `[written by ...]` or the bare
    `[NN/NN]` form used in README. `[used by ...]` is a read tag and ignored.
    """
    sources = [REPO_ROOT / "README.md", NOTEBOOKS_DIR / "notebook_setup.md"]
    tags: dict = {}
    fname_re = re.compile(r"([A-Za-z0-9_\.\-]+\.(?:json|jsonl|pkl|gml|graphml|csv|gz))")
    for s in sources:
        if not s.exists():
            continue
        for line in s.read_text().splitlines():
            # skip lines that use shell-glob notation like {test,full}_X or [_test] —
            # they describe a family of filenames, not a single one
            if "{" in line or "[_" in line:
                continue
            m = _DOC_TAG_RE.search(line)
            if not m:
                continue
            verb = m.group(1) or "written by"
            if verb == "used by":
                continue
            stage = m.group(2) or m.group(4)
            idx = m.group(3) or m.group(5)
            for fname_match in fname_re.finditer(line):
                fname = fname_match.group(1)
                if fname.startswith("_") or fname.startswith("."):
                    continue
                tags[fname] = f"{stage}/{idx}"
    return tags


def validate(G: nx.DiGraph) -> list[str]:
    """Compare doc tags against parser output. Returns drift messages."""
    doc_tags = parse_doc_tags()
    drift: list = []
    parser_writers: dict = {}
    for u, v, d in G.edges(data=True):
        if d.get("direction") != "write":
            continue
        fname = Path(v).name
        # collapse stage/index from notebook id like 02_Processing/02_sanity_check_...
        m = re.match(r"^(\d+)_[A-Za-z_]+/(\d+)_", u)
        if m:
            parser_writers.setdefault(fname, set()).add(f"{m.group(1)}/{m.group(2)}")

    for fname, doc_writer in doc_tags.items():
        if "[" in fname:
            # bracketed names like _test patterns — skip; not exact filenames
            continue
        ours = parser_writers.get(fname)
        if ours is None:
            drift.append(f"docs say {fname} ← {doc_writer}, parser found no producer")
        elif doc_writer not in ours:
            drift.append(f"docs say {fname} ← {doc_writer}, parser says {sorted(ours)}")
    return drift


def cmd_build(args):
    notebooks = discover_notebooks()
    overrides = load_overrides()
    G, warns = build_graph(notebooks, overrides)
    for w in warns:
        print(f"warn: {w}", file=sys.stderr)

    dump_json(G, OUT_JSON)
    render_full(G, overrides.get("friendly_labels", {}), OUT_PNG_FULL)
    render_projection(G, OUT_PNG_NB)

    n_nb = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "notebook")
    n_art = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "artifact")
    n_w = sum(1 for _, _, d in G.edges(data=True) if d.get("direction") == "write")
    n_r = sum(1 for _, _, d in G.edges(data=True) if d.get("direction") == "read")
    print(f"built: {n_nb} notebooks, {n_art} artifacts, {n_w} write edges, {n_r} read edges")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PNG_FULL.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PNG_NB.relative_to(REPO_ROOT)}")


def cmd_validate(args):
    notebooks = discover_notebooks()
    overrides = load_overrides()
    G, _ = build_graph(notebooks, overrides)
    drift = validate(G)
    if not drift:
        print("no drift detected")
        return
    print(f"drift between docs and code ({len(drift)} entries):")
    for d in drift:
        print(f"  - {d}")


def _walk(G, start, fn):
    seen = set()
    out = []
    def rec(n):
        for nbr in fn(n):
            if nbr in seen:
                continue
            seen.add(nbr)
            out.append(nbr)
            rec(nbr)
    rec(start)
    return out


def cmd_query(args):
    notebooks = discover_notebooks()
    overrides = load_overrides()
    G, _ = build_graph(notebooks, overrides)
    target = args.notebook
    if target not in G:
        sys.exit(f"unknown notebook id: {target}\nknown: see {OUT_JSON}")
    if args.command == "downstream":
        nbrs = nx.descendants(G, target)
        label = "downstream of"
    else:
        nbrs = nx.ancestors(G, target)
        label = "upstream of"
    notebooks_only = sorted(n for n in nbrs if G.nodes[n].get("kind") == "notebook")
    artifacts_only = sorted(n for n in nbrs if G.nodes[n].get("kind") == "artifact")
    print(f"{label} {target}:")
    print(f"  notebooks ({len(notebooks_only)}):")
    for n in notebooks_only:
        print(f"    {n}")
    print(f"  artifacts ({len(artifacts_only)}):")
    for a in artifacts_only:
        print(f"    {a}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("build", help="(default) build graph, dump JSON, render PNGs")
    sub.add_parser("validate", help="check docs vs parser output")
    p_d = sub.add_parser("downstream", help="list everything that depends on a notebook")
    p_d.add_argument("notebook")
    p_u = sub.add_parser("upstream", help="list everything a notebook depends on")
    p_u.add_argument("notebook")
    args = ap.parse_args()

    if args.command in (None, "build"):
        cmd_build(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command in ("downstream", "upstream"):
        cmd_query(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
