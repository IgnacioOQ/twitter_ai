#!/usr/bin/env python3
"""
One-off patch: add DATASET_TYPE config and dual-dataset path support to all
05_Classifiers notebooks. Run once from the repo root; safe to delete afterward.

Changes per notebook
--------------------
00  add DATASET_TYPE + dual partitioned_folder, dual PRUNED_DICT_NAME lookup,
    EXPORT_HUMAN_SEED guard on optional export, Disconnect from Runtime footer
01  add DATASET_TYPE + partitioned_folder; fix INPUT_PATH (was hitl_folder)
02  add DATASET_TYPE + partitioned_folder; fix NEXT_BATCH_PATH (was hitl_folder)
03  add DATASET_TYPE + update partitioned_folder
04  add DATASET_TYPE + update partitioned_folder + dual FULL_CORPUS_PATH lookup
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2] / 'notebooks' / '05_Classifiers'


# ── helpers ───────────────────────────────────────────────────────────────────

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    with open(path, 'r', encoding='utf-8') as f:
        json.load(f)
    print(f'  saved: {path.name}')


def make_source(text):
    lines = text.split('\n')
    return [ln + '\n' for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def code_cell(text):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': make_source(text)}


def md_cell(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': make_source(text)}


def cell_src(cell):
    return ''.join(cell['source'])


def find_cell_idx(nb, marker):
    for i, c in enumerate(nb['cells']):
        if marker in cell_src(c):
            return i
    raise ValueError(f'Marker not found in any cell: {marker!r}')


def patch_substring(nb, marker, old, new):
    """Replace one occurrence of `old` inside the first cell containing `marker`."""
    i = find_cell_idx(nb, marker)
    s = cell_src(nb['cells'][i])
    if old not in s:
        raise ValueError(f'old string not found in cell {i}:\n  {old!r}')
    nb['cells'][i]['source'] = make_source(s.replace(old, new, 1))


def replace_cell_src(nb, marker, new_text):
    """Replace the entire source of the first cell containing `marker`."""
    i = find_cell_idx(nb, marker)
    nb['cells'][i]['source'] = make_source(new_text)


# ── shared constants ──────────────────────────────────────────────────────────

DATASET_TYPE_BLOCK = (
    "\n\n"
    "# --- DATASET TYPE ---\n"
    "# 'AI'  → AItrust_twits_pruned_dict.json    → Partitioned Data/AI Data/\n"
    "# 'Art' → AItrust_Art_pruned_twit_dict.json → Partitioned Data/Art Data/\n"
    "DATASET_TYPE = 'AI'"
)

DISCONNECT_CODE = (
    "if not RUNNING_LOCALLY:\n"
    "    from google.colab import runtime\n"
    "    runtime.unassign()"
)

# The marker that every notebook's cell 1 contains, used to locate that cell.
CELL1_MARKER = "RUNNING_LOCALLY = False\n\nif RUNNING_LOCALLY:"


# ── 00_hitl_data_preparation ─────────────────────────────────────────────────

def patch_00():
    path = NB_DIR / '00_hitl_data_preparation.ipynb'
    nb = load_nb(path)

    # 1. Insert DATASET_TYPE block right after "RUNNING_LOCALLY = False"
    patch_substring(nb, CELL1_MARKER,
        "RUNNING_LOCALLY = False",
        "RUNNING_LOCALLY = False" + DATASET_TYPE_BLOCK)

    # 2. Append dataset-type subdirectory to partitioned_folder
    patch_substring(nb,
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data' / f'{DATASET_TYPE} Data'")

    # 3. Replace single PRUNED_DICT_NAME line with dataset-type lookup dict
    old_name = (
        "PRUNED_DICT_NAME = 'AItrust_twits_pruned_dict_test.json' "
        "if USE_TEST_DATA else 'AItrust_twits_pruned_dict.json'"
    )
    new_name = (
        "_PRUNED_DICT_NAMES = {\n"
        "    'AI':  ('AItrust_twits_pruned_dict_test.json'    if USE_TEST_DATA else 'AItrust_twits_pruned_dict.json'),\n"
        "    'Art': ('AItrust_Art_pruned_twit_dict_test.json' if USE_TEST_DATA else 'AItrust_Art_pruned_twit_dict.json'),\n"
        "}\n"
        "assert DATASET_TYPE in _PRUNED_DICT_NAMES, "
        "f'Unknown DATASET_TYPE {DATASET_TYPE!r}. Choose \"AI\" or \"Art\".'\n"
        "PRUNED_DICT_NAME = _PRUNED_DICT_NAMES[DATASET_TYPE]"
    )
    patch_substring(nb, old_name, old_name, new_name)

    # 4. Replace the optional export code cell with an EXPORT_HUMAN_SEED guard
    replace_cell_src(nb,
        "seed = base_df.sample(n=min(10_000",
        "EXPORT_HUMAN_SEED = False  # set True to export a 10k human-seed CSV from the Base partition\n"
        "\n"
        "if EXPORT_HUMAN_SEED:\n"
        "    seed = base_df.sample(n=min(10_000, len(base_df)), random_state=42).copy()\n"
        "    for col in ('text', 'processed_text'):\n"
        "        if col in seed.columns:\n"
        "            seed[col] = seed[col].astype(str).str.replace('\\n', ' ', regex=False)\n"
        "\n"
        "    out_path = hitl_folder / 'hitl_review_batch_00.csv'\n"
        "    seed.to_csv(out_path, index=False)\n"
        "    print(f'Saved → {out_path}')\n"
        "else:\n"
        "    print('EXPORT_HUMAN_SEED=False — skipped. Set True to export the human seed batch.')"
    )

    # 5. Append Disconnect from Runtime section
    nb['cells'].append(md_cell("## Disconnect from Runtime"))
    nb['cells'].append(code_cell(DISCONNECT_CODE))

    save_nb(path, nb)


# ── 01_llm_bootstrap_labelling ───────────────────────────────────────────────

def patch_01():
    path = NB_DIR / '01_llm_bootstrap_labelling.ipynb'
    nb = load_nb(path)

    # 1. Insert DATASET_TYPE block
    patch_substring(nb, CELL1_MARKER,
        "RUNNING_LOCALLY = False",
        "RUNNING_LOCALLY = False" + DATASET_TYPE_BLOCK)

    # 2. Add partitioned_folder definition (not present in this notebook originally)
    patch_substring(nb,
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\nhitl_folder",
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\nhitl_folder",
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\n"
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data' / f'{DATASET_TYPE} Data'\n"
        "hitl_folder")

    # 3. Fix INPUT_PATH: the bootstrap pkl lives in partitioned_folder, not hitl_folder
    patch_substring(nb,
        "INPUT_PATH        = hitl_folder / 'llm_bootstrap_dataset.pkl'",
        "INPUT_PATH        = hitl_folder / 'llm_bootstrap_dataset.pkl'",
        "INPUT_PATH        = partitioned_folder / 'llm_bootstrap_dataset.pkl'")

    save_nb(path, nb)


# ── 02_hitl_training_loop ────────────────────────────────────────────────────

def patch_02():
    path = NB_DIR / '02_hitl_training_loop.ipynb'
    nb = load_nb(path)

    # 1. Insert DATASET_TYPE block
    patch_substring(nb, CELL1_MARKER,
        "RUNNING_LOCALLY = False",
        "RUNNING_LOCALLY = False" + DATASET_TYPE_BLOCK)

    # 2. Add partitioned_folder definition (not present in this notebook originally)
    patch_substring(nb,
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\nhitl_folder",
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\nhitl_folder",
        "classifiers_folder.mkdir(parents=True, exist_ok=True)\n"
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data' / f'{DATASET_TYPE} Data'\n"
        "hitl_folder")

    # 3. Fix NEXT_BATCH_PATH: pending batches live in partitioned_folder, not hitl_folder
    patch_substring(nb,
        "NEXT_BATCH_PATH = hitl_folder /",
        "NEXT_BATCH_PATH = hitl_folder /",
        "NEXT_BATCH_PATH = partitioned_folder /")

    save_nb(path, nb)


# ── 03_final_inference ───────────────────────────────────────────────────────

def patch_03():
    path = NB_DIR / '03_final_inference.ipynb'
    nb = load_nb(path)

    # 1. Insert DATASET_TYPE block
    patch_substring(nb, CELL1_MARKER,
        "RUNNING_LOCALLY = False",
        "RUNNING_LOCALLY = False" + DATASET_TYPE_BLOCK)

    # 2. Append dataset-type subdirectory to partitioned_folder
    patch_substring(nb,
        "partitioned_folder = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder = cleanedds_folder / 'Partitioned Data' / f'{DATASET_TYPE} Data'")

    save_nb(path, nb)


# ── 04_full_dataset_inference ────────────────────────────────────────────────

def patch_04():
    path = NB_DIR / '04_full_dataset_inference.ipynb'
    nb = load_nb(path)

    # 1. Insert DATASET_TYPE block
    patch_substring(nb, CELL1_MARKER,
        "RUNNING_LOCALLY = False",
        "RUNNING_LOCALLY = False" + DATASET_TYPE_BLOCK)

    # 2. Append dataset-type subdirectory to partitioned_folder
    patch_substring(nb,
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data'",
        "partitioned_folder    = cleanedds_folder / 'Partitioned Data' / f'{DATASET_TYPE} Data'")

    # 3. Replace single FULL_CORPUS_PATH line with dataset-type lookup dict
    old_corpus = "FULL_CORPUS_PATH = datasets_folder / 'AItrust_twits_dict.json'"
    new_corpus = (
        "_FULL_CORPUS_PATHS = {\n"
        "    'AI':  datasets_folder / 'AItrust_twits_dict.json',\n"
        "    'Art': datasets_folder / 'AItrust_Art_twit_dict.json',  # TODO: confirm Art corpus path\n"
        "}\n"
        "FULL_CORPUS_PATH = _FULL_CORPUS_PATHS[DATASET_TYPE]"
    )
    patch_substring(nb, old_corpus, old_corpus, new_corpus)

    save_nb(path, nb)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Patching 05_Classifiers notebooks...')
    patch_00()
    patch_01()
    patch_02()
    patch_03()
    patch_04()
    print('Done. Re-run src/scripts/pipeline_graph.py to regenerate docs/pipeline_graph.*')
