#!/usr/bin/env python3
"""
One-off patch: fix RAM exhaustion in 00_hitl_data_preparation, preserve `ref_id`
on retweets so 03/04 can do the orphan lookup without re-extracting the nested
`referenced_tweets_dictionary`, and fix two latent bugs in 00:
  - the partition-slicing cell (`llm_bootstrap_df`, `base_df`, ...) was missing
  - the manifest cell was duplicated and the surviving copy dropped `retweets`

Run once from the repo root; safe to delete afterward.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2] / 'notebooks' / '05_Classifiers'


# ── helpers (same shape as patch_classifiers_dual_dataset.py) ────────────────

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


def find_cell_idx(nb, marker, start=0):
    for i in range(start, len(nb['cells'])):
        if marker in cell_src(nb['cells'][i]):
            return i
    raise ValueError(f'Marker not found: {marker!r}')


def find_all_cell_idx(nb, marker):
    return [i for i, c in enumerate(nb['cells']) if marker in cell_src(c)]


# ── notebook 00: streaming load + missing slicing + manifest dedupe ──────────

CELL5_NEW = """USE_TEST_DATA = False  # True → AItrust_twits_pruned_dict_test.json (development); False → full dataset

_PRUNED_DICT_NAMES = {
    'AI':  ('AItrust_twits_pruned_dict_test.json'    if USE_TEST_DATA else 'AItrust_twits_pruned_dict.json'),
    'Art': ('AItrust_Art_pruned_twit_dict_test.json' if USE_TEST_DATA else 'AItrust_Art_pruned_twit_dict.json'),
}
assert DATASET_TYPE in _PRUNED_DICT_NAMES, f'Unknown DATASET_TYPE {DATASET_TYPE!r}. Choose \"AI\" or \"Art\".'
PRUNED_DICT_NAME = _PRUNED_DICT_NAMES[DATASET_TYPE]
PRUNED_DICT_PATH = cleanedds_folder / PRUNED_DICT_NAME

assert PRUNED_DICT_PATH.exists(), (
    f'Pruned dict not found: {PRUNED_DICT_PATH}. '
    f'Run notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb first.'
)

# Stream the JSONL line-by-line and keep only the 7 fields the HITL pipeline needs.
# `pd.read_json(..., lines=True)` materialises every column (including nested
# public_metrics, entities, ...) as object dtype, blowing past a 51 GB Colab
# session on a 15+ GB JSONL. Matches the streaming pattern in
# 02_Processing/02_sanity_check_and_network_generation.ipynb and
# 03_Analysis_and_Modeling/02_extract_examples.ipynb.
import json
import tqdm

records = []
print(f'Streaming {PRUNED_DICT_PATH}...')
with open(PRUNED_DICT_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm.tqdm(f, desc='Reading tweets', unit=' tweets'):
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        pm = t.get('public_metrics') or {}
        rt = t.get('referenced_tweets_dictionary')
        ref_id = None
        if isinstance(rt, dict):
            rid = rt.get('id')
            if rid is not None:
                ref_id = str(rid)
        records.append({
            'id':             str(t.get('id', '')),
            'text':           str(t.get('text', '')),
            'processed_text': str(t.get('processed_text', '')),
            'type':           str(t.get('type', '')),
            'likes':          int(pm.get('like_count', 0) or 0),
            'retweets':       int(pm.get('retweet_count', 0) or 0),
            'ref_id':         ref_id,
        })

df = pd.DataFrame(records)
del records  # free the intermediate list before continuing
df['type']     = df['type'].astype('category')
df['likes']    = df['likes'].astype('int32')
df['retweets'] = df['retweets'].astype('int32')
print(f'Loaded {len(df):,} tweets')
print(f'Columns: {list(df.columns)}')
print(f'Memory:  {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')"""


CELL7_NEW = """# Cell 5 already streamed the columns we need and applied compact dtypes;
# only the label placeholder columns are added here.
df['predicted_label'] = pd.NA
df['human_label']     = pd.NA

print(f'Normalised: {len(df):,} tweets')
print('Type breakdown:')
print(df['type'].value_counts().to_string())"""


CELL9_NEW = """RETWEET_TYPE = 'retweeted'

# ref_id is kept on retweets (used for the orphan lookup in 03 and 04) and
# dropped from non-retweets where it would just waste memory.
retweets_df = df[df['type'] == RETWEET_TYPE].copy().reset_index(drop=True)
df          = df[df['type'] != RETWEET_TYPE].drop(columns=['ref_id']).reset_index(drop=True)

print(f'Retweets split out:   {len(retweets_df):>9,}  → retweets_dataset.pkl (label by lookup at merge time)')
print(f'Partitionable corpus: {len(df):>9,}  → originals + replies + quotes (used for train/HITL/inference)')"""


PARTITION_SLICE_CELL = """LLM_BOOTSTRAP_SIZE = 10_000
BASE_SIZE          = 100_000
BATCH_SIZE         = 50_000
N_HITL_BATCHES     = 4
HITL_SIZE          = N_HITL_BATCHES * BATCH_SIZE  # 4 × 50k = 200k

needed = LLM_BOOTSTRAP_SIZE + BASE_SIZE + HITL_SIZE
assert len(df) >= needed, (
    f'Not enough partitionable tweets ({len(df):,}) for the requested partition sizes ({needed:,}).'
)

a = LLM_BOOTSTRAP_SIZE
b = a + BASE_SIZE
c = b + HITL_SIZE
llm_bootstrap_df = df.iloc[:a].reset_index(drop=True)
base_df          = df.iloc[a:b].reset_index(drop=True)
hitl_df          = df.iloc[b:c].reset_index(drop=True)
inference_df     = df.iloc[c:].reset_index(drop=True)

print(f'  LLM bootstrap: n={len(llm_bootstrap_df):>10,}')
print(f'  Base:          n={len(base_df):>10,}')
print(f'  HITL:          n={len(hitl_df):>10,}')
print(f'  Inference:     n={len(inference_df):>10,}')"""


CELL15_NEW = """partitioned_folder.mkdir(parents=True, exist_ok=True)
hitl_folder.mkdir(parents=True, exist_ok=True)

retweets_df.to_pickle(partitioned_folder / 'retweets_dataset.pkl')
print(f'Saved retweets_dataset.pkl       ({len(retweets_df):>10,} tweets)')

llm_bootstrap_df.to_pickle(partitioned_folder / 'llm_bootstrap_dataset.pkl')
print(f'Saved llm_bootstrap_dataset.pkl  ({len(llm_bootstrap_df):>10,} tweets)')

base_df.to_pickle(partitioned_folder / 'base_dataset.pkl')
print(f'Saved base_dataset.pkl           ({len(base_df):>10,} tweets)')

inference_df.to_pickle(partitioned_folder / 'inference_dataset.pkl')
print(f'Saved inference_dataset.pkl      ({len(inference_df):>10,} tweets)')

n_batches = int(np.ceil(len(hitl_df) / BATCH_SIZE)) if len(hitl_df) else 0
for i in range(n_batches):
    chunk = hitl_df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reset_index(drop=True)
    out   = partitioned_folder / f'hitl_pending_batch_{i+1:02d}.pkl'
    chunk.to_pickle(out)
    print(f'Saved {out.name} ({len(chunk):,} tweets)')"""


MANIFEST_NEW = """def _ids(d):
    return d['id'].astype(str).tolist() if 'id' in d.columns else d.index.astype(str).tolist()

partition_ids = {
    'llm_bootstrap': _ids(llm_bootstrap_df),
    'base':          _ids(base_df),
    'inference':     _ids(inference_df),
    'retweets':      _ids(retweets_df),
}

n_batches = int(np.ceil(len(hitl_df) / BATCH_SIZE)) if len(hitl_df) else 0
for i in range(n_batches):
    start = i * BATCH_SIZE
    stop  = min(start + BATCH_SIZE, len(hitl_df))
    partition_ids[f'hitl_batch_{i+1:02d}'] = _ids(hitl_df.iloc[start:stop])

seen = set()
for name, ids in partition_ids.items():
    s = set(ids)
    overlap = s & seen
    assert not overlap, f'Partition {name!r} overlaps existing partitions on {len(overlap)} ids'
    seen |= s
    print(f'  {name:>16}: {len(ids):>10,} ids')
print(f'  {\"TOTAL\":>16}: {len(seen):>10,} unique ids — all partitions disjoint')

manifest_path = partitioned_folder / 'partition_ids.pkl'
with open(manifest_path, 'wb') as f:
    pickle.dump(partition_ids, f)
print(f'Wrote → {manifest_path}')"""


def patch_00():
    path = NB_DIR / '00_hitl_data_preparation.ipynb'
    nb = load_nb(path)

    # Idempotency: if the streaming load is already in place, this patch was applied.
    if any('Streaming' in cell_src(c) and 'tqdm.tqdm(f' in cell_src(c) for c in nb['cells']):
        print(f'  skipped: 00_hitl_data_preparation.ipynb (already patched)')
        return

    # 1. Streaming JSONL load (replaces the pd.read_json that OOMs)
    i = find_cell_idx(nb, 'PRUNED_DICT_PATH = cleanedds_folder / PRUNED_DICT_NAME')
    nb['cells'][i]['source'] = make_source(CELL5_NEW)

    # 2. Simplified normalisation (most work now happens during streaming)
    i = find_cell_idx(nb, "df = df[['id', 'text', 'processed_text', 'type', 'likes', 'retweets']].copy()")
    nb['cells'][i]['source'] = make_source(CELL7_NEW)

    # 3. Retweet split — preserve ref_id on retweets, drop it from non-retweets
    i = find_cell_idx(nb, "RETWEET_TYPE = 'retweeted'")
    nb['cells'][i]['source'] = make_source(CELL9_NEW)

    # 4. Insert the missing partition-slicing cell immediately after the shuffle.
    shuffle_idx = find_cell_idx(nb, "SAMPLING_ALPHA = 0.5")
    nb['cells'].insert(shuffle_idx + 1, code_cell(PARTITION_SLICE_CELL))

    # 5. Save cell — no more `hitl_batches` dict (manifest reads hitl_df slices directly)
    i = find_cell_idx(nb, "llm_bootstrap_df.to_pickle(partitioned_folder / 'llm_bootstrap_dataset.pkl')")
    nb['cells'][i]['source'] = make_source(CELL15_NEW)

    # 6. Manifest — keep one canonical version (with retweets), delete the duplicate
    manifest_idxs = [
        idx for idx, c in enumerate(nb['cells'])
        if c['cell_type'] == 'code'
        and 'partition_ids = {' in cell_src(c)
        and "_ids(llm_bootstrap_df)" in cell_src(c)
    ]
    assert manifest_idxs, 'Expected at least one manifest cell'
    nb['cells'][manifest_idxs[0]]['source'] = make_source(MANIFEST_NEW)
    for idx in sorted(manifest_idxs[1:], reverse=True):
        del nb['cells'][idx]

    save_nb(path, nb)


# ── notebook 03: prefer the new ref_id column for the orphan lookup ──────────

OLD_03 = """if 'referenced_tweets_dictionary' not in retweets_df.columns:
    print('WARNING: retweets_dataset.pkl has no referenced_tweets_dictionary column.')
    print('         00_hitl_data_preparation.ipynb dropped it during normalisation.')
    print('         Every retweet will fall through to the no-reference fallback in Pass 3.')
    retweets_df['ref_id'] = None
else:
    retweets_df['ref_id'] = retweets_df['referenced_tweets_dictionary'].apply(_ref_id)"""

NEW_03 = """# Prefer the precomputed `ref_id` column (written by 00 since the RAM fix).
# Fall back to extracting from `referenced_tweets_dictionary` for older partitions.
if 'ref_id' in retweets_df.columns:
    retweets_df['ref_id'] = retweets_df['ref_id'].where(retweets_df['ref_id'].notna(), None)
elif 'referenced_tweets_dictionary' in retweets_df.columns:
    retweets_df['ref_id'] = retweets_df['referenced_tweets_dictionary'].apply(_ref_id)
else:
    print('WARNING: retweets_dataset.pkl has no ref_id / referenced_tweets_dictionary column.')
    print('         Every retweet will fall through to the no-reference fallback in Pass 3.')
    retweets_df['ref_id'] = None"""


def patch_03():
    path = NB_DIR / '03_final_inference.ipynb'
    nb = load_nb(path)
    i = find_cell_idx(nb, "if 'referenced_tweets_dictionary' not in retweets_df.columns:")
    s = cell_src(nb['cells'][i])
    assert OLD_03 in s, f'Old ref_id block not found in 03 cell {i}'
    nb['cells'][i]['source'] = make_source(s.replace(OLD_03, NEW_03))
    save_nb(path, nb)


# ── notebook 04: same lookup-preference change against new_retweets ──────────

OLD_04 = """if 'referenced_tweets_dictionary' not in new_retweets.columns:
    print('WARNING: full corpus has no referenced_tweets_dictionary column.')
    print('         Every retweet will fall through to the no-reference fallback in Pass 3.')
    new_retweets['ref_id'] = None
else:
    new_retweets['ref_id'] = new_retweets['referenced_tweets_dictionary'].apply(_ref_id)"""

NEW_04 = """# Prefer the precomputed `ref_id` column where available (e.g. corpus produced
# by an updated 02_Processing step). Fall back to extracting from the nested dict.
if 'ref_id' in new_retweets.columns:
    new_retweets['ref_id'] = new_retweets['ref_id'].where(new_retweets['ref_id'].notna(), None)
elif 'referenced_tweets_dictionary' in new_retweets.columns:
    new_retweets['ref_id'] = new_retweets['referenced_tweets_dictionary'].apply(_ref_id)
else:
    print('WARNING: full corpus has no ref_id / referenced_tweets_dictionary column.')
    print('         Every retweet will fall through to the no-reference fallback in Pass 3.')
    new_retweets['ref_id'] = None"""


def patch_04():
    path = NB_DIR / '04_full_dataset_inference.ipynb'
    nb = load_nb(path)
    i = find_cell_idx(nb, "if 'referenced_tweets_dictionary' not in new_retweets.columns:")
    s = cell_src(nb['cells'][i])
    assert OLD_04 in s, f'Old ref_id block not found in 04 cell {i}'
    nb['cells'][i]['source'] = make_source(s.replace(OLD_04, NEW_04))
    save_nb(path, nb)


if __name__ == '__main__':
    print('Patching 05_Classifiers notebooks (RAM + ref_id + latent bugs)...')
    patch_00()
    patch_03()
    patch_04()
    print('Done.')
