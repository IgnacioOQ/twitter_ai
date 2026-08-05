import os
import json
import glob
from pathlib import Path

# Always analyze relative to this script's directory (notebooks/)
script_dir = Path(__file__).parent
notebooks = glob.glob(str(script_dir / '**' / '*.ipynb'), recursive=True)

results = []

for nb_path in notebooks:
    if '.ipynb_checkpoints' in nb_path:
        continue
        
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        continue
        
    has_time = False
    has_disconnect = False
    has_tqdm = False
    code_cells_count = 0
    time_cells_count = 0
    cells_missing_metadata = 0
    cells_missing_metadata_id = 0
    cells_top_level_id = 0
    # Only nbformat_minor 0 is Colab flavor, where ids belong at metadata.id alone.
    # At 4.5+ a top-level 'id' is required by the schema, so it is not a defect there.
    colab_flavor = nb.get('nbformat_minor') == 0

    for cell in nb.get('cells', []):
        # Colab cell-shape checks: Colab's loader reads cell.metadata.id, so a
        # cell without a 'metadata' key crashes the whole notebook load from
        # GitHub (TypeError: Cannot read properties of undefined (reading 'id')).
        # For nbformat_minor 0 (Colab flavor) ids belong at metadata.id, not as
        # a top-level 'id' field.
        if 'metadata' not in cell:
            cells_missing_metadata += 1
        elif not cell['metadata'].get('id'):
            cells_missing_metadata_id += 1
        if colab_flavor and 'id' in cell:
            cells_top_level_id += 1
        if cell.get('cell_type') == 'code':
            code_cells_count += 1
            source = "".join(cell.get('source', []))
            
            if '%%time' in source:
                has_time = True
                time_cells_count += 1
            
            if 'tqdm' in source or 'progress' in source.lower():
                has_tqdm = True
                
            if 'kernel.disconnect()' in source or 'runtime.unassign()' in source:
                has_disconnect = True
                
    # Collapse state: sections fold to one line only when every section h1 (other
    # than the notebook title) is listed in metadata.colab.collapsed_sections.
    # Colab rewrites this on every save, so it drifts — see notebook_setup.md
    # § Collapsible Sections.
    collapsed = nb.get('metadata', {}).get('colab', {}).get('collapsed_sections') or []
    h1_ids = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        for line in "".join(cell.get('source', [])).split("\n"):
            stripped = line.strip()
            if stripped.startswith('#'):
                if stripped.startswith('# '):
                    h1_ids.append(cell.get('metadata', {}).get('id') or cell.get('id'))
                break
    # The first h1 is the title and is meant to stay expanded.
    section_ids = h1_ids[1:]
    uncollapsed = [i for i in section_ids if i not in collapsed]

    # Get path relative to the notebooks folder for cleaner display
    rel_path = os.path.relpath(nb_path, script_dir)

    results.append({
        'path': rel_path,
        'has_time': has_time,
        'time_cells_count': time_cells_count,
        'code_cells_count': code_cells_count,
        'has_tqdm': has_tqdm,
        'has_disconnect': has_disconnect,
        'cells_missing_metadata': cells_missing_metadata,
        'cells_missing_metadata_id': cells_missing_metadata_id,
        'cells_top_level_id': cells_top_level_id,
        'colab_ok': cells_missing_metadata == 0,
        'section_count': len(section_ids),
        'uncollapsed': len(uncollapsed),
    })

hdr = (f"{'Notebook':<50} | {'%%time':<10} | {'tqdm':<5} | {'disconnect':<10} | "
       f"{'colab_ok':<8} | {'folded':<8}")
print(hdr)
print("-" * len(hdr))
for r in sorted(results, key=lambda x: x['path']):
    time_ratio = f"{r['time_cells_count']}/{r['code_cells_count']}"
    folded = f"{r['section_count'] - r['uncollapsed']}/{r['section_count']}"
    print(f"{r['path']:<50} | {time_ratio:<10} | {str(r['has_tqdm']):<5} | "
          f"{str(r['has_disconnect']):<10} | {str(r['colab_ok']):<8} | {folded:<8}")

# Notebooks that will open expanded (or have no foldable sections at all)
drifted = [r for r in sorted(results, key=lambda x: x['path'])
           if r['uncollapsed'] or r['section_count'] == 0]
if drifted:
    print("\nSections not collapsed by default (see notebook_setup.md § Collapsible Sections):")
    for r in drifted:
        if r['section_count'] == 0:
            print(f"  {r['path']}: no `#` sections — nothing can be folded")
        else:
            print(f"  {r['path']}: {r['uncollapsed']}/{r['section_count']} section(s) "
                  f"missing from collapsed_sections")

# Detail any notebooks that would fail to load in Colab
broken = [r for r in sorted(results, key=lambda x: x['path']) if not r['colab_ok'] or r['cells_top_level_id']]
if broken:
    print("\nColab cell-shape issues (loader reads cell.metadata.id; missing 'metadata' crashes the GitHub load):")
    for r in broken:
        issues = []
        if r['cells_missing_metadata']:
            issues.append(f"{r['cells_missing_metadata']} cell(s) missing 'metadata' (breaks Colab load)")
        if r['cells_missing_metadata_id']:
            issues.append(f"{r['cells_missing_metadata_id']} cell(s) missing 'metadata.id' (not addressable by id)")
        if r['cells_top_level_id']:
            issues.append(f"{r['cells_top_level_id']} cell(s) with top-level 'id' (nbformat_minor 0 keeps ids at metadata.id)")
        print(f"  {r['path']}: " + "; ".join(issues))
