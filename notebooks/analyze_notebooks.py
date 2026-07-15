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
    cells_top_level_id = 0

    for cell in nb.get('cells', []):
        # Colab cell-shape checks: Colab's loader reads cell.metadata.id, so a
        # cell without a 'metadata' key crashes the whole notebook load from
        # GitHub (TypeError: Cannot read properties of undefined (reading 'id')).
        # For nbformat_minor 0 (Colab flavor) ids belong at metadata.id, not as
        # a top-level 'id' field.
        if 'metadata' not in cell:
            cells_missing_metadata += 1
        if 'id' in cell:
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
        'cells_top_level_id': cells_top_level_id,
        'colab_ok': cells_missing_metadata == 0
    })

print(f"{'Notebook':<50} | {'%%time':<10} | {'tqdm':<5} | {'disconnect':<10} | {'colab_ok':<8}")
print("-" * 95)
for r in sorted(results, key=lambda x: x['path']):
    time_ratio = f"{r['time_cells_count']}/{r['code_cells_count']}"
    print(f"{r['path']:<50} | {time_ratio:<10} | {str(r['has_tqdm']):<5} | {str(r['has_disconnect']):<10} | {str(r['colab_ok']):<8}")

# Detail any notebooks that would fail to load in Colab
broken = [r for r in sorted(results, key=lambda x: x['path']) if not r['colab_ok'] or r['cells_top_level_id']]
if broken:
    print("\nColab cell-shape issues (loader reads cell.metadata.id; missing 'metadata' crashes the GitHub load):")
    for r in broken:
        issues = []
        if r['cells_missing_metadata']:
            issues.append(f"{r['cells_missing_metadata']} cell(s) missing 'metadata' (breaks Colab load)")
        if r['cells_top_level_id']:
            issues.append(f"{r['cells_top_level_id']} cell(s) with top-level 'id' (Colab flavor keeps ids at metadata.id)")
        print(f"  {r['path']}: " + "; ".join(issues))
