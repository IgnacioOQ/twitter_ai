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
    
    for cell in nb.get('cells', []):
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
        'has_disconnect': has_disconnect
    })

print(f"{'Notebook':<50} | {'%%time':<10} | {'tqdm':<5} | {'disconnect':<10}")
print("-" * 80)
for r in sorted(results, key=lambda x: x['path']):
    time_ratio = f"{r['time_cells_count']}/{r['code_cells_count']}"
    print(f"{r['path']:<50} | {time_ratio:<10} | {str(r['has_tqdm']):<5} | {str(r['has_disconnect']):<10}")
