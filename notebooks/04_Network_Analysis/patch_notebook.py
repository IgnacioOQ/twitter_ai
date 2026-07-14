import json
import re
import numpy as np

file_path = '/Users/ignacio/Documents/VS Code/GitHub Repositories/twitter_ai/notebooks/04_Network_Analysis/02_network_visualization.ipynb'

with open(file_path, 'r') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if '"linlog":  dict(lin_log_mode=True,  outbound_attraction_distribution=True,' in line:
                source[i+1] = source[i+1].replace('scaling_ratio=2.0', 'scaling_ratio=5.0')
            if 'center = np.median(xy, axis=0)' in line:
                source[i] = '    center = np.mean(xy, axis=0)\n'
            
            # The size change spans multiple lines, we can replace it directly if we match the start
            # But the simplest is string replacement on the joined source.

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'cg_nodes["size"] = np.sqrt(cg_nodes["in_degree"]) * 2 + 1' in source_str:
            new_size_logic = """        n_tiers = 7
        tier_sizes = np.array([3, 6, 12, 22, 35, 55, 85])
        log_deg = np.log1p(cg_nodes["in_degree"])
        norm_deg = (log_deg - log_deg.min()) / (log_deg.max() - log_deg.min() + 1e-9)
        tier_indices = np.clip(np.floor(norm_deg * n_tiers).astype(int), 0, n_tiers - 1)
        cg_nodes["size"] = tier_sizes[tier_indices]
"""
            source_str = source_str.replace('        cg_nodes["size"] = np.sqrt(cg_nodes["in_degree"]) * 2 + 1\n', new_size_logic)
            
            # recreate source lines
            # In nbformat, lines typically end with \n except the last one
            new_source = []
            lines = source_str.split('\n')
            for i, l in enumerate(lines):
                if i < len(lines) - 1:
                    new_source.append(l + '\n')
                else:
                    if l: # if last line is not empty
                        new_source.append(l)
            cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(notebook, f, indent=1)
