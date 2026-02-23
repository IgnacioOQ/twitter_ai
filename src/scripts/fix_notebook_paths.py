import os
import glob
import json
import re

NOTEBOOKS_DIR = 'notebooks'

# Typical hardcoded path variations
HARDCODED_PREFIXES = [
    r'/content/drive/My Drive/Colab Projects/AI Public Trust/',
    r'/content/drive/MyDrive/AI Public Trust/',
    r'/content/drive/MyDrive/Colab Projects/AI Public Trust/'
]

# We want to replace hardcoded strings with a standardized parameter approach:
# BASE_PATH = '/content/drive/MyDrive/AI Public Trust'
# data_sets_folder = BASE_PATH + '/Data Sets/'

def process_notebook(filepath):
    print(f"Processing: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        new_source = []
        cell_modified = False
        
        for line in cell.get('source', []):
            original_line = line
            
            # Check if this cell defines the base path, if so, standardize it
            if re.search(r"SHARED_FOLDER_PATH\s*=", line) or re.search(r"BASE\s*=", line) or re.search(r"BASE_PATH\s*=", line):
                # Standardize top level variable
                # Let's not touch the definition cell itself if it's already using a variable like BASE,
                # but we will standardize hardcoded strings later in the line
                pass
            
            # Replace hardcoded prefixes with f-strings or string concatenation if they are inside strings
            for prefix in HARDCODED_PREFIXES:
                # Naive replacement inside string literals
                # Ideally, they'd use a BASE variable. Let's just normalize the prefix first.
                normalized_prefix = '/content/drive/MyDrive/AI Public Trust/'
                if prefix in line and prefix != normalized_prefix:
                    line = line.replace(prefix, normalized_prefix)
                    cell_modified = True
                    modified = True
            
            new_source.append(line)
            
        if cell_modified:
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            # Add back newlines that json.dump drops to match jupyter formatting somewhat
            f.write('\n')
        print(f"  -> Modified {filepath}")

def main():
    notebooks = glob.glob(f'{NOTEBOOKS_DIR}/**/*.ipynb', recursive=True)
    for nb in notebooks:
        process_notebook(nb)

if __name__ == "__main__":
    main()
