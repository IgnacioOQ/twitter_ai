import os
import glob
import json
import re

NOTEBOOKS_DIR = 'notebooks'

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
        in_shell_block = False
        shell_commands_buffer = []
        
        # A simple state machine to group consecutive ! commands
        
        for line in cell.get('source', []):
            stripped_line = line.strip()
            
            # Check if line starts with ! (ignoring leading whitespace and comments)
            # Actually, standard colab commands might have #! or just !
            
            # Match !pip, !git, #!pip, #!git
            if re.match(r"^#?\s*!(pip|git)", stripped_line):
                # We found a shell command
                cell_modified = True
                modified = True
                
                # If command is commented out, keep it commented out in the python execution?
                # Best to just extract the actual command.
                
                # Remove leading '#' if it exists so we can process the command
                if stripped_line.startswith('#'):
                    # It was a commented out command. Let's just wrap it as a python comment.
                    new_source.append(f"# Command was commented out: {stripped_line}\n")
                    continue
                
                # It's an active shell command
                command = stripped_line.lstrip('!')
                
                if not in_shell_block:
                    # Start of a new shell block
                    in_shell_block = True
                    new_source.append("import os\n")
                    new_source.append("if not RUNNING_LOCALLY:\n")
                    new_source.append("    print('Running Colab setup shell commands...')\n")
                
                # Add to python block
                new_source.append(f"    os.system('{command}')\n")
            else:
                # Not a shell command
                if in_shell_block:
                    # End of shell block
                    in_shell_block = False
                    new_source.append("else:\n")
                    new_source.append("    print('Running locally: Skipping Colab shell setup.')\n")
                    new_source.append("\n") # formatting
                
                new_source.append(line)
                
        # Handle case where cell ends with a shell block
        if in_shell_block:
            new_source.append("else:\n")
            new_source.append("    print('Running locally: Skipping Colab shell setup.')\n")
            new_source.append("\n")
                
        if cell_modified:
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f"  -> Modified {filepath}")

def main():
    notebooks = glob.glob(f'{NOTEBOOKS_DIR}/**/*.ipynb', recursive=True)
    for nb in notebooks:
        process_notebook(nb)

if __name__ == "__main__":
    main()
