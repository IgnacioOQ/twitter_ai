import os
import glob
import json

NOTEBOOKS_DIR = 'notebooks'

SETUP_BLOCK = [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# --- ENVIRONMENT SWITCH ---\n",
    "# Set to True if running on local machine with Google Drive Desktop mounted\n",
    "# Set to False if running in Google Colab cloud\n",
    "RUNNING_LOCALLY = True\n",
    "\n",
    "if RUNNING_LOCALLY:\n",
    "    # Standard macOS path for Google Drive Desktop\n",
    "    BASE_PATH = Path('/Volumes/GoogleDrive/MyDrive/AI Public Trust')\n",
    "else:\n",
    "    # Google Colab cloud path\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    BASE_PATH = Path('/content/drive/MyDrive/AI Public Trust')\n",
    "\n",
    "# Pre-compute critical paths used across notebooks\n",
    "twits_folder = BASE_PATH / 'Raw Data/Twits/'\n",
    "test_folder = BASE_PATH / 'Raw Data/'\n",
    "datasets_folder = BASE_PATH / 'Data Sets'\n",
    "cleanedds_folder = BASE_PATH / 'Data Sets/Cleaned Data'\n",
    "networks_folder = BASE_PATH / 'Data Sets/Networks/'\n",
    "literature_folder = BASE_PATH / 'Literature/'\n",
    "topic_models_folder = BASE_PATH / 'Models/Topic Modeling/'\n"
]

def add_setup_cell(filepath):
    print(f"Injecting into: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Create the new code cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "env_switch_setup"
        },
        "outputs": [],
        "source": SETUP_BLOCK
    }
    
    # We will inject it as the first cell in the notebook, right after the title if it's a markdown cell
    # A cleaner approach is simply to insert it at index 0 or 1.
    cells = nb.get('cells', [])
    
    if len(cells) > 0 and cells[0].get('cell_type') == 'markdown':
        # Insert after the first markdown (which is usually the title)
        cells.insert(1, new_cell)
    else:
        # Insert at the very top
        cells.insert(0, new_cell)
        
    nb['cells'] = cells
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        f.write('\n')
        
    print(f"  -> Done.")

def main():
    notebooks = glob.glob(f'{NOTEBOOKS_DIR}/**/*.ipynb', recursive=True)
    for nb in notebooks:
        add_setup_cell(nb)

if __name__ == "__main__":
    main()
