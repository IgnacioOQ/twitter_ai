"""
create_notebook.py
──────────────────
Helper script for creating valid Jupyter notebooks (.ipynb) programmatically.

USAGE
─────
1. Edit the `cells` list and `OUTPUT_PATH` at the bottom of this file.
2. Run:  python3 notebooks/create_notebook.py
3. Open the generated .ipynb in VS Code or JupyterLab.

WHY THIS EXISTS
───────────────
.ipynb files are JSON. Hand-writing JSON with embedded Python source code is
error-prone (escaping, newlines, commas). This script uses Python's json module
to guarantee valid output and validates the file on every run.
"""

import json
import os
from pathlib import Path


# ── Cell builders ─────────────────────────────────────────────────────────────

def make_source(text: str) -> list:
    """
    Convert a plain Python multiline string into the list-of-strings format
    required by the .ipynb spec.

    Rules:
      - Each line except the last must end with '\\n'.
      - The last line must NOT end with '\\n'.
      - If the string ends with a newline the last element is dropped
        (empty trailing entry is not valid).
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    result = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return result


def md(text: str) -> dict:
    """Create a Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(text),
    }


def code(text: str) -> dict:
    """Create a Code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_source(text),
    }


# ── Notebook writer ────────────────────────────────────────────────────────────

def write_notebook(path: str, cells: list) -> None:
    """
    Serialise `cells` into a valid .ipynb file at `path`.
    Raises json.JSONDecodeError if the resulting file cannot be parsed back.
    """
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.8",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    # ── Validate round-trip ──────────────────────────────────────────────────
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)  # will raise if JSON is malformed

    print(f"✅  Written and validated: {path}")


# ── Standard setup cell (Cell 1 per notebook_setup.md) ────────────────────────

def setup_cell(extra_paths: str = "") -> str:
    """
    Return the canonical Cell 1 source string.
    Pass additional path definitions as `extra_paths`.
    """
    base = """\
import os
from pathlib import Path
import sys

# --- ENVIRONMENT SWITCH ---
# True  → local machine with Google Drive Desktop mounted
# False → Google Colab cloud
RUNNING_LOCALLY = False

if RUNNING_LOCALLY:
    _REPO_ROOT = str(Path(os.getcwd()).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    BASE_PATH = Path('/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust')
else:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/My Drive/Colab Projects/AI Public Trust')

# Pre-compute critical paths
twits_folder        = BASE_PATH / 'Raw Data/Twits/'
test_folder         = BASE_PATH / 'Raw Data/'
datasets_folder     = BASE_PATH / 'Data Sets'
cleanedds_folder    = BASE_PATH / 'Data Sets/Cleaned Data'
networks_folder     = BASE_PATH / 'Data Sets/Networks/'
literature_folder   = BASE_PATH / 'Literature/'
topic_models_folder = BASE_PATH / 'Models/Topic Modeling/'"""
    if extra_paths:
        return base + "\n" + extra_paths
    return base


# ════════════════════════════════════════════════════════════════════════════════
# HITL CLASSIFIER NOTEBOOKS
# Run: python3 notebooks/create_notebook.py
# ════════════════════════════════════════════════════════════════════════════════

NB_DIR = "notebooks/05_Classifiers"

EXTRA_PATHS = "hitl_folder = datasets_folder / 'Classifiers_Data' / 'HITL'"

# ── Notebook 1: Data Preparation ─────────────────────────────────────────────

nb1_cells = [
    md("# 01 - HITL Data Preparation\n\n"
       "Splits the **cleaned** tweets dataset into three partitions:\n"
       "- **Base** (~100 000 tweets): source for initial ground-truth labelling\n"
       "- **HITL batches** (~200 000 tweets, 4 × 50 000): for iterative human review\n"
       "- **Final Inference** (remainder): classified by the final model in notebook 03\n\n"
       "Run this notebook **once** at the start of the project."),

    code(setup_cell(EXTRA_PATHS)),

    code("import os\n"
         "if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "from pathlib import Path"),

    code("CLEANED_DATA_PATH = cleanedds_folder / 'cleaned_tweets.pkl'\n"
         "print(f'Loading from {CLEANED_DATA_PATH}')\n"
         "if CLEANED_DATA_PATH.suffix == '.pkl':\n"
         "    df = pd.read_pickle(CLEANED_DATA_PATH)\n"
         "else:\n"
         "    df = pd.read_csv(CLEANED_DATA_PATH)\n"
         "print(f'Loaded {len(df):,} tweets')"),

    md("## Normalise Columns"),

    code("if 'public_metrics.like_count' in df.columns:\n"
         "    df['likes']    = df['public_metrics.like_count']\n"
         "    df['retweets'] = df['public_metrics.retweet_count']\n"
         "elif 'like_count' in df.columns:\n"
         "    df['likes']    = df['like_count']\n"
         "    df['retweets'] = df['retweet_count']\n"
         "else:\n"
         "    df['likes'] = df['retweets'] = 0\n"
         "\n"
         "if 'tweet_id' in df.columns:\n"
         "    df['id'] = df['tweet_id']\n"
         "\n"
         "keep = [c for c in ['id', 'text', 'likes', 'retweets'] if c in df.columns]\n"
         "df = df[keep].copy()\n"
         "df['predicted_label'] = np.nan\n"
         "df['human_label']     = np.nan\n"
         "\n"
         "df = df.sample(frac=1, random_state=42).reset_index(drop=True)\n"
         "print(df.head())"),

    md("## Partition the Dataset"),

    code("BASE_SIZE = 100_000\n"
         "HITL_SIZE = 200_000\n"
         "\n"
         "if len(df) < BASE_SIZE + HITL_SIZE:\n"
         "    print('Warning: dataset smaller than intended splits; adjusting.')\n"
         "    BASE_SIZE = min(len(df), BASE_SIZE)\n"
         "    HITL_SIZE = min(len(df) - BASE_SIZE, HITL_SIZE)\n"
         "\n"
         "base_df      = df.iloc[:BASE_SIZE].copy()\n"
         "hitl_df      = df.iloc[BASE_SIZE:BASE_SIZE + HITL_SIZE].copy()\n"
         "inference_df = df.iloc[BASE_SIZE + HITL_SIZE:].copy()\n"
         "\n"
         "print(f'Base: {len(base_df):,}  HITL: {len(hitl_df):,}  Inference: {len(inference_df):,}')"),

    md("## Save Partitions"),

    code("hitl_folder.mkdir(parents=True, exist_ok=True)\n"
         "\n"
         "base_df.to_pickle(hitl_folder / 'base_dataset.pkl')\n"
         "inference_df.to_pickle(hitl_folder / 'inference_dataset.pkl')\n"
         "\n"
         "BATCH_SIZE = 50_000\n"
         "n_batches  = int(np.ceil(len(hitl_df) / BATCH_SIZE))\n"
         "for i in range(n_batches):\n"
         "    chunk = hitl_df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]\n"
         "    out   = hitl_folder / f'hitl_pending_batch_{i+1:02d}.pkl'\n"
         "    chunk.to_pickle(out)\n"
         "    print(f'Saved {out.name} ({len(chunk):,} tweets)')"),

    md("## Export Iteration-0 Review Batch\n\n"
       "No model exists yet, so we export 10 000 random tweets as the ground-truth seed.\n"
       "Label the `human_label` column and save the file before running notebook 02."),

    code("seed = base_df.sample(n=min(10_000, len(base_df)), random_state=42).copy()\n"
         "if 'text' in seed.columns:\n"
         "    seed['text'] = seed['text'].astype(str).str.replace('\\n', ' ', regex=False)\n"
         "\n"
         "out_path = hitl_folder / 'hitl_review_batch_00.csv'\n"
         "seed.to_csv(out_path, index=False)\n"
         "print(f'Saved → {out_path}')"),
]

# ── Notebook 2: Active Learning Loop ─────────────────────────────────────────

nb2_cells = [
    md("# 02 - HITL Active Learning Loop\n\n"
       "Re-run this notebook after each human labelling round.\n"
       "Increment `PENDING_BATCH_TO_PROCESS` each time (1 → 2 → 3 → 4)."),

    code(setup_cell(EXTRA_PATHS)),

    code("if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "    import subprocess\n"
         "    subprocess.run(['pip', 'install', '-q', 'transformers', 'torch',\n"
         "                    'sentence-transformers', 'lightgbm', 'scikit-learn', 'datasets'])\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import time\n"
         "import glob\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import torch\n"
         "from pathlib import Path\n"
         "from sklearn.feature_extraction.text import TfidfVectorizer\n"
         "from sklearn.linear_model import LogisticRegression\n"
         "from sklearn.model_selection import train_test_split\n"
         "from sklearn.metrics import accuracy_score\n"
         "import lightgbm as lgb\n"
         "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
         "                          Trainer, TrainingArguments, pipeline)\n"
         "from datasets import Dataset"),

    md("## Configuration"),

    code("PENDING_BATCH_TO_PROCESS = 1   # change to 2, 3, 4 for subsequent iterations\n"
         "NEXT_BATCH_PATH = hitl_folder / f'hitl_pending_batch_{PENDING_BATCH_TO_PROCESS:02d}.pkl'"),

    md("## 1. Load All Labeled Data"),

    code("labeled_files = sorted(glob.glob(str(hitl_folder / 'hitl_review_batch_*.csv')))\n"
         "print(f'Found {len(labeled_files)} labeled batch(es).')\n"
         "\n"
         "dfs = []\n"
         "for f in labeled_files:\n"
         "    tmp = pd.read_csv(f)\n"
         "    if 'human_label' in tmp.columns:\n"
         "        dfs.append(tmp.dropna(subset=['human_label']))\n"
         "\n"
         "if not dfs:\n"
         "    raise ValueError('No labeled data found. Label hitl_review_batch_00.csv first.')\n"
         "\n"
         "train_df = pd.concat(dfs, ignore_index=True)\n"
         "train_df['text'] = train_df['text'].astype(str)\n"
         "print(f'Total labeled examples: {len(train_df):,}')\n"
         "\n"
         "X_tr, X_val, y_tr, y_val = train_test_split(\n"
         "    train_df['text'], train_df['human_label'], test_size=0.1, random_state=42)"),

    md("## 2. Fast Models (TF-IDF + Logistic Regression & LightGBM)"),

    code("t0  = time.time()\n"
         "vec = TfidfVectorizer(max_features=25_000)\n"
         "Xtr_v  = vec.fit_transform(X_tr)\n"
         "Xval_v = vec.transform(X_val)\n"
         "print(f'TF-IDF vectorisation: {time.time()-t0:.1f}s')\n"
         "\n"
         "t0 = time.time()\n"
         "lr = LogisticRegression(max_iter=1000)\n"
         "lr.fit(Xtr_v, y_tr)\n"
         "lr_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lr_acc = accuracy_score(y_val, lr.predict(Xval_v))\n"
         "lr_inf = time.time()-t0\n"
         "print(f'LR   train {lr_tr:.1f}s | infer {lr_inf:.2f}s | acc {lr_acc:.4f}')\n"
         "\n"
         "t0 = time.time()\n"
         "lgbm = lgb.LGBMClassifier(n_estimators=200, random_state=42)\n"
         "lgbm.fit(Xtr_v, y_tr)\n"
         "lg_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lg_acc = accuracy_score(y_val, lgbm.predict(Xval_v))\n"
         "lg_inf = time.time()-t0\n"
         "print(f'LGBM train {lg_tr:.1f}s | infer {lg_inf:.2f}s | acc {lg_acc:.4f}')"),

    md("## 3. Twitter-RoBERTa Fine-Tuning\n\n"
       "`cardiffnlp/twitter-roberta-base` — pre-trained on 58 M tweets."),

    code("model_name = 'cardiffnlp/twitter-roberta-base'\n"
         "tokenizer  = AutoTokenizer.from_pretrained(model_name)\n"
         "\n"
         "unique_labels = list(train_df['human_label'].unique())\n"
         "label2id = {str(v): i for i, v in enumerate(unique_labels)}\n"
         "id2label  = {i: str(v) for i, v in enumerate(unique_labels)}\n"
         "\n"
         "def tokenize(batch):\n"
         "    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)\n"
         "\n"
         "hf_train = Dataset.from_dict({\n"
         "    'text':  X_tr.tolist(),\n"
         "    'label': [label2id[str(y)] for y in y_tr]}).map(tokenize, batched=True)\n"
         "hf_val = Dataset.from_dict({\n"
         "    'text':  X_val.tolist(),\n"
         "    'label': [label2id[str(y)] for y in y_val]}).map(tokenize, batched=True)\n"
         "\n"
         "model = AutoModelForSequenceClassification.from_pretrained(\n"
         "    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)\n"
         "\n"
         "args = TrainingArguments(\n"
         "    output_dir='./roberta_results',\n"
         "    evaluation_strategy='epoch', save_strategy='epoch',\n"
         "    learning_rate=2e-5,\n"
         "    per_device_train_batch_size=16, per_device_eval_batch_size=32,\n"
         "    num_train_epochs=3, weight_decay=0.01, load_best_model_at_end=True)\n"
         "\n"
         "def compute_metrics(ep):\n"
         "    logits, labels = ep\n"
         "    return {'accuracy': accuracy_score(labels, np.argmax(logits, axis=-1))}\n"
         "\n"
         "trainer = Trainer(model=model, args=args,\n"
         "                  train_dataset=hf_train, eval_dataset=hf_val,\n"
         "                  compute_metrics=compute_metrics)"),

    code("t0 = time.time()\n"
         "trainer.train()\n"
         "print(f'RoBERTa train time: {time.time()-t0:.1f}s')\n"
         "\n"
         "res = trainer.evaluate()\n"
         "print(f'RoBERTa val accuracy: {res[\"eval_accuracy\"]:.4f}')\n"
         "\n"
         "best_path = hitl_folder / 'best_roberta_model'\n"
         "trainer.save_model(str(best_path))\n"
         "tokenizer.save_pretrained(str(best_path))\n"
         "print(f'Model saved to {best_path}')"),

    md("## 4. Predict Next Batch and Export Active-Learning Sample"),

    code("assert NEXT_BATCH_PATH.exists(), f'Batch not found: {NEXT_BATCH_PATH}'\n"
         "\n"
         "pending = pd.read_pickle(NEXT_BATCH_PATH)\n"
         "print(f'Running inference on {len(pending):,} tweets...')\n"
         "\n"
         "clf_pipe = pipeline('text-classification', model=trainer.model,\n"
         "                    tokenizer=tokenizer,\n"
         "                    device=0 if torch.cuda.is_available() else -1,\n"
         "                    return_all_scores=True)\n"
         "\n"
         "preds = []\n"
         "t0 = time.time()\n"
         "for i in range(0, len(pending), 500):\n"
         "    preds.extend(clf_pipe(pending['text'].iloc[i:i+500].astype(str).tolist()))\n"
         "print(f'Inference done in {time.time()-t0:.1f}s')\n"
         "\n"
         "pending['predicted_label'] = [max(s, key=lambda x: x['score'])['label'] for s in preds]\n"
         "pending['confidence']      = [max(s, key=lambda x: x['score'])['score'] for s in preds]"),

    code("N_UNCERTAIN = 5_000\n"
         "N_RANDOM    = 5_000\n"
         "\n"
         "uncertain = pending.nsmallest(min(N_UNCERTAIN, len(pending)), 'confidence')\n"
         "pool      = pending.drop(uncertain.index)\n"
         "random_s  = pool.sample(n=min(N_RANDOM, len(pool)), random_state=42)\n"
         "\n"
         "export = pd.concat([uncertain, random_s]).sample(frac=1, random_state=42)\n"
         "export['human_label'] = np.nan\n"
         "export['text'] = export['text'].astype(str).str.replace('\\n', ' ', regex=False)\n"
         "\n"
         "out = hitl_folder / f'hitl_review_batch_{PENDING_BATCH_TO_PROCESS:02d}.csv'\n"
         "export.to_csv(out, index=False)\n"
         "print(f'Exported {len(export):,} tweets for review to {out}')\n"
         "print('Next: fill human_label, save, increment PENDING_BATCH_TO_PROCESS, re-run.')"),
]

# ── Notebook 3: Final Inference ───────────────────────────────────────────────

EXTRA_PATHS_3 = ("hitl_folder = datasets_folder / 'Classifiers_Data' / 'HITL'\n"
                 "out_folder  = datasets_folder / 'Classifiers_Data' / 'Final'\n"
                 "out_folder.mkdir(parents=True, exist_ok=True)")

nb3_cells = [
    md("# 03 - Final Inference\n\n"
       "Applies the best trained model to the complete unseen dataset and\n"
       "merges results with all human-labeled data into one annotated file."),

    code(setup_cell(EXTRA_PATHS_3)),

    code("if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "    import subprocess\n"
         "    subprocess.run(['pip', 'install', '-q', 'transformers', 'torch'])\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import glob\n"
         "import time\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import torch\n"
         "from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline"),

    md("## 1. Load Best Model"),

    code("best_path = hitl_folder / 'best_roberta_model'\n"
         "if not best_path.exists():\n"
         "    raise FileNotFoundError('Model not found. Run notebook 02 first.')\n"
         "\n"
         "tokenizer = AutoTokenizer.from_pretrained(str(best_path))\n"
         "model     = AutoModelForSequenceClassification.from_pretrained(str(best_path))\n"
         "device    = 0 if torch.cuda.is_available() else -1\n"
         "clf_pipe  = pipeline('text-classification', model=model,\n"
         "                     tokenizer=tokenizer, device=device, return_all_scores=True)\n"
         "print('Model loaded.')"),

    md("## 2. Load Inference Dataset"),

    code("df = pd.read_pickle(hitl_folder / 'inference_dataset.pkl')\n"
         "print(f'Inference dataset: {len(df):,} tweets')"),

    md("## 3. Run Inference"),

    code("preds = []\n"
         "t0    = time.time()\n"
         "for i in range(0, len(df), 500):\n"
         "    preds.extend(clf_pipe(df['text'].iloc[i:i+500].astype(str).tolist()))\n"
         "    if i % 50_000 == 0 and i > 0:\n"
         "        print(f'  {i:,}/{len(df):,} processed...')\n"
         "print(f'Inference done in {time.time()-t0:.1f}s')\n"
         "\n"
         "df['predicted_label'] = [max(s, key=lambda x: x['score'])['label'] for s in preds]\n"
         "df['confidence']      = [max(s, key=lambda x: x['score'])['score'] for s in preds]"),

    md("## 4. Merge with Human Labels and Save"),

    code("labeled_files = sorted(glob.glob(str(hitl_folder / 'hitl_review_batch_*.csv')))\n"
         "human_dfs = []\n"
         "for f in labeled_files:\n"
         "    tmp = pd.read_csv(f)\n"
         "    if 'human_label' in tmp.columns:\n"
         "        tmp['predicted_label'] = tmp['human_label'].fillna(tmp.get('predicted_label'))\n"
         "        human_dfs.append(tmp.dropna(subset=['human_label']))\n"
         "\n"
         "if human_dfs:\n"
         "    human_df = pd.concat(human_dfs, ignore_index=True)\n"
         "    human_df['is_human_labeled'] = True\n"
         "else:\n"
         "    human_df = pd.DataFrame()\n"
         "\n"
         "df['human_label']      = np.nan\n"
         "df['is_human_labeled'] = False\n"
         "\n"
         "final_df = pd.concat([df, human_df], ignore_index=True)\n"
         "print(f'Final dataset: {len(final_df):,} rows')\n"
         "\n"
         "final_df.to_pickle(out_folder / 'final_annotated_tweets.pkl')\n"
         "final_df.to_csv(out_folder / 'final_annotated_tweets.csv', index=False)\n"
         "print('Saved to Classifiers_Data/Final/')"),
]

# ── Write all three notebooks ─────────────────────────────────────────────────

if __name__ == "__main__":
    write_notebook(f"{NB_DIR}/01_hitl_data_preparation.ipynb", nb1_cells)
    write_notebook(f"{NB_DIR}/02_hitl_training_loop.ipynb",    nb2_cells)
    write_notebook(f"{NB_DIR}/03_final_inference.ipynb",       nb3_cells)

