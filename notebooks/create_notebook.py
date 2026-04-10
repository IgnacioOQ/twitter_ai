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
twits_folder          = BASE_PATH / 'Raw Data/Twits/'
test_folder           = BASE_PATH / 'Raw Data/'
datasets_folder       = BASE_PATH / 'Data Sets'
cleanedds_folder      = BASE_PATH / 'Data Sets/Cleaned Data'
networks_folder       = BASE_PATH / 'Data Sets/Networks/'
literature_folder     = BASE_PATH / 'Literature/'
topic_models_folder   = BASE_PATH / 'Models/Topic Modeling/'
classifiers_folder    = BASE_PATH / 'Models/Classifiers/'
classifiers_folder.mkdir(parents=True, exist_ok=True)"""
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

    md("## 2. Sentence Embeddings\n\nGenerate sentence embeddings using `all-MiniLM-L6-v2` for use in the boosted tree and logistic regression."),

    code("from sentence_transformers import SentenceTransformer\n"
         "\n"
         "t0 = time.time()\n"
         "st_model = SentenceTransformer('all-MiniLM-L6-v2')\n"
         "embeddings_tr  = st_model.encode(X_tr.tolist(), show_progress_bar=True)\n"
         "embeddings_val = st_model.encode(X_val.tolist(), show_progress_bar=True)\n"
         "print(f'Sentence embedding time: {time.time()-t0:.1f}s')"),

    md("## 3. Bag-of-Words (CountVectorizer)"),

    code("from sklearn.feature_extraction.text import CountVectorizer\n"
         "\n"
         "bow_vec = CountVectorizer(max_features=50_000)\n"
         "X_bow_tr  = bow_vec.fit_transform(X_tr)\n"
         "X_bow_val = bow_vec.transform(X_val)\n"
         "print(f'BoW shape: {X_bow_tr.shape}')"),

    md("## 4. Logistic Regression"),

    code("from sklearn.metrics import roc_auc_score\n"
         "\n"
         "# --- On sentence embeddings ---\n"
         "t0 = time.time()\n"
         "lr_embed = LogisticRegression(max_iter=1000)\n"
         "lr_embed.fit(embeddings_tr, y_tr)\n"
         "lr_embed_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lr_embed_preds = lr_embed.predict(embeddings_val)\n"
         "lr_embed_inf   = time.time()-t0\n"
         "print(f'LR (embed) train {lr_embed_tr:.1f}s | infer {lr_embed_inf:.2f}s | acc {accuracy_score(y_val, lr_embed_preds):.4f}')\n"
         "\n"
         "# --- On BoW ---\n"
         "t0 = time.time()\n"
         "lr_bow = LogisticRegression(max_iter=1000)\n"
         "lr_bow.fit(X_bow_tr, y_tr)\n"
         "lr_bow_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lr_bow_preds = lr_bow.predict(X_bow_val)\n"
         "lr_bow_inf   = time.time()-t0\n"
         "print(f'LR (BoW)   train {lr_bow_tr:.1f}s | infer {lr_bow_inf:.2f}s | acc {accuracy_score(y_val, lr_bow_preds):.4f}')"),

    md("## 5. Boosted Tree (LightGBM native API)\n\nMatches the approach in `Classifiers_Training_Final.ipynb` — uses `lgb.Dataset` and `lgb.train()` with a params dict. Run with both sentence embeddings and BoW."),

    code("# LightGBM params (binary classification, AUC metric)\n"
         "lgb_param = {\n"
         "    'objective':    'binary',\n"
         "    'boosting_type': 'gbdt',\n"
         "    'metric':       'auc',\n"
         "    'verbose':      -1,\n"
         "}\n"
         "NUM_BOOST_ROUND = 1000\n"
         "\n"
         "# --- On sentence embeddings ---\n"
         "train_data_embed = lgb.Dataset(embeddings_tr, label=(y_tr.astype(int) if hasattr(y_tr, 'astype') else list(y_tr)))\n"
         "t0 = time.time()\n"
         "clf_embed = lgb.train(lgb_param, train_data_embed, num_boost_round=NUM_BOOST_ROUND)\n"
         "lgb_embed_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lgb_embed_preds = (clf_embed.predict(embeddings_val) > 0.5).astype(int)\n"
         "lgb_embed_inf   = time.time()-t0\n"
         "print(f'LGB (embed) train {lgb_embed_tr:.1f}s | infer {lgb_embed_inf:.2f}s | acc {accuracy_score(y_val, lgb_embed_preds):.4f}')\n"
         "clf_embed.save_model(str(classifiers_folder / 'lgb_embed.txt'))"),

    code("# --- On BoW ---\n"
         "import scipy.sparse\n"
         "X_bow_tr_dense  = X_bow_tr.toarray()  if scipy.sparse.issparse(X_bow_tr)  else X_bow_tr\n"
         "X_bow_val_dense = X_bow_val.toarray() if scipy.sparse.issparse(X_bow_val) else X_bow_val\n"
         "\n"
         "train_data_bow = lgb.Dataset(X_bow_tr_dense, label=(y_tr.astype(int) if hasattr(y_tr, 'astype') else list(y_tr)))\n"
         "t0 = time.time()\n"
         "clf_bow = lgb.train(lgb_param, train_data_bow, num_boost_round=NUM_BOOST_ROUND)\n"
         "lgb_bow_tr = time.time()-t0\n"
         "t0 = time.time()\n"
         "lgb_bow_preds = (clf_bow.predict(X_bow_val_dense) > 0.5).astype(int)\n"
         "lgb_bow_inf   = time.time()-t0\n"
         "print(f'LGB (BoW)   train {lgb_bow_tr:.1f}s | infer {lgb_bow_inf:.2f}s | acc {accuracy_score(y_val, lgb_bow_preds):.4f}')\n"
         "clf_bow.save_model(str(classifiers_folder / 'lgb_bow.txt'))"),

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
         "best_path = classifiers_folder / 'best_roberta_model'\n"
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

    code("best_path = classifiers_folder / 'best_roberta_model'\n"
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

# ── Notebook 4: Full-Dataset Mass Inference (~17M tweets) ────────────────────

EXTRA_PATHS_4 = ("hitl_folder        = datasets_folder / 'Classifiers_Data' / 'HITL'\n"
                 "full_inference_folder = datasets_folder / 'Classifiers_Data' / 'Full_Inference'\n"
                 "full_inference_folder.mkdir(parents=True, exist_ok=True)")

nb4_cells = [
    md("# 04 - Full Dataset Mass Inference\n\n"
       "Classifies the **entire remaining tweet corpus** (~17 million tweets) using the\n"
       "models trained in notebook 02. Models are loaded from `Models/Classifiers/`.\n\n"
       "This notebook is designed to run on Colab with GPU acceleration and processes tweets\n"
       "in chunks to avoid out-of-memory errors."),

    code(setup_cell(EXTRA_PATHS_4)),

    code("if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "    import subprocess\n"
         "    subprocess.run(['pip', 'install', '-q', 'transformers', 'torch',\n"
         "                    'sentence-transformers', 'lightgbm'])\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import os\n"
         "import re\n"
         "import time\n"
         "import pickle\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import tqdm\n"
         "import torch\n"
         "import lightgbm as lgb\n"
         "from pathlib import Path\n"
         "from sentence_transformers import SentenceTransformer\n"
         "from sklearn.feature_extraction.text import CountVectorizer\n"
         "from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline"),

    md("## Configuration\n\n"
       "Set `CHUNK_SIZE` based on available RAM. 10 000 is safe for Colab; increase if you have more memory."),

    code("CHUNK_SIZE    = 10_000   # tweets processed per batch\n"
         "ROBERTA_BATCH = 64       # pipeline batch size for RoBERTa (GPU dependent)\n"
         "SAVE_EVERY    = 100_000  # checkpoint: save intermediate results every N tweets"),

    md("## 1. Text Cleaning"),

    code("def clean_tweet(text: str) -> str:\n"
         "    text = re.sub(r'http\\S+', '', text)          # remove URLs\n"
         "    text = re.sub(r'@[A-Za-z0-9_]+', '', text)   # remove @mentions\n"
         "    text = text.replace('#', '')                  # strip # from hashtags\n"
         "    text = re.sub(r'\\bRT\\b', '', text)           # remove RT markers\n"
         "    text = re.sub(r'\\n', ' ', text)\n"
         "    text = re.sub(r'\\s+', ' ', text).strip()\n"
         "    return text"),

    md("## 2. Load Full Tweet Dataset\n\n"
       "Expects the full cleaned tweet corpus as a DataFrame or pickle dict.\n"
       "Adjust the path and loading logic to match your data format."),

    code("# ── Adjust this path and format to your actual full corpus ──────────────\n"
         "FULL_DATA_PATH = cleanedds_folder / 'all_cleaned_tweets.pkl'\n"
         "\n"
         "print(f'Loading full corpus from {FULL_DATA_PATH}...')\n"
         "if FULL_DATA_PATH.suffix == '.pkl':\n"
         "    full_df = pd.read_pickle(FULL_DATA_PATH)\n"
         "elif FULL_DATA_PATH.suffix == '.csv':\n"
         "    full_df = pd.read_csv(FULL_DATA_PATH)\n"
         "else:\n"
         "    raise ValueError('Unknown format. Update FULL_DATA_PATH.')\n"
         "\n"
         "# Ensure id column exists\n"
         "if 'tweet_id' in full_df.columns and 'id' not in full_df.columns:\n"
         "    full_df['id'] = full_df['tweet_id']\n"
         "\n"
         "print(f'Total tweets to classify: {len(full_df):,}')"),

    md("## 3. Load Trained Models"),

    code("# ── Twitter-RoBERTa (primary model) ─────────────────────────────────────\n"
         "best_model_path = classifiers_folder / 'best_roberta_model'\n"
         "if not best_model_path.exists():\n"
         "    raise FileNotFoundError(f'RoBERTa model not found at {best_model_path}. Run notebook 02 first.')\n"
         "\n"
         "print('Loading Twitter-RoBERTa...')\n"
         "tokenizer = AutoTokenizer.from_pretrained(str(best_model_path))\n"
         "model     = AutoModelForSequenceClassification.from_pretrained(str(best_model_path))\n"
         "device    = 0 if torch.cuda.is_available() else -1\n"
         "roberta_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer,\n"
         "                        device=device, batch_size=ROBERTA_BATCH, return_all_scores=True)\n"
         "print(f'RoBERTa loaded. Device: {\"GPU\" if device==0 else \"CPU\"}')"),

    code("# ── LightGBM (sentence embeddings) ──────────────────────────────────────\n"
         "lgb_embed_path = classifiers_folder / 'lgb_embed.txt'\n"
         "lgb_bow_path   = classifiers_folder / 'lgb_bow.txt'\n"
         "\n"
         "clf_embed = lgb.Booster(model_file=str(lgb_embed_path)) if lgb_embed_path.exists() else None\n"
         "clf_bow   = lgb.Booster(model_file=str(lgb_bow_path))   if lgb_bow_path.exists()   else None\n"
         "\n"
         "if clf_embed:\n"
         "    print('LightGBM (embed) loaded.')\n"
         "if clf_bow:\n"
         "    print('LightGBM (BoW) loaded.')"),

    code("# ── Sentence Transformer (for LightGBM embed) ───────────────────────────\n"
         "if clf_embed:\n"
         "    st_model = SentenceTransformer('all-MiniLM-L6-v2')\n"
         "    print('Sentence transformer loaded.')"),

    md("## 4. Run Mass Inference\n\n"
       "Processes tweets in chunks. Results are saved incrementally every `SAVE_EVERY` tweets\n"
       "to avoid losing progress on long runs."),

    code("results = []\n"
         "t_start = time.time()\n"
         "n_total = len(full_df)\n"
         "\n"
         "for chunk_start in tqdm.tqdm(range(0, n_total, CHUNK_SIZE)):\n"
         "    chunk = full_df.iloc[chunk_start:chunk_start + CHUNK_SIZE].copy()\n"
         "    texts_raw = chunk['text'].astype(str).tolist()\n"
         "    texts_clean = [clean_tweet(t) for t in texts_raw]\n"
         "\n"
         "    row_results = {'id': chunk['id'].tolist()\n"
         "                   if 'id' in chunk.columns else list(range(chunk_start, chunk_start+len(chunk)))}\n"
         "\n"
         "    # ── RoBERTa predictions ──────────────────────────────────────────\n"
         "    scores = roberta_pipe(texts_clean)\n"
         "    row_results['roberta_label']      = [max(s, key=lambda x: x['score'])['label'] for s in scores]\n"
         "    row_results['roberta_confidence'] = [max(s, key=lambda x: x['score'])['score'] for s in scores]\n"
         "\n"
         "    # ── LightGBM (embed) predictions ────────────────────────────────\n"
         "    if clf_embed:\n"
         "        embeds = st_model.encode(texts_clean, show_progress_bar=False)\n"
         "        lgb_embed_probs = clf_embed.predict(embeds)\n"
         "        row_results['lgb_embed_label'] = (lgb_embed_probs > 0.5).astype(int).tolist()\n"
         "        row_results['lgb_embed_prob']  = lgb_embed_probs.tolist()\n"
         "\n"
         "    results.append(pd.DataFrame(row_results))\n"
         "\n"
         "    # ── Checkpoint save ──────────────────────────────────────────────\n"
         "    processed_so_far = chunk_start + len(chunk)\n"
         "    if processed_so_far % SAVE_EVERY < CHUNK_SIZE:\n"
         "        checkpoint_df = pd.concat(results, ignore_index=True)\n"
         "        checkpoint_path = full_inference_folder / f'checkpoint_{processed_so_far}.pkl'\n"
         "        checkpoint_df.to_pickle(checkpoint_path)\n"
         "        print(f'Checkpoint saved at {processed_so_far:,} tweets ({time.time()-t_start:.0f}s elapsed)')\n"
         "\n"
         "print(f'\nDone. Total time: {time.time()-t_start:.1f}s')\n"
         "results_df = pd.concat(results, ignore_index=True)"),

    md("## 5. Merge Predictions Back and Save Final Output"),

    code("# Merge predictions onto the original dataframe (keeping all original columns)\n"
         "final_df = full_df.merge(results_df, on='id', how='left')\n"
         "\n"
         "print(f'Final annotated dataset: {len(final_df):,} rows')\n"
         "print(final_df[['id', 'text', 'roberta_label', 'roberta_confidence']].head())"),

    code("# Save as both pickle (fast I/O) and CSV (interoperability)\n"
         "out_pkl = full_inference_folder / 'full_inference_annotated.pkl'\n"
         "out_csv = full_inference_folder / 'full_inference_annotated.csv'\n"
         "\n"
         "final_df.to_pickle(out_pkl)\n"
         "final_df.to_csv(out_csv, index=False)\n"
         "print(f'Saved to:\n  {out_pkl}\n  {out_csv}')"),
]


# ── Write all four notebooks ──────────────────────────────────────────────────

if __name__ == "__main__":
    write_notebook(f"{NB_DIR}/01_hitl_data_preparation.ipynb", nb1_cells)
    write_notebook(f"{NB_DIR}/02_hitl_training_loop.ipynb",    nb2_cells)
    write_notebook(f"{NB_DIR}/03_final_inference.ipynb",       nb3_cells)
    write_notebook(f"{NB_DIR}/04_full_dataset_inference.ipynb", nb4_cells)


