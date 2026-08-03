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

# ── Notebook 1: LLM Bootstrap Labelling ──────────────────────────────────────

nb1_cells = [
    md("# 01 - LLM Bootstrap Labelling\n\n"
       "Uses an LLM (Gemini API) to **bootstrap** the HITL classifier seed labels.\n"
       "Replaces — or precedes — the human seed step described in `classification_strategy.md` (Step 0).\n\n"
       "**Inputs:** `llm_bootstrap_dataset.pkl` (~10 000 tweets carved out by `00_hitl_data_preparation.ipynb`).\n"
       "This subset is disjoint from `base_dataset.pkl`, the HITL batches, and `inference_dataset.pkl` —\n"
       "see `partition_ids.pkl` for the manifest.\n\n"
       "**Pipeline:**\n"
       "1. Load `llm_bootstrap_dataset.pkl`.\n"
       "2. For each tweet, call the LLM with a prompt containing the per-category criteria.\n"
       "3. Parse the JSON response, validate the label, retry on transient errors.\n"
       "4. Checkpoint every N tweets to survive Colab disconnects.\n"
       "5. Save the final CSV with the **same schema** as `hitl_review_batch_*.csv`\n"
       "   so it drops straight into `02_hitl_training_loop.ipynb`.\n\n"
       "Two cells are **placeholders** that must be filled in before running:\n"
       "the category list + criteria, and the Gemini API key + model id."),

    code(setup_cell(EXTRA_PATHS)),

    code("import os\n"
         "if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "    import subprocess\n"
         "    subprocess.run(['pip', 'install', '-q', 'google-genai'])\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import json\n"
         "import re\n"
         "import time\n"
         "from pathlib import Path\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import tqdm\n"
         "from google import genai\n"
         "from google.genai import types"),

    md("## Configuration\n\n"
       "Tune the constants below per run. `SMOKE_TEST` keeps the pipeline cheap during development."),

    code("# ── Run mode ────────────────────────────────────────────────────────────\n"
         "SMOKE_TEST   = True       # True → label SMOKE_TEST_N tweets only; flip to False for the full run\n"
         "SMOKE_TEST_N = 100\n"
         "\n"
         "# ── LLM ─────────────────────────────────────────────────────────────────\n"
         "# Gemini model id. Check availability first (per content/reference/GEMINI_MODELS_REF.md;\n"
         "# error handling in content/how-to/LLM_ERROR_HANDLING_SKILL.md):\n"
         "#   'gemini-2.5-flash-lite'  — cheapest stable (DEFAULT); fine for one-shot classification\n"
         "#   'gemini-2.5-flash'       — standard, more capable (thinking ON by default — see DISABLE_THINKING)\n"
         "#   'gemini-2.5-pro'         — most capable; CANNOT disable thinking (min budget 128)\n"
         "#   'gemini-2.0-flash-lite' / 'gemini-2.0-flash' — DEPRECATED (EOL ~June 2026)\n"
         "MODEL_NAME  = 'gemini-2.5-flash-lite'\n"
         "TEMPERATURE = 0.0         # deterministic classification; raise only if you want sampling diversity\n"
         "\n"
         "# Thinking is opt-out on gemini-2.5+ models (it inflates token cost a lot for tasks that\n"
         "# don't need step-by-step reasoning). For one-shot classification we want it OFF; the\n"
         "# config built below only attaches a ThinkingConfig when the chosen model is in the 2.5+\n"
         "# family. Set False ONLY if you deliberately want the model to reason before answering.\n"
         "DISABLE_THINKING = True\n"
         "\n"
         "# ── Retry / backoff ─────────────────────────────────────────────────────\n"
         "MAX_RETRIES     = 3\n"
         "INITIAL_BACKOFF = 2.0     # seconds; doubled on each retry\n"
         "\n"
         "# ── I/O ─────────────────────────────────────────────────────────────────\n"
         "INPUT_PATH        = hitl_folder / 'llm_bootstrap_dataset.pkl'\n"
         "OUTPUT_CSV        = hitl_folder / 'llm_bootstrap_labels.csv'\n"
         "OUTPUT_PKL        = hitl_folder / 'llm_bootstrap_labels_full.pkl'\n"
         "CHECKPOINT_PREFIX = 'llm_bootstrap_checkpoint'\n"
         "CHECKPOINT_EVERY  = 1_000  # save partial results every N tweets"),

    md("## Categories and Criteria\n\n"
       "**TODO — fill these in before running.**\n\n"
       "- `CATEGORIES` is the closed list of allowed labels. The LLM must return one of these strings.\n"
       "- `CATEGORY_CRITERIA` is the per-category description that goes into the prompt.\n"
       "  Be precise: include a definition, 2-3 positive examples, and 1-2 exclusions per category."),

    code("# TODO: list every allowed category label exactly as you want it written in the output CSV.\n"
         "CATEGORIES: list[str] = [\n"
         "    # 'category_a',\n"
         "    # 'category_b',\n"
         "    # 'category_c',\n"
         "]\n"
         "\n"
         "# TODO: write the full criteria for each category. The LLM sees this verbatim.\n"
         "CATEGORY_CRITERIA: str = \"\"\"\n"
         "<<< FILL IN THE PER-CATEGORY CRITERIA HERE >>>\n"
         "\"\"\".strip()\n"
         "\n"
         "assert CATEGORIES, 'CATEGORIES is empty — fill it in before running.'\n"
         "assert '<<< FILL IN' not in CATEGORY_CRITERIA, 'CATEGORY_CRITERIA still contains the placeholder.'"),

    md("## API Key\n\n"
       "**TODO — provide a Gemini API key before running.**\n\n"
       "On Colab, store the key as a notebook secret named `GEMINI_API_KEY` (left sidebar → key icon)\n"
       "and the cell below picks it up via `userdata.get`. Locally, set the `GEMINI_API_KEY`\n"
       "environment variable. **Never hard-code the key in the notebook.**"),

    code("API_KEY = ''  # leave empty — populated below from Colab secrets / env var\n"
         "\n"
         "if not RUNNING_LOCALLY:\n"
         "    try:\n"
         "        from google.colab import userdata\n"
         "        API_KEY = userdata.get('GEMINI_API_KEY')\n"
         "    except Exception as e:\n"
         "        print(f'Colab userdata lookup failed: {e}')\n"
         "else:\n"
         "    API_KEY = os.environ.get('GEMINI_API_KEY', '')\n"
         "\n"
         "assert API_KEY, 'GEMINI_API_KEY not set. Add it as a Colab secret or env var before running.'\n"
         "assert MODEL_NAME, 'MODEL_NAME is empty — set it in the Configuration cell.'\n"
         "\n"
         "client = genai.Client(api_key=API_KEY)\n"
         "\n"
         "config_kwargs = dict(\n"
         "    temperature=TEMPERATURE,\n"
         "    response_mime_type='application/json',\n"
         ")\n"
         "if MODEL_NAME.startswith('gemini-2.5-pro'):\n"
         "    # gemini-2.5-pro cannot turn thinking off; minimum thinking_budget is 128.\n"
         "    config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=128)\n"
         "    print(f'Note: {MODEL_NAME} cannot disable thinking; pinning thinking_budget=128')\n"
         "elif DISABLE_THINKING and MODEL_NAME.startswith('gemini-2.5'):\n"
         "    config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=0)\n"
         "    print(f'Thinking disabled for {MODEL_NAME} (DISABLE_THINKING=True)')\n"
         "elif not DISABLE_THINKING and MODEL_NAME.startswith('gemini-2.5'):\n"
         "    print(f'WARNING: thinking is ENABLED for {MODEL_NAME} — expect higher token cost')\n"
         "GEN_CONFIG = types.GenerateContentConfig(**config_kwargs)\n"
         "\n"
         "print(f'LLM client ready: {MODEL_NAME}')"),

    md("## Prompt and Response Schema\n\n"
       "The LLM is asked to return a strict JSON object:\n"
       "```\n"
       "{\"label\": \"<one of CATEGORIES>\", \"confidence\": <float 0-1>, \"rationale\": \"<one short sentence>\"}\n"
       "```\n"
       "Anything else is treated as a parse error: it is retried, and on final failure the row is marked `PARSE_ERROR`."),

    code("def build_prompt(tweet_text: str) -> str:\n"
         "    return (\n"
         "        'You are a tweet classifier for a research project on AI public trust.\\n'\n"
         "        'Classify the tweet into exactly one of the following categories:\\n'\n"
         "        f'{\", \".join(CATEGORIES)}\\n\\n'\n"
         "        'Per-category criteria:\\n'\n"
         "        f'{CATEGORY_CRITERIA}\\n\\n'\n"
         "        'Return ONLY a JSON object with this exact schema (no prose, no markdown fences):\\n'\n"
         "        '{\"label\": \"<one of the categories above>\", '\n"
         "        '\"confidence\": <number between 0 and 1>, '\n"
         "        '\"rationale\": \"<one short sentence>\"}\\n\\n'\n"
         "        f'Tweet:\\n\"\"\"{tweet_text}\"\"\"'\n"
         "    )"),

    md("## Classification Function\n\n"
       "Single-tweet wrapper: build prompt → call LLM → parse + validate JSON → retry on transient errors."),

    code("PARSE_ERROR_RESULT = {'label': 'PARSE_ERROR', 'confidence': 0.0, 'rationale': ''}\n"
         "\n"
         "def _strip_code_fences(raw: str) -> str:\n"
         "    raw = raw.strip()\n"
         "    if raw.startswith('```'):\n"
         "        raw = re.sub(r'^```(?:json)?\\s*', '', raw)\n"
         "        raw = re.sub(r'\\s*```$', '', raw)\n"
         "    return raw.strip()\n"
         "\n"
         "def classify_tweet(text: str) -> dict:\n"
         "    prompt = build_prompt(text)\n"
         "    last_error = ''\n"
         "    for attempt in range(MAX_RETRIES):\n"
         "        try:\n"
         "            resp = client.models.generate_content(\n"
         "                model=MODEL_NAME,\n"
         "                contents=prompt,\n"
         "                config=GEN_CONFIG,\n"
         "            )\n"
         "            raw  = _strip_code_fences(resp.text or '')\n"
         "            parsed = json.loads(raw)\n"
         "            label = str(parsed.get('label', '')).strip()\n"
         "            if label not in CATEGORIES:\n"
         "                raise ValueError(f'label {label!r} not in CATEGORIES')\n"
         "            return {\n"
         "                'label': label,\n"
         "                'confidence': float(parsed.get('confidence', 0.0)),\n"
         "                'rationale': str(parsed.get('rationale', ''))[:500],\n"
         "            }\n"
         "        except Exception as e:\n"
         "            last_error = f'{type(e).__name__}: {e}'\n"
         "            if attempt + 1 < MAX_RETRIES:\n"
         "                time.sleep(INITIAL_BACKOFF * (2 ** attempt))\n"
         "    return {**PARSE_ERROR_RESULT, 'rationale': last_error[:500]}"),

    md("## Load Input"),

    code("assert INPUT_PATH.exists(), f'Input not found: {INPUT_PATH}. Run 00_hitl_data_preparation.ipynb first.'\n"
         "df = pd.read_pickle(INPUT_PATH)\n"
         "print(f'Loaded {len(df):,} tweets from {INPUT_PATH.name}')\n"
         "\n"
         "if SMOKE_TEST:\n"
         "    df = df.sample(n=min(SMOKE_TEST_N, len(df)), random_state=42).reset_index(drop=True)\n"
         "    print(f'SMOKE_TEST mode → using {len(df)} tweets')\n"
         "\n"
         "for col in ('id', 'text', 'likes', 'retweets'):\n"
         "    if col not in df.columns:\n"
         "        df[col] = '' if col in ('id', 'text') else 0\n"
         "\n"
         "df['text'] = df['text'].astype(str)"),

    md("## Run Classification with Checkpointing"),

    code("results: list[dict] = []\n"
         "t0 = time.time()\n"
         "\n"
         "for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='LLM labelling'):\n"
         "    classification = classify_tweet(row['text'])\n"
         "    results.append({\n"
         "        'id': row['id'],\n"
         "        'text': row['text'],\n"
         "        'likes': row.get('likes', 0),\n"
         "        'retweets': row.get('retweets', 0),\n"
         "        'predicted_label': classification['label'],\n"
         "        'confidence': classification['confidence'],\n"
         "        'rationale': classification['rationale'],\n"
         "        'human_label': '',\n"
         "    })\n"
         "    if (len(results) % CHECKPOINT_EVERY) == 0:\n"
         "        ckpt = hitl_folder / f'{CHECKPOINT_PREFIX}_{len(results)}.pkl'\n"
         "        pd.DataFrame(results).to_pickle(ckpt)\n"
         "        tqdm.tqdm.write(f'checkpoint → {ckpt.name} ({time.time()-t0:.0f}s elapsed)')\n"
         "\n"
         "out_df = pd.DataFrame(results)\n"
         "print(f'Done. Total time: {time.time()-t0:.1f}s')\n"
         "print(out_df['predicted_label'].value_counts(dropna=False))"),

    md("## Save Output\n\n"
       "Two artifacts:\n"
       "- **`llm_bootstrap_labels.csv`** — same schema as `hitl_review_batch_*.csv`, ready to drop into `02_hitl_training_loop.ipynb`.\n"
       "- **`llm_bootstrap_labels_full.pkl`** — same data **plus** `confidence` and `rationale` columns for inspection."),

    code("hitl_schema_cols = ['id', 'text', 'likes', 'retweets', 'predicted_label', 'human_label']\n"
         "out_df[hitl_schema_cols].to_csv(OUTPUT_CSV, index=False)\n"
         "out_df.to_pickle(OUTPUT_PKL)\n"
         "\n"
         "n_errors = (out_df['predicted_label'] == 'PARSE_ERROR').sum()\n"
         "print(f'Saved → {OUTPUT_CSV}')\n"
         "print(f'Saved → {OUTPUT_PKL}')\n"
         "print(f'PARSE_ERROR rows: {n_errors:,} / {len(out_df):,} ({n_errors/max(len(out_df),1):.1%})')"),
]

# ── Notebook 0: Data Preparation ─────────────────────────────────────────────

nb0_cells = [
    md("# 00 - HITL Data Preparation\n\n"
       "Splits the **cleaned** tweets dataset into four non-overlapping partitions:\n"
       "- **LLM Bootstrap** (~10 000 tweets): labelled by an LLM in `01_llm_bootstrap_labelling.ipynb`\n"
       "- **Base** (~100 000 tweets): reserve pool / source for any human seed labelling\n"
       "- **HITL batches** (~200 000 tweets, 4 × 50 000): iterative human review\n"
       "- **Final Inference** (remainder): classified by the final model in notebook 03\n\n"
       "All partitions are mutually disjoint. A `partition_ids.pkl` manifest is written so any\n"
       "downstream notebook can verify which subset a tweet belongs to.\n\n"
       "Run this notebook **once** at the start of the project, before any of `01`, `02`, `03`."),

    code(setup_cell(EXTRA_PATHS)),

    code("import os\n"
         "if not RUNNING_LOCALLY:\n"
         "    print('Running Colab setup...')\n"
         "else:\n"
         "    print('Running locally: skipping Colab setup.')"),

    code("import pickle\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "from pathlib import Path"),

    md("## Load the Pruned Tweet Dict\n\n"
       "Source: `cleanedds_folder / 'AItrust_twits_pruned_dict.json'` — the JSONL output of\n"
       "`notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb`. Each line is one\n"
       "tweet with the standard Twitter API v2 schema: `id`, `text`, `processed_text`,\n"
       "`created_at`, `type`, `public_metrics` (nested), `referenced_tweets`, ...\n\n"
       "Toggle `USE_TEST_DATA = True` to load the smaller `*_test.json` variant for development."),

    code("USE_TEST_DATA = False  # True → AItrust_twits_pruned_dict_test.json (development); False → full dataset\n"
         "\n"
         "PRUNED_DICT_NAME = 'AItrust_twits_pruned_dict_test.json' if USE_TEST_DATA else 'AItrust_twits_pruned_dict.json'\n"
         "PRUNED_DICT_PATH = cleanedds_folder / PRUNED_DICT_NAME\n"
         "\n"
         "assert PRUNED_DICT_PATH.exists(), (\n"
         "    f'Pruned dict not found: {PRUNED_DICT_PATH}. '\n"
         "    f'Run notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb first.'\n"
         ")\n"
         "\n"
         "print(f'Loading {PRUNED_DICT_PATH}...')\n"
         "df = pd.read_json(PRUNED_DICT_PATH, lines=True)\n"
         "print(f'Loaded {len(df):,} tweets')\n"
         "print(f'Columns: {list(df.columns)}')"),

    md("## Normalise Columns\n\n"
       "Pull `likes` and `retweets` out of the nested `public_metrics` dict, cast `id` to string\n"
       "(Twitter snowflake IDs exceed 2^53 and lose precision as Python floats), and keep only\n"
       "the columns the HITL pipeline needs: `id`, `text`, `likes`, `retweets`."),

    code("def _pm_field(pm, key, default=0):\n"
         "    return pm.get(key, default) if isinstance(pm, dict) else default\n"
         "\n"
         "if 'public_metrics' in df.columns:\n"
         "    df['likes']    = df['public_metrics'].apply(lambda pm: _pm_field(pm, 'like_count', 0)).astype(int)\n"
         "    df['retweets'] = df['public_metrics'].apply(lambda pm: _pm_field(pm, 'retweet_count', 0)).astype(int)\n"
         "else:\n"
         "    df['likes'] = df['retweets'] = 0\n"
         "\n"
         "df['id']   = df['id'].astype(str)\n"
         "df['text'] = df['text'].astype(str)\n"
         "\n"
         "df = df[['id', 'text', 'likes', 'retweets']].copy()\n"
         "df['predicted_label'] = np.nan\n"
         "df['human_label']     = np.nan\n"
         "print(f'Normalised: {len(df):,} tweets')"),

    md("## Attention-Weighted Shuffle\n\n"
       "Tweet engagement (likes + retweets) follows a heavy-tailed (near power-law) distribution:\n"
       "a small number of viral tweets carry most of the attention, while the long tail gets almost\n"
       "none. A **uniform** random sample of 10 000 tweets from this corpus would consist almost\n"
       "entirely of low-engagement tweets — the LLM and the classifier would never see the\n"
       "discourse-shaping content.\n\n"
       "We instead do an **attention-weighted permutation**: each tweet's selection probability is\n"
       "proportional to `(likes + retweets + 1) ** SAMPLING_ALPHA`. The downstream slice still cuts\n"
       "the dataframe into contiguous partitions, but the partitions are now *stratified by\n"
       "influence* — the LLM Bootstrap slice (first 10 000) tends to pick up the influential head,\n"
       "the base / HITL / inference slices follow with progressively lighter engagement.\n\n"
       "`SAMPLING_ALPHA` is the smoothing exponent:\n"
       "- `0.0` — uniform random (recovers the previous behaviour).\n"
       "- `0.5` — square-root rule (default): head is well-represented, body still well-covered.\n"
       "- `1.0` — proportional to attention: heavily concentrates on the head."),

    code("SAMPLING_ALPHA = 0.5  # 0 → uniform; 0.5 → sqrt smoothing (default); 1 → proportional to attention\n"
         "\n"
         "if SAMPLING_ALPHA > 0:\n"
         "    attention = (df['likes'] + df['retweets'] + 1).astype(float)\n"
         "    weights   = attention ** SAMPLING_ALPHA\n"
         "    df = df.sample(frac=1, weights=weights, random_state=42).reset_index(drop=True)\n"
         "    print(f'Attention-weighted shuffle (alpha={SAMPLING_ALPHA})')\n"
         "else:\n"
         "    df = df.sample(frac=1, random_state=42).reset_index(drop=True)\n"
         "    print('Uniform shuffle (alpha=0)')"),

    md("## Partition the Dataset\n\n"
       "Slices the shuffled dataframe into four contiguous, disjoint partitions:\n"
       "`LLM_BOOTSTRAP_SIZE` first, then `BASE_SIZE`, then `HITL_SIZE`, then the remainder."),

    code("LLM_BOOTSTRAP_SIZE = 10_000\n"
         "BASE_SIZE          = 100_000\n"
         "HITL_SIZE          = 200_000\n"
         "\n"
         "required = LLM_BOOTSTRAP_SIZE + BASE_SIZE + HITL_SIZE\n"
         "if len(df) < required:\n"
         "    print(f'Warning: dataset has {len(df):,} rows, less than the {required:,} required; shrinking later partitions.')\n"
         "    LLM_BOOTSTRAP_SIZE = min(len(df), LLM_BOOTSTRAP_SIZE)\n"
         "    BASE_SIZE          = min(len(df) - LLM_BOOTSTRAP_SIZE, BASE_SIZE)\n"
         "    HITL_SIZE          = max(0, len(df) - LLM_BOOTSTRAP_SIZE - BASE_SIZE)\n"
         "\n"
         "a = LLM_BOOTSTRAP_SIZE\n"
         "b = a + BASE_SIZE\n"
         "c = b + HITL_SIZE\n"
         "\n"
         "llm_bootstrap_df = df.iloc[:a].copy()\n"
         "base_df          = df.iloc[a:b].copy()\n"
         "hitl_df          = df.iloc[b:c].copy()\n"
         "inference_df     = df.iloc[c:].copy()\n"
         "\n"
         "print(f'LLM bootstrap: {len(llm_bootstrap_df):,}')\n"
         "print(f'Base:          {len(base_df):,}')\n"
         "print(f'HITL:          {len(hitl_df):,}')\n"
         "print(f'Inference:     {len(inference_df):,}')"),

    md("## Engagement Distribution by Partition\n\n"
       "Sanity check that the attention-weighted shuffle stratified the partitions as intended.\n"
       "On a heavy-tailed corpus you should see the LLM Bootstrap slice carry a much higher\n"
       "median / mean / max engagement than the Inference slice."),

    code("def _summarise(name, part):\n"
         "    if not len(part):\n"
         "        print(f'  {name:>14}: (empty)')\n"
         "        return\n"
         "    att = (part['likes'] + part['retweets']).astype(int)\n"
         "    print(f'  {name:>14}: n={len(part):>9,}  median={att.median():>6.0f}  mean={att.mean():>9.1f}  '\n"
         "          f'p95={att.quantile(0.95):>7.0f}  max={att.max():>9,}')\n"
         "\n"
         "print('Engagement (likes + retweets) by partition:')\n"
         "_summarise('LLM bootstrap', llm_bootstrap_df)\n"
         "_summarise('Base',          base_df)\n"
         "_summarise('HITL',          hitl_df)\n"
         "_summarise('Inference',     inference_df)"),

    md("## Save Partitions"),

    code("hitl_folder.mkdir(parents=True, exist_ok=True)\n"
         "\n"
         "llm_bootstrap_df.to_pickle(hitl_folder / 'llm_bootstrap_dataset.pkl')\n"
         "base_df.to_pickle(hitl_folder / 'base_dataset.pkl')\n"
         "inference_df.to_pickle(hitl_folder / 'inference_dataset.pkl')\n"
         "\n"
         "BATCH_SIZE   = 50_000\n"
         "n_batches    = int(np.ceil(len(hitl_df) / BATCH_SIZE)) if len(hitl_df) else 0\n"
         "hitl_batches = {}\n"
         "for i in range(n_batches):\n"
         "    chunk = hitl_df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]\n"
         "    out   = hitl_folder / f'hitl_pending_batch_{i+1:02d}.pkl'\n"
         "    chunk.to_pickle(out)\n"
         "    hitl_batches[f'hitl_batch_{i+1:02d}'] = chunk\n"
         "    print(f'Saved {out.name} ({len(chunk):,} tweets)')"),

    md("## Partition Manifest\n\n"
       "Write a `partition_ids.pkl` mapping every partition name to its tweet IDs and assert\n"
       "that the partitions are pairwise disjoint. Downstream notebooks can load this manifest\n"
       "to verify which subset any tweet belongs to."),

    code("def _ids(d):\n"
         "    return d['id'].astype(str).tolist() if 'id' in d.columns else d.index.astype(str).tolist()\n"
         "\n"
         "partition_ids = {\n"
         "    'llm_bootstrap': _ids(llm_bootstrap_df),\n"
         "    'base':          _ids(base_df),\n"
         "    'inference':     _ids(inference_df),\n"
         "}\n"
         "for name, batch_df in hitl_batches.items():\n"
         "    partition_ids[name] = _ids(batch_df)\n"
         "\n"
         "seen = set()\n"
         "for name, ids in partition_ids.items():\n"
         "    s = set(ids)\n"
         "    overlap = s & seen\n"
         "    assert not overlap, f'Partition {name!r} overlaps existing partitions on {len(overlap)} ids'\n"
         "    seen |= s\n"
         "    print(f'  {name:>16}: {len(ids):>10,} ids')\n"
         "print(f'  {\"TOTAL\":>16}: {len(seen):>10,} unique ids — all partitions disjoint')\n"
         "\n"
         "manifest_path = hitl_folder / 'partition_ids.pkl'\n"
         "with open(manifest_path, 'wb') as f:\n"
         "    pickle.dump(partition_ids, f)\n"
         "print(f'Wrote → {manifest_path}')"),

    md("## (Optional) Export Human Seed Review Batch\n\n"
       "If you intend to run the **human-only** seed path (label 10 000 tweets by hand), this\n"
       "cell exports the seed CSV. Skip it if you are using the **LLM bootstrap** path\n"
       "(`01_llm_bootstrap_labelling.ipynb`) — both paths produce the same downstream artifact:\n"
       "a labelled CSV that `02_hitl_training_loop.ipynb` ingests as initial training data."),

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
    write_notebook(f"{NB_DIR}/00_hitl_data_preparation.ipynb",   nb0_cells)
    write_notebook(f"{NB_DIR}/01_llm_bootstrap_labelling.ipynb", nb1_cells)
    write_notebook(f"{NB_DIR}/02_hitl_training_loop.ipynb",      nb2_cells)
    write_notebook(f"{NB_DIR}/03_final_inference.ipynb",         nb3_cells)
    write_notebook(f"{NB_DIR}/04_full_dataset_inference.ipynb",  nb4_cells)


