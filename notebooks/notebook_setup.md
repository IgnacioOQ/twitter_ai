# Notebook Setup Guide
- status: active
- type: guideline
- id: notebook_setup_guide
- last_checked: 2026-07-15
<!-- content -->
This document defines the canonical structure for the **Setup section** of every notebook in this project. All notebooks must follow this pattern to ensure consistency and correct execution both locally and on Google Colab.

## Setup Section Structure
- status: active
<!-- content -->
The Setup section is always the **first section** of the notebook. It consists of the following cells, in order:

### Cell 1 — Environment Switch & Paths
- status: active
<!-- content -->
A single code cell that:
1. Detects whether the notebook is running locally or on Colab via the `RUNNING_LOCALLY` flag.
2. Mounts Google Drive (Colab) or adds the repo root to `sys.path` (local).
3. Defines all shared folder path variables used throughout the notebook.

```python
import os
from pathlib import Path
import sys

# --- ENVIRONMENT SWITCH ---
# Set to True if running on local machine with Google Drive Desktop mounted
# Set to False if running in Google Colab cloud
RUNNING_LOCALLY = False

if RUNNING_LOCALLY:
    # --- REPO ROOT ON sys.path (so `from src.*` works locally) ---
    _REPO_ROOT = str(Path(os.getcwd()).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    # Standard macOS path for Google Drive Desktop
    BASE_PATH = Path('/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust')

else:
    # Google Colab cloud path
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/My Drive/Colab Projects/AI Public Trust')

# Pre-compute critical paths used across notebooks
twits_folder      = BASE_PATH / 'Raw Data/Twits/'
test_folder       = BASE_PATH / 'Raw Data/'
datasets_folder   = BASE_PATH / 'Data Sets'
cleanedds_folder  = BASE_PATH / 'Data Sets/Cleaned Data'
networks_folder   = BASE_PATH / 'Data Sets/Networks/'
literature_folder = BASE_PATH / 'Literature/'
topic_models_folder = BASE_PATH / 'Models/Topic Modeling/'
```

### Cell 2 — Git Clone (Colab only, when needed)
- status: active
<!-- content -->
Required only for notebooks that import from the `src/` package while running on Colab (since Colab has no local copy of the repo). Skip this cell for notebooks that do not use `src` imports.

The cell is always guarded by `if not RUNNING_LOCALLY` so it is a no-op locally.

```python
import os
if not RUNNING_LOCALLY:
    print('Running Colab setup shell commands...')
    !git clone https://github.com/IgnacioOQ/twitter_ai.git
else:
    print('Running locally: Skipping Colab shell setup.')
```

Followed immediately by a cell that changes into the cloned repo **and adds it to `sys.path`** so that `from src.*` imports resolve correctly on Colab:

```python
import sys, os
if not RUNNING_LOCALLY:
    os.chdir('twitter_ai')
    _repo_root = os.getcwd()
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
```

> **Note:** Use `os.chdir()` instead of the `%cd` magic so the call can be guarded by `if not RUNNING_LOCALLY`. The `sys.path` update mirrors what Cell 1 already does for the local branch.

### Cell 3 — pip Installs (Colab only, when needed)
- status: active
<!-- content -->
Required only for packages that are not pre-installed in Colab (e.g. `igraph`, `leidenalg`, `powerlaw`). Always guarded by `if not RUNNING_LOCALLY`.

```python
import os
if not RUNNING_LOCALLY:
    print('Running Colab setup shell commands...')
    !pip install igraph leidenalg powerlaw
else:
    print('Running locally: Skipping Colab shell setup.')
```

Only install what the specific notebook actually needs. Do not add blanket installs.

### Cell 4 — Explicit Library Imports
- status: active
<!-- content -->
A single code cell listing **all** third-party and standard-library imports the notebook uses, written **explicitly** — one import per line.

**Rules:**
- Do **not** use wildcard imports (`from module import *`) for external libraries.
- Do **not** use `from src.network.imports import *` — all imports must be listed here directly.
- Group imports in the following order: standard library → third-party → IPython/Jupyter utilities → `pathlib`.
- Only import what the notebook actually uses.

**Example (network analysis notebooks):**

```python
from datetime import datetime, timedelta, timezone
import json
import os
import pickle
import random
import re
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import pytz
import seaborn as sns
import tqdm
from IPython.display import Javascript
from pathlib import Path
```

**Example (processing notebooks):**

```python
from datetime import datetime, timedelta, timezone
import json, heapq, itertools, csv
import os
import pickle
import random
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from pathlib import Path
```

### Cell 5 — src Imports (when applicable)
- status: active
<!-- content -->
If the notebook uses functions from the `src/` package, import them **after** the library imports. Wildcard imports are acceptable here since the `src` modules are internal and well-controlled.

```python
from src.network.network_utils import *
from src.network.network_pruning import *
from src.network.network_modularity import *
```

Only import the `src` submodules that the notebook actually uses.

## Summary Table
- status: active
<!-- content -->

| Cell | Content | Required? |
| :--- | :--- | :--- |
| 1 | Environment switch + paths | Always |
| 2 | `git clone` + `%cd` | Only if `src/` imports are needed on Colab |
| 3 | `pip install` | Only if non-default packages are needed |
| 4 | Explicit library imports | Always |
| 5 | `src/` imports | Only if the notebook uses internal `src` modules |

## Teardown Section — Disconnect from Runtime
- status: active
<!-- content -->
Every **processing notebook** should **finish** with a short teardown section
whose final cell disconnects from the Colab runtime, freeing the GPU/instance
for other work. This matters most for GPU notebooks, where an idle runtime keeps
a scarce accelerator reserved after the work is done.

Place it as the **last section** of the notebook, after all artifacts have been
written to Drive. Guard it with `if not RUNNING_LOCALLY` so it is a no-op
locally, and be aware that it **terminates the runtime** — any unsaved in-memory
state is lost, so run it only once you are truly finished.

```python
# ---- Disconnect from the Colab runtime (frees the GPU/instance) ------------ #
# Run this LAST, once artifacts are saved: it terminates the runtime, so any
# unsaved in-memory state is lost. No-op locally.
if not RUNNING_LOCALLY:
    from google.colab import runtime
    runtime.unassign()
```

## Google Drive Data Directory Structure
- status: active
<!-- content -->
All data lives under `My Drive/Colab Projects/AI Public Trust/` (the `BASE_PATH`).
The path variables defined in Cell 1 map to the following folders and canonical files:

```
AI Public Trust/
├── Raw Data/                                  (test_folder = BASE_PATH / 'Raw Data/')
│   ├── testing.json                           ← single test batch (raw API)
│   └── Twits/                                 (twits_folder = BASE_PATH / 'Raw Data/Twits/')
│       └── tweets_YYYY-MM-DDTHH:MM:SS.json   ← raw API harvest files
│
├── Data Sets/                                 (datasets_folder = BASE_PATH / 'Data Sets')
│   ├── AItrust_twits_dict_test.json           ← unprocessed test twit dict   [written by 02_Processing/01]
│   ├── AItrust_author_dict_test.json          ← unprocessed test author dict [written by 02_Processing/01]
│   ├── AItrust_twits_dict.json                ← unprocessed full twit dict   [written by 02_Processing/01]
│   ├── AItrust_author_dict.json               ← unprocessed full author dict [written by 02_Processing/01]
│   │
│   ├── Cleaned Data/                          (cleanedds_folder = BASE_PATH / 'Data Sets/Cleaned Data')
│   │   ├── AItrust_twits_pruned_dict_test.json          [written by 02_Processing/02]
│   │   ├── AItrust_Art_pruned_twit_dict_test.json       [written by 02_Processing/02]
│   │   ├── AItrust_twits_pruned_dict.json               [written by 02_Processing/02]
│   │   ├── AItrust_Art_pruned_twit_dict.json            [written by 02_Processing/02]
│   │   ├── top_test_ai_tweets.csv                       [written by 02_Processing/02]
│   │   ├── top_test_art_tweets.csv                      [written by 02_Processing/02]
│   │   ├── test_basic_counts_dict.pkl                   [written by 02_Processing/02]
│   │   ├── test_timeline_dict.pkl                       [written by 02_Processing/02]
│   │   ├── test_author_corpus_dict.pkl                  [written by 02_Processing/02]
│   │   ├── full_basic_counts_dict.pkl                   [written by 02_Processing/02]
│   │   ├── full_timeline_dict.pkl                       [written by 02_Processing/02]
│   │   ├── full_author_corpus_dict.pkl                  [written by 02_Processing/02]
│   │   ├── AItrust_twits_pruned_dict_test_with_sentiment.json   [written by 03_Analysis/01]
│   │   ├── AItrust_pruned_twits_with_sentiment.json             [written by 03_Analysis/01]
│   │   ├── AItrust_pruned_twits_with_sentiment_cleaned.json     [written by 02_Processing/03]
│   │   ├── AItrust_topics_k5_metadata.json                      [written by 03_Analysis/03]
│   │   ├── AItrust_pruned_twits_with_sentiment_and_topics_k5.json     [written by 03_Analysis/03]
│   │   ├── AItrust_pruned_twits_with_sentiment_and_topics_k5.jsonl.gz [written by 03_Analysis/03]
│   │   └── top_retweets_by_topic_100.csv                        [written by 02_Processing/03]
│   │
│   └── Networks/                              (networks_folder = BASE_PATH / 'Data Sets/Networks/')
│       ├── test_network_dict.pkl              [written by 02_Processing/02]
│       ├── full_network_dict.pkl              [written by 02_Processing/02]
│       ├── Test_Network.json                  [written by 02_Processing/02]
│       ├── Full_Network.json                  [written by 02_Processing/02]
│       └── Full_Network.gml                  [used by 04_Network_Analysis/01]
│
├── Literature/                                (literature_folder = BASE_PATH / 'Literature/')
└── Models/
    └── Topic Modeling/                        (topic_models_folder = BASE_PATH / 'Models/Topic Modeling/')
        └── lda_k5_topics_metadata.json        [written by 03_Analysis/03]
```

## Common Mistakes
- status: active
<!-- content -->
- **Using `from src.network.imports import *`** — this hides what is actually imported and makes the notebook harder to debug. Always list imports explicitly in Cell 4.
- **Missing `igraph as ig`** — `igraph` must be imported explicitly even though it is pip-installed in Cell 3.
- **Putting imports before pip installs** — Cell 4 must come after Cell 3, since packages like `igraph` and `leidenalg` must be installed before they can be imported.
