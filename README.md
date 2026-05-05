# Twitter AI Dataset Analysis

Research pipeline for studying public discourse and trust around **artificial intelligence** on Twitter / X. The repository ingests raw Twitter API data, cleans and structures it into tweet/author dictionaries, runs sentiment and topic models, builds and analyses the retweet/mention network, and trains a Human-in-the-Loop (HITL) classifier that is then applied to the full ~17M-tweet corpus.

The work centres on a single longitudinal dataset (`AI Public Trust`) and is written as a sequence of numbered Jupyter notebooks designed to run on **Google Colab** with the data persisted to **Google Drive**. A shared `src/` package holds reusable Python (network construction, pruning, modularity) so notebooks stay focused on analysis steps.

## Quick Orientation

- **Subject:** AI-related tweets, the authors who post them, and the network they form.
- **Compute:** Google Colab (primary) or local Python with Google Drive Desktop mounted.
- **Storage:** Google Drive — `My Drive/Colab Projects/AI Public Trust/`.
- **Language:** Python 3.10, Jupyter notebooks, plus markdown definitions for AI agents and conventions.
- **Entry point for new readers:** [notebooks/notebook_setup.md](notebooks/notebook_setup.md) — describes how every notebook is structured and which Drive paths it touches.

## Repository Structure

```
twitter_ai/
├── README.md                    ← this file
├── TODO_WORKFLOW.md             ← cross-session task backlog for coding agents
├── notebooks/                   ← the analytical pipeline (see below)
│   ├── notebook_setup.md        ← canonical setup pattern for every notebook
│   ├── notebook_authoring.md    ← authoring conventions
│   ├── create_notebook.py       ← scaffolding helper
│   ├── 01_Ingestion/
│   ├── 02_Processing/
│   ├── 03_Analysis_and_Modeling/
│   ├── 04_Network_Analysis/
│   ├── 05_Classifiers/          ← HITL classification (see classification_strategy.md)
│   └── 06_Experiments/
├── src/                         ← reusable Python package, importable as `from src.*`
│   ├── network/                 ← graph generation, pruning, modularity, utilities
│   └── scripts/                 ← one-off maintenance scripts (path fixes, env-switch injection, etc.)
├── agents/                      ← markdown definitions for AI agents that operate on the repo
│   ├── MC_AGENT.md              ← Master Control Agent
│   ├── MCP_AGENT.md             ← MCP Agent (and MCP_AGENT_EXTENDED.md)
│   ├── LINEARIZE_AGENT.md
│   ├── NOTEBOOK_SKILL.md
│   └── AGENTS_LOG.md            ← intervention history
└── docs/                        ← long-form documentation
    ├── MD_CONVENTIONS.md        ← markdown / metadata conventions
    ├── MCP_EXPLANATION.md
    └── MCP_SKLEARN_PLAN.md
```

## Notebook Pipeline

The pipeline is divided into six sequential stages. Every notebook follows the [Setup pattern](notebooks/notebook_setup.md) (environment switch, Drive mount, optional repo clone, explicit imports, `src/` imports) so it runs identically locally and on Colab.

| Stage | Folder | What it does |
| :--- | :--- | :--- |
| 1 | [01_Ingestion/](notebooks/01_Ingestion/) | Configure shared Drive folders; mine raw tweets from the Twitter API into `Raw Data/Twits/`. |
| 2 | [02_Processing/](notebooks/02_Processing/) | Convert raw API JSON to tweet/author dictionaries; run sanity checks; build the retweet/mention network; clean tweet text. |
| 3 | [03_Analysis_and_Modeling/](notebooks/03_Analysis_and_Modeling/) | Sentiment analysis (Twitter-RoBERTa); example extraction; LDA topic models for tweets and authors; embedding maps. |
| 4 | [04_Network_Analysis/](notebooks/04_Network_Analysis/) | Pure graph analysis on the `igraph`/`networkx` network: degree distributions, modularity (Leiden), pruning, power-law fits. |
| 5 | [05_Classifiers/](notebooks/05_Classifiers/) | Human-in-the-Loop active learning (LightGBM + Twitter-RoBERTa); final inference on the HITL remainder; mass inference on the full ~17M-tweet corpus. See [classification_strategy.md](notebooks/05_Classifiers/classification_strategy.md). |
| 6 | [06_Experiments/](notebooks/06_Experiments/) | Isolated probes (e.g. TP-Bigrams) that do not feed downstream stages. |

Notebooks are numbered within each stage (`01_*`, `02_*`, ...) and use snake_case names. Outputs of stage *N* are inputs of stage *N+1*.

## Google Drive Data Layout

All persistent data lives under `My Drive/Colab Projects/AI Public Trust/` (referred to as `BASE_PATH` in every notebook). The structure below is the canonical layout — see [notebooks/notebook_setup.md](notebooks/notebook_setup.md) for the authoritative version with per-file write provenance.

```
AI Public Trust/                                  (BASE_PATH)
├── Raw Data/
│   ├── testing.json                              ← single test batch (raw API)
│   └── Twits/
│       └── tweets_YYYY-MM-DDTHH:MM:SS.json       ← raw API harvest files
│
├── Data Sets/
│   ├── AItrust_twits_dict[_test].json            ← unprocessed tweet dicts   [02/01]
│   ├── AItrust_author_dict[_test].json           ← unprocessed author dicts  [02/01]
│   │
│   ├── Cleaned Data/
│   │   ├── AItrust_twits_pruned_dict[_test].json                       [02/02]
│   │   ├── AItrust_Art_pruned_twit_dict[_test].json                    [02/02]
│   │   ├── {test,full}_basic_counts_dict.pkl                           [02/02]
│   │   ├── {test,full}_timeline_dict.pkl                               [02/02]
│   │   ├── {test,full}_author_corpus_dict.pkl                          [02/02]
│   │   ├── AItrust_pruned_twits_with_sentiment[_cleaned].json          [03/01, 02/03]
│   │   ├── AItrust_topics_k5_metadata.json                             [03/03]
│   │   ├── AItrust_pruned_twits_with_sentiment_and_topics_k5.json(.gz) [03/03]
│   │   ├── top_test_{ai,art}_tweets.csv                                [02/02]
│   │   └── top_retweets_by_topic_100.csv                               [02/03]
│   │
│   ├── Networks/
│   │   ├── {test,full}_network_dict.pkl                                [02/02]
│   │   ├── {Test,Full}_Network.json                                    [02/02]
│   │   └── Full_Network.gml                                            [04/01]
│   │
│   └── Classifiers_Data/
│       ├── HITL/
│       │   ├── base_dataset.pkl, inference_dataset.pkl
│       │   ├── hitl_pending_batch_0{1..4}.pkl
│       │   └── hitl_review_batch_0{0..4}.csv     ← human-labelled rounds
│       ├── Final/
│       │   └── final_annotated_tweets.{pkl,csv}
│       └── Full_Inference/
│           ├── checkpoint_*.pkl                  ← every 100k tweets
│           └── full_inference_annotated.{pkl,csv}
│
├── Literature/                                   ← reference papers, notes
└── Models/
    ├── Topic Modeling/
    │   └── lda_k5_topics_metadata.json           [03/03]
    └── Classifiers/
        ├── best_roberta_model/                   ← Twitter-RoBERTa weights + tokenizer
        ├── lgb_embed.txt                         ← LightGBM (sentence embeddings)
        └── lgb_bow.txt                           ← LightGBM (bag of words)
```

The bracketed tags (`[02/02]`, `[03/03]`, ...) indicate which notebook *stage/index* writes each file.

## Setup

### Option 1 — Google Colab (recommended)

1. Open any notebook directly from this repo on Colab.
2. The first cell mounts your Google Drive at `/content/drive` and resolves `BASE_PATH` to `MyDrive/AI Public Trust`.
3. If the notebook imports from `src/`, the second cell runs `git clone https://github.com/IgnacioOQ/twitter_ai.git` into the Colab session and adds the repo root to `sys.path`.

### Option 2 — Local

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/) and ensure `My Drive/Colab Projects/AI Public Trust/` is synced.
2. Set `RUNNING_LOCALLY = True` in Cell 1 of the notebook.
3. Create a Python 3.10 venv and install dependencies as needed (typical stack: `pandas`, `numpy`, `scikit-learn`, `igraph`, `leidenalg`, `powerlaw`, `networkx`, `transformers`, `sentence-transformers`, `lightgbm`, `tqdm`, `matplotlib`, `seaborn`).

## Conventions

- **Notebook setup:** every notebook follows the five-cell pattern in [notebooks/notebook_setup.md](notebooks/notebook_setup.md). Wildcard imports from third-party libraries are forbidden; `from src.*` wildcards are acceptable.
- **Markdown documents:** governance / agent / log files follow the metadata schema in [docs/MD_CONVENTIONS.md](docs/MD_CONVENTIONS.md).
- **Cross-session tasks for coding agents:** [TODO_WORKFLOW.md](TODO_WORKFLOW.md) at the repo root holds pending tasks; agents pick them up and remove them on completion.

## Status

Active research project. The processing, analysis, and network stages are stable; the HITL classifier loop in stage 05 is the most recently added pipeline component. See `agents/AGENTS_LOG.md` for an intervention history.
