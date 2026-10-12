# Twitter AI Dataset Analysis

Research pipeline for studying public discourse and trust around **artificial intelligence** on Twitter / X. The repository ingests raw Twitter API data, cleans and structures it into tweet/author dictionaries, builds the retweet network and its author backbones, detects communities, fits author-level topic, sentiment and emotion profiles, renders the figures and an interactive author map, and trains a Human-in-the-Loop (HITL) classifier that is then applied to the full ~17M-tweet corpus.

The work centres on a single longitudinal dataset (`AI Public Trust`) and is written as a sequence of numbered Jupyter notebooks designed to run on **Google Colab** with the data persisted to **Google Drive**. A shared `src/` package holds reusable Python (network construction, pruning, modularity, the author analysis and the figure renderers); the later stages also run from the command line.

## Quick Orientation

- **Subject:** AI-related tweets, the authors who post them, and the network they form.
- **Compute:** Google Colab, local Python or HPC; downstream commands accept explicit input/output directories.
- **Storage:** Google Drive — `My Drive/Colab Projects/AI Public Trust/`.
- **Language:** Python 3.10, Jupyter notebooks, plus markdown definitions for AI agents and conventions.
- **Entry point for new readers:** [notebooks/notebook_setup.md](notebooks/notebook_setup.md) — describes how every notebook is structured and which Drive paths it touches.
- **Pipeline trace:** [docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md](docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md) — tweet and author counts at every stage, the code behind them, runs done and owed, and an inventory of every file.
- **Dataset statistics:** [docs/DATASET_STATISTICS.md](docs/DATASET_STATISTICS.md) — narrative report of collection, filtering, networks, communities and author topics, with the full summary statistics.
- **Pipeline graph:** [docs/pipeline_graph_notebooks.png](docs/pipeline_graph_notebooks.png), built by [src/scripts/pipeline_graph.py](src/scripts/pipeline_graph.py) (see [Pipeline Graph](#pipeline-graph)).
- **Author analysis:** [src/author_analysis/README.md](src/author_analysis/README.md) — author profiles, topic fitting, matching and summaries.
- **Community analysis and figures:** [docs/blog_analysis/README.md](docs/blog_analysis/README.md) — aggregate tables and light/dark figure commands.
- **Model-specific configuration:** [config/retweeted_once_k12](config/retweeted_once_k12) — twelve-topic settings, reviewed labels, shared colours, artifact fingerprints and result links. Generated datasets, models and figures stay outside version control.
- **2D/3D network viewer:** [visualizations/author_network/README.md](visualizations/author_network/README.md) — browser viewer and layout tools.

## Repository Structure

```
twitter_ai/
├── README.md                    ← this file
├── notebooks/                   ← the analytical pipeline (see below)
│   ├── notebook_setup.md        ← canonical setup pattern for every notebook
│   ├── notebook_authoring.md    ← authoring conventions
│   ├── create_notebook.py       ← scaffolding helper
│   ├── analyze_notebooks.py     ← structural checks (Colab cell shape, foldable sections)
│   ├── 01_Ingestion/
│   ├── 02_Processing/
│   ├── 03_Analysis_and_Modeling/
│   ├── 04_Network_Analysis/
│   ├── 05_Classifiers/          ← HITL classification (see classification_strategy.md)
│   └── 06_Experiments/
├── src/                         ← reusable Python package, importable as `from src.*`
│   ├── network/                 ← graph generation, pruning, modularity, utilities
│   │   └── visualization/       ← matched-author layout, 3D layout and viewer export
│   ├── author_analysis/         ← author topics, affect, matching, community comparison (CLI + notebook)
│   ├── blog_analysis/           ← weekly aggregation and the figure renderers
│   └── scripts/                 ← pipeline_graph.py and one-off maintenance scripts
├── config/
│   └── retweeted_once_k12/      ← settings, labels and provenance of the twelve-topic fit
├── visualizations/
│   └── author_network/          ← offline WebGL author viewer
├── tests/                       ← unit tests (python -m unittest discover -s tests)
├── agents/                      ← markdown definitions for AI agents that operate on the repo
│   ├── MC_AGENT.md              ← Master Control Agent
│   ├── MCP_AGENT.md             ← MCP Agent (and MCP_AGENT_EXTENDED.md)
│   ├── LINEARIZE_AGENT.md
│   ├── NOTEBOOK_SKILL.md
│   └── AGENTS_LOG.md            ← intervention history
└── docs/                        ← long-form documentation
    ├── AUTHOR_AND_NETWORK_PIPELINE_TRACE.md  ← stage-by-stage counts, runs owed, file inventory
    ├── DATASET_STATISTICS.md    ← dataset report and summary statistics
    ├── TOPIC_EMOTION_MIXING_PLAN.md  ← plan: topic × emotion joint matrices
    ├── blog_analysis/README.md  ← figure commands
    ├── MD_CONVENTIONS.md        ← markdown / metadata conventions
    ├── COLAB_MCP_WORKFLOW.md
    ├── pipeline_graph.png       ← bipartite map of notebooks and scripts ↔ Drive artifacts
    ├── pipeline_graph_notebooks.png  ← projected DAG of notebooks and scripts (overview)
    ├── pipeline_graph.json      ← persisted graph, queryable with `jq`
    └── pipeline_overrides.yaml  ← what the graph builder cannot parse: declared edges and src/ scripts
```

## Pipeline Graph

[src/scripts/pipeline_graph.py](src/scripts/pipeline_graph.py) derives a producer/consumer graph of the whole pipeline: notebooks and `src/` scripts on one side, the Drive files they read and write on the other. It parses each notebook's code for the files it opens. What it cannot parse — paths built inside helpers or `src/` workflows, and the `src/` stages that run from the command line — is declared in [docs/pipeline_overrides.yaml](docs/pipeline_overrides.yaml). Run it from the repository root:

```bash
python3 src/scripts/pipeline_graph.py              # rebuild docs/pipeline_graph.{json,png} and docs/pipeline_graph_notebooks.png
python3 src/scripts/pipeline_graph.py validate     # check the [stage/index] tags in this README and notebook_setup.md against the graph
python3 src/scripts/pipeline_graph.py upstream   03_Analysis_and_Modeling/04_lda_author_topics_hpc
python3 src/scripts/pipeline_graph.py downstream 02_Processing/02_sanity_check_and_network_generation
```

It needs `nbformat`, `networkx`, `matplotlib` and `pyyaml`. In the images, squares are notebooks and diamonds are `src/` scripts. Rebuild and validate after any change to a notebook or a `src/` stage; how to read the graph is in [§12 of the pipeline trace](docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md#12-the-pipeline-graph).

## Notebook Pipeline

The notebooks are divided into six stages. Every notebook follows the [Setup pattern](notebooks/notebook_setup.md) (environment switch, Drive mount, optional repo clone, explicit imports, `src/` imports) so it runs identically locally and on Colab.

| Stage | Folder | What it does |
| :--- | :--- | :--- |
| 1 | [01_Ingestion/](notebooks/01_Ingestion/) | Configure shared Drive folders; mine raw tweets from the Twitter API into `Raw Data/Twits/`. |
| 2 | [02_Processing/](notebooks/02_Processing/) | Convert raw API JSON to tweet/author dictionaries; filter to the AI and AI+Art corpora; build the retweet network and cut its two author backbones (retweeted once, LWCC). |
| 3 | [03_Analysis_and_Modeling/](notebooks/03_Analysis_and_Modeling/) | Sentiment and emotion classification (Twitter-RoBERTa); example extraction; tweet-level LDA; author-level topics, affect and community comparison per backbone (`04`, which runs `src/author_analysis`); embedding maps. |
| 4 | [04_Network_Analysis/](notebooks/04_Network_Analysis/) | Community detection with five methods on each backbone, degree distributions and power-law fits; GPU maps of a backbone. |
| 5 | [05_Classifiers/](notebooks/05_Classifiers/) | Human-in-the-Loop active learning (LightGBM + Twitter-RoBERTa); final inference on the HITL remainder; mass inference on the full ~17M-tweet corpus. See [classification_strategy.md](notebooks/05_Classifiers/classification_strategy.md). |
| 6 | [06_Experiments/](notebooks/06_Experiments/) | Isolated probes (e.g. TP-Bigrams) that do not feed downstream stages. |

Notebooks are numbered within each stage (`01_*`, `02_*`, ...) and use snake_case names. The order is not strictly by stage number: the author analysis in stage 3 reads the community partitions of stage 4. After it, command-line stages in `src/` build the figure inputs, the weekly series, figures 1–11 and the author viewer ([docs/blog_analysis/README.md](docs/blog_analysis/README.md)).

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
│   │   ├── ai_full_classified_<model>.json                             ← per-tweet sentiment/emotion (01b, copied from the cluster)
│   │   ├── AItrust_pruned_twits_with_sentiment.json                    [03/01]
│   │   ├── AItrust_topics_k5_metadata.json                             [03/03]
│   │   ├── AItrust_pruned_twits_with_sentiment_and_topics_k5.json      [03/03]
│   │   ├── Partitioned Data/AI Data/                                   ← HITL partitions (base, inference, pending batches)  [05/00]
│   │   └── Tweet Sheets/
│   │       ├── top_test_ai_tweets.csv                                  [03/02]
│   │       └── top_test_art_tweets.csv                                 [03/02]
│   │
│   ├── Networks/                                                     (one subfolder per stage since 2026-10-10)
│   │   ├── 1_retweet_dicts/full_network_dict.pkl                       [02/02]
│   │   ├── 2_full_graphs/Full_Network_{Influence,InfoFlow}.gml         [02/02]
│   │   ├── 3_backbones/Full_RetweetedOnce_{Influence,InfoFlow}.gml     [02/02]  ← authors retweeted ≥ 1x
│   │   ├── 3_backbones/Full_LWCC_{Influence,InfoFlow}.gml              [02/02]  ← LWCC, self-loops removed (direction-neutral)
│   │   ├── 4_communities/{RetweetedOnce,LWCC}/<stem>_InfoFlow_<method>.gml   [04/01]
│   │   ├── 4_communities/{RetweetedOnce,LWCC}/<stem>_author_communities.json [04/01]
│   │   ├── 5_visualizations/<NETWORK>/                                 [04/02]
│   │   └── test/                                                       [02/02]  ← test_network_dict.pkl and every Test_* graph
│   │
│   └── Classifiers_Data/
│       ├── HITL/
│       │   ├── llm_bootstrap_labels.csv          ← LLM seed labels
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
    │   ├── LDA/lda_k5_topics_metadata.json       [03/03]
    │   └── LDA/backbone_analysis/networks/<backbone>/   ← author topics, affect, tables, figure inputs, figures, viewer (03/04 + src/)
    └── Classifiers/
        ├── best_roberta_model/                   ← Twitter-RoBERTa weights + tokenizer
        ├── lgb_embed.txt                         ← LightGBM (sentence embeddings)
        └── lgb_bow.txt                           ← LightGBM (bag of words)
```

The bracketed tags (`[02/02]`, `[03/03]`, ...) indicate which notebook *stage/index* writes each file.

## Setup

### Option 1 — Google Colab (recommended)

1. Open any notebook directly from this repo on Colab.
2. The first cell mounts your Google Drive at `/content/drive` and resolves `BASE_PATH` to `My Drive/Colab Projects/AI Public Trust`.
3. If the notebook imports from `src/`, the second cell runs `git clone https://github.com/IgnacioOQ/twitter_ai.git` into the Colab session and adds the repo root to `sys.path`.

### Option 2 — Local

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/) and ensure `My Drive/Colab Projects/AI Public Trust/` is synced.
2. Set `RUNNING_LOCALLY = True` in Cell 1 of the notebook.
3. Create a Python 3.10+ venv and install dependencies as needed (typical stack: `pandas`, `numpy`, `scikit-learn`, `igraph`, `leidenalg`, `powerlaw`, `networkx`, `transformers`, `sentence-transformers`, `lightgbm`, `tqdm`, `matplotlib`, `seaborn`).

## Author-content workflow

The authoritative author workflow is [src/author_analysis](src/author_analysis/README.md).
Its notebook and command-line entry points call the same implementation: all post types,
network-defined author populations, and separate topic models for LWCC and RetweetedOnce.
The same fitted bundles feed sentiment/emotion summaries, weekly analysis and the network viewer.
The retweeted-once backbone has been fitted (164,291 authors, K = 12; [config/retweeted_once_k12](config/retweeted_once_k12));
the LWCC backbone has not.

## Conventions

- **Notebook setup:** every notebook follows the five-cell pattern in [notebooks/notebook_setup.md](notebooks/notebook_setup.md). Wildcard imports from third-party libraries are forbidden; `from src.*` wildcards are acceptable.
- **Markdown documents:** governance / agent / log files follow the metadata schema in [docs/MD_CONVENTIONS.md](docs/MD_CONVENTIONS.md).
- **Pipeline documentation:** every change to a notebook, a `src/` stage or the Drive layout updates [docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md](docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md) and rebuilds the [pipeline graph](#pipeline-graph).

## Status

Active research project. Corpus, networks, communities and the retweeted-once author analysis are complete; the runs still owed (LWCC author analysis and map, sentiment-file verification, tweet-level topics) are listed in [§11 of the pipeline trace](docs/AUTHOR_AND_NETWORK_PIPELINE_TRACE.md#11-runs-owed). See `agents/AGENTS_LOG.md` for an intervention history.
