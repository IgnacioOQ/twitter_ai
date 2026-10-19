---
status: active
type: reference
id: twitter_ai.author_and_network_pipeline_trace
description: Step-by-step trace of the pipeline — how many tweets and authors exist at each stage, the code that decided it, which run produced each number, what is still owed, and an inventory of every file the notebooks and src/ stages read or write.
label: [dataset, network, authors, provenance, audit]
volatility: evolving
scope: project-specific
repository: [twitter_ai]
last_checked: '2026-10-12'
---

# Author & Network Pipeline Trace

This document answers one question for every step of the pipeline: **how many tweets and authors are there at this point, and which code decided that?** [DATASET_STATISTICS.md](DATASET_STATISTICS.md) gives the narrative and the master statistics; this document adds the code, the run that produced each number, what is still owed, and the file inventory (§13). Where the two overlap, the numbers agree.

Numbers are copied from the outputs stored in the committed notebooks, except the author-topic numbers of §7, which come from [config/retweeted_once_k12/provenance.json](../config/retweeted_once_k12/provenance.json) (that run is not stored in a notebook).

> **Status 2026-10-12.**
>
> - **Done.** Corpus and retweet graph (`02_Processing/02`); both backbones (`02/02`); five community partitions per backbone (`04_Network_Analysis/01`); map of the retweeted-once backbone (`04/02`); author topics, affect, community comparison and figures for the **retweeted-once backbone** — 164,291 eligible authors, K = 12 (§7, §8). All on Drive.
> - **Owed.** Author topics on the LWCC backbone; the LWCC map; the `01b` verification run (8a) on the cluster, so the stored classified files are recorded as verified (§9); `03/03`, `03/05`, `03/06` on the current corpus. Full list in §11.
> - The networks built before 2026-10-07 used the wrong edge direction; everything derived from them is superseded (§5.3).

**Reading conventions.** Notebook ids are `stage/index`: `02/02` = `02_Processing/02_sanity_check_and_network_generation`, `04/01` = `04_Network_Analysis/01_network_analysis`, `04/02` = `04_Network_Analysis/02_network_visualization`, `03/04` = `03_Analysis_and_Modeling/04_lda_author_topics_hpc`, `03/01b` = `03_Analysis_and_Modeling/01b_sentiment_emotion_hpc`. "Cell N" in §§1–4 points into the pre-2026-10 version of `02/02` (in git history); the current file has the same corpus cells under the same headings, and its 2026-10-10 run reproduced every count. **Influence** orientation = edge retweeter → retweeted (the graph as built); **information-flow** orientation = the transpose, edge retweeted → retweeter (the direction content travels). The information-flow files are the ones analysed.

## 0. Answer sheet

| Stage | Where | Tweets | Authors | Notes |
|---|---|---:|---:|---|
| Raw API dumps flattened | `02/01` cell 16 | 36,560,405 records | 31,989,322 author records | both with duplicates |
| Unique tweet ids in raw file | `02/02` cell 20 | 25,637,570 | — | |
| **AI corpus** (`AItrust_twits_pruned_dict.json`) | `02/02` cell 20 | **17,410,035** | **4,775,711** | dedup + AI keyword + English + date ≥ 2022-10-31 |
| **AI+Art corpus** (`AItrust_Art_pruned_twit_dict.json`) | `02/02` cell 20 | **3,583,101** | **1,440,802** | AI corpus ∩ 60 art keywords; nested in the AI corpus |
| Retweets in the AI corpus | `02/02` cell 44 | 9,638,407 | — | the only tweets that create edges |
| Retweet graph | `02/02` cell 52 | — | **3,379,040** | 7,768,720 edges, weight 9,638,407 |
| Authors retweeted at least once | `02/02` cell 51 | — | 374,368 | outer keys of the retweet counter |
| **LWCC backbone** (`Full_LWCC_InfoFlow.gml`) | `02/02` | — | **3,264,499** | self-loops removed, largest weakly connected component; 98.14 % of retweets |
| **Retweeted-once backbone** (`Full_RetweetedOnce_InfoFlow.gml`) | `02/02` | — | **202,710** | retweeted ≥ 1× by someone else (363,618), then largest component; 13.17 % of retweets |
| Community partitions | `04/01` | — | 202,710 / 3,264,499 | five methods per backbone; every author in every partition |
| **Author topics, retweeted-once** | `03/04` + `src/author_analysis` | all post types | **164,291** | authors with ≥ 3 non-empty cleaned posts; K = 12 (§7) |
| Author topics, LWCC | `03/04` | — | — | not yet fitted |
| Test branch, AI / AI+Art | `02/02` cell 17 | 881 / 216 | 723 / 198 | from one raw window file (233,094 records) |

## 1. Stage 1 — Raw API pages → line-delimited dictionaries

Notebook: [02_Processing/01_api_data_to_dictionaries.ipynb](../notebooks/02_Processing/01_api_data_to_dictionaries.ipynb)

`process_tweet()` (cell 10, *Tweet Processing Functions*) keeps a fixed subset of fields per tweet:

```python
twit_dict['id'] = original_tweet['id']
twit_dict['text']=original_tweet['text']
twit_dict['created_at']=original_tweet['created_at']
twit_dict['public_metrics']=original_tweet['public_metrics']
twit_dict['author_id']=original_tweet['author_id']
twit_dict['type']='original'
try:
  twit_dict['type']=original_tweet['referenced_tweets'][0]['type']
  twit_dict['referenced_tweets']=original_tweet['referenced_tweets'][0]['id']
except:
  pass
```

`author_id` is copied verbatim from the API, so every later author count is a count of **distinct Twitter user ids**.

**Where the duplicates come from.** Cell 16 (*Process All Data*) writes both the page's `data` tweets and its `includes.tweets` (referenced tweets pulled in by the API expansions), so a tweet referenced by many others is written many times. Author objects from `includes.users` are written once per appearance.

| Counter (end of cell 16) | Value |
|---|---:|
| Raw window files processed | 5,016 |
| Tweet records written | 36,560,405 |
| Author records written | 31,989,322 |

The author file is never deduplicated and feeds no author count; every author count below comes from the `author_id` field of tweets. Test branch (cell 12): one file `testing.json` → 233,094 tweet records, 1,272 author records.

## 2. Stage 2 — Pruning to the AI and AI+Art corpora

Notebook: [02_Processing/02_sanity_check_and_network_generation.ipynb](../notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb)

**Filters** (cells 12, 14, 15 — *Prunning Functions*, *Keywords Filter*): an AI-keyword regex plus allowed standalone/hashtag/mention forms of `AI` and `AGI`, with an `#airdrop`/`@airdrop` denylist; an English test (`lang` starts with `en`, or the `seems_english()` heuristic); a date cutoff `created_at ≥ 2022-10-31`; and, for the AI+Art corpus only, a whole-word match against the 60 terms of [keywords.txt](../notebooks/02_Processing/keywords.txt). The rules are spelled out in [DATASET_STATISTICS.md §3](DATASET_STATISTICS.md).

**The single pruning pass** (cell 20, *Prune Full DS*). The order of the checks makes the funnel additive:

```python
for line in tqdm.tqdm(in_f, total=36560405, desc="Pruning Full DS"):
    twit = json.loads(line); twid = twit.get('id'); txt = (twit.get('text') or '').strip()
    if twid is None or not txt:                      stats['dropped_missing_id_or_text'] += 1; continue
    seen_all_ids.add(twid)
    if twid in seen_written:                         stats['duplicates_skipped'] += 1; continue
    has_ai = bool(pattern.search(txt) or has_allowed_ai_form(txt) or has_allowed_agi_form(txt))
    if not has_ai:                                   stats['dropped_no_ai_keyword'] += 1; continue
    is_en = (lang.startswith('en') or seems_english(txt))
    if not is_en:                                    stats['dropped_non_english'] += 1; continue
    if date_only < earliest_date:                    stats['dropped_before_date'] += 1; continue
    ai_f.write(...); seen_written.add(twid); unique_authors_ai.add(author_id)
    if tweet_has_keyword(twit['text']):
        art_f.write(...); unique_authors_art.add(author_id)
```

Two consequences:

- **"AI+Art authors" = authors with at least one art-keyword tweet.** It is not a classification of authors.
- **Dedup happens before the keyword test**, on tweet id, keeping the first occurrence. Only *written* ids are remembered, so a dropped tweet that reappears is re-tested, not counted as a duplicate.

| Funnel line (cell 20, full run) | Count |
|---|---:|
| Total records read | 36,560,405 |
| Unique tweet ids seen | 25,637,570 |
| Duplicates skipped | 8,494,403 |
| Dropped (no AI keyword) | 10,429,950 |
| Dropped (non-English) | 76,363 |
| Dropped (before 2022-10-31) | 149,654 |
| **AI corpus written** | **17,410,035** |
| **AI+Art corpus written** | **3,583,101** |
| **Unique authors, AI corpus** | **4,775,711** |
| **Unique authors, AI+Art corpus** | **1,440,802** |
| Date range | 2022-10-31 → 2023-02-27 |
| Tweet types | original 4,061,626 · retweeted 9,638,407 · replied_to 3,221,048 · quoted 488,954 |

Check: 8,494,403 + 10,429,950 + 76,363 + 149,654 + 17,410,035 = 36,560,405 ✓. Test branch (cell 17): 233,094 records → 1,459 unique ids → 881 AI tweets / 723 authors, 216 AI+Art tweets / 198 authors, 2022-11-05 → 2022-11-15.

## 3. Derived dictionaries — timeline, author corpus, retweet counter

Same notebook, cell 44 (*Generate Full Timeline and Network Dictionaries*), one pass over the **AI corpus only**:

```python
type_of_network = 'retweeted'                       # cell 43
for line in tqdm.tqdm(AItrust_pruned_twits, total=17410035):
    twit = json.loads(line)
    if twit['type'] == type_of_network:
        author = str(twit['author_id'])
        referenced_author = str(twit['referenced_tweets_dictionary']['author_id'])
        network_dict.setdefault(referenced_author, {}).setdefault(author, 0)
        network_dict[referenced_author][author] += 1     # (written out long-hand in the cell)
    author_corpus_dict.setdefault(str(twit['author_id']), []).append(twit['text'])
```

`network_dict[retweeted_author][retweeter]` = number of times the retweeter retweeted that author. Unresolvable referenced tweets would raise and be counted as `exceptions`; the stored count is **0**, so all 9,638,407 retweets became edge weight. Outputs: `Networks/1_retweet_dicts/full_network_dict.pkl`, and in `Cleaned Data/`: `full_basic_counts_dict.pkl`, `full_timeline_dict.pkl`, `full_author_corpus_dict.pkl` (4,775,711 authors).

## 4. The retweet graph

**Construction** (cell 51, *Full Network*):

```python
G = nx.DiGraph()
for key in tqdm.tqdm(network_dict):            # key = retweeted author
    for referenced in network_dict[key]:       # 'referenced' is actually the RETWEETER
        G.add_edge(referenced, key, weight=network_dict[key][referenced])   # retweeter -> retweeted
```

In this (influence) orientation, weighted **in-strength = retweets received** and **out-strength = retweets made**. The outer loop has 374,368 keys, so 374,368 authors were retweeted at least once; the other 3,004,672 nodes only retweet. Node names are `str(author_id)`; in a GML written by NetworkX the author id is the `label` attribute and `id` is a sequential index — always match authors on `label`.

**Topology** (cell 52):

| Metric | Value |
|---|---:|
| Nodes (authors) | 3,379,040 |
| Directed edges (unique retweeter→retweeted pairs) | 7,768,720 |
| Total weight | 9,638,407 (= number of retweets) |
| Self-loops | 32,216 |
| Weakly connected components | 49,464 |
| Largest WCC | 3,264,499 (96.61 %) |
| Most retweeted author | 1156482001, 92,733 retweets received |
| Most active retweeter | 1356588046616043527, 45,503 retweets made |

`02/02` writes the graph in both orientations to `2_full_graphs/Full_Network_{Influence,InfoFlow}.gml`, each read back and checked for identical node and edge sets. Test branch: 881 tweets → 446 nodes, 334 edges, weight 363, 122 components, largest 32.

**Three author populations.** 4,775,711 corpus authors ⊃ 3,379,040 in the retweet graph: 1,396,671 authors (29 %) wrote only originals, replies or quotes and have no edge.

## 5. The author backbones

`02/02` cuts both backbones from the full graph: drop self-loops, optionally apply a degree cut, keep the largest weakly connected component. Each is written in both orientations to `Networks/3_backbones/`; the `InfoFlow` file is the one `04/01`, `04/02` and the author analysis read. `02/02` asserts that the two orientations hold the same author set and edge count.

### 5.1 The two backbones

| Backbone | Rule | Authors | Edges | Retweets kept | Files |
|---|---|---:|---:|---:|---|
| **LWCC** | self-loops removed (25,135 inside the giant component), largest weakly connected component | **3,264,499** | 7,670,516 | 9,458,703 (98.14 %) | `Full_LWCC_{Influence,InfoFlow}.gml` |
| **Retweeted once** | self-loops removed; in-strength ≥ 1 on the influence graph = retweeted at least once by someone else (363,618 authors); then largest component | **202,710** | 882,530 | 1,269,747 (13.17 %) | `Full_RetweetedOnce_{Influence,InfoFlow}.gml` |

They answer different questions. The **LWCC** keeps every retweeter and every edge, so its communities reflect who amplifies whom; inside it only 317,533 authors are retweeted and 3,095,219 retweet. The **retweeted-once** backbone is exactly the amplified population, but it drops 87 % of the retweet volume, because pure retweeters leave with their edges; inside it 132,603 authors are retweeted and 143,043 retweet.

Retweeted-once cut, step by step (influence graph):

| Step | Authors | Edges | Weight |
|---|---:|---:|---:|
| Full network | 3,379,040 | 7,768,720 | 9,638,407 |
| Self-loops removed | 3,379,040 | 7,736,504 | 9,606,191 |
| Retweeted ≥ 1× by someone else | 363,618 | 890,006 | — |
| Largest weakly connected component | **202,710** | 882,530 | 1,269,747 |

374,368 authors were retweeted but only 363,618 pass: the other 10,750 were retweeted only by themselves. The component step removes 161,000 authors but almost no edges — they are isolates whose only retweeters were pure retweeters.

Distributions of retweets received (`04/01`, flow orientation): heavy-tailed on both backbones (power-law α ≈ 1.87 retweeted-once, 1.84 LWCC), but a lognormal fits better (p < 0.001).

### 5.2 Runs

`02/02` last ran on Colab on **2026-10-10** (finished 04:37 EDT), into the stage folders, with `generate_data = True`; outputs stored in the notebook. Every count above reproduced the 2026-10-09 run exactly, and every GML passed its read-back check. Timing: 13 min 35 s for the retweet counter, 17 min 34 s to export both full graphs, 20 min 59 s for the LWCC cell, about 4 min for the retweeted-once cut and export. Statistics are also in `Cleaned Data/full_dual_network_stats.json` (`raw`, `lwcc`, `pruned`).

### 5.3 Superseded: the retweeter backbone

Until 2026-10-07 the analysed backbone was `Final_OutThreshold1.gml` (1,984,599 authors, 4,472,376 edges). It was cut with out-strength ≥ 1 on the influence graph, which keeps authors who **made** a retweet, not authors who **were retweeted**. Its community partitions, maps, the 198,326 "matched authors" and every figure and number in the blog draft (`docs/Public Perceptions of AI and Art.md`) descend from it. The files are deleted from Drive and no notebook produces them; the rule survives only as `prune_by_out_strength_threshold` in [src/network/network_pruning.py](../src/network/network_pruning.py), called by nothing. The corrected run (§7, §8) replaces those results.

## 6. Communities and maps (`04_Network_Analysis`)

### 6.1 Five partitions per backbone (`04/01`)

[04_Network_Analysis/01_network_analysis.ipynb](../notebooks/04_Network_Analysis/01_network_analysis.ipynb) runs five methods on the `InfoFlow` file of each backbone through `src.network.network_modularity.run_modularity_workflow`. Label propagation, Louvain and Leiden (fast) run on the undirected collapse with summed weights (882,530 → 864,436 edges on retweeted-once; 7,652,030 on LWCC); Leiden (directed) and Infomap run on the directed graph. Last run: Colab, **2026-10-10** (13:01 CEST), no errors, outputs stored in the notebook.

| Method | Retweeted once: communities | Q | Largest five | LWCC: communities | Q | Largest five |
|---|---:|---:|---|---:|---:|---|
| label_propagation | 4,489 | 0.652 | 87,114 · 58,319 · 4,339 · 2,985 · 2,453 | 36,189 | 0.668 | 856,505 · 768,583 · 36,238 · 35,361 · 33,563 |
| louvain | 1,031 | 0.742 | 48,166 · 47,736 · 17,357 · 17,313 · 12,662 | 2,331 | 0.763 | 620,710 · 384,156 · 307,544 · 291,793 · 182,394 |
| leiden_fast | 1,119 | 0.747 | 46,206 · 44,280 · 21,796 · 21,455 · 13,166 | 2,983 | 0.774 | 609,447 · 375,776 · 294,813 · 291,054 · 177,380 |
| leiden_directed | 1,089 | 0.744 | 47,527 · 46,239 · 19,810 · 19,705 · 13,531 | 2,951 | 0.772 | 619,125 · 386,553 · 327,295 · 269,842 · 188,686 |
| infomap | 17,117 | 0.415 (7.946 bits) | 2,786 · 1,363 · 1,227 · 857 · 810 | 136,744 | 0.513 (13.530 bits) | 36,398 · 31,596 · 30,223 · 28,873 · 27,060 |

Agreement, aligned by author id (every author is in every partition):

| Pair | Retweeted once AMI / ARI | LWCC AMI / ARI |
|---|---|---|
| leiden_fast ↔ leiden_directed | 0.817 / 0.851 | 0.847 / 0.879 |
| louvain ↔ leiden_directed | 0.802 / 0.868 | 0.785 / 0.817 |
| louvain ↔ leiden_fast | 0.783 / 0.816 | 0.816 / 0.850 |
| label_propagation ↔ modularity methods | 0.541–0.546 / 0.44–0.48 | 0.549–0.559 / 0.46–0.47 |
| infomap ↔ everything | 0.258–0.311 / < 0.01 | 0.423–0.521 / 0.011–0.024 |

The three modularity methods agree closely; label propagation sits in the middle and puts 1.6M LWCC authors into two communities; Infomap gives many small random-walk modules, a different kind of partition. **Leiden (directed) is the partition the author analysis uses** (`--community-algorithm leiden_directed`, the default). The modularity methods are randomised, so a re-run moves community counts by a few per cent (the LWCC Leiden-directed and Infomap partitions reproduced exactly between the 2026-10-09 and 2026-10-10 runs).

Outputs, per backbone in `Networks/4_communities/<RetweetedOnce|LWCC>/`: five `<stem>_InfoFlow_<method>.gml` (the backbone with one vertex attribute `community_<method>`, read back as `community<method without underscores>`, float → int) and `<stem>_author_communities.json` = `{author_id: {method: community}}`. The JSON is what downstream code reads.

**Caveat.** The GMLs of the three undirected methods hold the undirected collapse, so a "degree" read from them is total degree, not retweets received. For retweets received, read the backbone in `3_backbones/` (out-strength of the `InfoFlow` file) and join on `label`.

### 6.2 Maps (`04/02`)

[04_Network_Analysis/02_network_visualization.ipynb](../notebooks/04_Network_Analysis/02_network_visualization.ipynb) renders one registry entry per run on a Colab GPU (ForceAtlas2 `classic` and `linlog`, light and dark), into `Networks/5_visualizations/<NETWORK>/`.

- **`RetweetedOnce_leiden_fast` rendered 2026-10-10** (8.2 min): 202,710 nodes, 864,436 undirected edges, 1,119 communities, the 16 largest (coloured) cover 91 % of nodes. Files: `social_map_{classic,linlog}_{light,dark}.png`, `positions_{classic,linlog}.parquet`, `05_degree_ccdf.png`, `06_adjacency_blocks.png`. Its `in_degree` column is total degree (§6.1 caveat).
- **LWCC maps: not rendered.**
- The trailing DrL cell fails on Colab (`ModuleNotFoundError: src`, the notebook has no clone cell), so there is no DrL map.

The interactive author viewer of §8 is a separate, newer path built from the author-analysis outputs.

## 7. Author topics and affect (`03/04` + `src/author_analysis`)

### 7.1 The workflow

[03_Analysis_and_Modeling/04_lda_author_topics_hpc.ipynb](../notebooks/03_Analysis_and_Modeling/04_lda_author_topics_hpc.ipynb) is a thin driver: it imports [src/author_analysis](../src/author_analysis/README.md) and runs its stages for each backbone. The same stages run from the command line (`python -m src.author_analysis.run <stage> --data-root DATA --output-root OUT --backbone <RetweetedOnce|LWCC>`). On Colab, `DATA` = the Drive project folder and `OUT` = `Models/Topic Modeling/LDA/backbone_analysis/`; elsewhere both come from `TWITTER_AI_DATA_ROOT` / `TWITTER_AI_OUTPUT_ROOT`. Results go to `OUT/networks/<backbone>/`.

| Stage | Reads | Writes (under `OUT/networks/<backbone>/`) | What it does |
|---|---|---|---|
| `prepare` | AI corpus, backbone community JSON | `topics/author_documents.jsonl`, `documents.json` | one document per backbone author: all post types (including retweets), cleaned with the shared rules; keeps authors with ≥ 3 non-empty cleaned posts |
| `topics` | the documents | `topics/grid_results.csv`, `model.json`, `top_terms.csv`, `lda_model.pkl`, `vectorizer.pkl`, `author_topics.csv` | grid on a seeded 100,000-author sample, then a refit of the selected model on every eligible author |
| `affect` | AI corpus, sentiment and emotion files of `01b` | `affect/author_sentiment.csv`, `author_emotions.csv`, `coverage.json` | mean sentiment and emotion scores over every post of each eligible author; **fails if any post is missing or mismatched** |
| `match` | community JSON, topics, affect | `data/matched_authors.parquet` | one row per author: topic mixture, affect, community |
| `summarize` | matched authors | `tables/*_network_subset.csv` | corpus sentiment/emotion, topic prevalence, topic × sentiment/emotion |
| `communities` | matched authors | `communities/figures/*.csv` | community × topic composition, capture, enrichment; sentiment/emotion means and standardised differences; omega-squared |

Grid: TF-IDF unigram and unigram+bigram × K ∈ {8, 12, 16, 20} × α ∈ {0.05, 0.1, 0.3} × η ∈ {0.01, 0.1}; `min_df = 5`, `max_df = 0.7`, 50 iterations, seed 42; scored by coherence (0.25), diversity (0.30), non-collapse (0.15) and evenness (0.30). A final fit whose dominant topic holds > 90 % of authors, or grows > 15 points over selection, stops the later stages for review. A `--model-config` JSON can fix the selected model explicitly.

### 7.2 The retweeted-once run (2026-10-11)

Recorded in [config/retweeted_once_k12/](../config/retweeted_once_k12/) (`model.json`, `provenance.json`, `topics.csv`, `communities.csv`), source revision `8ee7555`. The run itself is not stored in a notebook; `provenance.json` carries the SHA-256 of the corpus, the community JSON, the author documents, the model, the vectorizer and the author-topic table, so a later run can be checked against it.

- **Population.** 202,710 backbone authors → **164,291 eligible** (≥ 3 non-empty cleaned posts).
- **Model.** TF-IDF unigram, **K = 12**, α = 0.3, η = 0.1, seed 42. Chosen by semantic review of terms and posts, **not** the automatic grid winner; `model.json` fixes it as `selected_model`.
- **Topics** ([topics.csv](../config/retweeted_once_k12/topics.csv), reviewed labels with confidence): 0 Education & fandom (mixed) · 1 AI images & avatars (mixed) · 2 AI art & artists' debate · 3 Writing, reviews & mixed content · 4 Games & fandom (mixed) · 5 AI research & industry · 6 Crypto promotion & giveaways · 7 AI crypto launchpads · 8 General AI use, tools & writing · 9 AI art sharing & NFTs · 10 Politics & AI bias · 11 Crypto & blockchain. Four are labelled low confidence (0, 1, 3, 4).
- **Communities** ([communities.csv](../config/retweeted_once_k12/communities.csv)): the 21 largest Leiden-directed communities among the eligible authors, each labelled from c-TF-IDF terms and 25 posts by distinct authors. Largest: C1 AI tools, news & discussion (39,397) · C0 AI-art debate & criticism (37,737) · C2 AI debate, politics & trading (16,019) · C3 AI in education & research (15,120) · C4 AI-art sharing & NFTs (11,623). The contrast figure compares C0 and C4.
- **Stability** (seed 42 vs 43, same vocabulary): 76.5 % of authors keep their leading topic after matching topics by membership; mean membership change 0.17; rank correlation of topic net sentiment 0.87; 5 of 54 substantial community–topic enrichments (community ≥ 1,000 authors, |log₂| ≥ 0.5 in both fits) flip sign. Approved for exploratory use, not as a robust taxonomy.
- **Results**: [Drive results folder](https://drive.google.com/drive/folders/1C4-R40VDIyhn5yFovLAmVjUBE5RSqDLc) (`Dark/`, `Data/`, `PNG/`, `SVG/`, `Labels and colours.html`, all dated 2026-10-11) and the interactive [author atlas](https://davidfreeborn.github.io/twitter-authors-map/).

Profiles include retweets, so they describe content contributed **and amplified**, not beliefs. Topic numbers are specific to this fit; do not compare them across backbones or refits.

### 7.3 LWCC

Not fitted. The workflow is the same with `--backbone LWCC`; the K = 12 labels and community descriptions above must not be reused for it.

## 8. Figures and the author viewer

These stages read the outputs of §7 and refit nothing.

| Step | Code | Reads | Writes (under `OUT/networks/<backbone>/`) |
|---|---|---|---|
| Figure inputs | `run figure-inputs` (`src/author_analysis/export_figures.py`) | matched authors, `tables/`, `communities/figures/`; optional `--topic-labels`, `--community-labels` | `figure_inputs/` — `matched_authors.json`, `matched_author_ids.json`, `topic_labels.csv`, `community_labels.csv`, copies of the tables and matrices |
| Projection | `run projection` (`project_topics.py`) | matched authors | `figure_inputs/umap_fixed_topic_scores_seed0.csv` |
| Weekly series | `src/blog_analysis/weekly_aggregation.py` | matched ids, AI corpus, sentiment/emotion files, saved model and vectorizer | `weekly/weekly_{sentiment,emotions,topics_long,topics_wide}.csv` |
| Figures 1, 4, 6–11 | `src/blog_analysis/render_non_temporal_figures.py` | `figure_inputs/` | `figures/` (light and dark PNG/SVG, one CSV per figure) |
| Figures 2, 3, 5 | `src/blog_analysis/render_temporal_figures.py` | `weekly/` | `figures/` |
| Layout | `run layout` (`src/network/visualization/build_matched_layout.py`) | matched authors, backbone `InfoFlow.gml` | `layout/matched_nodes.parquet`, `matched_edges.parquet` |
| Viewer bundle | `run export` (`export_viewer.py`), then `build_layout_3d` | layout | `viewer/nodes.json` + 3D coordinates; served by `src.network.visualization.serve` into [visualizations/author_network/](../visualizations/author_network/README.md) |

Weekly rules: weeks start Monday (UTC); affect is averaged per author-week, then each active author weighs equally; the incomplete last week (collection ends 2023-02-27 12:00 UTC) is excluded from plotted trends. Commands: [docs/blog_analysis/README.md](blog_analysis/README.md). Generated figures and tables are not tracked in git (the old `outputs/` folder was removed on 2026-10-11).

## 9. Sentiment and emotion (`03/01`, `03/01b`)

- **`03/01`** (CardiffNLP sentiment, Colab) wrote `Cleaned Data/AItrust_pruned_twits_with_sentiment.json`, all 17,410,035 tweets (2026-05-12). Its block reuse checked only line counts, so the content is unverified; nothing in §7 reads it.
- **`03/01b`** (cluster, ten models on the test sets; `twitter-roberta-base-sentiment-latest` and `twitter-roberta-base-emotion-multilabel-latest` on both full corpora). The full run stored in the notebook is **faulty**: its block resume trusted line counts, so blocks from other runs were reused.

  | Output | Lines | Expected | State |
  |---|---:|---:|---|
  | AI × sentiment | 17,410,035 | 17,410,035 | unverified |
  | AI × emotion | 17,401,060 | 17,410,035 | 8,975 short and misaligned |
  | AI+Art × sentiment | 4,057,340 | 3,583,101 | AI-corpus content |
  | AI+Art × emotion | 3,583,101 | 3,583,101 | unverified |

  Fixed in code on 2026-10-10: one order-preserving writer, block folders keyed by dataset × model × block count, a block reused only if its tweet ids equal the input slice, merge to `.tmp` then verify then rename. **8a** verifies existing outputs tweet by tweet (`aligned` / `same ids` / `FAIL`), **8b** re-runs failures, **8c** removes legacy block folders. 8a has not been run or stored.

- **What the author analysis used.** The `affect` stage reads the two AI-corpus files (`ai_full_classified_*`) and refuses to run if any post of an eligible author is missing, duplicated or mismatched. The 2026-10-11 run passed that check, so the files it read covered every post of the 164,291 authors; the repair that made the AI emotion file complete is not recorded in the repository. The AI+Art files are used by nothing downstream.

## 10. Other notebooks

| Notebook | Input | State |
|---|---|---|
| `03/02_extract_examples` | AI and AI+Art test files | example sheets only |
| `03/03_lda_tweet_topics` | `AItrust_pruned_twits_with_sentiment.json` (17.41M) | rewired 2026-10-10 to the current corpus; **not run** (no stored outputs). Trains on the first 25,000 tweets, not a random sample. Earlier results came from a first-generation 2025 corpus (22.4M lines, 14.9M unique ids, no longer on Drive) and must not be quoted next to the 17.41M corpus |
| `03/05_embedding_mapping` | current AI corpus | pointed at the current corpus 2026-10-10; **not run**. `.npy`/`.pkl` files in `Models/Topic Modeling/` predate it |
| `03/05b_embedding_datamap_template` | OpenAlex sample (not on Drive) | a collaborator's template; no edges in this pipeline |
| `03/06_topic_modeling_appendix` | AI test corpus | fixed 2026-10-10; **not run** |
| `05_Classifiers/*` | AI and AI+Art tweet files | tweet-level labels; independent of the networks |

**AI+Art authors.** The only AI+Art author set is the 1,440,802 authors with ≥ 1 art-keyword tweet (§2). No AI+Art retweet network and no author-level art label exist; the `05_Classifiers` notebooks label tweets, not authors.

## 11. Runs owed

| Run | Where | Needs |
|---|---|---|
| Author topics, affect, communities on **LWCC** | `03/04` (`BACKBONES = ["LWCC"]`) or the CLI | a smoke run first; grid on a 100,000-author sample, final fit on ~3M authors (run time unknown) |
| LWCC maps | `04/02`, `NETWORK = 'LWCC_leiden_fast'` | Colab GPU |
| `01b` 8a (then 8b for each failure, then 8c) | cluster | records which stored classified files are verified (§9) |
| Tweet-level topics | `03/03` on Colab | — |
| Embedding map, topic appendix | `03/05`, `03/06` on Colab | — |
| Blog draft numbers | `docs/Public Perceptions of AI and Art.md` | replace the superseded-backbone numbers (§5.3) with the results of §7.2 |

Independent of the backbone, the blog draft also says the graph records "retweets, replies and quotations" (retweets only), rounds 4,775,711 authors to "4.7 million" (4.8 million) and describes bot filtering that the pipeline does not do (the only spam rule is the `#airdrop` denylist).

## 12. The pipeline graph

[src/scripts/pipeline_graph.py](../src/scripts/pipeline_graph.py) builds a producer/consumer graph of the pipeline: notebooks and `src/` scripts on one side, data files on the other. It parses each notebook's code for the files it opens; I/O the parser cannot see is declared in [pipeline_overrides.yaml](pipeline_overrides.yaml). It writes [pipeline_graph.json](pipeline_graph.json), [pipeline_graph.png](pipeline_graph.png) (notebooks and scripts ↔ files) and [pipeline_graph_notebooks.png](pipeline_graph_notebooks.png) (notebooks and scripts only; squares are notebooks, diamonds are `src/` scripts, stage 7 is the figures).

```bash
python3 src/scripts/pipeline_graph.py              # rebuild the three files in docs/
python3 src/scripts/pipeline_graph.py validate     # compare provenance tags in README.md / notebook_setup.md with the graph
python3 src/scripts/pipeline_graph.py upstream   03_Analysis_and_Modeling/04_lda_author_topics_hpc
python3 src/scripts/pipeline_graph.py downstream 02_Processing/02_sanity_check_and_network_generation
```

**State 2026-10-12:** 23 notebooks, 7 scripts, 124 files, 119 write edges, 96 read edges; 99 declared edges; `validate` reports no drift.

**What the overrides declare.**

- `manual_edges` — paths built inside helpers (`export_network()` in `02/02`), written by `src/` workflows (`run_modularity_workflow`, every `src/author_analysis` stage), or chosen from registries at run time. Each entry names the notebook or script, the file, the direction and a note. Declared edges carry `manual: true` in the JSON.
- `scripts` — the `src/` stages that run from the command line (§8): layout, viewer export, figure inputs, projection, weekly aggregation and the two renderers. They have no notebook, so without this section they would not appear.
- `glob_artifacts` — families of files folded into one node (classifier checkpoints, the `ai_/art_full_classified_*.json` outputs).

**Reading the graph.**

- Files under `OUT/networks/{backbone}/` are drawn at the Colab default `OUT` (`Models/Topic Modeling/LDA/backbone_analysis/`) with `{backbone}` = `RetweetedOnce` or `LWCC`.
- `01b` writes its classified files on the cluster; the author analysis reads the Drive copies. The copy is manual, so the graph shows no edge from `01b` to `03/04`.
- The `BASE_PATH divergence` warning is expected: `01b` runs on the cluster path, the other notebooks on Drive.
- The graph records what the code can read and write, not what exists on Drive.

**After any change to a notebook or a `src/` stage:** rebuild, run `validate` (must print `no drift detected`), declare any path the parser cannot see in the overrides, run `python3 notebooks/analyze_notebooks.py` for the touched notebook, and update §13.

## 13. File inventory

Every file the notebooks and declared scripts read or write, from [pipeline_graph.json](pipeline_graph.json). Locations are relative to `My Drive/Colab Projects/AI Public Trust/` unless marked HPC. *Written by* / *read by* use the notebook ids of the reading conventions; templated names (`{…}`) are filled at run time. A file with no writer is produced outside the repository; a file with no reader is a terminal output.

**Live listings** (shared read-only; `https://drive.google.com/embeddedfolderview?id=<id>#list` lists any folder): [Data Sets/](https://drive.google.com/drive/folders/1ktJRebXjsxVgcjjii6uGmw68ZySQUMSa?usp=sharing) (`1ktJRebXjsxVgcjjii6uGmw68ZySQUMSa`), `Cleaned Data/` (`1b7uBOgqghM-_dQSsVVmA2GH0lySeuxt_`), [Networks/](https://drive.google.com/drive/folders/1PlVu_Li9nSI7IDLLfURp09s_1bAGXpMq?usp=sharing) (`1PlVu_Li9nSI7IDLLfURp09s_1bAGXpMq`), [Colab notebooks](https://drive.google.com/drive/folders/167u-upKGI1lqbgeWW6ygw2Zbsgmts52a?usp=sharing) (`167u-upKGI1lqbgeWW6ygw2Zbsgmts52a`). The notebooks on Drive import `src.*` from a fresh clone of GitHub `main`, so a `src/` change must be pushed before a notebook that uses it is run on Colab.

### 13.1 `Raw Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `testing.json` | — | `02/01` | One raw API page (233,094 records with duplicates); source of every `_test` file. |
| `Twits/tweets_<timestamp>.json` | — | `02/01` | Raw API harvest, one file per 10-minute window; `02/01` iterates the folder. |

### 13.2 `Data Sets/`

Besides the files below and the folders of §§13.3–13.8, `Data Sets/` holds `Backup/`, `Classification/` and `Side Sentiment Analysis/`, which no notebook reads or writes.

| File | Written by | Read by | What it is |
|---|---|---|---|
| `AItrust_twits_dict.json` | `02/01` | `02/02` | Stage-1 tweet dictionary, with duplicates (36,560,405 lines). |
| `AItrust_author_dict.json` | `02/01` | `02/02` | Stage-1 author dictionary, with duplicates (31,989,322 lines). |
| `AItrust_twits_dict_test.json`, `AItrust_author_dict_test.json` | `02/01` | `02/01`, `02/02` | Test-branch counterparts. |

### 13.3 `Data Sets/Cleaned Data/`

Subfolders: `Partitioned Data/` (§13.4), `Tweet Sheets/` (§13.5), `Sentiment Analysis/` (copies of `01b`'s test outputs, valid), `Blocks/` (empty), `Topic Modeling/Kylie's Work/` (May 2026 outputs of the deleted `04c`, fitted on ~8,000 low-id accounts because of a GML `id`/`label` bug; invalid, superseded by §7).

| File | Written by | Read by | What it is |
|---|---|---|---|
| `AItrust_twits_pruned_dict.json` | `02/02` | `02/02`, `03/01`, `03/04`, `03/05`, weekly aggregation | **AI corpus**, 17,410,035 tweets with `processed_text` (§2). |
| `AItrust_Art_pruned_twit_dict.json` | `02/02` | — | **AI+Art corpus**, 3,583,101 tweets. |
| `AItrust_twits_pruned_dict_test.json` | `02/02` | `02/02`, `03/01`, `03/02`, `03/05`, `03/06` | Test AI corpus (881 tweets). |
| `AItrust_Art_pruned_twit_dict_test.json` | `02/02` | `02/02`, `03/02` | Test AI+Art corpus (216 tweets). |
| `ai_full_classified_twitter-roberta-base-sentiment-latest.json`, `ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json` | copied from HPC (`03/01b`) | `03/04` (affect), weekly aggregation | Per-tweet sentiment and emotion scores for the AI corpus (§9). |
| `AItrust_pruned_twits_with_sentiment.json` | `03/01` | `03/03` | AI corpus with CardiffNLP sentiment (§9). |
| `AItrust_pruned_twits_test_with_sentiment.json`, `AItrust_twits_pruned_dict_test_with_sentiment.json` | `03/01` | `03/01` | Sentiment test outputs (old and current names). |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5.json` | `03/03` | `03/03` | Tweets with K = 5 topic fields; merged from per-block files. Not on Drive until `03/03` runs. |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5_block{NNN}.json` | `03/03` | — | Per-block outputs of that enrichment. |
| `AItrust_pruned_twits_with_sentiment__SAMPLE{SAMPLE_ENRICH_N}.json` | `03/03` | — | Optional input sample (`SAMPLE_ENRICH = True`). |
| `AItrust_topics_k5_metadata.json` | `03/03` | — | Top terms and settings of the K = 5 tweet-topic model. |
| `dataset_statistics_summary.json` | `02/02` | — | Master summary; mirrored at `notebooks/02_Processing/dataset_statistics_summary.json`. |
| `full_pruning_stats.json` | `02/02` | — | Funnel counters of the pruning pass (§2). |
| `full_dual_network_stats.json` | `02/02` | — | Full graph (both orientations), LWCC and retweeted-once statistics (§5). |
| `full_network_stats.json` | — | `02/02` | Topology statistics from the pre-2026-10 `02/02`; read by the summary-report cell. |
| `full_author_corpus_dict.pkl` | `02/02` | `02/02` | author id → raw tweet texts, 4,775,711 authors. |
| `full_basic_counts_dict.pkl` | `02/02` | — | Tweet-type counts. |
| `full_timeline_dict.pkl` | `02/02` | `02/02` | tweet id → `created_at`. |
| `test_*` (`author_corpus_dict.pkl`, `basic_counts_dict.pkl`, `dual_network_stats.json`, `pruning_stats.json`, `timeline_dict.pkl`) | `02/02` | `02/02` (except the stats JSON) | Test-branch counterparts. |

### 13.4 `Data Sets/Cleaned Data/Partitioned Data/AI Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `base_dataset.pkl` | `05/00` | `05/00b`, `05/03` | HITL seed partition. |
| `hitl_pending_batch_{i + 1}.pkl` | `05/00` | `05/02` (as `hitl_pending_batch_{PENDING_BATCH_TO_PROCESS}.pkl`) | Pending HITL batches, one per round. |
| `inference_dataset.pkl` | `05/00` | `05/03` | Remainder held out for final inference. |
| `llm_bootstrap_dataset.pkl` | `05/00` | `05/01`, `05/03` | Partition sent to the LLM labeller. |
| `partition_ids.pkl` | `05/00` | `05/04` | Tweet ids per partition. |
| `retweets_dataset.pkl` | `05/00` | `05/03` | Retweets split off from the partitions. |

### 13.5 `Data Sets/Cleaned Data/Tweet Sheets/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `top_test_ai_tweets.csv`, `top_test_art_tweets.csv` | `03/02` | — | Example sheets of top test tweets. |

### 13.6 `Data Sets/Classifiers_Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `HITL/base_playground_sample.csv` | `05/00b` | — | Sample for drafting label definitions. |
| `HITL/hitl_review_batch_00.csv` | `05/00` | — | First human-review sheet. |
| `HITL/hitl_review_batch_{PENDING_BATCH_TO_PROCESS}.csv` | `05/02` | — | Later review sheets. |
| `HITL/llm_bootstrap_labels_<stamp>.csv` | `05/01` | `05/02`, `05/03` | LLM bootstrap labels, one file per run (stamp = start date and time); consumers read the newest. The pre-2026-10-19 unstamped `llm_bootstrap_labels.csv` is the 2,000-tweet run. |
| `HITL/llm_bootstrap_labels_full_<stamp>.pkl` | `05/01` | — | Full bootstrap output with confidence and rationale. |
| `HITL/llm_bootstrap_seen_ids_AI.json` | `05/01` | `05/01` | Ids already sent to the LLM (resume without re-billing). |
| `HITL/llm_bootstrap_usage_{timestamp}.json` | `05/01` | — | API usage and cost per run. |
| `Final/final_annotated_tweets.pkl` (+ `.csv`) | `05/03` | `05/04` | Final annotations on the HITL remainder. |
| `Full_Inference/full_inference_annotated.pkl` (+ `.csv`) | `05/04` | — | Classifier annotations for the full AI corpus. |
| `Full_Inference/checkpoint_{n_processed}.pkl` | `05/04` | — | Checkpoints every 100k tweets. |

### 13.7 `Data Sets/Networks/`

One subfolder per stage (since 2026-10-10). The folder variables are one block, repeated in the Setup cell of `02/02`, `04/01` and `04/02`.

| File | Written by | Read by | What it is |
|---|---|---|---|
| `1_retweet_dicts/full_network_dict.pkl` | `02/02` | `02/02` | `{retweeted_author: {retweeter: n}}`: 374,368 retweeted authors, 9,638,407 retweets. |
| `2_full_graphs/Full_Network_Influence.gml` | `02/02` | — | Full retweet graph, influence orientation (reference). |
| `2_full_graphs/Full_Network_InfoFlow.gml` | `02/02` | `04/01` | Full retweet graph, information-flow orientation. |
| `3_backbones/Full_LWCC_Influence.gml` | `02/02` | — | LWCC backbone, influence orientation (reference). |
| `3_backbones/Full_LWCC_InfoFlow.gml` | `02/02` | `04/01`, `04/02`, layout | **LWCC backbone**, 3,264,499 authors (§5). |
| `3_backbones/Full_RetweetedOnce_Influence.gml` | `02/02` | — | Retweeted-once backbone, influence orientation (reference). |
| `3_backbones/Full_RetweetedOnce_InfoFlow.gml` | `02/02` | `04/01`, `04/02`, layout | **Retweeted-once backbone**, 202,710 authors (§5). |
| `3_backbones/90TS_LWCC.gml` (+ `.graphml`) | `04/01` | — | Optional reference backbone (`RUN_STRATEGY_2`): drop the lowest total-strength nodes until 10 % of weight is lost, then LWCC (2,315,573 authors). Not clustered. |
| `4_communities/<RetweetedOnce\|LWCC>/Full_<backbone>_InfoFlow_<method>.gml` | `04/01` | `04/01`; `…_leiden_fast.gml` also `04/02` | Backbone with the `community_<method>` attribute; five methods (§6.1). |
| `4_communities/<RetweetedOnce\|LWCC>/Full_<backbone>_author_communities.json` | `04/01` | `03/04` | `{author_id: {method: community}}`, all authors × five methods. |
| `5_visualizations/<NETWORK>/positions_{classic,linlog}.parquet` | `04/02` | — | Node tables per ForceAtlas2 recipe; the PNG maps sit beside them (§6.2). |
| `5_visualizations/<NETWORK>/positions_drl.parquet` | `04/02` | — | DrL layout (cell fails on Colab; §6.2). |
| `5_visualizations/<NETWORK>/network_with_layout.graphml` | `04/02` | — | Optional GraphML export for Gephi. |
| `5_visualizations/<NETWORK>/demo_network.gml` | `04/02` | `04/02` | Synthetic fallback when the selected network is missing. |
| `test/test_network_dict.pkl` | `02/02` | `02/02` | Test retweet counter. |
| `test/Test_{Network,LWCC,RetweetedOnce}_{Influence,InfoFlow}.gml` | `02/02` | — | Test graphs (the test data is too small to carry a retweeted-once backbone: 3 nodes). |

### 13.8 `Models/Topic Modeling/LDA/backbone_analysis/networks/<backbone>/`

The Colab default `OUT` of `03/04` (§7). The 2026-10-11 retweeted-once results are in the [Drive results folder](https://drive.google.com/drive/folders/1C4-R40VDIyhn5yFovLAmVjUBE5RSqDLc) (§7.2).

| File | Written by | Read by | What it is |
|---|---|---|---|
| `topics/author_documents.jsonl`, `documents.json` | `03/04` (prepare) | `03/04` (topics) | One cleaned document per eligible author; corpus metadata and hash. |
| `topics/grid_results.csv` | `03/04` (topics) | — | Model-selection grid. |
| `topics/model.json` | `03/04` (topics) | later stages | Settings, fit statistics, review flags; the later stages stop if the fit needs review. |
| `topics/top_terms.csv` | `03/04` (topics) | — | 20 top terms per topic. |
| `topics/lda_model.pkl`, `vectorizer.pkl` | `03/04` (topics) | weekly aggregation | The final model. |
| `topics/author_topics.csv` | `03/04` (topics) | `03/04` (affect, match) | Topic mixture and dominant topic per eligible author. |
| `affect/author_sentiment.csv`, `author_emotions.csv`, `coverage.json` | `03/04` (affect) | `03/04` (match) | Mean sentiment and emotion per author over all posts. |
| `data/matched_authors.parquet` (+ `.csv`) | `03/04` (match) | summarize, communities, layout, figure inputs, projection | One row per author: topics, affect, community. |
| `tables/*_network_subset.csv` | `03/04` (summarize) | figure inputs | Corpus and topic-level summaries. |
| `communities/figures/*.csv` | `03/04` (communities) | figure inputs | Community × topic and affect matrices. |
| `figure_inputs/…` | figure inputs, projection | renderers, weekly aggregation | Inputs of the figures (§8). |
| `weekly/weekly_*.csv` | weekly aggregation | temporal renderer | Author-balanced weekly series. |
| `figures/*` | renderers | — | Figures 1–11, light and dark. |
| `layout/matched_{nodes,edges}.parquet` | layout | viewer export | Matched-author subgraph for the viewer. |
| `viewer/nodes.json` | viewer export | `visualizations/author_network` | Viewer bundle. |

### 13.9 `Models/Topic Modeling/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `LDA/lda_grid_results.csv` | `03/03` | `03/03` | Tweet-topic grid. |
| `LDA/lda_k5_doc_topic_matrix.npy` | `03/03` | — | Document × topic matrix of the training sample. |
| `LDA/lda_k5_topics_metadata.json` | `03/03` | — | Metadata of the K = 5 tweet-topic model. |
| `bow_test_corpus.pkl` | `03/06` | `03/06` | Bag-of-words test corpus. |
| `full_sentences_corpus_embedding.pkl`, `test_sentences_corpus_embedding.pkl` | `03/05` | `03/05` | Per-tweet dictionaries fed to the embedding step. |
| `hf_embeddings.npy`, `processed_sentence_transformer_embeddings.npy`, `sentence_transformer_reduced_embeddings.npy` | `03/05` | `03/05` | Embeddings and their reduction. |

### 13.10 HPC: `/projects/ComputationalPhilosophyLab/TwitterDataAnalysis/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Data Sets/Cleaned Data/ai_full_classified_*.json`, `art_full_classified_*.json` | `03/01b` (8b) | `03/01b` (8a) | Per-model full outputs (§9); the AI files are copied to Drive for §7. |
| `Data Sets/Cleaned Data/{ai,art}_test_classified_{model_alias}.json` | `03/01b` | `03/01b` | Test outputs per model (copies on Drive under `Cleaned Data/Sentiment Analysis/`). |
| `Data Sets/Cleaned Data/Blocks/{ds_key}__{alias}__{num_blocks}blocks/block_{bid}.json` | `03/01b` (8b) | `03/01b` | Per-block outputs, reused only when their ids equal the input slice. |
| `Models/Sentiment/model_summary_test.csv`, `validation_{ds_key}_{model_alias}.csv` | `03/01b` | `03/01b` | Test-set validation summaries. |

When `03/04` runs on the cluster, `TWITTER_AI_DATA_ROOT` must point at a folder with the Drive layout (`Data Sets/Cleaned Data/`, `Data Sets/Networks/`). The outputs of the deleted `04c` under `Models/Topic Modeling/LDA/author_level/` are invalid (§13.3) and read by nothing.
