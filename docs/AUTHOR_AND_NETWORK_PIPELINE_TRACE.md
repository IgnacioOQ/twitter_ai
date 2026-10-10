---
status: active
type: reference
id: twitter_ai.author_and_network_pipeline_trace
description: Cell-by-cell trace of every tweet count, author count and network step across 02_Processing, 03_Analysis_and_Modeling and the pruning step of 04_Network_Analysis — numbers, the exact code that produced them, the discrepancies found, and a full inventory of every file the pipeline reads or writes (§12).
label: [dataset, network, authors, provenance, audit]
volatility: evolving
scope: project-specific
repository: [twitter_ai]
last_checked: '2026-10-10'
---

# Author & Network Pipeline Trace

This document answers one question for every step of the pipeline: **how many tweets and authors are there at this point, and which lines of code decided that?** It complements [DATASET_STATISTICS.md](DATASET_STATISTICS.md), which gives the narrative and the master statistics. Where the two overlap, the numbers agree; this document adds the code, the test-run numbers, every downstream consumer in `03_Analysis_and_Modeling`, and three discrepancies that the stored notebook outputs reveal.

> **Status 2026-10-09.** Shape of the pipeline after today's clean-up:
>
> - `02_Processing/` holds two notebooks: `01_api_data_to_dictionaries` and `02_sanity_check_and_network_generation`. The latter is the October 2026 rewrite of the network section (called `02b` until it was renamed over the original on 2026-10-09). It writes the full retweet graph in both orientations and two backbones — retweeted-once (202,710 authors) and LWCC (3,264,499 expected) — The full re-run **completed on 2026-10-09 (10:45 EDT) without errors**; its outputs are stored in the notebook, and `Full_LWCC_{Influence,InfoFlow}.gml` are now on Drive (3,264,499 authors, 7,670,516 edges, 98.14 % of the retweet weight). The pre-2026-10 original and `03_cleaning_tweets` were deleted (git history keeps both).
> - `04_Network_Analysis/` holds two notebooks: `01_network_analysis` clusters both backbones with five methods **on their information-flow files** (`<stem>_InfoFlow.gml`, since 2026-10-09; the 2026-10-07 run used the influence orientation for four of the five methods) and exports one community JSON per backbone — its full run **completed on 2026-10-09 (18:42 CEST) without errors**, so all ten partitions and both JSONs are on Drive (§9.3); `02_network_visualization` renders either backbone on a GPU runtime (its 2026-10-08 run failed on the cu13 RAPIDS image; fixed, not yet re-run; §9.5). Both are the October rewrites (`01`, `02b`), renamed over their originals on 2026-10-09; the originals and the one-off `patch_notebook.py` are deleted (git history).
> - The blog-figure package is on `main` under `src/blog_analysis/` (§11). The code that built its inputs — the 198,326 matched authors with community, topic, sentiment and emotion — is in no repository we can reach (§9.4); the published figures descend from the superseded retweeter backbone.
> - **2026-10-10 — `Data Sets/Networks/` reorganised by stage.** One subfolder per pipeline stage, each written by one notebook: `1_retweet_dicts/`, `2_full_graphs/`, `3_backbones/`, `4_communities/<RetweetedOnce|LWCC>/`, `5_visualizations/<NETWORK>/`, `test/` (§12.7). The folder variables are one block, repeated in the Setup cell of `02/02`, `04/01` and `04/02`; file names are unchanged. The flat folder — current outputs and every legacy file — was emptied on 2026-10-10 and the network stage is being re-run `02` (with `generate_data = True`: `full_network_dict.pkl` lived there) → `04/01` → `04/02`. **`02` finished on 2026-10-10 (04:37 EDT) without errors**, its outputs stored in the notebook; every count reproduces the 2026-10-09 run exactly, and `1_retweet_dicts/`, `2_full_graphs/`, `3_backbones/` and `test/` hold exactly the files of §12.7 (checked against the public listing). `4_communities/` and `5_visualizations/` stay empty until `04/01` and `04/02` run; `04/01` needs the `networks-stage-layout` branch on GitHub `main`, because on Colab it clones `main` for `src.network.*` (`run_modularity_workflow(..., output_folder_path=)`). The folder is shared read-only at [Drive: Data Sets/Networks/](https://drive.google.com/drive/folders/1PlVu_Li9nSI7IDLLfURp09s_1bAGXpMq?usp=sharing) — the public listing shows file names, dates and sizes and is how a session can check what is on Drive. The numbers in this document come from the runs before the reorganisation. `03/04` and `03/04c` now read `3_backbones/Full_LWCC_InfoFlow.gml` (same graph as the legacy LWCC file).
> - **2026-10-10 — `03_Analysis_and_Modeling` clean-up (in progress, branch `analysis-03-cleanup`).** Plan: delete the superseded notebooks; fix `01b`'s block resume (§6.2); fix `04c`'s author filter and sentiment join (§6.4); point `03`, `05`, `06` at the current corpus (§6.3, §6.5). Done so far: **B1** — `01_sentiment_analysis_v2` (2025 corpus, never run; `01b` covers its models) and `04_lda_author_topics` (every section is in `04c`) deleted; both remain in git history. **B2** — `01b` renamed `01b_sentiment_emotion_hpc` and its full-run section rewritten with verified blocks and a verifier for existing outputs (§6.2.2); the stored AI+Art sentiment and AI emotion files are known bad and the other two unverified until 8a runs on the cluster. **B4** — `03_lda_tweet_topics` rewired to the current sentiment file: two cells that could not run fixed, recovery/test/empty cells removed, analysis section kept and pointed at the enriched output (§6.3). **B5** — `05_embedding_mapping` pointed at the current corpus and made runnable; the collaborator's OpenAlex workflow moved to `05b_embedding_datamap_template` (§6.5). Drive `Cleaned Data/` was pruned the same day (§12.3).
> - Still open: the author-level LDA loader bug (§6.4), the sentiment block drift (§6.2), the first-generation corpus in `03/03` (§6.3), the viz re-render, and the blog figures on the corrected backbone (§11.3).
>
> **Reading conventions.** `02/02`, `04/01` and `04/02` without a cell number mean today's notebooks (`02_sanity_check_and_network_generation`, `01_network_analysis`, `02_network_visualization`); passages dated before 2026-10-09 call the same notebooks `02b`, `01` and `04/02`, and this document now writes `02`, `01` and `04/02` for them. A reference **with a cell number**, such as `02/02 cell 44` or `04/01 cell 18`, points into the **deleted original** of that notebook, whose stored outputs are the source of every number in §§1–5; the current `02` reproduces the corpus cells unchanged under the same headings, while the current `01` and `04/02` share only the method with their originals. `02/03` is the deleted cleaning notebook. `Full_Network.gml`, `.graphml`, `.gexf`, `.json`, `Test_Network.*`, `LWCC.gml/.graphml`, every `Final_OutThreshold1*` file, `viz_outputs_Final_leiden_fast/`, `AItrust_pruned_twits_with_sentiment_cleaned.json` and `top_retweets_by_topic_*.csv` are legacy files with no producing notebook (the `Networks/` ones were deleted in the 2026-10-10 reorganisation); `Full_Network_Influence.gml` is the same graph as `Full_Network.gml`.

All numbers below are **copied from the outputs stored in the committed `.ipynb` files**. Nothing was re-run. Cell references use the form `cell N` = 0-based position in the notebook's cell list, together with the nearest markdown heading, so a cell can be found either way.

## 0. Answer sheet

| Stage | Notebook / cell | Tweets | Authors | Notes |
|---|---|---:|---:|---|
| Raw API dumps flattened | `02/01` cell 16 | 36,560,405 records | 31,989,322 author records | both with duplicates |
| Unique tweet ids in raw file | `02/02` cell 20 | 25,637,570 | — | |
| **AI corpus** (`AItrust_twits_pruned_dict.json`) | `02/02` cell 20 | **17,410,035** | **4,775,711** | dedup + AI keyword + English + date ≥ 2022-10-31 |
| **AI+Art corpus** (`AItrust_Art_pruned_twit_dict.json`) | `02/02` cell 20 | **3,583,101** | **1,440,802** | AI corpus ∩ 60 art keywords; nested in AI corpus |
| Retweets inside the AI corpus | `02/02` cell 44 | 9,638,407 | — | the only tweets that create edges |
| Retweet network, all nodes | `02/02` cell 52 | — | **3,379,040** | 7,768,720 edges, weight 9,638,407 |
| Authors who were retweeted (dict outer keys) | `02/02` cell 51 | — | 374,368 | tqdm total of the outer loop |
| Largest weakly connected component | `02/02` cell 52, `04/01` cell 13 | — | 3,264,499 | 96.61 % of nodes |
| Total-strength 90 % pruning + LWCC (`90TS_LWCC.gml`) | `04/01` cell 16 (gated in today's `04/01`) | — | 2,315,573 | reference only, not clustered |
| Out-strength ≥ 1 (made ≥ 1 retweet) — **legacy** | deleted `04/01` cell 18 | — | 3,159,105 | before LWCC |
| **Out-strength ≥ 1 + LWCC** (`Final_OutThreshold1.gml`) — **legacy** | deleted `04/01` cell 18 | — | **1,984,599** | wrong population (retweeters); the set behind every community result before 2026-10-07 and the blog draft (§5.2) |
| Sentiment v4, AI corpus | `03/01b` (old Section 8) | 17,410,035 | — | emotion file short by 8,975 and misaligned (§6.2.1) |
| Sentiment v4, AI+Art corpus | `03/01b` (old Section 8) | 3,583,101 | — | sentiment file holds 4,057,340 lines of AI-corpus content (§6.2.1) |
| Tweet-level LDA / cleaning / top-K | `02/03`, `03/03` | 21,466,173 lines read | — | **old-generation corpus**, not the 17.41M one (see §6.3) |
| Author-level LDA v1 (notebook deleted 2026-10-10) | `03/04` cell 14 | 1,000,000 read → 4,558 kept | 1,704 | LCC filter matches the wrong field (see §6.4) |
| Author-level LDA v2 | `03/04c` cell 30 | 17,410,035 read → 65,615 kept | 8,000 → 3,682 (≥ 3 tweets) | same bug |
| Test branch, AI / AI+Art | `02/02` cell 17 | 881 / 216 | 723 / 198 | from one raw window file (233,094 records) |
| **In-strength ≥ 1 + LWCC** (`Full_RetweetedOnce_Influence.gml`, `02`) | `02` full pruning cell | — | **202,710** | authors retweeted ≥ 1× by someone else (363,618 before LWCC); 13.17 % of retweet weight; the corrected backbone (§9.3) |
| **LWCC, both directions** (`Full_LWCC_{Influence,InfoFlow}.gml`, `02`) | `02` full LWCC cell | — | **3,264,499** | direction-neutral: 25,135 self-loops removed inside the giant component, 7,670,516 edges, weight 9,458,703 (98.14 %) — the same graph as the pre-2026-10 strategy 1 (`04/01` cell 13), now in both orientations; run 2026-10-09; `01` clusters it beside the retweeted-once backbone (§5.0, §9.3) |
| Blog analysis "matched authors" | branch `origin/add-blog-figures-and-community-analysis` | — | **198,326** | with Leiden-directed community, K=12 topics, sentiment, emotion; provenance not in repo (see §9.4); within 2.2 % of the 202,710 above |

There is **no** author count anywhere in `02`–`04` for "AI+Art authors that are also in the pruned network". The only AI+Art author set computed is the 1,440,802 of `02/02` cell 20. See §7.

## 1. Stage 1 — Raw API pages → line-delimited dictionaries

Notebook: [02_Processing/01_api_data_to_dictionaries.ipynb](../notebooks/02_Processing/01_api_data_to_dictionaries.ipynb)

**What a tweet record is** — `process_tweet()` (cell 10, heading *Tweet Processing Functions*):

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

**Where the duplicates come from** — cell 16 (heading *Process All Data*) writes two kinds of records per page: the page's `data` tweets and the page's `includes.tweets` (referenced tweets pulled in by the API expansions). Both go through `print(dictionary, file=AItrust_twits_dict)`, so a tweet referenced by many others is written many times. Author objects from `includes.users` are written once per appearance as well.

Counts printed at the end of cell 16:

| Counter | Value |
|---|---:|
| Raw window files processed | 5,016 |
| Tweet records written (`count_processed`) | 36,560,405 |
| Author records written (`count_processed_authors`) | 31,989,322 |

The author file is never deduplicated and is not used by any downstream author count; every author count below is derived from the `author_id` field of tweets.

Test branch (cell 12): one file `testing.json` → 233,094 tweet records, 1,272 author records.

## 2. Stage 2 — Pruning to the AI and AI+Art corpora

Notebook: [02_Processing/02_sanity_check_and_network_generation.ipynb](../notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb) (cell numbers below are those of the pre-2026-10 version; the headings are unchanged in the current file)

### 2.1 The filters (cells 12, 14, 15 — heading *Prunning Functions*, *Keywords Filter*)

AI keyword test, cell 12:

```python
pattern = re.compile(r'(?:\bChatGPT\b|\bChat-GPT\b|\bGPT(?:-?3|-?4)?\b|\bLLMs?\b|\bBARD\b|\bBERT\b|'
                     r'\bLaMDA\b|\bLLaMA\b|\bMed-PaLM\b|Bing AI|artificial intelligence|large language models)', re.IGNORECASE)
_RE_AI_ALLOWED  = re.compile(r'(?:(?<![\'])?\bAI\b|#AI[\w-]*|@AI[\w-]*|AI-[A-Za-z])', re.IGNORECASE)
_RE_AGI_ALLOWED = re.compile(r'(?:\bAGI\b|#AGI[\w-]*|@AGI[\w-]*|AGI-[A-Za-z])', re.IGNORECASE)
_AI_DENYLIST_RE = re.compile(r'(?i)^(?:#|@)airdrop\w*$')
```

Art keyword test, cell 14 — 60 terms loaded from [keywords.txt](../notebooks/02_Processing/keywords.txt), whole-word, case-insensitive:

```python
_alts = '|'.join(re.escape(kw) for kw in _keywords)
keywords_pattern = re.compile(r'\b(' + _alts + r')\b', re.IGNORECASE)
def tweet_has_keyword(text: str) -> bool: ...
```

Date cutoff, cell 15: `earliest_date = datetime(2022, 10, 31, tzinfo=timezone.utc).date()`.

### 2.2 The single pruning pass (cell 20 — heading *Prune Full DS*)

The order of the checks is what makes the funnel additive. Relevant lines:

```python
seen_written = set(); unique_authors_ai = set(); unique_authors_art = set()
for line in tqdm.tqdm(in_f, total=36560405, desc="Pruning Full DS"):
    twit = json.loads(line); twid = twit.get('id'); txt = (twit.get('text') or '').strip()
    if twid is None or not txt:                      stats['dropped_missing_id_or_text'] += 1; continue
    seen_all_ids.add(twid)
    if twid in seen_written:                         stats['duplicates_skipped'] += 1; continue
    has_ai = bool(pattern.search(txt) or has_allowed_ai_form(txt) or has_allowed_agi_form(txt))
    if not has_ai:                                   stats['dropped_no_ai_keyword'] += 1; continue
    lang = (twit.get('lang') or '').lower()
    is_en = (lang.startswith('en') or seems_english(txt))
    if not is_en:                                    stats['dropped_non_english'] += 1; continue
    date_only = datetime.fromisoformat(tw_date.replace('Z', '+00:00')).date()
    if date_only < earliest_date:                    stats['dropped_before_date'] += 1; continue
    ...
    ai_f.write(json.dumps(twit, ensure_ascii=False) + '\n'); stats['written_ai_only'] += 1; seen_written.add(twid)
    author_id = str(twit.get('author_id', ''))
    if author_id: unique_authors_ai.add(author_id)
    if tweet_has_keyword(twit['text']):
        art_f.write(json.dumps(twit, ensure_ascii=False) + '\n'); stats['written_ai_art'] += 1
        if author_id: unique_authors_art.add(author_id)
```

Two consequences worth being able to state:

- **"Unique authors (AI+Art)" = authors with at least one tweet that matches an art keyword.** It is not a classification of authors; an author with 500 AI tweets and 1 art-matching tweet is in the set.
- **Dedup happens before the keyword test**, on tweet id, keeping the first occurrence in file order. The `seen_written` set holds only *written* ids, so a tweet that was dropped (no keyword) and reappears is re-tested, not counted as a duplicate.

Output of cell 20 (full run):

| Funnel line | Count |
|---|---:|
| Total records read | 36,560,405 |
| Unique tweet IDs seen | 25,637,570 |
| Duplicates skipped | 8,494,403 |
| Dropped (no AI keyword) | 10,429,950 |
| Dropped (non-English) | 76,363 |
| Dropped (before 2022-10-31) | 149,654 |
| **AI-only written** | **17,410,035** |
| **AI+Art written** | **3,583,101** |
| **Unique authors (AI-only)** | **4,775,711** |
| **Unique authors (AI+Art)** | **1,440,802** |
| Date range | 2022-10-31 → 2023-02-27 |
| Tweet types | original 4,061,626 · retweeted 9,638,407 · replied_to 3,221,048 · quoted 488,954 |

Check: 8,494,403 + 10,429,950 + 76,363 + 149,654 + 17,410,035 = 36,560,405 ✓.

Test branch, cell 17 (same code on `AItrust_twits_dict_test.json`): 233,094 records → 1,459 unique ids → 881 AI tweets / 723 authors, 216 AI+Art tweets / 198 authors, dates 2022-11-05 → 2022-11-15. Cell 18 re-reads both test files and confirms 881 and 216 lines with no duplicate ids.

## 3. Derived dictionaries — timeline, author corpus, network dict

Same notebook, cell 44 (heading *Generate Full Timeline and Network Dictionaries*). One pass over the **AI corpus only** (`AItrust_twits_pruned_dict.json`); the AI+Art file is never used for networks.

```python
type_of_network = 'retweeted'                       # cell 43
for line in tqdm.tqdm(AItrust_pruned_twits, total=17410035):
    twit = json.loads(line)
    ...
    if twit['type'] == type_of_network:
        author = str(twit['author_id'])
        referenced_author = str(twit['referenced_tweets_dictionary']['author_id'])
        if referenced_author in network_dict:
            if author in network_dict[referenced_author]:
                network_dict[referenced_author][author] += 1
            else:
                network_dict[referenced_author][author] = 1
        else:
            network_dict[referenced_author] = dict()
            network_dict[referenced_author][author] = 1
    # Author Corpus Dict
    author = str(twit['author_id'])
    author_corpus_dict.setdefault(author, []).append(twit['text'])   # (written out long-hand in the cell)
```

So `network_dict[retweeted_author][retweeter] = number of times retweeter retweeted retweeted_author`. The outer key is the **author who was retweeted**; the inner key is the **author who retweeted**.

Note on `referenced_tweets_dictionary`: Stage 1 sets it to the string `'N/A'` when the referenced tweet could not be resolved from the page's `includes`. For such a record `['author_id']` raises and the `except` branch increments `basic_counts_dict['exceptions']`. The stored output shows `'exceptions': 0`, so **all 9,638,407 retweets resolved to a retweeted author** and all of them became edge weight.

Output of cell 44:

| Quantity | Value |
|---|---:|
| Lines read / timeline entries / unique ids | 17,410,035 (all three equal) |
| `basic_counts_dict` | original 4,061,626 · retweeted 9,638,407 · replied_to 3,221,048 · quoted 488,954 · total 17,410,035 · exceptions 0 |
| `len(author_corpus_dict)` | not printed here; cell 59 loads it and reports 4,775,711 (= AI-corpus authors, as expected since every kept tweet has an author) |

Outputs: `Networks/full_network_dict.pkl`, `Cleaned Data/full_basic_counts_dict.pkl`, `full_timeline_dict.pkl`, `full_author_corpus_dict.pkl`.

## 4. The retweet graph

### 4.1 Construction (cell 51 — heading *Full Network*)

```python
G = nx.DiGraph()
for key in tqdm.tqdm(network_dict):            # key = retweeted author
    G.add_node(key)
    for referenced in network_dict[key]:       # 'referenced' here is actually the RETWEETER
        G.add_node(referenced)
        weight = network_dict[key][referenced]
        G.add_edge(referenced, key, weight=weight)   # edge: retweeter -> retweeted author
```

The variable name `referenced` is misleading (in cell 44 the inner key is the retweeter), but the edge that is written is **retweeter → retweeted author**, which is what [DATASET_STATISTICS.md §7](DATASET_STATISTICS.md) documents. Therefore, for every node:

- weighted **in-degree / in-strength = retweets received** (how often the author was retweeted);
- weighted **out-degree / out-strength = retweets made** (how often the author retweeted others).

The tqdm line of this cell (`374368/374368`) is the number of outer keys, i.e. **374,368 authors were retweeted at least once** inside the AI corpus. The remaining 3,004,672 nodes are authors who only appear as retweeters.

Node names are `str(author_id)`. When `nx.write_gml` serialises the graph (cell 53), each node gets a sequential integer `id` and the author id goes into `label`. This matters in §6.4.

### 4.2 Topology (cell 52)

```python
num_nodes = len(G.nodes); num_edges = len(G.edges)
total_weight = sum(d.get('weight', 1) for _, _, d in G.edges(data=True))
num_wcc = nx.number_weakly_connected_components(G)
lwcc_nodes = len(max(nx.weakly_connected_components(G), key=len))
top_retweeted  = sorted(G.in_degree(weight='weight'),  key=lambda x: x[1], reverse=True)[:10]
top_retweeters = sorted(G.out_degree(weight='weight'), key=lambda x: x[1], reverse=True)[:10]
```

| Metric | Value |
|---|---:|
| Nodes (authors) | 3,379,040 |
| Directed edges (unique retweeter→retweeted pairs) | 7,768,720 |
| Total weight | 9,638,407 (= number of retweet tweets, as it must) |
| Weakly connected components | 49,464 |
| Largest WCC | 3,264,499 (96.61 %) |
| Most retweeted author | 1156482001, 92,733 retweets received |
| Most active retweeter | 1356588046616043527, 45,503 retweets made |

Serialisation (cells 53–55) to `Full_Network.gml / .graphml / .gexf / .json`, each read back and checked with `G.nodes == F.nodes` and `G.edges == F.edges` → `True`.

Test branch (cells 26, 33, 34): 881 tweets → 153 retweeted authors → 446 nodes, 334 edges, weight 363, 122 components, largest 32.

### 4.3 The three author populations of Stage 2 (cell 59 — *Dataset Statistics & Summary Report*)

```python
'unique_authors_pruned_ai':  total_authors_corpus,              # len(full_author_corpus_dict.pkl) = 4,775,711
'unique_authors_pruned_art': prune_stats.get('unique_authors_art', 0),   # 1,440,802
'unique_authors_in_network': net_stats.get('num_nodes', 0),     # 3,379,040
```

Why 3,379,040 < 4,775,711: an author enters the graph only by retweeting or being retweeted *within the AI corpus*. 1,396,671 authors (29 %) wrote only originals, replies or quotes and have no edge.

## 5. The author backbones

Every later notebook is meant to be restricted to an author set cut from the retweet graph, so the cut belongs in this trace. Since 2026-10-09 the cuts are made in `02_Processing/02` (both backbones) and read by `04_Network_Analysis/01`; the pre-2026-10 `04/01` made its own cut, which is described in §5.1 because every community result stored before 2026-10-07 and the blog draft (§11) descend from it.

### 5.0 The backbones in the current pipeline

All start from the full influence graph (`Full_Network_Influence.gml`, 3,379,040 authors, 7,768,720 edges), drop the 32,216 self-loops, cut, and keep the largest weakly connected component. Each is written in both orientations (`*_Influence.gml`: retweeter → retweeted; `*_InfoFlow.gml`: the transpose, edge retweeted → retweeter). **The information-flow file is the canonical one:** `04/01` clusters both backbones on it with five methods (`BACKBONE_STEMS`, outputs `4_communities/<backbone>/<stem>_InfoFlow_<method>.gml`) and `04/02` renders it; the influence files are written for reference and read by nothing.

| Backbone | Written by | Rule | Nodes | Edges | Weight kept | Files |
|---|---|---|---:|---:|---:|---|
| **LWCC** (direction-neutral) | `02`, full LWCC cell (helper `export_lwcc_both_directions`) | no degree cut; self-loops removed; giant component | **3,264,499** | 7,670,516 | 98.14 % (9,458,703 retweets) | `3_backbones/Full_LWCC_{Influence,InfoFlow}.gml` (identical to the pre-2026-10 strategy 1) |
| **Retweeted once** | `02`, full pruning cell (helper `prune_retweeted_at_least_once`) | in-strength ≥ 1 on the influence graph = out-strength ≥ 1 on the flow graph (363,618 authors), then LWCC | **202,710** | 882,530 | 13.17 % (1,269,747 retweets) | `3_backbones/Full_RetweetedOnce_{Influence,InfoFlow}.gml` |
| Total-strength 90 % (reference only) | `04/01`, gated by `RUN_STRATEGY_2` (`prune_network_by_deletion`) | delete lowest in+out strength nodes until 10 % of weight is lost, then LWCC | 2,315,573 | 6,721,590 | 90 % | `3_backbones/90TS_LWCC.gml` (only when the flag is on; not clustered) |

The two clustered backbones answer different questions. The LWCC keeps every retweeter and every edge, so community structure is informed by who amplifies whom; "retweeted at least once" can then be applied as an author filter on its community table. The retweeted-once backbone is exactly the amplified population, but it discards 87 % of the retweet volume because pure retweeters leave with their edges (§9.3). The `02` notebook asserts, for each backbone, that the two orientations hold the same author set and the same number of edges.

### 5.1 Legacy: the out-strength backbone of 1,984,599 retweeters

The pre-2026-10 `04_Network_Analysis/01_network_analysis.ipynb` (deleted 2026-10-09, in git history) read `Full_Network.gml` and cut three backbones of its own:

| Strategy | Cell | Rule | Nodes | Edges | Output (legacy files, removed with the 2026-10-10 reorganisation) |
|---|---|---|---:|---:|---|
| 1 — LWCC only | 13 | no pruning | 3,264,499 | 7,670,516 | `LWCC.gml` (same graph as today's `Full_LWCC_Influence.gml`) |
| 2 — total-strength 90 % | 16 | delete lowest in+out strength nodes until 10 % of weight is lost (958,959 nodes removed), then LWCC | 2,315,573 | 6,721,590 | `90TS_LWCC.gml` (still produced by today's `04/01` when `RUN_STRATEGY_2` is on) |
| 3 — out-strength ≥ 1 | 18 | keep nodes with out-strength ≥ 1 (3,159,105 of 3,379,040), then LWCC | **1,984,599** | 4,472,376 | `Final_OutThreshold1.gml` — **wrong population**, no longer produced |

Strategy 3 was the set used for community detection, the AMI table, `Final_OutThreshold1_author_communities.json` and the social-media maps until 2026-10-07, and it is the population behind the blog draft's 198,326 matched authors (§9.4, §11). Its rule is still in [src/network/network_pruning.py:226](../src/network/network_pruning.py#L226), now called by no notebook:

```python
def prune_by_out_strength_threshold(g, threshold=1.0):
    ...
    g_clean = g.simplify(loops=True, multiple=False)          # self-loops removed first
    out_str = g_clean.strength(mode="out", weights=weight_attr)
    nodes_to_keep = [i for i, s in enumerate(out_str) if s >= threshold]
    ...
    pruned = pruned.components(mode="weak").giant()
```

Given the edge direction of §4.1, **`out_strength >= 1` kept authors who made at least one retweet (of someone else)**. It did *not* select authors who were retweeted. The threshold deletes the *node*, so an author who was retweeted (even heavily) but never retweeted anyone was removed outright, together with all edges pointing at them. Of the 3,159,105 retweeters, 1,984,599 sat in one connected component; the other 1,174,506 were in small islands and were dropped by the LWCC step. Weight kept: 60.46 % of the original.

The intended population, "authors retweeted at least once", is the in-strength condition on the same graph; that is what today's `02` implements (§5.0, second row).

Two readings of the same graph keep this straight. Under the **influence** reading (the graph as built), A → B means A retweeted B, in-strength is retweets received, and the legacy rule kept nodes with ≥ 1 *outgoing* edge. Under the **information-flow** reading (the transpose), A → B means B retweeted A, and the same rule keeps nodes that *received* information from ≥ 1 source while dropping every pure source. "Retweeted at least once" is a source condition: in-strength in the influence graph, out-strength in the flow graph. Direction also matters for Infomap, whose random walker follows edge direction: the deleted `04/01` ran it on the influence orientation, i.e. against information flow, while today's `04/01` runs it on the `*_InfoFlow.gml` file. The directed Leiden quality function is invariant under transposition and the three undirected methods are unaffected.

The deleted notebook's cell 27 (*Export community memberships as JSON*) read the `label` attribute (= author id) of each `Final_OutThreshold1_<method>.gml` and wrote `Final_OutThreshold1_author_communities.json` = `{author_id: {method: community}}`; its key count (expected 1,984,599) was never stored.

### 5.2 Does the backbone choice reach the sentiment or topic analyses?

Not yet. No notebook in `02` or `03` reads any backbone file or community JSON. What each analysis actually restricted on:

| Analysis | Author restriction intended | Author restriction actually applied |
|---|---|---|
| Sentiment v1 (`03/01`) | none | none — all 17,410,035 tweets |
| Sentiment v4 (`03/01b`) | none | none — 17,410,035 AI and 3,583,101 AI+Art tweets |
| Tweet-level LDA, topics in time, top-K per topic (`03/03`; the deleted `02/03`) | none | none — stored results on the first-generation corpus; rewired to the current one 2026-10-10, not yet re-run (§6.3) |
| Author-level LDA v1 / v2 (`03/04`, deleted 2026-10-10; `03/04c`) | authors in the LWCC (3,264,499; both retweeters and retweeted, no direction involved) | legacy low-id accounts only, because the loader matches GML `id` instead of `label` (§6.4) |
| Classifiers (`05_Classifiers`) | none | none — they read the AI+Art tweet file directly |

So the sentiment and tweet-topic results cover every author in the corpus, including the 1,396,671 who never retweeted or were retweeted, and the author-level LDA was designed against the undirected component, not the retweeter set. The choice between "everyone in the component" (LWCC, 3,264,499) and "authors who were retweeted" (202,710) becomes consequential the first time a downstream notebook restricts to one of the `04/01` JSON exports; the author-level LDA re-run (§8) and the blog figures on the corrected backbone (§11.3) are where that will happen.

## 6. Consumers in `03_Analysis_and_Modeling` (and `02/03`)

### 6.1 `01_sentiment_analysis.ipynb` (v1, CardiffNLP sentiment only)

Cell 18 (heading *For Full Data Set*):

```python
input_path        = cleanedds_folder / 'AItrust_twits_pruned_dict.json'
final_output_path = cleanedds_folder / 'AItrust_pruned_twits_with_sentiment.json'
num_blocks = 10
total_tweets = count_lines(input_path)           # → 17410035
```

Stored output: `📊 Total tweets: 17410035`; ten blocks of 1,741,003 lines (last 1,741,008) all reported complete; merged to `.../Cleaned Data v2/AItrust_pruned_twits_with_sentiment.json`. The code has since been pointed at `Cleaned Data/`, and on Drive (listing of 2026-10-10) `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` is dated 2026-05-12 with its ten block files dated 2026-05-09…12 — i.e. **this 17.41M-tweet output is the file now in `Cleaned Data/`**. No first-generation sentiment file is left on Drive. The block files were deleted on 2026-10-10 after the merge; `Cleaned Data v2/` is not inside `Data Sets/`.

**Same resume rule as the old `01b` (§6.2.1), unverified content.** Cell 18 reuses a block file with **≥ 95 %** of the expected lines and its merge skips missing blocks silently. The stored run reused all ten blocks at *exact* counts, so nothing in the log is wrong — but nothing checked that the blocks held the right tweets, and empty-text tweets are written ahead of their batch here too (order only). `03` reads the file tweet by tweet, so order does not matter to it; content would. A cheap check is to compare the first/last tweet ids of each 1,741,003-line slice with `AItrust_twits_pruned_dict.json` (the verifier of `01b` §6.2.2 does exactly this for its own outputs). Not done yet.

### 6.2 `01b_sentiment_emotion_hpc.ipynb` (HPC, 10 models; `01b_sentiment_emotion_v4_hpc.ipynb` until 2026-10-10)

Datasets (`cleanedds_folder` = `/projects/ComputationalPhilosophyLab/TwitterDataAnalysis/Data Sets/Cleaned Data`):

| Key | File | Tweets |
|---|---|---:|
| `ai_test` | `AItrust_twits_pruned_dict_test.json` | 881 |
| `art_test` | `AItrust_Art_pruned_twit_dict_test.json` | 216 |
| `ai_full` | `AItrust_twits_pruned_dict.json` | 17,410,035 |
| `art_full` | `AItrust_Art_pruned_twit_dict.json` | 3,583,101 |

The ten models run on the two test sets in one pass each (Section 4, `classify_file`). Two models — `twitter-roberta-base-sentiment-latest` and `twitter-roberta-base-emotion-multilabel-latest` — run on both full corpora (Section 8): four jobs, each split into blocks so a killed GPU session can resume.

#### 6.2.1 The stored full run (before 2026-10-10) is not trustworthy

The merged result files, as counted by the stored Section 9 output:

| Result file | Lines | Expected | Δ |
|---|---:|---:|---:|
| `ai_full × twitter-roberta-base-sentiment-latest` | 17,410,035 | 17,410,035 | 0 |
| `ai_full × twitter-roberta-base-emotion-multilabel-latest` | 17,401,060 | 17,410,035 | **−8,975** |
| `art_full × twitter-roberta-base-emotion-multilabel-latest` | 3,583,101 | 3,583,101 | 0 |
| `art_full × twitter-roberta-base-sentiment-latest` | **4,057,340** | 3,583,101 | **+474,239** |

**Mechanism.** Three defects of the old Section 8 combine:

1. **Resume trusted line counts.** An existing block file was reused when it held **≥ 95 %** of the expected lines — no upper bound, no check that it held the right tweets. A block from another run (other corpus, other block layout, other input version) passes.
2. **AI+Art blocks had no dataset in their folder name.** The AI+Art cell wrote to `Blocks/<model alias>/`; the AI cell to `Blocks/ai_full__<model alias>/` (the prefix was evidently added later, and only there). Any earlier run that used `Blocks/<model alias>/` left blocks the AI+Art run then accepted.
3. **Empty-text tweets jumped their batch.** A tweet whose preprocessed text was empty was written immediately, while the tweets before it were still waiting in the batch buffer, so output order differed from input order even without any resume.

The old merge only opened files in `w` mode and concatenated whole blocks — nothing was ever appended twice, so this section's earlier explanation ("a block appended twice") was wrong.

**Evidence in the stored outputs.**

- *AI+Art sentiment.* The run's log shows all ten blocks *skipped* ("Block i done. Skipping."): every block pre-existed. Its sanity check prints as lines 1–3 the very same tweets as lines 1–3 of the **AI** sentiment file (`Hey you need perfect scores and completely AI proof content…`, `RT @lallamapic…`, `RT @lexfridman…`), and the first of them contains none of the 60 art keywords. The file holds AI-corpus content and is 474,239 lines too long. **Its label distribution (cell "9a") must not be used.**
- *AI emotion.* Block 0 was accepted at 1,732,028 of 1,741,003 lines, and block 9 was re-run after being found at 802,664 lines. The file's first line is a different tweet from the first line of the AI sentiment file, i.e. block 0 came from another run. The file is 8,975 tweets short **and** misaligned.
- *AI+Art emotion.* Its line count is right, but its blocks also sat in an un-prefixed folder and were all skipped; correct count does not prove correct content.
- *AI sentiment.* No anomaly visible in counts or first lines; unverified.

Full-corpus label distributions stored in the old run (for the record only; the AI+Art row is on the corrupt file):

| Model × corpus | n | negative | neutral | positive |
|---|---:|---:|---:|---:|
| sentiment-latest × AI General | 17,410,035 | 21.2 % | 47.9 % | 30.8 % |
| sentiment-latest × AI+Art | 4,057,340 (corrupt) | 31.4 % | 42.1 % | 26.5 % |

#### 6.2.2 The fix (2026-10-10, branch `analysis-03-cleanup`)

- **One order-preserving writer**, `classify_lines()` (Section 2), used by both the test pass and the full runs: empty-text tweets wait in the same queue, so output order = input order.
- **Block folders** `Blocks/<dataset>__<model alias>__<N>blocks/` — never shared across datasets, models or block layouts.
- **A block is reused only if its tweet ids equal the input slice exactly** (`block_is_valid`: same count, same order). Otherwise it is recomputed and checked again.
- **Merge to `<output>.tmp`, verify, then rename.**
- **8a — verify existing outputs (CPU)**: `verify_classified_output()` compares every output with its input tweet by tweet and reports `aligned` (same ids, same order), `same ids` (same set, different order — still fine for joins by tweet id) or `FAIL` (count differs, unknown or duplicated ids, tweets without this model's result). The status table is `FULL_STATUS`.
- **8b — (re)run (GPU)**: `RERUN = 'failed'` re-runs only the jobs 8a failed; `'all'` re-runs everything.
- **8c — legacy block folders**: lists the old `Blocks/<model alias>/` and `Blocks/ai_full__<model alias>/` folders; deletes them only with `DELETE_LEGACY_BLOCK_DIRS = True`.
- **Section 9 loads only outputs that passed** 8a/8b.

Tested locally with a stub in place of the model on a synthetic 103-tweet corpus with 12 empty-text tweets: a fresh run is `aligned`; a second run reuses all blocks; a block with one line removed is the only one recomputed; a rotated file is reported `same ids`, a padded file `FAIL`; output order equals input order in both the test and the full path. Not yet run on the HPC data — **the 8a verdicts for the four stored outputs are the next step** (owed: run 8a on the cluster, then 8b for every job that fails, then 8c with the flag on).

### 6.3 `02/03_cleaning_tweets.ipynb` (deleted 2026-10-09) and `03/03_lda_tweet_topics.ipynb` — the old-generation corpus, and the 2026-10-10 rewiring

Both read `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (cell 3 of `02/03`; cell 8 of `03/03`, where it is also the source for the K = 5 enrichment file `AItrust_pruned_twits_with_sentiment_and_topics_k5.json`). Their stored outputs do **not** match the 17.41M corpus:

- `02/03` cells 13, 16 and 21 all set `TOTAL_LINES = 22416373` and their progress bars stop at **21,466,173** lines, the last of which is malformed (`line 21466173: JSONDecodeError`). The file therefore has 21,466,173 lines and was cut mid-record.
- `02/03` cell 13: of those lines, 12,537,708 were skipped as non-original, leaving 8,928,465 originals for the top-150-per-topic extraction. The 17.41M corpus has only 4,061,626 originals.
- `03/05_embedding_mapping.ipynb` cell 29 streams `AItrust_pruned_twits.json` with `total=22416373` and cell 30 reports `len(corpus_dict)` (unique ids) as **14,885,897**, i.e. the older pruned file had ~7.5M duplicate ids.

So there were **two generations** of the corpus:

| Generation | File | Lines | Unique ids | Produced by |
|---|---|---:|---:|---|
| 2025 | `AItrust_pruned_twits.json` → `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (+ `_and_topics_k5`) | 22,416,373 (sentiment copy truncated at 21,466,173) | 14,885,897 | earlier version of `02/02`, before dedup and the stricter keyword filter — **no longer on Drive** (listing of 2026-10-10) |
| 2026 | `AItrust_twits_pruned_dict.json`, `AItrust_Art_pruned_twit_dict.json`, `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (written as `Cleaned Data v2/…`, §6.1) | 17,410,035 / 3,583,101 | = lines | `02/02` (the corpus file is dated 2026-08-22, when `processed_text` was added; the sentiment file 2026-05-12) |

The tweet-level LDA results stored before 2026-10-10 (k = 5 topics, topic × sentiment counts, "topics in time", top retweets per topic CSV) were all computed on the first generation; none of those output files is on Drive any more. Its absolute counts (e.g. Topic 0: 4,867,664 positive / 7,051,674 neutral / 3,534,272 negative in `02/03` cell 22) cannot be reconciled with the 17.41M corpus and should not be quoted next to it.

(The "Disconnected from runtime" timestamps stored in the notebooks — `02/01` 2025-08-20, `03/01` 2025-08-26, `02/02` 2026-08-22 — are the last time the *disconnect cell* ran, not necessarily when the cells above it ran, so they date sessions only loosely.)

**`03/03` rewired (2026-10-10, B4 of the 03 clean-up).** The notebook's input name already matched the current file, so the input did not change; what changed:

- **Paths cell would have crashed**: it built `DATA_DIR` from `BASE`, which was commented out (the Setup cell defines `BASE_PATH`) → `NameError`. It now uses `cleanedds_folder` and `topic_models_folder / 'LDA'`.
- **Imports cell would have crashed**: two pasted install blocks sat unindented under `except ImportError:` / `except Exception:` (umap, gensim) → `IndentationError`. Replaced by plain imports; the Setup cell installs `gensim umap-learn pyLDAvis` (unpinned — the numpy 1.26.4 / scipy 1.10.1 downgrade and the forced runtime restart are gone; those pins have no wheels for Colab's Python 3.13). pyLDAvis's scikit-learn adapter is imported from `pyLDAvis.lda_model` (3.4+) with `pyLDAvis.sklearn` as fallback; the old code fell back to installing pyLDAvis 3.2.2.
- **Analysis section** (now *4 · Analysis — topics × sentiment*) reads the plain enriched file `OUTPUT_JSONL` (it read a `.jsonl.gz` only the deleted merge-recovery section produced), takes topic labels from `TOPIC_LABELS[K_TARGET]` (the hand-written 2025 labels — "General AI / Tools & Utility", "Memes / Culture", "Crypto / NFT / Giveaways / Hype", "Tech / Research / ML Ethics", "ChatGPT vs Google Bard" — described the 2025 model), and computes monthly proportions with `groupby(...).transform('sum')` (the old `groupby.apply` loses the grouping columns on pandas 3). The sanity peek prints `sentiment_label` (it printed a non-existent `sentiment`).
- **Removed**: *3 · Merge Recovery* (Drive FUSE workaround that wrote the `.gz`), *TEMPORARY TESTS* (an "author-lite" K = 5 experiment writing `authorlite_k5_topics_metadata.json`), 21 empty cells, and a comment-only duplicate of the header. 66 → 27 cells.
- **Tested**: the five analysis cells run on a synthetic enriched file (pandas 3.0.2). The training/enrichment half (scikit-learn LDA, gensim coherence, pyLDAvis) was not run locally.
- **Known limitation, unchanged**: the model trains on the **first** 25,000 tweets of the input (`TRAIN_SAMPLE_MAX`), not a random sample.

### 6.4 `04_lda_author_topics.ipynb` (v1) and `04c_lda_author_topics_v2.ipynb` — the LCC filter matches the wrong GML field

*`04_lda_author_topics.ipynb` was deleted on 2026-10-10 (every section of it is in `04c`); its cell references below point into git history.*

Both notebooks intend to restrict author documents to authors in the largest connected component. The loader (identical in `03/04` cell 13 and `03/04c` cell 25):

```python
def load_lcc_node_ids_lightweight(gml_path):
    nodes = set()
    with open(gml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id "):
                node_id = line[3:].strip().strip('"')
                nodes.add(node_id)
    return nodes
lcc_nodes = load_lcc_node_ids_lightweight(LCC_GML)      # → "Loaded LCC node count: 3264499"
```

and the filter (`03/04` cell 14, `03/04c` cells 30 and 63):

```python
aid = str(tw.get("author_id", "")).strip()
if lcc_nodes is not None and aid not in lcc_nodes:
    continue
```

As §4.1 notes, in a GML written by NetworkX (and re-written by igraph in `04/01` cell 13 as `LWCC.gml`) the `id` line is the **sequential integer index 0…3,264,498**; the author id is in the `label` line. The loader collects indices, so `aid not in lcc_nodes` is true for every author whose Twitter id is larger than 3,264,498 — which is every account created after early 2007. The notebook's own check in `03/04c` cell 29 shows this:

```text
LCC nodes sample: ['680530', '1314135', '2206707', '895806', '1036202']
Tweet author_id: 1306286218150391810
str(author_id) in lcc_nodes: False
```

Effect on the stored runs:

| Run | LCC file | Tweets read | Kept | Authors | Then |
|---|---|---:|---:|---:|---|
| v1, `03/04` cell 14 | `Largest_Weakly_Connected_Component.gml` (Drive; not produced by any committed notebook) | 1,000,000 | 4,558 | 1,704 | LDA on 1,704 docs |
| v2, `03/04c` cell 30 | `LWCC.gml` | 17,410,035 | 65,615 | 8,000 | `MIN_TWEETS_PER_AUTHOR = 3` → 3,682 (cell 33); 108-model grid, UMAP, HDBSCAN on these |
| v2, `03/04c` cell 63 | same | 17,410,035 | 65,615 written to `AI_pruned_tweets_with_topic_weights_v2.json` | — | |

The correct filter would retain the large majority of tweets: all 9,638,407 retweets are by graph authors, 96.6 % of graph nodes are in the LWCC, so at least ~55 % of the corpus should pass. 0.38 % passed. The 8,000 retained authors are not a sample of the component; they are the subset of legacy low-id accounts, selected by account age.

The same notebook defines a correct loader that is never called (`03/04c` cell 16):

```python
def load_lcc_node_ids(gml_path):
    g = nx.read_gml(gml_path)           # NetworkX uses 'label' as the node name
    return set(str(n) for n in g.nodes())
```

Minimal fix: in `load_lcc_node_ids_lightweight` replace `line.startswith("id ")` with `line.startswith("label ")` (the value is quoted in the file, and `.strip('"')` already handles that), or point `LCC_GML` at `Final_OutThreshold1.gml` and use the `label` field, which is the author set §5 defines. Every result of `03/04` and `03/04c` (grid search, canonical model, author-topic UMAP/HDBSCAN, topic weights file) needs to be regenerated after that; nothing in those notebooks can be interpreted as "authors of the connected component" as it stands.

### 6.5 Other notebooks in `03`

- `01_sentiment_analysis_v2.ipynb` — **deleted 2026-10-10**: no stored outputs, read only the first-generation `AItrust_pruned_twits.json`, and its models are covered by `01b`.
- `02_extract_examples.ipynb` — reads the AI and AI+Art *test* files only; no counts stored.
- `05_embedding_mapping.ipynb` — sentence embeddings → UMAP → HDBSCAN on the test corpus, then on the full corpus restricted to tweets with `like_count > 0`. **Updated 2026-10-10 (B5):** reads the current `AItrust_twits_pruned_dict{_test,}.json` (it read the first-generation `AItrust_pruned_twits{_test,}.json`); every path joined with `/` (they were `folder + 'name'` on a `Path` → `TypeError`, which the graph showed as the unresolvable `Cleaned DataAItrust_pruned_twits*.json`); `%%time` moved to line 1 in three cells (below a comment it is a `UsageError`); empty cells dropped (100 → 43 cells with the template moved out). The corpus-loading cells were run on a synthetic test file; the embedding/UMAP/HDBSCAN half was not run locally. `.npy`/`.pkl` files already in `Models/Topic Modeling/` predate this and come from the first-generation corpus.
- `05b_embedding_datamap_template.ipynb` — **split out of `05` on 2026-10-10.** A collaborator's workflow for OpenAlex scientific abstracts (Specter-2 embeddings → UMAP → EVoC / HDBSCAN → cluster labels from a *local* Llama 3 70B via `llama_cpp` → DataMapPlot), shared as a template. It reads none of this project's files (`one_millions_filtered_OA_sample_longer_abstracts.bz` is not on Drive), so it has no edges in the pipeline graph.
- `06_topic_modeling_appendix.ipynb` — sklearn/gensim LDA demos; no corpus counts.

## 7. The AI+Art author subset — what exists and what does not

What exists:

1. **1,440,802 authors** with ≥ 1 art-keyword tweet in the AI corpus — `02/02` cell 20, `unique_authors_art`. Nested inside the 4,775,711.
2. **3,583,101 AI+Art tweets** with v4 sentiment and emotion labels — `03/01b`; the stored sentiment file is corrupt (4,057,340 lines of AI-corpus content, §6.2.1) and must be regenerated with the fixed Section 8.
3. The test-branch counterpart: 198 authors / 216 tweets.

What does not exist anywhere in `02`–`04`:

- an AI+Art author set intersected with the retweet network (3,379,040), its LWCC (3,264,499) or the pruned network (1,984,599);
- an AI+Art retweet network (the network pass in `02/02` cell 44 reads only the AI file);
- a per-author "share of art tweets" or any author-level art label. The `05_Classifiers` notebooks label *tweets* from the AI+Art file; they do not aggregate to authors.

Any of these is a single pass: load `Final_OutThreshold1_author_communities.json` (keys = pruned-network author ids, once §5's direction question is settled), stream `AItrust_Art_pruned_twit_dict.json`, and intersect on `str(author_id)`.

## 8. What a re-run is needed for, and what it is not

Nothing in §§1–5 needs re-running: every number is in the stored outputs and the funnel is internally consistent.

Re-runs that would change results:

| Item | Notebook | Why |
|---|---|---|
| Author-level LDA | `03/04c` (`03/04` deleted 2026-10-10) | LCC filter matches GML indices, not author ids (§6.4) — all stored results are on the wrong author set |
| AI+Art sentiment (full) | `03/01b` 8a → 8b | stored file is AI-corpus content, 474,239 lines too long (§6.2.1); regenerate with the verified block runner (§6.2.2) |
| AI emotion-multilabel (full) | `03/01b` 8a → 8b | block 0 from another run: 8,975 short and misaligned (§6.2.1); regenerate (§6.2.2) |
| AI sentiment, AI+Art emotion (full) | `03/01b` 8a | unverified; 8a decides whether 8b re-runs them (§6.2.2) |
| Tweet-level LDA, topics × sentiment, topics in time | `03/03` | stored results were on the first-generation corpus (§6.3); notebook rewired 2026-10-10 to `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (the 2026 file) — run it on Colab |
| Pruned network + communities + JSON export | `04/01` | the intended population is "retweeted ≥ 1"; the rule implemented is "retweeted someone ≥ 1" (§5, §9). **Done 2026-10-07** via `02` + `01` (§9.3); `04/01` outputs are superseded, not deleted. **Extended 2026-10-09**: `02` / `01` also produce and cluster the direction-neutral LWCC backbone (`Full_LWCC_*`); both run 2026-10-09, `01` on the information-flow files of both backbones (§9.3) |
| Network visualisations (ForceAtlas2, DrL, adjacency blocks) | `04/02` | rendered on `Final_OutThreshold1_leiden_fast.gml`, i.e. the retweeter backbone (§9.2). **2026-10-08 run failed** before layout (cuGraph install on the new cu13 Colab image; patched 2026-10-09, §9.5). To be re-run on `Full_RetweetedOnce_InfoFlow_leiden_fast.gml` and on `Full_LWCC_InfoFlow_leiden_fast.gml`, one network per run |
| Community-based blog figures 7–11 (and weekly figures 2, 3, 5 if the matched set is network-derived) | branch `add-blog-figures-and-community-analysis` | only if the 198,326 matched authors / their `leiden_directed` labels descend from `Final_OutThreshold1` — to be verified (§9.4) |

**Corrected notebooks (added 2026-10-07; all three renamed over their originals on 2026-10-09).** Three copies implemented the fix without touching the originals: [02_Processing/02_sanity_check_and_network_generation.ipynb](../notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb) (named `02b_influence_and_flow_networks.ipynb` until the rename) builds the influence and information-flow networks and the in-strength ≥ 1 backbone (`Full_RetweetedOnce_{Influence,InfoFlow}.gml`); [04_Network_Analysis/01_network_analysis.ipynb](../notebooks/04_Network_Analysis/01_network_analysis.ipynb) (named `01b_network_analysis_retweeted_once.ipynb` until the rename) runs the five community methods on it (Infomap on the flow file), aligns AMI/ARI by author id and exports `Full_RetweetedOnce_author_communities.json`; [04_Network_Analysis/02_network_visualization.ipynb](../notebooks/04_Network_Analysis/02_network_visualization.ipynb) (named `02b_network_visualization_retweeted_once.ipynb` until the rename) renders the maps from those files (registry default `RetweetedOnce_leiden_fast`).

**Extended 2026-10-09 — a second, direction-neutral backbone.** `02` additionally exports the largest weakly connected component of the full network, self-loops removed, in both orientations (`Full_LWCC_{Influence,InfoFlow}.gml`; helpers `largest_weakly_connected_component` and `export_lwcc_both_directions`, which cuts the information-flow copy from the transpose on the same author set and asserts equal node sets and edge counts; stats under `lwcc` in `full_dual_network_stats.json`). `01` loops over `BACKBONE_STEMS = ['Full_RetweetedOnce', 'Full_LWCC']` for the backbone check, the five methods, AMI/ARI and the export, writing `Full_LWCC_InfoFlow_<method>.gml` (all five methods on the flow file since the same day) and `Full_LWCC_author_communities.json`; its own Strategy-1 recompute cell was removed as redundant, and a `SKIP_EXISTING_COMMUNITY_FILES` flag makes the multi-hour run resumable. `04/02` lists seven `LWCC_*` registry keys next to the `RetweetedOnce_*` ones. The `Full_LWCC_*` backbones are on Drive since the `02` run of 2026-10-09, their five partitions and community JSON since the `01` run of the same evening (§9.3).

## 9. Impact of the pruning-direction error across the study

*Added 2026-10-07 after the direction question was settled.*

### 9.1 What exactly is wrong, and what is not

- **The graph is right.** `Full_Network.gml` has edges retweeter → retweeted, as documented. In-strength is retweets received, out-strength is retweets made (§4.1).
- **The pruning rule is the wrong side of the edge.** `prune_by_out_strength_threshold(threshold=1.0)` keeps authors who *made* ≥ 1 retweet. The intended population was authors who *were retweeted* ≥ 1 times (in-strength). The out-strength rule deletes every retweeted-only author outright (§5).
- **Everything downstream of `Final_OutThreshold1.gml` is therefore a study of retweeters**, 1,984,599 of them, not of the amplified accounts. That file is the sole input of the five community partitions, the AMI/ARI comparison, the community JSON export, and the social-media maps.
- **Nothing in the tweet-level analyses depends on it** (§5.2). Sentiment, emotion and tweet-topic results are computed on every tweet of the corpus.

### 9.2 Stage-by-stage impact

| Stage | Notebook / artefact | Reads the pruned network? | Verdict | What a fix requires |
|---|---|---|---|---|
| Corpora, author counts, funnel | `02/01`, `02/02` cells 12–20 | no | **unaffected** | nothing |
| Retweet graph, topology, top-10 lists | `02/02` cells 44–55 | no (it produces the input) | **unaffected** — direction is correct | nothing |
| Pruning strategies 1 and 2 (`LWCC.gml`, `90TS_LWCC.gml`) | `04/01` cells 13, 16 | no | **direction-neutral** (LWCC ignores direction; total strength = in + out) | nothing — strategy 1 is since 2026-10-09 also written by `02` as `Full_LWCC_{Influence,InfoFlow}.gml` so that `01` can cluster it |
| Pruning strategy 3 (`Final_OutThreshold1.gml`) | `04/01` cell 18 | — | **wrong population** (retweeters) | replace with an in-strength rule (`02` notebook) or a direction-neutral backbone (§9.3) — **done**: `Full_RetweetedOnce_*.gml`, 202,710 authors |
| Community detection, five methods; modularity / codelength; AMI-ARI table | `04/01` cells 21–25 | yes | **valid only as a partition of retweeters**; Infomap additionally walks against information flow on this edge orientation | re-run `run_modularity_workflow` on the new backbone (≈ 10 min per method on 2M nodes); Infomap on the flow (transposed) file — **done** in `01` (§9.3 results table); **2026-10-09**: `01` re-runs all five methods on the information-flow file of both backbones — **done** (§9.3, second results block) |
| Community JSON export (`Final_OutThreshold1_author_communities.json`) | `04/01` cell 27 | yes | same | **done**: `Full_RetweetedOnce_author_communities.json` and `Full_LWCC_author_communities.json` (`01`, 2026-10-09) |
| Social-media maps (ForceAtlas2 classic / linlog, DrL), degree CCDF, community-blocked adjacency, `positions_*.parquet`, `network_with_layout.graphml` | `04/02` (`NETWORK = 'Final_leiden_fast'`; stored output: 1,984,599 nodes, 1,458 communities, top 16 cover 87 %) | yes | **maps of the retweeter backbone** | re-render (layout ≈ 15 min GPU budget per recipe) — the `04/02` run of 2026-10-08 **failed before layout** (cu13 cuGraph install; patched 2026-10-09, §9.5); pending for `RetweetedOnce_leiden_fast` and `LWCC_leiden_fast` |
| Sentiment v1 / v4, per tweet | `03/01`, `03/01b` | no | **unaffected** | nothing — but any *per-community* aggregate of these scores inherits the partition |
| Tweet-level LDA, topics in time, top-K per topic | `03/03`, `02/03` (deleted) | no | unaffected by direction; stored results on the first-generation corpus (§6.3) | rewired 2026-10-10; re-run |
| Author-level LDA v1 / v2 | `03/04` (deleted 2026-10-10), `03/04c` | no — reads the LWCC backbone (`LWCC.gml` until 2026-10-10, now `3_backbones/Full_LWCC_InfoFlow.gml`, same graph) | unaffected by direction; broken by the GML `id`/`label` bug (§6.4) | fix loader, re-run |
| Topic–emotion mixing plan | `docs/TOPIC_EMOTION_MIXING_PLAN.md` | no — consumes `04c` θ | inherits §6.4 | after `04c` is re-run |
| Classifiers | `05_Classifiers` | no | **unaffected** | nothing |
| Blog analysis: 198,326 matched authors, 21 Leiden-directed communities, figures 1–11 | branch `origin/add-blog-figures-and-community-analysis` (commit `da01e80`, 2026-08-21, one commit ahead of `main`, unmerged) | **unknown** — inputs not committed | see §9.4 | verify provenance first |

What survives untouched: every number in §§1–4 of this document, the two corpora, all per-tweet labels, and the LWCC / total-strength backbones.

### 9.3 Is degree pruning worth doing at all?

The concern is legitimate: an in-strength ≥ 1 *subgraph* keeps at most the 374,368 authors who were ever retweeted (§4.1) and only the edges *among* them (a retweeted author retweeting another retweeted author). Pure retweeters, who carry most of the 9.6M retweets, disappear with their edges, so the surviving weight will be a small fraction of the original and the graph will fragment before the LWCC step. The `02` notebook prints the node count and weight kept after each step so this can be read off directly.

The options, with their populations:

| Backbone | Rule | Nodes | Direction-sensitive? | Comment |
|---|---|---:|---|---|
| A. Full LWCC (`LWCC.gml`) | self-loops removed, giant component | 3,264,499 | no | since 2026-10-09 `02` writes it in both orientations as `Full_LWCC_{Influence,InfoFlow}.gml` and `01` clusters it (run 2026-10-09); the legacy `LWCC.gml` was deleted in the 2026-10-10 reorganisation |
| B. Total-strength 90 % (`90TS_LWCC.gml`) | drop lowest in+out nodes until 10 % of weight is lost, then LWCC | 2,315,573 | no | written by `01` only when `RUN_STRATEGY_2` is on; keeps both heavy retweeters and heavily retweeted |
| C. Out-strength ≥ 1 (`Final_OutThreshold1.gml`) | made ≥ 1 retweet | 1,984,599 | yes — retweeters | current; wrong population |
| D. In-strength ≥ 1 (`02`: `*_RetweetedOnce_*`) | received ≥ 1 retweet | ≤ 374,368 before LWCC | yes — retweeted | intended population, but a sparse subgraph |

Two things are being conflated by any degree-threshold *subgraph*: the graph on which communities are detected, and the author population reported on. They need not be the same set. Communities can be detected on a direction-neutral backbone (A or B), where every retweeter still contributes structure, and the "retweeted at least once" criterion can then be applied as an **author filter on the community table** (keep rows whose in-strength ≥ 1 in the full graph) rather than as a graph cut. That keeps the retweeters' edges in the clustering while reporting only on amplified authors, and the choice of population becomes a one-line filter that can be changed without re-running Leiden. The in-strength of every author is available from `Full_Network.gml` (`G.in_degree(weight='weight')`).

**The `02` numbers (Colab run completed 2026-10-07, stored in the notebook and in `Cleaned Data/full_dual_network_stats.json`):**

| Step (influence network) | Nodes | Edges | Weight |
|---|---:|---:|---:|
| Full network | 3,379,040 | 7,768,720 | 9,638,407 |
| Self-loops removed | 3,379,040 | 7,736,504 | 9,606,191 (32,216 self-retweets) |
| In-strength ≥ 1 (retweeted ≥ 1× by someone else) | 363,618 | 890,006 | — |
| Largest weakly connected component | **202,710** | 882,530 | 1,269,747 (**13.17 %** of original) |

Two readings of that table. First, 374,368 authors appear as retweeted in the dictionary but only 363,618 pass the threshold: the other 10,750 were retweeted *only by themselves*. Second, the component step costs 161,000 authors but almost no edges, so the retweeted-once authors outside the giant component are overwhelmingly isolates — authors whose only retweeters were pure retweeters, now deleted. The backbone keeps 13 % of the retweet volume: the 87 % lost is retweets *by* authors who were never themselves retweeted.

**The LWCC backbone (Colab run completed 2026-10-09, stored in the notebook and in `full_dual_network_stats.json` under `lwcc`):** 3,264,499 nodes, 7,670,516 edges, weight 9,458,703 — 96.61 % of the authors and 98.14 % of the retweet volume — after removing the 25,135 self-retweets that sit inside the giant component (the other 7,081 of the 32,216 self-loops are in the small components); node set and edge count identical in both orientations (asserted). Exactly the strategy-1 graph of the pre-2026-10 `04/01`. Timing on a high-RAM CPU runtime: 2 min 18 s to build both full graphs, 22 min 49 s to export them, 29 min 18 s for the LWCC cell (two GML write/read-back passes of ~12 min each), 5 min 17 s for the retweeted-once backbone; the retweeted-once numbers reproduce the 2026-10-07 run exactly.

**Re-run into the stage folders (Colab, finished 2026-10-10 04:37 EDT; stored in the notebook, commit `run(02)` of 2026-10-10):** no errors, cells executed 1–43 in order, source identical to the committed notebook. `generate_data = True` rebuilt `full_network_dict.pkl` (13 min 35 s; 17,410,035 tweets, 0 exceptions; 4,061,626 original / 9,638,407 retweeted / 3,221,048 replied / 488,954 quoted) and every downstream number reproduced exactly: full graph 3,379,040 authors / 7,768,720 edges / weight 9,638,407; LWCC 3,264,499 / 7,670,516 / 9,458,703 after removing 25,135 self-loops; retweeted-once 363,618 → 202,710 authors / 882,530 edges / 1,269,747 (13.17 %). Every GML export passed its read-back check. Timing: 17 min 34 s to export both full graphs, 20 min 59 s for the LWCC cell, 2 min 44 s + 1 min 27 s for the retweeted-once cut and export. The Drive listing shows `1_retweet_dicts/full_network_dict.pkl`, `2_full_graphs/Full_Network_{Influence,InfoFlow}.gml`, `3_backbones/Full_{LWCC,RetweetedOnce}_{Influence,InfoFlow}.gml` and the seven files of `test/`.

A coincidence, now explained (§9.4): 202,710 is within 2.2 % of the blog analysis's **198,326 matched authors**, but the draft states the matched set was cut from the 1,984,599-node out-strength backbone, i.e. it is retweeters with topic data, not retweeted authors. The similar size is accidental.

Recommendation: D is small but connected (one component, 882,530 edges), which is enough to carry community detection; it is also exactly the intended population. Run `01` on it. If community structure on 200k authors turns out too coarse for the research question, fall back to A or B for detection and apply "retweeted ≥ 1" as the downstream author filter, as described above.

**2026-10-09:** the fallback to A is wired in rather than hypothetical. `01` runs the five methods on both D and A in one pass (`BACKBONE_STEMS = ['Full_RetweetedOnce', 'Full_LWCC']`), so the two partitions can be compared side by side before the population decision of §11.4 is taken. Expected cost: each method on the 3.26M-node LWCC is 10–20 min, roughly 1.5–2 h for the five on both backbones; the AMI cell holds five 3.26M-entry membership dicts (~3 GB).

**`01` results (Colab run completed 2026-10-07, CPU runtime, no errors; superseded by the 2026-10-09 run below for the four methods that then ran on the influence file, the Infomap row still current):**

Inside the backbone, 132,603 of the 202,710 authors still receive ≥ 1 retweet and 143,043 make ≥ 1; the other 70,107 receivers lost all their retweeters to the cut and remain only because they retweet other backbone authors. Degree Gini 0.74; the out-strength tail fits a power law with α = 2.50 (x_min = 59) but is not distinguishable from a lognormal (p = 0.59).

| Method | Input file | Communities | Modularity | Largest five |
|---|---|---:|---:|---|
| label_propagation | `…_Influence.gml` (undirected collapse: 864,436 edges) | 4,363 | 0.682 | 91,832 · 53,195 · 8,122 · 4,149 · 2,977 |
| louvain | same | 1,116 | 0.740 | 53,222 · 47,368 · 19,587 · 12,677 · 9,775 |
| leiden_fast | same | 1,089 | 0.747 | 46,403 · 44,165 · 22,561 · 19,687 · 12,847 |
| leiden_directed | `…_Influence.gml` (directed) | 1,077 | 0.744 | 47,906 · 46,415 · 18,555 · 18,469 · 13,446 |
| infomap | `…_InfoFlow.gml` (directed, flow orientation) | 17,055 | 0.417 (codelength 7.947 bits) | 2,790 · 2,352 · 844 · 838 · 771 |

Agreement (aligned by author id, all 202,710 present in every partition): leiden_fast ↔ leiden_directed AMI 0.804 / ARI 0.846; louvain ↔ either Leiden AMI ≈ 0.75; label propagation ↔ modularity methods AMI ≈ 0.57; Infomap ↔ everything AMI ≈ 0.31 and ARI < 0.01. Infomap's 17,055 small modules on the flow orientation are a different kind of partition (random-walk traps on a sparse directed graph), not a noisier version of the modularity ones — the same pattern as on the old backbone, where it gave 36,122 communities and ARI 0.08. The two Leiden variants are the stable choice; the two largest Leiden-directed communities (47,906 and 46,415) happen to be close in size to the two largest validated communities of the blog analysis (44,718 and 41,574), but those were computed on the retweeter backbone (§9.4), so the resemblance says nothing about their membership.

Export: `Full_RetweetedOnce_author_communities.json`, 202,710 authors × 5 methods.

**`01` results on the information-flow files of both backbones (Colab run completed 2026-10-09, 18:42 CEST, CPU runtime, no errors; outputs stored in the notebook):**

Backbone checks passed on both files (one weak component, unique author labels, directed, no self-loops). `Full_RetweetedOnce_InfoFlow.gml`: 202,710 authors, 882,530 edges, weight 1,269,747, 132,603 / 143,043 authors retweeted / retweeting inside the backbone — identical to 2026-10-07. `Full_LWCC_InfoFlow.gml`: 3,264,499 authors, 7,670,516 edges, weight 9,458,703; only 317,533 authors are retweeted inside it and 3,095,219 retweet, so 90 % of its authors have out-strength 0. `plot_report` now reads the flow orientation, where out-degree is distinct retweeters and out-strength is retweets received; its numbers therefore differ from the 2026-10-07 ones, which described retweets *made*: retweeted-once out-degree Gini 0.849, out-strength tail α = 1.87 (x_min = 4); LWCC α = 1.84 (x_min = 39, Gini skipped for size); a lognormal fits better than a power law on both (p < 0.001).

| Method (input: `<stem>_InfoFlow.gml`) | Retweeted-once: communities | Q | Largest five | LWCC: communities | Q | Largest five |
|---|---:|---:|---|---:|---:|---|
| label_propagation (undirected collapse) | 4,364 | 0.683 | 91,490 · 53,187 · 7,704 · 4,267 · 3,007 | 36,500 | 0.670 | 842,069 · 772,998 · 38,155 · 35,155 · 33,631 |
| louvain (same) | 974 | 0.743 | 47,789 · 47,062 · 19,348 · 17,650 · 12,795 | 2,299 | 0.763 | 619,251 · 400,258 · 303,337 · 278,129 · 180,860 |
| leiden_fast (same) | 1,110 | 0.747 | 46,569 · 44,029 · 22,665 · 20,280 · 12,742 | 3,041 | 0.774 | 608,582 · 371,580 · 311,703 · 279,936 · 176,298 |
| leiden_directed (directed) | 1,057 | 0.744 | 47,728 · 47,247 · 18,194 · 17,101 · 13,952 | 2,951 | 0.772 | 619,125 · 386,553 · 327,295 · 269,842 · 188,686 |
| infomap (directed) | 17,055 * | 0.417 * | 2,790 · 2,352 · 844 · 838 · 771 * | 136,744 | 0.513 (codelength 13.530 bits) | 36,398 · 31,596 · 30,223 · 28,873 · 27,060 |

\* Not recomputed: `Full_RetweetedOnce_InfoFlow_infomap.gml` already existed from the 2026-10-07 run, which computed it on this same file (same author set, edge count and weight), so `SKIP_EXISTING_COMMUNITY_FILES` reused it; the values are those of the 2026-10-07 table above.

Agreement (aligned by author id; every author present in every partition on both backbones):

| Pair | Retweeted-once AMI / ARI | LWCC AMI / ARI |
|---|---|---|
| leiden_fast ↔ leiden_directed | 0.792 / 0.824 | 0.844 / 0.884 |
| louvain ↔ leiden_fast | 0.798 / 0.824 | 0.824 / 0.864 |
| louvain ↔ leiden_directed | 0.774 / 0.832 | 0.798 / 0.831 |
| label_propagation ↔ modularity methods | 0.565–0.572 / 0.46–0.48 | 0.546–0.558 / 0.46–0.48 |
| infomap ↔ everything | 0.260–0.312 / < 0.01 | 0.423–0.524 / 0.011–0.024 |

The 2026-10-07 pattern holds on both backbones: the three modularity methods agree closely, label propagation lands in the middle, and Infomap is a different kind of partition. On the LWCC, label propagation puts 1.6M of the 3.26M authors into two communities, and Infomap leaves singleton modules. Run time: 2 min 43 s for the backbone checks, 36 min for the undirected methods, 21 min for the directed ones, **51 min for the AMI/ARI cell** (the Infomap pairs on the LWCC dominate) and 47 s for the export — about 1 h 53 min in all.

Export: `Full_RetweetedOnce_author_communities.json` (202,710 authors × 5 methods) and `Full_LWCC_author_communities.json` (3,264,499 authors × 5 methods).

### 9.4 The 198,326 matched authors and the two community figures

The number you remembered as "198,326" is the **matched-author set of the blog analysis**, which is not on `main`. It is in commit `da01e80` ("Add validated community analysis and blog figures", 2026-08-21) on the unmerged remote branch `origin/add-blog-figures-and-community-analysis`, one commit ahead of `main`. To inspect it without switching branches: `git show origin/add-blog-figures-and-community-analysis:docs/blog_analysis/README.md`.

**What the branch contains** (80 files, no notebooks):

- `docs/blog_analysis/README.md` — scope, the 21 validated community labels, figure table, reproduction commands;
- `src/blog_analysis/` — `render_non_temporal_figures.py` (figures 4, 6–11), `render_temporal_figures.py` (2, 3, 5), `weekly_aggregation.py`, `community_tfidf.py`, `label_matrices.py`, `validate_community_labels.py`, `validate_outputs.py`, styles;
- `outputs/blog_figures/{light,dark}/{png,svg}/01–11_*.{png,svg}` and one compact CSV per figure under `outputs/blog_figures/data/`;
- `outputs/community_analysis/` — `community_labels.csv` (21 rows), c-TF-IDF terms, label revision history, de-identified audit diagnostics, and eight labelled matrices (`topic_enrichment_labelled.csv`, `sentiment_means_labelled.csv`, …).

**What it does not contain**, by design (README, "Reproducibility"): the inputs. `data_sets/blog_analysis/matched_authors.json` (198,326 records with `leiden_directed`, `dominant_topic`, `topic_0…topic_11`, `positive/neutral/negative`, eleven emotion means), `matched_author_ids.json`, the unlabelled `matrices/`, the HPC `tables/`, the frozen gensim K = 12 model and dictionary, and the UMAP projection. Both renderers hard-fail unless exactly 198,326 authors are loaded. The scripts explicitly "reuse completed analyses" and do not rerun Leiden, LDA, sentiment or layout. The excluded visualiser codebase (`davidfreeborn.github.io/twitter-authors-map`) is where `nodes.json` / `matched_nodes.json` originate (`validate_outputs.py` lists those names as forbidden in the package).

**Provenance of the matched set — unverified.** Nothing in either branch states how the 198,326 were selected or on which graph their `leiden_directed` labels were computed. Two facts constrain it:

- 198,326 / 1,984,599 = **9.99 %** of the out-strength backbone, and the 21 displayed communities cover 95.2 % of the matched set — consistent with "nodes of `Final_OutThreshold1` that also have a K = 12 topic profile", i.e. retweeters with enough original text;
- but the community sizes (44,718; 41,574; 17,422; …) are not proportional subsets of the `04/01` Leiden-directed sizes (455,672; 254,068; 203,605; …), and the topic model is a gensim K = 12 model that no notebook in this repository produces, so the Leiden run may equally have been a separate one on the matched subgraph.

**Resolved 2026-10-08 by the blog draft itself** (`docs/Public Perceptions of AI and Art.pdf`, section *Communities*): "The full network contained 1,984,599 authors and 4,472,376 connections; the combined analysis used the 198,326 authors with matching topic, sentiment and emotion data." Those are the node and edge counts of `Final_OutThreshold1.gml` (§5), so the matched set is the subset of the **out-strength (retweeter) backbone** that has topic, sentiment and emotion records, and its `leiden_directed` labels are the `04/01` partition restricted to that subset (946 of the 1,417 communities survive the restriction). The paragraph below is kept as the reasoning that preceded this confirmation.

Until `matched_authors.json` is traced to its generating code, the honest statement is: **if** the matched set or its communities descend from `Final_OutThreshold1`, figures 7–11 (and the author-balanced weekly figures 2, 3, 5, which restrict to the matched ids) are figures about retweeters and inherit §9.1; if the communities were recomputed on a graph built another way, the direction question has to be answered for that graph. A direct test: join `matched_author_ids.json` against the node labels of `Final_OutThreshold1.gml` — a near-100 % hit rate settles the first clause.

**Figure 7 — "Topic enrichment by community, log₂ ratios relative to all matched authors".** `render_non_temporal_figures.py::community_topic_enrichment_data` reads the *precomputed* `data_sets/blog_analysis/matrices/topic_enrichment.csv` (rows `C<id>`, 12 topic columns), merges the display labels, melts it to `data/07_community_topic_enrichment.csv` (columns `community_id, display_label, n_authors, topic_label, log2_enrichment, topic_id`), and `plot_community_topic_enrichment` draws it with `imshow`, `RdBu_r`, `TwoSlopeNorm` centred at 0, colour-bar label "log2 enrichment relative to all matched authors". **The log₂ ratio itself is not computed anywhere in the repository**; the committed `topic_enrichment_labelled.csv` is the only copy of its values (e.g. Anti-AI-Art Discourse × Web3/DeFi = −4.88; AI/Crypto Launchpads × Web3/DeFi = +3.82). From its name and the README caveat it is log₂ of (mean topic weight in community ÷ mean topic weight over all 198,326 authors), but that is inferred, not read from code.

**Figure 9 — "Topic-weighted net sentiment by community".** Computed in-script by `topic_weighted_net_data` from `matched_authors.json`:

```python
matched["net_sentiment"] = matched["positive"] - matched["negative"]          # per author
community = matched[matched["leiden_directed"] == record.community_id]
weights   = community[f"topic_{topic_id}"]                                   # author's weight on the topic
membership  = weights.sum()                                                  # "author-equivalents"
effective_n = membership**2 / (weights**2).sum()                             # Kish effective n
weighted_net = np.average(net, weights=weights)
adequate = membership >= 10.0 and effective_n >= 30.0
```

Cells that fail the support rule are written to `data/09_….csv` with `plotted_weighted_net_sentiment = NaN` and rendered grey (`cmap.set_bad("#77777d")`) by `plot_topic_weighted_net`; the README reports 37 of 252 cells masked and a plotted range of −0.368 to +0.798. Both figures iterate over the 21 rows of `community_labels.csv`, so the community set is fixed by that file.

To regenerate either figure the branch needs the external inputs restored under `data_sets/blog_analysis/` and then `python src/blog_analysis/render_non_temporal_figures.py --input-root data_sets/blog_analysis --labels outputs/community_analysis/community_labels.csv --output-root <dir>`. Figure 7 cannot be regenerated from scratch with the committed code because its input matrix is upstream of everything in the repository.

### 9.5 The network visualisation notebooks

Three places:

- The pre-2026-10 `04_Network_Analysis/02_network_visualization.ipynb` (deleted 2026-10-09; the rewrite now carries the name): loaded one of the `Final_OutThreshold1*.gml` files (configured `NETWORK = 'Final_leiden_fast'`), undirects, keeps the giant component (1,984,599 nodes), colours the 16 largest precomputed communities, lays out with GPU ForceAtlas2 (`classic`, `linlog`) and, since 2026-08-12, igraph DrL via `src/network/network_utils.py::compute_drl_layout`; writes `social_map_<recipe>_<light|dark>.png`, `positions_<recipe>.parquet`, `05_degree_ccdf.png`, `06_adjacency_blocks.png`, optionally `network_with_layout.graphml`, into `Networks/viz_outputs_Final_leiden_fast/`. Every rendered map is therefore a map of the retweeter backbone (§9.2).
- [04_Network_Analysis/02_network_visualization.ipynb](../notebooks/04_Network_Analysis/02_network_visualization.ipynb) (named `02b_network_visualization_retweeted_once.ipynb` until 2026-10-09): the same pipeline with a registry of the `02` / `01` files — seven `RetweetedOnce_*` keys and, since 2026-10-09, seven `LWCC_*` keys; one network per run, output in `Networks/viz_outputs_<NETWORK>/`. Its 2026-10-08 run (`RetweetedOnce_leiden_fast`, A100) loaded and preprocessed the 202,710-node graph (undirected: 864,436 edges, one component, 1,089 Leiden-fast communities, top 16 = 91 % of nodes) and then failed at `import cudf`: the cuGraph install cell probed only `*-cu12` packages, and Colab GPU runtimes now ship RAPIDS cu13, so an unpinned cu12 wheel overwrote the preinstalled stack. Both visualisation notebooks now detect the CUDA suffix first and refuse an unpinned install (2026-10-09); the failed outputs were stripped from the committed copy. Still open: the trailing DrL cell imports `src.network.network_utils` although the notebook has no clone cell, so it cannot run on Colab.
- The interactive DrL map (`davidfreeborn.github.io/twitter-authors-map`), maintained outside this repository; the blog branch's `nodes.json` naming refers to it.

## 10. The pipeline graph — machine-derived double check

*Added 2026-10-07.*

[src/scripts/pipeline_graph.py](../src/scripts/pipeline_graph.py) derives a bipartite producer/consumer graph from the notebooks themselves: it parses every `.ipynb` with Python's AST, resolves the path variables of each setup cell, and records which Drive files each notebook opens for reading and writing. It writes [docs/pipeline_graph.json](pipeline_graph.json) (queryable), [docs/pipeline_graph.png](pipeline_graph.png) (notebooks ↔ artifacts) and [docs/pipeline_graph_notebooks.png](pipeline_graph_notebooks.png) (notebook-only DAG). It is the independent check on everything this document says by hand.

```bash
python3 src/scripts/pipeline_graph.py              # rebuild the three files in docs/ from the live notebooks
python3 src/scripts/pipeline_graph.py validate     # compare the provenance tags in README.md / notebook_setup.md with the parser
python3 src/scripts/pipeline_graph.py upstream   04_Network_Analysis/01_network_analysis
python3 src/scripts/pipeline_graph.py downstream 02_Processing/02_sanity_check_and_network_generation
```

### 10.1 State on 2026-10-07

- The committed graph dated from 2026-05-18 and covered 19 notebooks. Rebuilt: **28 notebooks, 159 artifacts, 147 write edges, 125 read edges**; `validate` reports **no drift** between the two Drive directory listings and the graph.
- `validate` initially flagged two **wrong provenance tags** in `notebooks/notebook_setup.md` / `README.md`, now corrected: `top_test_ai_tweets.csv` and `top_test_art_tweets.csv` are written by `03_Analysis_and_Modeling/02`, not `02_Processing/02`; `Full_Network.gml` is written by `02_Processing/02`, not `04_Network_Analysis/01` (which only reads it).
- The three corrected notebooks (§9, "Corrected notebooks") are registered in [docs/pipeline_overrides.yaml](pipeline_overrides.yaml) as **alternatives** of their originals (dashed edges; both appeared in the graph until 2026-10-09, when all three corrected copies were renamed over their originals and the pairs were dropped), and the new network artifacts are listed in both directory listings with `[written by …/02b]` / `[…/01b]` tags.
- **Rebuilt 2026-10-09** after the LWCC extension: **28 notebooks, 169 artifacts, 155 write edges, 132 read edges**; 53 declared edges (20 added: the `02` LWCC writes for the Test and Full branches, the `01` reads of all four backbone files and its five LWCC community writes and read-backs, and the two community JSONs — `COMMUNITIES_JSON` is now a dict comprehension over `BACKBONE_STEMS`, which the parser cannot resolve, so `Full_RetweetedOnce_author_communities.json` moved from parsed to declared). `validate` clean.
- **Rebuilt 2026-10-09 (clean-up)** after the deletions and renames of `02_Processing/02`, `02_Processing/03`, `04_Network_Analysis/01` and `04_Network_Analysis/02`: **24 notebooks, 146 artifacts, 114 write edges, 101 read edges**; 45 declared edges, 2 alternative edges. The deleted notebooks' edges are gone, so `Full_Network.gml`, `LWCC.gml`, every `Final_OutThreshold1*` file, `viz_outputs_Final_leiden_fast/` and `02/03`'s outputs no longer appear in the graph; they remain on Drive as legacy files (status block, *Reading conventions*). `validate` clean.
- **Rebuilt 2026-10-10** after the stage-folder reorganisation of `Data Sets/Networks/` (§12.7): **24 notebooks, 146 artifacts, 114 write edges, 99 read edges**. The 37 declared network paths in `pipeline_overrides.yaml` were remapped to the stage folders; the parser resolves the new `<stage>_folder / 'name'` paths of `02`, `04/01` and `04/02` on its own, and `03/04c` now reads `3_backbones/Full_LWCC_InfoFlow.gml` on the HPC path.
- **Rebuilt 2026-10-10 (03 clean-up, B1)** after deleting `03/01_sentiment_analysis_v2` and `03/04_lda_author_topics`: **22 notebooks, 133 artifacts, 110 write edges, 88 read edges**. The 13 artifacts only those two touched left the graph (the first-generation `AItrust_pruned_twits{,_test}.json`, the sentiment-v2 outputs, the v1 author-LDA grid/models under `LDA/author_level/`, the Colab-local sentiment copy) and their §12 rows were removed. `validate` clean.
- **Rebuilt 2026-10-10 (B2)** after renaming `01b` and rewriting its Section 8: **22 notebooks, 133 artifacts, 110 write edges, 88 read edges**. The full-output and block paths are now built inside `output_path_for()` / `block_dir_for()`, so six declared edges were added to `pipeline_overrides.yaml` (writes and read-backs of `{ai,art}_full_classified_{alias}.json` and `Blocks/{ds_key}__{alias}__{num_blocks}blocks/block_{bid}.json`). `validate` clean.
- **Rebuilt 2026-10-10 (B4)** after rewiring `03/03`: **22 notebooks, 131 artifacts, 110 write edges, 88 read edges**. With `DATA_DIR` / `MODELS_DIR` now derived from the Setup-cell folders, the parser resolves `03`'s real reads (the sentiment file, its own enriched output and grid CSV) and writes; the phantom `.jsonl` input, the `authorlite_*` experiment files, the block/stem templates and the declared `.jsonl.gz` write (merge-recovery section, removed) left the graph. `validate` clean after dropping the `.gz` line from the folder trees.
- **Rebuilt 2026-10-10 (B5)** after splitting `03/05`: **23 notebooks, 129 artifacts, 110 write edges, 88 read edges**. `05`'s two unresolvable reads (`Cleaned DataAItrust_pruned_twits{,_test}.json`) became reads of the current corpus files; `05b` has no edges. `validate` clean.

### 10.2 What the parser cannot see — and how it is covered

The parser only resolves literal paths and simple `folder / 'name'` expressions whose variables are bound in the same notebook. It is blind to:

1. paths assembled **inside a user-defined function** (the `export_network()` helper of `02`, the `block_folder / f'…_{block_id}.json'` merges of `03/01`);
2. files written by **`src/` workflows** (`run_modularity_workflow` writes `<input stem>_<method>.gml`), so every community-annotated GML — old and new — was invisible as a *write*;
3. **registry dictionaries** resolved at run time (`NETWORKS[NETWORK]` in the visualisation notebooks), so those notebooks showed only the demo network as input;
4. a path held in a variable assigned in one cell and opened in another, and f-strings with run-time values (`top_retweets_by_topic_{K}.csv`).

Rather than widen the parser, the script now accepts **hand-declared edges** (`manual_edges:` in the overrides file; each entry names the notebook, the Drive path, the direction and a note saying which cell produces it). 33 such edges were added, every one checked against the notebook source quoted earlier in this document. They carry `manual: true` in the JSON, so a reader can always tell a parsed edge from a declared one:

```bash
jq '[.edges[] | select(.manual)] | length' docs/pipeline_graph.json
```

Notebook ids may carry a letter suffix in provenance tags (the former `02b`, `01b`), so corrected copies were validated like any other notebook while they coexisted with their originals.

### 10.3 Hand-off table for the network stage

Each row was checked twice: against the notebook source (§§4–5, §9) and against the rebuilt graph (`upstream` / `downstream` queries). Rows marked *legacy* describe files that the deleted originals wrote (flat in `Networks/` until the 2026-10-10 reorganisation, which deleted them); they are kept so the stored numbers in §§5 and 9 stay traceable. Current files carry their stage folder (§12.7).

| Producer | File (in `Data Sets/Networks/`) | Consumer(s) | Edge source |
|---|---|---|---|
| `02/02` cell 44 (same cell in today's `02`, gated by `generate_data`) | `1_retweet_dicts/full_network_dict.pkl` | `02` (network build) | parsed |
| `02/02` cell 53 (deleted original) | `Full_Network.gml` — **legacy** | read only by the deleted `04/01`; `Full_Network_Influence.gml` is the same graph | no edge since 2026-10-09 |
| `02` | `2_full_graphs/Full_Network_Influence.gml` (reference), `2_full_graphs/Full_Network_InfoFlow.gml` | `01` (optional inspection and strategy 2, both on the InfoFlow file) | declared |
| `02` | `3_backbones/Full_RetweetedOnce_Influence.gml` | *(reference orientation; read by nothing since 2026-10-09)* | declared |
| `02` | `3_backbones/Full_RetweetedOnce_InfoFlow.gml` | `01` (backbone check; all five methods), `04/02` registry | declared |
| `01` | `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_{label_propagation,louvain,leiden_fast,leiden_directed,infomap}.gml` (the 2026-10-07 `…_Influence_<method>.gml` files are legacy) | `01` (AMI/ARI, JSON export), `04/02` | declared |
| `01` | `4_communities/RetweetedOnce/Full_RetweetedOnce_author_communities.json` | *(no consumer yet — the author set for downstream restriction)* | declared (parsed until 2026-10-09) |
| `02` | `3_backbones/Full_LWCC_Influence.gml` (reference), `3_backbones/Full_LWCC_InfoFlow.gml` | `01` (backbone check; all five methods on the InfoFlow file), `04/02` registry; since 2026-10-10 also `03/04c` (author-level LDA, HPC copy) | declared (`03/04c`: parsed) |
| `01` | `4_communities/LWCC/Full_LWCC_InfoFlow_{label_propagation,louvain,leiden_fast,leiden_directed,infomap}.gml` | `01` (AMI/ARI, JSON export), `04/02` | declared |
| `01` | `4_communities/LWCC/Full_LWCC_author_communities.json` | *(no consumer yet)* | declared |
| `04/01` cell 18 (deleted original) | `Final_OutThreshold1.gml` — **legacy** | the deleted `04/01` community cells | no edge since 2026-10-09 |
| `04/01` cells 21, 23 (deleted original) | `Final_OutThreshold1_<method>.gml` — **legacy** | the deleted `04/01` AMI + export and the deleted `04/02`; no longer in today's `04/02` registry (InfoFlow keys only since 2026-10-09) | no edge since 2026-10-09 |
| `04/01` cell 27 (deleted original) | `Final_OutThreshold1_author_communities.json` — **legacy** | *(none in repo; §9.4 for the blog branch)* | no edge since 2026-10-09 |

### 10.4 Caveats when reading the graph

- **Alternative edges are bidirectional**, and the `upstream` / `downstream` walkers follow them, so a query on `01` also lists `04/01` and its outputs (and vice versa). Read the notebook list of a query with the alternatives in mind, or inspect the JSON and drop `direction == "alternative"` edges.
- The `BASE_PATH divergence` warning is expected: `03/01b` and `03/04c` run on the HPC path, every other notebook on Drive. The parser strips both prefixes, so artifacts still line up by relative path.
- The parser records what the *code* can write, not what *exists* on Drive. A file can be in the graph and absent on disk (e.g. before `02` finishes), or on disk and absent from the graph (first-generation files such as `AItrust_pruned_twits.json`, §6.3, whose producing cells no longer exist).

### 10.5 Procedure after any notebook change

1. `python3 src/scripts/pipeline_graph.py` — rebuild; commit the three `docs/pipeline_graph*` files with the notebook.
2. `python3 src/scripts/pipeline_graph.py validate` — must print `no drift detected`; if it names a file, either the directory listing or a `manual_edges` entry is stale.
3. If the notebook writes a file through a helper, a `src/` function or a registry, add a `manual_edges` entry with a note naming the cell; do not leave the edge implicit.
4. `python3 notebooks/analyze_notebooks.py` — `colab_ok` and `folded` must be clean for the touched notebook.
5. If the change adds or renames a file, add it to the inventory in §12 (one row: location, name, writer, readers, description).

## 11. Audit of the blog draft against the corrected pipeline

*Added 2026-10-08. Draft: `docs/Public Perceptions of AI and Art.pdf` (16 pages, "Public Perceptions of AI and Art: Mapping a Cultural Shockwave").*

### 11.1 Where the draft's numbers come from

| Draft section | Population behind the numbers | Depends on the network? |
|---|---|---|
| Introduction, timeline | — | no |
| "Finding the Human Signal in 36 Million Tweets" | full funnel and the two corpora (§2) | no |
| "Sentiment and Emotion Profile" (means for AI-and-art vs general AI; weekly sentiment and emotion) | the **198,326 matched authors** ("in the full 198,326-author analysis"); weekly means are author-balanced over matched ids | **yes** — matched set is cut from the out-strength backbone |
| "Topics" (K = 12 LDA, dominant-topic shares, UMAP, weekly topic prevalence, net sentiment by topic) | matched authors; fig. 6 reads `topic_sentiment_fuzzy_network_subset.csv` | **yes** |
| "Communities" | `Final_OutThreshold1_leiden_directed` restricted to matched authors: 946 communities, 21 largest = 95.2 % | **yes** — this is the retweeter partition (§9.1) |
| "Sentiment, Topic and Community" (omega-squared, community sentiment means, enrichment heatmap, topic-weighted net sentiment, within-community SD, the two-community contrast and its quoted posts) | matched authors × those communities | **yes** |
| "Exploring the Network" (interactive map: 198,326 authors, 853,133 connections, 187,095 in the largest component) | matched subgraph of the retweeter backbone | **yes** |
| "The Larger Picture" | restates the above | **yes** (every claim except the corpus-level sentiment contrast, which is *also* computed on matched authors in the draft) |

Only the dataset description is network-independent. Everything from "Sentiment and Emotion Profile" onward is computed on authors who *made* at least one retweet (and had enough text for a topic profile), which the draft presents as the public reaction.

### 11.2 Statements that need correcting

Independent of the backbone choice (factual against the stored pipeline):

1. **"Its connections record retweets, replies and quotations between authors."** The graph is built with `type_of_network = 'retweeted'` only (§3); replies and quotes create no edges. Say "retweets".
2. **"The full network contained 1,984,599 authors and 4,472,376 connections."** That is the pruned out-strength backbone. The full retweet network has 3,379,040 authors and 7,768,720 connections (§4.2).
3. **"17.4 million tweets from 4.7 million unique authors."** 4,775,711 rounds to 4.8 million.
4. **"We rigorously filtered out the noise … crypto bots."** The only bot/spam rule in the pipeline is the `#airdrop`/`@airdrop` denylist inside the AI-keyword test (§2.1); there is no bot detection. Soften to what was done.
5. **"the data we analyzed focuses on the intense 120-day window"** — fine, but the collection ended 27 February 12:00 UTC (blog README), so the last ISO week is incomplete; the draft already omits it from the plots.

Dependent on the backbone (will change when the matched set is rebuilt on `Full_RetweetedOnce_*`):

6. Every number in the "Sentiment and Emotion Profile" section (0.330 / 0.159, 0.194 / 0.341, 0.476 / 0.500; the emotion means; all weekly values and the mid-December peak).
7. Every number in "Topics" (120,665 / 60.8 %, 38,980 / 19.7 %, 57.3 %, 46.6 %, +0.719, +0.573, −0.143) and the UMAP.
8. Everything in "Communities": the 1,984,599 / 4,472,376 framing, 946 communities, 188,765 / 95.2 %, all 21 community sizes and their labels (the labels were derived from c-TF-IDF on *these* members; a new partition needs relabelling and re-validation).
9. Everything in "Sentiment, Topic and Community": 21.1 % / 32.6 % / 27.6 %, 0.769, +3.82, −0.368 / +0.798, 0.500 / 0.133, 0.277 / 0.614, 0.497 / 0.149, the median SD 0.103, and the quoted example posts (they were sampled from community members).
10. "Exploring the Network": 198,326 / 853,133 / 187,095 and the screenshot.
11. "The Larger Picture": the claim that community membership is associated more with sentiment than with topic, and the mid-December peak, are both matched-set results.

A secondary caveat for item 6 regardless of backbone: the AI+Art sentiment result file holds 474,239 surplus lines (§6.2). If the author-level means were built from it, some tweets are double-counted inside author averages; rebuild that file first.

### 11.3 What has to be re-run, in order

In this repository:

| Step | Notebook / script | Status |
|---|---|---|
| Both networks + retweeted-once backbone | `02_Processing/02` | done 2026-10-07 (as `02b`); re-run 2026-10-09 with identical numbers |
| Five partitions + community JSON on the backbone | `04_Network_Analysis/01` | done 2026-10-07 (as `01b`, influence orientation); re-run on the information-flow file 2026-10-09 (§9.3) |
| Social-media maps | `04_Network_Analysis/02` | run of 2026-10-08 failed before layout (cu13 cuGraph install); patched 2026-10-09; pending |
| LWCC backbone in both directions | `02_Processing/02` (extension of 2026-10-09) | done 2026-10-09 |
| Five partitions + community JSON on the LWCC backbone | `04_Network_Analysis/01` | done 2026-10-09 (§9.3) |
| Whole network stage again, into the stage folders of §12.7, after `Data Sets/Networks/` was emptied (2026-10-10) | `02_Processing/02` (`generate_data = True`) → `04_Network_Analysis/01` → `04_Network_Analysis/02` | `02` done 2026-10-10 (identical numbers, §9.3); `04/01` (needs the layout branch on `main`), `04/02` pending |
| Tweet-level topics on the current corpus | `03_Analysis_and_Modeling/03` (Colab) | rewired 2026-10-10 (§6.3); run open |
| Verified full sentiment/emotion outputs | `03/01b` 8a (CPU) → 8b (GPU) → 8c (legacy blocks) on the cluster | code fixed 2026-10-10 (§6.2.2); runs open |

Outside this repository (the blog branch's scripts consume files that are produced elsewhere — `data_sets/blog_analysis/matched_authors.json`, the K = 12 gensim model, the `matrices/` and `tables/` folders; see §9.4):

| Step | Where | Input that changes |
|---|---|---|
| Rebuild the matched-author set: authors of `Full_RetweetedOnce_Influence.gml` that have topic, sentiment and emotion records; carry `leiden_directed` from `Full_RetweetedOnce_author_communities.json` | the external matching code (not in repo) | backbone + partition |
| Refit or re-apply the K = 12 author-topic model on the new author documents (the blog's topic model is not `03/04c`; it is a gensim model kept under `data_sets/blog_analysis/models/`) | external | author set |
| Recompute the community matrices (`topic_enrichment`, `topic_composition`, `topic_capture`, `sentiment_means`, `emotion_means`, standardised variants, `omega_squared`) and `topic_sentiment_fuzzy_network_subset.csv` | external (HPC tables) | author set + partition |
| c-TF-IDF terms, label drafting and the 25-post audits | `src/blog_analysis/community_tfidf.py`, `validate_community_labels.py` (branch) | partition |
| Attach labels to matrices | `src/blog_analysis/label_matrices.py` | labels |
| Weekly author-balanced tables | `src/blog_analysis/weekly_aggregation.py` | matched ids |
| Figures 2–11 | `render_temporal_figures.py`, `render_non_temporal_figures.py` (both assert exactly 198,326 authors — that constant must be removed) | all of the above |
| Interactive DrL map | `davidfreeborn.github.io/twitter-authors-map` codebase | matched subgraph |

Figure 1 (timeline) needs nothing.

### 11.4 Two decisions the re-run forces

- **Which population the blog is about.** On the retweeted-once backbone the matched set will be at most 202,710 authors and in practice fewer (only those with enough text for a topic profile). That is a study of *amplified* authors. If the intended subject is "the public reaction", a direction-neutral backbone with "retweeted ≥ 1" as a reported filter (§9.3) may fit the prose better. Decide before rebuilding the matched set, because every downstream number depends on it. Since 2026-10-09 both backbones are produced and clustered in the same `02` / `01` run (§9.3), so the decision can be made on the two partitions side by side rather than before clustering.
- **Whether the community labels carry over.** They were validated for specific memberships. A new partition on a different author set needs the c-TF-IDF and audit steps rerun; reusing the 21 names would be a claim without evidence.

## 12. File inventory — every artifact of the pipeline

*Added 2026-10-09.* Every file the notebooks read or write, derived from [docs/pipeline_graph.json](pipeline_graph.json) (133 artifacts after the 2026-10-10 rebuild: parsed edges plus the declared ones of `pipeline_overrides.yaml`). Location is the folder, relative to `BASE_PATH` = `My Drive/Colab Projects/AI Public Trust/` unless marked HPC or Colab-local; *written by* / *read by* are notebook ids (`stage/index`, `b` suffixes in older documents denote the corrected copies that were renamed over their originals on 2026-10-09); templated names (`{…}`) are resolved at run time. Descriptions are from the notebooks and §§1–9 of this document. A file with no writer is produced outside the repository or by a cell that no longer exists; a file with no reader is a terminal output. Legacy files that no notebook reads or writes any more (`Test_Network.*`, `Full_Network.graphml/.gexf/.json` from the pre-2026-10 `02/02`; `LWCC.gml/.graphml`, `Final_OutThreshold1*` and `viz_outputs_Final_leiden_fast/` from the pre-2026-10 `04/01` and `04/02`; `Full_RetweetedOnce_Influence_<method>.gml` from the 2026-10-07 run on the influence orientation; `AItrust_pruned_twits_with_sentiment_cleaned.json` and `top_retweets_by_topic_*.csv` from the deleted `02/03`) still sit on Drive but are not listed. To regenerate the producer/consumer columns after a notebook change, follow §10.5 and re-derive this table from the JSON.

### 12.1 `Raw Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `testing.json` | — | `02/01` | One raw API page (a single test batch, 233,094 records with duplicates); source of every `_test` file. |
| `tweets_2023-02-07T17:00:00.json` | — | `02/01` | The same literal example referenced at the `Raw Data/` root in a `02/01` sanity cell (no such file is expected there). |
| `Twits/tweets_2023-02-07T17:00:00.json` | — | `02/01` | Raw API harvest files, one per 10-minute window (`tweets_<ISO timestamp>.json`); `02/01` iterates the whole folder — the parser only sees this literal example. |

### 12.2 `Data Sets/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `AItrust_author_dict.json` | `02/01` | `02/02` | Stage-1 author dictionary: one author record per line, with duplicates (31,989,322 lines; §1). |
| `AItrust_author_dict_test.json` | `02/01` | `02/01`, `02/02` | Test-branch counterpart of the author dictionary. |
| `AItrust_twits_dict.json` | `02/01` | `02/02` | Stage-1 tweet dictionary: line-delimited JSON, one tweet record per line, flattened from the raw pages, **with duplicates** (36,560,405 lines; §1). |
| `AItrust_twits_dict_test.json` | `02/01` | `02/01`, `02/02` | Test-branch counterpart of the tweet dictionary, built from `testing.json`. |

### 12.3 `Data Sets/Cleaned Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `AItrust_Art_pruned_twit_dict.json` | `02/02` | — | **AI+Art corpus**: the 3,583,101 AI-corpus tweets that also match one of the 60 art keywords (§2). |
| `AItrust_Art_pruned_twit_dict_test.json` | `02/02` | `02/02`, `03/02` | Test-branch AI+Art corpus (216 tweets). |
| `AItrust_pruned_twits_test_with_sentiment.json` | `03/01` | `03/01` | Sentiment v1 test output (older file name). |
| `AItrust_pruned_twits_with_sentiment.json` | `03/01` | `03/03` | AI corpus (17,410,035 tweets) with CardiffNLP sentiment (`sentiment_label`, `sentiment_score`; `03/01`, §6.1). The 2026 file since the first-generation one left Drive; input of the tweet-topic notebook. |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5.json` | `03/03` | `03/03` | Tweets with sentiment and K = 5 LDA topic fields (`lda_k5_topic_id`, `_dist`, `_label`); merged from `03/03`'s blocks and read back by its analysis section. Not on Drive until `03/03` is re-run. |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5_block{NNN}.json` | `03/03` | — | Per-block outputs of the enrichment (Section 2, step 10), deleted and rewritten at each run and merged into the file above. The parser renders the name from `OUTPUT_JSONL.stem` literally. |
| `AItrust_pruned_twits_with_sentiment__SAMPLE{SAMPLE_ENRICH_N}.json` | `03/03` | — | Optional sample of the input when `SAMPLE_ENRICH = True` (off by default); with it the outputs get a `__SAMPLE…` suffix. |
| `AItrust_topics_k5_metadata.json` | `03/03` | — | Top terms and settings of the K = 5 tweet-topic model. |
| `AItrust_twits_pruned_dict.json` | `02/02` | `02/02`, `03/01`, `03/05` | **AI corpus**: 17,410,035 tweets after dedup + AI keyword + English + date ≥ 2022-10-31, with `processed_text` added (§2). |
| `AItrust_twits_pruned_dict_test.json` | `02/02` | `02/02`, `03/01`, `03/02`, `03/05` | Test-branch AI corpus (881 tweets). |
| `AItrust_twits_pruned_dict_test_with_sentiment.json` | `03/01` | — | Sentiment v1 test output (current file name). |
| `dataset_statistics_summary.json` | `02/02` | — | Master summary of the dataset (timeframe, funnel, authors, typology, network topology); mirrored in the repository at `notebooks/02_Processing/dataset_statistics_summary.json`. |
| `full_author_corpus_dict.pkl` | `02/02` | `02/02` | author id → list of that author's raw tweet texts, 4,775,711 authors (input for author-level text models; §3). |
| `full_basic_counts_dict.pkl` | `02/02` | — | Tweet-type counts of the AI corpus (original / retweeted / replied_to / quoted; §3). |
| `full_dual_network_stats.json` | `02/02` | — | `02` statistics for the full network in both orientations (`raw`), the LWCC backbone (`lwcc`, since 2026-10-09) and the retweeted-once backbone (`pruned`), including the pruning funnel and top-N lists (§9.3). |
| `full_network_stats.json` | — | `02/02` | Topology statistics of the full retweet graph written by the pre-2026-10 `02/02` (nodes, edges, weight, components, top-10 lists; §4.2); read by the summary-report cell. |
| `full_pruning_stats.json` | `02/02` | — | Funnel counters of the full pruning pass (lines read, duplicates, drops per rule, kept counts, date range, tweet types; §2.2). |
| `full_timeline_dict.pkl` | `02/02` | `02/02` | tweet id → `created_at` for every AI-corpus tweet (the daily-volume timeline; §3). |
| `test_author_corpus_dict.pkl` | `02/02` | `02/02` | Test-branch author corpus dictionary. |
| `test_basic_counts_dict.pkl` | `02/02` | `02/02` | Test-branch tweet-type counts. |
| `test_dual_network_stats.json` | `02/02` | — | Same for the test branch. |
| `test_pruning_stats.json` | `02/02` | `02/02` | Same counters for the test branch. |
| `test_timeline_dict.pkl` | `02/02` | `02/02` | Test-branch timeline dictionary. |

### 12.4 `Data Sets/Cleaned Data/Partitioned Data/AI Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `base_dataset.pkl` | `05/00` | `05/00b`, `05/03` | HITL seed partition of the AI corpus: the tweets to be labelled first (`05/00`). |
| `hitl_pending_batch_{i + 1}.pkl` | `05/00` | — | Pending HITL batches written at preparation time, one per round. |
| `hitl_pending_batch_{PENDING_BATCH_TO_PROCESS}.pkl` | — | `05/02` | The pending batch the training loop (`05/02`) consumes in a given round (same files, read side). |
| `inference_dataset.pkl` | `05/00` | `05/03` | The remainder of the AI corpus held out for final inference (`05/03`). |
| `llm_bootstrap_dataset.pkl` | `05/00` | `05/01`, `05/03` | Partition sent to the LLM bootstrap labeller (`05/01`). |
| `partition_ids.pkl` | `05/00` | `05/04` | Tweet ids per partition; `05/04` uses it to exclude already-handled tweets from full inference. |
| `retweets_dataset.pkl` | `05/00` | `05/03` | Retweets split off from the partitions; annotated separately in `05/03`. |

### 12.5 `Data Sets/Cleaned Data/Tweet Sheets/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `top_test_ai_tweets.csv` | `03/02` | — | Example sheet: top test tweets of the AI corpus (`03/02`). |
| `top_test_art_tweets.csv` | `03/02` | — | Example sheet: top test tweets of the AI+Art corpus (`03/02`). |

### 12.6 `Data Sets/Classifiers_Data/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Final/final_annotated_tweets.csv` | `05/03` | — | CSV copy of the same. |
| `Final/final_annotated_tweets.pkl` | `05/03` | `05/04` | Final classifier annotations on the HITL remainder (`05/03`); input of full inference. |
| `Full_Inference/checkpoint_{n_processed}.pkl` | `05/04` | — | Checkpoints of the full-corpus inference every 100k tweets (`05/04`). |
| `Full_Inference/full_inference_annotated.csv` | `05/04` | — | CSV copy of the same. |
| `Full_Inference/full_inference_annotated.pkl` | `05/04` | — | Classifier annotations for the full AI corpus (`05/04`). |
| `HITL/base_playground_sample.csv` | `05/00b` | — | Sample used to draft and test the label definitions (`05/00b`). |
| `HITL/hitl_review_batch_00.csv` | `05/00` | — | First human-review sheet (`05/00`). |
| `HITL/hitl_review_batch_{PENDING_BATCH_TO_PROCESS}.csv` | `05/02` | — | Human-review sheets of the later rounds (`05/02`). |
| `HITL/llm_bootstrap_labels.csv` | `05/01` | `05/02`, `05/03` | LLM bootstrap labels per tweet (`05/01`); seed labels for the HITL loop and the final model. |
| `HITL/llm_bootstrap_labels_full.pkl` | `05/01` | — | Full LLM bootstrap output including confidence and rationale fields. |
| `HITL/llm_bootstrap_seen_ids_AI.json` | `05/01` | `05/01` | Ids already sent to the LLM; lets `05/01` resume without re-billing. |
| `HITL/llm_bootstrap_usage_{usage_record['timestamp'].replace(':', '')}.json` | `05/01` | — | Per-run API usage / cost record of the bootstrap labelling. |

### 12.7 `Data Sets/Networks/`

*Reorganised 2026-10-10.* One subfolder per pipeline stage, each written by one notebook. The folder variables — `retweet_dicts_folder`, `full_graphs_folder`, `backbones_folder`, `communities_folder`, `visualizations_folder`, `test_networks_folder` — are one block, repeated verbatim in the Setup cell of `02/02`, `04/01` and `04/02`. File names are unchanged. The flat layout before that date also held the legacy files named in *Reading conventions*, `Full_RetweetedOnce_Influence_<method>.gml` (2026-10-07), `leiden_runs/` and the `viz_outputs_*/` folders; all of them were deleted with the reorganisation.

**Live listing:** [Drive: Data Sets/Networks/](https://drive.google.com/drive/folders/1PlVu_Li9nSI7IDLLfURp09s_1bAGXpMq?usp=sharing) (shared link; the page lists file names, modification dates and sizes, so it can be fetched to check what exists before trusting this table).

| Folder | Written by | Holds |
|---|---|---|
| `1_retweet_dicts/` | `02/02` | the AI-corpus retweet counter |
| `2_full_graphs/` | `02/02` | the full retweet graph, both orientations |
| `3_backbones/` | `02/02` (`90TS_LWCC` by `04/01`, optional) | the two analysed backbones, both orientations |
| `4_communities/RetweetedOnce/`, `4_communities/LWCC/` | `04/01` | five annotated graphs and the community JSON per backbone |
| `5_visualizations/<NETWORK>/` | `04/02` | maps and layouts per rendered network (§12.8) |
| `test/` | `02/02` (test branch) | the test-corpus counter and every `Test_*` graph |

| File | Written by | Read by | What it is |
|---|---|---|---|
| `1_retweet_dicts/full_network_dict.pkl` | `02/02` | `02/02` | Nested counter `{retweeted_author: {retweeter: n}}` for the AI corpus: 374,368 retweeted authors, 9,638,407 retweets (§3). |
| `2_full_graphs/Full_Network_Influence.gml` | `02/02` | — | **Full retweet graph**, influence orientation (retweeter → retweeted); identical to the legacy `Full_Network.gml` (3,379,040 authors, 7,768,720 edges). |
| `2_full_graphs/Full_Network_InfoFlow.gml` | `02/02` | `04/01` | `02` full graph, information-flow orientation (the transpose; same counts). |
| `3_backbones/90TS_LWCC.gml` | `04/01` | — | Strategy 2: drop lowest total-strength nodes until 10 % of weight is lost, then LWCC — 2,315,573 authors (§5). Written by `01` only when `RUN_STRATEGY_2` is on. |
| `3_backbones/90TS_LWCC.graphml` | `04/01` | — | GraphML copy of strategy 2. |
| `3_backbones/Full_LWCC_Influence.gml` | `02/02` | — | **LWCC backbone**: self-loops removed, largest weakly connected component of the full graph — 3,264,499 authors, 7,670,516 edges, weight 9,458,703 (= the pre-2026-10 strategy 1); influence orientation, written for reference. |
| `3_backbones/Full_LWCC_InfoFlow.gml` | `02/02` | `04/01` | Transpose of the LWCC backbone (edge retweeted → retweeter); the file `04/01` clusters and `04/02` renders. |
| `3_backbones/Full_RetweetedOnce_Influence.gml` | `02/02` | — | **Retweeted-once backbone**: self-loops removed, in-strength ≥ 1 (363,618 authors), then LWCC — **202,710 authors, 882,530 edges, weight 1,269,747** (13.17 % of retweets); influence orientation (§9.3). |
| `3_backbones/Full_RetweetedOnce_InfoFlow.gml` | `02/02` | `04/01` | Transpose of the retweeted-once backbone (edge retweeted → retweeter); the file `04/01` clusters and `04/02` renders. |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_author_communities.json` | `04/01` | — | `{author_id: {method: community_id}}` for the 202,710 backbone authors and all five methods (`01`); the author set for downstream restriction. |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_infomap.gml` | `04/01` | `04/01` | Retweeted-once backbone (flow orientation) with the `community_infomap` vertex attribute (read back as `communityinfomap`, float → int) (`01`). |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_label_propagation.gml` | `04/01` | `04/01` | Retweeted-once backbone (flow orientation) with the `community_label_propagation` vertex attribute (read back as `communitylabelpropagation`, float → int) (`01`). |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_leiden_directed.gml` | `04/01` | `04/01` | Retweeted-once backbone (flow orientation) with the `community_leiden_directed` vertex attribute (read back as `communityleidendirected`, float → int) (`01`). |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_leiden_fast.gml` | `04/01` | `04/01`, `04/02` | Retweeted-once backbone (flow orientation) with the `community_leiden_fast` vertex attribute (read back as `communityleidenfast`, float → int) (`01`). |
| `4_communities/RetweetedOnce/Full_RetweetedOnce_InfoFlow_louvain.gml` | `04/01` | `04/01` | Retweeted-once backbone (flow orientation) with the `community_louvain` vertex attribute (read back as `communitylouvain`, float → int) (`01`). |
| `4_communities/LWCC/Full_LWCC_author_communities.json` | `04/01` | — | Same for the LWCC backbone (3,264,499 authors × 5 methods, `01`). |
| `4_communities/LWCC/Full_LWCC_InfoFlow_infomap.gml` | `04/01` | `04/01` | LWCC backbone (flow orientation) with the `community_infomap` vertex attribute (`01`). |
| `4_communities/LWCC/Full_LWCC_InfoFlow_label_propagation.gml` | `04/01` | `04/01` | LWCC backbone (flow orientation) with the `community_label_propagation` vertex attribute (`01`). |
| `4_communities/LWCC/Full_LWCC_InfoFlow_leiden_directed.gml` | `04/01` | `04/01` | LWCC backbone (flow orientation) with the `community_leiden_directed` vertex attribute (`01`). |
| `4_communities/LWCC/Full_LWCC_InfoFlow_leiden_fast.gml` | `04/01` | `04/01` | LWCC backbone (flow orientation) with the `community_leiden_fast` vertex attribute (`01`). |
| `4_communities/LWCC/Full_LWCC_InfoFlow_louvain.gml` | `04/01` | `04/01` | LWCC backbone (flow orientation) with the `community_louvain` vertex attribute (`01`). |
| `test/Test_LWCC_Influence.gml` | `02/02` | — | `02` test LWCC backbone, influence orientation (added 2026-10-09). |
| `test/Test_LWCC_InfoFlow.gml` | `02/02` | — | Transpose of the test LWCC backbone. |
| `test/test_network_dict.pkl` | `02/02` | `02/02` | Nested counter `{retweeted_author: {retweeter: n}}` for the test corpus (§3). |
| `test/Test_Network_Influence.gml` | `02/02` | — | Test graph, influence orientation (retweeter → retweeted); identical to the legacy `Test_Network.gml`. |
| `test/Test_Network_InfoFlow.gml` | `02/02` | — | `02` test graph, information-flow orientation (retweeted → retweeter): the transpose. |
| `test/Test_RetweetedOnce_Influence.gml` | `02/02` | — | `02` test backbone: in-strength ≥ 1 then LWCC (3 nodes — the test data is too small to carry one). |
| `test/Test_RetweetedOnce_InfoFlow.gml` | `02/02` | — | Transpose of the test backbone. |

### 12.8 `Data Sets/Networks/5_visualizations/<NETWORK>/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `RetweetedOnce_leiden_fast/demo_network.gml` | `04/02` | `04/02` | Synthetic stochastic-block-model fallback written by `04/02` only when the selected `.gml` is missing, so Run-all still completes. |
| `RetweetedOnce_leiden_fast/network_with_layout.graphml` | `04/02` | — | Optional GraphML with x/y and community for Gephi (`EXPORT_GRAPHML`, ~1.2 GB at 2M nodes; `04/02`). |
| `RetweetedOnce_leiden_fast/positions_drl.parquet` | `04/02` | — | Node table of the igraph DrL layout (`04/02`; the DrL cell imports `src.network.network_utils` and needs the repo on `sys.path`). |
| `RetweetedOnce_leiden_fast/positions_{_recipe}.parquet` | `04/02` | — | Node table per ForceAtlas2 recipe (`classic`, `linlog`): x, y, label, degree, in-degree, community, rank (`04/02`). The maps themselves — `social_map_<recipe>_<light|dark>.png`, `05_degree_ccdf.png`, `06_adjacency_blocks.png` — are written beside it by `savefig`, which the parser does not record. |

### 12.9 `Models/Topic Modeling/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `bow_test_corpus.pkl` | `03/06` | `03/06` | Bag-of-words test corpus of the topic-modelling appendix (`03/06`). |
| `full_sentences_corpus_embedding.pkl` | `03/05` | `03/05` | `{tweet id: {text, processed_text, date, public_metrics}}` for the full corpus — the input of the embedding step, not embeddings (`03/05`, written when `do_corpus = True`). |
| `hf_embeddings.npy` | `03/05` | `03/05` | Hugging Face embeddings array (`03/05`). |
| `LDA/lda_grid_results.csv` | `03/03` | `03/03` | Perplexity (and coherence) per K of the tweet-topic grid search (`TOPIC_GRID`). |
| `LDA/lda_k5_doc_topic_matrix.npy` | `03/03` | — | Document × topic matrix of the training sample for the K = 5 model. |
| `LDA/lda_k5_topics_metadata.json` | `03/03` | — | Metadata of the K = 5 tweet-topic LDA (`03/03`). |
| `processed_sentence_transformer_embeddings.npy` | `03/05` | `03/05` | Post-processed sentence-transformer embeddings (`03/05`). |
| `sentence_transformer_reduced_embeddings.npy` | `03/05` | `03/05` | Dimensionality-reduced embeddings for the map (`03/05`). |
| `test_sentences_corpus.pkl` | — | `03/06` | Test sentence corpus read by the appendix (no producer in the repository). |
| `test_sentences_corpus_embedding.pkl` | `03/05` | `03/05` | Same per-tweet dictionary for the test corpus (`03/05`, `do_corpus = True`). |

### 12.10 `HPC: /projects/ComputationalPhilosophyLab/TwitterDataAnalysis/Data Sets/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Cleaned Data/ai_full_classified_{alias}.json` | `03/01b` | `03/01b` | Merged per-model output for the AI corpus (8b); verified against the input by 8a (§6.2.2). Stored emotion file bad (§6.2.1). |
| `Cleaned Data/AI_pruned_tweets_with_topic_weights_v2.json` | `03/04c` | `03/04c` | Author-LDA v2: tweets with topic weights (`03/04c`; on the wrong author set, §6.4). |
| `Cleaned Data/ai_test_classified_{model_alias}.json` | `03/01b` | `03/01b` | Test output per model, AI corpus (one pass; copies on Drive under `Cleaned Data/Sentiment Analysis/`). |
| `Cleaned Data/AItrust_twits_pruned_dict.json` | — | `03/04c` | HPC copy of the AI corpus (17,410,035 tweets), read by author-LDA v2. |
| `Cleaned Data/art_full_classified_{alias}.json` | `03/01b` | `03/01b` | Merged per-model output for the AI+Art corpus (8b); verified by 8a. Stored sentiment file is AI-corpus content (§6.2.1). |
| `Cleaned Data/art_test_classified_{model_alias}.json` | `03/01b` | `03/01b` | Test output per model, AI+Art corpus (one pass; copies on Drive under `Cleaned Data/Sentiment Analysis/`). |
| `Cleaned Data/author_sentiment_mean.pkl` | `03/04c` | — | Author-level mean sentiment computed in `03/04c`. |
| `Cleaned Data/Blocks/{ds_key}__{alias}__{num_blocks}blocks/block_{bid}.json` | `03/01b` | `03/01b` | Per-job, per-block outputs of the full runs (8b), reused only when their ids equal the input slice (§6.2.2). The pre-2026-10-10 folders `Blocks/<alias>/` and `Blocks/ai_full__<alias>/` are legacy (8c). |
| `Networks/3_backbones/Full_LWCC_InfoFlow.gml` | — | `03/04c` | HPC copy of `02`'s LWCC backbone, to be copied over from Drive (until 2026-10-10 `03/04c` read the legacy `LWCC.gml`, the same graph); `03/04c` filters authors on its GML `id` field instead of `label` (§6.4). |

### 12.11 `HPC: /projects/ComputationalPhilosophyLab/TwitterDataAnalysis/Models/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Sentiment/model_summary_test.csv` | `03/01b` | — | v4 validation summary across the 10 models (`03/01b`). |
| `Sentiment/validation_{ds_key}_{model_alias}.csv` | `03/01b` | `03/01b` | v4 per-dataset, per-model validation metrics. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/ALL_REPRESENTATIONS_LDA_FULL_GRID.csv` | `03/04c` | `03/04c` | v2 108-model grid results (representations × K × α × η). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/ALL_REPRESENTATIONS_LDA_SUMMARY.csv` | `03/04c` | — | Summary of the v2 grid. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/author_docs.csv` | `03/04c` | — | v2 author documents (one row per author) fed to the LDA grid. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/bow_bigram/author_topics_k{K}_top_terms.csv` | — | `03/04c` | v2 top terms per topic, `bow_bigram` representation (read back for inspection). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/bow_unigram/author_topics_k{K}_top_terms.csv` | — | `03/04c` | v2 top terms per topic, `bow_unigram` representation (read back for inspection). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/candidate_diversity_scores.csv` | `03/04c` | — | Topic-diversity scores of the candidate models. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/dominant_topics_lowest_catchall_k12.csv` | `03/04c` | — | Dominant topic per author for the chosen K = 12 model. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/embedding_lowest_catchall_k12.npy` | `03/04c` | — | UMAP embedding of θ for the chosen model. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/focused_grid_results.csv` | `03/04c` | `03/04c` | v2 focused grid around the best K. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/hdbscan_author_clusters.csv` | `03/04c` | — | HDBSCAN clusters of authors on the UMAP embedding. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/high_k_grid_results.csv` | `03/04c` | — | v2 high-K grid. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_bigram/author_topic_matrix_tfidf_bigram_k{K}_a{a_tag}_e{e_tag}.csv` | — | `03/04c` | v2 θ matrix, `tfidf_bigram`, per grid point. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_bigram/author_topics_k{K}_top_terms.csv` | — | `03/04c` | v2 top terms per topic, `tfidf_bigram` representation (read back for inspection). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_bigram/author_topics_tfidf_bigram_k{K}_a{a_tag}_e{e_tag}_top_terms.csv` | — | `03/04c` | v2 top terms, `tfidf_bigram`, per grid point. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_bigram/theta_v3stopwords_tfidf_bigram_k{K}_a{a_tag}_e{e_tag}.csv` | `03/04c` | — | v2 θ matrix with the v3 stop-word list, `tfidf_bigram`. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/author_topic_matrix_tfidf_unigram_k{best_k}_a0p01_e0p1.csv` | — | `03/04c` | v2 θ matrix, `tfidf_unigram`, best K, α = 0.01. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/author_topic_matrix_tfidf_unigram_k{best_k}_a0p1_e0p1.csv` | — | `03/04c` | v2 θ matrix, `tfidf_unigram`, best K, α = 0.1. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/author_topics_k{K}_top_terms.csv` | — | `03/04c` | v2 top terms per topic, `tfidf_unigram` representation (read back for inspection). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/author_topics_tfidf_unigram_k{best_k}_a0p01_e0p1_top_terms.csv` | — | `03/04c` | v2 top terms for the best-K unigram model. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/lda_model_k16_a0p1_e0p1.pkl` | — | `03/04c` | v2 fitted LDA model, K = 16 (read). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/lda_model_k{K}_a0p1_e0p1.pkl` | `03/04c` | — | v2 fitted LDA models per K (written). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/vectorizer_k16_a0p1_e0p1.pkl` | — | `03/04c` | v2 TF-IDF vectorizer, K = 16 (read). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/tfidf_unigram/vectorizer_k{K}_a0p1_e0p1.pkl` | `03/04c` | — | v2 TF-IDF vectorizers per K (written). |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/theta_lowest_catchall_k12.npy` | `03/04c` | — | Author × topic matrix (θ) of the chosen K = 12 model. |
| `Topic Modeling/LDA/author_level/full_v2_cleaned/umap_dimensionality_comparison.csv` | `03/04c` | — | UMAP settings comparison for the author map. |

### 12.12 `Colab-local /content/`

None since 2026-10-10. The only entry was the Colab-local copy of the sentiment file that `03/04` read; it went with that notebook.

