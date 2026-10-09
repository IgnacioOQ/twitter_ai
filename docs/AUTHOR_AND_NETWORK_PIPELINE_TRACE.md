---
status: active
type: reference
id: twitter_ai.author_and_network_pipeline_trace
description: Cell-by-cell trace of every tweet count, author count and network step across 02_Processing, 03_Analysis_and_Modeling and the pruning step of 04_Network_Analysis — numbers, the exact code that produced them, the discrepancies found, and a full inventory of every file the pipeline reads or writes (§12).
label: [dataset, network, authors, provenance, audit]
volatility: evolving
scope: project-specific
repository: [twitter_ai]
last_checked: '2026-10-09'
---

# Author & Network Pipeline Trace

This document answers one question for every step of the pipeline: **how many tweets and authors are there at this point, and which lines of code decided that?** It complements [DATASET_STATISTICS.md](DATASET_STATISTICS.md), which gives the narrative and the master statistics. Where the two overlap, the numbers agree; this document adds the code, the test-run numbers, every downstream consumer in `03_Analysis_and_Modeling`, and three discrepancies that the stored notebook outputs reveal.

> **Status 2026-10-08.** The corrected pipeline has been executed through community detection: `02b` (both networks + the retweeted-once backbone, 202,710 authors) and `01b` (five partitions + JSON export) ran on Colab without errors and their outputs are stored in the notebooks (§9.3). `02b-viz` is running. The original `04/01` / `04/02` outputs on Drive are superseded but not deleted. Still open: the author-level LDA fix (§6.4), the sentiment block drift (§6.2), the first-generation corpus in the tweet-topic notebooks (§6.3), and the blog draft, whose network-dependent sections are built on the retweeter backbone (§11).
>
> **Status 2026-10-09.** The `02b-viz` run of 2026-10-08 failed before layout (the cuGraph install cell assumed a cu12 RAPIDS image; Colab GPU runtimes now ship cu13 — fixed in both visualisation notebooks, §9.5). The corrected pipeline was then **extended to a second backbone**: `02b` also exports the direction-neutral LWCC in both orientations (`Full_LWCC_{Influence,InfoFlow}.gml`), `01b` clusters both backbones in one pass (`BACKBONE_STEMS`), and `02b-viz` lists `LWCC_*` registry keys (§8 "Corrected notebooks", §9.3, §10.3). **None of the LWCC outputs exist on Drive yet**; the three notebooks are queued for a re-run.
>
> **Rename 2026-10-09.** The original `02_Processing/02_sanity_check_and_network_generation.ipynb` was deleted and the corrected `02b_influence_and_flow_networks.ipynb` was renamed to that name (`git mv`, history preserved). Throughout this document `02b` means today's `02_sanity_check_and_network_generation.ipynb`, and `02/02` with a cell number means the deleted original, whose stored outputs are the source of §§1–4. `Full_Network.gml`, `.graphml`, `.gexf`, `.json` and `Test_Network.*` are now legacy files on Drive with no producing notebook; `Full_Network_Influence.gml` is the same graph.

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
| Total-strength 90 % pruning + LWCC | `04/01` cell 16 | — | 2,315,573 | not used downstream |
| Out-strength ≥ 1 (made ≥ 1 retweet) | `04/01` cell 18 | — | 3,159,105 | before LWCC |
| **Out-strength ≥ 1 + LWCC** (`Final_OutThreshold1.gml`) | `04/01` cell 18 | — | **1,984,599** | the set used for community detection |
| Sentiment v4, AI corpus | `03/01b` cell 35 | 17,410,035 | — | emotion run short by 8,975 (see §6.2) |
| Sentiment v4, AI+Art corpus | `03/01b` cell 34 | 3,583,101 | — | sentiment result file holds 4,057,340 lines (see §6.2) |
| Tweet-level LDA / cleaning / top-K | `02/03`, `03/03` | 21,466,173 lines read | — | **old-generation corpus**, not the 17.41M one (see §6.3) |
| Author-level LDA v1 | `03/04` cell 14 | 1,000,000 read → 4,558 kept | 1,704 | LCC filter matches the wrong field (see §6.4) |
| Author-level LDA v2 | `03/04c` cell 30 | 17,410,035 read → 65,615 kept | 8,000 → 3,682 (≥ 3 tweets) | same bug |
| Test branch, AI / AI+Art | `02/02` cell 17 | 881 / 216 | 723 / 198 | from one raw window file (233,094 records) |
| **In-strength ≥ 1 + LWCC** (`Full_RetweetedOnce_Influence.gml`, `02b`) | `02b` full pruning cell | — | **202,710** | authors retweeted ≥ 1× by someone else (363,618 before LWCC); 13.17 % of retweet weight; the corrected backbone (§9.3) |
| **LWCC, both directions** (`Full_LWCC_{Influence,InfoFlow}.gml`, `02b`) | `02b` full LWCC cell | — | **3,264,499** (expected) | direction-neutral: self-loops removed, giant component — the same graph as strategy 1 (`04/01` cell 13: 3,264,499 nodes, 7,670,516 edges), now in both orientations; added 2026-10-09, not yet run; `01b` clusters it beside the retweeted-once backbone (§9.3) |
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

Notebook: [02_Processing/02_sanity_check_and_network_generation.ipynb](../notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb)

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

## 5. Where the author set shrinks to 1,984,599

Notebook: [04_Network_Analysis/01_network_analysis.ipynb](../notebooks/04_Network_Analysis/01_network_analysis.ipynb). This is outside the `02`/`03` folders but it is the step that produces the author set every later notebook is meant to be restricted to, so it belongs in this trace.

All three strategies start from `Full_Network.gml`, drop self-loops (32,216 of them — authors retweeting themselves), prune, then keep the largest weakly connected component.

| Strategy | Cell | Rule | Nodes | Edges | Output |
|---|---|---|---:|---:|---|
| 1 — LWCC only | 13 | no pruning | 3,264,499 | 7,670,516 | `LWCC.gml` |
| 2 — total-strength 90 % | 16 | delete lowest in+out strength nodes until 10 % of weight is lost (958,959 nodes removed), then LWCC | 2,315,573 | 6,721,590 | `90TS_LWCC.gml` |
| 3 — out-strength ≥ 1 | 18 | keep nodes with out-strength ≥ 1 (3,159,105 of 3,379,040), then LWCC | **1,984,599** | 4,472,376 | `Final_OutThreshold1.gml` |

The rule of strategy 3 is in [src/network/network_pruning.py:226](../src/network/network_pruning.py#L226):

```python
def prune_by_out_strength_threshold(g, threshold=1.0):
    ...
    g_clean = g.simplify(loops=True, multiple=False)          # self-loops removed first
    out_str = g_clean.strength(mode="out", weights=weight_attr)
    nodes_to_keep = [i for i, s in enumerate(out_str) if s >= threshold]
    ...
    pruned = pruned.components(mode="weak").giant()
```

Given the edge direction of §4.1, **`out_strength >= 1` keeps authors who made at least one retweet (of someone else)**. It does *not* select authors who were retweeted. The threshold deletes the *node*, so an author who was retweeted (even heavily) but never retweeted anyone is removed outright, together with all edges pointing at them. Of the 3,159,105 retweeters, 1,984,599 sit in one connected component; the other 1,174,506 are in small islands and are dropped by the LWCC step. Weight kept: 60.46 % of the original.

If the intended population was "authors retweeted at least once" the equivalent filter is `mode="in"` on the same function; nothing else in the pipeline would need to change, but every community file and the JSON export below would have to be regenerated.

Two readings of the same graph help keep this straight. Under the **influence** reading (the graph as built), A → B means A retweeted B, in-strength is retweets received, and the current rule keeps nodes with ≥ 1 *outgoing* edge. Under the **information-flow** reading (the transpose), A → B means B retweeted A, and the same rule keeps nodes that *received* information from ≥ 1 source while dropping every pure source. "Retweeted at least once" is a source condition: in-strength in the influence graph, out-strength in the flow graph. Direction also matters for Infomap (`04/01`, directed methods), whose random walker follows edge direction and therefore walks against information flow on the graph as built; the directed Leiden quality function is invariant under transposition and the three undirected methods are unaffected.

### 5.1 Does the pruning direction reach the sentiment or topic analyses?

Not yet. No notebook in `02` or `03` reads `Final_OutThreshold1.gml` or `Final_OutThreshold1_author_communities.json`. What each analysis actually restricted on:

| Analysis | Author restriction intended | Author restriction actually applied |
|---|---|---|
| Sentiment v1 (`03/01`) | none | none — all 17,410,035 tweets |
| Sentiment v4 (`03/01b`) | none | none — 17,410,035 AI and 3,583,101 AI+Art tweets |
| Tweet-level LDA, topics in time, top-K per topic (`03/03`, `02/03`) | none | none — on the first-generation corpus (§6.3) |
| Author-level LDA v1 / v2 (`03/04`, `03/04c`) | authors in the Strategy-1 LWCC (3,264,499; both retweeters and retweeted, no direction involved) | legacy low-id accounts only, because the loader matches GML `id` instead of `label` (§6.4) |
| Classifiers (`05_Classifiers`) | none | none — they read the AI+Art tweet file directly |

So the sentiment and tweet-topic results cover every author in the corpus, including the 1,396,671 who never retweeted or were retweeted, and the author-level LDA was designed against the undirected component, not the out-strength set. The choice between "everyone in the component", "authors who retweeted" and "authors who were retweeted" becomes consequential the first time a downstream notebook restricts to the 1,984,599-author set; the author-level LDA re-run (§8) is where that will happen.

Cell 27 (*Export community memberships as JSON*) reads the `label` attribute (= author id) of each `Final_OutThreshold1_<method>.gml` and writes `Final_OutThreshold1_author_communities.json` = `{author_id: {method: community}}`. Its output is not stored in the committed notebook, so the exported key count (expected 1,984,599) is unverified here.

## 6. Consumers in `03_Analysis_and_Modeling` (and `02/03`)

### 6.1 `01_sentiment_analysis.ipynb` (v1, CardiffNLP sentiment only)

Cell 18 (heading *For Full Data Set*):

```python
input_path        = cleanedds_folder / 'AItrust_twits_pruned_dict.json'
final_output_path = cleanedds_folder / 'AItrust_pruned_twits_with_sentiment.json'
num_blocks = 10
total_tweets = count_lines(input_path)           # → 17410035
```

Stored output: `📊 Total tweets: 17410035`; ten blocks of 1,741,003 lines (last 1,741,008) all reported complete; merged to `.../Cleaned Data v2/AItrust_pruned_twits_with_sentiment.json`. Note the **`Cleaned Data v2`** folder: this run wrote next to, not over, the older file of the same name in `Cleaned Data/` (see §6.3).

### 6.2 `01b_sentiment_emotion_v4_hpc.ipynb` (v4, HPC, 10 models)

Datasets registered in cell 14/34/35 (`cleanedds_folder` = `/projects/ComputationalPhilosophyLab/TwitterDataAnalysis/Data Sets/Cleaned Data`):

| Key | File | Tweets |
|---|---|---:|
| `ai_test` | `AItrust_twits_pruned_dict_test.json` | 881 |
| `art_test` | `AItrust_Art_pruned_twit_dict_test.json` | 216 |
| `ai_full` | `AItrust_twits_pruned_dict.json` | 17,410,035 |
| `art_full` | `AItrust_Art_pruned_twit_dict.json` | 3,583,101 |

Full runs (cells 34 and 35) use the same block scheme as v1 (`total = sum(1 for _ in f)`, 10 blocks, skip a block whose output already exists). Cell 36 then counts the lines of the merged result files:

| Result file | Lines | Expected | Δ |
|---|---:|---:|---:|
| `ai_full × twitter-roberta-base-sentiment-latest` | 17,410,035 | 17,410,035 | 0 |
| `ai_full × twitter-roberta-base-emotion-multilabel-latest` | 17,401,060 | 17,410,035 | **−8,975** |
| `art_full × twitter-roberta-base-emotion-multilabel-latest` | 3,583,101 | 3,583,101 | 0 |
| `art_full × twitter-roberta-base-sentiment-latest` | **4,057,340** | 3,583,101 | **+474,239** |

Both deltas are explained by the stored block log of cell 35: block 0 of the AI emotion run was accepted with 1,732,028 lines instead of 1,741,003 (exactly 8,975 short, the skip rule only requires ≥ 95 %), and the art sentiment file contains 474,239 surplus lines, consistent with a block having been appended twice during a resumed run. Every percentage reported for "AI+Art sentiment (full)" in cell 39 is computed on the 4,057,340-line file, so it double-counts part of the corpus.

Full-corpus label distributions from cell 39 (for the record):

| Model × corpus | n | negative | neutral | positive |
|---|---:|---:|---:|---:|
| sentiment-latest × AI General | 17,410,035 | 21.2 % | 47.9 % | 30.8 % |
| sentiment-latest × AI+Art | 4,057,340 (see above) | 31.4 % | 42.1 % | 26.5 % |

### 6.3 `02/03_cleaning_tweets.ipynb` and `03/03_lda_tweet_topics.ipynb` — the old-generation corpus

Both read `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (cell 3 of `02/03`; cell 8 of `03/03`, where it is also the source for the K = 5 enrichment file `AItrust_pruned_twits_with_sentiment_and_topics_k5.json`). Their stored outputs do **not** match the 17.41M corpus:

- `02/03` cells 13, 16 and 21 all set `TOTAL_LINES = 22416373` and their progress bars stop at **21,466,173** lines, the last of which is malformed (`line 21466173: JSONDecodeError`). The file therefore has 21,466,173 lines and was cut mid-record.
- `02/03` cell 13: of those lines, 12,537,708 were skipped as non-original, leaving 8,928,465 originals for the top-150-per-topic extraction. The 17.41M corpus has only 4,061,626 originals.
- `03/05_embedding_mapping.ipynb` cell 29 streams `AItrust_pruned_twits.json` with `total=22416373` and cell 30 reports `len(corpus_dict)` (unique ids) as **14,885,897**, i.e. the older pruned file had ~7.5M duplicate ids.

So there are **two generations** of the corpus on disk:

| Generation | File | Lines | Unique ids | Produced by |
|---|---|---:|---:|---|
| 2025 | `AItrust_pruned_twits.json` → `Cleaned Data/AItrust_pruned_twits_with_sentiment.json` (+ `_and_topics_k5`) | 22,416,373 (sentiment copy truncated at 21,466,173) | 14,885,897 | earlier version of `02/02`, before dedup and the stricter keyword filter |
| 2026 | `AItrust_twits_pruned_dict.json`, `AItrust_Art_pruned_twit_dict.json`, `Cleaned Data v2/AItrust_pruned_twits_with_sentiment.json` | 17,410,035 / 3,583,101 | = lines | `02/02` run completed 2026-08-22 (cell 61) |

The tweet-level LDA (k = 5 topics, topic × sentiment counts, "topics in time", top retweets per topic CSV) is all computed on the first generation. Its absolute counts (e.g. Topic 0: 4,867,664 positive / 7,051,674 neutral / 3,534,272 negative in `02/03` cell 22) cannot be reconciled with the 17.41M corpus and should not be quoted next to it.

(The "Disconnected from runtime" timestamps stored in the notebooks — `02/01` 2025-08-20, `03/01` 2025-08-26, `02/02` 2026-08-22 — are the last time the *disconnect cell* ran, not necessarily when the cells above it ran, so they date sessions only loosely.)

### 6.4 `04_lda_author_topics.ipynb` (v1) and `04c_lda_author_topics_v2.ipynb` — the LCC filter matches the wrong GML field

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

- `01_sentiment_analysis_v2.ipynb` — no stored outputs; paths reference `AItrust_pruned_twits.json` (first generation).
- `02_extract_examples.ipynb` — reads the AI and AI+Art *test* files only; no counts stored.
- `05_embedding_mapping.ipynb` — first-generation corpus (§6.3); cell 30 keeps tweets with `like_count > 0` for embedding; count not stored.
- `06_topic_modeling_appendix.ipynb` — sklearn/gensim LDA demos; no corpus counts.

## 7. The AI+Art author subset — what exists and what does not

What exists:

1. **1,440,802 authors** with ≥ 1 art-keyword tweet in the AI corpus — `02/02` cell 20, `unique_authors_art`. Nested inside the 4,775,711.
2. **3,583,101 AI+Art tweets** with v4 sentiment and emotion labels — `03/01b` (with the 4,057,340-line caveat for the sentiment file).
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
| Author-level LDA, both versions | `03/04`, `03/04c` | LCC filter matches GML indices, not author ids (§6.4) — all stored results are on the wrong author set |
| AI+Art sentiment (full) | `03/01b` cell 34 | result file has 474,239 surplus lines; delete `art_full__twitter-roberta-base-sentiment-latest` blocks and re-merge (§6.2) |
| AI emotion-multilabel (full) | `03/01b` cell 35 | block 0 short by 8,975 lines; lower the skip rule from 95 % to 100 % or delete block 0 (§6.2) |
| Tweet-level LDA, topics-in-time, top-K per topic | `03/03`, `02/03` | computed on the first-generation 22.4M-line corpus (§6.3); must be re-run on `Cleaned Data v2/AItrust_pruned_twits_with_sentiment.json` to be comparable with everything else |
| Pruned network + communities + JSON export | `04/01` | the intended population is "retweeted ≥ 1"; the rule implemented is "retweeted someone ≥ 1" (§5, §9). **Done 2026-10-07** via `02b` + `01b` (§9.3); `04/01` outputs are superseded, not deleted. **Extended 2026-10-09**: `02b` / `01b` also produce and cluster the direction-neutral LWCC backbone (`Full_LWCC_*`), not yet run |
| Network visualisations (ForceAtlas2, DrL, adjacency blocks) | `04/02` | rendered on `Final_OutThreshold1_leiden_fast.gml`, i.e. the retweeter backbone (§9.2). **2026-10-08 run failed** before layout (cuGraph install on the new cu13 Colab image; patched 2026-10-09, §9.5). To be re-run on `Full_RetweetedOnce_Influence_leiden_fast.gml` and on `Full_LWCC_Influence_leiden_fast.gml`, one network per run |
| Community-based blog figures 7–11 (and weekly figures 2, 3, 5 if the matched set is network-derived) | branch `add-blog-figures-and-community-analysis` | only if the 198,326 matched authors / their `leiden_directed` labels descend from `Final_OutThreshold1` — to be verified (§9.4) |

**Corrected notebooks (added 2026-10-07).** Three copies implement the fix without touching the originals: [02_Processing/02b_influence_and_flow_networks.ipynb](../notebooks/02_Processing/02b_influence_and_flow_networks.ipynb) builds the influence and information-flow networks and the in-strength ≥ 1 backbone (`Full_RetweetedOnce_{Influence,InfoFlow}.gml`); [04_Network_Analysis/01b_network_analysis_retweeted_once.ipynb](../notebooks/04_Network_Analysis/01b_network_analysis_retweeted_once.ipynb) runs the five community methods on it (Infomap on the flow file), aligns AMI/ARI by author id and exports `Full_RetweetedOnce_author_communities.json`; [04_Network_Analysis/02b_network_visualization_retweeted_once.ipynb](../notebooks/04_Network_Analysis/02b_network_visualization_retweeted_once.ipynb) renders the maps from those files (registry default `RetweetedOnce_leiden_fast`).

**Extended 2026-10-09 — a second, direction-neutral backbone.** `02b` additionally exports the largest weakly connected component of the full network, self-loops removed, in both orientations (`Full_LWCC_{Influence,InfoFlow}.gml`; helpers `largest_weakly_connected_component` and `export_lwcc_both_directions`, which cuts the information-flow copy from the transpose on the same author set and asserts equal node sets and edge counts; stats under `lwcc` in `full_dual_network_stats.json`). `01b` loops over `BACKBONE_STEMS = ['Full_RetweetedOnce', 'Full_LWCC']` for the backbone check, the five methods, AMI/ARI and the export, writing `Full_LWCC_Influence_<method>.gml`, `Full_LWCC_InfoFlow_infomap.gml` and `Full_LWCC_author_communities.json`; its own Strategy-1 recompute cell was removed as redundant, and a `SKIP_EXISTING_COMMUNITY_FILES` flag makes the multi-hour run resumable. `02b-viz` lists seven `LWCC_*` registry keys next to the `RetweetedOnce_*` ones. Until the three notebooks are re-run, no `Full_LWCC_*` file exists on Drive.

## 9. Impact of the pruning-direction error across the study

*Added 2026-10-07 after the direction question was settled.*

### 9.1 What exactly is wrong, and what is not

- **The graph is right.** `Full_Network.gml` has edges retweeter → retweeted, as documented. In-strength is retweets received, out-strength is retweets made (§4.1).
- **The pruning rule is the wrong side of the edge.** `prune_by_out_strength_threshold(threshold=1.0)` keeps authors who *made* ≥ 1 retweet. The intended population was authors who *were retweeted* ≥ 1 times (in-strength). The out-strength rule deletes every retweeted-only author outright (§5).
- **Everything downstream of `Final_OutThreshold1.gml` is therefore a study of retweeters**, 1,984,599 of them, not of the amplified accounts. That file is the sole input of the five community partitions, the AMI/ARI comparison, the community JSON export, and the social-media maps.
- **Nothing in the tweet-level analyses depends on it** (§5.1). Sentiment, emotion and tweet-topic results are computed on every tweet of the corpus.

### 9.2 Stage-by-stage impact

| Stage | Notebook / artefact | Reads the pruned network? | Verdict | What a fix requires |
|---|---|---|---|---|
| Corpora, author counts, funnel | `02/01`, `02/02` cells 12–20 | no | **unaffected** | nothing |
| Retweet graph, topology, top-10 lists | `02/02` cells 44–55 | no (it produces the input) | **unaffected** — direction is correct | nothing |
| Pruning strategies 1 and 2 (`LWCC.gml`, `90TS_LWCC.gml`) | `04/01` cells 13, 16 | no | **direction-neutral** (LWCC ignores direction; total strength = in + out) | nothing — strategy 1 is since 2026-10-09 also written by `02b` as `Full_LWCC_{Influence,InfoFlow}.gml` so that `01b` can cluster it |
| Pruning strategy 3 (`Final_OutThreshold1.gml`) | `04/01` cell 18 | — | **wrong population** (retweeters) | replace with an in-strength rule (`02b` notebook) or a direction-neutral backbone (§9.3) — **done**: `Full_RetweetedOnce_*.gml`, 202,710 authors |
| Community detection, five methods; modularity / codelength; AMI-ARI table | `04/01` cells 21–25 | yes | **valid only as a partition of retweeters**; Infomap additionally walks against information flow on this edge orientation | re-run `run_modularity_workflow` on the new backbone (≈ 10 min per method on 2M nodes); Infomap on the flow (transposed) file — **done** in `01b` (§9.3 results table); **2026-10-09**: `01b` also runs all five methods on `Full_LWCC_*` (pending) |
| Community JSON export (`Final_OutThreshold1_author_communities.json`) | `04/01` cell 27 | yes | same | **done**: `Full_RetweetedOnce_author_communities.json` (`01b`); pending: `Full_LWCC_author_communities.json` |
| Social-media maps (ForceAtlas2 classic / linlog, DrL), degree CCDF, community-blocked adjacency, `positions_*.parquet`, `network_with_layout.graphml` | `04/02` (`NETWORK = 'Final_leiden_fast'`; stored output: 1,984,599 nodes, 1,458 communities, top 16 cover 87 %) | yes | **maps of the retweeter backbone** | re-render (layout ≈ 15 min GPU budget per recipe) — the `02b-viz` run of 2026-10-08 **failed before layout** (cu13 cuGraph install; patched 2026-10-09, §9.5); pending for `RetweetedOnce_leiden_fast` and `LWCC_leiden_fast` |
| Sentiment v1 / v4, per tweet | `03/01`, `03/01b` | no | **unaffected** | nothing — but any *per-community* aggregate of these scores inherits the partition |
| Tweet-level LDA, topics in time, top-K per topic | `03/03`, `02/03` | no | unaffected by direction; still on the first-generation corpus (§6.3) | re-run on the 2026 corpus for other reasons |
| Author-level LDA v1 / v2 | `03/04`, `03/04c` | no — reads `LWCC.gml` (strategy 1) | unaffected by direction; broken by the GML `id`/`label` bug (§6.4) | fix loader, re-run |
| Topic–emotion mixing plan | `docs/TOPIC_EMOTION_MIXING_PLAN.md` | no — consumes `04c` θ | inherits §6.4 | after `04c` is re-run |
| Classifiers | `05_Classifiers` | no | **unaffected** | nothing |
| Blog analysis: 198,326 matched authors, 21 Leiden-directed communities, figures 1–11 | branch `origin/add-blog-figures-and-community-analysis` (commit `da01e80`, 2026-08-21, one commit ahead of `main`, unmerged) | **unknown** — inputs not committed | see §9.4 | verify provenance first |

What survives untouched: every number in §§1–4 of this document, the two corpora, all per-tweet labels, and the LWCC / total-strength backbones.

### 9.3 Is degree pruning worth doing at all?

The concern is legitimate: an in-strength ≥ 1 *subgraph* keeps at most the 374,368 authors who were ever retweeted (§4.1) and only the edges *among* them (a retweeted author retweeting another retweeted author). Pure retweeters, who carry most of the 9.6M retweets, disappear with their edges, so the surviving weight will be a small fraction of the original and the graph will fragment before the LWCC step. The `02b` notebook prints the node count and weight kept after each step so this can be read off directly.

The options, with their populations:

| Backbone | Rule | Nodes | Direction-sensitive? | Comment |
|---|---|---:|---|---|
| A. Full LWCC (`LWCC.gml`) | self-loops removed, giant component | 3,264,499 | no | `LWCC.gml` exists; since 2026-10-09 `02b` also writes it in both orientations as `Full_LWCC_{Influence,InfoFlow}.gml` and `01b` clusters it (pending run) |
| B. Total-strength 90 % (`90TS_LWCC.gml`) | drop lowest in+out nodes until 10 % of weight is lost, then LWCC | 2,315,573 | no | already exists; keeps both heavy retweeters and heavily retweeted |
| C. Out-strength ≥ 1 (`Final_OutThreshold1.gml`) | made ≥ 1 retweet | 1,984,599 | yes — retweeters | current; wrong population |
| D. In-strength ≥ 1 (`02b`: `*_RetweetedOnce_*`) | received ≥ 1 retweet | ≤ 374,368 before LWCC | yes — retweeted | intended population, but a sparse subgraph |

Two things are being conflated by any degree-threshold *subgraph*: the graph on which communities are detected, and the author population reported on. They need not be the same set. Communities can be detected on a direction-neutral backbone (A or B), where every retweeter still contributes structure, and the "retweeted at least once" criterion can then be applied as an **author filter on the community table** (keep rows whose in-strength ≥ 1 in the full graph) rather than as a graph cut. That keeps the retweeters' edges in the clustering while reporting only on amplified authors, and the choice of population becomes a one-line filter that can be changed without re-running Leiden. The in-strength of every author is available from `Full_Network.gml` (`G.in_degree(weight='weight')`).

**The `02b` numbers (Colab run completed 2026-10-07, stored in the notebook and in `Cleaned Data/full_dual_network_stats.json`):**

| Step (influence network) | Nodes | Edges | Weight |
|---|---:|---:|---:|
| Full network | 3,379,040 | 7,768,720 | 9,638,407 |
| Self-loops removed | 3,379,040 | 7,736,504 | 9,606,191 (32,216 self-retweets) |
| In-strength ≥ 1 (retweeted ≥ 1× by someone else) | 363,618 | 890,006 | — |
| Largest weakly connected component | **202,710** | 882,530 | 1,269,747 (**13.17 %** of original) |

Two readings of that table. First, 374,368 authors appear as retweeted in the dictionary but only 363,618 pass the threshold: the other 10,750 were retweeted *only by themselves*. Second, the component step costs 161,000 authors but almost no edges, so the retweeted-once authors outside the giant component are overwhelmingly isolates — authors whose only retweeters were pure retweeters, now deleted. The backbone keeps 13 % of the retweet volume: the 87 % lost is retweets *by* authors who were never themselves retweeted.

A coincidence, now explained (§9.4): 202,710 is within 2.2 % of the blog analysis's **198,326 matched authors**, but the draft states the matched set was cut from the 1,984,599-node out-strength backbone, i.e. it is retweeters with topic data, not retweeted authors. The similar size is accidental.

Recommendation: D is small but connected (one component, 882,530 edges), which is enough to carry community detection; it is also exactly the intended population. Run `01b` on it. If community structure on 200k authors turns out too coarse for the research question, fall back to A or B for detection and apply "retweeted ≥ 1" as the downstream author filter, as described above.

**2026-10-09:** the fallback to A is wired in rather than hypothetical. `01b` runs the five methods on both D and A in one pass (`BACKBONE_STEMS = ['Full_RetweetedOnce', 'Full_LWCC']`), so the two partitions can be compared side by side before the population decision of §11.4 is taken. Expected cost: each method on the 3.26M-node LWCC is 10–20 min, roughly 1.5–2 h for the five on both backbones; the AMI cell holds five 3.26M-entry membership dicts (~3 GB).

**`01b` results (Colab run completed 2026-10-07, CPU runtime, no errors; outputs stored in the notebook):**

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

Two places:

- [04_Network_Analysis/02_network_visualization.ipynb](../notebooks/04_Network_Analysis/02_network_visualization.ipynb) on `main`: loads one of the `Final_OutThreshold1*.gml` files (configured `NETWORK = 'Final_leiden_fast'`), undirects, keeps the giant component (1,984,599 nodes), colours the 16 largest precomputed communities, lays out with GPU ForceAtlas2 (`classic`, `linlog`) and, since 2026-08-12, igraph DrL via `src/network/network_utils.py::compute_drl_layout`; writes `social_map_<recipe>_<light|dark>.png`, `positions_<recipe>.parquet`, `05_degree_ccdf.png`, `06_adjacency_blocks.png`, optionally `network_with_layout.graphml`, into `Networks/viz_outputs_Final_leiden_fast/`. Every rendered map is therefore a map of the retweeter backbone (§9.2).
- [04_Network_Analysis/02b_network_visualization_retweeted_once.ipynb](../notebooks/04_Network_Analysis/02b_network_visualization_retweeted_once.ipynb): the same pipeline with a registry of the `02b` / `01b` files — seven `RetweetedOnce_*` keys and, since 2026-10-09, seven `LWCC_*` keys; one network per run, output in `Networks/viz_outputs_<NETWORK>/`. Its 2026-10-08 run (`RetweetedOnce_leiden_fast`, A100) loaded and preprocessed the 202,710-node graph (undirected: 864,436 edges, one component, 1,089 Leiden-fast communities, top 16 = 91 % of nodes) and then failed at `import cudf`: the cuGraph install cell probed only `*-cu12` packages, and Colab GPU runtimes now ship RAPIDS cu13, so an unpinned cu12 wheel overwrote the preinstalled stack. Both visualisation notebooks now detect the CUDA suffix first and refuse an unpinned install (2026-10-09); the failed outputs were stripped from the committed copy. Still open: the trailing DrL cell imports `src.network.network_utils` although the notebook has no clone cell, so it cannot run on Colab.
- The interactive DrL map (`davidfreeborn.github.io/twitter-authors-map`), maintained outside this repository; the blog branch's `nodes.json` naming refers to it.

## 10. The pipeline graph — machine-derived double check

*Added 2026-10-07.*

[src/scripts/pipeline_graph.py](../src/scripts/pipeline_graph.py) derives a bipartite producer/consumer graph from the notebooks themselves: it parses every `.ipynb` with Python's AST, resolves the path variables of each setup cell, and records which Drive files each notebook opens for reading and writing. It writes [docs/pipeline_graph.json](pipeline_graph.json) (queryable), [docs/pipeline_graph.png](pipeline_graph.png) (notebooks ↔ artifacts) and [docs/pipeline_graph_notebooks.png](pipeline_graph_notebooks.png) (notebook-only DAG). It is the independent check on everything this document says by hand.

```bash
python3 src/scripts/pipeline_graph.py              # rebuild the three files in docs/ from the live notebooks
python3 src/scripts/pipeline_graph.py validate     # compare the provenance tags in README.md / notebook_setup.md with the parser
python3 src/scripts/pipeline_graph.py upstream   04_Network_Analysis/01b_network_analysis_retweeted_once
python3 src/scripts/pipeline_graph.py downstream 02_Processing/02b_influence_and_flow_networks
```

### 10.1 State on 2026-10-07

- The committed graph dated from 2026-05-18 and covered 19 notebooks. Rebuilt: **28 notebooks, 159 artifacts, 147 write edges, 125 read edges**; `validate` reports **no drift** between the two Drive directory listings and the graph.
- `validate` initially flagged two **wrong provenance tags** in `notebooks/notebook_setup.md` / `README.md`, now corrected: `top_test_ai_tweets.csv` and `top_test_art_tweets.csv` are written by `03_Analysis_and_Modeling/02`, not `02_Processing/02`; `Full_Network.gml` is written by `02_Processing/02`, not `04_Network_Analysis/01` (which only reads it).
- The three corrected notebooks (§9, "Corrected notebooks") are registered in [docs/pipeline_overrides.yaml](pipeline_overrides.yaml) as **alternatives** of their originals (dashed edges; both appear in the graph), and the new network artifacts are listed in both directory listings with `[written by …/02b]` / `[…/01b]` tags.
- **Rebuilt 2026-10-09** after the LWCC extension: **28 notebooks, 169 artifacts, 155 write edges, 132 read edges**; 53 declared edges (20 added: the `02b` LWCC writes for the Test and Full branches, the `01b` reads of all four backbone files and its five LWCC community writes and read-backs, and the two community JSONs — `COMMUNITIES_JSON` is now a dict comprehension over `BACKBONE_STEMS`, which the parser cannot resolve, so `Full_RetweetedOnce_author_communities.json` moved from parsed to declared). `validate` clean.

### 10.2 What the parser cannot see — and how it is covered

The parser only resolves literal paths and simple `folder / 'name'` expressions whose variables are bound in the same notebook. It is blind to:

1. paths assembled **inside a user-defined function** (the `export_network()` helper of `02b`, the `block_folder / f'…_{block_id}.json'` merges of `03/01`);
2. files written by **`src/` workflows** (`run_modularity_workflow` writes `<input stem>_<method>.gml`), so every community-annotated GML — old and new — was invisible as a *write*;
3. **registry dictionaries** resolved at run time (`NETWORKS[NETWORK]` in the visualisation notebooks), so those notebooks showed only the demo network as input;
4. a path held in a variable assigned in one cell and opened in another, and f-strings with run-time values (`top_retweets_by_topic_{K}.csv`).

Rather than widen the parser, the script now accepts **hand-declared edges** (`manual_edges:` in the overrides file; each entry names the notebook, the Drive path, the direction and a note saying which cell produces it). 33 such edges were added, every one checked against the notebook source quoted earlier in this document. They carry `manual: true` in the JSON, so a reader can always tell a parsed edge from a declared one:

```bash
jq '[.edges[] | select(.manual)] | length' docs/pipeline_graph.json
```

Notebook ids may now carry a letter suffix (`02b`, `01b`) in provenance tags, so the corrected copies are validated like any other notebook.

### 10.3 Hand-off table for the network stage

Each row was checked twice: against the notebook source (§§4–5, §9) and against the rebuilt graph (`upstream` / `downstream` queries).

| Producer | File (in `Data Sets/Networks/`) | Consumer(s) | Edge source |
|---|---|---|---|
| `02/02` cell 44 | `full_network_dict.pkl` | `02/02` cell 51, `02b` | parsed |
| `02/02` cell 53 | `Full_Network.gml` | `04/01` (all three strategies) | parsed |
| `02b` | `Full_Network_Influence.gml`, `Full_Network_InfoFlow.gml` | `01b` (optional inspection, strategy 2), `02b-viz` registry | declared |
| `02b` | `Full_RetweetedOnce_Influence.gml` | `01b` (backbone check; undirected methods; directed Leiden), `02b-viz` registry | declared |
| `02b` | `Full_RetweetedOnce_InfoFlow.gml` | `01b` (Infomap) | declared |
| `01b` | `Full_RetweetedOnce_Influence_{label_propagation,louvain,leiden_fast,leiden_directed}.gml`, `Full_RetweetedOnce_InfoFlow_infomap.gml` | `01b` (AMI/ARI, JSON export), `02b-viz` | declared |
| `01b` | `Full_RetweetedOnce_author_communities.json` | *(no consumer yet — the author set for downstream restriction)* | declared (parsed until 2026-10-09) |
| `02b` | `Full_LWCC_Influence.gml`, `Full_LWCC_InfoFlow.gml` — *added 2026-10-09, not yet on Drive* | `01b` (backbone check; undirected methods + directed Leiden on Influence; Infomap on InfoFlow), `02b-viz` registry | declared |
| `01b` | `Full_LWCC_Influence_{label_propagation,louvain,leiden_fast,leiden_directed}.gml`, `Full_LWCC_InfoFlow_infomap.gml` — *not yet on Drive* | `01b` (AMI/ARI, JSON export), `02b-viz` | declared |
| `01b` | `Full_LWCC_author_communities.json` — *not yet on Drive* | *(no consumer yet)* | declared |
| `04/01` cell 18 | `Final_OutThreshold1.gml` — **superseded** | `04/01` community cells | parsed |
| `04/01` cells 21, 23 | `Final_OutThreshold1_<method>.gml` — **superseded** | `04/01` AMI + export, `04/02` | declared |
| `04/01` cell 27 | `Final_OutThreshold1_author_communities.json` — **superseded** | *(none in repo; §9.4 for the blog branch)* | parsed |

### 10.4 Caveats when reading the graph

- **Alternative edges are bidirectional**, and the `upstream` / `downstream` walkers follow them, so a query on `01b` also lists `04/01` and its outputs (and vice versa). Read the notebook list of a query with the alternatives in mind, or inspect the JSON and drop `direction == "alternative"` edges.
- The `BASE_PATH divergence` warning is expected: `03/01b` and `03/04c` run on the HPC path, every other notebook on Drive. The parser strips both prefixes, so artifacts still line up by relative path.
- The parser records what the *code* can write, not what *exists* on Drive. A file can be in the graph and absent on disk (e.g. before `02b` finishes), or on disk and absent from the graph (first-generation files such as `AItrust_pruned_twits.json`, §6.3, whose producing cells no longer exist).

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
| Both networks + retweeted-once backbone | `02_Processing/02b` | done 2026-10-07 |
| Five partitions + community JSON on the backbone | `04_Network_Analysis/01b` | done 2026-10-07 |
| Social-media maps | `04_Network_Analysis/02b` | run of 2026-10-08 failed before layout (cu13 cuGraph install); patched 2026-10-09; pending |
| LWCC backbone in both directions | `02_Processing/02b` (extension of 2026-10-09) | pending |
| Five partitions + community JSON on the LWCC backbone | `04_Network_Analysis/01b` | pending |
| AI+Art sentiment file without the surplus block | `03/01b` cell 34 (delete the `art_full__twitter-roberta-base-sentiment-latest` blocks, re-merge) | open |

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

- **Which population the blog is about.** On the retweeted-once backbone the matched set will be at most 202,710 authors and in practice fewer (only those with enough text for a topic profile). That is a study of *amplified* authors. If the intended subject is "the public reaction", a direction-neutral backbone with "retweeted ≥ 1" as a reported filter (§9.3) may fit the prose better. Decide before rebuilding the matched set, because every downstream number depends on it. Since 2026-10-09 both backbones are produced and clustered in the same `02b` / `01b` run (§9.3), so the decision can be made on the two partitions side by side rather than before clustering.
- **Whether the community labels carry over.** They were validated for specific memberships. A new partition on a different author set needs the c-TF-IDF and audit steps rerun; reusing the 21 names would be a claim without evidence.

## 12. File inventory — every artifact of the pipeline

*Added 2026-10-09.* Every file the notebooks read or write, derived from [docs/pipeline_graph.json](pipeline_graph.json) (164 artifacts after the 2026-10-09 rebuild: parsed edges plus the declared ones of `pipeline_overrides.yaml`). Location is the folder, relative to `BASE_PATH` = `My Drive/Colab Projects/AI Public Trust/` unless marked HPC or Colab-local; *written by* / *read by* are notebook ids (`stage/index`, `b` = corrected copy); templated names (`{…}`) are resolved at run time. Descriptions are from the notebooks and §§1–9 of this document. A file with no writer is produced outside the repository or by a cell that no longer exists; a file with no reader is a terminal output. Legacy files that no notebook reads or writes any more (`Test_Network.*`, `Full_Network.graphml/.gexf/.json`, written by the pre-2026-10 `02/02`) still sit on Drive but are not listed. To regenerate the producer/consumer columns after a notebook change, follow §10.5 and re-derive this table from the JSON.

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
| `AItrust_pruned_twits.json` | — | `03/01` | First-generation AI corpus (22.4M-line generation, §6.3). No producing cell survives; read by sentiment v2. |
| `AItrust_pruned_twits_classified_{MODEL_ALIAS}.json` | `03/01` | `03/01` | Sentiment v2 output per model alias on the first-generation corpus. |
| `AItrust_pruned_twits_test.json` | — | `03/01` | First-generation test corpus; read by sentiment v2. |
| `AItrust_pruned_twits_test_classified_{MODEL_ALIAS}.json` | `03/01` | `03/01` | Sentiment v2 test output per model alias. |
| `AItrust_pruned_twits_test_with_sentiment.json` | `03/01` | `03/01` | Sentiment v1 test output (older file name). |
| `AItrust_pruned_twits_with_sentiment.json` | `03/01` | `02/03`, `03/01` | AI corpus with CardiffNLP sentiment (v1, `03/01`) — first-generation corpus (§6.1, §6.3); input of `02/03` and of the tweet-topic notebook. |
| `AItrust_pruned_twits_with_sentiment.jsonl` | — | `03/03` | Line-delimited copy of the v1 sentiment file read by the tweet-topic LDA (`03/03`); no producing cell in the repository. |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5.json` | `03/03` | — | Tweets with sentiment and K = 5 LDA topic weights (`03/03`). |
| `AItrust_pruned_twits_with_sentiment_and_topics_k5.jsonl.gz` | `03/03` | `03/03` | Compressed line-delimited version of the same, read back by `03/03`. |
| `AItrust_pruned_twits_with_sentiment_cleaned.json` | `02/03` | — | The v1 sentiment file after the `02/03` text-cleaning pass. |
| `AItrust_topics_k5_metadata.json` | `03/03` | — | Top terms and settings of the K = 5 tweet-topic model. |
| `AItrust_twits_pruned_dict.json` | `02/02` | `02/02`, `03/01` | **AI corpus**: 17,410,035 tweets after dedup + AI keyword + English + date ≥ 2022-10-31, with `processed_text` added (§2). |
| `AItrust_twits_pruned_dict_test.json` | `02/02` | `02/02`, `03/01`, `03/02` | Test-branch AI corpus (881 tweets). |
| `AItrust_twits_pruned_dict_test_with_sentiment.json` | `03/01` | — | Sentiment v1 test output (current file name). |
| `Cleaned DataAItrust_pruned_twits.json` | — | `03/05` | **Path bug**: `03/05` joins `cleanedds_folder` and the file name without a separator, so this path can never resolve. Intended: `Cleaned Data/AItrust_pruned_twits.json`. |
| `Cleaned DataAItrust_pruned_twits_test.json` | — | `03/05` | Same path bug, test file. |
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
| `top_retweets_by_topic_100.csv` | `02/03` | — | The 100 most-retweeted tweets per topic (`02/03`). |
| `top_retweets_by_topic_{K}.csv` | `02/03` | — | Templated version of the same for other K. |
| `{BLOCK_BASENAME}{block_idx}.json` | `03/03` | — | Per-block intermediate outputs of the tweet-topic inference (`03/03`; names resolved at run time). |
| `{OUT_STEM}.jsonl` | `03/03` | — | Merged line-delimited output of the tweet-topic inference (`03/03`; name resolved at run time). |

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

| File | Written by | Read by | What it is |
|---|---|---|---|
| `90TS_LWCC.gml` | `04/01`, `04/01b` | — | Strategy 2: drop lowest total-strength nodes until 10 % of weight is lost, then LWCC — 2,315,573 authors (§5). `01b` keeps the recompute cell behind `RUN_STRATEGY_2`. |
| `90TS_LWCC.graphml` | `04/01`, `04/01b` | — | GraphML copy of strategy 2. |
| `Final_OutThreshold1.gml` | `04/01` | — | **Superseded** strategy 3 of `04/01`: out-strength ≥ 1 then LWCC — 1,984,599 *retweeters*, 4,472,376 edges (§5, §9.1). |
| `Final_OutThreshold1.graphml` | `04/01` | — | GraphML copy of the superseded backbone. |
| `Final_OutThreshold1_author_communities.json` | `04/01` | — | Superseded `{author_id: {method: community}}` export for the 1,984,599 retweeters; the blog branch's `leiden_directed` labels descend from it (§9.4). |
| `Final_OutThreshold1_infomap.gml` | `04/01` | `04/01` | Superseded retweeter backbone with the `community_infomap` attribute (`04/01`). Infomap ran on the influence orientation, i.e. against information flow. |
| `Final_OutThreshold1_label_propagation.gml` | `04/01` | `04/01` | Superseded retweeter backbone with the `community_label_propagation` attribute (`04/01`). |
| `Final_OutThreshold1_leiden_directed.gml` | `04/01` | `04/01` | Superseded retweeter backbone with the `community_leiden_directed` attribute (`04/01`). |
| `Final_OutThreshold1_leiden_fast.gml` | `04/01` | `04/01`, `04/02` | Superseded retweeter backbone with the `community_leiden_fast` attribute (`04/01`). |
| `Final_OutThreshold1_louvain.gml` | `04/01` | `04/01` | Superseded retweeter backbone with the `community_louvain` attribute (`04/01`). |
| `Full_LWCC_author_communities.json` | `04/01b` | — | Same for the LWCC backbone (3.26M authors); pending. |
| `Full_LWCC_Influence.gml` | `02/02` | `04/01b` | **LWCC backbone**: self-loops removed, largest weakly connected component of the full graph — expected 3,264,499 authors, 7,670,516 edges (= strategy 1); influence orientation. Added 2026-10-09, **not yet on Drive**. |
| `Full_LWCC_Influence_label_propagation.gml` | `04/01b` | `04/01b` | LWCC backbone with the `community_label_propagation` vertex attribute; `01b`, pending. |
| `Full_LWCC_Influence_leiden_directed.gml` | `04/01b` | `04/01b` | LWCC backbone with the `community_leiden_directed` vertex attribute; `01b`, pending. |
| `Full_LWCC_Influence_leiden_fast.gml` | `04/01b` | `04/01b` | LWCC backbone with the `community_leiden_fast` vertex attribute; `01b`, pending. |
| `Full_LWCC_Influence_louvain.gml` | `04/01b` | `04/01b` | LWCC backbone with the `community_louvain` vertex attribute; `01b`, pending. |
| `Full_LWCC_InfoFlow.gml` | `02/02` | `04/01b` | Transpose of the LWCC backbone; the Infomap input. Not yet on Drive. |
| `Full_LWCC_InfoFlow_infomap.gml` | `04/01b` | `04/01b` | LWCC backbone, flow orientation, with the Infomap partition; `01b`, pending. |
| `Full_Network.gml` | — | `04/01` | **Legacy** full retweet graph written by the pre-2026-10 version of `02/02`: edge retweeter → retweeted, weight = retweets, 3,379,040 authors, 7,768,720 edges (§4); identical to `Full_Network_Influence.gml`. Node name = `str(author_id)` stored as GML `label`; GML `id` is a positional index. |
| `full_network_dict.pkl` | `02/02` | `02/02` | Nested counter `{retweeted_author: {retweeter: n}}` for the AI corpus: 374,368 retweeted authors, 9,638,407 retweets (§3). |
| `Full_Network_Influence.gml` | `02/02` | `04/01b` | **Full retweet graph**, influence orientation (retweeter → retweeted); identical to the legacy `Full_Network.gml` (3,379,040 authors, 7,768,720 edges). |
| `Full_Network_InfoFlow.gml` | `02/02` | — | `02` full graph, information-flow orientation (the transpose; same counts). |
| `Full_RetweetedOnce_author_communities.json` | `04/01b` | — | `{author_id: {method: community_id}}` for the 202,710 backbone authors and all five methods (`01b`); the author set for downstream restriction. |
| `Full_RetweetedOnce_Influence.gml` | `02/02` | `04/01b` | **Retweeted-once backbone**: self-loops removed, in-strength ≥ 1 (363,618 authors), then LWCC — **202,710 authors, 882,530 edges, weight 1,269,747** (13.17 % of retweets); influence orientation (§9.3). |
| `Full_RetweetedOnce_Influence_label_propagation.gml` | `04/01b` | `04/01b` | Retweeted-once backbone with the `community_label_propagation` vertex attribute (read back as `communitylabelpropagation`, float → int); `01b`, 2026-10-07. |
| `Full_RetweetedOnce_Influence_leiden_directed.gml` | `04/01b` | `04/01b` | Retweeted-once backbone with the `community_leiden_directed` vertex attribute (read back as `communityleidendirected`, float → int); `01b`, 2026-10-07. |
| `Full_RetweetedOnce_Influence_leiden_fast.gml` | `04/01b` | `04/01b`, `04/02b` | Retweeted-once backbone with the `community_leiden_fast` vertex attribute (read back as `communityleidenfast`, float → int); `01b`, 2026-10-07. |
| `Full_RetweetedOnce_Influence_louvain.gml` | `04/01b` | `04/01b` | Retweeted-once backbone with the `community_louvain` vertex attribute (read back as `communitylouvain`, float → int); `01b`, 2026-10-07. |
| `Full_RetweetedOnce_InfoFlow.gml` | `02/02` | `04/01b` | Transpose of the retweeted-once backbone; the Infomap input. |
| `Full_RetweetedOnce_InfoFlow_infomap.gml` | `04/01b` | `04/01b` | Retweeted-once backbone, flow orientation, with the Infomap partition (17,055 modules); `01b`, 2026-10-07. |
| `LWCC.gml` | `04/01` | — | Strategy 1 of `04/01`: full graph, self-loops removed, giant component — 3,264,499 authors, 7,670,516 edges (igraph-written; §5). Same graph as `Full_LWCC_Influence.gml`. |
| `LWCC.graphml` | `04/01` | — | GraphML copy of strategy 1. |
| `Test_LWCC_Influence.gml` | `02/02` | — | `02` test LWCC backbone, influence orientation (added 2026-10-09). |
| `Test_LWCC_InfoFlow.gml` | `02/02` | — | Transpose of the test LWCC backbone. |
| `test_network_dict.pkl` | `02/02` | `02/02` | Nested counter `{retweeted_author: {retweeter: n}}` for the test corpus (§3). |
| `Test_Network_Influence.gml` | `02/02` | — | Test graph, influence orientation (retweeter → retweeted); identical to the legacy `Test_Network.gml`. |
| `Test_Network_InfoFlow.gml` | `02/02` | — | `02` test graph, information-flow orientation (retweeted → retweeter): the transpose. |
| `Test_RetweetedOnce_Influence.gml` | `02/02` | — | `02` test backbone: in-strength ≥ 1 then LWCC (3 nodes — the test data is too small to carry one). |
| `Test_RetweetedOnce_InfoFlow.gml` | `02/02` | — | Transpose of the test backbone. |

### 12.8 `Data Sets/Networks/viz_outputs_<NETWORK>/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Final_leiden_fast/demo_network.gml` | `04/02` | `04/02` | Synthetic stochastic-block-model fallback written by `04/02` only when the selected `.gml` is missing, so Run-all still completes. |
| `Final_leiden_fast/network_with_layout.graphml` | `04/02` | — | Optional GraphML with x/y and community for Gephi (`EXPORT_GRAPHML`, ~1.2 GB at 2M nodes; `04/02`). |
| `Final_leiden_fast/positions_drl.parquet` | `04/02` | — | Node table of the igraph DrL layout (`04/02`; the DrL cell imports `src.network.network_utils` and needs the repo on `sys.path`). |
| `Final_leiden_fast/positions_{_recipe}.parquet` | `04/02` | — | Node table per ForceAtlas2 recipe (`classic`, `linlog`): x, y, label, degree, in-degree, community, rank (`04/02`). The maps themselves — `social_map_<recipe>_<light|dark>.png`, `05_degree_ccdf.png`, `06_adjacency_blocks.png` — are written beside it by `savefig`, which the parser does not record. |
| `RetweetedOnce_leiden_fast/demo_network.gml` | `04/02b` | `04/02b` | Synthetic stochastic-block-model fallback written by `04/02b` only when the selected `.gml` is missing, so Run-all still completes. |
| `RetweetedOnce_leiden_fast/network_with_layout.graphml` | `04/02b` | — | Optional GraphML with x/y and community for Gephi (`EXPORT_GRAPHML`, ~1.2 GB at 2M nodes; `04/02b`). |
| `RetweetedOnce_leiden_fast/positions_drl.parquet` | `04/02b` | — | Node table of the igraph DrL layout (`04/02b`; the DrL cell imports `src.network.network_utils` and needs the repo on `sys.path`). |
| `RetweetedOnce_leiden_fast/positions_{_recipe}.parquet` | `04/02b` | — | Node table per ForceAtlas2 recipe (`classic`, `linlog`): x, y, label, degree, in-degree, community, rank (`04/02b`). The maps themselves — `social_map_<recipe>_<light|dark>.png`, `05_degree_ccdf.png`, `06_adjacency_blocks.png` — are written beside it by `savefig`, which the parser does not record. |

### 12.9 `Models/Topic Modeling/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `bow_test_corpus.pkl` | `03/06` | `03/06` | Bag-of-words test corpus of the topic-modelling appendix (`03/06`). |
| `full_sentences_corpus_embedding.pkl` | `03/05` | `03/05` | Sentence-transformer embeddings of the full corpus (`03/05`). |
| `hf_embeddings.npy` | `03/05` | `03/05` | Hugging Face embeddings array (`03/05`). |
| `LDA/author_level/ALL_REPRESENTATIONS_LDA_FULL_GRID.csv` | — | `03/04` | Author-LDA v1 grid results over representations × K × α × η (`03/04`; computed on the wrong author set, §6.4). |
| `LDA/author_level/ALL_REPRESENTATIONS_LDA_SUMMARY.csv` | — | `03/04` | Summary of the v1 grid. |
| `LDA/author_level/tfidf_unigram/author_topic_matrix_tfidf_unigram_k{K}_a{a_tag}_e{e_tag}.csv` | — | `03/04` | v1 author × topic matrix (θ) for one grid point. |
| `LDA/author_level/tfidf_unigram/author_topics_tfidf_unigram_k{K}_a{a_tag}_e{e_tag}_top_terms.csv` | — | `03/04` | v1 top terms per topic for one grid point. |
| `LDA/author_level/tfidf_unigram/lda_model_k16_a0p1_e0p1.pkl` | — | `03/04` | v1 fitted LDA model, K = 16 (read). |
| `LDA/author_level/tfidf_unigram/lda_model_k{K}_a0p1_e0p1.pkl` | `03/04` | — | v1 fitted LDA models per K (written). |
| `LDA/author_level/tfidf_unigram/vectorizer_k16_a0p1_e0p1.pkl` | — | `03/04` | v1 TF-IDF vectorizer, K = 16 (read). |
| `LDA/author_level/tfidf_unigram/vectorizer_k{K}_a0p1_e0p1.pkl` | `03/04` | — | v1 TF-IDF vectorizers per K (written). |
| `LDA/authorlite_k5_topics_metadata.json` | `03/03` | — | Metadata of the K = 5 "author-lite" topic model (`03/03`). |
| `LDA/authorlite_k{K_TARGET}_topics_metadata.json` | — | `03/03` | Templated read of the same for the configured K. |
| `LDA/lda_k5_topics_metadata.json` | `03/03` | — | Metadata of the K = 5 tweet-topic LDA (`03/03`). |
| `processed_sentence_transformer_embeddings.npy` | `03/05` | `03/05` | Post-processed sentence-transformer embeddings (`03/05`). |
| `sentence_transformer_reduced_embeddings.npy` | `03/05` | `03/05` | Dimensionality-reduced embeddings for the map (`03/05`). |
| `test_sentences_corpus.pkl` | — | `03/06` | Test sentence corpus read by the appendix (no producer in the repository). |
| `test_sentences_corpus_embedding.pkl` | `03/05` | `03/05` | Sentence-transformer embeddings of the test corpus (`03/05`). |

### 12.10 `HPC: /projects/ComputationalPhilosophyLab/TwitterDataAnalysis/Data Sets/`

| File | Written by | Read by | What it is |
|---|---|---|---|
| `Cleaned Data/ai_full_classified_{alias}.json` | `03/01b` | `03/01b` | v4 merged per-model output for the AI corpus. |
| `Cleaned Data/AI_pruned_tweets_with_topic_weights_v2.json` | `03/04c` | `03/04c` | Author-LDA v2: tweets with topic weights (`03/04c`; on the wrong author set, §6.4). |
| `Cleaned Data/ai_test_classified_{model_alias}.json` | `03/01b` | `03/01b` | v4 test output per model, AI corpus. |
| `Cleaned Data/AItrust_twits_pruned_dict.json` | — | `03/04c` | HPC copy of the AI corpus (17,410,035 tweets), read by author-LDA v2. |
| `Cleaned Data/art_full_classified_{alias}.json` | `03/01b` | `03/01b` | v4 merged per-model output for the AI+Art corpus (the sentiment file holds 474,239 surplus lines, §6.2). |
| `Cleaned Data/art_test_classified_{model_alias}.json` | `03/01b` | `03/01b` | v4 test output per model, AI+Art corpus. |
| `Cleaned Data/author_sentiment_mean.pkl` | `03/04c` | — | Author-level mean sentiment computed in `03/04c`. |
| `Cleaned Data/Blocks/ai_full__{model_name.split('/')[-1]}/block_{bid}.json` | `03/01b` | `03/01b` | Sentiment/emotion v4: per-model, per-block inference outputs over the AI corpus (`03/01b`, 10 models; block 0 of the emotion model is short by 8,975 lines, §6.2). |
| `Networks/LWCC.gml` | — | `03/04c` | HPC copy of `LWCC.gml`; `03/04c` filters authors on its GML `id` field instead of `label` (§6.4). |

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

| File | Written by | Read by | What it is |
|---|---|---|---|
| `AItrust_pruned_twits_with_sentiment.json` | — | `03/04` | Colab-local copy of the v1 sentiment file read by author-LDA v1 (`03/04`); exists only inside a running Colab session. |

