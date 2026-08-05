# Classification Strategy
- status: active
- type: guideline
- id: classification_strategy
- last_checked: 2026-08-01
<!-- content -->

This document describes the classification workflow used in this project to label the full AI-Twitter dataset using a **Human-in-the-Loop (HITL) Active Learning** strategy.

---

## Overview

The goal is to train a text classifier on tweets and use it to annotate the entire dataset. Rather than labelling data randomly, we use an iterative active learning loop where a human reviews the predictions the model is *least confident about*, feeding corrections back into the training set. This maximises label quality while minimising human effort.

---

## Label Taxonomy

**`categories.md` is the source of truth for what the labels mean.** This document describes the *machinery*; that one describes the *label set* the machinery carries.

The taxonomy is currently **two labels**, deliberately:

| Label | Meaning |
| :--- | :--- |
| `originality` | The tweet appeals to originality — newness, creativity, copying, theft — **as a criterion for the value of art**. |
| `none` | Residual bucket: no category in the current taxonomy applies. |

Starting at one substantive category keeps the bootstrap's errors attributable — a wrong label traces to a specific clause of one definition rather than to an unknown interaction between several. `categories.md` carries the decision test, worked examples, exclusions, confidence calibration, and the checklist for adding category *N+1* (including the two things that must be decided at that point: what happens to existing `none` rows, and whether the single-label assumption still holds).

The prompt itself — scaffold and criteria together, in the form the model receives — is `llm_bootstrap_prompt.md` in this folder. Because notebook `01` does not clone this repo on Colab, no markdown here is readable at runtime, so that file is embedded in the notebook as a literal. **The embedding is mechanical: edit `llm_bootstrap_prompt.md`, then run `python3 notebooks/05_Classifiers/sync_prompt.py`.** `sync_prompt.py --check` exits non-zero while the notebook and the prompt file disagree. The notebook writes the same file back out next to `llm_bootstrap_labels.csv`, byte-identical, so the copy a reviewer reads is the copy in git.

---

## Dataset Type

Every notebook in this stage exposes a `DATASET_TYPE` constant (default `'AI'`) in its setup cell. The value selects which pruned corpus to consume and which subdirectory under `Cleaned Data/Partitioned Data/` to read and write:

| `DATASET_TYPE` | Pruned source (JSONL from `02_Processing/02`) | Partitioned outputs land in |
| :--- | :--- | :--- |
| `'AI'` | `AItrust_twits_pruned_dict.json` | `Cleaned Data/Partitioned Data/AI Data/` |
| `'Art'` | `AItrust_Art_pruned_twit_dict.json` | `Cleaned Data/Partitioned Data/Art Data/` |

Each corpus is partitioned, HITL-labelled, and classified end-to-end. Notebook `00` must be run **once per `DATASET_TYPE`** to seed the partition pickles for that corpus; notebooks `01`-`04` are then run per dataset, each with the matching `DATASET_TYPE` set. The trained model artifacts under `Models/Classifiers/` and the HITL labelling CSVs under `Classifiers_Data/HITL/` are **not** dataset-type-aware — the operator is expected to manage those artifacts between runs to avoid clobbering across corpora.

---

## Dataset Partitioning

The **pruned tweets dataset** for the active `DATASET_TYPE` (`AItrust_twits_pruned_dict.json` for AI, `AItrust_Art_pruned_twit_dict.json` for Art — both JSONL outputs of `02_Processing/02_sanity_check_and_network_generation.ipynb`) is loaded by `00_hitl_data_preparation.ipynb`. Each tweet carries `id`, `text`, `processed_text` (lowercase / URL-stripped / @-mention-stripped / RT-marker-stripped), `type` (one of `original`, `replied_to`, `quoted`, `retweeted`), and a nested `public_metrics` dict.

### Retweets are split out before partitioning

Retweets (`type == 'retweeted'`) are routed to a separate `retweets_dataset.pkl` and **never enter the train/test partitions**. A retweet's `text` is bit-for-bit identical to its referenced original's, so leaving retweets in would (1) leak the same string between training partitions and the held-out Inference partition, inflating measured generalisation, and (2) burn classifier compute predicting labels that are by construction the original's label. At merge time in Step 4, each retweet inherits its original's predicted (or human-confirmed) label via `referenced_tweets_dictionary` — no model call needed.

`replied_to` and `quoted` carry the user's own commentary text on top of a reference, so their text is genuinely new and they remain in the partitionable corpus alongside originals.

### The partitionable corpus

The partitionable corpus (`type ∈ {original, replied_to, quoted}`) is then shuffled with an **attention-weighted permutation** — each tweet's selection probability is proportional to `(likes + retweets + 1) ** SAMPLING_ALPHA` (default `0.5`, i.e. square-root smoothing). The dataframe is then sliced into four non-overlapping partitions:

| Partition | Size | Purpose |
| :--- | :--- | :--- |
| **LLM Bootstrap** | ~10 000 tweets | Labelled by an LLM (Gemini) in `01_llm_bootstrap_labelling.ipynb` to seed the training set |
| **Base** | ~100 000 tweets | Reserve pool / source for any human seed labelling that supplements the LLM bootstrap |
| **HITL Batches** | ~200 000 tweets (4 × 50 000) | Used for iterative human-in-the-loop review |
| **Final Inference** | Remainder of partitionable corpus | Classified by the final model, merged with the labelled data and the retweet-lookup labels |
| **Retweets** *(separate)* | All `type == 'retweeted'` rows | Bypass the model; labels inherited from referenced originals at Step 4 merge |

A `partition_ids.pkl` manifest is written to `Cleaned Data/Partitioned Data/{DATASET_TYPE} Data/`, mapping each partition name to its tweet IDs, with an inline assertion that all partitions are pairwise disjoint. Any downstream notebook can load this manifest to verify which subset a tweet belongs to.

The **remaining ~17 million tweets** in the full corpus are classified separately in notebook 04 using the final trained model.

---

## Workflow

### Step 0 — Initial Labelling (once)

The seed training set is produced by one of two paths — they can also be combined.

**Path A — LLM Bootstrap (default).** The tweets in the **LLM Bootstrap** partition are labelled by an LLM (Gemini) in `01_llm_bootstrap_labelling.ipynb`. The notebook embeds the full per-category criteria in the prompt, constrains the response with a native JSON schema (`label`, `confidence`, `rationale`) whose `label` enum is derived from `CATEGORIES`, retries on transient errors, and writes `llm_bootstrap_labels.csv` with the **same schema** as `hitl_review_batch_*.csv`. This CSV is read by `02_hitl_training_loop.ipynb` exactly like a human-labelled batch.

> **The partition is ~10 000 tweets but the notebook currently ships a hard cap of `MAX_LLM_TWEETS = 2_000` per run**, enforced independently of `SMOKE_TEST` at three layers (dataframe truncation, an assertion before the loop, and a per-call counter in `classify_tweet` that raises rather than exceed the budget). It caps *tweets*, not requests: with `N_LABEL_PASSES = 2` the request ceiling is 4 000. The cap is there because the prompt in `llm_bootstrap_prompt.md` is still being tuned; the full partition should not be spent on a definition that has not yet survived a read of its own low-confidence rows. Raising it is a deliberate edit to the Configuration cell.

> **`CHECKPOINT_EVERY` / `CHECKPOINT_PREFIX` are declared in the Configuration cell but never used** — the labelling loop writes nothing until it finishes. A disconnect partway through loses the whole run. The clean-stop paths (budget exceeded, quota exhausted, fatal request error) do save what completed; an unhandled kernel death does not.

**Which tweets get labelled, and the record of it.** Selection is a fixed permutation of the partition (sorted by `id`, permuted with `SELECTION_SEED`, sliced from the head), so a smoke run's tweets are a strict subset of a full run's and successive runs *extend* the labelled set rather than redraw it. Every id sent to the LLM is appended to `Classifiers_Data/HITL/llm_bootstrap_seen_ids_{DATASET_TYPE}.json`, which accumulates across runs.

> **This file, not `partition_ids.pkl`, defines the bootstrap training set.** The manifest records which tweets are *eligible* for bootstrap labelling; with the cap, only a fraction is actually labelled and the remaining ~9 000 of the partition stay unseen — legitimately usable as held-out evaluation data. Any downstream accuracy measurement must exclude the ids in the basket, or it is scoring the model on its own training data.

**Token accounting** (`TOKENOPT_REF.md` §16-17): a pre-flight cell projects tokens and cost before the loop and warns above `COST_ALERT_USD`; the run cell reports the billed `usage_metadata` including the implicit-cache hit rate; each run appends `llm_bootstrap_usage_<timestamp>.json`. Measured prompt size is ~4 389 chars (~1 100 input tokens), essentially all of it the static scaffold that is identical on every call and therefore implicit-cache eligible. The current smoke run (`SMOKE_TEST_N = 500`) issues 1 000 requests over two passes and measures ~1.1 M tokens; a run that reached the 2 000-tweet cap would issue 4 000 and measure ~4.4 M.

**Token ceiling.** `MAX_SESSION_TOKENS = 6_000_000` bounds total tokens per run independently of the row cap. It has to be sized against the *request* ceiling (`MAX_LLM_TWEETS × N_LABEL_PASSES` = 4 000 calls, ~5.6 M tokens at a padded 1 400 per call), not against the tweet count — sized below that it stops being a backstop and silently truncates runs. It fires only if per-call cost inflates unexpectedly. Enforcement is between calls (`usage_metadata` arrives with the response), so overshoot is bounded by a single call. **A breach stops the loop cleanly instead of raising:** completed rows are saved and basketed, and `stopped_early` is recorded in both the usage record and the basket's run log. The clean stop is load-bearing — `CHECKPOINT_EVERY = 1_000` means an uncaught exception partway through a capped run would otherwise discard the entire run.

The `llm_bootstrap_labels_full.pkl` companion file carries `confidence` and `rationale` alongside the labels, and is the artifact to read when tuning the criteria — the `rationale` instructs the model to quote the phrase that decided the label, so a disagreement is traceable to a specific clause.

**Prompt record for reviewers.** The Save section writes `llm_bootstrap_prompt.md` beside the CSV — the repo's own copy, byte for byte, asserted on write. The humans filling in `human_label` therefore read exactly the text the model was given, and the author tuning the prompt edits that same file in git. It carries no run metadata on purpose: a generated header would make the two copies differ, and then neither could be trusted as *the* prompt. Model, passes, per-pass temperatures, max output tokens, the effective thinking budget, whether the response schema was enforced, and a `sha256` of the prompt file go into `llm_bootstrap_usage_<timestamp>.json` instead. That fingerprint is what ties a CSV to the prompt version that produced it once the prompt has been tuned further — the decoding fields are read from the *effective* config rather than the Configuration cell, because the client cell probes `thinking_budget` and silently falls back when a model rejects it.

**Path B — Human Seed (alternative or supplement).** 10 000 tweets are sampled at random from the **Base** partition and exported to `hitl_review_batch_00.csv` by `00_hitl_data_preparation.ipynb`. The export is guarded by an `EXPORT_HUMAN_SEED` flag at the bottom of `00` (default `False` — skipped); set it to `True` and re-run that cell to produce the seed CSV. A human annotates the `human_label` column. Run this in addition to Path A if you want a human-verified subset on top of the LLM labels.

No model prediction is needed at this stage. The output of either path (or both) is the initial training set consumed by Step 1.

---

### Step 1 — Train (repeated each iteration)

All available human-labeled CSVs are loaded and merged. A suite of classifiers is trained and evaluated with explicit **training time** and **inference time** logging:

#### Embeddings

Two embedding strategies are used in parallel:

| Embedding | Tool | Notes |
| :--- | :--- | :--- |
| **Sentence Embeddings** | `SentenceTransformer('all-MiniLM-L6-v2')` | Dense semantic vectors, used by LightGBM and LogReg |
| **Bag of Words** | `CountVectorizer(max_features=50 000)` | Sparse token counts, used by LightGBM and LogReg |

#### Models

| Model | Embedding | Implementation | Notes |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Sentence + BoW | `sklearn.linear_model.LogisticRegression` | Fast baseline, run twice |
| **Boosted Tree (LightGBM)** | Sentence + BoW | Native API: `lgb.Dataset` + `lgb.train()` | 1 000 boost rounds, AUC metric, run twice — matches `Classifiers_Training_Final.ipynb` |
| **Twitter-RoBERTa** | Transformer (128 tokens) | HuggingFace `Trainer` | `cardiffnlp/twitter-roberta-base`, pre-trained on 58 M tweets — primary model |

Training time and inference time are logged for every model.

#### Model Storage

All trained models are saved to `BASE_PATH / 'Models/Classifiers/'`:

```
Models/Classifiers/best_roberta_model/   ← Twitter-RoBERTa weights + tokenizer
Models/Classifiers/lgb_embed.txt         ← LightGBM (sentence embeddings)
Models/Classifiers/lgb_bow.txt           ← LightGBM (bag of words)
```

---

### Step 2 — Active Learning Sample (repeated each iteration)

The best model is applied to the **next pending 50k batch**. From those predictions, exactly **10 000 tweets** are selected for human review:

- **5 000** tweets chosen *at random* (to avoid sampling bias)
- **5 000** tweets where the model's **confidence was lowest** (closest to the decision boundary — the ones the model is most uncertain about)

These 10 000 tweets are exported to `hitl_review_batch_XX.csv`, containing:

```
id | text | processed_text | type | likes | retweets | predicted_label | human_label
```

`text` is the raw tweet body (what the human reads to label); `processed_text` is the cleaned variant carried alongside so downstream training can pick whichever input the chosen model prefers. `type` is always one of `original`, `replied_to`, or `quoted` here — retweets are excluded from HITL review by construction. The human then fills in the `human_label` column and saves the file.

---

### Step 3 — Repeat

Steps 1–2 are repeated up to **4 times** (one per 50k batch), incorporating each new round of human labels into the growing training set. The expectation is that model accuracy improves with each iteration.

---

### Step 4 — Final Inference (HITL Remainder + Retweet Merge)

Once Step 3 is complete, every tweet's label is produced in three passes.

**Pass 1 — Classify the partitionable corpus.** Apply the finalised Twitter-RoBERTa model to the **Final Inference partition**. Combined with the HITL human labels and the LLM bootstrap labels, this gives a `labels` dict keyed by tweet ID — one label per *non-retweet* in the labelled corpus.

**Pass 2 — Promote orphan originals.** For each retweet in `retweets_dataset.pkl`, read `ref_id = row['referenced_tweets_dictionary']['id']`. Retweets where `ref_id ∉ labels` are **orphans**: their referenced original was filtered out by upstream pruning but their own row survived. Group the orphans by `ref_id`, pick one representative per group (highest `(likes + retweets)`, then lowest `id` to break ties deterministically), classify *only* the representative once, and write the prediction to a second dict:

```python
orphan_originals = { ref_id: predicted_label, ... }   # one entry per missing referenced original
```

A viral missing original retweeted 5 000 times needs one model call here, not 5 000 — and all 5 000 sibling retweets are guaranteed to inherit the same label by construction.

**Pass 3 — Look up retweet labels.** For every retweet:

```python
label = labels.get(ref_id) or orphan_originals.get(ref_id) or model_predict(row['text'])
```

The third branch is a safety net for retweets whose `referenced_tweets_dictionary` is empty or missing the `id` key — rare, but can happen if upstream parsing failed for the reference object. It is the only place we re-invoke the model per-row in Step 4.

**Anticipated edge cases:**

- **Empty `referenced_tweets_dictionary`.** Caught by the third branch above. Worth instrumenting: log the rate after the first run; if it exceeds ~1%, investigate the upstream JSONL rather than absorbing the cost silently.
- **Multiple references on a single tweet** (e.g. a tweet that is both a reply and a quote). The Twitter v2 API can attach more than one reference, but the upstream `02_sanity_check_and_network_generation.ipynb` already collapses these to a single `referenced_tweets_dictionary` per tweet, so we treat its `id` as authoritative here.
- **A `ref_id` that points to a tweet inside the partitionable corpus but not yet classified at merge time.** Cannot occur given Pass 1 ordering (every non-retweet is labelled before Pass 2 runs), but assert it explicitly so a future refactor can't silently break the invariant.
- **Cross-partition consistency.** Since all retweets in an orphan group read off the same `orphan_originals[ref_id]`, sibling retweets of the same missing original always receive identical labels. The same property holds trivially for the standard lookup path.

**Provenance.** A `label_source` column is written on every row of the final annotated file:

| Value | Meaning |
| :--- | :--- |
| `human` | HITL-labelled (`human_label` non-null after a review round) |
| `llm_bootstrap` | Gemini-labelled in `01_llm_bootstrap_labelling.ipynb` |
| `model_original` | Twitter-RoBERTa on a partitionable tweet (`type ∈ {original, replied_to, quoted}`) |
| `model_synthetic_retweet` | Twitter-RoBERTa on a representative retweet, used as the canonical label for a missing referenced original |
| `model_no_reference` | Twitter-RoBERTa on a retweet whose `referenced_tweets_dictionary` was empty / unparseable |
| `lookup` | Inherited from a labelled original or a synthetic original via `ref_id` (no model call) |

This lets downstream analysis report the share of labels from each source and audit the orphan handling without re-running anything.

**Output:**

```
Data Sets/Classifiers_Data/Final/final_annotated_tweets.pkl
Data Sets/Classifiers_Data/Final/final_annotated_tweets.csv
```

---

### Step 5 — Full Dataset Mass Inference (~17M tweets)

The final model is then applied to the **entire remaining tweet corpus**. This notebook:

- Loads models from `Models/Classifiers/`
- Processes tweets in configurable chunks (default 10 000) with `tqdm` progress bars
- Applies standard text cleaning (remove URLs, @mentions, RT markers)
- Saves checkpoints every 100 000 tweets to prevent data loss on Colab disconnects
- Saves final output to `Data Sets/Classifiers_Data/Full_Inference/full_inference_annotated.pkl`

This annotated corpus feeds into `03_Analysis_and_Modeling` and `04_Network_Analysis` for the main paper.

---

## Notebooks

All notebooks consume `DATASET_TYPE` (default `'AI'`); set it to `'Art'` to run the same pipeline against the Art corpus.

| Notebook | Run when |
| :--- | :--- |
| `00_hitl_data_preparation.ipynb` | **Once per `DATASET_TYPE`, first** — partitions the data into LLM Bootstrap / Base / HITL / Inference and writes the `partition_ids.pkl` manifest |
| `01_llm_bootstrap_labelling.ipynb` | **Once per `DATASET_TYPE`, after `00`** — runs the LLM over the LLM Bootstrap partition to produce `llm_bootstrap_labels.csv`. Reads its label set and criteria from `categories.md` (pasted in, see [Label Taxonomy](#label-taxonomy)); capped at `MAX_LLM_TWEETS` per run |
| `02_hitl_training_loop.ipynb` | **After each labelling round** — trains all models and exports the next review batch |
| `03_final_inference.ipynb` | **Once per `DATASET_TYPE`** — classifies the HITL remainder and merges with labelled data |
| `04_full_dataset_inference.ipynb` | **Once per `DATASET_TYPE`** — classifies the entire remaining corpus |

---

## Data Folder Structure

Partition outputs from notebook 00 live under `Cleaned Data/Partitioned Data/` (next to the upstream pruned tweet dict). Labelling artifacts and final inference outputs live under `Classifiers_Data/`.

```
Data Sets/
├── Cleaned Data/
│   ├── AItrust_twits_pruned_dict.json          ← upstream input from 02/02 (AI corpus)
│   ├── AItrust_Art_pruned_twit_dict.json       ← upstream input from 02/02 (Art corpus)
│   └── Partitioned Data/                       ← outputs of 00_hitl_data_preparation.ipynb
│       ├── AI Data/                            ← written when DATASET_TYPE='AI'
│       │   ├── llm_bootstrap_dataset.pkl       ← ~10 000 tweet partition for LLM bootstrap
│       │   ├── base_dataset.pkl
│       │   ├── inference_dataset.pkl
│       │   ├── retweets_dataset.pkl            ← retweets — labels inherited at merge time
│       │   ├── hitl_pending_batch_01.pkl       ← ... 04.pkl
│       │   └── partition_ids.pkl               ← {partition_name: [tweet_id, ...]} manifest
│       └── Art Data/                           ← written when DATASET_TYPE='Art' (same layout)
│           └── ...
│
└── Classifiers_Data/
    ├── HITL/
    │   ├── llm_bootstrap_labels.csv        ← LLM-labelled seed (HITL CSV schema)
    │   ├── llm_bootstrap_labels_full.pkl   ← labels + confidence + rationale
    │   ├── llm_bootstrap_prompt.md         ← the repo's prompt file, copied verbatim
    │   ├── llm_bootstrap_seen_ids_AI.json  ← ids the LLM has seen — EXCLUDE from eval
    │   ├── llm_bootstrap_usage_*.json      ← measured token/cost record, one per run
    │   ├── llm_bootstrap_checkpoint_*.pkl  ← intermediate saves every 1 000 tweets
    │   ├── hitl_review_batch_00.csv        ← optional human-labelled seed (Path B)
    │   └── hitl_review_batch_01.csv        ← ... 04.csv (filled after each loop)
    ├── Final/
    │   ├── final_annotated_tweets.pkl
    │   └── final_annotated_tweets.csv
    └── Full_Inference/
        ├── checkpoint_100000.pkl      ← intermediate saves
        ├── full_inference_annotated.pkl
        └── full_inference_annotated.csv

Models/
└── Classifiers/
    ├── best_roberta_model/
    ├── lgb_embed.txt
    └── lgb_bow.txt
```

---

## Key Decisions

- **Why split retweets out instead of routing them through the classifier?** A retweet's `text` is bit-for-bit identical to its referenced original (Twitter's `RT @user: <text>` is the same content). Two problems if we left retweets in the partitionable corpus: (1) **text leakage** — the same string could land in both a training partition and the held-out Inference partition, polluting our generalisation measurement; (2) **wasted compute** — running Twitter-RoBERTa on each retweet predicts a label that is by construction the original's label, with the added risk that the model labels a retweet differently from its original on a borderline case. Splitting retweets into `retweets_dataset.pkl` and inheriting their labels at merge time fixes both. `replied_to` and `quoted` tweets carry the user's own commentary text on top of a reference, so their text is genuinely new and they stay in the partitionable corpus.
- **Why promote one retweet per orphan group instead of running the classifier on every orphan?** A missing original can have anywhere from 1 to 50 000+ orphan retweets, all carrying identical text. Classifying each retweet independently wastes GPU time proportional to virality, and introduces a consistency risk: the model could land on different sides of the decision boundary on two near-identical inputs. Picking one representative per `ref_id`, classifying it once, and writing the prediction to `orphan_originals[ref_id]` reduces model passes by one to two orders of magnitude on viral missing originals and guarantees sibling consistency. The representative is chosen by descending `(likes + retweets)` and then ascending `id` so the choice is deterministic across reruns.
- **Why keep both `text` and `processed_text` in every partition?** The two embedding pipelines want different inputs. `cardiffnlp/twitter-roberta-base` was pre-trained on near-raw tweets (it expects @-mentions, hashtags, casing, even URLs as `<url>` placeholders) and performs best on `text`. Classical bag-of-words and sentence-embedding pipelines benefit from the cleaning already done by `02_sanity_check_and_network_generation.ipynb` (`processed_text`), which collapses noise that would otherwise blow up vocabulary size or perturb dense vectors. Carrying both columns means every downstream notebook picks the right input for its model without having to re-run the cleaner.
- **Why attention-weighted sampling instead of uniform random?** Tweet engagement on this corpus follows a heavy-tailed distribution — a small number of viral tweets carry most of the discourse. A uniform random sample of 10 000 tweets would consist almost entirely of low-engagement tweets, leaving the LLM and the classifier blind to the content that actually shapes the conversation. Weighting selection probability by `(likes + retweets + 1) ** SAMPLING_ALPHA` over-represents the influential head while still drawing from the body of the distribution. `SAMPLING_ALPHA = 0.5` (square-root smoothing) is the default; `0` recovers uniform, `1` is fully proportional to attention. The downstream slice (LLM Bootstrap → Base → HITL → Inference) is *stratified by influence* — the LLM Bootstrap subset carries the highest-engagement tweets so the LLM and the seed training set learn from the most informative content first.
- **Why bootstrap with an LLM instead of going straight to human seed labelling?** Hand-labelling 10 000 tweets is the most expensive single step in the original loop. An LLM with a well-specified prompt can produce the same 10 000 labels in well under an hour for a few dollars — good enough to seed the first round of HITL training. The human review then concentrates where it pays off most: correcting the model's uncertain predictions in Steps 2-3, not creating labels from scratch. The LLM bootstrap subset is **disjoint** from the Base partition, so a human seed CSV (Path B) can be added on top without double-labelling any tweet.
- **Why Twitter-RoBERTa over generic BERT?** `cardiffnlp/twitter-roberta-base` was specifically pre-trained on Twitter data (informal language, hashtags, mentions, abbreviations), making it significantly better at tweet classification than domain-general transformers.
- **Why not LLaMA?** Large generative LLMs (7B+ parameters) require dedicated GPU infrastructure and LoRA/PEFT fine-tuning pipelines. RoBERTa (125M parameters) achieves excellent results within standard Colab GPU limits.
- **Why 5k random + 5k uncertain?** Pure uncertainty sampling can introduce bias by clustering around a narrow slice of the input space. Mixing in a random sample ensures the training set remains diverse.
- **Why the native LightGBM API (`lgb.train`) instead of the sklearn wrapper?** The native API gives direct control over `lgb.Dataset` construction, the params dict, and `num_boost_rounds`, matching the approach in `Classifiers_Training_Final.ipynb` and enabling finer-grained model inspection.
- **Why checkpoint saves in notebook 04?** Classifying 17M tweets on Colab can take many hours. Google Colab sessions disconnect unpredictably; saving every 100k tweets means at most ~100k tweets of work is lost.
