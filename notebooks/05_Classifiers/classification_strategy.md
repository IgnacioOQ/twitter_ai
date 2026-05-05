# Classification Strategy
- status: active
- type: guideline
- id: classification_strategy
- last_checked: 2026-04-10
<!-- content -->

This document describes the classification workflow used in this project to label the full AI-Twitter dataset using a **Human-in-the-Loop (HITL) Active Learning** strategy.

---

## Overview

The goal is to train a text classifier on tweets and use it to annotate the entire dataset. Rather than labelling data randomly, we use an iterative active learning loop where a human reviews the predictions the model is *least confident about*, feeding corrections back into the training set. This maximises label quality while minimising human effort.

---

## Dataset Partitioning

The **pruned tweets dataset** at `cleanedds_folder / 'AItrust_twits_pruned_dict.json'` (JSONL output of `02_Processing/02_sanity_check_and_network_generation.ipynb`) is loaded by `00_hitl_data_preparation.ipynb`, then shuffled once with an **attention-weighted permutation** — each tweet's selection probability is proportional to `(likes + retweets + 1) ** SAMPLING_ALPHA` (default `0.5`, i.e. square-root smoothing). The dataframe is then sliced into four non-overlapping partitions:

| Partition | Size | Purpose |
| :--- | :--- | :--- |
| **LLM Bootstrap** | ~10 000 tweets | Labelled by an LLM (Gemini) in `01_llm_bootstrap_labelling.ipynb` to seed the training set |
| **Base** | ~100 000 tweets | Reserve pool / source for any human seed labelling that supplements the LLM bootstrap |
| **HITL Batches** | ~200 000 tweets (4 × 50 000) | Used for iterative human-in-the-loop review |
| **Final Inference** | Remainder (~300k) | Classified by the final model, merged with the labelled data |

A `partition_ids.pkl` manifest is written alongside the partition files mapping each partition name to its tweet IDs, with an inline assertion that the partitions are pairwise disjoint. Any downstream notebook can load this manifest to verify which subset a tweet belongs to.

The **remaining ~17 million tweets** in the full corpus are classified separately in notebook 04 using the final trained model.

---

## Workflow

### Step 0 — Initial Labelling (once)

The seed training set is produced by one of two paths — they can also be combined.

**Path A — LLM Bootstrap (default).** The ~10 000 tweets in the **LLM Bootstrap** partition are labelled by an LLM (Gemini) in `01_llm_bootstrap_labelling.ipynb`. The notebook embeds the full per-category criteria in the prompt, parses a strict JSON response per tweet (`label`, `confidence`, `rationale`), retries on transient errors, checkpoints every 1 000 rows, and writes `llm_bootstrap_labels.csv` with the **same schema** as `hitl_review_batch_*.csv`. This CSV is read by `02_hitl_training_loop.ipynb` exactly like a human-labelled batch.

**Path B — Human Seed (alternative or supplement).** 10 000 tweets are sampled at random from the **Base** partition and exported to `hitl_review_batch_00.csv` by `00_hitl_data_preparation.ipynb`. A human annotates the `human_label` column. Run this in addition to Path A if you want a human-verified subset on top of the LLM labels.

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
id | text | likes | retweets | predicted_label | human_label
```

A human then fills in the `human_label` column and saves the file.

---

### Step 3 — Repeat

Steps 1–2 are repeated up to **4 times** (one per 50k batch), incorporating each new round of human labels into the growing training set. The expectation is that model accuracy improves with each iteration.

---

### Step 4 — Final Inference (HITL Remainder)

Once the iterative loop is complete and the model quality is satisfactory, the finalised Twitter-RoBERTa model is applied to the **Final Inference partition** (~300k tweets). Results are merged with all human-labeled data into a single annotated file:

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

| Notebook | Run when |
| :--- | :--- |
| `00_hitl_data_preparation.ipynb` | **Once, first** — partitions the data into LLM Bootstrap / Base / HITL / Inference and writes the `partition_ids.pkl` manifest |
| `01_llm_bootstrap_labelling.ipynb` | **Once, after `00`** — runs the LLM over the LLM Bootstrap partition to produce `llm_bootstrap_labels.csv` |
| `02_hitl_training_loop.ipynb` | **After each labelling round** — trains all models and exports the next review batch |
| `03_final_inference.ipynb` | **Once** — classifies the HITL remainder and merges with labelled data |
| `04_full_dataset_inference.ipynb` | **Once** — classifies the entire ~17M tweet corpus |

---

## Data Folder Structure

```
Data Sets/
└── Classifiers_Data/
    ├── HITL/
    │   ├── llm_bootstrap_dataset.pkl       ← ~10 000 tweet partition for LLM bootstrap
    │   ├── llm_bootstrap_labels.csv        ← LLM-labelled seed (HITL CSV schema)
    │   ├── llm_bootstrap_labels_full.pkl   ← labels + confidence + rationale
    │   ├── llm_bootstrap_checkpoint_*.pkl  ← intermediate saves every 1 000 tweets
    │   ├── partition_ids.pkl               ← {partition_name: [tweet_id, ...]} manifest
    │   ├── base_dataset.pkl
    │   ├── inference_dataset.pkl
    │   ├── hitl_pending_batch_01.pkl       ← ... 04.pkl
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

- **Why attention-weighted sampling instead of uniform random?** Tweet engagement on this corpus follows a heavy-tailed distribution — a small number of viral tweets carry most of the discourse. A uniform random sample of 10 000 tweets would consist almost entirely of low-engagement tweets, leaving the LLM and the classifier blind to the content that actually shapes the conversation. Weighting selection probability by `(likes + retweets + 1) ** SAMPLING_ALPHA` over-represents the influential head while still drawing from the body of the distribution. `SAMPLING_ALPHA = 0.5` (square-root smoothing) is the default; `0` recovers uniform, `1` is fully proportional to attention. The downstream slice (LLM Bootstrap → Base → HITL → Inference) is *stratified by influence* — the LLM Bootstrap subset carries the highest-engagement tweets so the LLM and the seed training set learn from the most informative content first.
- **Why bootstrap with an LLM instead of going straight to human seed labelling?** Hand-labelling 10 000 tweets is the most expensive single step in the original loop. An LLM with a well-specified prompt can produce the same 10 000 labels in well under an hour for a few dollars — good enough to seed the first round of HITL training. The human review then concentrates where it pays off most: correcting the model's uncertain predictions in Steps 2-3, not creating labels from scratch. The LLM bootstrap subset is **disjoint** from the Base partition, so a human seed CSV (Path B) can be added on top without double-labelling any tweet.
- **Why Twitter-RoBERTa over generic BERT?** `cardiffnlp/twitter-roberta-base` was specifically pre-trained on Twitter data (informal language, hashtags, mentions, abbreviations), making it significantly better at tweet classification than domain-general transformers.
- **Why not LLaMA?** Large generative LLMs (7B+ parameters) require dedicated GPU infrastructure and LoRA/PEFT fine-tuning pipelines. RoBERTa (125M parameters) achieves excellent results within standard Colab GPU limits.
- **Why 5k random + 5k uncertain?** Pure uncertainty sampling can introduce bias by clustering around a narrow slice of the input space. Mixing in a random sample ensures the training set remains diverse.
- **Why the native LightGBM API (`lgb.train`) instead of the sklearn wrapper?** The native API gives direct control over `lgb.Dataset` construction, the params dict, and `num_boost_rounds`, matching the approach in `Classifiers_Training_Final.ipynb` and enabling finer-grained model inspection.
- **Why checkpoint saves in notebook 04?** Classifying 17M tweets on Colab can take many hours. Google Colab sessions disconnect unpredictably; saving every 100k tweets means at most ~100k tweets of work is lost.
