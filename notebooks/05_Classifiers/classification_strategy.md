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

The full **cleaned tweets dataset** (output of `02_Processing/03_cleaning_tweets.ipynb`) is shuffled once with a fixed random seed and divided into three non-overlapping partitions:

| Partition | Size | Purpose |
| :--- | :--- | :--- |
| **Base** | ~100 000 tweets | Source of the initial ground-truth labels and training data |
| **HITL Batches** | ~200 000 tweets (4 × 50 000) | Used for iterative human-in-the-loop review |
| **Final Inference** | Remainder (~300k) | Classified by the final model, merged with the labeled data |

The **remaining ~17 million tweets** in the full corpus are then classified separately in notebook 04 using the final trained model.

---

## Workflow

### Step 0 — Initial Labelling (once)

10 000 tweets are sampled **at random** from the Base partition and exported to `hitl_review_batch_00.csv`. A human annotates the `human_label` column.  
This is the seed training set; no model prediction is needed at this stage.

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
| `01_hitl_data_preparation.ipynb` | **Once** — partitions the data and creates the first labelling CSV |
| `02_hitl_training_loop.ipynb` | **After each labelling round** — trains all models and exports the next review batch |
| `03_final_inference.ipynb` | **Once** — classifies the HITL remainder and merges with labeled data |
| `04_full_dataset_inference.ipynb` | **Once** — classifies the entire ~17M tweet corpus |

---

## Data Folder Structure

```
Data Sets/
└── Classifiers_Data/
    ├── HITL/
    │   ├── base_dataset.pkl
    │   ├── inference_dataset.pkl
    │   ├── hitl_pending_batch_01.pkl  ← ... 04.pkl
    │   ├── hitl_review_batch_00.csv   ← human-labeled seed
    │   └── hitl_review_batch_01.csv   ← ... 04.csv (filled after each loop)
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

- **Why Twitter-RoBERTa over generic BERT?** `cardiffnlp/twitter-roberta-base` was specifically pre-trained on Twitter data (informal language, hashtags, mentions, abbreviations), making it significantly better at tweet classification than domain-general transformers.
- **Why not LLaMA?** Large generative LLMs (7B+ parameters) require dedicated GPU infrastructure and LoRA/PEFT fine-tuning pipelines. RoBERTa (125M parameters) achieves excellent results within standard Colab GPU limits.
- **Why 5k random + 5k uncertain?** Pure uncertainty sampling can introduce bias by clustering around a narrow slice of the input space. Mixing in a random sample ensures the training set remains diverse.
- **Why the native LightGBM API (`lgb.train`) instead of the sklearn wrapper?** The native API gives direct control over `lgb.Dataset` construction, the params dict, and `num_boost_rounds`, matching the approach in `Classifiers_Training_Final.ipynb` and enabling finer-grained model inspection.
- **Why checkpoint saves in notebook 04?** Classifying 17M tweets on Colab can take many hours. Google Colab sessions disconnect unpredictably; saving every 100k tweets means at most ~100k tweets of work is lost.
