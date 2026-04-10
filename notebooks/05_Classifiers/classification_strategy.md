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
| **Final Inference** | Remainder | Classified by the final model for the main paper |

---

## Workflow

### Step 0 — Initial Labelling (once)

10 000 tweets are sampled **at random** from the Base partition and exported to `hitl_review_batch_00.csv`. A human annotates the `human_label` column.  
This is the seed training set; no model prediction is needed at this stage.

---

### Step 1 — Train (repeated each iteration)

All available human-labeled CSVs are loaded and merged. A suite of classifiers is trained and timed:

| Model | Embedding | Notes |
| :--- | :--- | :--- |
| Logistic Regression | TF-IDF (25k features) | Fast baseline |
| LightGBM | TF-IDF (25k features) | Strong tree-based baseline |
| **Twitter-RoBERTa** | Transformer (128 tokens) | `cardiffnlp/twitter-roberta-base`, pre-trained on 58 M tweets — primary model |

Training time and inference time are logged for every model to assist with cost/quality trade-offs on subsequent iterations.

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

### Step 4 — Final Inference

Once the iterative loop is complete and the model quality is satisfactory, the finalised Twitter-RoBERTa model is applied to the **entire Final Inference partition**. Results are merged with all human-labeled data into a single annotated file:

```
Data Sets/Classifiers_Data/Final/final_annotated_tweets.pkl
Data Sets/Classifiers_Data/Final/final_annotated_tweets.csv
```

This annotated dataset feeds into `03_Analysis_and_Modeling` and `04_Network_Analysis`.

---

## Notebooks

| Notebook | Run when |
| :--- | :--- |
| `01_hitl_data_preparation.ipynb` | **Once** — partitions the data and creates the first labelling CSV |
| `02_hitl_training_loop.ipynb` | **After each labelling round** — trains models and exports the next review batch |
| `03_final_inference.ipynb` | **Once at the end** — classifies the full remaining dataset |

---

## Key Decisions

- **Why Twitter-RoBERTa over generic BERT?** `cardiffnlp/twitter-roberta-base` was specifically pre-trained on Twitter data (informal language, hashtags, mentions, abbreviations), making it significantly better at tweet classification than domain-general transformers.
- **Why not LLaMA?** Large generative LLMs (7B+ parameters) require dedicated GPU infrastructure and LoRA/PEFT fine-tuning pipelines. RoBERTa (125M parameters) achieves excellent results within standard Colab GPU limits.
- **Why 5k random + 5k uncertain?** Pure uncertainty sampling can introduce bias by clustering around a narrow slice of the input space. Mixing in a random sample ensures the training set remains diverse.
