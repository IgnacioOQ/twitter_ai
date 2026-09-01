# Classification Categories
- status: active
- type: guideline
- id: classification_categories
- last_checked: 2026-08-01
<!-- content -->

This document is the **source of truth for what the labels mean**. The prompt that puts those meanings in front of the model is `llm_bootstrap_prompt.md` in this folder — see [Where the criteria live](#where-the-criteria-live).

> **The sync obligation is now mechanical.** Notebook `01` does **not** clone this repo on Colab — it only mounts Drive — so no markdown in this folder is readable at runtime and the prompt has to be embedded in the notebook. That embedding is done by `sync_prompt.py`, not by hand, and `sync_prompt.py --check` fails while the notebook and the prompt file disagree.

---

## Label set

The pipeline uses a **multi-label classification** approach. A single tweet can invoke more than one of these conceptual frameworks simultaneously. The categories correspond to answers to the questions "What is art?" and "What makes art valuable?".

| Label | Meaning |
| :--- | :--- |
| `intentionalism` | Meaning/value is determined by the artist's intent. |
| `anti_intentionalism` | Meaning/value is independent of the artist's intent; set by language/imagery. |
| `cognitivism` | Art is valuable because it gives knowledge, understanding, or makes us think. |
| `expressivism` | Art is valuable because it expresses or evokes emotion. |
| `hedonism` | Art is valuable because of the pleasure or enjoyment it provides. |
| `originality` | Art is valuable because it is non-trivially new, creative, and not copied/stolen. |
| `achievement` | Art is valuable because it requires human effort, skill, and mastery. |
| `none` | No category in the current taxonomy applies. |

`none` is a **residual bucket.** It means "no category applied *as of the run that produced it*". 

---

## Where the criteria live

**The criteria are no longer duplicated here.** They live in `llm_bootstrap_prompt.md` in this
folder, which is the single authored copy of the prompt — scaffold and criteria together, in
the form the model actually receives.

That file is embedded into notebook `01` verbatim by `sync_prompt.py`, and written back out
next to `llm_bootstrap_labels.csv` unchanged, so the version a reviewer reads in Drive, the
version in the notebook, and the version in git are the same bytes.

To change what the model is told:

```bash
$EDITOR notebooks/05_Classifiers/llm_bootstrap_prompt.md    # edit inside the fence
python3 notebooks/05_Classifiers/sync_prompt.py             # embed it in the notebook
```

`sync_prompt.py --check` exits non-zero while the two disagree, which is the form to wire into
a pre-commit hook.

---

## Confidence and rationale are the interpretability surface

`01` writes the labels in two shapes. `llm_bootstrap_labels.csv` carries only the HITL schema (one-hot columns for multi-label); `llm_bootstrap_labels_full.pkl` additionally carries `confidence` and `rationale` for each predicted label. **The `.pkl` is the one to read when tuning this file.** The intended loop:

1. Run with `SMOKE_TEST = True` (capped — see below).
2. Open the `.pkl`, sort by `confidence` ascending, and read the bottom rows. Low-confidence rows are where the definition is underspecified.
3. Edit the fence in `llm_bootstrap_prompt.md`, run `sync_prompt.py`, re-run the notebook. Repeat.

The `rationale` column instructs the model to quote the deciding phrase, so a disagreement can usually be traced to a specific clause rather than argued in the abstract.

---

## Run-size cap, selection, and the seen-ids basket

Notebook `01` enforces a hard ceiling of `MAX_LLM_TWEETS = 2_000` tweets per run, at three independent layers (config constant → dataframe truncation + assertion → a per-call counter inside `classify_tweet` that raises rather than exceed the budget). The cap binds **regardless of `SMOKE_TEST`**. Raising it is a deliberate edit, not a side effect of flipping the smoke-test flag. It caps *tweets*, not requests: with `N_LABEL_PASSES = 2` the request ceiling is 4 000.

This exists because the criteria in `llm_bootstrap_prompt.md` are still being tuned. The full ~10 000-tweet bootstrap partition should not be spent on a definition that has not yet survived a read-through of its own low-confidence rows.

A second, independent ceiling caps **token** spend: `MAX_SESSION_TOKENS = 6_000_000` per run. It must be sized against the *request* ceiling (`MAX_LLM_TWEETS × N_LABEL_PASSES` = 4 000 calls, ~5.6 M tokens at a padded 1 400 per call), not against the tweet count — sized below that it stops being a backstop and silently truncates runs instead. The row cap is what should normally stop a run; this fires only if something inflates per-call cost (a much longer prompt, or a model whose thinking budget is not actually off). It can only be checked *between* calls, since `usage_metadata` arrives with the response, so the guarantee is "no further calls once crossed", with overshoot bounded by one call. **If it fires the run stops cleanly rather than raising** — completed rows are saved, basketed, and flagged `stopped_early` in both JSON artifacts. That matters because `CHECKPOINT_EVERY = 1_000`: an uncaught exception partway through a capped run would discard everything.

**Selection is a fixed permutation, not a fresh sample.** The partition is sorted by `id`, permuted once with `SELECTION_SEED`, and sliced from the head. So the 500 tweets of a smoke run are a strict **subset** of the 2 000 of a full run — successive runs extend the labelled set instead of drawing a new one. (Two `df.sample(n=…)` calls sharing a seed but differing in `n` do *not* nest; that was the earlier behaviour and it would have produced two largely disjoint sets.) Treat `SELECTION_SEED` as frozen once a real run has happened — changing it re-draws which tweets get labelled.

**Every tweet sent to the LLM is recorded** in `Classifiers_Data/HITL/llm_bootstrap_seen_ids_{DATASET_TYPE}.json`, which accumulates across runs (`ids`, `parse_error_ids`, and a per-run log). **These ids are training data — exclude them from any held-out evaluation set.**

`partition_ids.pkl` cannot serve this purpose. It records which tweets are *eligible* for bootstrap labelling (~10 000), but with the cap only a fraction is ever labelled; the remaining ~9 000 of the bootstrap partition stay unseen and are legitimately usable as held-out data. The basket is the only record of where that line actually falls.

**Token spend is measured, not estimated.** A pre-flight cell prints projected tokens and cost before the loop runs (and warns above `COST_ALERT_USD`); the run cell reports the `usage_metadata` the API actually billed, including the implicit-cache hit rate; and each run appends a `llm_bootstrap_usage_<timestamp>.json`. Roughly 96% of every prompt is the identical criteria scaffold, so the cache-hit figure is the number to watch — if it is near zero, the Batch API (50% cheaper, and this workload is not latency-sensitive) is the next lever.

---

## Adding a category

The taxonomy is designed to grow. When adding category *N+1*:

1. Add a `### <label>` block inside the fence in `llm_bootstrap_prompt.md` with the same parts — Definition, Example.
2. Add the label to the [Label set](#label-set) table.
3. Add the label string to `CATEGORIES` in the Label Set cell of notebook `01`, then run `python3 notebooks/05_Classifiers/sync_prompt.py`. The response schema derives its `enum` from `CATEGORIES`, so no other cell needs editing — and the Prompt Builder cell asserts `CATEGORIES` against the fence's category line.
