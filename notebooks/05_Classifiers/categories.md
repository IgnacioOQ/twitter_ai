# Classification Categories
- status: active
- type: guideline
- id: classification_categories
- last_checked: 2026-08-01
<!-- content -->

This document is the **source of truth for what the labels mean**. The prompt that puts those meanings in front of the model is `llm_bootstrap_prompt.md` in this folder — see [Where the criteria live](#where-the-criteria-live).

> **The sync obligation is now mechanical.** Notebook `01` does **not** clone this repo on Colab — it only mounts Drive — so no markdown in this folder is readable at runtime and the prompt has to be embedded in the notebook. That embedding is done by `sync_prompt.py`, not by hand, and `sync_prompt.py --check` fails while the notebook and the prompt file disagree.

The taxonomy deliberately starts at **one substantive category**. The point is a labelling run whose errors are legible: you read the model's `rationale` column, disagree with a specific decision, tighten one definition, and re-run. Growing the label space before that loop works makes every error harder to attribute. See [Adding a category](#adding-a-category) for the extension procedure.

---

## Label set

| Label | Meaning |
| :--- | :--- |
| `originality` | The tweet appeals to originality — newness, creativity, copying, theft — **as a criterion for the value of art**. |
| `none` | No category in the current taxonomy applies. |

`none` is a **residual bucket, not the negation of `originality`.** The distinction matters the moment a second category exists: a row labelled `none` means "no category applied *as of the run that produced it*", so `none` rows from an earlier run are stale evidence about a category that did not exist yet. Naming the complement `none` rather than `not_originality` is what makes that reading available — the label does not have to be renamed or remapped when the taxonomy grows.

---

## The discriminating condition

The category is **not** "mentions copying" and **not** "is about AI art". It is an *evaluative move* with two halves:

> *this work is (not) new / (not) copied* **→ therefore** *it is (not) good, real, or valuable art.*

Vocabulary alone fails the test (a tweet can say "stolen" about wages). An art evaluation alone fails it (a tweet can call AI art soulless without any claim about newness). Both halves must be present.

This framing sits inside a wider debate the project already maps: `docs/Theories on the Value of Art.md` lists ten competing accounts of what makes art valuable — hedonism, expressivism, cognitivism, formal value, moral, social, political, process, achievement, pluralism. Originality is not one of the ten in its own right; it lives closest to Gaut's cluster property *"being an exercise of creative imagination"*. That is precisely why the exclusions below are worth stating explicitly: **most tweets that evaluate art are appealing to one of the other nine accounts**, and those are the errors this category is most likely to absorb if the boundary is left implicit.

---

## Where the criteria live

**The criteria are no longer duplicated here.** They live in `llm_bootstrap_prompt.md` in this
folder, which is the single authored copy of the prompt — scaffold and criteria together, in
the form the model actually receives.

That file is embedded into notebook `01` verbatim by `sync_prompt.py`, and written back out
next to `llm_bootstrap_labels.csv` unchanged, so the version a reviewer reads in Drive, the
version in the notebook, and the version in git are the same bytes. Previously this section
held one copy and the notebook held another, kept in step by hand.

To change what the model is told:

```bash
$EDITOR notebooks/05_Classifiers/llm_bootstrap_prompt.md    # edit inside the fence
python3 notebooks/05_Classifiers/sync_prompt.py             # embed it in the notebook
```

`sync_prompt.py --check` exits non-zero while the two disagree, which is the form to wire into
a pre-commit hook.

This document keeps what the prompt has no room for: why the taxonomy is shaped this way, what
`none` is for, the expected class balance, and what must be settled before adding a category.

One thing to fix there when the first results land: inside that fence, **examples 1-3 are real corpus tweets** carried over from the original draft, but **examples 4-5 are author-written placeholders**. Replace them with real `none` tweets from the smoke run — the negative cases that actually occur in this corpus, rather than the ones we imagined.

---

## Confidence and rationale are the interpretability surface

`01` writes the labels in two shapes. `llm_bootstrap_labels.csv` carries only the HITL schema; `llm_bootstrap_labels_full.pkl` additionally carries `confidence` and `rationale`. **The `.pkl` is the one to read when tuning this file.** The intended loop:

1. Run with `SMOKE_TEST = True` (capped — see below).
2. Open the `.pkl`, sort by `confidence` ascending, and read the bottom rows. Low-confidence rows are where the definition is underspecified.
3. Read a sample of *high*-confidence `originality` rows too — a confident wrong answer means a boundary is missing from the Exclusions, which is the more expensive failure.
4. Edit the fence in `llm_bootstrap_prompt.md`, run `sync_prompt.py`, re-run the notebook. Repeat.

The `rationale` column instructs the model to quote the deciding phrase, so a disagreement can usually be traced to a specific clause rather than argued in the abstract.

---

## Run-size cap, selection, and the seen-ids basket

Notebook `01` enforces a hard ceiling of `MAX_LLM_TWEETS = 2_000` tweets per run, at three independent layers (config constant → dataframe truncation + assertion → a per-call counter inside `classify_tweet` that raises rather than exceed the budget). The cap binds **regardless of `SMOKE_TEST`**. Raising it is a deliberate edit, not a side effect of flipping the smoke-test flag. It caps *tweets*, not requests: with `N_LABEL_PASSES = 2` the request ceiling is 4 000.

This exists because the criteria in `llm_bootstrap_prompt.md` are still being tuned. The full ~10 000-tweet bootstrap partition should not be spent on a definition that has not yet survived a read-through of its own low-confidence rows.

A second, independent ceiling caps **token** spend: `MAX_SESSION_TOKENS = 6_000_000` per run. It must be sized against the *request* ceiling (`MAX_LLM_TWEETS × N_LABEL_PASSES` = 4 000 calls, ~5.6 M tokens at a padded 1 400 per call), not against the tweet count — sized below that it stops being a backstop and silently truncates runs instead. The row cap is what should normally stop a run; this fires only if something inflates per-call cost (a much longer prompt, or a model whose thinking budget is not actually off). It can only be checked *between* calls, since `usage_metadata` arrives with the response, so the guarantee is "no further calls once crossed", with overshoot bounded by one call. **If it fires the run stops cleanly rather than raising** — completed rows are saved, basketed, and flagged `stopped_early` in both JSON artifacts. That matters because `CHECKPOINT_EVERY = 1_000`: an uncaught exception partway through a capped run would discard everything.

**Selection is a fixed permutation, not a fresh sample.** The partition is sorted by `id`, permuted once with `SELECTION_SEED`, and sliced from the head. So the 1 000 tweets of a smoke run are a strict **subset** of the 2 000 of a full run — successive runs extend the labelled set instead of drawing a new one. (Two `df.sample(n=…)` calls sharing a seed but differing in `n` do *not* nest; that was the earlier behaviour and it would have produced two largely disjoint sets.) Treat `SELECTION_SEED` as frozen once a real run has happened — changing it re-draws which tweets get labelled.

**Every tweet sent to the LLM is recorded** in `Classifiers_Data/HITL/llm_bootstrap_seen_ids_{DATASET_TYPE}.json`, which accumulates across runs (`ids`, `parse_error_ids`, and a per-run log). **These ids are training data — exclude them from any held-out evaluation set.**

`partition_ids.pkl` cannot serve this purpose. It records which tweets are *eligible* for bootstrap labelling (~10 000), but with the cap only a fraction is ever labelled; the remaining ~9 000 of the bootstrap partition stay unseen and are legitimately usable as held-out data. The basket is the only record of where that line actually falls.

**Token spend is measured, not estimated.** A pre-flight cell prints projected tokens and cost before the loop runs (and warns above `COST_ALERT_USD`); the run cell reports the `usage_metadata` the API actually billed, including the implicit-cache hit rate; and each run appends a `llm_bootstrap_usage_<timestamp>.json`. Roughly 96% of every prompt is the identical criteria scaffold, so the cache-hit figure is the number to watch — if it is near zero, the Batch API (50% cheaper, and this workload is not latency-sensitive) is the next lever.

---

## Expected class balance

With `DATASET_TYPE = 'AI'`, `originality` is expected to be a **small minority** of the corpus — the partition is drawn from AI-trust discourse at large, not from art discourse. Two consequences:

- Measure prevalence on the first capped run before spending the full partition. If `originality` lands in single-digit percentages, a 10 000-tweet bootstrap yields only a few hundred positives, which is thin for seeding a classifier.
- If prevalence is too low to be useful, the options are: run the bootstrap against `DATASET_TYPE = 'Art'` where the base rate should be far higher, or pre-filter the partition on the cue vocabulary before labelling and accept that the seed set is enriched rather than representative (which changes what the downstream accuracy numbers mean, and must be recorded if done).

---

## Adding a category

The taxonomy is designed to grow. When adding category *N+1*:

1. Add a `### <label>` block inside the fence in `llm_bootstrap_prompt.md` with the same four parts — Definition, Decision test, Examples (positive **and** negative), Exclusions — and add the label to the category line near the top of that fence.
2. Add the label to the [Label set](#label-set) table.
3. Add the label string to `CATEGORIES` in the Label Set cell of notebook `01`, then run `python3 notebooks/05_Classifiers/sync_prompt.py`. The response schema derives its `enum` from `CATEGORIES`, so no other cell needs editing — and the Prompt Builder cell asserts `CATEGORIES` against the fence's category line, so doing step 3 without step 1 fails on the next run rather than silently sending a prompt that never mentions the new label.
4. Revisit the exclusions of **every existing category** — a new category almost always carves territory out of an old one's `none` bucket, and the old category's exclusion list is where that boundary has to be written down.
5. Decide what happens to existing `none` rows. They were labelled against the old taxonomy, so any tweet belonging to the new category is currently sitting in `none`. Either re-run the affected rows or treat the new category as valid only from that run forward — and record which, because it changes what the label set means.
6. Re-check the single-label assumption (below).

`docs/Theories on the Value of Art.md` is the natural source for the next categories: each of the ten accounts of artistic value is a candidate, and several already appear in this file's exclusion list — *achievement / skill*, *expression / emotion*, *meaning / understanding*, *moral value* — which is where the boundary work has already been half-done.

### The single-label assumption

Notebook `01` is **single-label by construction**: the prompt asks for exactly one category and the response schema admits exactly one `label` string. That is sound for a two-label taxonomy where one label is a residual bucket.

It stops being sound once two substantive categories exist, and the original draft of this file already anticipated it — example 3 was annotated *"originality more implicit, automation higher"*, which is a per-category score across two categories on a single tweet, not a single choice between them. **Resolve this before adding category 2**, not after: the decision changes the prompt, the response schema, the CSV schema (`predicted_label` becomes one column per category, or a list), the HITL review interface, and every metric in `02_hitl_training_loop.ipynb`. The cheap interim option is to keep single-label and add a *precedence rule* ("if two categories apply, label the one the tweet's main point rests on"); the honest option is multi-label with a score per category.
