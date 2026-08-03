# Classification Categories
- status: active
- type: guideline
- id: classification_categories
- last_checked: 2026-08-01
<!-- content -->

This document is the **source of truth for the label taxonomy** used by the classifiers in this folder. `01_llm_bootstrap_labelling.ipynb` carries a verbatim copy of the [Criteria block](#criteria-block-verbatim-prompt-text) below in its `CATEGORY_CRITERIA` cell.

> **Sync obligation.** Notebook `01` does **not** clone this repo on Colab — it only mounts Drive — so `categories.md` is not on disk at runtime and the criteria cannot be read from this file. The text is embedded in the notebook instead. **When you edit the Criteria block, re-paste it into cell 7 of `01_llm_bootstrap_labelling.ipynb`.**

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

## Criteria block (verbatim prompt text)

Everything inside the fence is copied into `CATEGORY_CRITERIA` in cell 7 of notebook `01` **exactly as written**. It is deliberately terse and imperative — it is re-sent on every one of the run's API calls (`TOKENOPT_REF.md` §3).

```text
### originality

Definition. Originality in art refers to something which is non-trivially new in a work of
art. It relates to ideas of creativity (in the positive) and ideas about copying (in the
negative). Tweets that are referencing the importance of originality may mention creativity,
newness, difference to prior works, or theft, copying, tracing, plagiarising, replicating.
Label this category when that appeal is used as a criterion for the value of art.

Decision test. Label `originality` only when BOTH halves are present:
  (a) the tweet invokes newness or its absence — creative, original, novel, derivative,
      copy, steal, trace, plagiarise, replicate, rip off, regurgitate, unoriginal; AND
  (b) that invocation does evaluative work about art — it is offered as a reason the
      work, the practice, or the maker is good, bad, real, fake, valuable or worthless art.
Vocabulary alone is not enough. An evaluation of art on some other ground is not enough.

Positive examples.

1. -> originality, high confidence. Plagiarism is named outright as the condition under which
   the art would be unacceptable.
i don't mind ai art as long as it's not plagiarism...i also think some people a bit too lazy with it, seen some make a few posts with ai art that you can clearly see have flaws. yours is great, i really like it. but i feel like others should touch up the ai art before posting...

2. -> originality, medium confidence. The appeal is carried by the analogy rather than stated
   directly, and the scare-quoted "create" is doing the evaluative work.
and farmers learned from other farmers how to crow their crop and harvest. what's your point? the difference is that no farmer or artist is taking another's product and mixing it with yet another stolen product to "create" something. ai is just fancy photoshop for thieves.

3. -> originality, medium confidence. Theft and derivation are explicit, but the tweet is
   framed inside the jobs/automation argument, which competes for the tweet's main point.
sick and absolutely fucking tired of seeing people defending ai art "don't worry they're not taking away your jobs! people thought the same when cameras were invented!" homie that is not the point ai literally steals from artists, it takes whole ass aspects from existing pieces

Negative examples.

4. -> none. The objection is consent and payment, not that the output fails to be new.
they scraped every portfolio on the internet without asking and pay us nothing for it

5. -> none. Art is being evaluated, but on expression and emotion, not on newness.
ai art is empty. there is no human feeling behind any of it.

Exclusions — label `none`:
- Other value criteria. The tweet evaluates art on a ground other than newness: skill or
  effort, soul or emotion, meaning or understanding, beauty or formal qualities, morality,
  social or political function, or the experience of making it.
- Economic, consent or labour objections. The complaint is pay, permission, licensing or
  jobs rather than the work being derivative. If the tweet ALSO argues the output is not
  genuinely new, label originality instead.
- Novelty talk outside art. New models, products, research results, memes.
- Non-evaluative mention. Reporting, defining or quoting a copying dispute with no claim
  about artistic worth.

### none

No category above applies. This is the residual bucket — it is not a claim that the tweet
is unrelated to art or to AI.

Confidence.
0.8-1.0   Explicit: the vocabulary is present and the link to art's worth is stated outright.
0.5-0.79  Implicit: the appeal is inferred from framing, or shares the tweet with a competing
          theme (jobs, automation, consent) of equal or greater weight.
0.0-0.49  Contested: one plausible reading supports the label, another equally plausible
          reading does not.
Score confidence for whichever label you chose, including `none`.

In `rationale`, quote the phrase from the tweet that decided the label.
```

**Examples 1-3 are real corpus tweets** carried over from the original draft with the confidence annotations attached to them. **Examples 4-5 are author-written placeholders** — replace them with real `none` tweets drawn from the first smoke-test run, which will surface the negative cases that actually occur in this corpus rather than the ones we imagined.

---

## Confidence and rationale are the interpretability surface

`01` writes two artifacts. `llm_bootstrap_labels.csv` carries only the HITL schema; `llm_bootstrap_labels_full.pkl` additionally carries `confidence` and `rationale`. **The `.pkl` is the one to read when tuning this file.** The intended loop:

1. Run with `SMOKE_TEST = True` (capped — see below).
2. Open the `.pkl`, sort by `confidence` ascending, and read the bottom rows. Low-confidence rows are where the definition is underspecified.
3. Read a sample of *high*-confidence `originality` rows too — a confident wrong answer means a boundary is missing from the Exclusions, which is the more expensive failure.
4. Edit the Criteria block, re-paste into cell 7, re-run. Repeat.

The `rationale` column instructs the model to quote the deciding phrase, so a disagreement can usually be traced to a specific clause rather than argued in the abstract.

---

## Run-size cap, selection, and the seen-ids basket

Notebook `01` enforces a hard ceiling of `MAX_LLM_TWEETS = 1_000` tweets per run, at three independent layers (config constant → dataframe truncation + assertion → a per-call counter inside `classify_tweet` that raises rather than exceed the budget). The cap binds **regardless of `SMOKE_TEST`**. Raising it is a deliberate edit, not a side effect of flipping the smoke-test flag.

This exists because the criteria above are still being tuned. The full ~10 000-tweet bootstrap partition should not be spent on a definition that has not yet survived a read-through of its own low-confidence rows.

A second, independent ceiling caps **token** spend: `MAX_SESSION_TOKENS = 2_000_000` per run. A full 1 000-tweet run measures ~1.4 M tokens, so the row cap is what normally stops a run and this is a backstop — it fires only if something inflates per-call cost (a much longer criteria block, a model whose thinking budget is not actually off, or a raised `MAX_LLM_TWEETS`). It can only be checked *between* calls, since `usage_metadata` arrives with the response, so the guarantee is "no further calls once crossed", with overshoot bounded by one call. **If it fires the run stops cleanly rather than raising** — completed rows are saved, basketed, and flagged `stopped_early` in both JSON artifacts. That matters because `CHECKPOINT_EVERY = 1_000`: an uncaught exception partway through a capped run would discard everything.

**Selection is a fixed permutation, not a fresh sample.** The partition is sorted by `id`, permuted once with `SELECTION_SEED`, and sliced from the head. So the 100 tweets of a smoke run are a strict **subset** of the 1 000 of a full run — successive runs extend the labelled set instead of drawing a new one. (Two `df.sample(n=…)` calls sharing a seed but differing in `n` do *not* nest; that was the earlier behaviour and it would have produced two largely disjoint sets.) Treat `SELECTION_SEED` as frozen once a real run has happened — changing it re-draws which tweets get labelled.

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

1. Add a `### <label>` block to the Criteria block above with the same four parts — Definition, Decision test, Examples (positive **and** negative), Exclusions.
2. Add the label to the [Label set](#label-set) table.
3. Add the label string to `CATEGORIES` in cell 7 of notebook `01`, and re-paste the whole Criteria block into `CATEGORY_CRITERIA`. The response schema derives its `enum` from `CATEGORIES`, so no other cell needs editing.
4. Revisit the exclusions of **every existing category** — a new category almost always carves territory out of an old one's `none` bucket, and the old category's exclusion list is where that boundary has to be written down.
5. Decide what happens to existing `none` rows. They were labelled against the old taxonomy, so any tweet belonging to the new category is currently sitting in `none`. Either re-run the affected rows or treat the new category as valid only from that run forward — and record which, because it changes what the label set means.
6. Re-check the single-label assumption (below).

`docs/Theories on the Value of Art.md` is the natural source for the next categories: each of the ten accounts of artistic value is a candidate, and several already appear in this file's exclusion list — *achievement / skill*, *expression / emotion*, *meaning / understanding*, *moral value* — which is where the boundary work has already been half-done.

### The single-label assumption

Notebook `01` is **single-label by construction**: the prompt asks for exactly one category and the response schema admits exactly one `label` string. That is sound for a two-label taxonomy where one label is a residual bucket.

It stops being sound once two substantive categories exist, and the original draft of this file already anticipated it — example 3 was annotated *"originality more implicit, automation higher"*, which is a per-category score across two categories on a single tweet, not a single choice between them. **Resolve this before adding category 2**, not after: the decision changes the prompt, the response schema, the CSV schema (`predicted_label` becomes one column per category, or a list), the HITL review interface, and every metric in `02_hitl_training_loop.ipynb`. The cheap interim option is to keep single-label and add a *precedence rule* ("if two categories apply, label the one the tweet's main point rests on"); the honest option is multi-label with a score per category.
