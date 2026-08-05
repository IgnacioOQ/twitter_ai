# LLM Bootstrap Labelling — Prompt

**This file is the prompt.** The text inside the fence below is sent to the model verbatim,
once per tweet, with `{{TWEET}}` replaced by the tweet being classified and nothing else
added. There is no system instruction, no conversation history, and no context carried
between calls: whatever the model knows about this task, it knows from the fence below.

It is the single authored copy. `notebooks/05_Classifiers/01_llm_bootstrap_labelling.ipynb`
embeds this file verbatim as `PROMPT_DOC` (Colab does not clone the repo at runtime, so it
cannot be read from disk there) and writes it back out unchanged next to the labels it
produces. The copy sitting beside `llm_bootstrap_labels.csv` in Google Drive is byte-identical
to this one — that is asserted, not hoped for.

`categories.md` in this folder explains *why* the taxonomy is shaped this way — what `none`
means, the expected class balance, what has to be decided before adding a category. This file
is what the model is actually told.

## Editing this file

Edit the fence, then run the sync script, then re-run the notebook:

```bash
python3 notebooks/05_Classifiers/sync_prompt.py
```

The script pastes this file into the notebook's `c-prompt` cell and verifies the round-trip.
**Editing the notebook cell by hand instead will be overwritten**, and the sync script fails
loudly if the two have diverged, so the repo can never disagree with itself about what was
sent. Nothing outside the fence reaches the model — the prose in this file is for you and for
whoever reviews the labels.

The tuning loop that this is built for: run with `SMOKE_TEST = True`, open
`llm_bootstrap_labels_full.pkl`, sort by `confidence` ascending and read the bottom rows —
those are where the criteria are underspecified. Then read the *high*-confidence
`originality` rows, because a confident wrong answer means a boundary is missing from the
exclusions, which is the more expensive failure. Edit the fence, sync, re-run.

## The prompt

````text
You are a tweet classifier for a research project on AI public trust.
Classify the tweet into exactly one of the following categories:
originality, none

Per-category criteria:
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

Return ONLY a JSON object with this exact schema (no prose, no markdown fences):
{"label": "<one of the categories above>", "confidence": <number between 0 and 1>, "rationale": "<one short sentence>"}

Tweet:
"""{{TWEET}}"""
````

## What constrains the reply

The reply shape is enforced **server-side** by the API through a JSON schema, not merely
requested in the prose above. `label` cannot come back as anything outside the label set, and
the reply cannot be wrapped in prose or markdown fences — the two failures that used to
produce `PARSE_ERROR` rows.

```json
{
  "type": "object",
  "properties": {
    "label":      {"type": "string", "enum": ["originality", "none"]},
    "confidence": {"type": "number"},
    "rationale":  {"type": "string"}
  },
  "required": ["label", "confidence", "rationale"]
}
```

The enum is generated from the notebook's `CATEGORIES` list rather than copied from here, and
the notebook asserts that list against the category line in the fence above. Adding a category
therefore means editing three things that are checked against each other: `CATEGORIES`, the
category line in the fence, and the criteria describing it.

The notebook also caps output at 128 tokens and pins the thinking budget to 0 where the model
accepts it. If rationales start coming back truncated they surface as `PARSE_ERROR` rows;
raise `MAX_OUTPUT_TOKENS` before suspecting anything else.

## The CSV this produces

`llm_bootstrap_labels.csv`, one row per tweet, in the same schema as a human
`hitl_review_batch_*.csv`:

| Column | Meaning |
| :--- | :--- |
| `id` | Tweet id. |
| `text` | The tweet verbatim. This exact string is what replaced `{{TWEET}}`. |
| `likes`, `retweets` | Engagement counts, for your context only. **Not** part of the prompt — the model saw the text alone and knew nothing about how the tweet performed. |
| `predicted_label` | The model's answer. Leave it as it is; it is the record of what the model said. |
| `human_label` | **Yours.** Empty on delivery — fill it in for every row you review. |

The model's `confidence`, its `rationale`, and whether the independent passes agreed are
**not** in the CSV. They are in the sibling `llm_bootstrap_labels_full.pkl`. The rationale is
instructed to quote the phrase that decided the label, so a disagreement can usually be traced
to a specific clause above rather than argued in the abstract.

A `predicted_label` of `PARSE_ERROR` is not a label. It means the call failed after every
retry, and the `rationale` field in the pickle holds the error text instead of a reason. Treat
those rows as unlabelled.

Before overriding a label, read the fence. A label that looks wrong is often the criteria
working exactly as written — which is a reason to edit this file, not just that row.

## How `predicted_label` was decided

- The same tweet is labelled **more than once**, independently, each pass through a fresh
  client. The exact number of passes and their temperatures are recorded per run in
  `llm_bootstrap_usage_<timestamp>.json`.
- **Pass 1 is the operative label** — the one in the CSV. Where passes disagree the row is
  flagged in `passes_agree` (in the pickle) rather than silently resolved; with two passes
  there is no majority to take.
- Those disagreements are the highest-value rows to review: they mark where the criteria
  under-determine the answer, which is a defect in this file rather than in the tweet.
- `confidence` in the pickle is the **mean across passes**. It is the model's own self-report,
  calibrated by the scale at the end of the fence. It is not an accuracy estimate.

## Which run produced the CSV next to me

This file is the prompt and nothing else, so it carries no run metadata — that would make the
Drive copy differ from the repo copy, and then neither could be trusted. Model, passes,
temperatures, row count, token spend, and a `sha256` of this file are recorded per run in
`llm_bootstrap_usage_<timestamp>.json` in the same folder. Match a CSV to the prompt that
produced it through that fingerprint: this file changes as the criteria are tuned, and an old
CSV was labelled under an older version of it.
