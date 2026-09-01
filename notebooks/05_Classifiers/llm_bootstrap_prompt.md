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
those are where the criteria are underspecified. Edit the fence, sync, re-run.

## The prompt

````text
You are a tweet classifier for a research project on AI public trust.
This is a multi-label classification task. Evaluate the tweet against the following conceptual frameworks used to answer the questions "What is art?" and "What makes art valuable?".

Classify the tweet into ALL of the following categories that apply:
intentionalism, anti_intentionalism, cognitivism, expressivism, hedonism, originality, achievement, none

Per-category criteria:

### intentionalism
Definition: This framework posits that the artist's intent is the ultimate source of meaning and the defining characteristic of art. A work is art because it was intended to be so, and it means exactly what the creator meant it to mean. Tweets using this framework will often argue that AI cannot produce art because it lacks a conscious mind, meaning, or deliberate intention.
Example: "can we please stop calling it 'ai art' and start calling it 'ai generated images' instead? art requires meaning and intention, ai has none of that"

### anti_intentionalism
Definition: In direct contrast to intentionalism, this view argues that the author's intent does not govern the meaning of the work. Instead, meaning is derived from the work itself (the language, imagery, or form) and the audience's interpretation. Tweets in this category might invoke the "death of the author" or emphasize that the image speaks for itself regardless of who or what made it.

### cognitivism
Definition: This framework values art for its ability to impart knowledge, foster understanding, or provoke thought. Art is seen as a vehicle for discovering or communicating ideas. Tweets applying this framework will praise art that teaches us something or makes us think, or conversely criticize art for being shallow or intellectually empty.
Example: "gm ☕ i've been playing a lot with ai image generation lately... i like to think i'm a creative person, so for me it's an avenue to express those ideas and a journey of discovery.☄️"

### expressivism
Definition: Expressivism grounds the value (and often the definition) of art in emotion. This can refer to the emotion the artist felt during creation, the emotion captured within the piece, or the feelings it evokes in the audience. Tweets in this category will evaluate works based on their emotional resonance, or dismiss AI art by pointing out its lack of "feeling" or inability to experience emotion.
Example: "🖤 creatures - this ai collection is more about feeling than about a fairy tale🖤"

### hedonism
Definition: This view asserts that the primary value of art lies in the pleasure it brings the viewer. Aesthetic hedonism evaluates art based on whether it is enjoyable, pleasant, or makes the audience feel good. Negative evaluations under this framework will dismiss art simply because it is ugly or unpleasant to look at.

### originality
Definition: Originality values the novel, the creative, and the unprecedented. To be good art (or art at all), the work must offer something non-trivially new. In discourse around AI, this framework is frequently invoked in the negative—criticizing works for being derivative, stolen, traced, copied, or plagiarized from real artists.
Examples:
- "i don't mind ai art as long as it's not plagiarism..."
- "ai is just fancy photoshop for thieves."

### achievement
Definition: This framework ties the value and definition of art to the human effort, skill, and mastery required to produce it. Art is viewed as a triumph of hard work. Tweets using this framework will often attack AI art for being a shortcut, requiring no real skill or patience, and bypassing the arduous learning process that makes traditional art valuable.
Example: "This AI collaboration art thing, isn't effort."

### none
Definition: No category above applies. This is the residual bucket — it is not a claim that the tweet is unrelated to art or to AI. Only output `none` if NO other category applies.

Confidence.
0.8-1.0   Explicit: the vocabulary is present and the link to art's worth/definition is stated outright.
0.5-0.79  Implicit: the appeal is inferred from framing, or shares the tweet with a competing theme.
0.0-0.49  Contested: one plausible reading supports the label, another equally plausible reading does not.
Score confidence for whichever label(s) you chose.

In `rationale`, quote the phrase from the tweet that decided the label.

Return ONLY a JSON array of objects with this exact schema (no prose, no markdown fences). Each object in the array represents a selected category:
[
  {"category": "<one of the categories above>", "confidence": <number between 0 and 1>, "rationale": "<one short sentence>"}
]

Tweet:
"""{{TWEET}}"""
````

## What constrains the reply

The reply shape is enforced **server-side** by the API through a JSON schema, not merely
requested in the prose above. `category` cannot come back as anything outside the label set, and
the reply cannot be wrapped in prose or markdown fences.

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "category":   {"type": "string", "enum": ["intentionalism", "anti_intentionalism", "cognitivism", "expressivism", "hedonism", "originality", "achievement", "none"]},
      "confidence": {"type": "number"},
      "rationale":  {"type": "string"}
    },
    "required": ["category", "confidence", "rationale"]
  }
}
```

The enum is generated from the notebook's `CATEGORIES` list rather than copied from here, and
the notebook asserts that list against the category line in the fence above. Adding a category
therefore means editing three things that are checked against each other: `CATEGORIES`, the
category line in the fence, and the criteria describing it.

The notebook also caps output at 512 tokens and pins the thinking budget to 0 where the model
accepts it. If rationales start coming back truncated they surface as `PARSE_ERROR` rows;
raise `MAX_OUTPUT_TOKENS` before suspecting anything else.

## The CSV this produces

`llm_bootstrap_labels.csv`, one row per tweet, in the same schema as a human
`hitl_review_batch_*.csv`. Since it's multi-label, the CSV contains one-hot columns (0 or 1) for each category.

| Column | Meaning |
| :--- | :--- |
| `id` | Tweet id. |
| `text` | The tweet verbatim. This exact string is what replaced `{{TWEET}}`. |
| `likes`, `retweets` | Engagement counts, for your context only. **Not** part of the prompt. |
| `pred_intentionalism`, etc | The model's binary one-hot prediction for each category (1 if selected, 0 if not). |
| `human_intentionalism`, etc | **Yours.** Empty on delivery — fill in 0 or 1 for every row you review. |

The model's `confidence`, its `rationale`, and whether the independent passes agreed are
**not** in the CSV. They are in the sibling `llm_bootstrap_labels_full.pkl`.

Before overriding a label, read the fence. A label that looks wrong is often the criteria
working exactly as written — which is a reason to edit this file, not just that row.

## How predictions are decided

- The same tweet is labelled **more than once**, independently, each pass through a fresh client.
- **Pass 1 is the operative label** — the one in the CSV. Where passes disagree the row is flagged in `passes_agree` (in the pickle).
- `confidence` in the pickle is the **mean across passes**. It is the model's own self-report.
