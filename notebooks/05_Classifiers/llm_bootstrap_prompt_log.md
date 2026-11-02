# LLM Bootstrap Prompt — Change Log
- status: active
- type: log
- id: llm_bootstrap_prompt_log
- last_checked: 2026-11-02
<!-- content -->

A record of every version of the prompt in `llm_bootstrap_prompt.md`: what changed, why, and
what the runs made with it showed. `categories.md` says what the labels mean and
`classification_strategy.md` says how the labels are used; this file says how the text the
model reads got to its current form, so that a labels file on Drive can be traced back to the
criteria that produced it, and so that a problem already fixed once is not re-introduced.

## How to add an entry

1. Edit the fence in `llm_bootstrap_prompt.md`, then run
   `python3 notebooks/05_Classifiers/sync_prompt.py`. It prints the **file fingerprint**
   (`sha256:` of the whole markdown file, stored in the notebooks as `PROMPT_DOC_SHA256` and
   written to every usage record as `prompt_doc_sha256`).
2. Add a row to the table below and an entry under *Versions*, newest first. Fill in the
   **sent fingerprint** (`prompt_sent_sha256` — the fence alone, which is what the model
   receives) from the next run's `llm_bootstrap_usage_<stamp>.json`; it changes only when the
   fence changes, so prose-only edits share it with the version before.
3. State the evidence: which run, which rows, which report section. "It read better" is not
   an entry. The low-confidence rows of `llm_bootstrap_labels_full_<stamp>.pkl`, the pass
   disagreement study, the co-occurrence block, and the `none` audit are the usual sources.
4. After the next run, come back and record what it showed under the same entry.

Fingerprints are the first 12 hex characters of the SHA-256, as the notebooks print them.

## Versions

| # | Date | Commit | File fingerprint | Sent fingerprint | Change |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 8 | 2026-11-02 | — | `e4e7586cad52` | `7a9e1e9d8857` | `value` counts casual praise/criticism, not seller promotion; criteria apply literally, no argument needed |
| 7 | 2026-11-02 | `c3473ed` | `cdb1c2c846ba` | `96d53ecfad1b` | `none` is exclusive; parser enforces it |
| 6 | 2026-10-19 | `1b83a71` | `c4956915e763` | `6245fd0a8962` | prose only (audit documented); fence unchanged |
| 5 | 2026-10-19 | `e7a6d7b` | `fccda9f2dbf2` | `6245fd0a8962` | prose only (stamped file names); fence unchanged |
| 4 | 2026-09-28 | `8ce0cc9` | `41593454982f` | `6245fd0a8962` | added the question labels `definition` and `value`; two-group framing |
| 3 | 2026-09-01 | `6101b1e` | `38c352174afb` | `7d478ff552cd` | multi-label: seven aesthetic frameworks + `none`, JSON array reply |
| 2 | 2026-08-05 | `2887e76` | `f18081896ac8` | — | prose only; fence unchanged |
| 1 | 2026-08-05 | `834f8e5` | `9394390d6274` | — | first version of the file: one category, `originality` + `none`, single-label reply |

Before version 1 (2026-08-01, commit `fde6909`) the criteria were a literal inside the
notebook, with a second copy in `categories.md`; the two drifted, which is why the file and
`sync_prompt.py` exist. The 2026-08-03 smoke runs belong to that period: 100 rows on
`gemini-2.5-flash-lite` came back 100% `PARSE_ERROR` twice, and the move to
`gemini-3.1-flash-lite` the same morning fixed it (0 errors on 100 and then 500 rows).

### 8 — `value` threshold and literal reading (2026-11-02)

**Change.** Two additions to the fence, nothing removed:

- After the two-group framing: *"The tweets are ordinary social-media posts, not essays.
  Apply each criterion as written: a tweet does not need to make an argument, or to use
  philosophical vocabulary, to meet it."*
- In `value`: casual praise or criticism of a specific work, performance or creator counts,
  with no argument attached; a seller promoting its own product, token or collection is
  advertising and does not count on its own. One example added: *"this is what ai voice
  generation is meant to be"*.

**Evidence: the `none` audit of run `20261019_1550`.** Section 6 of `01` sent the 500
most-engaged `none`-only rows to `gemini-4-argon` (thinking `low`) with the same prompt
(version 6). Argon kept `none` on 486 (97.2%), relabelled 13 (2.6%), 1 parse error. Labels it
assigned: `value` 11, `cognitivism` 4, `hedonism` 2, `definition` 1, `originality` 1, at
audit confidences of 0.45–0.75. Median engagement was the same for relabelled rows (58) and
rows kept as `none` (62), so engagement does not predict the miss.

What the 13 rows have in common is **pass 1's reason for `none`**, not the tweets. Eleven of
the thirteen pass-1 rationales invoke a bar the fence never sets — *"does not engage in
philosophical discourse / arguments regarding the definition or value of art"* — while the
`value` criterion already says that *"the language of liking or disliking a work"* counts.
Argon applied the criterion as written: *"this is what ai voice generation is meant to be"*
(127,733 likes), *"amazing AI artists"*, *"cuz its cute"*, *"Sapiens is what ChatGPT would
produce if…"*. So the two models disagree about **where the threshold sits**, not about how
hard the tweets are to read, and that is a prompt problem. The opening framing ("two central
questions", "conceptual frameworks") is the likely source of the philosophical bar, hence the
general sentence rather than a `value`-only fix.

Not every relabel is an improvement, which is why the amendment also says what does *not*
count: the crypto and NFT promotions (*"The quality produced by @PixiaAi"*, *"Another sick
free claim for our holders!"*) are advertising, and the "Book Corner" reference-work plug is
marketing copy. `cognitivism` on *"watch movies … to know more about your terrorist nation"*
is a stretch (political, not about art) and is left alone for now.

**Decision recorded: no thinking budget for the labelling model.** The audit cannot show
that thinking helps — Argon differs from `gemini-3.1-flash-lite` in size as well as in
thinking, and the misses above are threshold misses that reasoning does not fix. Thinking
tokens are billed as output ($1.50/M on 3.1 Flash-Lite) and count against
`MAX_OUTPUT_TOKENS = 512`, so a budget also means raising the cap or paying for truncated
`PARSE_ERROR` rows. Where thinking might help is the 102 pass-disagreement rows and the
`none`-beside-a-label hedge (version 7), which are consistency problems; that is a smoke-run
A/B to do once human labels exist. The ~130 rows to label first: the 13 Argon relabels, the
18 `none`+label rows, the 102 disagreements.

**Pricing checked 2026-11-02** against https://ai.google.dev/gemini-api/docs/pricing:
3.1 Flash-Lite $0.25 / $1.50 per M in/out (batch half); 4 Argon $2.00 / $10.00, **doubling on
2026-12-01**; 3.8 Flash $0.75 / $3.75, doubling 2027-01-01. Argon bills thinking tokens as
output and the notebooks' usage code counts visible output only, so the recorded audit cost
($2.23) is a floor.

**Run with this version:** none yet.

### 7 — `none` is exclusive (2026-11-02, `c3473ed`)

**Change.** Added to the `none` criterion: *"`none` is never combined with another category.
If you select any other category, do not also output `none`, even when you are unsure;
express the doubt through that category's confidence instead."* In the same commit the
parsers of `01` and `01b` (`enforce_none_exclusive`) drop `none` from any reply that also
carries another label, before `pred_*`, `passes_agree` or the audit see it, and count how
often they did; `01b` prints the count in its run report and both notebooks write it to the
usage record (`none_dropped`).

**Evidence: run `20261019_1550` (version 6).** 18 of 2,000 rows carried `none` beside another
label (13 with `value`, 5 with `definition`), although the fence already said *"Only output
`none` if NO other category applies."* The pattern in the rationales is a substantive label
followed by a hedging `none` (*"While the tweet critiques music, it focuses on…"*). Those rows
would have trained `02` on contradictory targets. The prompt line alone could not be relied
on, hence the parser rule; the dropped count is what tells whether the line helped.

**Run with this version:** `01`, 2026-11-02, 2,000 tweets, two passes, audit on — in
progress at the time of writing; the dropped count is printed at the end of the Labelling
Loop cell.

### 6 and 5 — prose only (2026-10-19)

The fence did not change (`prompt_sent_sha256` stays `6245fd0a8962`); the prose around it was
updated for the stamped file names (`llm_bootstrap_labels_<stamp>.csv`) and to document the
pass-disagreement study and the second opinion on `none`.

**Run `20261019_1550`** (`01`, `gemini-3.1-flash-lite`, 2,000 tweets, two passes at
temperatures 0.0 and 0.5, 0 parse errors): 1,621 rows (81%) `none`-only; `value` 269,
`achievement` 111, `definition` 100, `originality` 64, `hedonism` 61, `cognitivism` 46,
`expressivism` 21, `intentionalism` 16, `anti_intentionalism` 1. Labels per labelled row: one
110, two 234, three 29, four 6. Passes agreed on 1,898 rows (94.9%). Audit as under version 8.
Measured usage ~6.48M tokens over 4,000 calls (~1,620 per call), ~$2.00 at the standard rate;
audit 502 calls, 0.96M visible tokens, $2.23.

### 4 — question labels (2026-09-28, `8ce0cc9`)

**Change.** Added `definition` and `value` as a second group of labels — a tweet that
*raises* "What is art?" or "What makes art valuable?" — distinct from the seven frameworks
that *answer* them; added the two-group framing sentence and the rule that a tweet can carry
both. Fence grew from 4,674 to 6,731 characters.

**Why.** With frameworks only, a tweet that judged a work good or bad, or denied that
something was art, without appealing to a nameable framework had nowhere to go but `none`.

**Runs:** 2026-09-28, 500 rows then 2,000 rows (the unstamped `llm_bootstrap_labels.csv`),
0 parse errors in both.

### 3 — multi-label, seven frameworks (2026-09-01, `6101b1e`)

**Change.** Replaced the single-label `originality`/`none` task with a multi-label one over
`intentionalism`, `anti_intentionalism`, `cognitivism`, `expressivism`, `hedonism`,
`originality`, `achievement` and `none`; the reply became a JSON array of
`{category, confidence, rationale}` objects. Each framework got a one-paragraph definition
and, where available, one example tweet. The detailed decision test and negative examples of
version 1 were not carried over.

**Runs:** 2026-09-01, two runs of 500 rows, 29 parse errors each (5.8%) — the array reply was
then only prompt-instructed; the server-side response schema came later in the notebook.

### 1 — one category (2026-08-05, `834f8e5`)

**Change.** First version of the file, created so the notebook, `categories.md` and the
reviewers' copy on Drive could not disagree. One substantive category, `originality`, plus
`none`; a two-part decision test (newness vocabulary **and** evaluative work about art);
positive and negative examples; a confidence scale; single-label JSON object reply.

**Runs:** 2026-08-05, two runs of 500 rows on `gemini-3.1-flash-lite`, 0 parse errors.
