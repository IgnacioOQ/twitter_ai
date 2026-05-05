# TODO Workflow
- status: active
- type: plan
- id: twitter_ai.todo_workflow
- description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
- label: [planning, agent]
- injection: excluded
- volatility: evolving
- owner: agent
- last_checked: 2026-05-05
<!-- content -->
Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `WORKLOG.md` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls. To bootstrap a new repository, copy this template (`content/templates/TODO_WORKFLOW_TEMPLATE.md` in the knowledge base) to the repo root as `TODO_WORKFLOW.md` and fill in the `{{repo}}` and `{{YYYY-MM-DD}}` placeholders.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
3. After completing one or more tasks, assess whether a WORKLOG.md entry is warranted — see Phase 5 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
4. Confirm a task is still valid before executing; conditions may have changed since it was written.

**Adding tasks (session authors):**
- Copy the template below (without fences), fill in all fields, and insert it as a new `##` block above the Template section, preceded and followed by `---`.
- Be precise: include target file paths, specific tool calls, expected outcomes, and a verification step.
- Any `knowledge_base_update` call requires a current `content_hash` — capture it with a `knowledge_base_read` at execution time, not when writing the task.

---

## Build LLM-Based Bootstrap Classification Pipeline (Colab)
- status: todo
- type: task
- id: todo.llm_bootstrap_classifier
- description: Build a Colab notebook that uses an LLM API (e.g. Gemini) to label tweets with categories using prompt-embedded criteria, to bootstrap the HITL classifier seed labels in stage 05.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-05
<!-- content -->
**Context:** The HITL classifier in `notebooks/05_Classifiers/` currently relies on a human to label 10 000 seed tweets in `hitl_review_batch_00.csv` (see `notebooks/05_Classifiers/classification_strategy.md`, Step 0 — Initial Labelling). An LLM (Gemini API, or equivalent — Anthropic, OpenAI) can pre-label these tweets cheaply: the human can then review/correct rather than label from scratch, or the LLM labels can serve as the initial training set directly. The pipeline must run on Colab, take one tweet at a time, send it to the LLM with a prompt that contains the full category criteria, parse the response, and persist the classification.

**Preconditions:**
- The cleaned tweet dataset produced by `notebooks/02_Processing/03_cleaning_tweets.ipynb` exists at `BASE_PATH/Data Sets/Cleaned Data/AItrust_pruned_twits_with_sentiment_cleaned.json`.
- `notebooks/05_Classifiers/01_hitl_data_preparation.ipynb` has been run and produced `BASE_PATH/Data Sets/Classifiers_Data/HITL/base_dataset.pkl`.
- A Gemini API key (or chosen alternative) is available. On Colab, load it via `from google.colab import userdata; userdata.get('GEMINI_API_KEY')` — never hard-code.
- The final list of categories and per-category criteria has been confirmed with the user (or extracted from `classification_strategy.md` if documented there).

**Steps:**
1. Read `notebooks/05_Classifiers/classification_strategy.md` end-to-end to confirm the schema of `hitl_review_batch_*.csv` (`id | text | likes | retweets | predicted_label | human_label`), the category list, and the role of Step 0.
2. Read `notebooks/notebook_setup.md` and replicate its 5-cell setup pattern in the new notebook (env switch + paths → optional `git clone` → optional `pip install` → explicit imports → `src.*` imports if needed).
3. Confirm the category list and per-category criteria with the user before drafting the prompt. Persist the criteria as a constant inside the notebook (or, if long, in a sibling `notebooks/05_Classifiers/llm_bootstrap_prompt.md` referenced from the notebook).
4. Create `notebooks/05_Classifiers/00_llm_bootstrap_labelling.ipynb`. The `notebooks/create_notebook.py` helper may be used for scaffolding.
5. Implement the pipeline inside the notebook:
   - Load `base_dataset.pkl` and (for development) sample N=100 tweets via a `SMOKE_TEST` flag.
   - Build the LLM client; load the API key from Colab `userdata`.
   - Construct one instruction prompt containing: task description, category list, per-category criteria, and a strict output schema such as JSON `{"label": "<category>", "confidence": <0-1>, "rationale": "<one short sentence>"}`.
   - Iterate over tweets; for each tweet call the LLM with prompt + tweet text, parse the JSON response, append to a results list. Validate that `label` is one of the declared categories; on parse failure, retry once, then mark the row as `predicted_label = "PARSE_ERROR"` and log the raw response.
   - Add retry-with-exponential-backoff for rate-limit (`429`) and transient errors (`5xx`); cap at 3 retries.
   - Checkpoint every 1 000 labelled tweets to `BASE_PATH/Data Sets/Classifiers_Data/HITL/llm_bootstrap_checkpoint_<n>.pkl` so a Colab disconnect does not lose progress, mirroring the checkpoint pattern in `04_full_dataset_inference.ipynb`.
6. Persist the final output as `BASE_PATH/Data Sets/Classifiers_Data/HITL/llm_bootstrap_labels.csv` with the **exact same schema** as `hitl_review_batch_*.csv`: `id | text | likes | retweets | predicted_label | human_label`. `predicted_label` is the LLM's label; `human_label` is left blank for downstream review.
7. Add a leading markdown cell that documents: which LLM model + version was used, the prompt strategy, rate-limit handling, and how to feed the output into `02_hitl_training_loop.ipynb` (either as the seed CSV in place of `hitl_review_batch_00.csv`, or as an additional labelled batch).
8. Update `notebooks/05_Classifiers/classification_strategy.md` to add the LLM-bootstrap option as a precursor or alternative to human seed labelling in Step 0. This file is in the working repo (not the KB), so edit it directly with `Edit`.

**Verification:**
- Notebook runs end-to-end on the 100-tweet `SMOKE_TEST` subset and writes a CSV at the expected path with the expected columns.
- A `value_counts()` on `predicted_label` shows only declared categories plus, at most, a small fraction of `PARSE_ERROR` rows.
- `02_hitl_training_loop.ipynb` can read the LLM-labelled CSV without schema changes (load it, confirm column dtypes match).
- A `WORKLOG.md` entry is appended documenting the new notebook, the model + prompt used, and any non-obvious decisions.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).

---

## Task Template

Copy the block below (without the outer fences), fill in all fields, and insert it as a new `## [Task Title]` task block.

````markdown
## [Task Title]
- status: todo
- type: task
- id: todo.[short_id]
- description: One-sentence description of what this task accomplishes.
- owner: agent
- blocked_by: []
- last_checked: {{YYYY-MM-DD}}
<!-- content -->
**Context:** Why this task exists and what triggered it. Include the KB path or repo file path it operates on.

**Preconditions:** Any state that must be true before starting (prior tasks complete, files present, etc.). Write `none` if there are none.

**Steps:**
1. (Include specific tool calls where possible, e.g., `knowledge_base_read(path="content/...", sections=["..."])`)
2. ...

**Verification:** How to confirm the task is complete (e.g., a grep that should return one match, a status field that should read `done`).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).
````
