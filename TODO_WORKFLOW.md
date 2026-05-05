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

## Run HITL Data Preparation Notebook
- status: todo
- type: task
- id: todo.run_hitl_data_prep
- description: Execute `00_hitl_data_preparation.ipynb` end-to-end to produce the four partitions and the partition manifest, after the human has reviewed the notebook for errors.
- owner: agent
- blocked_by: [human_review]
- last_checked: 2026-05-05
<!-- content -->
**Context:** The data-prep notebook `notebooks/05_Classifiers/00_hitl_data_preparation.ipynb` was scaffolded recently (cells defined in `notebooks/create_notebook.py`). It loads the JSONL pruned tweet dict produced by `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb`, normalises columns out of the nested `public_metrics` dict, performs an attention-weighted permutation (`SAMPLING_ALPHA = 0.5` default), and slices the dataframe into four disjoint partitions (LLM Bootstrap, Base, HITL ×4, Inference) plus a `partition_ids.pkl` manifest with a disjointness assertion. Several assumptions were inferred from `02_sanity_check_and_network_generation.ipynb` (field names, schema, file path) that may not exactly match the live data — the human should sanity-check the notebook before it is run for real.

**Human action required (do this first):**
> **Please open `notebooks/05_Classifiers/00_hitl_data_preparation.ipynb`, read each cell end-to-end, and revise anything that does not match the actual upstream data or your intended flow.** Pay particular attention to: (1) the `PRUNED_DICT_NAME` / `USE_TEST_DATA` toggle and whether the path resolves; (2) the `_pm_field` extraction (`like_count` vs `retweet_count` vs other names in the actual `public_metrics` dict); (3) the `SAMPLING_ALPHA` default — verify it produces the engagement gradient you want; (4) the partition sizes (`LLM_BOOTSTRAP_SIZE`, `BASE_SIZE`, `HITL_SIZE`). Confirm in chat once done; the agent will then proceed.

**Preconditions:**
- The human has reviewed and (if necessary) revised `00_hitl_data_preparation.ipynb`.
- The pruned tweet dataset exists at `BASE_PATH / 'Data Sets/Cleaned Data/AItrust_twits_pruned_dict.json'` (or `..._test.json` if `USE_TEST_DATA = True`).

**Steps:**
1. Re-read the human-revised notebook in full before doing anything else.
2. Run all cells (Colab or local) with the configuration the human chose. Do not silently change `SAMPLING_ALPHA` or any partition size.
3. Confirm the partition manifest's disjointness assertion passes (the cell will raise if any partition overlaps).
4. Inspect the engagement-distribution diagnostic — the printed table should show a clear gradient on `median` / `mean` / `p95` from LLM Bootstrap (highest) down to Inference (lowest). If the gradient is flat, raise `SAMPLING_ALPHA` and rerun; if the head dominates too aggressively (LLM Bootstrap mostly the same handful of viral tweets), lower it.

**Verification:**
- These files exist in `BASE_PATH/Data Sets/Classifiers_Data/HITL/`: `llm_bootstrap_dataset.pkl`, `base_dataset.pkl`, `inference_dataset.pkl`, `hitl_pending_batch_0{1..4}.pkl`, and `partition_ids.pkl`.
- `partition_ids.pkl` deserialises to a dict with keys `llm_bootstrap`, `base`, `hitl_batch_01..04`, `inference` and the disjointness check printed `all partitions disjoint`.
- The engagement diagnostic shows the expected gradient.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md and append a `WORKLOG.md` entry recording: partition sizes, the `SAMPLING_ALPHA` used, any revisions the human made to the notebook, and the engagement-by-partition table.

---

## Set Up LLM Bootstrap Labelling
- status: todo
- type: task
- id: todo.setup_llm_bootstrap
- description: Configure and run `01_llm_bootstrap_labelling.ipynb` on the LLM Bootstrap partition, after the human has read the notebook and supplied categories, criteria, model id, and API key.
- owner: agent
- blocked_by: [todo.run_hitl_data_prep, human_review]
- last_checked: 2026-05-05
<!-- content -->
**Context:** The LLM bootstrap notebook `notebooks/05_Classifiers/01_llm_bootstrap_labelling.ipynb` was scaffolded recently (cells in `notebooks/create_notebook.py`). It loads `llm_bootstrap_dataset.pkl` (~10 000 tweets carved out by `00_hitl_data_preparation.ipynb`), builds an LLM prompt from a configurable category list and per-category criteria, calls Gemini with retry/backoff and strict JSON parsing, checkpoints every 1 000 rows, and writes `llm_bootstrap_labels.csv` (HITL schema) plus `llm_bootstrap_labels_full.pkl` (with `confidence` + `rationale`). Two cells are placeholders — the category list + criteria, and the Gemini model id + API key — that **must** be filled before running.

**Human action required (do this first):**
> **Please open `notebooks/05_Classifiers/01_llm_bootstrap_labelling.ipynb`, read each cell, and revise the prompt-builder or pipeline if anything looks wrong.** Then provide the agent with: **(1)** the closed list of category labels (multi-class), **(2)** the per-category criteria text (definitions, positive examples, exclusions), **(3)** the Gemini model id (e.g. `gemini-2.0-flash`), and **(4)** confirmation that the `GEMINI_API_KEY` Colab secret has been set. The agent will fill in the placeholders and run the smoke test only after receiving all four.

**Preconditions:**
- `todo.run_hitl_data_prep` has been completed and `BASE_PATH/Data Sets/Classifiers_Data/HITL/llm_bootstrap_dataset.pkl` exists.
- The human has reviewed `01_llm_bootstrap_labelling.ipynb` and supplied: category list, per-category criteria, Gemini model id, and confirmation that the API key is configured as a Colab secret named `GEMINI_API_KEY`.

**Steps:**
1. Re-read the human-revised notebook in full before editing.
2. Fill the `CATEGORIES` list and `CATEGORY_CRITERIA` string with the values supplied by the human (in the marked TODO cell — the file has `assert` statements that fail loudly until the placeholders are gone).
3. Set `MODEL_NAME` in the Configuration cell to the supplied Gemini model id.
4. Run the notebook with `SMOKE_TEST = True` (default 100-tweet subset) on Colab. Inspect the `value_counts()` of `predicted_label` printed at the end — every row should be one of the declared categories or `PARSE_ERROR`, and PARSE_ERROR rate should be < 5%. If higher, adjust the prompt (likely the JSON schema instruction or category criteria) and rerun the smoke test before scaling up.
5. Once the smoke test looks healthy, flip `SMOKE_TEST = False` and run on the full ~10 000-tweet LLM Bootstrap partition.
6. Confirm `llm_bootstrap_labels.csv` was written with the HITL schema (`id | text | likes | retweets | predicted_label | human_label`) and the `_full.pkl` companion was written with `confidence` + `rationale` columns.

**Verification:**
- `BASE_PATH/Data Sets/Classifiers_Data/HITL/llm_bootstrap_labels.csv` has ~10 000 rows.
- `predicted_label` distribution is sensible: no single category > 90 %, no declared category at 0 %, PARSE_ERROR rate < 5 %.
- The CSV loads cleanly from `02_hitl_training_loop.ipynb` (read it, confirm column dtypes match the human-labelled CSV schema).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md and append a `WORKLOG.md` entry recording: the Gemini model used, the categories, the prompt strategy, the PARSE_ERROR rate, the label distribution (`value_counts()` table), and any prompt iterations needed to get the smoke test below 5 % PARSE_ERROR.

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
