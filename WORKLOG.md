# Worklog
- status: active
- type: log
- id: twitter_ai.worklog
- description: Append-only working history of significant agent interventions, difficult problems solved, and major changes to this repository.
- label: [agent]
- injection: excluded
- volatility: evolving
- last_checked: 2026-05-05
<!-- content -->
Append-only working history. Newest entries first.
Add an entry whenever you solve a difficult problem, make a significant change, or complete a major task.

---

## 2026-05-05 — Build classifier-stage data prep + LLM bootstrap pipeline
- status: done
- type: task
- id: twitter_ai.worklog.2026_05_05_classifier_pipeline
- last_checked: 2026-05-05
<!-- content -->
**What:**
- Authored two new notebooks under `notebooks/05_Classifiers/` via the `notebooks/create_notebook.py` cell-spec pattern: `00_hitl_data_preparation.ipynb` and `01_llm_bootstrap_labelling.ipynb` (numbering swapped mid-session so chronological order matches file order).
- **Data prep** (`00_*`): reads the JSONL pruned tweet dict at `cleanedds_folder / 'AItrust_twits_pruned_dict.json'` (output of `02_Processing/02_sanity_check_and_network_generation.ipynb`) via `pd.read_json(lines=True)`, normalises columns out of the nested `public_metrics` dict, casts `id` to `str` (Twitter snowflakes overflow float64), performs an **attention-weighted permutation** (`weights = (likes + retweets + 1) ** SAMPLING_ALPHA`, default `0.5`), and slices into four disjoint partitions: LLM Bootstrap (10 000), Base (100 000), HITL ×4 batches of 50 000, Inference (remainder). Writes `partition_ids.pkl` manifest with an inline pairwise-disjointness assertion and emits a per-partition engagement-distribution diagnostic.
- **LLM bootstrap** (`01_*`): Gemini-based labelling pipeline with strict JSON output schema (`label`, `confidence`, `rationale`), retry-with-exponential-backoff (3 retries, 2 s base), validation that the returned label is in the declared `CATEGORIES`, `PARSE_ERROR` fallback after final retry, checkpoint every 1 000 rows, output CSV matching the existing `hitl_review_batch_*.csv` schema. Two cells are placeholders (categories + criteria; model id + API key) protected by `assert` statements so the notebook refuses to run until they're filled.
- Updated `notebooks/05_Classifiers/classification_strategy.md`: new partition table row for LLM Bootstrap; Step 0 rewritten as Path A (LLM) / Path B (human); attention-weighted sampling rationale added to Key Decisions; corrected upstream notebook reference to `02_Processing/02` (was incorrectly `03_cleaning_tweets`); new artifact entries in Data Folder Structure.
- Replaced the build-pipeline task in `TODO_WORKFLOW.md` with two execution-stage tasks (`todo.run_hitl_data_prep`, `todo.setup_llm_bootstrap`), both gated on `human_review` so a fresh agent must wait for the human to read and revise the notebooks before running them.

**Why:**
- Initial spec relied on two wrong assumptions surfaced during user review: uniform random sampling, and the wrong upstream file (`03_cleaning_tweets`'s sentiment-cleaned output rather than `02`'s pruned dict). Tweet engagement is heavy-tailed (near power-law), so a uniform 10 000-tweet sample would consist almost entirely of low-engagement tweets and the LLM/classifier would never see the discourse-shaping content.
- Numbering swap (data prep `01→00`, LLM bootstrap `00→01`) makes the file numbers reflect actual run order, which matters for someone opening the folder cold.

**Outcome:** Both notebooks regenerate cleanly from `create_notebook.py` and parse as valid Python. Pipeline waits on human review before execution per the new TODO blocks. No code committed.

**KB changes:** None.

**Follow-up:** Human to read both notebooks and provide categories + per-category criteria + Gemini model id + Colab `GEMINI_API_KEY` secret confirmation before the agent picks up `todo.setup_llm_bootstrap`. The `SAMPLING_ALPHA` default (`0.5`) is a starting point — the engagement-distribution diagnostic in the data-prep notebook will tell the user whether to raise it (flat gradient → too uniform) or lower it (head dominates → over-concentrated).

---

## 2026-05-05 — Bootstrap governance files and rewrite README
- status: done
- type: task
- id: twitter_ai.worklog.2026_05_05_bootstrap_governance
- last_checked: 2026-05-05
<!-- content -->
**What:** Created `TODO_WORKFLOW.md` and `WORKLOG.md` at the repo root from the KB templates (`content/templates/TODO_WORKFLOW_TEMPLATE.md`, `content/templates/WORKLOG_TEMPLATE.md`). Rewrote `README.md` to cover the full repo structure, the six-stage notebook pipeline (including the previously undocumented `05_Classifiers/` and renumbered `06_Experiments/`), the Google Drive data layout under `BASE_PATH = AI Public Trust/`, and pointers to the agents/ and docs/ governance files.

**Why:** The repository had no cross-session task backlog or worklog, so coding-agent sessions had nowhere to leave or pick up pending work, and no audit trail of significant interventions. The previous README was minimal and missing the HITL classifier stage, the `src/` package, and the Drive data structure — a new reader (human or agent) could not get a working mental model from it alone.

**Outcome:** `README.md` rewritten; `TODO_WORKFLOW.md` and `WORKLOG.md` created. No code changes. Follow-up: none — future sessions can now use these files per Phase 5 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
