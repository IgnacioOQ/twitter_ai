# TODO Workflow
- status: active
- type: plan
- id: twitter_ai.todo_workflow
- description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
- label: [planning, agent]
- injection: excluded
- volatility: evolving
- owner: agent
- last_checked: 2026-05-06
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
> **Please open `notebooks/05_Classifiers/01_llm_bootstrap_labelling.ipynb`, read each cell, and revise the prompt-builder or pipeline if anything looks wrong.** Then provide the agent with: **(1)** the closed list of category labels (multi-class), **(2)** the per-category criteria text (definitions, positive examples, exclusions), **(3)** the Gemini model id (e.g. `gemini-2.5-flash-lite`), and **(4)** confirmation that the `GEMINI_API_KEY` Colab secret has been set. The agent will fill in the placeholders and run the smoke test only after receiving all four.

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

## Preserve `referenced_tweets_dictionary` in 00 Output
- status: todo
- type: task
- id: todo.preserve_ref_dict_in_00
- description: Keep the `referenced_tweets_dictionary` column in the partitioned dataframe so notebooks 03 and 04 can resolve retweet labels via lookup instead of falling through to per-row model fallback.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-06
<!-- content -->
**Context:** Cell 7 of `notebooks/05_Classifiers/00_hitl_data_preparation.ipynb` currently narrows the normalised dataframe to `['id', 'text', 'processed_text', 'type', 'likes', 'retweets']`, dropping the `referenced_tweets_dictionary` field that came from `02_Processing/02_sanity_check_and_network_generation.ipynb`. Both `03_final_inference.ipynb` (Pass 2 / Pass 3) and `04_full_dataset_inference.ipynb` need this field on `retweets_dataset.pkl` to look up each retweet's referenced original. Without it, every retweet in the partitioned set falls through to the per-row `model_no_reference` fallback and the `lookup` / `model_synthetic_retweet` provenance values never appear — defeating the orphan-promotion optimisation we designed in this redesign.

**Preconditions:** none. Best done before `todo.run_hitl_data_prep` so 00 is run only once with the correct schema.

**Steps:**
1. Edit cell 7 of `notebooks/05_Classifiers/00_hitl_data_preparation.ipynb`:
   - Add `'referenced_tweets_dictionary'` to the `required` set used by the assert (or, if the upstream JSONL might not always carry it, drop it from `required` and warn instead of raising — confirm by spot-checking the live JSONL).
   - Add `'referenced_tweets_dictionary'` to the column whitelist on the `df = df[[...]].copy()` line.
2. Add a one-liner after the assert printing the share of rows with a non-null `referenced_tweets_dictionary` (sanity check that the field actually carries through).
3. Update the cell-6 markdown to mention the new column.

**Verification:**
- After re-running 00, `pd.read_pickle(partitioned_folder / 'retweets_dataset.pkl').columns` includes `referenced_tweets_dictionary`.
- The Pass-2 warning in 03 (`WARNING: retweets_dataset.pkl has no referenced_tweets_dictionary column.`) does not fire.

**On completion:** Delete this entire task block.

---

## Update 01 and 02 to Read Partition Pickles from `Partitioned Data/`
- status: todo
- type: task
- id: todo.fix_partition_paths_in_01_02
- description: Point `01_llm_bootstrap_labelling.ipynb` and `02_hitl_training_loop.ipynb` at the new `Cleaned Data/Partitioned Data/` location for the partition pickles.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-06
<!-- content -->
**Context:** `00_hitl_data_preparation.ipynb` now writes partition pickles (`llm_bootstrap_dataset.pkl`, `base_dataset.pkl`, `inference_dataset.pkl`, `retweets_dataset.pkl`, `hitl_pending_batch_*.pkl`, `partition_ids.pkl`) to `Cleaned Data/Partitioned Data/`. Notebooks 01 and 02 still read them from the old `Classifiers_Data/HITL/` location and will fail at file-open time when run.

**Preconditions:** none.

**Steps:**
1. **`01_llm_bootstrap_labelling.ipynb`:**
   - Add `partitioned_folder = cleanedds_folder / 'Partitioned Data'` to the path block in Cell 1 (next to `hitl_folder`).
   - Change `INPUT_PATH = hitl_folder / 'llm_bootstrap_dataset.pkl'` to `INPUT_PATH = partitioned_folder / 'llm_bootstrap_dataset.pkl'`.
   - Leave `OUTPUT_CSV`, `OUTPUT_PKL`, `CHECKPOINT_PREFIX` writing to `hitl_folder` — labelling artifacts stay there.
2. **`02_hitl_training_loop.ipynb`:**
   - Add `partitioned_folder = cleanedds_folder / 'Partitioned Data'` to the path block in Cell 1.
   - Change `NEXT_BATCH_PATH = hitl_folder / f'hitl_pending_batch_{...}.pkl'` to use `partitioned_folder` instead.
   - Grep the rest of the notebook for `hitl_folder /` and assess each match: partition pickles → `partitioned_folder`; labelling artifacts (`hitl_review_batch_*.csv`) and trained-model writes → unchanged.

**Verification:**
- Re-running 01: `INPUT_PATH.exists()` passes after 00 has been run.
- Re-running 02 with `PENDING_BATCH_TO_PROCESS = 1`: `NEXT_BATCH_PATH.exists()` passes and the notebook reaches the model-training step.

**On completion:** Delete this entire task block.

---

## Confirm `FULL_CORPUS_PATH` in 04
- status: todo
- type: task
- id: todo.confirm_full_corpus_path
- description: Verify the file path used by `04_full_dataset_inference.ipynb` for the ~17M-tweet corpus and update the configuration cell with the correct value.
- owner: agent
- blocked_by: [human_review]
- last_checked: 2026-05-06
<!-- content -->
**Context:** `04_full_dataset_inference.ipynb` defaults to `FULL_CORPUS_PATH = datasets_folder / 'AItrust_twits_dict.json'` (the unprocessed dump from `02_Processing/01`, per the README data-folder map). The README and `classification_strategy.md` reference "the remaining ~17 million tweets" but never pin a single canonical file — the corpus may live in `AItrust_twits_dict.json`, in many `Raw Data/Twits/*.json` harvest files, or in a yet-unproduced larger pruned export. The notebook will fail at `_load_corpus(FULL_CORPUS_PATH)` if the path is wrong.

**Human action required (do this first):**
> **Please confirm which file (or files) hold the full ~17M tweet corpus.** Candidates: (a) `Data Sets/AItrust_twits_dict.json` — current default; (b) raw API harvest under `Raw Data/Twits/*.json` aggregated together; (c) something else. Reply with the path. The agent will pin it (and adjust the loader to glob multiple files if needed for option b).

**Preconditions:** human has supplied the canonical path.

**Steps:**
1. Update `FULL_CORPUS_PATH` in the Configuration cell of `04_full_dataset_inference.ipynb` to the human-supplied value.
2. If the corpus is split across many JSON files, replace the single `_load_corpus(path)` call with a glob + concat. Keep the in-cell `_load_corpus` helper for whichever single-file format ends up handling each chunk.
3. If the chosen file does not include `processed_text`, leave the existing fallback in cell 3 in place (it sets `processed_text = ''` for missing rows). Twitter-RoBERTa uses `text` regardless.
4. Confirm `referenced_tweets_dictionary` is present on the chosen corpus — Pass 2 needs it for retweet lookup. If it isn't, raise a follow-up task before running.

**Verification:** `FULL_CORPUS_PATH.exists()` is True and `_load_corpus(FULL_CORPUS_PATH)` returns a dataframe with at least `id`, `text`, `type`.

**On completion:** Delete this entire task block.

---

## Decide `create_notebook.py` Governance
- status: todo
- type: task
- id: todo.create_notebook_governance
- description: Decide whether `notebooks/create_notebook.py` remains the source of truth for the HITL-stack notebooks (and re-sync it) or is retired as a one-time scaffold.
- owner: agent
- blocked_by: [todo.preserve_ref_dict_in_00, todo.fix_partition_paths_in_01_02, human_review]
- last_checked: 2026-05-06
<!-- content -->
**Context:** `notebooks/create_notebook.py` was the original scaffold for notebooks 00-04. It is currently out of sync with the live `.ipynb` files in three places:
- `nb0_cells` — pre-dates the `Partitioned Data` folder, the retweet split, the `processed_text` / `type` columns, and the multi-column newline scrubbing in the human-seed CSV cell.
- `nb1_cells` and `nb2_cells` — `INPUT_PATH` / `NEXT_BATCH_PATH` still resolve through `hitl_folder` (will be fixed by `todo.fix_partition_paths_in_01_02` on the .ipynb side; the script will silently re-introduce the bug if regenerated).
- `nb3_cells` and `nb4_cells` — original Apr-10 versions; the live notebooks were rewritten for the three-pass merge protocol (Step 4 / Step 5 of `classification_strategy.md`). Running the script as-is would clobber the rewrite.

**Human action required (do this first):**
> **Pick a path:**
> (a) **Re-sync** — agent updates `nb0_cells` … `nb4_cells` in `create_notebook.py` to mirror the current `.ipynb` content, so the script can be re-run safely. Keeps it as the canonical scaffold.
> (b) **Retire** — move it to `notebooks/archive/` and treat the `.ipynb` files as the source of truth from now on. Stops the drift; loses regeneration ability.

**Preconditions:** the .ipynb-side fixes (`todo.preserve_ref_dict_in_00`, `todo.fix_partition_paths_in_01_02`) have been applied so re-syncing has the right target.

**Steps (option a):**
1. Read each live notebook in full and rewrite the corresponding `nbX_cells` list in `create_notebook.py`.
2. Run `python3 notebooks/create_notebook.py` and confirm `git diff` shows no changes to any `.ipynb`.
3. Add a one-line header comment reminding maintainers to re-sync after every notebook hand-edit.

**Steps (option b):**
1. `git mv notebooks/create_notebook.py notebooks/archive/create_notebook.py.archived`.
2. Drop the `create_notebook.py` mention from the README's `notebooks/` tree.
3. Add a brief note in `archive/` explaining the script was the original scaffold.

**Verification:** option a — `git diff` after running the script shows no `.ipynb` changes. Option b — file no longer exists at the old path; README updated.

**On completion:** Delete this entire task block.

---

## Run HITL Active-Learning Loop (4 rounds)
- status: todo
- type: task
- id: todo.run_hitl_training_loop
- description: Run `02_hitl_training_loop.ipynb` iteratively (up to 4 rounds), labelling each emitted `hitl_review_batch_XX.csv` between rounds, per Steps 1-3 of the strategy doc.
- owner: agent
- blocked_by: [todo.run_hitl_data_prep, todo.setup_llm_bootstrap, todo.fix_partition_paths_in_01_02]
- last_checked: 2026-05-06
<!-- content -->
**Context:** Implements Steps 1-3 of `notebooks/05_Classifiers/classification_strategy.md`. Each round trains the model on whatever labels exist (LLM bootstrap + any human reviews so far), predicts the next pending HITL batch, exports a 10k tweet review CSV (5k uncertain + 5k random), waits for human labels, and repeats. There are 4 pending batches.

**Preconditions:**
- `00_hitl_data_preparation.ipynb` has been run (`todo.run_hitl_data_prep` complete) and produced `Partitioned Data/hitl_pending_batch_0{1..4}.pkl`.
- `01_llm_bootstrap_labelling.ipynb` has produced `llm_bootstrap_labels.csv` (`todo.setup_llm_bootstrap` complete).
- `02_hitl_training_loop.ipynb` has been path-fixed (`todo.fix_partition_paths_in_01_02` complete).

**Steps:**
1. Set `PENDING_BATCH_TO_PROCESS = 1`, run the notebook end-to-end on Colab GPU. Inspect val accuracy from each model section (LR / LightGBM / RoBERTa) and record it.
2. Open `hitl_review_batch_01.csv`, fill `human_label` for as many rows as you can reasonably review (target: all 10k), save.
3. Increment `PENDING_BATCH_TO_PROCESS` and re-run. Repeat through batch 4 (or stop earlier if val accuracy plateaus).
4. Append a `WORKLOG.md` entry per round with val-accuracy delta and model timings.

**Verification:**
- `Models/Classifiers/best_roberta_model/`, `lgb_embed.txt`, `lgb_bow.txt` exist and are timestamped after the latest round.
- `hitl_review_batch_01..04.csv` (or however many rounds you ran) all have `human_label` populated for the rows reviewed.

**On completion:** Delete this entire task block; the running `WORKLOG.md` entries already capture the round-by-round detail.

---

## Run Final Inference (Notebook 03)
- status: todo
- type: task
- id: todo.run_final_inference
- description: Execute `03_final_inference.ipynb` to produce `final_annotated_tweets.{pkl,csv}` per Step 4 of the strategy doc (HITL Remainder + Retweet Merge).
- owner: agent
- blocked_by: [todo.run_hitl_training_loop, todo.preserve_ref_dict_in_00]
- last_checked: 2026-05-06
<!-- content -->
**Context:** Implements Step 4 (HITL Remainder + Retweet Merge) of `notebooks/05_Classifiers/classification_strategy.md`. Three passes: (1) classify partitionable corpus with the trained RoBERTa; (2) promote orphan retweet originals (one model call per missing-original group); (3) look up retweet labels. Output goes to `Data Sets/Classifiers_Data/Final/`.

**Preconditions:**
- `02_hitl_training_loop.ipynb` has been run for the desired number of rounds (`todo.run_hitl_training_loop` complete) — `best_roberta_model/` exists.
- `00_hitl_data_preparation.ipynb` was rerun *after* `todo.preserve_ref_dict_in_00` so `retweets_dataset.pkl` carries `referenced_tweets_dictionary`.
- `llm_bootstrap_labels.csv` and `hitl_review_batch_*.csv` are in `Classifiers_Data/HITL/`.

**Steps:**
1. Run the notebook end-to-end on Colab (GPU recommended for Pass 1).
2. Inspect Pass 2 output: the rate of "no usable ref_id" should be < 1%. If higher, follow the in-notebook hint to investigate `02_Processing/02_sanity_check_and_network_generation.ipynb` before trusting Pass 3.
3. Inspect the printed `label_source` `value_counts()` — entries should be present for `human`, `llm_bootstrap`, `model_original`, `lookup`, and likely `model_synthetic_retweet`. `model_no_reference` should be a small minority.

**Verification:**
- `Data Sets/Classifiers_Data/Final/final_annotated_tweets.pkl` and `.csv` exist.
- `final_df['label_source'].notna().all()` — every row has provenance.
- Total row count ≈ partitionable corpus + retweets from 00.

**On completion:** Delete this entire task block; append a `WORKLOG.md` entry recording the `label_source` distribution and any anomalies in the orphan/no-reference rates.

---

## Run Full-Dataset Mass Inference (Notebook 04)
- status: todo
- type: task
- id: todo.run_full_dataset_inference
- description: Execute `04_full_dataset_inference.ipynb` to label the leftover ~17M tweet corpus and produce `full_inference_annotated.{pkl,csv}` per Step 5 of the strategy doc.
- owner: agent
- blocked_by: [todo.run_final_inference, todo.confirm_full_corpus_path]
- last_checked: 2026-05-06
<!-- content -->
**Context:** Implements Step 5 of `notebooks/05_Classifiers/classification_strategy.md`. Reuses the labels from notebook 03's `final_annotated_tweets.pkl` so we never re-classify a tweet we already have a label for, classifies new non-retweets in chunks with checkpointing, then resolves new retweets via the same orphan-promotion pattern as 03.

**Preconditions:**
- `03_final_inference.ipynb` has been run (`todo.run_final_inference` complete) and produced `final_annotated_tweets.pkl`.
- `FULL_CORPUS_PATH` in the notebook has been confirmed (`todo.confirm_full_corpus_path` complete).

**Steps:**
1. Confirm GPU is available on the Colab runtime (mass inference on ~17M tweets is impractical on CPU).
2. Run the notebook end-to-end. Monitor `tqdm` progress and the periodic checkpoint prints.
3. If the runtime disconnects mid-run, the most recent `Full_Inference/checkpoint_<n>.pkl` can be loaded back into `labels` to resume. Modify the Pass 1 loop to skip the first `n` already-labelled rows of `new_originals`.
4. After Pass 3, inspect the `label_source` distribution. Dominant values should be `lookup` (sibling retweets) and `model_original` (new originals/replies/quotes). `model_synthetic_retweet` should reflect the count of distinct missing-original groups in the new retweet population.

**Verification:**
- `Data Sets/Classifiers_Data/Full_Inference/full_inference_annotated.pkl` and `.csv` exist.
- Row count ≈ `final_annotated_tweets` + new originals + new retweets.

**On completion:** Delete this entire task block; append a `WORKLOG.md` entry recording total rows labelled, runtime, GPU model used, and the `label_source` distribution.

---

## Verify 02/02 Test Branch Runs End-to-End Against In-Repo Test Dicts
- status: todo
- type: task
- id: todo.verify_02_02_local_test_run
- description: Human-driven verification that `02_sanity_check_and_network_generation.ipynb` cells 1–43 complete successfully with `RUNNING_LOCALLY=True` and `USE_LOCAL_TEST_DATA=True`, against the in-repo `data_sets/` test dicts.
- owner: human
- blocked_by: []
- last_checked: 2026-05-06
<!-- content -->
**Context:** In session 2026-05-06 the agent wired `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 to point at `<repo>/data_sets/` when `USE_LOCAL_TEST_DATA=True`, and smoke-tested only the read patterns in cells 5 and 6 (both files parse cleanly: 233,094 twits, 1,272 authors, expected JSONL schemas). The downstream "Test Data" branch (Prune Test DS, Test Timeline, Test Network, Test Author Corpus) was **not** executed yet — it makes structural assumptions about field names (e.g. `referenced_tweets`, `referenced_tweets_dictionary`, `public_metrics`) and date filters (`earliest_date = 2022-10-31`) that should be confirmed against the live test data. The "Sanity Test Full" cells (8–9), "Prune Full DS" cells (22–24), and the entire "Full Data" branch (cells ~46–60) are expected to fail because the full (`AItrust_twits_dict.json`, no `_test`) inputs are not in `data_sets/`; those cells should be **skipped**, not investigated.

**Human action required:**
> **Open `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb`, set `RUNNING_LOCALLY = True` and `USE_LOCAL_TEST_DATA = True` in cell 1, and run the test-branch cells (1–7, 10–21, 25–43).** Skip cells 8–9 and 22–24 (Full sanity / Prune Full DS) and 46+ (entire Full branch). Watch for: (a) any cell that errors on a missing field in the test dict, (b) zero-row outputs from the prune step that suggest `earliest_date` cuts the test data too aggressively, (c) network-generation cells producing a graph with 0 edges. Report back whichever happens first; do **not** debug silently.

**Preconditions:**
- `data_sets/AItrust_twits_dict_test.json` and `data_sets/AItrust_author_dict_test.json` exist at the repo root (confirmed present 2026-05-06).
- Cell 1 of the notebook contains the two toggles `RUNNING_LOCALLY` and `USE_LOCAL_TEST_DATA` (already wired this session).

**Steps:**
1. Run cells 1–7 (setup + sanity check test). Confirm no exceptions and that the printed sample tweets/authors look sensible.
2. Run cells 10–21 (prune functions + Prune Test DS). Confirm `data_sets/Cleaned Data/AItrust_twits_pruned_dict_test.json` and `AItrust_Art_pruned_twit_dict_test.json` are written and contain >0 rows. If 0 rows, drop or widen `earliest_date` and rerun.
3. Run cells 25–43 (Test timeline, Test network, Test author corpus). Confirm the network graph reports >0 nodes and >0 edges; confirm the timeline and author corpus pickles deserialise cleanly.

**Verification:**
- The following files exist after the run, all written into the in-repo path:
  - `data_sets/Cleaned Data/AItrust_twits_pruned_dict_test.json`
  - `data_sets/Cleaned Data/AItrust_Art_pruned_twit_dict_test.json`
  - `data_sets/Cleaned Data/test_basic_counts_dict.pkl`
  - `data_sets/Cleaned Data/test_timeline_dict.pkl`
  - `data_sets/Cleaned Data/test_author_corpus_dict.pkl`
  - `data_sets/Networks/test_network_dict.pkl`
  - `data_sets/Networks/Test_Network.gml` (and the `.graphml` / `.gexf` / `.json` variants if their cells were run)
- The test network has >0 nodes and >0 edges.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md and append a `WORKLOG.md` entry recording: the test-network node/edge counts, the pruned-twit row counts (AI-only and AI+Art), any cells that needed a tweak, and whether further follow-ups were spawned (e.g. structural mismatches in `referenced_tweets_dictionary`).

---

## Harmonise BASE_PATH Across create_notebook.py and 02/02 Cell 1
- status: todo
- type: task
- id: todo.harmonise_base_path_drift
- description: Resolve the BASE_PATH inconsistency between `notebooks/create_notebook.py` and `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` so every notebook in the project resolves to the same Drive folder.
- owner: agent
- blocked_by: [human_review]
- last_checked: 2026-05-06
<!-- content -->
**Context:** While reviewing scaffolding in session 2026-05-06, two BASE_PATH definitions were found to disagree:

1. `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1:
   - Local: `Path('/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust')`
   - Colab: `Path('/content/drive/MyDrive/AI Public Trust')` ← note `MyDrive/AI Public Trust`
2. `notebooks/create_notebook.py` `setup_cell()` (lines 121, 125):
   - Local: `Path('/Volumes/GoogleDrive/My Drive/Colab Projects/AI Public Trust')`
   - Colab: `Path('/content/drive/My Drive/Colab Projects/AI Public Trust')` ← note `My Drive/Colab Projects/AI Public Trust`

`README.md` and `notebooks/notebook_setup.md` both state the canonical layout is `My Drive/Colab Projects/AI Public Trust/`. The Colab path in 02/02 cell 1 (`MyDrive/AI Public Trust`) skips the `Colab Projects/` parent and collapses `My Drive` into `MyDrive` — most likely a stale path from before the `Colab Projects/` reorganisation. Until the canonical path is settled, **do not touch `setup_cell()`'s BASE_PATH** (the path-shape change in `todo.create_notebook_local_test_toggle` would otherwise bake the drift in).

**Human action required (do this first):**
> **Please confirm which path is the live Drive layout.** Open Drive Desktop / drive.google.com and check whether `AI Public Trust/` lives at the root of `MyDrive` or under `MyDrive/Colab Projects/`. Reply with the canonical absolute path. If both paths exist (e.g. a copy lives at the root), say so — that may indicate a separate cleanup TODO.

**Preconditions:**
- The human has confirmed the canonical Drive path.

**Steps:**
1. Update `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 so its Colab branch uses the canonical path. The local-Drive branch already matches the README layout — leave it unchanged unless the human says otherwise.
2. Update `notebooks/create_notebook.py` `setup_cell()` (lines 121 and 125) so both branches use the canonical path.
3. Grep the rest of the repo for `'MyDrive/AI Public Trust'`, `'My Drive/Colab Projects/AI Public Trust'`, `'/content/drive/MyDrive'`, `'/content/drive/My Drive'` and update any other notebook or doc that disagrees with the canonical path.
4. Update `notebooks/notebook_setup.md` and `README.md` only if their stated canonical path turns out to be wrong (the human's reply settles this).

**Verification:**
- `git grep -E "MyDrive|My Drive"` returns the canonical path everywhere except documented historical notes.
- Cell 1 of every notebook resolves to the same Drive folder when `RUNNING_LOCALLY=False`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md and append a `WORKLOG.md` entry recording: which path was canonical, the list of files updated, and any orphaned files left behind on Drive.

---

## Add USE_LOCAL_TEST_DATA Toggle to create_notebook.py setup_cell()
- status: todo
- type: task
- id: todo.create_notebook_local_test_toggle
- description: Update the `setup_cell()` helper in `notebooks/create_notebook.py` so future notebooks emitted by the scaffolder include the `USE_LOCAL_TEST_DATA` flag introduced in 02/02 cell 1.
- owner: agent
- blocked_by: [todo.harmonise_base_path_drift]
- last_checked: 2026-05-06
<!-- content -->
**Context:** In session 2026-05-06 the agent wired `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 to read the in-repo `data_sets/` folder when `RUNNING_LOCALLY=True` AND `USE_LOCAL_TEST_DATA=True`. That edit lives in 02/02 cell 1 only — the `setup_cell()` function in `notebooks/create_notebook.py` (the scaffolder used to (re)generate the HITL classifier notebooks `05_Classifiers/00..04`) still emits the old single-flag pattern. To make the local-test pattern propagate to any future notebook scaffolded by this script, `setup_cell()` needs to be updated to mirror the new shape.

**The new shape (already in 02/02 cell 1) is:**
1. Add `USE_LOCAL_TEST_DATA = False` directly under `RUNNING_LOCALLY = False` with the explanatory comment block from 02/02.
2. In the `if RUNNING_LOCALLY:` branch, nest a `USE_LOCAL_TEST_DATA` check that sets `BASE_PATH = Path(_REPO_ROOT) / 'data_sets'` when True; keep the existing local-Drive path as the `else`.
3. Replace the unconditional folder-path block with a `if RUNNING_LOCALLY and USE_LOCAL_TEST_DATA:` branch where `datasets_folder = BASE_PATH`, `cleanedds_folder = BASE_PATH / 'Cleaned Data'`, `networks_folder = BASE_PATH / 'Networks'`, both subfolders auto-created via `.mkdir(parents=True, exist_ok=True)`. The `else` keeps today's `BASE_PATH / 'Data Sets/...'` shape. `twits_folder`, `test_folder`, `literature_folder`, `topic_models_folder`, and `classifiers_folder` (with its existing `.mkdir(...)` call) stay common.

Compare against `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 for the verbatim pattern. The `extra_paths` parameter behaviour must be preserved (it is appended after the standard folder block — the HITL notebooks rely on it for `hitl_folder`, `out_folder`, `full_inference_folder`).

**Side effect to manage carefully:** Running `python3 notebooks/create_notebook.py` regenerates **all five** HITL notebooks (`05_Classifiers/00_hitl_data_preparation.ipynb`, `01_llm_bootstrap_labelling.ipynb`, `02_hitl_training_loop.ipynb`, `03_final_inference.ipynb`, `04_full_dataset_inference.ipynb`) from the cell lists in the script — any uncommitted edits in those notebooks are clobbered. At session start the working tree had `M notebooks/05_Classifiers/00_hitl_data_preparation.ipynb` (and uncommitted edits in `02_sanity_check_and_network_generation.ipynb` itself), so do **not** rerun the scaffolder until those edits are committed or otherwise preserved.

**Preconditions:**
- The pending uncommitted edits in `05_Classifiers/00_hitl_data_preparation.ipynb` are committed, or the human has explicitly confirmed they are safe to discard.
- `todo.harmonise_base_path_drift` is resolved or explicitly deferred (otherwise the new template bakes the path-drift in).

**Steps:**
1. Read `notebooks/create_notebook.py` lines 102–139 (the existing `setup_cell()` function) and `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 (the new pattern).
2. Edit `setup_cell()` to match the new shape described above. Preserve the `extra_paths` append behaviour (it is critical for the HITL notebooks).
3. Do **not** automatically run `create_notebook.py`. Show the human the diff and let them decide when to regenerate.
4. After the human confirms regeneration is safe, run `python3 notebooks/create_notebook.py` and inspect cell 1 of one regenerated notebook (e.g. `05_Classifiers/00_hitl_data_preparation.ipynb`) to confirm the new toggle is present and the indentation matches the canonical pattern.

**Verification:**
- A regenerated HITL notebook's cell 1 contains both `RUNNING_LOCALLY = False` and `USE_LOCAL_TEST_DATA = False`, the nested `if USE_LOCAL_TEST_DATA:` branch, and the conditional folder-path block with `.mkdir(...)` calls — character-for-character matching the structure in 02/02 cell 1.
- The HITL notebooks still validate as JSON (the `write_notebook()` round-trip check in `create_notebook.py` passes).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md and append a `WORKLOG.md` entry recording: the diff to `setup_cell()`, whether the HITL notebooks were regenerated, and any followups (e.g. `notebook_setup.md` update if not already done — see `todo.notebook_setup_md_local_test`).

---

## Document USE_LOCAL_TEST_DATA Pattern in notebook_setup.md
- status: todo
- type: task
- id: todo.notebook_setup_md_local_test
- description: Update the canonical Cell 1 pattern in `notebooks/notebook_setup.md` to include the `USE_LOCAL_TEST_DATA` toggle introduced in 02/02 cell 1.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-06
<!-- content -->
**Context:** `notebooks/notebook_setup.md` is the canonical setup-pattern document referenced by `README.md`, every notebook author, and the agent's auto-memory entry "Notebook Setup Pattern". Its "Cell 1 — Environment Switch & Paths" section currently shows a two-branch pattern (Colab vs local Drive). In session 2026-05-06 the agent extended 02/02 cell 1 to support a third mode — `RUNNING_LOCALLY=True` AND `USE_LOCAL_TEST_DATA=True` — which points `BASE_PATH` at the in-repo `data_sets/` folder so the test branch of a notebook can run end-to-end without Google Drive. That toggle is currently undocumented; future notebook authors and agents will not know it exists unless `notebook_setup.md` describes it.

**Preconditions:**
- The `USE_LOCAL_TEST_DATA` shape in 02/02 cell 1 is the agreed-upon canonical version (i.e. no further iteration is expected — `todo.verify_02_02_local_test_run` ideally signed off first).

**Steps:**
1. Read `notebooks/notebook_setup.md` Cell 1 section (currently lines ~14–54) and `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` cell 1 (the verbatim source).
2. Update the Cell 1 code block in `notebook_setup.md` to include `USE_LOCAL_TEST_DATA` and the conditional folder-path branch that auto-creates `cleanedds_folder` / `networks_folder` under `<repo>/data_sets/`.
3. Add a short prose note under the code block explaining when to flip the flag (smoke-testing locally without Drive, tied to the in-repo `data_sets/` test fixtures), and the constraint that it only takes effect when `RUNNING_LOCALLY=True` (the Colab branch ignores it — flipping it under Colab is a no-op).
4. If "Common Mistakes" at the bottom of the doc is the right place, add a bullet warning that `USE_LOCAL_TEST_DATA=True` while `RUNNING_LOCALLY=False` is silently ignored.
5. Update `last_checked` in the doc's metadata block to today's date.

**Verification:**
- The Cell 1 code block in `notebook_setup.md` is character-for-character identical to 02/02 cell 1's source (modulo any section-header comments that prefix subsections in the doc).
- The auto-memory entry "Notebook Setup Pattern" in the agent's `MEMORY.md` does not contradict the new content. If it does, surface the contradiction to the human; do not silently amend memory.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. A WORKLOG entry is optional unless the doc edit also touched "Common Mistakes" or surfaced a contradiction with auto-memory.

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
