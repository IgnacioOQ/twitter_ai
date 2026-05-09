# Worklog
- status: active
- type: log
- id: twitter_ai.worklog
- description: Append-only working history of significant agent interventions, difficult problems solved, and major changes to this repository.
- label: [agent]
- injection: excluded
- volatility: evolving
- last_checked: 2026-05-06
<!-- content -->
Append-only working history. Newest entries first.
Add an entry whenever you solve a difficult problem, make a significant change, or complete a major task.

---

## 2026-05-06 — Wire 02/02 to in-repo `data_sets/` for local smoke-testing
- status: done
- type: task
- id: twitter_ai.worklog.2026_05_06_local_test_data_toggle
- last_checked: 2026-05-06
<!-- content -->
**What:**
- The user added a new top-level `data_sets/` folder to the repo containing the test fixtures `AItrust_twits_dict_test.json` (313 MB, 233,094 JSONL records) and `AItrust_author_dict_test.json` (743 KB, 1,272 records). Goal was to make the test branch of `notebooks/02_Processing/02_sanity_check_and_network_generation.ipynb` runnable on a laptop without Google Drive Desktop.
- Modified cell 1 of `02_sanity_check_and_network_generation.ipynb` to add a `USE_LOCAL_TEST_DATA = False` toggle alongside `RUNNING_LOCALLY`. When both are `True`, `BASE_PATH = <repo>/data_sets/` and `datasets_folder = BASE_PATH`; `cleanedds_folder = BASE_PATH / 'Cleaned Data'` and `networks_folder = BASE_PATH / 'Networks'` are routed under it and auto-created via `.mkdir(parents=True, exist_ok=True)`. Default behaviour (Colab + local-Drive) is unchanged.
- Edit had to go through a Python `json.load`/`json.dump` round-trip because the `.ipynb` exceeds the Read tool's 25 000-token limit and `NotebookEdit` requires a prior Read. Verified the edit was isolated to cell 1 by parsing both pre-edit and post-edit cell sources and comparing — only `cells[1].source` changed; cell count and all other sources unchanged. The pre-existing diff vs `HEAD` (cells 12+) is the user's own uncommitted work, not from this session.
- Smoke-tested the new wiring with a Python script that mimicked cell 1's path resolution and replayed the read patterns from cells 5 and 6: both files parsed cleanly via line-by-line `json.loads`. Twit schema confirmed: `id`, `text`, `created_at`, `public_metrics`, `author_id`, `type`, `referenced_tweets`, `conversation_id`, `entities`, `referenced_tweets_dictionary`. Author schema confirmed: `description`, `public_metrics`, `created_at`, `id`, `entities`, `name`, `username`, `verified`.
- Did **not** execute the full test-branch (cells 1–43) end-to-end — that's a human action captured as `todo.verify_02_02_local_test_run` because pruning + network generation against 233 K records is slow and I'd rather have the human eyeball the intermediate outputs.
- Reviewed `notebooks/create_notebook.py` and `notebooks/notebook_setup.md` while considering propagation. Surfaced two issues: (1) BASE_PATH drift — 02/02 cell 1's Colab branch uses `Path('/content/drive/MyDrive/AI Public Trust')` while `setup_cell()` (lines 121, 125) uses `Path('/content/drive/My Drive/Colab Projects/AI Public Trust')`; the latter matches `README.md` and `notebook_setup.md` so the former is most likely a stale artifact. (2) `setup_cell()` and the canonical `notebook_setup.md` don't yet describe the new toggle, so any newly scaffolded notebook will lack it.
- Updated `TODO_WORKFLOW.md` with four self-contained follow-up tasks (in suggested execution order):
  - `todo.verify_02_02_local_test_run` (owner: human) — the end-to-end run.
  - `todo.harmonise_base_path_drift` (blocked on human confirming canonical Drive path).
  - `todo.create_notebook_local_test_toggle` (blocked on the path-drift task and on committing the pending edits in `00_hitl_data_preparation.ipynb` to avoid a regen clobber).
  - `todo.notebook_setup_md_local_test` (doc update; ideally after verification).

**Why:**
- Self-contained smoke testing without Drive shortens the iteration loop on `02/02` and the `src/network/` modules. It also makes the test fixture available as a real input for any future regression check on network generation. The chain reaction (scaffolder + canonical doc updates) was deliberately deferred so we don't bake the path-drift inconsistency into the new template, and so the user can preserve their uncommitted edits in `05_Classifiers/00_hitl_data_preparation.ipynb` before any scaffolder regen.

**Outcome:**
- Cell 1 of `02_sanity_check_and_network_generation.ipynb` now supports three modes (Colab, local-Drive, local-test). Read patterns verified against both in-repo test dicts. No regressions in the default modes. Four follow-ups queued in `TODO_WORKFLOW.md` with full context, preconditions, steps, and verification gates.

**KB changes:** None — deferred until `todo.verify_02_02_local_test_run` confirms the wiring works end-to-end. Auto-memory's "Notebook Setup Pattern" entry is consistent with the change but should gain a `USE_LOCAL_TEST_DATA` line once the doc update task ships.

**Follow-up:** See the four new tasks in `TODO_WORKFLOW.md`. Nothing else outstanding from this session.

---

## 2026-05-05 — Migrate LLM bootstrap to google-genai SDK + Gemini 2.5 family
- status: done
- type: task
- id: twitter_ai.worklog.2026_05_05_genai_sdk_migration
- last_checked: 2026-05-05
<!-- content -->
**What:**
- Swapped the LLM bootstrap notebook (`notebooks/05_Classifiers/01_llm_bootstrap_labelling.ipynb`) from the deprecated `google-generativeai` package to the unified `google-genai` SDK. New call shape: `client = genai.Client(api_key=...)` + `client.models.generate_content(model=..., contents=..., config=...)`. Imports updated to `from google import genai; from google.genai import types`.
- Web-verified model pricing and the `ThinkingConfig` API. Found that the 2.0 Flash family (including `gemini-2.0-flash-lite`) is being phased out (EOL ~June 2026) and that `gemini-2.5-flash-lite` is now the cheapest stable model (~$0.10/M in, $0.40/M out; batch mode $0.05/$0.20).
- Set `MODEL_NAME = 'gemini-2.5-flash-lite'` as the notebook default. Added a `DISABLE_THINKING = True` knob with three-branch logic that: (a) on `gemini-2.5-pro`, pins `thinking_budget=128` because Pro cannot disable thinking; (b) on `gemini-2.5-flash` / `gemini-2.5-flash-lite` with `DISABLE_THINKING=True`, sets `thinking_budget=0`; (c) on those models with `DISABLE_THINKING=False`, prints a cost warning. No-op on 2.0 models.
- Updated `TODO_WORKFLOW.md` (example model id in the human-action block: `gemini-2.0-flash` → `gemini-2.5-flash-lite`).
- KB updates (user-approved): `content/how-to/GEMINI_ERROR_HANDLING_SKILL.md` — expanded model list to include 2.5-flash-lite / 2.5-pro / 3.x previews, marked 2.0 family deprecated, added new "Disabling Thinking on 2.5 Models" subsection with the `ThinkingConfig` snippet + Pro-min-128 caveat + the python-genai #1842 tools-present silent-ignore gotcha. `content/how-to/MCP_SKILL.md` — appended a 2026-05 deprecation note to the gemini-2.0 paragraph pointing readers at the updated GEMINI doc.

**Why:** Session-start choice of `google-generativeai` + `gemini-2.0-flash-lite` was based on training knowledge; KB and web verification revealed both were superseded. Migrating now avoids a forced rewrite when the 2.0 family is shut down. Propagating the finding to the canonical KB doc means future agents in any project that uses the KB inherit the correct model list automatically.

**Outcome:** Notebook regenerates cleanly from `create_notebook.py`, parses as valid Python, runs the correct three-branch thinking logic. Two KB docs updated (with `.kb_backups/` snapshots). No code committed.

**KB changes:**
- Updated `content/how-to/GEMINI_ERROR_HANDLING_SKILL.md` (additive — model list + new section).
- Updated `content/how-to/MCP_SKILL.md` (one-paragraph deprecation note).

**Follow-up:** None for the SDK / model layer. The pending TODOs (`todo.run_hitl_data_prep`, `todo.setup_llm_bootstrap`) still apply unchanged — both wait on human review.

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

## 2026-05-09 — Pipeline graph + BASE_PATH harmonisation
- type: entry
- id: twitter_ai.worklog.2026_05_09_pipeline_graph_and_basepath
- last_checked: 2026-05-09
<!-- content -->
**Task:** Build a queryable + visual representation of the data pipeline implicit across the 19 notebooks (the user can't easily see what produces/consumes what), and resolve the long-standing `todo.harmonise_base_path_drift` so artifact nodes resolve to a single canonical Drive path.

**Outcome:**
- New `src/scripts/pipeline_graph.py` (~430 lines) — static analyzer that walks every `.ipynb`, extracts path-variable bindings (canonical Colab branch), tracks local vars across cells in source order, recognizes read/write idioms (`pickle.dump`, `json.dump/load`, `pd.read_*`, `df.to_*`, `nx.read_gml/write_gml`, `ig.Graph.Read_GML`, `np.save/load`, `with open(...)`), resolves `/`-concat and `+`-concat paths, strips Jupyter magics before `ast.parse`. CLI: `build` (default, dumps JSON + 2 PNGs), `validate` (compares parser output vs README/notebook_setup.md `[stage/index]` tags), `downstream NB`, `upstream NB`.
- New `docs/pipeline_overrides.yaml` — small annotations file (v1/v2 sentiment alternatives, stage-05 checkpoint glob patterns, friendly labels). Expected to stay <50 lines; everything else derived.
- New generated artifacts under `docs/`: `pipeline_graph.json` (78 artifact nodes, 19 notebook nodes, 59 write + 56 read edges), `pipeline_graph.png` (full bipartite, stage-band layout), `pipeline_graph_notebooks.png` (notebook-only DAG projection).
- `README.md` — added pointer + run command under Repository Structure.
- **BASE_PATH harmonisation:** canonical Colab path is `/content/drive/My Drive/Colab Projects/AI Public Trust` (matches `02_extract_examples.ipynb`, the README data-layout block, and the canonical `notebooks/notebook_setup.md`). Updated **14 notebook files** (replaced wrong `MyDrive/AI Public Trust` references), `notebooks/notebook_setup.md` (line 44 cell-1 example), `src/scripts/inject_env_switch.py` (template lines 18 and 23 — local Volume now uses `My Drive/Colab Projects/...` instead of `MyDrive/...`, Colab branch likewise), `src/scripts/fix_notebook_paths.py` (re-pointed the normalisation target to the canonical, otherwise re-running it would re-introduce the drift), `agents/NOTEBOOK_SKILL.md` (two narrative refs), `README.md` (one narrative ref). After re-running `pipeline_graph.py` the BASE_PATH-divergence stderr warning is gone and all artifact nodes share the single prefix `My Drive/Colab Projects/AI Public Trust`.

**Key decisions:**
- **Code-first sourcing.** The graph is rebuilt from the live `.ipynb` files on every run; there is no hand-curated manifest to maintain. Decision driven by the user's note that the repo is not finalised — any docs-based manifest would silently drift.
- **Stage-band layout** instead of `nx.topological_generations`. Disconnected notebooks (stage 6 `tp_bigrams_test`, the two stage-1 ingestion notebooks whose paths can't fully resolve) all landed in the same generation under topological layout, crowding the picture. y = -stage matches the user's mental model and the README's stage table.
- **Strip the `/content/drive/` runtime prefix** from all artifact paths, also at emit time (not only at env-collection time). f-string-derived paths (`03/03_lda_tweet_topics`'s templated outputs) were leaking the prefix; emit-time strip catches them uniformly.
- **`fix_notebook_paths.py` updated, not retired.** The script previously normalised everything *to* the wrong Colab path; flipping its target plus removing the canonical from `HARDCODED_PREFIXES` makes the script idempotent under the new canonical.

**Validation findings (kept, not fixed):** `pipeline_graph.py validate` flags `Full_Network.gml` as documented `[04/01]` in the README data layout though the parser correctly identifies `02/02` as the producer (`04/01` is the consumer). Not fixed in this session — flagged for the README maintainer.

**KB changes:** none — `content/how-to/GRAPH_REPRESENTATION_SKILL.md` already covered the relevant cookbook recipes (§3.3, §6.7, §6.12, §6.5). The notebook-AST-static-analysis tactics (cell-magic strip, cross-cell var binding, igraph-vs-networkx aliases, `+`-concat handling, `np.save`/`np.load` direct-path detection) are arguably worth capturing as a separate `NOTEBOOK_STATIC_IO_GRAPH_SKILL.md` if a similar parser is needed for another repo — deferred until that need surfaces.

**Follow-up:**
- `todo.create_notebook_local_test_toggle` is now unblocked (its `blocked_by` was `todo.harmonise_base_path_drift`).
- Three notebooks remain at 0/0 in the graph: stage-01 `01_shared_folder_setup` (no real I/O), stage-01 `02_twitter_api_mining` (uses an unresolved `parent` variable in its own setup — pre-canonical layout), stage-06 `01_tp_bigrams_test` (confirmed isolated, synthetic data only). Acceptable; the graph is honest about what the parser can and can't see.
