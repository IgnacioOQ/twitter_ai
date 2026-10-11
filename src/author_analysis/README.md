# Author analysis

One implementation serves the command-line runner and
`notebooks/03_Analysis_and_Modeling/04_lda_author_topics_hpc.ipynb`.
Each backbone selects its authors, includes all post types (including retweets),
and fits its own topic model. The notebook imports this package; it does not carry
an alternative fitting implementation.

## Setup and inputs

Use Python 3.10+ and install `src/author_analysis/requirements.txt` in a virtual environment.
Run commands from the repository root. DATA contains the notebook data layout:

- `Data Sets/Cleaned Data/AItrust_twits_pruned_dict.json`: canonical General AI posts.
- `Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-sentiment-latest.json`.
- `Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json`.
- `Data Sets/Networks/4_communities/<backbone>/Full_<backbone>_author_communities.json`.
- `Data Sets/Networks/3_backbones/Full_<backbone>_InfoFlow.gml`.

Use a new OUT directory for each production run. Inside the repository, outputs must
be under ignored `data_sets/`. Original classified files must be validated and repaired
before affect aggregation. Overrides: `--tweets`, `--sentiment`, `--emotions`,
`--communities`, `--graph`. Author IDs are exact decimal strings; GML uses `label`
by default (`--author-id-attribute` overrides this).

## Run one backbone

```sh
python -m src.author_analysis.run prepare --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run topics --data-root DATA --output-root OUT --backbone RetweetedOnce --jobs 8
python -m src.author_analysis.run affect --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run match --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run summarize --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run communities --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run layout --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run export --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run figure-inputs --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run projection --data-root DATA --output-root OUT --backbone RetweetedOnce
```

Repeat with `LWCC`. Results are separate under `OUT/networks/<backbone>/`.
`--check` checks required input presence without writing. `--smoke` uses a limited
corpus and reduced grid under `OUT/smoke/networks/<backbone>/`; pass it consistently
to every stage in a smoke run. Smoke results are not production results.

`prepare` cleans raw text with the shared notebook rules and requires at least three
nonempty cleaned posts per author. `topics` uses TF-IDF unigram and unigram/bigram
representations, K=8/12/16/20, alpha=0.05/0.1/0.3, eta=0.01/0.1, seed 42,
min_df=5, max_df=0.7 and 50 batch-LDA iterations. Selection uses a seeded sample of
up to 100,000 authors and up to 20,000 coherence documents. The combined score keeps
the notebook weights: coherence 0.25, diversity 0.30, non-collapse 0.15, evenness 0.30.
The selected configuration is refitted on all qualifying authors of that backbone.
The final fit records dominant-topic counts, mean memberships and empty feature rows.
A final dominant-topic share above 90%, or an increase above 15 percentage points
relative to selection, requires review and stops dependent analysis. These are
review triggers, not requirements that real topics be balanced. The fitted outputs
are preserved for diagnosis. Passing them does not establish semantic quality or
stability; inspect terms and representative posts before interpreting topics.
CPU workers are controlled by `--jobs`; the default is the Slurm allocation or one.
The full grid and final fit should first be benchmarked on a smoke run.

Optional `--model-config settings.json` overrides fields of `ModelSettings` in
`topic_model.py`; pass the same file to preparation and training. For example,
`{"selected_model": ["tfidf_unigram", 12, 0.1, 0.1]}` chooses the final configuration
explicitly after recording grid results. Settings and file hashes are saved with each
model. A limited corpus is accepted only with `--smoke`.

`affect` averages all canonical posts for the topic-eligible authors, using the same
all-post inclusion rule for sentiment and emotions. It checks post ID, author, text,
type, date, duplicate IDs, score validity and complete coverage. Missing posts fail;
authors are not silently dropped. Affect counts can exceed topic-document counts when
some canonical posts become empty after topic cleaning. `match` binds the affect
outputs to the fitted author table and preserves community assignments. These are
content contributed and amplified, not direct measurements of personal beliefs.

Topics and labels are specific to each fitted model. Numeric topic labels are the
default; provide `--topic-labels labels.csv` (`topic_id,label`) to summaries,
communities, layout and figure-inputs for the exact model. Do not equate topic numbers
between backbones. `--community-algorithm` selects the comparison partition; the
default is `leiden_directed`. Constant-score standardised differences and effect sizes
are undefined and recorded as missing values, not as zero effects.

## Figures and weekly analysis

```sh
python src/blog_analysis/render_non_temporal_figures.py --input-root OUT/networks/RetweetedOnce/figure_inputs --output-root OUT/networks/RetweetedOnce/figures --figures umap topic_sentiment community_topic_enrichment community_sentiment topic_weighted_net emotion_sd
python src/blog_analysis/weekly_aggregation.py --backbone RetweetedOnce --matched-authors OUT/networks/RetweetedOnce/figure_inputs/matched_author_ids.json --posts "DATA/Data Sets/Cleaned Data/AItrust_twits_pruned_dict.json" --sentiment-classified "DATA/Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-sentiment-latest.json" --emotion-classified "DATA/Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json" --model-dir OUT/networks/RetweetedOnce/topics --out-dir OUT/networks/RetweetedOnce/weekly
python src/blog_analysis/render_temporal_figures.py --weekly-dir OUT/networks/RetweetedOnce/weekly --output-root OUT/networks/RetweetedOnce/weekly_figures --topic-labels OUT/networks/RetweetedOnce/figure_inputs/topic_labels.csv --collection-end TIMESTAMP
```

TIMESTAMP is the actual timezone-aware collection end. Weekly topics use the saved
vectorizer/model and shared cleaning, including retweets, without refitting. Weekly
affect averages post scores within author-week, then gives each active author equal
weight. These weekly aggregation rules extend the backbone notebook. Use only the
validated classified inputs supplied to the affect stage.

See [viewer setup](../../visualizations/author_network/README.md) for native 3D
and offline display. No classifier inference, network reconstruction or community
detection is repeated by the downstream figure/viewer stages.

Tests: `python -m unittest discover -s tests -v`. Set `SFDP_RUNNER` to the compiled
Graphviz helper for the native 3D integration test. Generated data, models, figures,
run records and exploratory notes stay outside version control.

## Supplied twelve-topic configuration

config/retweeted_once_k12 describes 164,291 eligible RetweetedOnce authors.
All 202,710 network authors belong to the full network's largest weakly connected
component. The model uses unigram TF-IDF, K=12, alpha=0.3, eta=0.1, seed=42 and
50 maximum iterations. It is selected by semantic review, not the automatic grid winner.

Pass --model-config config/retweeted_once_k12/model.json to prepare and topics.
Pass --topic-labels config/retweeted_once_k12/topics.csv to downstream stages.
For figure-inputs and export, also pass --community-labels config/retweeted_once_k12/communities.csv.
Counts are checked against the actual population. See the [figure commands](../../docs/blog_analysis/README.md).

provenance.json records input/model hashes, software versions, author counts, review
qualifications and result links. Add --provenance config/retweeted_once_k12/provenance.json
to downstream run commands to bind presentation to the exact supplied fit.
Do not reuse its model-specific names for LWCC or another fit. Refitting can differ
across dependency versions/platforms even with the same seed; review terms and posts.

A controlled seed-42/43 comparison gives 76.5% leading-topic agreement after
author-membership cosine matching. Some matched topics differ in meaning.
Five of 54 substantial community-topic enrichments reverse direction between fits
(community size >=1,000 and absolute log2 enrichment >=0.5 in both).
These are descriptive filters, not universal robustness thresholds.
All-post profiles include amplification; topic-weighted affect is author-level,
not sentiment of posts about a topic.

## Controlled stability checks

Use trusted bundles containing model.json, author_topics.csv, lda_model.pkl and
vectorizer.pkl. A repeat also needs the reference author_documents.jsonl.
Pickles execute Python when loaded; use your trusted analysis artifacts.

    python -m src.author_analysis.check_stability --reference OUT/networks/RetweetedOnce/topics --repeat-seed 43 --jobs 8 --profiles OUT/networks/RetweetedOnce/data/matched_authors.parquet --output DIAGNOSTIC

The repeat keeps the exact fitted vocabulary and IDF and does not replace the reference.
For saved fits, replace --repeat-seed 43 with --candidate CANDIDATE.
Outputs include leading-topic agreement, adjusted Rand index, membership total
variation and optional affect/enrichment comparisons. The output directory must be new.
Inspect terms and posts separately: membership matching is not semantic matching.

## Adopt an explicitly reviewed saved fit

    python -m src.author_analysis.adopt_reviewed_fit --source SAVED_TOPICS --prepared PREPARED_TOPICS --review REVIEW.json --output NEW_OUT/networks/RetweetedOnce/topics

REVIEW.json must identify model_sha256 and set semantic_review_completed,
sensitivity_review_completed and approved_for_exploratory_use to true.
The supplied provenance.json applies only to its exact identified fit.
PREPARED_TOPICS contains author_documents.jsonl and documents.json from prepare.

Adoption verifies fingerprints, authors, numerical quality and a deterministic
sample of saved probabilities against the fitted model. It refuses existing output
directories, unreviewed fits and diagnostic-only bundles. Source files stay intact.
Run affect and downstream stages against NEW_OUT; cached results are not silently copied.
Adoption does not trigger a model sweep or full-corpus fit.
