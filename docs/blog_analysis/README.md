# Author-analysis figures

The renderers consume explicit tables and saved projections; they do not fit models
or assign communities. Start with the [author workflow](../../src/author_analysis/README.md).
The supplied config/retweeted_once_k12 describes an exploratory twelve-topic fit
for 164,291 eligible RetweetedOnce authors. Its provenance.json links the results.
Generated figures and tables are not tracked in this repository.

## Non-temporal figures

Run from the repository root. DATA and OUT are your input and output directories.
Inside the checkout, generated files belong under ignored data_sets/.

    python src/blog_analysis/render_non_temporal_figures.py --input-root OUT/networks/RetweetedOnce/figure_inputs --output-root OUT/networks/RetweetedOnce/figures --topic-labels config/retweeted_once_k12/topics.csv --labels config/retweeted_once_k12/communities.csv --contrast config/retweeted_once_k12/provenance.json --figures timeline umap topic_sentiment community_topic_enrichment community_sentiment topic_weighted_net emotion_sd community_contrast

The figure-inputs stage supplies matched authors, labels, tables and community
matrices. The projection stage supplies umap_fixed_topic_scores_seed0.csv.
The renderer never fits UMAP. Community contrast reads two communities and selected
topic IDs from the contrast configuration, grouping the remaining topics as Other topics.

## Weekly figures

    python src/blog_analysis/weekly_aggregation.py --backbone RetweetedOnce --matched-authors OUT/networks/RetweetedOnce/figure_inputs/matched_author_ids.json --posts "DATA/Data Sets/Cleaned Data/AItrust_twits_pruned_dict.json" --sentiment-classified "DATA/Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-sentiment-latest.json" --emotion-classified "DATA/Data Sets/Cleaned Data/ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json" --model-dir OUT/networks/RetweetedOnce/topics --out-dir OUT/networks/RetweetedOnce/weekly
    python src/blog_analysis/render_temporal_figures.py --weekly-dir OUT/networks/RetweetedOnce/weekly --output-root OUT/networks/RetweetedOnce/figures --topic-labels config/retweeted_once_k12/topics.csv --collection-end 2023-02-27T12:00:00Z

Supply the actual collection end for other corpora. Weeks begin Monday in UTC.
Affect first averages posts within author-week, then weights active authors equally.
Weekly topics use the saved model/vectorizer on author-week documents, including
retweets. The incomplete final week is recorded but excluded from plotted trends.

## Presentation and interpretation

Both renderers read names and exact hex colours from topics.csv; the viewer exporter
uses that same file. Light and dark PNG/SVG variants are generated together.
Community, sentiment and emotion palettes use the shared style file. Font fallback
is available without Arial; identical pixels also depend on fonts and library versions.

Topics are fuzzy and exploratory, not exclusive author categories. Community labels
describe prevalent content, not every author or their beliefs. Affect values are model
scores, not verified feelings. Topic-weighted affect averages author profiles, not
topic-specific post sentiment. Keep semantic and seed-sensitivity qualifications
alongside interpretation.
