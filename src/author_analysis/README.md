# Author analysis

Python 3.10+ scripts for the independent content branch and its join to either
network backbone. Install dependencies in a virtual environment:

```sh
python -m pip install -r src/author_analysis/requirements.txt
```

Run from the repository root. DATA is the AI Public Trust data directory used by
the notebooks. OUT is a separate writable output directory (use ignored
`data_sets/` if it is inside the repository).

```sh
python -m src.author_analysis.run prepare --data-root DATA --output-root OUT
python -m src.author_analysis.run topics --data-root DATA --output-root OUT
python -m src.author_analysis.run match --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run summarize --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run communities --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run layout --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run export --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run figure-inputs --data-root DATA --output-root OUT --backbone RetweetedOnce
python -m src.author_analysis.run projection --data-root DATA --output-root OUT --backbone RetweetedOnce
```

Use `LWCC` to run the other backbone. Content profiles are shared, while all joined
tables, summaries, coordinates and viewer files are isolated under
`OUT/networks/<backbone>/`. `--check` verifies required file presence without
writing or executing. Actual runs overwrite their named outputs.

`prepare` reads the General AI CardiffNLP emotion and sentiment JSON Lines under
`DATA/Data Sets/Cleaned Data/`. It requires three eligible posts per author and
excludes both `retweeted` and `retweet` records. It writes author means and text
under `OUT/content/`. `topics` fits an independent unigram LDA model there:
K12, alpha0.01, eta0.1, seed42, ten passes, minimum document frequency5 and maximum
fraction0.7. Use `--topics`, `--alpha`, `--eta`, `--seed` and `--min-documents` to
choose parameters explicitly. Models and parameters are written to `content/models/`.

If full-content profiles already exist, skip preparation and fitting and supply
`--profiles-dir PATH` to `match`. This avoids refitting just to change networks.
The directory must contain `ai_general_author_sentiment.csv`,
`ai_general_author_emotions.csv` and `ai_general_author_topics.csv`.

The default network inputs follow the notebooks:

- `Data Sets/Networks/3_backbones/Full_<backbone>_InfoFlow.gml`
- `Data Sets/Networks/4_communities/<backbone>/Full_<backbone>_author_communities.json`

Override these with `--graph` and `--communities`. `--author-id-attribute` defaults
to `label`, which contains the notebooks' exact author IDs. Float IDs, duplicate
IDs/JSON keys, missing edge endpoints and invalid score distributions are rejected.
`--community-algorithm` defaults to `leiden_directed`. The match stage writes
coverage counts as well as the joined table. Community comparisons use up to21
communities, with an up-to15 sensitivity comparison.

Topic labels default to numeric names. Supply `--topic-labels labels.csv` with
`topic_id,label` for the exact model in use to summaries, communities and layout.
A freshly trained model must not inherit labels from a different fit.

`figure-inputs` exports matched IDs, profiles, matrices and labels under
`OUT/networks/<backbone>/figure_inputs/`. `projection` fits a presentation-only
UMAP to existing topic scores (cosine distance, up to30 neighbours, minimum
distance0.1, seed0). It changes no topic score or community assignment.

```sh
python src/blog_analysis/render_non_temporal_figures.py --input-root OUT/networks/RetweetedOnce/figure_inputs --output-root OUT/networks/RetweetedOnce/figures --figures umap topic_sentiment community_topic_enrichment community_sentiment topic_weighted_net emotion_sd
```

The weekly aggregator accepts `figure_inputs/matched_author_ids.json`, the
eligible-post file and classified score files, plus the model/dictionary under
`content/models/`. It excludes both pure-retweet spellings by default and uses
the loaded model's topic count. Render weekly outputs with
`render_temporal_figures.py --weekly-dir PATH --output-root PATH --collection-end TIMESTAMP`;
the timezone-aware timestamp must be the actual collection end.

See [viewer setup](../../visualizations/author_network/README.md) for 3D generation
and display. Tests: `python -m unittest discover -s tests -v`. To also run the native
3D integration test, set `SFDP_RUNNER` to the compiled Graphviz helper. Data, model
binaries, generated figures and execution results are not committed.
