# Validated community analysis and blog figures

This package integrates the completed network-aligned Twitter/X analysis for the **198,326 matched authors**. It contains compact derived tables, validated community labels, final light/dark figures and path-portable post-processing scripts. It deliberately excludes raw posts, author-level exports, network files, model files and the separate interactive-visualiser codebase.

The interactive DrL network map remains separately maintained at [davidfreeborn.github.io/twitter-authors-map](https://davidfreeborn.github.io/twitter-authors-map/).

## Analytical scope

- Collection window: **30 October 2022 00:00 UTC–27 February 2023 12:00 UTC**. The last ISO week is incomplete.
- Matched sample: **198,326 authors** with fixed topic, sentiment, emotion and Leiden-community assignments.
- Displayed communities: the 21 largest existing Leiden-directed communities, covering **188,765 authors (95.2%)**.
- Weekly unit: post scores are averaged within author-week, then each qualifying author contributes once to the weekly mean. Weeks begin Monday in UTC.
- Topic model: the existing fixed twelve-topic model and dictionary. No model fitting occurs in the included weekly or figure scripts.
- UMAP: a presentation-only projection of the fixed 12-score author profiles. It is not a clustering result. The renderer requires the existing fixed-seed projection and cannot fit UMAP.

The fixed topic order and colours are:

| ID | Topic | Colour |
|---:|---|---|
| 0 | Marketing/Social | `#e31a1c` |
| 1 | AI Art Discourse | `#387db8` |
| 2 | AI Tools/Code | `#4db04a` |
| 3 | Bard/LLMs | `#994fa3` |
| 4 | Visual AI Art | `#ff8000` |
| 5 | Bot/Spam | `#ffff33` |
| 6 | Tech/Programming | `#a65729` |
| 7 | News/Updates | `#f782bf` |
| 8 | NFT/Crypto | `#999999` |
| 9 | Web3/DeFi | `#66c2a6` |
| 10 | General AI | `#fc8c61` |
| 11 | Trading/Invest | `#8ca1cc` |

Sentiment uses the agreed traffic-light scheme: positive `#2ca25f`, neutral `#e3b505`, and negative `#d73027`. Topic, community, sentiment and emotion colours match the deployed visualiser at 8-bit display precision.

## Community labels and validation

Labels describe **prevalent post content**, not every author, every post, or authors' identities, occupations, locations, intentions or relationships. They were drafted from class-based TF-IDF terms and then conservatively revised. Fourteen labels with a stance, geographic, behavioural or identity implication received a deterministic cross-author audit: **25 posts from 25 distinct authors per community (350 posts total)**. The public repository retains aggregate sample diagnostics and the accepted evidence summaries; raw text and author/post identifiers are intentionally excluded.

For stance-sensitive labels, sampled posts were reviewed as supportive, opposed, mixed/argumentative or unclear. Geographic review separated discussion about a place, regionally associated language, explicit self-reported location and unsupported inferred location. Promotional/behavioural review recorded link, hashtag, promotional-language, template-duplication and account-concentration signals. Repetition or promotion alone was not treated as proof of automation or coordination.

| Community ID | Full validated label | Short display label | Authors | Confidence | Status |
|---:|---|---|---:|---|---|
| 1 | General ChatGPT/AI discussion | General ChatGPT Discussion | 44718 | medium | validated |
| 0 | Anti-AI-art discourse | Anti-AI-Art Discourse | 41574 | high | validated |
| 3 | General ChatGPT/AI debate | General ChatGPT Debate | 17422 | low | provisional |
| 8 | AI-art and NFT posting | AI-Art & NFT Posting | 12679 | high | validated |
| 2 | ChatGPT for SEO and content marketing | ChatGPT SEO/Content | 10348 | high | validated |
| 12 | AI-token/launchpad and general-AI posting | AI Tokens, Launchpads & AI | 7502 | medium | validated |
| 4 | AI/crypto launchpad and airdrop promotion | AI/Crypto Launchpads | 7497 | high | validated |
| 9 | AI-blockchain/token promotion | AI-Blockchain Promotion | 7256 | high | validated |
| 16 | Data-science/ML technology headlines | Data Science/ML Headlines | 7132 | high | validated |
| 7 | AI-art/authenticity disputes | AI-Art Authenticity Disputes | 5490 | medium | validated |
| 10 | AI news and tech press coverage | ChatGPT & AI/Tech News | 4542 | high | validated |
| 6 | Mixed AI art and ChatGPT discourse | Mixed AI Art/ChatGPT | 4098 | low | validated |
| 5 | Mixed NFT, AI-token and general-AI posting | NFTs, AI Tokens & AI | 3607 | medium | validated |
| 18 | ChatGPT in education and teaching | ChatGPT in Education | 3281 | high | validated |
| 14 | India-linked ChatGPT/AI news | India-Linked ChatGPT/AI News | 2834 | low | provisional |
| 13 | ChatGPT discussion with regional-language markers | ChatGPT + Regional-Language Markers | 2699 | low | provisional |
| 15 | AI-art criticism and pop-culture mix | AI-Art Criticism & Pop Culture | 2005 | low | provisional |
| 11 | Anime AI art (Stable Diffusion) | Anime AI Art/Stable Diffusion | 1245 | high | validated |
| 17 | Mixed/general ChatGPT discourse | Mixed ChatGPT Usage | 1166 | low | validated |
| 24 | Medical imaging and radiology AI | Medical Imaging AI | 1060 | high | validated |
| 28 | Mixed AI/creative-content posting | Mixed AI/Creative Content | 610 | low | provisional |

C3, C13, C14, C15 and C28 remain explicitly provisional. The canonical label table, c-TF-IDF terms, revision history and de-identified audit diagnostics are under [`outputs/community_analysis/`](../../outputs/community_analysis/).

### Association results

Across the top 21 communities, community membership was associated on average with **21.1% of topic-weight variation**, **32.6% of sentiment-probability variation**, and **27.6% of emotion-score variation** (omega-squared). The strongest topic associations were Web3/DeFi (63.3%), AI Art Discourse (53.4%) and NFT/Crypto (33.1%); the strongest emotion associations were trust (44.3%), disgust (41.7%) and anger (39.7%). Dominant topic and community membership had Cramer's V = 0.330. These are descriptive associations, not causal effects or measures of label purity.

## Final figure set

All figure variants live under [`outputs/blog_figures/`](../../outputs/blog_figures/), separated into `light/png`, `light/svg`, `dark/png` and `dark/svg`. Each figure has one compact CSV under `data/`.

| # | Figure | Key result or role | Necessary caveat |
|---:|---|---|---|
| 1 | AI-development timeline | Eight verified public events provide context for the collection window. | Selective context; it does not establish causal effects on posting. |
| 2 | Weekly sentiment | Complete-week net sentiment ranged from -0.022 to +0.387. | Variation combines within-author change and changes in participating authors; the incomplete final week is excluded visually but retained in the CSV. |
| 3 | Weekly emotions | Joy had the largest complete-week range, 0.393–0.643. | Eleven emotion probabilities are non-exclusive and use separate panel scales. |
| 4 | UMAP of fixed topic profiles | AI Art Discourse was dominant for 120,665 authors (60.8%); General AI for 38,980 (19.7%). | Existing fixed-seed projection only; coloured overlay is capped at 5,000 authors per topic and does not encode prevalence. Author-level coordinates are not committed. |
| 5 | Weekly topic prevalence | AI Art Discourse had the largest author-week-weighted share (46.6%). | Uses the frozen K=12 model; changes describe active-author composition, not newly fitted topics. |
| 6 | Topic sentiment and net | Net sentiment was highest for Web3/DeFi (+0.719) and lowest for AI Art Discourse (-0.143). | Topic membership and sentiment are author-level aggregates rather than post-level topic/sentiment labels. |
| 7 | Community-topic enrichment | Strongest over-representation was AI/Crypto Launchpads × Web3/DeFi (+3.82 log2). | Enrichment is relative; small absolute topic mass can produce large ratios. |
| 8 | Community sentiment | Highest positive mean: AI/Crypto Launchpads (0.769); highest negative mean: Anti-AI-Art Discourse (0.500). | Community means do not imply homogeneity. |
| 9 | Topic-weighted net sentiment by community | Supported cells ranged from -0.368 to +0.798; 37 of 252 cells are masked. | Figure 9 uses **author-level fuzzy topic weighting**. Cells with weight <10 author-equivalents or Kish effective n <30 are inadequate and masked. It cannot be interpreted as how a community feels “when discussing” a topic. |
| 10 | Within-community emotion variation | Maximum author-level SD was 0.269; median community-emotion SD was 0.103. | Dispersion is not emotional valence. |
| 11 | AI-art community comparison | Negative sentiment: 0.500 vs 0.133; joy: 0.277 vs 0.614. | Both communities remain heterogeneous; labels summarise prevalent content. |

The separate author-coverage diagnostics created during review are intentionally omitted from the main package.

## Reproducibility

The figures described above are saved historical results. New results use the
single backbone-specific workflow in [the author-analysis instructions](../../src/author_analysis/README.md).
That workflow includes retweets, fits separate topic models for RetweetedOnce and
LWCC, aggregates verified sentiment and emotion scores, and exports the figure inputs.
The notebook calls the same implementation as the command-line runner.

Use the current commands in those instructions to regenerate non-temporal and weekly
figures. Weekly inference loads the selected backbone's saved scikit-learn vectorizer
and model. It uses the shared text cleaning and all post types; it does not refit the
model. Temporal rendering requires the actual collection-end timestamp. Old topic
labels, sample counts and community interpretations must not be transferred to a new
fit. Historical committed figures remain unchanged until explicitly regenerated.
