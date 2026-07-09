---
status: todo
type: plan
id: twitter.topic_emotion_mixing
description: Implementation plan for fusing author-level LDA topic distributions (04c v2) with author-level 11-class emotion intensities (01b v4 multilabel) into topic-by-emotion joint matrices via A.T @ R. Builds two complementary joints — a row-stochastic "shape" joint P(emotion|topic) and an un-normalized "intensity" joint (mean sigmoid per emotion, incl. a net trust-minus-fear index) — with author-universe alignment, diagnostics, and visualization.
label: [planning, backend]
injection: informational
volatility: initial_draft
last_checked: '2026-07-09'
---
# Topic–Emotion Mixing (author-level, 11-class emotion)

Fuse the per-author **topic distribution** (output of the author-dictionary LDA, [notebooks/03_Analysis_and_Modeling/04c_lda_author_topics_v2.ipynb](../notebooks/03_Analysis_and_Modeling/04c_lda_author_topics_v2.ipynb)) with the per-author **emotion profile** (aggregated from the per-tweet 11-class multilabel emotion classifier in [notebooks/03_Analysis_and_Modeling/01b_sentiment_emotion_v4_hpc.ipynb](../notebooks/03_Analysis_and_Modeling/01b_sentiment_emotion_v4_hpc.ipynb)) into `T × M` matrices that summarize which emotions co-occur with which topics across the author population. The core operation is a matrix product; the rest of the plan handles model selection, building the emotion matrix from per-tweet scores, input alignment, the two normalization views, diagnostics, and persistence so the result is interpretable rather than a number-blob.

**Scope decisions (locked 2026-07-09):**

- **Level = author-level.** We compute `Aᵀ @ R`, matching the artifacts already on disk (04c persists the author–topic matrix `θ`; it does **not** persist the fitted LDA/vectorizer, so per-tweet topic vectors from the v2 model are not available without re-running it). Author-level fusion is the *conditional-independence (mean-field) estimate* of the topic–emotion joint — see Mathematical Setup. The exact tweet-level joint (`Σ_tweets P(t|tweet)·P(m|tweet)`) is the stronger claim but is **out of scope here**; the gap between the two *is* the within-author topic–emotion correlation, left unmeasured at this level.
- **Signal = 11-class emotion only.** The multilabel emotion model `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest` (labels: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust). The 3-class sentiment model is a separate, cleaner fusion (single-label softmax, naturally row-stochastic) and is **not covered here**. Note **trust** and **fear** are emotion labels directly, so the intensity joint speaks to the *AI Public Trust* research question head-on.
- **Corpus = AI-General.** The v2 author-LDA was built on `AItrust_twits_pruned_dict.json` (AI-General), so `A`'s author universe is AI-General. Build `R` from `ai_full_classified_…emotion-multilabel-latest.json`. The AI+Art subset has no author-topic matrix and is excluded.

## Mathematical Setup

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.math_setup
estimate: 0h
```

This task is documentation-only — it pins down the notation the implementation tasks refer to. No code is produced here.

Let:

```text
N = number of authors (intersection of A and R universes; see Validate & align)
T = number of topics   (from the selected author-dictionary LDA model)
M = 11 emotions         (multilabel classifier label set, fixed order)

A ∈ R^{N×T}    rows L1-normalized, A[n, t] = P(topic=t | author=n)      (LDA θ, already normalized)

R ∈ R^{N×M}    R[n, m] = mean over author n's tweets of  sigmoid_score(emotion=m)   ∈ [0, 1]
                 Built from the per-tweet `scores` (all 11 sigmoids), NEVER the thresholded `labels`,
                 so it is independent of the 0.5 decision threshold.
                 Rows are NOT normalized — each entry is a mean per-emotion intensity.

S ∈ R^{N×M}    rows L1-normalized, S[n, :] = R[n, :] / Σ_m R[n, m]      "emotional signature share"
```

Because the emotion model is **multilabel** (independent sigmoids), a tweet can fire zero, one, or several emotions, so the naïve `(#tweets labeled m)/(#tweets)` is **not** row-stochastic. Averaging the raw soft scores into `R` and normalizing once (into `S`) fixes this while retaining confidence information. Do **not** renormalize per tweet (`scores/scores.sum()` on each tweet) — AI discourse is heavy with affect-flat news tweets, and per-tweet renormalization makes a near-zero tweet vote for a near-*uniform* distribution, biasing every author toward uniform.

**Two complementary joints.** Only the first needs the row-stochastic step:

```text
Topic mass:   m_t = Σ_n A[n, t]                              "authorial mass of topic t"

SHAPE joint (relative profile):
    E_shape = Aᵀ @ S                         shape (T, M);   row t sums to m_t
    P(emotion | topic) = E_shape / m_t[:, None]              rows sum to 1
    P(topic | emotion) = E_shape / E_shape.sum(axis=0)       cols sum to 1

INTENSITY joint (absolute, in sigmoid units):
    E_int = (Aᵀ @ R) / m_t[:, None]          shape (T, M);   E_int[t, m] ∈ [0, 1]
          = topic-mass-weighted average intensity of emotion m among authors of topic t
    net_trust_fear[t] = E_int[t, 'trust'] − E_int[t, 'fear']  "headline AI-trust index per topic"
```

Why keep both: row-normalizing into `P(emotion|topic)` forces a **zero-sum reallocation** across the 11 emotions, which *erases how emotional a topic is* — a dry technical topic and an incendiary one can have identical shapes. `E_int` preserves absolute emotional salience (a topic can be high or low on everything), which for a trust/fear question is as important as the relative shape. Lead with `E_int`; keep `P(emotion|topic)` as the "what's the emotional signature" view.

**Mean-field interpretation (state honestly in any write-up).** Drawing a random author `n` uniformly, `E_shape[t,m] = Σ_n P(t|n)·P(m|n) = N·P(topic, emotion)` **only if topic and emotion are independent within an author**. So `Aᵀ@S` credits an author's emotion to *all* their topics proportionally — an author who writes 60% "AI safety" / 40% "AI art" and is generally anxious contributes anxiety to both, even if it is entirely about safety. This is the fundamental limitation of the author-level view.

**Optional author-volume weighting.** Each author contributes equally because `A` is row-stochastic (`R` is per-author-averaged). If a 5,000-tweet author should outweigh a 5-tweet author, introduce weights `w ∈ R^N` (`w_n = tweet_count_n / Σ tweet_count`) and replace `A` with `A * w[:, None]` throughout (recomputing `m_t` accordingly). AI Twitter has prolific, affect-flat news/bot accounts, so expect the tweet-weighted view to look *less* emotional than the per-author view; the divergence between them is a first-class diagnostic (see Diagnostics). Keep unweighted and weighted as **separate** outputs — the comparison is the point.

## Select the author-topic model `A`

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.select_model
estimate: 2h
```

04c produces a **108-model grid** (4 representations × K∈{5,12,20} × α∈{0.01,0.1,0.5} × η∈{0.01,0.1,0.5}), each saved as `author_topic_matrix_{rep}_k{K}_a{a}_e{e}.csv`. Fusion needs **one** `A` (2–3 candidates for robustness is fine). Selection is not purely a coherence contest:

- **Coherence** — start from `ALL_REPRESENTATIONS_LDA_FULL_GRID.csv`; the notebook already ranks the top models by `c_v`.
- **Small α is preferable *for fusion*, even at a small coherence cost.** Small `doc_topic_prior` (α = 0.01) makes each author's topic row *peaked* → authors are effectively single-topic → `Aᵀ@S` blurs emotion across an author's topics *less* (tightening the mean-field confound above). Large α (0.5) makes rows diffuse and maximizes the confound. This criterion is invisible in the coherence grid, so apply it explicitly.
- **K interpretability** — prefer K where topics stay legible and per-topic effective author counts (see Diagnostics) don't get thin; K=20 tends to fragment. K∈{5,12} is the likely sweet spot.
- **Name the topics.** Read the matching `author_topics_{rep}_k{K}_…_top_terms.csv` and assign a short human label per topic. Without names the heatmaps are unreadable. Persist the `topic_index → label` mapping alongside the chosen `A`.

Record the chosen `(representation, K, α, η)` and topic labels; downstream filenames encode them.

## Build the author emotion matrix `R`

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.build_emotion_matrix
estimate: 2h
```

`R` does not exist on disk — 01b writes only per-tweet JSONL. Build it with a single streaming pass over the classified AI-General file (~17.4M lines), restricting accumulation to the authors present in the selected `A` (cuts memory and makes the intersection explicit). Use `scores` (all 11 sigmoids), never `labels`.

### Sample code

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

EMOTION_LABELS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                  'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
EMO_MODEL = 'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest'

def build_emotion_matrix(classified_path, author_ids,
                         labels=EMOTION_LABELS, model=EMO_MODEL):
    """Stream per-tweet emotion classifications and average the *soft* sigmoid
    scores per author. Uses `scores` (all 11), never the thresholded `labels`,
    so the result is threshold-independent.

    Returns R_df: index=author_id (str), columns=labels, values in [0, 1].
    Rows are NOT normalized — each entry is a mean per-emotion intensity.
    """
    keep = set(map(str, author_ids))                 # dtype-safe restriction to A's authors
    m = len(labels)
    sums   = {a: np.zeros(m) for a in keep}
    counts = {a: 0 for a in keep}

    with open(classified_path, 'r', encoding='utf-8') as f:
        for line in f:
            tw = json.loads(line)
            aid = str(tw.get('author_id', '')).strip()
            if aid not in keep:
                continue
            res = tw.get('classifications', {}).get(model)
            if not res:
                continue
            sc = res.get('scores', {})
            sums[aid]   += np.array([sc.get(lbl, 0.0) for lbl in labels])
            counts[aid] += 1

    rows = {a: sums[a] / counts[a] for a in keep if counts[a] > 0}
    R_df = pd.DataFrame.from_dict(rows, orient='index', columns=labels)
    R_df.index.name = 'author_id'
    return R_df
```

Also emit a `tweet_counts` Series (the `counts` dict) — it is the volume-weighting vector and a diagnostic input.

## Validate and align inputs

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.validate_inputs
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.select_model, twitter.topic_emotion_mixing.build_emotion_matrix]
```

`A` and `R` come from separate pipelines. Align them explicitly before any multiplication — silent broadcasting on misaligned axes is the failure mode this task exists to prevent.

### Checks

- **Author-id dtype.** `A` is read from CSV (`author_id` parses as **int64**); `R` is keyed by **str** from JSONL. Coerce both indices to `str` *before* intersecting, or the intersection is silently empty — the exact trap the 04c author_id type-check guards against.
- **Same author universe.** Intersect indices; this restricts the analysis to `A`'s authors (LCC ∩ ≥3 tweets — a *connected, active* subpopulation; note as a scope caveat, not a bug). Drop, do not impute.
- **`A` row-stochastic** within `1e-6`; rows that violate are usually degenerate — drop them.
- **`R` bounds.** All entries in `[0, 1]` (they are mean sigmoids). `R` rows are **not** expected to sum to 1.
- **Affect-flat authors.** Drop authors whose total intensity `Σ_m R[n, m] ≈ 0` — `S` (= row-normalized `R`) would otherwise be undefined.
- **No NaNs** after the join; `A.shape[0] == R.shape[0] == N`.

### Sample code

```python
def align_and_validate(A_df, R_df, atol=1e-6):
    """Align two author-indexed matrices. A must be row-stochastic; R holds
    per-emotion mean sigmoids in [0, 1] (rows NOT normalized)."""
    A_df, R_df = A_df.copy(), R_df.copy()
    A_df.index = A_df.index.astype(str)              # int64 (CSV) vs str (JSONL) — coerce both
    R_df.index = R_df.index.astype(str)

    common = A_df.index.intersection(R_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping authors — check author_id dtype coercion first.")

    A, R = A_df.loc[common], R_df.loc[common]

    if A.isna().any().any() or R.isna().any().any():
        raise ValueError("NaN present after alignment; inspect upstream pipelines.")
    if not np.allclose(A.sum(axis=1), 1.0, atol=atol):
        raise ValueError("A is not row-stochastic; topic rows must sum to 1.")
    if ((R.values < -atol) | (R.values > 1 + atol)).any():
        raise ValueError("R has values outside [0, 1]; expected mean sigmoids.")

    flat = R.sum(axis=1) < atol                      # affect-flat authors → S undefined
    if flat.any():
        keep = ~flat
        A, R, common = A[keep], R[keep], common[keep]

    return (A.to_numpy(), R.to_numpy(),
            common, A.columns, R.columns)            # author, topic, emotion labels
```

## Compute the joint matrices

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.joint
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.validate_inputs]
```

Compute both the shape joint (`Aᵀ @ S`) and the intensity joint (`Aᵀ @ R`), plus their volume-weighted variants when `tweet_counts` is available. Keep weighted and unweighted as separate outputs — the comparison *is* the diagnostic for whether tweet volume matters.

### Sample code

```python
def compute_joints(A, R, tweet_counts=None, eps=1e-12):
    """Author-level topic×emotion fusion → two complementary joints.

    Returns:
      topic_mass              (T,)     Σ_n A[n, t]
      E_shape                 (T, M)   A.T @ S ; row t sums to topic_mass[t]
      p_emotion_given_topic   (T, M)   row-normalized E_shape (rows sum to 1)
      E_int                   (T, M)   mean sigmoid intensity per (topic, emotion), in [0, 1]
    plus *_w weighted variants when tweet_counts is supplied.
    """
    S = R / R.sum(axis=1, keepdims=True)                     # row-stochastic signature
    topic_mass = A.sum(axis=0)                               # (T,)

    E_shape = A.T @ S                                        # (T, M)
    p_egt   = E_shape / np.maximum(topic_mass[:, None], eps) # P(emotion | topic)
    E_int   = (A.T @ R) / np.maximum(topic_mass[:, None], eps)

    out = {"topic_mass": topic_mass, "E_shape": E_shape,
           "p_emotion_given_topic": p_egt, "E_int": E_int}

    if tweet_counts is not None:
        w = tweet_counts / tweet_counts.sum()
        Aw = A * w[:, None]
        mass_w = Aw.sum(axis=0)
        out["p_emotion_given_topic_w"] = (Aw.T @ S) / np.maximum(mass_w[:, None], eps)
        out["E_int_w"]                 = (Aw.T @ R) / np.maximum(mass_w[:, None], eps)

    return out
```

## Derive conditional distributions

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.conditionals
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.joint]
```

`E_shape` is most readable as a conditional distribution. `P(emotion|topic)` is already produced in the joint step (row-normalized); add the column-normalized direction. The intensity joint `E_int` is already in interpretable units and needs no conditional transform.

### Sample code

```python
def conditionals(E_shape, eps=1e-12):
    """Both conditional views of the shape joint. Each answers a different question."""
    # P(emotion | topic): each ROW is a distribution over emotions.
    p_emotion_given_topic = E_shape / np.maximum(E_shape.sum(axis=1, keepdims=True), eps)
    # P(topic | emotion): each COLUMN is a distribution over topics.
    p_topic_given_emotion = E_shape / np.maximum(E_shape.sum(axis=0, keepdims=True), eps)
    return {"p_emotion_given_topic": p_emotion_given_topic,   # rows sum to 1
            "p_topic_given_emotion": p_topic_given_emotion}   # cols sum to 1
```

## Diagnostics

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.diagnostics
estimate: 2h
blocked_by: [twitter.topic_emotion_mixing.joint]
```

Without sanity checks the joints are number-blobs. These surface the failure modes that make the result misleading rather than wrong.

### Checks

- **Per-topic entropy of `P(emotion | topic)`** — a topic whose emotion distribution matches the global marginal is uninformative. Compare each row's entropy to the entropy of the marginal `P(emotion)`; rows near the marginal have "no emotional signature distinct from the average author."
- **Total emotional intensity per topic** — `Σ_m E_int[t, m]`. The shape view hides this; a topic can be emotionally flat overall yet have a peaked *relative* profile. Report it so "distinctive shape" isn't confused with "emotionally charged."
- **Net trust−fear per topic** — `E_int[t,'trust'] − E_int[t,'fear']`. The headline index for *AI Public Trust*; rank topics by it.
- **Topic mass concentration** — `Σ_n A[n, t]`. If one topic captures most authorial mass, smaller topics' conditional rows rest on few effective authors → flag low-confidence.
- **Effective sample size per topic** — count authors with `A[n, t] > 0.1`. Topics below ~30 are thin evidence regardless of total mass.
- **Sensitivity to volume weighting** — KL divergence between `p_emotion_given_topic` and its weighted counterpart (a distribution-to-distribution comparison); for the intensity joint use mean-abs-diff or correlation between `E_int` and `E_int_w` (intensities aren't distributions). Large divergence ⇒ result depends strongly on author-vs-tweet unit choice.

### Sample code

```python
from scipy.stats import entropy

def diagnostics(A, joints, topic_labels, emotion_labels, threshold=0.1):
    """Per-topic diagnostics — one row per topic for easy inspection."""
    p    = joints["p_emotion_given_topic"]           # (T, M)
    Eint = joints["E_int"]                            # (T, M)
    tl   = list(emotion_labels)
    ti, fi = tl.index("trust"), tl.index("fear")

    marginal = p.mean(axis=0)                         # rough P(emotion)
    base_H   = entropy(marginal)
    row_H    = np.array([entropy(r) for r in p])

    return pd.DataFrame({
        "topic": topic_labels,
        "n_authors_above_threshold": (A > threshold).sum(axis=0),
        "total_topic_mass": A.sum(axis=0),
        "emotion_entropy": row_H,                     # lower = more distinctive shape
        "entropy_delta_vs_marginal": base_H - row_H,  # positive = more peaked than average
        "total_emotional_intensity": Eint.sum(axis=1),
        "net_trust_minus_fear": Eint[:, ti] - Eint[:, fi],
    }).sort_values("net_trust_minus_fear")
```

## Visualize

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.viz
estimate: 2h
blocked_by: [twitter.topic_emotion_mixing.conditionals, twitter.topic_emotion_mixing.diagnostics]
```

- **Intensity heatmap (lead):** `E_int` — rows = topics (named), columns = the 11 emotions, cell color = mean sigmoid intensity. Order rows by hierarchical clustering on the emotion profiles so emotionally-similar topics sit together. This is the single view that makes the result legible. Save `figures/topic_emotion_intensity_heatmap.{png,pdf}`.
- **Net trust−fear bar (headline):** per-topic `net_trust_minus_fear`, sorted — the direct AI-trust ranking. Save `figures/topic_net_trust_minus_fear.{png,pdf}`.
- **Shape heatmap:** `P(emotion | topic)` — same layout, for "relative emotional signature." Keep alongside the intensity heatmap so shape and magnitude are read together.
- **Entropy diagnostic bar:** per-topic emotion entropy — fastest way to spot uninformative topics.

## Persist outputs

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.persist
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.conditionals, twitter.topic_emotion_mixing.diagnostics]
```

Save with topic and emotion labels attached (parquet preserves them; npz does not). Encode the source model `(rep, K, α, η)` and the weighting choice in filenames so runs coexist:

- `R__{model}.parquet` — author × emotion mean-sigmoid matrix (the reusable intermediate)
- `E_int__{model}__{weighting}.parquet` — intensity joint `(T, M)`, incl. a `net_trust_minus_fear` column
- `p_emotion_given_topic__{model}__{weighting}.parquet` — shape joint, row-conditional
- `p_topic_given_emotion__{model}__{weighting}.parquet` — shape joint, column-conditional
- `diagnostics__{model}__{weighting}.parquet` — per-topic sanity + intensity + net-trust table
- `topic_labels__{model}.json` — the `topic_index → human label` mapping from Select-model

where `{weighting} ∈ {unweighted, volume_weighted}` and `{model}` e.g. `bow_bigram_k12_a0p01_e0p1`.

## Knowledge capture decision

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.kb_capture
estimate: 0.5h
```

Per MDDIA convention §11 (plans plan for knowledge capture):

- **Existing how-to relevant?** Check the KB for a fusion / co-occurrence-matrix how-to. If one exists, add a task here to update it with what the diagnostics reveal about failure modes (multilabel normalization, affect-flat dilution, mean-field confound).
- **New how-to warranted?** The reusable pattern here is narrower than generic `Aᵀ@S`: it is *"fuse a row-stochastic author-topic matrix with a per-author soft-score matrix built from a multilabel classifier, keeping both a normalized shape joint and an un-normalized intensity joint."* If the project will repeat it (topics × sentiment, topics × stance, topics × demographics), scaffold `TOPIC_FUSION_SKILL.md` as `volatility: initial_draft` and add a task to populate it once this plan completes.
- **Neither?** Mark this task `done` with a short rationale in the worklog and move on.

Decision deadline: before the joint-computation task closes — by then the generality of the pattern is clear.
