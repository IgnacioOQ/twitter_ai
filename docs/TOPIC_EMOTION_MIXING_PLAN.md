---
status: todo
type: plan
id: twitter.topic_emotion_mixing
description: Implementation plan for fusing author-level LDA topic distributions with author-level emotion distributions into a topic-by-emotion joint matrix via A.T @ S, including normalization, diagnostics, and visualization.
label: [planning, backend]
injection: informational
volatility: initial_draft
last_checked: '2026-05-18'
---
# Topic–Emotion Mixing

Fuse the per-author topic distribution (output of the author-dictionary LDA) with the per-author emotion distribution (aggregated from per-tweet sentiment predictions) into a single `T × M` matrix that summarizes which emotions co-occur with which topics across the author population. The core operation is a matrix product; the rest of the plan handles input alignment, normalization choices, sanity diagnostics, and persistence so the result is interpretable rather than a number-blob.

This plan is the right entry point when the artifact you want is a **population-level** topic-by-emotion summary. If you need per-tweet topic-by-emotion joints (e.g. "Alice writes joyfully about sports but bitterly about politics"), this approach collapses that within-author variation and you should plan a tweet-level fusion instead.

## Mathematical Setup

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.math_setup
estimate: 0h
```

This task is documentation-only — it pins down the notation that the implementation tasks below refer to. No code is produced here.

Let:

```text
N = number of authors
T = number of topics  (from LDA on the author-dictionary)
M = number of emotions (from per-tweet sentiment classifier)

A ∈ R^{N×T}    rows L1-normalized, A[n, t] = P(topic=t | author=n)
S ∈ R^{N×M}    rows L1-normalized, S[n, m] = P(emotion=m | author=n)
                 estimated as: (# tweets by n labeled emotion m) / (# tweets by n)
```

The unnormalized fusion is:

```text
E = A^T @ S         shape (T, M)
E[t, m] = Σ_n A[n, t] · S[n, m]
```

Row sums and column sums carry the meaningful marginals:

```text
Σ_m E[t, m] = Σ_n A[n, t]              "topic-mass" of topic t across authors
Σ_t E[t, m] = Σ_n S[n, m]              "emotion-mass" of emotion m across authors
```

From `E` we derive two conditional distributions:

```text
P(emotion | topic) :  row-normalize  →  E / E.sum(axis=1, keepdims=True)
P(topic   | emotion): col-normalize  →  E / E.sum(axis=0, keepdims=True)
```

Both conditionals throw away information the other retains, so persist both.

**Optional author-volume weighting.** Each author currently contributes equally because both `A` and `S` are row-stochastic. If a 5,000-tweet author should outweigh a 5-tweet author, introduce per-author weights `w ∈ R^N` (e.g. `w_n = tweet_count_n / Σ tweet_count`) and compute:

```text
E_weighted = (A * w[:, None])^T @ S
```

Decide once, up front, whether tweet volume is a feature or a confound, and document the choice in the output filenames.

## Validate and align inputs

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.validate_inputs
estimate: 2h
```

Both `A` and `S` come from separate pipelines (LDA over an author-dictionary, sentiment model over individual tweets). They need explicit alignment before any multiplication — silent broadcasting on misaligned axes is the failure mode this task exists to prevent.

### Checks

- Same author universe: the index of `A` and the index of `S` must contain the same author IDs. Drop authors with no tweets in `S` rather than imputing zeros.
- Same author ordering: reorder one matrix to match the other before stripping indices to numpy.
- Row-stochastic property: `A.sum(axis=1)` and `S.sum(axis=1)` should be `≈ 1` within `1e-6`. Authors that violate this are usually authors with zero tweets (degenerate `S` row); drop them.
- No NaNs after the join.
- Shape sanity: `A.shape[0] == S.shape[0] == N`.

### Sample code

```python
import numpy as np
import pandas as pd

def align_and_validate(
    A_df: pd.DataFrame,     # index = author_id, columns = topic_0..topic_{T-1}
    S_df: pd.DataFrame,     # index = author_id, columns = emotion labels
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, pd.Index]:
    """Align two author-indexed matrices and validate row-stochasticity.

    Returns numpy arrays plus the (author, topic, emotion) labels so that the
    downstream pipeline can re-attach them when persisting results.
    """
    # 1. Restrict to the intersection of authors. Anything we drop here is
    #    typically authors with zero observed tweets in S.
    common_authors = A_df.index.intersection(S_df.index)
    if len(common_authors) == 0:
        raise ValueError("No overlapping authors between topic and emotion matrices.")

    A = A_df.loc[common_authors]
    S = S_df.loc[common_authors]

    # 2. NaN check — should never trigger if upstream pipelines are healthy,
    #    but a silent NaN propagates into every downstream cell.
    if A.isna().any().any() or S.isna().any().any():
        raise ValueError("NaN values present after alignment; inspect upstream pipelines.")

    # 3. Row-stochastic check (within tolerance — small floating-point drift OK).
    if not np.allclose(A.sum(axis=1), 1.0, atol=atol):
        raise ValueError("A is not row-stochastic; topic distribution rows must sum to 1.")
    if not np.allclose(S.sum(axis=1), 1.0, atol=atol):
        raise ValueError("S is not row-stochastic; emotion distribution rows must sum to 1.")

    return (
        A.to_numpy(),
        S.to_numpy(),
        common_authors,         # author labels, length N
        A.columns,              # topic labels,  length T
        S.columns,              # emotion labels, length M
    )
```

## Compute the joint topic-emotion matrix

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.joint
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.validate_inputs]
```

Compute `E = A.T @ S`. Also produce the volume-weighted variant when a `tweet_counts` vector is available — keep them as separate outputs rather than swapping one for the other, because the comparison between them *is* the diagnostic for whether tweet volume matters.

### Sample code

```python
def compute_joint(
    A: np.ndarray,                       # (N, T)
    S: np.ndarray,                       # (N, M)
    tweet_counts: np.ndarray | None = None,  # (N,) — optional volume weighting
) -> dict[str, np.ndarray]:
    """Return the unweighted joint and (if weights supplied) the weighted joint.

    The unweighted joint treats each author as one unit regardless of activity.
    The weighted joint amplifies high-volume authors proportionally.
    """
    result = {"joint": A.T @ S}          # shape (T, M)

    if tweet_counts is not None:
        # Normalize weights to a probability distribution over authors so that
        # the weighted joint stays on the same scale as the unweighted one.
        w = tweet_counts / tweet_counts.sum()
        # Broadcasting: A * w[:, None] scales each author-row of A by w_n.
        result["joint_weighted"] = (A * w[:, None]).T @ S

    return result
```

## Derive conditional distributions

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.conditionals
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.joint]
```

`E` is a co-occurrence matrix in raw form; it is most readable as a conditional distribution. Compute both directions because each answers a different question.

### Sample code

```python
def conditionals(joint: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Row- and column-normalize the joint into the two conditional views.

    `eps` guards against zero rows/columns — a topic with no author mass at
    all (should not happen if validation passed) would otherwise produce NaN.
    """
    # P(emotion | topic): each ROW is a probability distribution over emotions.
    row_sums = joint.sum(axis=1, keepdims=True)
    p_emotion_given_topic = joint / np.maximum(row_sums, eps)

    # P(topic | emotion): each COLUMN is a probability distribution over topics.
    col_sums = joint.sum(axis=0, keepdims=True)
    p_topic_given_emotion = joint / np.maximum(col_sums, eps)

    return {
        "p_emotion_given_topic": p_emotion_given_topic,   # rows sum to 1
        "p_topic_given_emotion": p_topic_given_emotion,   # cols sum to 1
    }
```

## Diagnostics

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.diagnostics
estimate: 2h
blocked_by: [twitter.topic_emotion_mixing.joint]
```

Without sanity checks the joint matrix is a number-blob. These diagnostics surface the failure modes that make the result misleading rather than wrong.

### Checks

- **Per-topic entropy of `P(emotion | topic)`** — a topic whose emotion distribution matches the global emotion distribution is uninformative. Compare each row's entropy to the entropy of the marginal `P(emotion)`. Rows close to the marginal are saying "this topic has no emotional signature distinct from the average author".
- **Topic mass concentration** — `Σ_n A[n, t]` per topic. If one topic captures most of the authorial mass, every conditional row for the smaller topics is estimated from very few effective authors and should be flagged as low-confidence.
- **Author-count effective sample size per topic** — count authors with `A[n, t] > threshold` (e.g. 0.1). Topics where this drops below ~30 are estimated on thin evidence regardless of total mass.
- **Sensitivity to volume weighting** — KL divergence between `p_emotion_given_topic` and its weighted counterpart. A large divergence means the result depends strongly on whether you treat each author or each tweet as the unit.

### Sample code

```python
from scipy.stats import entropy

def diagnostics(
    A: np.ndarray,
    p_emotion_given_topic: np.ndarray,    # (T, M)
    topic_labels,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Per-topic diagnostics. Returns one row per topic for easy inspection."""
    # Marginal P(emotion) from the topic-conditional matrix's row sums, then
    # we'll compare each row's entropy to this baseline.
    marginal_emotion = p_emotion_given_topic.mean(axis=0)  # rough P(emotion)
    baseline_entropy = entropy(marginal_emotion)

    return pd.DataFrame({
        "topic": topic_labels,
        # Effective number of authors writing meaningfully about this topic.
        "n_authors_above_threshold": (A > threshold).sum(axis=0),
        # Total topic mass across authors.
        "total_topic_mass": A.sum(axis=0),
        # Lower entropy = more emotionally distinctive topic.
        "emotion_entropy": [entropy(row) for row in p_emotion_given_topic],
        # Positive value = this topic is more peaked than the average.
        "entropy_delta_vs_marginal": baseline_entropy - np.array(
            [entropy(row) for row in p_emotion_given_topic]
        ),
    }).sort_values("emotion_entropy")
```

## Visualize

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.viz
estimate: 2h
blocked_by: [twitter.topic_emotion_mixing.conditionals]
```

Produce a heatmap of `P(emotion | topic)` — rows are topics, columns are emotions, cell color is the conditional probability. Order rows by hierarchical clustering on emotion profiles so topics with similar emotional signatures sit next to each other; this is the single visualization that makes the result legible at a glance. Save to `figures/topic_emotion_heatmap.png` and a second copy as PDF for inclusion in any write-up.

A secondary plot: per-topic bar chart of the entropy diagnostic — fastest way to spot uninformative topics.

## Persist outputs

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.persist
estimate: 1h
blocked_by: [twitter.topic_emotion_mixing.conditionals, twitter.topic_emotion_mixing.diagnostics]
```

Save four artifacts with topic and emotion labels attached (parquet preserves them; npz does not):

- `joint.parquet` — unweighted `E`, shape `(T, M)`
- `joint_weighted.parquet` — volume-weighted `E`, if applicable
- `p_emotion_given_topic.parquet` — row-conditional view
- `p_topic_given_emotion.parquet` — column-conditional view
- `diagnostics.parquet` — per-topic sanity table from the diagnostics task

File naming should encode the weighting choice so the unweighted and weighted runs can coexist: e.g. `joint__unweighted.parquet` vs `joint__volume_weighted.parquet`.

## Knowledge capture decision

```yaml
status: todo
type: task
id: twitter.topic_emotion_mixing.kb_capture
estimate: 0.5h
```

Per MDDIA convention §11 (plans plan for knowledge capture):

- **Existing how-to relevant?** Check the KB for an existing fusion or co-occurrence-matrix how-to. If one exists, add a task here to update it with whatever the diagnostics step reveals about failure modes.
- **New how-to warranted?** If `A^T @ S` fusion of two row-stochastic author matrices is a pattern this project will repeat (e.g. topics × demographics, topics × behavioral clusters), scaffold `TOPIC_FUSION_SKILL.md` as `volatility: initial_draft` and add a task to populate it once this plan completes.
- **Neither?** Mark this task `done` with a short rationale in the worklog and move on.

Decision deadline: before the joint-computation task closes — by then the generality of the pattern is clear.
