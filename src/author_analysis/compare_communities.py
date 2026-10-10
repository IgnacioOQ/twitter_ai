#!/usr/bin/env python3
"""
Network Community Analysis
=========================
Analyzes topic, sentiment, and emotion profiles of the largest
Selected network communities among matched authors.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from pathlib import Path
from scipy import stats
from .common import configuration, validate_ids, topic_columns, topic_labels, EMOTIONS

CONFIG = configuration()
COMMUNITY = CONFIG['community_algorithm']
OUTPUT_DIR = Path(CONFIG['network']) / 'communities'
TOP_N = 21
SENSITIVITY_N = 15
N_TOPICS = 0
TOPIC_LABELS = {}


def load_data():
    global N_TOPICS, TOPIC_LABELS
    df = validate_ids(pd.read_parquet(CONFIG['matched']))
    columns = topic_columns(df)
    N_TOPICS = len(columns)
    TOPIC_LABELS = topic_labels(CONFIG['topic_labels'], N_TOPICS)
    sizes = df.groupby(COMMUNITY).size()
    if len(sizes) < 2 or len(df) <= len(sizes):
        raise ValueError('Community comparisons require at least two groups and residual degrees of freedom')
    return df, columns


def get_top_communities(df, n):
    """Get the n largest communities by matched-author count."""
    comm_sizes = df.groupby(COMMUNITY).size().sort_values(ascending=False)
    top_comms = comm_sizes.head(n).index.tolist()
    return top_comms, comm_sizes


def calculate_omega_squared(df, group_col, value_col):
    """Calculate omega-squared (bias-corrected eta-squared)."""
    groups = df.groupby(group_col)[value_col]

    grand_mean = df[value_col].mean()
    n_total = len(df)
    k = df[group_col].nunique()

    # SS_between
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)

    # SS_within
    ss_within = sum(((g - g.mean())**2).sum() for _, g in groups)

    # SS_total
    ss_total = ((df[value_col] - grand_mean)**2).sum()

    # MS_within
    ms_within = ss_within / (n_total - k) if n_total > k else 0

    # Omega-squared
    omega_sq = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    omega_sq = max(0, omega_sq)  # Floor at 0

    return omega_sq


def calculate_eta_squared(df, group_col, value_col):
    """Calculate eta-squared."""
    groups = df.groupby(group_col)[value_col]
    grand_mean = df[value_col].mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    ss_total = ((df[value_col] - grand_mean)**2).sum()

    return ss_between / ss_total if ss_total > 0 else 0


def calculate_cramers_v_corrected(df, group_col, category_col):
    """Calculate bias-corrected Cramer's V."""
    contingency = pd.crosstab(df[group_col], df[category_col])
    chi2 = stats.chi2_contingency(contingency)[0]
    n = len(df)
    r, k = contingency.shape

    # Bias correction
    phi2 = chi2 / n
    phi2_corrected = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    r_corrected = r - ((r-1)**2)/(n-1)
    k_corrected = k - ((k-1)**2)/(n-1)

    denom = min(k_corrected - 1, r_corrected - 1)
    if denom <= 0:
        return 0

    return np.sqrt(phi2_corrected / denom)


def create_community_summary(df, top_comms, topic_cols, output_path):
    """Create summary table for top communities."""
    print("\nCreating community summary table...")

    df_top = df[df[COMMUNITY].isin(top_comms)].copy()
    n_total = len(df)

    # Overall means
    overall_topic_means = df[topic_cols].mean()
    overall_sent = df[['positive', 'neutral', 'negative']].mean()
    overall_emo = df[EMOTIONS].mean()
    overall_emo_std = df[EMOTIONS].std()

    rows = []
    cumulative = 0

    for comm in top_comms:
        comm_df = df[df[COMMUNITY] == comm]
        n = len(comm_df)
        pct = 100 * n / n_total
        cumulative += pct

        # Topic composition
        topic_means = comm_df[topic_cols].mean()
        top_topics_idx = topic_means.values.argsort()[::-1][:3]
        top_topics = ", ".join([f"{TOPIC_LABELS[i]} ({100*topic_means.iloc[i]:.1f}%)"
                                for i in top_topics_idx])

        # Sentiment
        pos = comm_df['positive'].mean()
        neu = comm_df['neutral'].mean()
        neg = comm_df['negative'].mean()

        # Most distinctive emotions (by standardized difference)
        emo_means = comm_df[EMOTIONS].mean()
        emo_std_diff = (emo_means - overall_emo) / overall_emo_std
        top_emo_idx = np.abs(emo_std_diff.values).argsort()[::-1][:3]
        distinctive_emo = ", ".join([f"{EMOTIONS[i]} ({emo_std_diff.iloc[i]:+.2f}σ)"
                                     for i in top_emo_idx])

        rows.append({
            'community_id': comm,
            'n_authors': n,
            'pct_authors': round(pct, 2),
            'cumulative_pct': round(cumulative, 2),
            'top_topics': top_topics,
            'positive': round(pos, 3),
            'neutral': round(neu, 3),
            'negative': round(neg, 3),
            'distinctive_emotions': distinctive_emo
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path / "community_summary.csv", index=False)

    # Long tail info
    n_excluded = n_total - len(df_top)
    pct_excluded = 100 * n_excluded / n_total

    print(f"   Top {len(top_comms)} communities: {len(df_top):,} authors ({100-pct_excluded:.1f}%)")
    print(f"   Long tail excluded: {n_excluded:,} authors ({pct_excluded:.1f}%)")

    return summary_df, n_excluded, pct_excluded


def create_topic_heatmaps(df, top_comms, topic_cols, output_path):
    """Create topic composition, capture, and enrichment heatmaps."""
    print("\nCreating topic heatmaps...")

    df_top = df[df[COMMUNITY].isin(top_comms)].copy()

    # Community sizes for labels
    comm_sizes = df_top.groupby(COMMUNITY).size()

    # Overall means (among top communities)
    overall_means = df_top[topic_cols].mean()

    # --- Topic Composition (mean topic weights per community) ---
    composition = df_top.groupby(COMMUNITY)[topic_cols].mean()
    composition = composition.loc[top_comms]  # Order by size

    # Row labels with ID and n
    row_labels = [f"C{c} (n={comm_sizes[c]:,})" for c in top_comms]
    col_labels = [TOPIC_LABELS[i] for i in range(N_TOPICS)]

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(composition.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(N_TOPICS))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Community')
    ax.set_title(f'Topic Composition (Mean Weights) - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='Mean Topic Weight')
    plt.tight_layout()
    plt.savefig(output_path / "topic_composition.png", dpi=150)
    plt.savefig(output_path / "topic_composition.pdf")
    plt.close()

    # Save with nice labels
    composition_labeled = composition.copy()
    composition_labeled.index = row_labels
    composition_labeled.columns = col_labels
    composition_labeled.to_csv(output_path / "topic_composition.csv")

    # --- Topic Capture (share of topic's total weight in each community) ---
    topic_totals = df_top[topic_cols].sum()
    capture = df_top.groupby(COMMUNITY)[topic_cols].sum()
    capture = capture.loc[top_comms]
    capture = capture.div(topic_totals, axis=1) * 100  # Percentage

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(capture.values, aspect='auto', cmap='Blues')
    ax.set_xticks(range(N_TOPICS))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Community')
    ax.set_title(f'Topic Capture (% of Topic Weight) - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='% of Topic Captured')
    plt.tight_layout()
    plt.savefig(output_path / "topic_capture.png", dpi=150)
    plt.savefig(output_path / "topic_capture.pdf")
    plt.close()

    capture_labeled = capture.copy()
    capture_labeled.index = row_labels
    capture_labeled.columns = col_labels
    capture_labeled.to_csv(output_path / "topic_capture.csv")

    # --- Log2 Enrichment ---
    enrichment = pd.DataFrame(index=top_comms, columns=topic_cols, dtype=float)
    for col in topic_cols:
        enrichment[col] = np.log2(composition[col].values / overall_means[col])

    # Clip extreme values for visualization
    vmax = 3
    enrichment_clipped = enrichment.astype(float).clip(-vmax, vmax)

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(enrichment_clipped.values, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(N_TOPICS))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Community')
    ax.set_title(f'Log₂ Topic Enrichment (vs Overall) - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='Log₂ Enrichment')
    plt.tight_layout()
    plt.savefig(output_path / "topic_enrichment.png", dpi=150)
    plt.savefig(output_path / "topic_enrichment.pdf")
    plt.close()

    enrichment.index = row_labels
    enrichment.columns = col_labels
    enrichment.to_csv(output_path / "topic_enrichment.csv")

    print("   Saved: topic_composition, topic_capture, topic_enrichment")


def create_sentiment_heatmaps(df, top_comms, output_path):
    """Create sentiment heatmaps (raw means and standardized differences)."""
    print("\nCreating sentiment heatmaps...")

    df_top = df[df[COMMUNITY].isin(top_comms)].copy()
    sent_cols = ['positive', 'neutral', 'negative']

    # Community sizes for labels
    comm_sizes = df_top.groupby(COMMUNITY).size()
    row_labels = [f"C{c} (n={comm_sizes[c]:,})" for c in top_comms]

    # Overall means and stds
    overall_means = df_top[sent_cols].mean()
    overall_stds = df_top[sent_cols].std()

    # --- Raw Means ---
    sent_means = df_top.groupby(COMMUNITY)[sent_cols].mean()
    sent_means = sent_means.loc[top_comms]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(sent_means.values, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Community')
    ax.set_title(f'Mean Sentiment - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='Mean Score')
    plt.tight_layout()
    plt.savefig(output_path / "sentiment_means.png", dpi=150)
    plt.savefig(output_path / "sentiment_means.pdf")
    plt.close()

    sent_means.index = row_labels
    sent_means.to_csv(output_path / "sentiment_means.csv")

    # --- Standardized Differences ---
    sent_std = (sent_means.values - overall_means.values) / overall_stds.values
    sent_std_df = pd.DataFrame(sent_std, index=row_labels, columns=sent_cols)

    vmax = np.abs(sent_std).max()
    vmax = min(vmax, 3)  # Cap for readability

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sent_std, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Community')
    ax.set_title(f'Standardized Sentiment Differences - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='Std. Difference (σ)')
    plt.tight_layout()
    plt.savefig(output_path / "sentiment_standardized.png", dpi=150)
    plt.savefig(output_path / "sentiment_standardized.pdf")
    plt.close()

    sent_std_df.to_csv(output_path / "sentiment_standardized.csv")

    print("   Saved: sentiment_means, sentiment_standardized")


def create_emotion_heatmap(df, top_comms, output_path):
    """Create emotion heatmap showing standardized differences."""
    print("\nCreating emotion heatmap...")

    df_top = df[df[COMMUNITY].isin(top_comms)].copy()

    # Community sizes for labels
    comm_sizes = df_top.groupby(COMMUNITY).size()
    row_labels = [f"C{c} (n={comm_sizes[c]:,})" for c in top_comms]

    # Overall means and stds
    overall_means = df_top[EMOTIONS].mean()
    overall_stds = df_top[EMOTIONS].std()

    # Community means
    emo_means = df_top.groupby(COMMUNITY)[EMOTIONS].mean()
    emo_means = emo_means.loc[top_comms]

    # Standardized differences
    emo_std = (emo_means.values - overall_means.values) / overall_stds.values
    emo_std_df = pd.DataFrame(emo_std, index=row_labels, columns=EMOTIONS)

    vmax = np.abs(emo_std).max()
    vmax = min(vmax, 2)  # Cap for readability

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(emo_std, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=45, ha='right')
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Community')
    ax.set_title(f'Standardized Emotion Differences - Top {len(top_comms)} Leiden Communities')
    plt.colorbar(im, ax=ax, label='Std. Difference (σ)')
    plt.tight_layout()
    plt.savefig(output_path / "emotion_standardized.png", dpi=150)
    plt.savefig(output_path / "emotion_standardized.pdf")
    plt.close()

    emo_std_df.to_csv(output_path / "emotion_standardized.csv")

    # Also save raw means
    emo_means.index = row_labels
    emo_means.to_csv(output_path / "emotion_means.csv")

    print("   Saved: emotion_standardized, emotion_means")


def calculate_association_summary(df, top_comms, topic_cols, n_label):
    """Calculate association measures for top communities."""
    df_subset = df[df[COMMUNITY].isin(top_comms)].copy()
    n_authors = len(df_subset)
    coverage = 100 * n_authors / len(df)

    results = {
        'n_communities': len(top_comms),
        'n_authors': n_authors,
        'coverage_pct': round(coverage, 1)
    }

    # Topics - omega-squared for each topic
    topic_omega = {}
    topic_eta = {}
    for col in topic_cols:
        omega = calculate_omega_squared(df_subset, COMMUNITY, col)
        eta = calculate_eta_squared(df_subset, COMMUNITY, col)
        topic_omega[col] = omega
        topic_eta[col] = eta

    results['topic_mean_omega_sq'] = np.mean(list(topic_omega.values()))
    results['topic_mean_eta_sq'] = np.mean(list(topic_eta.values()))
    results['topic_omega_by_topic'] = topic_omega

    # Sentiment
    sent_omega = {}
    for col in ['positive', 'neutral', 'negative']:
        sent_omega[col] = calculate_omega_squared(df_subset, COMMUNITY, col)
    results['sentiment_mean_omega_sq'] = np.mean(list(sent_omega.values()))
    results['sentiment_omega'] = sent_omega

    # Emotions
    emo_omega = {}
    for col in EMOTIONS:
        emo_omega[col] = calculate_omega_squared(df_subset, COMMUNITY, col)
    results['emotion_mean_omega_sq'] = np.mean(list(emo_omega.values()))
    results['emotion_omega'] = emo_omega

    # Cramer's V for dominant topic
    results['cramers_v_dominant_topic'] = calculate_cramers_v_corrected(
        df_subset, COMMUNITY, 'dominant_topic')

    return results


def save_association_summary(results_21, results_15, output_path):
    """Save association summary to files."""
    print("\nSaving association summary...")

    # Main summary
    summary = {
        'primary_analysis': {
            'n_communities': results_21['n_communities'],
            'n_authors': results_21['n_authors'],
            'coverage_pct': results_21['coverage_pct'],
            'topic_mean_omega_sq': round(results_21['topic_mean_omega_sq'], 6),
            'sentiment_mean_omega_sq': round(results_21['sentiment_mean_omega_sq'], 6),
            'emotion_mean_omega_sq': round(results_21['emotion_mean_omega_sq'], 6),
            'cramers_v_dominant_topic': round(results_21['cramers_v_dominant_topic'], 4)
        },
        'sensitivity_check': {
            'n_communities': results_15['n_communities'],
            'n_authors': results_15['n_authors'],
            'coverage_pct': results_15['coverage_pct'],
            'topic_mean_omega_sq': round(results_15['topic_mean_omega_sq'], 6),
            'sentiment_mean_omega_sq': round(results_15['sentiment_mean_omega_sq'], 6),
            'emotion_mean_omega_sq': round(results_15['emotion_mean_omega_sq'], 6),
            'cramers_v_dominant_topic': round(results_15['cramers_v_dominant_topic'], 4)
        }
    }

    with open(output_path / "association_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Detailed topic omega-squared
    topic_df = pd.DataFrame({
        'topic': [TOPIC_LABELS[i] for i in range(N_TOPICS)],
        'omega_sq_21': [results_21['topic_omega_by_topic'][f'topic_{i}'] for i in range(N_TOPICS)],
        'omega_sq_15': [results_15['topic_omega_by_topic'][f'topic_{i}'] for i in range(N_TOPICS)]
    })
    topic_df = topic_df.sort_values('omega_sq_21', ascending=False)
    topic_df.to_csv(output_path / "topic_omega_squared.csv", index=False)

    # Emotion omega-squared
    emo_df = pd.DataFrame({
        'emotion': EMOTIONS,
        'omega_sq_21': [results_21['emotion_omega'][e] for e in EMOTIONS],
        'omega_sq_15': [results_15['emotion_omega'][e] for e in EMOTIONS]
    })
    emo_df = emo_df.sort_values('omega_sq_21', ascending=False)
    emo_df.to_csv(output_path / "emotion_omega_squared.csv", index=False)

    print("   Saved: association_summary.json, topic_omega_squared.csv, emotion_omega_squared.csv")

    return summary




def main():
    print("="*70)
    print("NETWORK COMMUNITY ANALYSIS")
    print("="*70)

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)

    figures_path = OUTPUT_DIR / "figures"
    tables_path = OUTPUT_DIR / "tables"

    # Load data
    df, topic_cols = load_data()

    # Get top communities
    top_21, comm_sizes = get_top_communities(df, TOP_N)
    top_15, _ = get_top_communities(df, SENSITIVITY_N)

    print(f"\nTop {TOP_N} communities selected for primary analysis")
    print(f"Top {SENSITIVITY_N} communities for sensitivity check")

    # Community summary
    summary_df, n_excluded, pct_excluded = create_community_summary(
        df, top_21, topic_cols, tables_path)

    # Topic heatmaps
    create_topic_heatmaps(df, top_21, topic_cols, figures_path)

    # Sentiment heatmaps
    create_sentiment_heatmaps(df, top_21, figures_path)

    # Emotion heatmap
    create_emotion_heatmap(df, top_21, figures_path)

    # Association measures
    print("\nCalculating association measures...")
    results_21 = calculate_association_summary(df, top_21, topic_cols, "top_21")
    results_15 = calculate_association_summary(df, top_15, topic_cols, "top_15")

    print(f"\n   Primary (Top 21, {results_21['coverage_pct']}% coverage):")
    print(f"      Topic ω²: {results_21['topic_mean_omega_sq']:.4f}")
    print(f"      Sentiment ω²: {results_21['sentiment_mean_omega_sq']:.4f}")
    print(f"      Emotion ω²: {results_21['emotion_mean_omega_sq']:.4f}")
    print(f"      Cramér's V: {results_21['cramers_v_dominant_topic']:.4f}")

    print(f"\n   Sensitivity (Top 15, {results_15['coverage_pct']}% coverage):")
    print(f"      Topic ω²: {results_15['topic_mean_omega_sq']:.4f}")
    print(f"      Sentiment ω²: {results_15['sentiment_mean_omega_sq']:.4f}")
    print(f"      Emotion ω²: {results_15['emotion_mean_omega_sq']:.4f}")
    print(f"      Cramér's V: {results_15['cramers_v_dominant_topic']:.4f}")

    # Save association summary
    assoc_summary = save_association_summary(results_21, results_15, tables_path)



    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nOutputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
