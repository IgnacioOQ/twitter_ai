#!/usr/bin/env python3
"""Build the non-temporal blog figures from existing fixed analysis outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "blog_figures"
DATA_OUT = OUT / "data"
FIG_OUT = OUT
WORK = REPO_ROOT / "data_sets" / "blog_analysis"
PALETTE_PATH = Path(__file__).resolve().parent / "styles" / "palette.json"
STYLE_PATH = Path(__file__).resolve().parent / "styles" / "blog_figures.mplstyle"

MATCHED_JSON = WORK / "matched_authors.json"
LABELS_CSV = REPO_ROOT / "outputs" / "community_analysis" / "community_labels.csv"
RAW_FIG_DIR = WORK / "matrices"
HPC_TABLES = WORK / "tables"
for directory in [DATA_OUT, FIG_OUT / "dark" / "png", FIG_OUT / "dark" / "svg", FIG_OUT / "light" / "png", FIG_OUT / "light" / "svg"]:
    directory.mkdir(parents=True, exist_ok=True)

PALETTE = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
TOPICS = PALETTE["topic_order"]
TOPIC_COLORS = PALETTE["topic_colors"]
SENTIMENT_COLORS = PALETTE["sentiment_colors"]
EMOTIONS = PALETTE["emotion_order"]
EMOTION_COLORS = PALETTE["emotion_colors"]
COMMUNITY_RGB = PALETTE["community_colors_rgb"]


def rgb_to_hex(rgb: list[float]) -> str:
    return "#" + "".join(f"{round(value * 255):02x}" for value in rgb)


def community_color(community_id: int) -> str:
    return rgb_to_hex(COMMUNITY_RGB[community_id % len(COMMUNITY_RGB)])


@contextmanager
def figure_theme(theme: str):
    tokens = PALETTE["themes"][theme]
    with plt.style.context(STYLE_PATH):
        with mpl.rc_context(
            {
                "figure.facecolor": tokens["background"],
                "figure.edgecolor": tokens["background"],
                "savefig.facecolor": tokens["background"],
                "axes.facecolor": tokens["background"],
                "axes.edgecolor": tokens["muted"],
                "axes.labelcolor": tokens["foreground"],
                "axes.titlecolor": tokens["foreground"],
                "text.color": tokens["foreground"],
                "xtick.color": tokens["foreground"],
                "ytick.color": tokens["foreground"],
                "grid.color": tokens["grid"],
                "legend.labelcolor": tokens["foreground"],
            }
        ):
            yield tokens


def finish_axes(ax: mpl.axes.Axes, tokens: dict, grid_axis: str | None = None) -> None:
    for spine in ax.spines.values():
        spine.set_color(tokens["muted"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=tokens["grid"], alpha=0.45)
        ax.set_axisbelow(True)


def save(fig: mpl.figure.Figure, stem: str, theme: str) -> None:
    png = FIG_OUT / theme / "png" / f"{stem}_{theme}.png"
    svg = FIG_OUT / theme / "svg" / f"{stem}_{theme}.svg"
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(svg, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def label_table() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    labels["community_id"] = labels["community_id"].astype(int)
    labels = labels.sort_values("n_authors", ascending=False).reset_index(drop=True)
    return labels


def load_raw_matrix(filename: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_FIG_DIR / filename, index_col=0)
    ids = df.index.to_series().str.extract(r"C(\d+)", expand=False).astype(int)
    df.insert(0, "community_id", ids.to_numpy())
    return df.reset_index(drop=True)


def timeline_data() -> pd.DataFrame:
    rows = [
        {"date": "2022-11-24", "event": "Stable Diffusion 2.0 released", "category": "Image generation", "source_title": "Stable Diffusion 2.0 Release", "source_url": "https://stability.ai/news-updates/stable-diffusion-v2-release"},
        {"date": "2022-11-30", "event": "ChatGPT research preview launched", "category": "Conversational AI", "source_title": "Introducing ChatGPT", "source_url": "https://openai.com/index/chatgpt/"},
        {"date": "2022-12-07", "event": "Stable Diffusion 2.1 released", "category": "Image generation", "source_title": "Stable Diffusion v2.1 and DreamStudio Updates", "source_url": "https://stability.ai/news-updates/stablediffusion2-1-release7-dec-2022"},
        {"date": "2023-01-23", "event": "Microsoft extends OpenAI partnership", "category": "Industry", "source_title": "Microsoft and OpenAI extend partnership", "source_url": "https://blogs.microsoft.com/blog/2023/01/23/microsoftandopenaiextendpartnership/"},
        {"date": "2023-02-01", "event": "ChatGPT Plus announced", "category": "Conversational AI", "source_title": "Introducing ChatGPT Plus", "source_url": "https://openai.com/index/chatgpt-plus/"},
        {"date": "2023-02-06", "event": "Google introduces Bard to trusted testers", "category": "Conversational AI", "source_title": "An important next step on our AI journey", "source_url": "https://blog.google/innovation-and-ai/technology/ai/bard-google-ai-search-updates/"},
        {"date": "2023-02-07", "event": "AI-powered Bing and Edge preview launched", "category": "Search and platform", "source_title": "Reinventing search with a new AI-powered Microsoft Bing and Edge", "source_url": "https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/"},
        {"date": "2023-02-24", "event": "Meta introduces LLaMA for research", "category": "Foundation model", "source_title": "Introducing LLaMA", "source_url": "https://ai.meta.com/blog/large-language-model-llama-meta-ai/"},
    ]
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data["dataset_start_utc"] = "2022-10-30T00:00:00Z"
    data["dataset_end_utc"] = "2023-02-27T12:00:00Z"
    data["date_verified"] = True
    data.to_csv(DATA_OUT / "01_ai_development_timeline.csv", index=False)
    return data


def plot_timeline() -> None:
    data = timeline_data()
    colors = {
        "Image generation": "#8f63b8",
        "Conversational AI": "#d9822b",
        "Industry": "#3f8f66",
        "Search and platform": "#3579a8",
        "Foundation model": "#b24f61",
    }
    start = pd.Timestamp("2022-10-30T00:00:00Z")
    end = pd.Timestamp("2023-02-27T12:00:00Z")
    y_offsets = [0.95, -1.00, 1.50, -1.48, 0.85, -0.95, 1.55, -1.48]
    x_offsets = [0, 0, 0, 0, -3, 0, 5, 0]

    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(16, 7.5))
            ax.hlines(0, start, end, color=tokens["muted"], linewidth=1.5)
            ax.scatter([start, end], [0, 0], s=45, color=tokens["foreground"], zorder=4)
            for row, y_offset, x_offset in zip(data.itertuples(), y_offsets, x_offsets):
                color = colors[row.category]
                text_x = row.date + pd.Timedelta(days=x_offset)
                ax.plot([row.date, text_x], [0, y_offset * 0.72], color=color, linewidth=1.1)
                ax.scatter(row.date, 0, s=72, color=color,
                           edgecolor=tokens["background"], linewidth=1.0, zorder=5)
                ax.text(
                    text_x, y_offset, f"{row.date.strftime('%d %b')}\n{row.event}",
                    ha="center", va="bottom" if y_offset > 0 else "top",
                    fontsize=9.3, linespacing=1.28, color=tokens["foreground"],
                )
            ax.text(start, -0.15, "Collection begins\n30 Oct 2022", ha="left", va="top",
                    color=tokens["muted"], fontsize=8.8)
            ax.text(end, 0.15, "Collection ends\n27 Feb 2023, 12:00 UTC", ha="right", va="bottom",
                    color=tokens["muted"], fontsize=8.8)
            ax.set_xlim(start - pd.Timedelta(days=4), end + pd.Timedelta(days=4))
            ax.set_ylim(-2.25, 2.25)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            ax.tick_params(axis="x", length=0, pad=7)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            handles = [Patch(facecolor=color, label=category) for category, color in colors.items()]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                       ncol=5, columnspacing=1.5)
            fig.subplots_adjust(left=0.045, right=0.985, top=0.87, bottom=0.15)
            save(fig, "01_ai_development_timeline", theme)

def load_matched() -> pd.DataFrame:
    print(f"Loading fixed matched-author data: {MATCHED_JSON}", flush=True)
    data = json.loads(MATCHED_JSON.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_records(data)
    if len(frame) != 198_326:
        raise RuntimeError(f"Expected 198326 authors, found {len(frame)}")
    return frame


def umap_projection() -> pd.DataFrame:
    """Load the existing fixed-seed projection; never fit or update UMAP here."""
    projection = WORK / "umap_fixed_topic_scores_seed0.csv"
    if not projection.exists():
        raise FileNotFoundError(
            f"Missing saved UMAP projection: {projection}. "
            "This renderer intentionally does not fit UMAP."
        )
    data = pd.read_csv(projection, dtype={"author_id": str})
    required = {"umap_1", "umap_2", "topic_id"}
    if not required.issubset(data.columns):
        raise ValueError(f"Saved projection lacks columns: {sorted(required - set(data.columns))}")
    if len(data) != 198_326:
        raise ValueError(f"Expected 198326 projected authors, found {len(data)}")
    data["topic_id"] = data["topic_id"].astype(int)
    data["topic_label"] = data["topic_id"].map(dict(enumerate(TOPICS)))
    return data


def plot_umap() -> None:
    data = umap_projection()
    data["rendered_topic_overlay"] = False
    for topic_id in range(12):
        index = data.index[data["topic_id"] == topic_id]
        if len(index) > 5_000:
            index = data.loc[index].sample(n=5_000, random_state=topic_id + 1).index
        data.loc[index, "rendered_topic_overlay"] = True
    summary = (
        data.groupby(["topic_id", "topic_label"], as_index=False)
        .agg(
            n_authors=("topic_id", "size"),
            rendered_overlay_authors=("rendered_topic_overlay", "sum"),
            umap_1_min=("umap_1", "min"),
            umap_1_max=("umap_1", "max"),
            umap_2_min=("umap_2", "min"),
            umap_2_max=("umap_2", "max"),
        )
        .sort_values("topic_id")
    )
    summary["projection_seed"] = 0
    summary["projection_status"] = "existing saved projection; no refit"
    summary.to_csv(DATA_OUT / "04_umap_12_topics.csv", index=False)

    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(14, 8.5))
            ax.scatter(
                data["umap_1"], data["umap_2"], s=0.22, alpha=0.08,
                linewidths=0, antialiased=False, color=tokens["muted"],
                rasterized=True, zorder=1,
            )
            for topic_id, topic_label in enumerate(TOPICS):
                subset = data[(data["topic_id"] == topic_id) & data["rendered_topic_overlay"]]
                ax.scatter(
                    subset["umap_1"], subset["umap_2"], s=2.2, alpha=0.68,
                    linewidths=0, antialiased=False,
                    color=TOPIC_COLORS[topic_label], rasterized=True, zorder=2,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            handles = [
                Line2D([], [], linestyle="", marker="o", markersize=5.5,
                       markerfacecolor=TOPIC_COLORS[label],
                       markeredgecolor=TOPIC_COLORS[label], alpha=1)
                for label in TOPICS
            ]
            legend = fig.legend(
                handles, TOPICS, loc="center left", bbox_to_anchor=(0.805, 0.52),
                ncol=1, title="Dominant topic",
            )
            plt.setp(legend.get_title(), weight="semibold")
            fig.subplots_adjust(left=0.055, right=0.79, top=0.985, bottom=0.075)
            save(fig, "04_umap_12_topics", theme)

def topic_sentiment_data() -> pd.DataFrame:
    data = pd.read_csv(HPC_TABLES / "topic_sentiment_fuzzy_network_subset.csv")
    data["topic_id"] = data["topic_id"].astype(int)
    data["topic_label"] = data["topic_id"].map(dict(enumerate(TOPICS)))
    data["net_sentiment"] = data["positive"] - data["negative"]
    data = data.sort_values("topic_id")
    columns = [
        "topic_id", "topic_label", "membership_weight", "positive", "neutral",
        "negative", "net_sentiment", "sum_check",
    ]
    data[columns].to_csv(DATA_OUT / "06_topic_sentiment_and_net.csv", index=False)
    return data[columns]


def plot_topic_sentiment() -> None:
    data = topic_sentiment_data()
    order = data.sort_values("net_sentiment")["topic_label"].tolist()
    draw = data.set_index("topic_label").loc[order].reset_index()
    y = np.arange(len(draw))
    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, (ax_share, ax_net) = plt.subplots(
                1, 2, figsize=(15.5, 9),
                gridspec_kw={"width_ratios": [1.7, 1]},
            )
            left = np.zeros(len(draw))
            bar_handles = []
            for sentiment in ["positive", "neutral", "negative"]:
                values = draw[sentiment].to_numpy()
                bars = ax_share.barh(
                    y, values, left=left,
                    color=SENTIMENT_COLORS[sentiment],
                    height=0.64,
                    label=sentiment.title(),
                )
                bar_handles.append(bars[0])
                left += values
            ax_share.set_yticks(y, draw["topic_label"])
            ax_share.set_xlim(0, 1)
            ax_share.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax_share.set_xlabel("Mean sentiment probability")
            finish_axes(ax_share, tokens, "x")

            net = draw["net_sentiment"].to_numpy()
            colors = [
                SENTIMENT_COLORS["positive"] if value >= 0
                else SENTIMENT_COLORS["negative"]
                for value in net
            ]
            ax_net.axvline(
                0, color=SENTIMENT_COLORS["neutral"],
                linewidth=1.2, zorder=0,
            )
            ax_net.hlines(y, 0, net, color=colors, linewidth=2.4, alpha=0.82)
            ax_net.scatter(net, y, color=colors, s=42, zorder=3)
            ax_net.set_yticks([])
            limit = max(abs(net.min()), abs(net.max())) * 1.16
            ax_net.set_xlim(-limit, limit)
            ax_net.set_xlabel("Net sentiment (positive - negative)")
            finish_axes(ax_net, tokens, "x")
            for value, y_pos in zip(net, y):
                ax_net.text(
                    value + (0.012 if value >= 0 else -0.012),
                    y_pos, f"{value:+.2f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=8,
                )

            fig.legend(
                bar_handles,
                ["Positive", "Neutral", "Negative"],
                loc="upper center",
                bbox_to_anchor=(0.42, 0.985),
                ncol=3,
            )
            fig.subplots_adjust(
                left=0.23, right=0.975, top=0.90,
                bottom=0.105, wspace=0.13,
            )
            save(fig, "06_topic_sentiment_and_net", theme)

def community_topic_enrichment_data(labels: pd.DataFrame) -> pd.DataFrame:
    matrix = load_raw_matrix("topic_enrichment.csv")
    matrix = labels[["community_id", "display_label", "n_authors"]].merge(
        matrix, on="community_id", how="left", validate="one_to_one"
    )
    long = matrix.melt(
        id_vars=["community_id", "display_label", "n_authors"],
        value_vars=TOPICS,
        var_name="topic_label",
        value_name="log2_enrichment",
    )
    long["topic_id"] = long["topic_label"].map({name: i for i, name in enumerate(TOPICS)})
    long.to_csv(DATA_OUT / "07_community_topic_enrichment.csv", index=False)
    return matrix


def plot_community_topic_enrichment(labels: pd.DataFrame) -> None:
    data = community_topic_enrichment_data(labels)
    values = data[TOPICS].to_numpy(dtype=float)
    span = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(18, 13))
            fig.patch.set_facecolor(tokens["background"])
            image = ax.imshow(
                values,
                aspect="auto",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span),
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(12), TOPICS, rotation=38, ha="right")
            ax.set_yticks(np.arange(len(data)), data["display_label"])
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(data), 1), minor=True)
            ax.grid(which="minor", color=tokens["background"], linewidth=1.1)
            ax.tick_params(which="minor", bottom=False, left=False)
            colorbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.03)
            colorbar.set_label("log2 enrichment relative to all matched authors")
            colorbar.ax.tick_params(colors=tokens["foreground"])
            colorbar.outline.set_edgecolor(tokens["muted"])
            fig.subplots_adjust(left=0.29, right=0.93, top=0.98, bottom=0.19)
            save(fig, "07_community_topic_enrichment", theme)


def community_sentiment_data(labels: pd.DataFrame) -> pd.DataFrame:
    matrix = load_raw_matrix("sentiment_means.csv")
    matrix = labels[["community_id", "display_label", "n_authors"]].merge(
        matrix, on="community_id", how="left", validate="one_to_one"
    )
    long = matrix.melt(
        id_vars=["community_id", "display_label", "n_authors"],
        value_vars=["positive", "neutral", "negative"],
        var_name="sentiment",
        value_name="mean_score",
    )
    long.to_csv(DATA_OUT / "08_community_sentiment_comparison.csv", index=False)
    return matrix


def plot_community_sentiment(labels: pd.DataFrame) -> None:
    data = community_sentiment_data(labels)
    y = np.arange(len(data))
    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(16, 12))
            fig.patch.set_facecolor(tokens["background"])
            left = np.zeros(len(data))
            for sentiment in ["positive", "neutral", "negative"]:
                values = data[sentiment].to_numpy()
                ax.barh(
                    y,
                    values,
                    left=left,
                    height=0.68,
                    color=SENTIMENT_COLORS[sentiment],
                    label=sentiment.title(),
                )
                left += values
            ax.set_yticks(y, data["display_label"])
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax.set_xlabel("Mean sentiment probability across authors")
            finish_axes(ax, tokens, "x")
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
            fig.subplots_adjust(left=0.31, right=0.97, top=0.91, bottom=0.09)
            save(fig, "08_community_sentiment_comparison", theme)


def topic_weighted_net_data(matched: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = matched.copy()
    matched["net_sentiment"] = matched["positive"] - matched["negative"]
    for record in labels.itertuples():
        community = matched[matched["leiden_directed"] == record.community_id]
        for topic_id, topic_label in enumerate(TOPICS):
            weights = community[f"topic_{topic_id}"].to_numpy(dtype=float)
            net = community["net_sentiment"].to_numpy(dtype=float)
            membership = weights.sum()
            weight_sq = np.square(weights).sum()
            effective_n = (membership * membership / weight_sq) if weight_sq > 0 else 0.0
            weighted_net = np.average(net, weights=weights) if membership > 0 else np.nan
            adequate = bool(membership >= 10.0 and effective_n >= 30.0)
            rows.append(
                {
                    "community_id": record.community_id,
                    "display_label": record.display_label,
                    "n_authors": record.n_authors,
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "membership_weight_author_equivalents": membership,
                    "effective_n_kish": effective_n,
                    "weighted_net_sentiment": weighted_net,
                    "adequate_support": adequate,
                    "plotted_weighted_net_sentiment": weighted_net if adequate else np.nan,
                    "support_rule": "membership_weight >= 10 and Kish effective_n >= 30",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_OUT / "09_topic_weighted_net_sentiment_by_community.csv", index=False)
    return out


def plot_topic_weighted_net(matched: pd.DataFrame, labels: pd.DataFrame) -> None:
    long = topic_weighted_net_data(matched, labels)
    matrix = long.pivot(index="display_label", columns="topic_label", values="plotted_weighted_net_sentiment")
    matrix = matrix.loc[labels["display_label"], TOPICS]
    values = matrix.to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    span = max(abs(valid.min()), abs(valid.max()))
    cmap = LinearSegmentedColormap.from_list(
        "net_sentiment", [SENTIMENT_COLORS["negative"], SENTIMENT_COLORS["net_zero"], SENTIMENT_COLORS["positive"]]
    )
    cmap.set_bad("#77777d")
    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(18, 13))
            fig.patch.set_facecolor(tokens["background"])
            image = ax.imshow(
                np.ma.masked_invalid(values),
                aspect="auto",
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span),
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(12), TOPICS, rotation=38, ha="right")
            ax.set_yticks(np.arange(len(matrix)), matrix.index)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
            ax.grid(which="minor", color=tokens["background"], linewidth=1.1)
            ax.tick_params(which="minor", bottom=False, left=False)
            colorbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.03)
            colorbar.set_label("Topic-weighted net sentiment (positive − negative)")
            colorbar.ax.tick_params(colors=tokens["foreground"])
            colorbar.outline.set_edgecolor(tokens["muted"])
            fig.subplots_adjust(left=0.29, right=0.93, top=0.98, bottom=0.19)
            save(fig, "09_topic_weighted_net_sentiment_by_community", theme)


def emotion_sd_data(matched: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in labels.itertuples():
        community = matched[matched["leiden_directed"] == record.community_id]
        for emotion in EMOTIONS:
            values = community[emotion].to_numpy(dtype=float)
            rows.append(
                {
                    "community_id": record.community_id,
                    "display_label": record.display_label,
                    "n_authors": record.n_authors,
                    "emotion": emotion,
                    "score_sd": np.std(values, ddof=1),
                    "score_scale": "raw CardiffNLP multilabel score (0–1)",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_OUT / "10_within_community_emotion_sd.csv", index=False)
    return out


def plot_emotion_sd(matched: pd.DataFrame, labels: pd.DataFrame) -> None:
    long = emotion_sd_data(matched, labels)
    matrix = long.pivot(index="display_label", columns="emotion", values="score_sd")
    matrix = matrix.loc[labels["display_label"], EMOTIONS]
    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(18, 13))
            fig.patch.set_facecolor(tokens["background"])
            image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="cividis", interpolation="nearest")
            ax.set_xticks(np.arange(len(EMOTIONS)), [name.title() for name in EMOTIONS], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(matrix)), matrix.index)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, len(EMOTIONS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
            ax.grid(which="minor", color=tokens["background"], linewidth=1.1)
            ax.tick_params(which="minor", bottom=False, left=False)
            colorbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.03)
            colorbar.set_label("Within-community standard deviation")
            colorbar.ax.tick_params(colors=tokens["foreground"])
            colorbar.outline.set_edgecolor(tokens["muted"])
            fig.subplots_adjust(left=0.29, right=0.93, top=0.98, bottom=0.17)
            save(fig, "10_within_community_emotion_sd", theme)


def contrast_data(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    composition = load_raw_matrix("topic_composition.csv")
    sentiment = load_raw_matrix("sentiment_means.csv")
    emotions = load_raw_matrix("emotion_means.csv")
    ids = [0, 8]
    label_map = labels.set_index("community_id")["display_label"].to_dict()
    composition = composition[composition["community_id"].isin(ids)].set_index("community_id")
    sentiment = sentiment[sentiment["community_id"].isin(ids)].set_index("community_id")
    emotions = emotions[emotions["community_id"].isin(ids)].set_index("community_id")

    key_topics = ["AI Art Discourse", "AI Tools/Code", "Bard/LLMs", "Visual AI Art", "General AI"]
    topic_rows = []
    for cid in ids:
        for topic in key_topics:
            topic_rows.append(
                {
                    "community_id": cid,
                    "display_label": label_map[cid],
                    "metric_group": "topic_share",
                    "metric": topic,
                    "value": composition.loc[cid, topic],
                    "is_other": False,
                }
            )
        topic_rows.append(
            {
                "community_id": cid,
                "display_label": label_map[cid],
                "metric_group": "topic_share",
                "metric": "Other topics",
                "value": 1 - composition.loc[cid, key_topics].sum(),
                "is_other": True,
            }
        )
    topic_out = pd.DataFrame(topic_rows)

    sentiment_rows = []
    for cid in ids:
        for metric in ["positive", "neutral", "negative"]:
            sentiment_rows.append(
                {
                    "community_id": cid,
                    "display_label": label_map[cid],
                    "metric_group": "sentiment",
                    "metric": metric,
                    "value": sentiment.loc[cid, metric],
                }
            )
    sentiment_out = pd.DataFrame(sentiment_rows)

    emotion_rows = []
    for cid in ids:
        for metric in EMOTIONS:
            emotion_rows.append(
                {
                    "community_id": cid,
                    "display_label": label_map[cid],
                    "metric_group": "emotion",
                    "metric": metric,
                    "value": emotions.loc[cid, metric],
                }
            )
    emotion_out = pd.DataFrame(emotion_rows)
    pd.concat([topic_out, sentiment_out, emotion_out], ignore_index=True).to_csv(
        DATA_OUT / "11_ai_art_community_contrast.csv", index=False
    )
    return topic_out, sentiment_out, emotion_out


def plot_contrast(labels: pd.DataFrame) -> None:
    topic_data, sentiment_data, emotion_data = contrast_data(labels)
    ids = [0, 8]
    display = labels.set_index("community_id")["display_label"].to_dict()
    topic_metrics = topic_data[topic_data["community_id"] == 0]["metric"].tolist()
    topic_colors = [TOPIC_COLORS.get(metric, "#b8b8bd") for metric in topic_metrics]

    for theme in ["dark", "light"]:
        with figure_theme(theme) as tokens:
            fig = plt.figure(figsize=(16.5, 10))
            grid = fig.add_gridspec(
                2, 2,
                width_ratios=[1.08, 1.2],
                height_ratios=[1, 1],
                wspace=0.36,
                hspace=0.52,
            )
            ax_topic = fig.add_subplot(grid[0, 0])
            ax_sent = fig.add_subplot(grid[1, 0])
            ax_emotion = fig.add_subplot(grid[:, 1])

            y = np.arange(2)
            left = np.zeros(2)
            topic_handles = []
            for metric, color in zip(topic_metrics, topic_colors):
                values = np.array([
                    topic_data[
                        (topic_data["community_id"] == cid)
                        & (topic_data["metric"] == metric)
                    ]["value"].iloc[0]
                    for cid in ids
                ])
                bars = ax_topic.barh(
                    y, values, left=left, color=color,
                    height=0.52, label=metric,
                )
                topic_handles.append(bars[0])
                left += values
            ax_topic.set_yticks(y, [display[cid] for cid in ids])
            ax_topic.invert_yaxis()
            ax_topic.set_xlim(0, 1)
            ax_topic.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax_topic.set_xlabel("Mean topic share")
            ax_topic.set_title("(a)", loc="left", pad=7)
            finish_axes(ax_topic, tokens, "x")

            left = np.zeros(2)
            sentiment_handles = []
            for metric in ["positive", "neutral", "negative"]:
                values = np.array([
                    sentiment_data[
                        (sentiment_data["community_id"] == cid)
                        & (sentiment_data["metric"] == metric)
                    ]["value"].iloc[0]
                    for cid in ids
                ])
                bars = ax_sent.barh(
                    y, values, left=left,
                    color=SENTIMENT_COLORS[metric],
                    height=0.52, label=metric.title(),
                )
                sentiment_handles.append(bars[0])
                left += values
            ax_sent.set_yticks(y, [display[cid] for cid in ids])
            ax_sent.invert_yaxis()
            ax_sent.set_xlim(0, 1)
            ax_sent.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax_sent.set_xlabel("Mean sentiment probability")
            ax_sent.set_title("(b)", loc="left", pad=7)
            finish_axes(ax_sent, tokens, "x")
            ax_sent.legend(
                sentiment_handles,
                ["Positive", "Neutral", "Negative"],
                loc="lower right",
                bbox_to_anchor=(1, 1.04),
                ncol=3,
                borderaxespad=0,
            )

            emotion_order = sorted(
                EMOTIONS,
                key=lambda emotion: abs(
                    emotion_data[
                        (emotion_data["community_id"] == 8)
                        & (emotion_data["metric"] == emotion)
                    ]["value"].iloc[0]
                    - emotion_data[
                        (emotion_data["community_id"] == 0)
                        & (emotion_data["metric"] == emotion)
                    ]["value"].iloc[0]
                ),
            )
            y_em = np.arange(len(emotion_order))
            vals0 = np.array([
                emotion_data[
                    (emotion_data["community_id"] == 0)
                    & (emotion_data["metric"] == emotion)
                ]["value"].iloc[0]
                for emotion in emotion_order
            ])
            vals8 = np.array([
                emotion_data[
                    (emotion_data["community_id"] == 8)
                    & (emotion_data["metric"] == emotion)
                ]["value"].iloc[0]
                for emotion in emotion_order
            ])
            for y_pos, value0, value8 in zip(y_em, vals0, vals8):
                ax_emotion.hlines(
                    y_pos, min(value0, value8), max(value0, value8),
                    color=tokens["grid"], linewidth=2.2,
                )
            point0 = ax_emotion.scatter(
                vals0, y_em, color=community_color(0),
                s=40, label=display[0], zorder=3,
            )
            point8 = ax_emotion.scatter(
                vals8, y_em, color=community_color(8),
                s=40, label=display[8], zorder=3,
            )
            ax_emotion.set_yticks(
                y_em, [emotion.title() for emotion in emotion_order]
            )
            ax_emotion.set_xlim(0, max(vals0.max(), vals8.max()) * 1.12)
            ax_emotion.set_xlabel("Mean emotion probability")
            ax_emotion.set_title("(c)", loc="left", pad=7)
            finish_axes(ax_emotion, tokens, "x")
            fig.legend(
                [point0, point8],
                [display[0], display[8]],
                loc="upper center",
                bbox_to_anchor=(0.76, 0.975),
                ncol=2,
            )

            fig.legend(
                topic_handles,
                topic_metrics,
                loc="lower center",
                bbox_to_anchor=(0.31, 0.015),
                ncol=3,
                title="Topic key",
            )
            fig.subplots_adjust(
                left=0.18, right=0.975,
                top=0.89, bottom=0.18,
            )
            save(fig, "11_ai_art_community_contrast", theme)

def main() -> None:
    global OUT, DATA_OUT, FIG_OUT, WORK
    global MATCHED_JSON, LABELS_CSV, RAW_FIG_DIR, HPC_TABLES

    choices = (
        "timeline", "umap", "topic_sentiment", "community_topic_enrichment",
        "community_sentiment", "topic_weighted_net", "emotion_sd", "contrast",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=WORK,
                        help="External fixed inputs; defaults to data_sets/blog_analysis")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--figures", nargs="+", choices=choices, default=list(choices))
    args = parser.parse_args()

    WORK = args.input_root
    OUT = args.output_root
    DATA_OUT = OUT / "data"
    FIG_OUT = OUT
    MATCHED_JSON = WORK / "matched_authors.json"
    LABELS_CSV = args.labels
    RAW_FIG_DIR = WORK / "matrices"
    HPC_TABLES = WORK / "tables"
    for directory in [
        DATA_OUT, FIG_OUT / "dark" / "png", FIG_OUT / "dark" / "svg",
        FIG_OUT / "light" / "png", FIG_OUT / "light" / "svg",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    requested = set(args.figures)
    labels = label_table() if requested & {
        "community_topic_enrichment", "community_sentiment",
        "topic_weighted_net", "emotion_sd", "contrast",
    } else None
    matched = load_matched() if requested & {"topic_weighted_net", "emotion_sd"} else None

    if "timeline" in requested:
        plot_timeline()
    if "umap" in requested:
        plot_umap()
    if "topic_sentiment" in requested:
        plot_topic_sentiment()
    if "community_topic_enrichment" in requested:
        plot_community_topic_enrichment(labels)
    if "community_sentiment" in requested:
        plot_community_sentiment(labels)
    if "topic_weighted_net" in requested:
        plot_topic_weighted_net(matched, labels)
    if "emotion_sd" in requested:
        plot_emotion_sd(matched, labels)
    if "contrast" in requested:
        plot_contrast(labels)
    print("Requested non-temporal figures complete", flush=True)

if __name__ == "__main__":
    main()
