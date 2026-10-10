#!/usr/bin/env python3
"""Render the three weekly blog figures from completed weekly aggregation tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import render_non_temporal_figures as common

COLLECTION_END: pd.Timestamp
RESULTS: Path


def prepare_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], utc=True)
    data["week_end"] = pd.to_datetime(data["week_end"], utc=True)
    data["boundary_status"] = np.where(
        data["week_start"] + pd.Timedelta(days=7) > COLLECTION_END,
        "incomplete: collection ended within this week", "complete",
    )
    data["plotted"] = data["boundary_status"].eq("complete")
    return data


def format_dates(ax, frame: pd.DataFrame) -> None:
    ax.set_xlim(frame["week_start"].min(), frame["week_start"].max() + pd.Timedelta(days=6))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.margins(x=0)


def plot_weekly_sentiment() -> None:
    data = prepare_weekly(pd.read_csv(RESULTS / "weekly_sentiment.csv"))
    data["net_sentiment"] = data["positive"] - data["negative"]
    data["score_scale"] = "raw model probabilities; author-balanced weekly mean"
    data["week_definition"] = "ISO week beginning Monday, UTC"
    data.to_csv(common.DATA_OUT / "02_weekly_sentiment.csv", index=False)
    draw = data[data["plotted"]]
    for theme in ("dark", "light"):
        with common.figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(13, 6.3))
            order = ("positive", "neutral", "negative")
            areas = ax.stackplot(
                draw["week_start"], *[draw[name].to_numpy() for name in order],
                labels=[name.title() for name in order],
                colors=[common.SENTIMENT_COLORS[name] for name in order],
                alpha=0.96, linewidth=0.45,
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean sentiment probability")
            ax.set_xlabel("Week beginning Monday (UTC)")
            format_dates(ax, draw)
            common.finish_axes(ax, tokens, "y")
            fig.legend(areas, [name.title() for name in order], loc="upper center",
                       bbox_to_anchor=(0.5, 0.985), ncol=3)
            fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.15)
            common.save(fig, "02_weekly_sentiment", theme)


def plot_weekly_emotions() -> None:
    data = prepare_weekly(pd.read_csv(RESULTS / "weekly_emotions.csv"))
    data["score_scale"] = "raw model probabilities; author-balanced weekly mean"
    data["week_definition"] = "ISO week beginning Monday, UTC"
    data.to_csv(common.DATA_OUT / "03_weekly_emotions.csv", index=False)
    draw = data[data["plotted"]]
    for theme in ("dark", "light"):
        with common.figure_theme(theme) as tokens:
            fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
            axes = axes.ravel()
            for index, emotion in enumerate(common.EMOTIONS):
                ax = axes[index]
                color = common.EMOTION_COLORS[emotion]
                values = draw[emotion]
                ax.plot(draw["week_start"], values, color=color, linewidth=1.55)
                ax.fill_between(draw["week_start"], values, color=color, alpha=0.10)
                ax.set_ylim(0, max(float(values.max()) * 1.12, 0.01))
                ax.set_title(emotion.title(), loc="left", pad=5)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                format_dates(ax, draw)
                common.finish_axes(ax, tokens, "y")
                ax.tick_params(axis="x", labelbottom=index in (8, 9, 10))
            axes[11].axis("off")
            fig.supxlabel("Week beginning Monday (UTC)", y=0.035)
            fig.supylabel("Mean emotion probability", x=0.022)
            fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.085,
                                hspace=0.38, wspace=0.22)
            common.save(fig, "03_weekly_emotions", theme)


def plot_weekly_topics() -> None:
    data = prepare_weekly(pd.read_csv(RESULTS / "weekly_topics_long.csv"))
    data["topic_id"] = data["topic_id"].astype(int)
    topic_names = data[['topic_id', 'topic_label']].drop_duplicates().sort_values('topic_id')
    if topic_names.topic_id.tolist() != list(range(len(topic_names))):
        raise ValueError('Weekly topic IDs must be contiguous with one label each')
    common.TOPICS = topic_names.topic_label.tolist()
    colors = list(common.PALETTE['topic_colors'].values())
    common.TOPIC_COLORS = {name: colors[i % len(colors)] for i, name in enumerate(common.TOPICS)}
    data["score_scale"] = "frozen-model topic probability; author-balanced weekly mean"
    data["week_definition"] = "ISO week beginning Monday, UTC"
    data["interpretation"] = "weekly composition among authors with usable topic text"
    data.to_csv(common.DATA_OUT / "05_weekly_topic_prevalence.csv", index=False)
    draw = data[data["plotted"]]
    wide = draw.pivot(index="week_start", columns="topic_id", values="mean_topic_share")
    wide = wide.reindex(columns=range(len(common.TOPICS))).sort_index()
    for theme in ("dark", "light"):
        with common.figure_theme(theme) as tokens:
            fig, ax = plt.subplots(figsize=(15, 7.2))
            areas = ax.stackplot(
                wide.index, *[wide[index].to_numpy() for index in range(len(common.TOPICS))],
                colors=[common.TOPIC_COLORS[common.TOPICS[index]] for index in range(len(common.TOPICS))],
                labels=[common.TOPICS[index] for index in range(len(common.TOPICS))],
                alpha=0.96, linewidth=0.35,
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean topic share")
            ax.set_xlabel("Week beginning Monday (UTC)")
            format_dates(ax, draw)
            common.finish_axes(ax, tokens, "y")
            fig.legend(areas, [common.TOPICS[index] for index in range(len(common.TOPICS))],
                       loc="center left", bbox_to_anchor=(0.79, 0.52),
                       ncol=1, title="Fixed topic")
            fig.subplots_adjust(left=0.08, right=0.77, top=0.97, bottom=0.14)
            common.save(fig, "05_weekly_topic_prevalence", theme)


def main() -> None:
    global RESULTS, COLLECTION_END
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-dir", type=Path, required=True)
    parser.add_argument("--collection-end", required=True, help="Actual collection end as a timezone-aware ISO timestamp")
    parser.add_argument("--output-root", type=Path, default=common.OUT)
    parser.add_argument("--figures", nargs="+", choices=("sentiment", "emotions", "topics"),
                        default=["sentiment", "emotions", "topics"])
    args = parser.parse_args()
    COLLECTION_END = pd.Timestamp(args.collection_end)
    if COLLECTION_END.tzinfo is None:
        parser.error('--collection-end must include a timezone')
    RESULTS = args.weekly_dir
    common.OUT = args.output_root
    common.DATA_OUT = args.output_root / "data"
    common.FIG_OUT = args.output_root
    common.DATA_OUT.mkdir(parents=True, exist_ok=True)
    requested = set(args.figures)
    if "sentiment" in requested:
        plot_weekly_sentiment()
    if "emotions" in requested:
        plot_weekly_emotions()
    if "topics" in requested:
        plot_weekly_topics()
    print("Requested weekly figures complete", flush=True)


if __name__ == "__main__":
    main()
