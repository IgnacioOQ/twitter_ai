"""Run an author-analysis stage using explicit inputs and separate backbone outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

STAGES = {
    'prepare': 'src.author_analysis.prepare_authors',
    'topics': 'src.author_analysis.fit_topics',
    'affect': 'src.author_analysis.affect',
    'match': 'src.author_analysis.match_authors',
    'summarize': 'src.author_analysis.summarize',
    'communities': 'src.author_analysis.compare_communities',
    'layout': 'src.network.visualization.build_matched_layout',
    'export': 'src.network.visualization.export_viewer',
    'figure-inputs': 'src.author_analysis.export_figures',
    'projection': 'src.author_analysis.project_topics',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--data-root', type=Path, required=True, help='AI Public Trust data directory')
    parser.add_argument('--output-root', type=Path, required=True, help='Separate writable analysis directory')
    parser.add_argument('--backbone', choices=['RetweetedOnce', 'LWCC'])
    parser.add_argument('--tweets', type=Path, help='Canonical General AI JSON Lines corpus')
    parser.add_argument('--sentiment', type=Path, help='Verified sentiment JSON Lines')
    parser.add_argument('--emotions', type=Path, help='Verified emotion JSON Lines')
    parser.add_argument('--model-config', type=Path, help='JSON overrides for shared model settings')
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--smoke', action='store_true', help='Small model run in isolated smoke outputs')
    parser.add_argument('--communities', type=Path, help='Override the selected backbone community JSON')
    parser.add_argument('--graph', type=Path, help='Override the selected backbone GML/GraphML')
    parser.add_argument('--author-id-attribute', default='label', help='Exact author ID vertex attribute')
    parser.add_argument('--community-algorithm', default='leiden_directed')
    parser.add_argument('--topic-labels', type=Path, help='Labels for this exact topic model; otherwise numeric')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if not args.backbone:
        parser.error('--backbone is required for network stages')
    from .topic_model import ModelSettings
    settings = ModelSettings.load(args.model_config, n_jobs=args.jobs, smoke=args.smoke)
    data = args.data_root.resolve()
    output = args.output_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / 'data_sets'):
        parser.error('Inside the repository, put outputs under ignored data_sets/')
    if args.smoke: output = output / 'smoke'
    network = output / 'networks' / str(args.backbone)
    graph = args.graph or data / 'Data Sets/Networks/3_backbones' / f'Full_{args.backbone}_InfoFlow.gml'
    community = args.communities or data / 'Data Sets/Networks/4_communities' / str(args.backbone) / f'Full_{args.backbone}_author_communities.json'
    matched = network / 'data/matched_authors.parquet'
    cleaned = data / 'Data Sets/Cleaned Data'
    tweets = args.tweets or cleaned / 'AItrust_twits_pruned_dict.json'
    sentiment = args.sentiment or cleaned / 'ai_full_classified_twitter-roberta-base-sentiment-latest.json'
    emotions = args.emotions or cleaned / 'ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json'
    inputs = {'prepare': [tweets, community],
              'topics': [network/'topics/author_documents.jsonl', network/'topics/documents.json', community],
              'affect': [tweets, sentiment, emotions, network/'topics/author_topics.csv'],
              'match': [community, network/'topics/author_topics.csv', network/'affect/author_sentiment.csv', network/'affect/author_emotions.csv'],
              'summarize': [matched], 'communities': [matched], 'layout': [matched, graph],
              'export': [network/'layout/matched_nodes.parquet', network/'layout/build_summary.json'],
              'figure-inputs': [matched, network/'tables/topic_sentiment_fuzzy_network_subset.csv', network/'communities/figures/topic_composition.csv'],
              'projection': [matched]}[args.stage]
    if args.topic_labels: inputs.append(args.topic_labels)
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing: parser.error('Missing inputs:\n' + '\n'.join(missing))
    if args.stage not in ['prepare', 'topics', 'affect']:
        model_metadata = network/'topics/model.json'
        if model_metadata.is_file():
            from .topic_quality import require_topic_review_passed
            require_topic_review_passed(json.loads(model_metadata.read_text(encoding='utf-8')))
    config = dict(data_root=str(data), network=str(network), backbone=args.backbone,
                  graph=str(graph.resolve()), communities=str(community.resolve()),
                  tweets=str(tweets.resolve()), sentiment=str(sentiment.resolve()), emotions=str(emotions.resolve()),
                  matched=str(matched), author_id_attribute=args.author_id_attribute,
                  community_algorithm=args.community_algorithm, seed=args.seed,
                  model_settings=settings.to_dict(), smoke=args.smoke,
                  topic_labels=str(args.topic_labels.resolve()) if args.topic_labels else None)
    if args.check:
        print(json.dumps({'stage': args.stage, 'inputs': [str(p) for p in inputs]}, indent=2))
        return 0
    for name in ['data','tables','logs','layout','viewer']:
        (network/name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TWITTER_AI_ANALYSIS_CONFIG=json.dumps(config), MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable, '-m', STAGES[args.stage]], cwd=repo, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
