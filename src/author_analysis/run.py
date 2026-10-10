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
    parser.add_argument('--profiles-dir', type=Path, help='Existing independent full-content profiles')
    parser.add_argument('--communities', type=Path, help='Override the selected backbone community JSON')
    parser.add_argument('--graph', type=Path, help='Override the selected backbone GML/GraphML')
    parser.add_argument('--author-id-attribute', default='label', help='Exact author ID vertex attribute')
    parser.add_argument('--community-algorithm', default='leiden_directed')
    parser.add_argument('--topic-labels', type=Path, help='Labels for this exact topic model; otherwise numeric')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topics', type=int, default=12)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--eta', type=float, default=.1)
    parser.add_argument('--min-documents', type=int, default=5)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.stage not in {'prepare', 'topics'} and not args.backbone:
        parser.error('--backbone is required for network stages')
    if args.topics < 2 or args.min_documents < 1 or args.alpha <= 0 or args.eta <= 0:
        parser.error('Invalid topic model parameters')
    data = args.data_root.resolve()
    output = args.output_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / 'data_sets'):
        parser.error('Inside the repository, put outputs under ignored data_sets/')
    content = output / 'content'
    profiles = args.profiles_dir.resolve() if args.profiles_dir else content
    network = output / 'networks' / str(args.backbone)
    graph = args.graph or data / 'Data Sets/Networks/3_backbones' / f'Full_{args.backbone}_InfoFlow.gml'
    community = args.communities or data / 'Data Sets/Networks/4_communities' / str(args.backbone) / f'Full_{args.backbone}_author_communities.json'
    matched = network / 'data/matched_authors.parquet'
    classified = [data / 'Data Sets/Cleaned Data' / f'ai_full_classified_twitter-roberta-base-{model}.json'
                  for model in ['emotion-multilabel-latest', 'sentiment-latest']]
    profile_files = [profiles / f'ai_general_author_{kind}.csv' for kind in ['sentiment', 'emotions', 'topics']]
    inputs = {'prepare': classified, 'topics': [content / 'ai_general_eligible_tweets.json'],
              'match': [community, *profile_files], 'summarize': [matched], 'communities': [matched],
              'layout': [matched, graph],
              'export': [network / 'layout/matched_nodes.parquet', network / 'layout/build_summary.json'],
              'figure-inputs': [matched, network / 'tables/topic_sentiment_fuzzy_network_subset.csv',
                                network / 'communities/figures/topic_composition.csv'],
              'projection': [matched]}[args.stage]
    if args.topic_labels: inputs.append(args.topic_labels)
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing: parser.error('Missing inputs:\n' + '\n'.join(missing))
    config = dict(data_root=str(data), content=str(content), profiles=str(profiles),
                  network=str(network), graph=str(graph.resolve()), communities=str(community.resolve()),
                  matched=str(matched), author_id_attribute=args.author_id_attribute,
                  community_algorithm=args.community_algorithm, seed=args.seed, topics=args.topics,
                  alpha=args.alpha, eta=args.eta, min_documents=args.min_documents,
                  topic_labels=str(args.topic_labels.resolve()) if args.topic_labels else None)
    if args.check:
        print(json.dumps({'stage': args.stage, 'inputs': [str(p) for p in inputs]}, indent=2))
        return 0
    if args.stage == 'prepare' and profiles != content:
        parser.error('prepare writes content profiles; omit --profiles-dir')
    for directory in ([content, content / 'models'] if args.stage in {'prepare', 'topics'} else
                      [network / name for name in ['data', 'tables', 'logs', 'layout', 'viewer']]):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TWITTER_AI_ANALYSIS_CONFIG=json.dumps(config), MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable, '-m', STAGES[args.stage]], cwd=repo, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
