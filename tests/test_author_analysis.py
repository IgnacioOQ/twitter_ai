"""Connected workflow checks using synthetic posts, profiles and both backbone formats."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import igraph as ig
import numpy as np
import pandas as pd
from src.author_analysis.common import EMOTIONS, exact_id, scores as validate_scores

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_partial_sunday_is_incomplete(self):
        sys.path.insert(0, str(ROOT / 'src/blog_analysis'))
        import render_temporal_figures as temporal
        frame = pd.DataFrame({'week_start': ['2023-01-02'], 'week_end': ['2023-01-08']})
        temporal.COLLECTION_END = pd.Timestamp('2023-01-08T12:00:00Z')
        self.assertFalse(temporal.prepare_weekly(frame).plotted.iloc[0])
        temporal.COLLECTION_END = pd.Timestamp('2023-01-09T00:00:00Z')
        self.assertTrue(temporal.prepare_weekly(frame).plotted.iloc[0])

    def test_probability_rounding_tolerance(self):
        frame = pd.DataFrame({'positive':[.3], 'neutral':[.4], 'negative':[.2997]})
        validate_scores(frame,list(frame),probabilities=True)
        self.assertEqual(frame.negative.iloc[0],.2997)
        frame['negative']=.2
        with self.assertRaises(ValueError):validate_scores(frame,list(frame),probabilities=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.output = self.root / 'output'
        self.profiles = self.output / 'content'
        self.profiles.mkdir(parents=True)

    def stage(self, name, backbone=None, *extra, success=True):
        command = [sys.executable, '-m', 'src.author_analysis.run', name,
                   '--data-root', str(self.data), '--output-root', str(self.output)]
        if backbone: command += ['--backbone', backbone]
        result = subprocess.run(command + list(extra), cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if success: self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else: self.assertNotEqual(result.returncode, 0)
        return result

    def fixture(self):
        ids = ['003'] + [str(90071992547409930 + i) for i in range(47)]
        rng = np.random.default_rng(42)
        sentiment = rng.dirichlet([1, 2, 3], len(ids))
        emotions = rng.random((len(ids), len(EMOTIONS)))
        topics = rng.dirichlet(np.ones(12), len(ids))
        for name, values, columns in [('sentiment', sentiment, ['positive','neutral','negative']),
                                      ('emotions', emotions, EMOTIONS),
                                      ('topics', topics, [f'topic_{i}' for i in range(12)])]:
            frame = pd.DataFrame(values, columns=columns)
            frame['author_id'] = ids
            if name != 'topics': frame['n_tweets'] = 3
            frame.to_csv(self.profiles / f'ai_general_author_{name}.csv', index=False)
        for backbone, subset in [('RetweetedOnce', ids), ('LWCC', ids[:24])]:
            directory = self.data / 'Data Sets/Networks/4_communities' / backbone
            directory.mkdir(parents=True)
            (directory / f'Full_{backbone}_author_communities.json').write_text(json.dumps({
                aid: {'leiden_directed': i % 4} for i, aid in enumerate(subset)}), encoding='utf-8')
            graph = ig.Graph(n=len(subset), directed=True)
            graph.vs['label'] = subset
            connected = len(subset) - (4 if backbone == 'RetweetedOnce' else 0)
            graph.add_edges([(i, (i+1) % connected) for i in range(connected)] +
                            [(i, (i+5) % connected) for i in range(connected)])
            graph.es['weight'] = [1 + i % 3 for i in range(graph.ecount())]
            path = self.data / 'Data Sets/Networks/3_backbones'
            path.mkdir(parents=True, exist_ok=True)
            graph.write_gml(str(path / f'Full_{backbone}_InfoFlow.gml'))
        return ids

    def test_both_backbones_join_summarize_layout_export(self):
        ids = self.fixture()
        for backbone, count in [('RetweetedOnce',48), ('LWCC',24)]:
            for stage in ['match','summarize','layout','export']:
                self.stage(stage, backbone)
            root = self.output / 'networks' / backbone
            matched = pd.read_parquet(root / 'data/matched_authors.parquet')
            self.assertEqual(len(matched), count)
            self.assertEqual(set(matched.author_id), set(ids[:count]))
            summary = pd.read_csv(root / 'tables/corpus_sentiment_network_subset.csv')
            self.assertAlmostEqual(summary.positive.iloc[0], matched.positive.mean())
            fuzzy = pd.read_csv(root / 'tables/topic_sentiment_fuzzy_network_subset.csv')
            self.assertAlmostEqual(fuzzy.positive.iloc[0], np.average(matched.positive, weights=matched.topic_0))
            nodes = pd.read_parquet(root / 'layout/matched_nodes.parquet')
            viewer = json.loads((root / 'viewer/nodes.json').read_text())
            self.assertEqual(nodes.author_id.tolist(), [row['id'] for row in viewer])
            self.assertIn('003', [row['id'] for row in viewer])
            np.testing.assert_allclose(nodes[['x','y']], [[row['x'],row['y']] for row in viewer])
        self.stage('communities', 'LWCC')
        self.stage('figure-inputs', 'LWCC')
        self.stage('projection', 'LWCC')
        root = self.output / 'networks/LWCC'
        result = subprocess.run([sys.executable,'src/blog_analysis/render_non_temporal_figures.py',
            '--input-root',str(root/'figure_inputs'),'--output-root',str(root/'rendered'),
            '--figures','umap','topic_sentiment','community_topic_enrichment','community_sentiment',
            'topic_weighted_net','emotion_sd'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(list((root/'rendered').rglob('*.png')))

    def test_rejects_duplicate_and_float_ids(self):
        self.fixture()
        path = self.profiles / 'ai_general_author_topics.csv'
        frame = pd.read_csv(path, dtype={'author_id':str})
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        self.assertIn('Duplicate author IDs', self.stage('match','LWCC',success=False).stderr)
        for value in [123.0, None, '1e18', True]:
            with self.assertRaises(ValueError): exact_id(value)
        frame.to_csv(path,index=False)
        community = self.data / 'Data Sets/Networks/4_communities/LWCC/Full_LWCC_author_communities.json'
        community.write_text('{"003":{"leiden_directed":0},"003":{"leiden_directed":1}}')
        self.assertIn('Duplicate JSON key', self.stage('match','LWCC',success=False).stderr)

    def test_preflight_no_writes_and_requires_backbone(self):
        self.fixture()
        before = set(self.root.rglob('*'))
        self.stage('match', 'LWCC', '--check')
        self.assertEqual(before, set(self.root.rglob('*')))
        self.assertIn('--backbone', self.stage('match', success=False).stderr)

    def test_prepare_and_train_topics(self):
        cleaned = self.data / 'Data Sets/Cleaned Data'
        cleaned.mkdir(parents=True)
        words = ['apple pear peach orange lemon melon', 'train rail carriage station engine track',
                 'river ocean beach island water coast', 'music piano guitar violin concert song']
        rows = []
        for i in range(32):
            for j, kind in enumerate(['original','quoted','replied_to','retweeted','retweet']):
                rows.append(dict(id=str(1000+i*10+j), author_id=str(90071992547409930+i),
                                 type=kind, created_at='2023-01-01T00:00:00Z', processed_text=words[i%4],
                                 classifications={
                    'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest':{'scores':dict.fromkeys(EMOTIONS,.2)},
                    'cardiffnlp/twitter-roberta-base-sentiment-latest':{'scores':{'positive':.3,'neutral':.4,'negative':.3}}}))
        for model in ['emotion-multilabel-latest','sentiment-latest']:
            (cleaned / f'ai_full_classified_twitter-roberta-base-{model}.json').write_text(
                '\n'.join(map(json.dumps,rows)),encoding='utf-8')
        self.stage('prepare')
        scores = pd.read_csv(self.profiles / 'ai_general_author_sentiment.csv', dtype={'author_id':str})
        self.assertEqual(len(scores),32)
        self.assertTrue((scores.n_tweets==3).all())
        self.stage('topics', None, '--topics','4')
        topics = pd.read_csv(self.profiles / 'ai_general_author_topics.csv', dtype={'author_id':str})
        self.assertEqual(set(scores.author_id),set(topics.author_id))
        np.testing.assert_allclose(topics.filter(regex='^topic_').sum(axis=1),1,atol=1e-5)
        self.assertTrue((self.profiles / 'models/selected_lda_model.model').is_file())
        ids = self.root / 'ids.json'
        ids.write_text(json.dumps(scores.author_id.tolist()))
        result = subprocess.run([sys.executable,'src/blog_analysis/weekly_aggregation.py',
            '--matched-authors',str(ids),'--eligible-posts',str(self.profiles/'ai_general_eligible_tweets.json'),
            '--sentiment-classified',str(cleaned/'ai_full_classified_twitter-roberta-base-sentiment-latest.json'),
            '--emotion-classified',str(cleaned/'ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json'),
            '--frozen-model',str(self.profiles/'models/selected_lda_model.model'),
            '--frozen-dictionary',str(self.profiles/'models/selected_dictionary.dict'),
            '--out-dir',str(self.root/'weekly')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        weekly=pd.read_csv(self.root/'weekly/weekly_sentiment.csv')
        self.assertEqual(int(weekly.n_scored_posts.sum()),96)
        result = subprocess.run([sys.executable,'src/blog_analysis/render_temporal_figures.py',
            '--weekly-dir',str(self.root/'weekly'),'--output-root',str(self.root/'weekly_figures'),
            '--collection-end','2023-02-27T12:00:00Z'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    @unittest.skipUnless(os.environ.get('SFDP_RUNNER'), 'Set SFDP_RUNNER to a compiled Graphviz helper')
    def test_native_3d_build_and_validate(self):
        self.fixture()
        for stage in ['match','layout','export']: self.stage(stage,'RetweetedOnce')
        root = self.output / 'networks/RetweetedOnce'
        result = subprocess.run([sys.executable,'-m','src.network.visualization.build_layout_3d',
            '--nodes',str(root/'layout/matched_nodes.parquet'),'--edges',str(root/'layout/matched_edges.parquet'),
            '--viewer-nodes',str(root/'viewer/nodes.json'),'--output-dir',str(root/'viewer'),
            '--sfdp-runner',os.environ['SFDP_RUNNER']],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        from src.network.visualization.validate_3d_layout import validate
        self.assertTrue(validate(root/'viewer')['valid'])
        meta=json.loads((root/'viewer/layout3d_meta.json').read_text())
        self.assertEqual(meta['node_count'],48)
        self.assertEqual(meta['rendered_node_count'],44)
        for module, extra in [
            ('evaluate_3d_layout', ['--nodes',str(root/'layout/matched_nodes.parquet'),
                '--edges',str(root/'layout/matched_edges.parquet'),'--pair-sample','1000','--knn','4',
                '--output',str(root/'viewer/evaluation.json')]),
            ('render_layout_3d', ['--output',str(root/'viewer/layout.png')])]:
            result = subprocess.run([sys.executable,'-m','src.network.visualization.'+module,
                '--coordinates',str(root/'viewer/layout3d.bin'), '--indices',str(root/'viewer/layout3d_indices.bin'),
                '--viewer-nodes',str(root/'viewer/nodes.json'),*extra],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        nodes=json.loads((root/'viewer/nodes.json').read_text())
        nodes.reverse()
        (root/'viewer/nodes.json').write_text(json.dumps(nodes))
        with self.assertRaises(ValueError):validate(root/'viewer')


if __name__ == '__main__':unittest.main()
