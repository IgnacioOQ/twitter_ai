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
from src.author_analysis.common import EMOTIONS, exact_id, scores as validate_scores, sha256

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

    def test_constant_scores_have_undefined_standardized_effect(self):
        from unittest.mock import patch
        import importlib
        with patch.dict(os.environ, {'TWITTER_AI_ANALYSIS_CONFIG':json.dumps({
                'community_algorithm':'leiden_directed','network':str(self.root)})}):
            module=importlib.import_module('src.author_analysis.compare_communities')
        frame=pd.DataFrame({'group':[0,0,1,1], 'value':[.2]*4})
        self.assertTrue(np.isnan(module.calculate_omega_squared(frame,'group','value')))
        result=module.standardized_difference(np.array([[.2]]),np.array([.2]),np.array([0.]))
        self.assertTrue(np.isnan(result).all())
        self.assertIsNone(module.finite_json({'value':float('nan')})['value'])

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
        for backbone, subset in [('RetweetedOnce', ids), ('LWCC', ids[:24])]:
            self.profiles = self.output / 'networks' / backbone
            (self.profiles/'topics').mkdir(parents=True)
            (self.profiles/'affect').mkdir()
            for name, values, columns in [('sentiment',sentiment,['positive','neutral','negative']),
                                         ('emotions',emotions,EMOTIONS),
                                         ('topics',topics,[f'topic_{i}' for i in range(12)])]:
                frame=pd.DataFrame(values[:len(subset)],columns=columns)
                frame['author_id']=subset
                frame['n_tweets']=3
                path=self.profiles/('topics/author_topics.csv' if name=='topics' else f'affect/author_{name}.csv')
                frame.to_csv(path,index=False)
            binding={'backbone':backbone,'author_topics_sha256':sha256(self.profiles/'topics/author_topics.csv')}
            (self.profiles/'topics/model.json').write_text(json.dumps(binding))
            binding.update({kind:{'profile_sha256':sha256(self.profiles/f'affect/author_{kind}.csv')}
                            for kind in ['sentiment','emotions']})
            (self.profiles/'affect/coverage.json').write_text(json.dumps(binding))
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
        path = self.output / 'networks/LWCC/topics/author_topics.csv'
        frame = pd.read_csv(path, dtype={'author_id':str})
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        self.assertIn('Duplicate author IDs', self.stage('match','LWCC',success=False).stderr)
        for value in [123.0, None, '1e18', True]:
            with self.assertRaises(ValueError): exact_id(value)
        frame.to_csv(path,index=False)
        community = self.data / 'Data Sets/Networks/4_communities/LWCC/Full_LWCC_author_communities.json'
        community.write_text('{"003":{"leiden_directed":0},"003":{"leiden_directed":1}}')
        self.assertIn('Duplicate JSON key', self.stage('match','LWCC',success=False).stderr)

    def test_rejects_model_from_other_backbone(self):
        self.fixture()
        path=self.output/'networks/LWCC/topics/model.json'
        info=json.loads(path.read_text())
        info['backbone']='RetweetedOnce'
        path.write_text(json.dumps(info))
        self.assertIn('different model runs',self.stage('match','LWCC',success=False).stderr)

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
                                 type=kind, created_at='2023-01-01T00:00:00Z', text=words[i%4], processed_text=words[i%4],
                                 classifications={
                    'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest':{'scores':dict.fromkeys(EMOTIONS,.2)},
                    'cardiffnlp/twitter-roberta-base-sentiment-latest':{'scores':{'positive':.3,'neutral':.4,'negative':.3}}}))
        for model in ['emotion-multilabel-latest','sentiment-latest']:
            (cleaned / f'ai_full_classified_twitter-roberta-base-{model}.json').write_text(
                '\n'.join(map(json.dumps,rows)),encoding='utf-8')
        canonical = cleaned / 'AItrust_twits_pruned_dict.json'
        canonical.write_text('\n'.join(map(json.dumps, rows)),encoding='utf-8')
        settings = self.root/'settings.json'
        settings.write_text(json.dumps({'k_grid':[4],'alpha_grid':[.1],'eta_grid':[.1],
            'representations':['tfidf_unigram'],'max_iter':3,'grid_sample_authors':32,'coherence_docs':32}))
        for backbone, n in [('RetweetedOnce',32),('LWCC',24)]:
            folder=self.data/'Data Sets/Networks/4_communities'/backbone
            folder.mkdir(parents=True)
            (folder/f'Full_{backbone}_author_communities.json').write_text(json.dumps({
                str(90071992547409930+i):{'leiden_directed':i%4} for i in range(n)}))
            for stage in ['prepare','topics','affect','match']:
                self.stage(stage,backbone,'--model-config',str(settings))
            topics=pd.read_csv(self.output/f'networks/{backbone}/topics/author_topics.csv',dtype={'author_id':str})
            self.assertEqual(len(topics),n)
            self.assertTrue((topics.n_tweets==5).all())
            np.testing.assert_allclose(topics.filter(regex='^topic_').sum(axis=1),1,atol=1e-5)
        self.profiles=self.output/'networks/RetweetedOnce'
        scores=pd.read_csv(self.profiles/'affect/author_sentiment.csv',dtype={'author_id':str})
        self.assertTrue((scores.n_tweets==5).all())
        self.assertTrue((self.profiles/'topics/lda_model.pkl').is_file())
        ids = self.root / 'ids.json'
        ids.write_text(json.dumps(scores.author_id.tolist()))
        result = subprocess.run([sys.executable,'src/blog_analysis/weekly_aggregation.py',
            '--matched-authors',str(ids),'--posts',str(canonical),
            '--sentiment-classified',str(cleaned/'ai_full_classified_twitter-roberta-base-sentiment-latest.json'),
            '--emotion-classified',str(cleaned/'ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json'),
            '--model-dir',str(self.profiles/'topics'),'--backbone','RetweetedOnce',
            '--out-dir',str(self.root/'weekly')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        weekly=pd.read_csv(self.root/'weekly/weekly_sentiment.csv')
        self.assertEqual(int(weekly.n_scored_posts.sum()),160)
        result = subprocess.run([sys.executable,'src/blog_analysis/render_temporal_figures.py',
            '--weekly-dir',str(self.root/'weekly'),'--output-root',str(self.root/'weekly_figures'),
            '--collection-end','2023-02-27T12:00:00Z'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        # Incomplete classifications must fail instead of quietly dropping posts/authors.
        emotion=cleaned/'ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json'
        emotion.write_text('\n'.join(map(json.dumps,rows[:-1])),encoding='utf-8')
        self.assertIn('missing 1 canonical posts',self.stage('affect','RetweetedOnce',success=False).stderr)

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
