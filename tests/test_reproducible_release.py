"""Integrity and portability regressions for the supplied analysis configuration."""
import json
import pickle
import shutil
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.author_analysis.common import sha256, topic_presentation
from src.author_analysis.check_stability import compare, repeat_fit, read_fit
from src.author_analysis.adopt_reviewed_fit import adopt
from src.author_analysis.topic_quality import assess_topic_fit

ROOT=Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fit_fixture(self):
        source=self.root/'fit';source.mkdir()
        texts=['apple orange pear fruit']*4+['train engine rail station']*4
        ids=[str(90071992547409930+i) for i in range(len(texts))]
        vectorizer=TfidfVectorizer();matrix=vectorizer.fit_transform(texts)
        model=LatentDirichletAllocation(n_components=2,doc_topic_prior=.1,topic_word_prior=.1,
            random_state=3,max_iter=10,n_jobs=1).fit(matrix)
        theta=model.transform(matrix)
        frame=pd.DataFrame(theta,columns=['topic_0','topic_1']);frame.insert(0,'author_id',ids)
        frame.to_csv(source/'author_topics.csv',index=False)
        (source/'author_documents.jsonl').write_text('\n'.join(json.dumps({'author_id':i,'text':t,'n_tweets':3})
            for i,t in zip(ids,texts)),encoding='utf-8')
        documents=dict(documents_sha256=sha256(source/'author_documents.jsonl'),
            tweets_sha256='a'*64,communities_sha256='b'*64,backbone='RetweetedOnce',includes_retweets=True)
        (source/'documents.json').write_text(json.dumps(documents))
        for name,obj in [('vectorizer.pkl',vectorizer),('lda_model.pkl',model)]:
            with (source/name).open('wb') as f:pickle.dump(obj,f)
        meta=dict(K=2,n_authors=len(ids),documents=documents,settings={'seed':3,'n_jobs':1},
            topic_quality=assess_topic_fit(theta))
        for key,name in [('model_sha256','lda_model.pkl'),('vectorizer_sha256','vectorizer.pkl'),('author_topics_sha256','author_topics.csv')]:
            meta[key]=sha256(source/name)
        (source/'model.json').write_text(json.dumps(meta))
        review=dict(model_sha256=meta['model_sha256'],semantic_review_completed=True,
            sensitivity_review_completed=True,approved_for_exploratory_use=True)
        return source,review

    def test_stability_permutation_and_author_alignment(self):
        a=pd.DataFrame([[.9,.1],[.8,.2],[.1,.9],[.2,.8]],index=['1','2','3','4'])
        b=a.iloc[::-1,[1,0]].copy()
        report,_,_=compare(a,b)
        self.assertEqual(report['cosine_aligned_dominant_agreement'],1.)
        self.assertEqual(report['mean_total_variation'],0.)
        self.assertEqual(report['matching'],[1,0])
        with self.assertRaises(ValueError):compare(a,b.rename(index={'1':'5'}))
        uncertain=pd.DataFrame([[.5,.5],[.5,.5]],index=['1','2'])
        self.assertIsNone(compare(uncertain,uncertain)[0]['jointly_confident_agreement'])

    def test_adoption_binds_review_inputs_scores_and_preserves_source(self):
        source,review=self.fit_fixture()
        before={p.name:sha256(p) for p in source.iterdir()}
        self.assertEqual(adopt(source,source,self.root/'adopted',review)['authors'],8)
        self.assertEqual(before,{p.name:sha256(p) for p in source.iterdir()})
        with self.assertRaises(ValueError):adopt(source,source,self.root/'adopted',review)
        with self.assertRaises(ValueError):adopt(source,source,self.root/'bad',dict(review,model_sha256='wrong'))
        with self.assertRaises(ValueError):adopt(source,source,self.root/'unreviewed',dict(review,sensitivity_review_completed=False))
        frame=pd.read_csv(source/'author_topics.csv',dtype={'author_id':str})
        frame[['topic_0','topic_1']]=frame[['topic_1','topic_0']].to_numpy()
        frame.to_csv(source/'author_topics.csv',index=False)
        meta=json.loads((source/'model.json').read_text())
        meta['author_topics_sha256']=sha256(source/'author_topics.csv')
        (source/'model.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'memberships disagree'):adopt(source,source,self.root/'scores',review)

    def test_controlled_repeat_preserves_vectorizer_and_ids(self):
        source,_=self.fit_fixture()
        output=self.root/'repeat'
        repeat_fit(source,output,4,1)
        ma,a=read_fit(source);mb,b=read_fit(output)
        self.assertEqual(sha256(source/'vectorizer.pkl'),sha256(output/'vectorizer.pkl'))
        self.assertTrue(a.index.equals(b.index))
        self.assertEqual(mb['settings']['seed'],4)
        self.assertTrue(mb['diagnostic_only'])

    def test_presentation_matches_exact_configuration(self):
        path=ROOT/'config/retweeted_once_k12/topics.csv'
        labels=topic_presentation(path)
        self.assertEqual(len(labels),12)
        self.assertEqual(labels.iloc[8].colour,'#ff6d3a')
        sys.path.insert(0,str(ROOT/'src/blog_analysis'))
        import render_non_temporal_figures as renderer
        renderer.configure_topics(path)
        self.assertEqual([renderer.TOPIC_COLORS[name] for name in renderer.TOPICS],labels.colour.tolist())
        bad=self.root/'bad.csv'
        pd.concat([labels,labels.iloc[:1]]).to_csv(bad,index=False)
        with self.assertRaises(ValueError):topic_presentation(bad)

    def test_provenance_preflight_rejects_mixed_model_files(self):
        from src.author_analysis.run import main
        source,_=self.fit_fixture()
        network=self.root/'output/networks/RetweetedOnce'
        shutil.copytree(source,network/'topics')
        (network/'data').mkdir()
        (network/'data/matched_authors.parquet').touch()
        meta=json.loads((source/'model.json').read_text())
        proof={key:meta[key] for key in ['model_sha256','vectorizer_sha256','author_topics_sha256']}
        proof.update(input_sha256={'documents_sha256':meta['documents']['documents_sha256']},
                     eligible_authors=8,backbone='RetweetedOnce',community_algorithm='leiden_directed')
        path=self.root/'provenance.json';path.write_text(json.dumps(proof))
        args=['summarize','--data-root',str(self.root),'--output-root',str(self.root/'output'),
              '--backbone','RetweetedOnce','--provenance',str(path),'--check']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args),0)
        with (network/'topics/author_topics.csv').open('a') as handle:handle.write('\n')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as failure:
            main(args)
        self.assertEqual(failure.exception.code,2)

    def test_examples_prefer_original_and_keep_exact_ids(self):
        authors=self.root/'authors.csv';authors.write_text('author_id\n003\n90071992547409931\n')
        posts=self.root/'posts.jsonl'
        rows=[dict(id='1',author_id='003',type='retweeted',text='train engine station railway carriage travel'),
              dict(id='2',author_id='003',type='original',text='apple orange pear fruit'),
              dict(id='3',author_id='90071992547409931',type='quoted',processed_text='painting artist canvas gallery')]
        posts.write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
        output=self.root/'examples.json'
        result=subprocess.run([sys.executable,'-m','src.network.visualization.build_examples',
            '--authors',str(authors),'--posts',str(posts),'--output',str(output)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(output.read_text())
        self.assertEqual(data['posts']['003'][0],'2')
        self.assertEqual(data['posts']['90071992547409931'][2],'quoted')
        self.assertEqual(data['n'],2)


if __name__=='__main__':unittest.main()
