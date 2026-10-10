"""Shared text processing and model settings for notebook, training and weekly inference."""
from dataclasses import dataclass, asdict, fields
import itertools
import json
import re
from pathlib import Path
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

METHOD_VERSION = 'backbone-all-posts-v1'
COMMUNITY_METHODS = ['label_propagation', 'louvain', 'leiden_fast', 'leiden_directed', 'infomap']
REPRESENTATIONS = {'tfidf_unigram': (1, 1), 'tfidf_bigram': (1, 2)}

@dataclass
class ModelSettings:
    min_tweets: int = 3
    grid_sample_authors: int = 100_000
    coherence_docs: int = 20_000
    k_grid: tuple = (8, 12, 16, 20)
    alpha_grid: tuple = (.05, .1, .3)
    eta_grid: tuple = (.01, .1)
    representations: tuple = ('tfidf_unigram', 'tfidf_bigram')
    max_df: float = .7
    min_df: int = 5
    max_iter: int = 50
    seed: int = 42
    n_jobs: int = 1
    selected_model: tuple | None = None
    max_posts: int | None = None

    def __post_init__(self):
        if self.min_tweets < 1 or self.grid_sample_authors < 2 or self.coherence_docs < 2:
            raise ValueError('Invalid author/sample thresholds')
        if not self.k_grid or any(k < 2 for k in self.k_grid): raise ValueError('Invalid topic counts')
        if not self.alpha_grid or not self.eta_grid or min(*self.alpha_grid, *self.eta_grid) <= 0:
            raise ValueError('Topic priors must be positive')
        if not self.representations or set(self.representations) - set(REPRESENTATIONS):
            raise ValueError('Unknown representation')
        if not 0 < self.max_df <= 1 or self.min_df < 1 or self.max_iter < 1 or self.n_jobs < 1:
            raise ValueError('Invalid model parameters')
        if self.max_posts is not None and self.max_posts < 1: raise ValueError('Invalid post limit')
        if self.selected_model is not None:
            rep, k, alpha, eta = self.selected_model
            if rep not in REPRESENTATIONS or k < 2 or alpha <= 0 or eta <= 0:
                raise ValueError('Invalid selected model')

    @classmethod
    def load(cls, path=None, *, n_jobs=1, smoke=False):
        values = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown model settings: {sorted(unknown)}')
        values['n_jobs'] = n_jobs
        if smoke:
            values.update(max_posts=500_000, grid_sample_authors=5_000, coherence_docs=2_000,
                          k_grid=[8], alpha_grid=[.1], eta_grid=[.1], max_iter=10)
        if values.get('max_posts') is not None and not smoke:
            raise ValueError('Post limits require --smoke and isolated smoke outputs')
        return cls(**values)

    def to_dict(self): return asdict(self)

STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | {
    'http','https','user','rt','amp','tco','www','co',
    'ai','chatgpt','gpt','openai','google','microsoft',
    'artificial','intelligence','machine','learning',
    'chat','bot','model','data','technology','tech',
    'latest','search','bing','chatbot','generative','llm','deeplearning',
    'like','just','don','know','think','people','make','going',
    'want','really','good','time','thing','say','way','look',
    'right','come','got','need','let','get','go','take',
    'see','tell','give','try','ask','feel','talk','keep',
    'start','lot','much','still','even','would','could',
    'one','also','well','back','day','year','new','said',
    'may','actually','thanks','use','work','write','read',
    'help','learn','build','create','tool','app','world','today','week',
    'lol','oh','yeah','yes','ok',
    'gonna','wanna','gotta','didnt','doesnt','dont','cant','wont',
    'im','ive','id','youre','hes','shes','theyre','thats',
    'theres','whats','whos','hows',
})

def clean_tweet_text(text):
    # Same rules as preprocess() in 02_Processing/02: lowercase, drop the RT marker, URLs and
    # @mentions, keep hashtag words, collapse whitespace.
    text = (text or "").lower()
    text = re.sub(r'^rt\s+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+:?\s*', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def top_terms(lda, vocab, n):
    return [[vocab[i] for i in comp.argsort()[::-1][:n]] for comp in lda.components_]

def diversity(tops):
    words = [w for t in tops for w in t]
    return len(set(words)) / len(words)

def collapse_ratio(tops, top_n=10, threshold=0.7):
    sets = [set(t[:top_n]) for t in tops]
    pairs = list(itertools.combinations(sets, 2))
    return sum(len(a & b) / top_n >= threshold for a, b in pairs) / len(pairs) if pairs else 0.0

def coherence_cv(tops, texts, dictionary, n_jobs=1):
    # Scored on unigram texts for both representations, so the two are comparable: a bigram
    # term contributes its two words (first occurrence kept, cut back to N_TOP_TERMS words).
    tops = [list(dict.fromkeys(w for term in t for w in term.split()))[:15] for t in tops]
    known = [[w for w in t if w in dictionary.token2id] for t in tops]
    known = [t for t in known if len(t) >= 2]
    if not known:
        return np.nan
    cm = CoherenceModel(topics=known, texts=texts, dictionary=dictionary, coherence="c_v",
                        processes=max(n_jobs - 1, 1))
    return float(cm.get_coherence())


def make_vectorizer(settings, representation='tfidf_unigram'):
    return TfidfVectorizer(lowercase=True, stop_words=STOPWORDS,
        max_df=settings.max_df, min_df=settings.min_df,
        ngram_range=REPRESENTATIONS[representation], dtype=np.float32)


def make_lda(settings, k, alpha, eta):
    return LatentDirichletAllocation(n_components=int(k), doc_topic_prior=float(alpha),
        topic_word_prior=float(eta), learning_method='batch', max_iter=settings.max_iter,
        random_state=settings.seed, n_jobs=settings.n_jobs)
