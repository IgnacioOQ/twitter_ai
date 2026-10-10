"""Flag concentrated or materially changed final fits before downstream analysis."""
import numpy as np


def assess_topic_fit(theta, *, selection_largest_share=None, zero_feature_authors=0):
    values = np.asarray(theta)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
        raise ValueError('Topic memberships must contain authors and at least two topics')
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Invalid final topic probabilities')
    counts = np.bincount(values.argmax(axis=1), minlength=values.shape[1])
    largest = float(counts.max() / len(values))
    reasons = []
    # These are review triggers, not an assumption that real topics must be balanced.
    if largest > .90:
        reasons.append('More than 90% of authors have the same dominant topic')
    if selection_largest_share is not None and largest - selection_largest_share > .15:
        reasons.append('Dominant-topic concentration increased by more than 15 percentage points after refitting')
    return {'status': 'review_required' if reasons else 'checks_passed',
            'reasons': reasons, 'authors': len(values), 'topics': values.shape[1],
            'dominant_counts': counts.tolist(), 'largest_dominant_topic_share': largest,
            'mean_topic_weights': values.mean(axis=0, dtype=np.float64).tolist(),
            'selection_largest_topic_share': selection_largest_share,
            'zero_feature_authors': int(zero_feature_authors),
            'interpretation': 'Passing these numerical checks does not establish semantic quality or stability.'}


def require_topic_review_passed(metadata):
    quality = metadata.get('topic_quality', {})
    if quality.get('status') == 'review_required':
        raise ValueError('Topic model requires review before downstream use: ' + '; '.join(quality['reasons']))
