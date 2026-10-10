import unittest
import numpy as np
from src.author_analysis.topic_quality import assess_topic_fit, require_topic_review_passed


class TopicQualityTests(unittest.TestCase):
    def test_concentrated_refit_is_preserved_but_cannot_pass_downstream(self):
        theta = np.tile([.8, .2], (100, 1))
        theta[-5:] = [.2, .8]
        report = assess_topic_fit(theta, selection_largest_share=.77, zero_feature_authors=1)
        self.assertEqual(report['dominant_counts'], [95, 5])
        self.assertAlmostEqual(report['largest_dominant_topic_share'], .95)
        self.assertAlmostEqual(report['mean_topic_weights'][0], .77)
        self.assertEqual(len(report['reasons']), 2)
        with self.assertRaisesRegex(ValueError, 'requires review'):
            require_topic_review_passed({'topic_quality': report})

    def test_large_refit_change_is_flagged_without_near_total_concentration(self):
        theta = np.eye(3)[[0]*60 + [1]*20 + [2]*20]
        self.assertEqual(assess_topic_fit(theta, selection_largest_share=.4)['status'], 'review_required')

    def test_uneven_topics_are_not_forced_to_be_balanced(self):
        theta = np.eye(3)[[0]*60 + [1]*30 + [2]*10]
        report = assess_topic_fit(theta, selection_largest_share=.55)
        self.assertEqual(report['status'], 'checks_passed')
        require_topic_review_passed({'topic_quality': report})

    def test_invalid_probabilities_are_rejected(self):
        for values in [[[.8,.8]], [[np.nan,.1]], [[-.1,1.1]]]:
            with self.assertRaises(ValueError): assess_topic_fit(values)
