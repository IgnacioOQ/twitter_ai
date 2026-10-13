# Twitter Authors Map: 13 October 2026

The [published atlas](https://davidfreeborn.github.io/twitter-authors-map/) now offers the reviewed eight-topic model alongside the twelve-topic default, plus followers, lifetime posts, posts analysed, retweets sent and retweets received. The release covers the same 164,291 matched RetweetedOnce authors. Original coordinates, community assignments, sentiment and emotion scores, example posts and twelve-topic memberships are unchanged.

The complete website code, optional payloads, checks and release history live in [twitter-authors-map](https://github.com/DavidFreeborn/twitter-authors-map). Its [methods](https://github.com/DavidFreeborn/twitter-authors-map/blob/master/METHODS.md) describe the fields and their limitations. The reusable viewer in this research repository continues to accept arbitrary compatible datasets and topic counts; the additional published lenses are specific to this fitted population.

## Topic model and colours

The eight-topic option reuses the reviewed grid runner-up: bigram TF-IDF, alpha 0.3, eta 0.01, seed 42 and 50 maximum iterations. Its combined grid score was 0.7156727875320591. It is not the rejected automatic winner and was not refitted for this release. Both fits use the same author documents (SHA-256 `c4be1839e6479cf455d45dba73bf4ef2070e226d9b6c6f31df66559b56a9adb5`).

An eight-topic group inherits a twelve-topic colour when strictly more than 60% of its dominant-topic authors share that twelve-topic assignment. The denominator is the eight-topic group size. More than one group may inherit the same colour. The payload records all intersections; `scripts/update_topic_colours.py` in the atlas repository reproduces the rule. This visual correspondence does not establish identical topic meaning.

## Counts and retweet direction

Followers and lifetime posts are historical stored public account metrics, not current Twitter counts or necessarily observations from a common date. The last stored row is used for duplicate author IDs; source order is not treated as a verified observation date. One displayed author has no recorded counters. Missing values remain missing. Posts analysed is the number of usable posts in each fitted author document, including retweets, under the existing cleaning and deduplication rules.

Retweets sent and received count weighted events in the full RetweetedOnce study network: 202,710 authors, 882,530 edges and 1,269,747 retweet events. In its information-flow orientation, an edge runs from original author to retweeter. Therefore sent uses incoming strength and received uses outgoing strength. These counts include authors outside the displayed subset; they are not lifetime or all-Twitter totals.

For every displayed author, sent plus received equals the existing `fd` field exactly. Across the full graph, total sent and total received each equal 1,269,747. The source graph SHA-256 is `56cae9e3a56a3f528c57b25096a6b08422693087a4c8a28ce715f88387eff53b`.

## Validation

The atlas checks exact string author IDs, complete coverage, unique records and topic probability sums. Full-data regression checks cover both topic models, all five numeric lenses, selection retention and replacement, failed downloads and retry, author details and unchanged original data. Colour checks cover the strict 60% boundary and exact inherited colours. Browser checks cover model and lens switching, author search, examples, selections in 2D and 3D, and desktop, tablet and phone layouts. Native 200% browser zoom, physical touch and screen-reader narration were not tested.
