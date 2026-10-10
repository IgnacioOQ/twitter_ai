"""Project supplied author topic memberships to two dimensions for display."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import umap
from .common import configuration, validate_ids, topic_columns, scores


def main():
    config=configuration()
    matched=validate_ids(pd.read_parquet(config['matched']))
    columns=topic_columns(matched)
    scores(matched,columns,probabilities=True)
    if len(matched)<4:raise ValueError('UMAP requires at least four matched authors')
    neighbors=min(30,len(matched)-1)
    coordinates=umap.UMAP(n_components=2,n_neighbors=neighbors,min_dist=.1,metric='cosine',
                          random_state=0,n_jobs=1).fit_transform(matched[columns].to_numpy(dtype=np.float32))
    out=Path(config['network'])/'figure_inputs'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'author_id':matched.author_id,'umap_1':coordinates[:,0],
                  'umap_2':coordinates[:,1],'topic_id':matched.dominant_topic.astype(int)}).to_csv(
        out/'umap_fixed_topic_scores_seed0.csv',index=False)
    (out/'projection_parameters.json').write_text(json.dumps(dict(n_neighbors=neighbors,min_dist=.1,
        metric='cosine',random_state=0,n_components=2,topics=len(columns))),encoding='utf-8')


if __name__=='__main__':main()
