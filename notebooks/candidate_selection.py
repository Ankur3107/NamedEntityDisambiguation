import pickle
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSHForest


with open('../data/candidate_selection/anchors_dict.pkl', 'rb') as f:
    anchors_dict = pickle.load(f)
    
with open('../data/candidate_selection/lsh_forest.pkl', 'rb') as f:
    forest = pickle.load(f)

top_wikidata = pd.read_csv('../data/candidate_selection/top_wikipages.csv')

    
def preprocess(text):
    text = re.sub(r'[^\w]','_',text)
    text = text.lower()
    return text

def predict(text, database, perms, num_results, forest):    
    #get top results for LSH forest
    text_preprocessed = preprocess(text)
    m = MinHash(num_perm=perms)
    for d in ngrams(text_preprocessed, 3):
        m.update("".join(d).encode('utf-8'))
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    result = database.iloc[idx_array]['wikidata_numeric_id'].astype(int)    
    return result

def get_candidates(entity, lsh_k, anchor_k):
    #get lsh_k candidates for entity from LSH and anchor_k candidates from anchor
    entity = preprocess(entity)
    anchors_candidates = anchors_dict[entity][:anchor_k]
    anchors_candidates = [int(candidate[0]) for candidate in anchors_candidates]
    lsh_candidates = predict(entity, top_wikidata, 128, lsh_k, forest).tolist()
    return set(anchors_candidates + lsh_candidates)
    