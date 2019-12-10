import pickle
import re
import unidecode
from collections import defaultdict

import pandas as pd
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk import ngrams
import spacy

from datasketch import MinHash, MinHashLSHForest

with open('../data/candidate_selection/anchors_dict.pkl', 'rb') as f:
    anchors_dict = pickle.load(f)
    anchors_dict = defaultdict(lambda: [], anchors_dict)

with open('../data/mod_anchor_candidates.pkl', 'rb') as f:
    candidates = pickle.load(f)
    
with open('../data/candidate_selection/lsh_forest.pkl', 'rb') as f:
    forest = pickle.load(f) 

with open('../data/knowledge_graph_data/id2idx_entity.pickle', 'rb') as f:
    # for graph embedding mapping
    id2idx_graph = pickle.load(f)

with open('../data/unicode_dict.pkl', 'rb') as f:
    unicode_dict = pickle.load(f)

# get corresponding wikipedia article title
with open('../data/wiki_id_to_page.pkl', 'rb') as f:
    wiki_id_to_page = pickle.load(f)
    
with open("../data/word2idx/word2idx.pkl", "rb") as f:
    # Map any unknown word to UNKNOWN, which has index 1
    # for mapping to model input
    word2idx = pickle.load(f)

dict_size = len(word2idx)
graph_embedding = np.load('../data/knowledge_graph_data/wiki_DistMult_entity.npy')    
top_wikidata = pd.read_csv('../data/candidate_selection/top_wikipages.csv')

# https://spacy.io/usage/linguistic-features#native-tokenizers
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

# nlp modelg
nlp = spacy.load('en_core_web_lg')


ERROR_ID = 29485
NED_model = load_model("final_NED_model.h5")
WINDOW_LENGTH = 10


# using spacy for now
def entity_recognition(text):
    """Given a text document, run a NER on it using SpaCy and return a dataframe with the following columns
    text: actual raw text input
    entity: identified entity text
    entity_start: character start position of entity in raw text
    entity_end: character end position of entity in raw text
    """
    import spacy
    import pandas as pd
    
    # entity recognition
    doc = nlp(text)
    
    # NER identify all entities
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
            continue
        entities.append([ent, ent.start_char, ent.end_char])
    entities = pd.DataFrame(entities, columns=['entity', 'entity_start', 'entity_end'])
    entities['text'] = text
    return entities

def extract_tokenized_vector(row, tokenized_text):
    """
    Helper function to extract tokenized vector of cleaned text
    """
    import numpy as np
    tokenized_vector = np.zeros(len(tokenized_text), dtype=int)
    # get span object to match characters with tokens
    char_to_token = tokenized_text.char_span(row.entity_start, row.entity_end)    
    
    # label corresponding tokens if there is an anchor link
    # to match anchor text rather than anchor link
    if not char_to_token:
        char_tokens = np.array([token.idx for token in tokenized_text])
        # to account for tokens at end of text
        if np.where(char_tokens >= row.entity_end)[0].size > 0:
            closest_start_token = char_tokens[np.where(char_tokens <= row.entity_start)[0][-1]]
            closest_end_token = char_tokens[np.where(char_tokens >= row.entity_end)[0][0]]-1
            char_to_token = tokenized_text.char_span(closest_start_token, closest_end_token)
            tokenized_vector[char_to_token.start:char_to_token.end] = 1
        else:
            tokenized_vector[np.where(char_tokens <= row.entity_start)[0][-1]:-1] = 1
    else:
        tokenized_vector[char_to_token.start:char_to_token.end] = 1
        
    return tokenized_vector

# might be easier/faster to subsequently change this unicode dict to fixed file
def text_cleaning(ner_data, regex_ls=[('&\w+;|&#[0-9]+;|&#[xX][a-fA-F0-9]+;', ''), 
                                                    ('[^a-zA-Z0-9\s]', ''), 
                                                    ('\s{2,}', ' '),
                                                    ('^ | $', ''),
                                                    ('[0-9]', '#')]):
    """Given the output of entity_recognition (dataframe), clean the text given a regex list 
    and return the dataframe with the cleaned text, entities, 
    and adjusted start and end characters of entity positions"""
    import re
    import spacy
    import pickle

    text = ner_data['text'].iloc[0]
    # clean text for disambiguation task
    text = replace_accents(text, unicode_dict)
    link_offset_idx = list(ner_data.apply(lambda i: (i.entity_start, i.entity_end), axis=1))
    
    for regex in regex_ls:
        match_idx = [(m.start(0), m.end(0)) for m in re.finditer(regex[0], text)]
        if not match_idx:
            continue    
        offset = 0
        for i in match_idx:
            replace_ls = adjust_entity_idx(text, 
                                           i[0]-offset,
                                           i[1]-offset,
                                           regex[1])
            # adjust link offsets
            for j, idx in enumerate(link_offset_idx):
                # account for matched item within entity
                if (i[0]-offset>=idx[0]) & (i[1]-offset<=idx[1]):
                    link_offset_idx[j] = (idx[0], idx[1]-replace_ls[1])
                # original index > replace_end needs to be offset
                elif idx[0] >= i[1]-offset:
                    link_offset_idx[j] = (idx[0]-replace_ls[1], idx[1]-replace_ls[1])
                else:
                    link_offset_idx[j] = (idx[0], idx[1])
                        
            # text preprocessing
            text = replace_ls[0]       
            offset += replace_ls[1]

    # update dataframe
    cleaned_ner_data = ner_data.copy()
    cleaned_ner_data['text'] = text.lower()
    cleaned_ner_data['entity_start'] = [i[0] for i in link_offset_idx]
    cleaned_ner_data['entity_end'] = [i[1] for i in link_offset_idx]
    cleaned_ner_data['entity'] = cleaned_ner_data.apply(lambda i: i.text[i.entity_start:i.entity_end], axis=1)
    
    # get tokenized vector of entity positions
    # helps in getting context windows
    # tokenization
    spacy_tokenizer = spacy.load('en_core_web_sm')
    spacy_tokenizer.tokenizer = WhitespaceTokenizer(spacy_tokenizer.vocab)
    tokenized_text = spacy_tokenizer(text, disable=['parser', 'tagger', 'ner'])
    
    cleaned_ner_data['tokenized_vector'] = (cleaned_ner_data
                                            .apply(lambda i: extract_tokenized_vector(i, tokenized_text), axis=1))
    
    return cleaned_ner_data
    
def replace_accents(text, unicode_dict):
    import re
    for rep, mat in unicode_dict.items():
        text = re.sub(mat, rep, text)
        
    return text

def adjust_entity_idx(text, replace_start, replace_end, replace_word):
    """
    Given a text, and a given start and end position of the original text to replace, the text to replace it by, 
    return a list consisting of the new text, an offset to shift the characters of the text, and the character beyond 
    which an offset is required
    """

    offset = replace_end - replace_start - len(replace_word)
    new_text = text[:replace_start] + replace_word + text[replace_end:]
    # original index > replace_end needs to be offset
    return [new_text, offset]

#right now candidate selection does an additional step of preprocessing after NER. 
def candidate_preprocess(text):
    """
    Given a text, do simple preprocessing of replacing non alphanumeric characters with _ and lowering case.
    Same as preprocess when constructing candidate_selection datasets.
    """
    text = re.sub(r'[^\w]','_',text)
    text = text.lower()
    return text


def lsh_predict(text, database, perms, num_results, forest):    
    """
    Get top results for LSH forest
    """
    text_preprocessed = candidate_preprocess(text)
    m = MinHash(num_perm=perms)
    for d in ngrams(text_preprocessed, 3):
        m.update("".join(d).encode('utf-8'))
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    result = database.iloc[idx_array]['wikidata_numeric_id'].astype(int)    
    return result

def get_candidate_list(entity, lsh_k, anchor_k):
    """
    Get lsh_k candidates for entity from LSH and anchor_k candidates from anchor
    """
    entity = candidate_preprocess(entity)
    anchors_candidates = anchors_dict[entity][:anchor_k]
    anchors_candidates = [int(candidate[0]) for candidate in anchors_candidates]
    lsh_candidates = lsh_predict(entity, top_wikidata, 128, lsh_k, forest).tolist()
    if anchors_candidates is None:
        anchors_candidates = []
    if lsh_candidates is None:
        lsh_candidates = []
    return list(set(anchors_candidates + lsh_candidates))

def disambiguate_candidates(ner_data, window_length=WINDOW_LENGTH):
    """
    Returns dataframe with disambiguated candidates
    """
    ner_data['predicted_wikidata'] = 29485
    for key, value in ner_data.iterrows():
        entity = value.entity
        split_text = value.text.split()
        candidate_list = value.candidate_list
        if len(candidate_list)==0:
            continue
        
        assert len(value.tokenized_vector) == len(split_text)
        
        # find context
        entity_idx = np.where(value.tokenized_vector)[0][0]
        context = split_text[max(0, entity_idx - window_length) : entity_idx] +\
                      split_text[entity_idx + 1 : entity_idx + window_length + 1]
        context_word2idx = [word2idx.get(word, 1) for word in context]
        # inputs
        lstm_input = pad_sequences(
            np.array([context_word2idx]),
            maxlen=window_length * 2,
            padding='post'
        )
        kge_candidates = [
            graph_embedding[id2idx_graph[entity_id]]
            if entity_id in id2idx_graph else None
            for entity_id in candidate_list
        ]
    
        # prediction
        candidate_probs = [
            NED_model.predict([lstm_input, kge_candidate.reshape(1,-1)])[0][0]
            if not kge_candidate is None else 0.0
            for kge_candidate in kge_candidates
        ]
        ner_data.loc[key,'predicted_wikidata'] = candidate_list[np.argmax(candidate_probs)]
    return ner_data

def candidate_selection_and_disambiguation(ner_data):
    # candidate selection and disambiguation
    ner_data['candidate_list'] = ner_data.apply(lambda i: get_candidate_list(i.entity, 10, 10), axis=1)
    # ner_data['predicted_wikidata'] = 29485
    ner_data = disambiguate_candidates(ner_data)
    ner_data['predicted_page'] = ner_data.apply(lambda i: wiki_id_to_page[i.predicted_wikidata], axis=1)
    
    return ner_data

def predict_entity(raw_data):
    """
    Given a raw text, return a dataframe consisting of the raw text, entity start and end positions, 
    and predicted Wikipedia page titles
    """
    # spacy NER (dependent on cases)
    ner_data = entity_recognition(raw_data)
    # preprocess text (without lowering cases)
    cleaned_ner_data = text_cleaning(ner_data)
    
    # candidate selection
    # entity disambiguation
    # link predicted Wikidata IDs with Wikipedia page titles
    pred_data = candidate_selection_and_disambiguation(cleaned_ner_data)
    
    # return predictions back to original dataframe with original raw text
    ner_data['predicted_page'] = pred_data['predicted_page']
    # return dataframe
    return ner_data
def text_link_preprocess(data):
    """
    Given a predicted dataset of entity links, edit/preprocess the text with Wikidata page links in the
    Wikilink extension format.
    i.e. Input: 'Apple was the brain-child of Steve Jobs. It has produced numerous innovations such as the IPhone.'
        Output: '[[Apple Inc.]] was the brain-child of [[Steve Jobs]]. 
        It has produced numerous innovations such as the [[iPhone]].'
    """
    import markdown
    
    # sort data in terms of entity links
    data.sort_values('entity_start', inplace=True)
    # for each entity, add in [[]] brackets
    # replace everything in brackets with corresponding actual Wikipedia page title
    text = data.text.loc[0]

    offset = 0
    for i in range(data.shape[0]):
#         replace_ls = adjust_entity_idx(text, 
#                                        data.loc[i, 'entity_start']-offset,
#                                        data.loc[i, 'entity_end']-offset,
#                                        '[['+data.loc[i, 'predicted_page']+']]')
        wikipedia_stub = '<a href="https://en.wikipedia.org/wiki/{}">{}</a>'.format(data.loc[i,'predicted_page'], data.loc[i, 'entity'])
        replace_ls = adjust_entity_idx(text, 
                                       data.loc[i, 'entity_start']-offset,
                                       data.loc[i, 'entity_end']-offset,
                                       wikipedia_stub)
                        
        # text preprocessing
        text = replace_ls[0]        
        offset += replace_ls[1]
        
    md = markdown.Markdown(extensions=['meta'])
    # return new text string
    return md.convert(text)