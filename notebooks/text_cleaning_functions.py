# this part takes a decent amount of time
# maybe can save as unicode dictionary file separately
def get_unicode_dict():
    import sys
    import unicodedata
    
    # get all unicode accented characters for alphabets
    unicode_dict = {}
    for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_str = '['
        if letter.isupper():
            caps = 'CAPITAL '
        else:
            caps = 'SMALL '
        for i in range(sys.maxunicode):
            try:
                if caps+'LETTER '+letter.upper()+' ' in unicodedata.name(chr(i)):
                    letter_str+=chr(i)
            except ValueError:
                continue
        unicode_dict[letter] = letter_str+']'
    return unicode_dict

def correct_whitespace_offset(data):
    data = data.copy()
    data['check'] = data.apply(lambda i: i.section_text[int(i.link_offset_start):int(i.link_offset_end)], axis=1)
    # drop empty anchors
    data = data[data['check']!='']
    # check for hanging whitespace at start
    wrong_link_offsets = data[data['check'].str.contains(r'^ ')] 
    data.loc[data['check'].str.contains(r'^ '), 'link_offset_start'] = [i+1 for i in wrong_link_offsets['link_offset_start'].values]
    # check for hanging whitespace at end
    wrong_link_offsets = data[data['check'].str.contains(r' $')] 
    data.loc[data['check'].str.contains(r' $'), 'link_offset_end'] = [i-1 for i in wrong_link_offsets['link_offset_end'].values]        
    return data

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

def text_preprocessing(data, regex_ls, unicode_dict):
    """Given the wikipedia text dataset, preprocess the text data in the column 'section_text'
    given a regex list to remove from the text data, and then adjust the corresponding link offsets
    under the columns 'link_offset_start', and 'link_offset_end'.
    """
    import re
    from tqdm import tqdm
    data = data.copy()
    
    # existing data has some wrong link offsets initially, these have additional spaces in front of the link anchors
    data = correct_whitespace_offset(data)
    # sort data first by section id, then by link offset
    data.sort_values(['source_section_id', 'link_offset_start'], inplace=True)
    
    # unidecode to replace accents
    data.loc[:,'section_text'] = data.apply(lambda i: replace_accents(i.section_text, unicode_dict), axis=1)
    data.loc[:,'link_anchor'] = data.apply(lambda i: replace_accents(i.link_anchor, unicode_dict), axis=1)
    
    # data cleaning while keeping track of link offsets
    for sid in tqdm(data['source_section_id'].unique()):
        replace_section = data[data['source_section_id'] == sid]
        text = replace_section['section_text'].iloc[0]
        # list of original link offsets
        link_offset_idx = list(replace_section.apply(lambda i: (i.link_offset_start, i.link_offset_end), axis=1))
    
        # remove text based on regex
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
   
        # edit original 
        # replace numbers with hash #
        # https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
#         data.loc[data['source_section_id'] == sid, 'section_text'] = re.sub('[0-9]', '#', text)
        data.loc[data['source_section_id'] == sid, 'section_text'] = text
        data.loc[data['source_section_id'] == sid, 'link_offset_start'] = [i[0] for i in link_offset_idx]
        data.loc[data['source_section_id'] == sid, 'link_offset_end'] = [i[1] for i in link_offset_idx]
        
    # a removed matched entity may leave whitespaces when we calculate offsets, this corrects for it
    # existing data has some wrong link offsets initially, these have additional spaces in front/behind the link anchors
    data = correct_whitespace_offset(data)
    
    data['link_offset_end'] = data['link_offset_end'].astype(int)
    data['link_offset_start'] = data['link_offset_start'].astype(int)
    
    return data

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
    
def extract_entities_and_labels_helper(section, tokenized_text):
    """
    Helper function for extract_entities_and_labels
    """
    import numpy as np
    tokenized_vector = np.zeros(len(tokenized_text))
    # get span object to match characters with tokens
    char_to_token = tokenized_text.char_span(section.link_offset_start, section.link_offset_end)
    
    # label corresponding tokens if there is an anchor link
    # to match anchor text rather than anchor link
    if not char_to_token:
        char_tokens = np.array([token.idx for token in tokenized_text])
        # to account for tokens at end of text
        if np.where(char_tokens >= section.link_offset_end)[0].size > 0:
            closest_start_token = char_tokens[np.where(char_tokens <= section.link_offset_start)[0][-1]]
            closest_end_token = char_tokens[np.where(char_tokens >= section.link_offset_end)[0][0]]-1
            char_to_token = tokenized_text.char_span(closest_start_token, closest_end_token)
            tokenized_vector[char_to_token.start:char_to_token.end] = 1
        else:
            tokenized_vector[np.where(char_tokens <= section.link_offset_start)[0][-1]:-1] = 1
    else:
        tokenized_vector[char_to_token.start:char_to_token.end] = 1 
        
    # extract other relevant information
    data = {(section.link_anchor, section.link_offset_start, 
             section.link_offset_end, 
             section.target_wikidata_numeric_id): tokenized_vector}
    return data

def drop_no_entities(ner_data, label_data):
    """
    Given NER data and true labels data, both in the format
    i.e. [['Apple is a good company', {('Apple', 0, 6, 0101102086): [1,0,0,0,0]}],...]
    check if there are text documents with either no entities identified by NER, or no
    entities with Wikidata links. Drop all of these text documents, and returns the NER 
    and true labels data.
    """
    # check to drop all texts without any ner_entity, or any true entity
    ner_entities_missing = [i for i, text in enumerate(ner_data) if not text[1]]
    true_entities_missing = [i for i, text in enumerate(label_data) if not text[1]]
    
    # drop entries if either does not have any entities
    new_ner_data = [text for i, text in enumerate(ner_data) if i not in ner_entities_missing+true_entities_missing]
    new_label_data = [text for i, text in enumerate(label_data) if i not in ner_entities_missing+true_entities_missing]
    return new_ner_data, new_label_data

def extract_entities_and_labels(data):
    """
    Given a data of wikipedia articles, extract two items, both in the format of a list of lists.
    Each entry in the outer list corresponds to a document. 
    For each document (inner list), the first entry is the actual text document, while the second entry is a dictionary.
    For the first item extracted, the dictionary has the true entities (wikidata links), 
    start characters, end characters, and true Wikidata entry IDs as the key of the dictionary.
    The dictionary has a vector of zeros, with ones at the positions of the corresponding token positions of the true entities
    i.e. ['Apple is a good company', {('Apple', 0, 6, 0101102086): [1,0,0,0,0]}]
    For the second item extracted, the item is largely the same. However, they contain the NER identified entities rather than
    the true entities. In the position of the true Wikidata entry IDs, they have an entity label instead.
    i.e. ['Apple is a good company', {('Apple', 0, 6, 'ORG'): [1,0,0,0,0]}]
    """
    import spacy
    import numpy as np
    import re
    true_entity = []
    ner_entity = []
    
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # for each section of text
    for sid in data['source_section_id'].unique():
        text = data.loc[data['source_section_id']==sid, 'section_text'].iloc[0]   
        
        # tokenizer+NER
        tokenized_text = nlp(text)
        
        # NER entities
        ner_entity_dict = {}
        for entity in tokenized_text.ents:
            if entity.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            ner_tokenized_vector = np.zeros(len(tokenized_text))
            char_to_token = tokenized_text.char_span(entity.start_char, entity.end_char)
            ner_tokenized_vector[char_to_token.start:char_to_token.end] = 1        
            ner_entity_dict[(entity.text, entity.start_char, entity.end_char, entity.label_)] = ner_tokenized_vector
    
        
        # actual wikidata links entities     
        # extract relevant data from dataframe 
        entity_data = (data[data['source_section_id']==sid]
                       .apply(lambda i: extract_entities_and_labels_helper(i, tokenized_text), axis=1))
              
        # form the correct data structure
        # wikidata link true entity
        # replace numbers with hash #
        # https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
        # these were originally kept for NER purposes
        new_text = re.sub('[0-9]', '#', text.lower())
        true_entity.append([new_text, dict()])
        for entity in list(entity_data):
            true_entity[-1][-1].update(entity)
            
        # ner identified entity
        ner_entity.append([new_text, ner_entity_dict])
    
    # drop text documents with either no entities identified by NER, or no entities with Wikidata links
    ner_entity, true_entity = drop_no_entities(ner_entity, true_entity)
    
    return true_entity, ner_entity

def extract_01_vector(data, axis):
    """
    Given data in the format 
    i.e. [['Apple is a good company', {('Apple', 0, 6, 0101102086): [1,0,0,0,0]}],...]
    extract all the 0-1 vectors within each dictionary, 
    concatenate them together along a given axis, and repeat for every single entry in the data, 
    where each entry in the data corresponds to a text document. 
    The function then returns 2 items, the first a list of arrays corresponding to the concatenated
    0-1 vectors, the second a list (of the same length) of dictionaries where the key, value pairs 
    correspond to the index within the arrays (from the 1st item) and the matching key pair of entity
    matches from the data respectively
    """
    import numpy as np
    key_idx_dict_ls = []
    concat_vec_all = []
    for text in data:
        key_idx_dict = {}
        concat_vec = []
        # to check if there are entities in the data
        if not text[1]:
            raise Exception('Drop text documents without entities!')
        else:
            for i, (key, val) in enumerate(text[1].items()):
                concat_vec.append(val)
                key_idx_dict[i] = key
            concat_vec = np.stack(concat_vec, axis=axis)

            # each entry in the list corresponds to a text
            concat_vec_all.append(concat_vec)
            key_idx_dict_ls.append(key_idx_dict)
    return concat_vec_all, key_idx_dict_ls

def drop_entities(ner_entity, true_entity):
    import numpy as np
    # for each identified NER entity in each text
    ner_entities, ner_key_idx_dict_ls = extract_01_vector(ner_entity, axis=1)
    # possible true entities with wikidata links
    true_entities, true_key_idx_dict_ls = extract_01_vector(true_entity, axis=0)
    entity_check_idx_ls = []
    for i in np.arange(len(true_entities)):
        # FOR NOW, AS LONG AS THERE IS SOME OVERLAP, WE KEEP THE TRUE ENTITY
        entity_check = np.sum(true_entities[i] @ ner_entities[i], axis=1)
    
        # entities which have a Wikidata link, but do not have a corresponding NER entity
        entity_check_idx = np.where(entity_check==0)[0]
        entity_check_idx_ls.append(entity_check_idx)
        
    for data, idx_ls, key_idx in zip(true_entity, entity_check_idx_ls, true_key_idx_dict_ls):
        for i in idx_ls:
            del data[1][key_idx[i]]
        
    ner_entity, true_entity = drop_no_entities(ner_entity, true_entity)        
    # possible true entities with wikidata links
    # for each identified NER entity in each text
    ner_entities, ner_key_idx_dict_ls = extract_01_vector(ner_entity, axis=1)
    true_entities, true_key_idx_dict_ls = extract_01_vector(true_entity, axis=0)
    entity_check_idx_ls = []
    for i in np.arange(len(ner_entities)):
        # FOR NOW, AS LONG AS THERE IS SOME OVERLAP, WE KEEP THE TRUE ENTITY
        entity_check = np.sum(true_entities[i] @ ner_entities[i], axis=0)
    
        # entities which have a Wikidata link, but do not have a corresponding NER entity
        entity_check_idx = np.where(entity_check==0)[0]
        entity_check_idx_ls.append(entity_check_idx)
        
    # drop all those NER entities with no corresponding wikidata link
    for data, idx_ls, key_idx in zip(ner_entity, entity_check_idx_ls, ner_key_idx_dict_ls):
        for i in idx_ls:
            del data[1][key_idx[i]]

    # drop document if there is no entity
    ner_entity, true_entity = drop_no_entities(ner_entity, true_entity)
    return ner_entity, true_entity