Please run the notebooks in the following order in order to recreate the datasets used for the final pipeline. Alternatively, we provide the datasets needed to run the pipeline here: 

For the second approach, please move the data to the relevant folder with respect to the notebook folder as we list below.

**redirect_wikidata_id.ipynb**

Purpose:

* Links redirected Wikipages to the target Wikipage's wikidata ID (Or vice versa in some cases).

Uses:

*  raw datasets

Creates: 

* ../data/wikipages_cleaned.csv

**load_data.sh + load_data.sql **

Purpose:

* Creates SQL database to store triplets related to Wikipedia pages. 

Uses:

*  ../data/wikipages_cleaned.csv

* ../data/raw/wikidata_20190805.qpq_item_statements.csv

* ../data/raw/enwiki_20190801.k_plaintext.csv

* ../data/raw/enwiki_20190801.k_link.csv

* ../data/raw/enwiki_20190801.k_raw_anchors.csv

Creates: N/A

**text_cleaning_functions.py**

Purpose: 

* Create functions used to clean textual information and extract entities from a NER to create a sample dataset. Only used to construct sample dataset. (TRAINING PURPOSES)

Uses: N/A

Creates: N/A

**create_candidate_list.ipynb**

Purpose: 

* Creates a candidate list dictionary from anchor text (for training)

Uses:

* ../data/raw/enwiki_20190801.k_raw_anchors.csv

* text_cleaning_functions.py

Creates:

* ../data/anchor_candidates.pkl

* ../data/mod_anchor_candidates.pkl (ensure each candidate list has at least 2 candidates)

**text_feature_engineering.ipynb**

Purpose: 

* Creates a sample dataset

Uses:

* text_cleaning_functions.py

* ../data/kensho.db

* ../data/mod_anchor_candidates.pkl

Creates:

* ../data/plaintext_link_sample_1percent.csv (cleaned text data)

* ../data/sample_data_1percent.csv (ner processed data)

* ../data/sample_labels.pkl (anchor text entity data)

* ../data/sample_data.pkl (ner identified entity data)

**create_word2idx.ipynb**

Purpose: 

* Create word2idx mappings so that words are mapped to numeric index for deep learning purposes

Uses: 

* ../data/sample_data_1percent.csv

Creates: 

* ../data/word2idx.pkl

**deep_learning_model.ipynb**

Purpose: 

* Final deep learning model + ablation

Uses:

* ../data/name_to_wiki_id.pkl

* ../data/knowledge_graph_data/id2text_entity.pickle

* ../data/knowledge_graph_data/idx2id_entity.pickle

* ../data/knowledge_graph_data/wiki_DistMult_entity.npy

* NED_models.py

* ../data/word2idx/word2idx.pkl

* ../data/sample_data_1percent.csv

* ../data/name_to_wiki_id.pkl

Creates:

* final_NED_model.h5

* Intermediate files (self use):

    * ../data/model1/lstm_input_list.npy

    * ../data/model1/graph_input_list.npy

**get_knowledge_graph_data.ipynb**

Purpose: 

* Preparing for knowledge graph training + index to wiki id conversion to identify entity embeddings

Uses:

* ../data/raw/wikidata_20190805.item.csv

* ../data/raw/wikidata_20190805.property.csv

* ../data/wikipages_triplets.csv

Creates:

* Under knowledge graph repo $kg_dir:

    * $kg_dir/train.txt

    * $kg_dir/valid.txt

    * $kg_dir/test.txt

    * $kg_dir/entities.dict

    * $kg_dir/relations.dict

* ../data/knowledge_graph_data/idx2id_edge.pickle

* ../data/knowledge_graph_data/idx2id_entity.pickle

* ../data/knowledge_graph_data/id2text_edge.pickle

* ../data/knowledge_graph_data/id2text_entity.pickle

* ../data/knowledge_graph_data/id2idx_entity.pickle

* ../data/knowledge_graph_data/wiki_DistMult_entity.npy

* ../data/knowledge_graph_data/wiki_DistMult_relation.npy

**candidate_selection.ipynb**

Purpose: 

* Makes the files necessary for candidate selection. Note that this new candidate selection was made after we trained our deep learning model. 

Uses: 

* wikipages_cleaned.csv

Creates: 

* ../data/candidate_selection/top_wikipages.csv

* ../data/candidate_selection/anchors_dict.pkl

* ../data/candidate_selection/lsh_forest.pkl

**pipeline_functions.py**

Purpose: 

* Create functions used to as part of the entity extraction pipeline

* Functions to clean textual information and extract entities from any typical text data, create candidate list of entities, disambiguate and predict entities, and to convert output to text form

Uses: 

* ../data/candidate_selection/anchors_dict.pkl

* ../data/candidate_selection/lsh_forest.pkl

* ../data/knowledge_graph_data/id2idx_entity.pickle

* ../data/unicode_dict.pkl 

    * Included in Github repo

    * Dictionary to replace unicode characters

* ../data/word2idx/word2idx.pkl

* ../data/knowledge_graph_data/wiki_DistMult_entity.npy

* ../data/candidate_selection/top_wikipages.csv

* final_NED_model.h5

Creates: N/A

**full_pipeline.py**

Purpose: 

* Creates a simple widget within Jupyter Notebook interface to demonstrate pipeline

Uses: 

* pipeline_functions.py

Creates: N/A

