from collections import defaultdict 
from knowledge_graph_extraction import *

def predict(db, table, text, entity_id_dict, top_k, centrality = 'degree', num_edges_away = 1, directed = True):
    #Get top k predictions for each entity
    wikidata_id_list = sum([value for value in entity_id_dict.values() if value is not None], [])
    if centrality == 'degree':
        graph_df = construct_subKG(db, table, wikidata_id_list, True)
        graph_dict = convert_graph_df_to_graph_dict(graph_df)
        indegree_dict = indegree_centrality(graph_df, directed)
        outdegree_dict = outdegree_centrality(graph_df, directed)
        centrality_dict = {k:indegree_dict[k] + outdegree_dict[k] for k in wikidata_id_list}
    else:
        #For other centrality algos, get all edges that are num_edges_away from candidate wikidata ids.
        source_graph, target_graph = get_triplet_table(db, table)
        relevant_source_graph = defaultdict(list, {})
        for wikidata_id in wikidata_id_list:
            relevant_triplets = get_all_k_degree_triplets(source_graph, wikidata_id, num_edges_away)
            for triplet in relevant_triplets:
                relevant_source_graph[triplet[0]].append(triplet[2])
        if centrality == 'betweenness':
            centrality_dict = betweenness_centrality(relevant_source_graph, directed) 
        elif centrality == 'closeness':
            centrality_dict = closeness_centrality(relevant_source_graph, directed)
        elif centrality == 'eigenvector':
            centrality_dict = eigenvector_centrality(relevant_source_graph, directed)
        centrality_dict = {k:centrality_dict[k] for k in wikidata_id_list}
        graph_dict = relevant_source_graph

    best_entity_dict = {}
    for entity, wikidata_ids in entity_id_dict.items():
        if wikidata_ids is not None:
            relevant_dict = defaultdict(lambda: 0, {k:v for k,v in centrality_dict.items() if k in wikidata_ids})
            best_entity = [t[0] for t in sorted(relevant_dict.items(), key=operator.itemgetter(1))[:top_k]]
            best_entity_dict[entity] = best_entity
    return (text, best_entity_dict), (text, graph_dict)


        
        