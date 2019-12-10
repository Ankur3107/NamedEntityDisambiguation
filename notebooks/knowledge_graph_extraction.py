import pandas as pd
import sqlite3
import numbers
import networkx as nx
from collections import defaultdict 
from collections.abc import Iterable
import operator


#Functions for baseline model using direct edges

def conduct_sql_query(db, query):
    #Conducts SQL query and returns results as DataFrame.
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute(query)
    results = c.fetchall()
    df = pd.DataFrame(results)
    if len(df) > 0:
        df.columns = [i[0] for i in c.description]
    conn.close()
    return df

def get_triplet_table(db, table):
    #Converts entire triplet SQL table to dictionary representation
    #key is source node and value is list of tuples
    #First item of tuple is source node, second is the edge property, and third item is target node.
    query = '''
            SELECT 
                source_item_id, edge_property_id, target_item_id 
            FROM 
                {} 
            ;
            '''.format(table)
    query_results = conduct_sql_query(db, query)
    if len(query_results) == 0:
        query_results = pd.DataFrame(columns = ['source_item_id', 'edge_property_id', 'target_item_id'])
    source_dict = defaultdict(list, {})
    target_dict = defaultdict(list, {})
    query_results.apply(lambda row:
                    (source_dict[row['source_item_id']].append(tuple(row)), 
                     target_dict[row['target_item_id']].append(tuple(row))),
                    axis = 1)
    return source_dict, target_dict

def find_source_nodes(db, table, wikidata_id_list):
    #Find all triplets that contain a node in wikidata_id_list as source.
    #Returns dictionary where key is source node and value is list of tuples
    #First item of tuple is source node, second is the edge property, and third item is target node.
    if len(wikidata_id_list) == 1: 
        wikidata_id_str = '(' + str(wikidata_id_list[0]) + ')'
    else:
        wikidata_id_str = str(tuple(wikidata_id_list))
    query = '''
            SELECT 
                source_item_id, edge_property_id, target_item_id 
            FROM 
                {} 
            WHERE 
                source_item_id IN {} 
            ;
            '''.format(table, wikidata_id_str)
    query_results = conduct_sql_query(db, query)
    if len(query_results) == 0:
        query_results = pd.DataFrame(columns = ['source_item_id', 'edge_property_id', 'target_item_id'])
    source_dict = defaultdict(list, {})
    target_dict = defaultdict(list, {})
    query_results.apply(lambda row:
                    (source_dict[row['source_item_id']].append(tuple(row)), 
                    target_dict[row['target_item_id']].append(tuple(row))),
                    axis = 1)
    return source_dict, target_dict
    
def find_target_nodes(db, table, wikidata_id_list):
    #Find all triplets that contain a node in wikidata_id_list as target.
    #Returns dictionary where key is target node and value is list of tuples
    #First item of tuple is source node, second is the edge property, and third item is source node.
    if len(wikidata_id_list) == 1: 
        wikidata_id_str = '(' + str(wikidata_id_list[0]) + ')'
    else:
        wikidata_id_str = str(tuple(wikidata_id_list))
    query = '''
            SELECT 
                source_item_id, edge_property_id, target_item_id 
            FROM 
                {} 
            WHERE 
                target_item_id IN {}
            ;
            '''.format(table, wikidata_id_str)
    query_results = conduct_sql_query(db, query)
    if len(query_results) == 0:
        query_results = pd.DataFrame(columns = ['source_item_id', 'edge_property_id', 'target_item_id'])
    target_dict = defaultdict(list, {})
    query_results.apply(lambda row:
                    target_dict[row['target_item_id']].append(tuple(row)), 
                    axis = 1)
    return target_dict

def construct_subKG(db, table, wikidata_id_list, direct_relations = False):
    #Constructs subKG from table of triplets where source and/or target are in wikidata_id_list
    #When direct_relations is True, both source and target are in wikidata_id_list
    #Else only one need to be in wikidata_id_list
    #Return list containing two dictionaries
    #First dictionary contains source as key and list of (source, edge, target) as item
    #Second dictionary contains target as key and list of (source, edge, target) as item
    if len(wikidata_id_list) == 1: 
        wikidata_id_str = '(' + str(wikidata_id_list[0]) + ')'
    else:
        wikidata_id_str = str(tuple(wikidata_id_list))
    and_or = 'OR'
    if direct_relations:
         and_or = 'AND'
    query = '''
            SELECT 
                source_item_id, edge_property_id, target_item_id 
            FROM 
                {} 
            WHERE 
                target_item_id IN {}
                {} source_item_id IN {}
            ;
            '''.format(table, wikidata_id_str, and_or, wikidata_id_str)
    query_results = conduct_sql_query(db, query)
    if len(query_results) == 0:
        query_results = pd.DataFrame(columns = ['source_item_id', 'edge_property_id', 'target_item_id'])
#     source_dict = defaultdict(list, {})
#     target_dict = defaultdict(list, {})
#     query_results.apply(lambda row:
#                     (source_dict[row['source_item_id']].append(tuple(row)), 
#                     target_dict[row['target_item_id']].append(tuple(row))), 
#                     axis = 1)
    return query_results


def construct_subKG_local(df, wikidata_id_list, direct_relations = False):
    #Constructs subKG from local df of triplets where source and/or target are in wikidata_id_list
    #When direct_relations is True, both source and target are in wikidata_id_list
    #Else only one need to be in wikidata_id_list
    #Returns as DF
    if direct_relations:
        query_results = df[(df['source_item_id'].isin(wikidata_id_list)) & (df['target_item_id'].isin(wikidata_id_list))]
    else:
        query_results = df[(df['source_item_id'].isin(wikidata_id_list)) | (df['target_item_id'].isin(wikidata_id_list))]
    return query_results

def degree(graph_dict, nodes = "all"):
    #Calculates the degrees of nodes in graph_dict
    #graph_dict shuld be a default dict with default item being empty list, 
        #with key being the node and value being the list of (source, edge, target) it is involved in.
    #If nodes = 'all', then get degree of all nodes in graph_dict and return as dictionary with key being the node
    #If nodes is a number, then get degree of that node
    #If nodes is an iterable, then get degree of each node in nodes and return as dictionary with key being the node
    #Note that if graph_dict does not contain the node, degree will be 0
    degree_dict = {}
    if nodes == "all":
        for key, value in graph_dict.items():
            degree_dict[key] = len(value)
        return degree_dict
    elif isinstance(nodes, numbers.Number):
        return len(graph_dict[nodes])
    elif isinstance(nodes, Iterable) and not isinstance(nodes, str):
        for node in nodes:
            degree_dict[node] = len(graph_dict[node])
        return degree_dict
    else:
        raise Exception("Invalid nodes argument!")
        
def convert_graph_dict_to_nx_graph(graph_dict, directed = False):
    #Converts graph_dict to a networkx graph object
    #graph_dict shuld be a default dict with default item being empty list, 
        #with key being the node and value being the list of (source, edge, target) it is involved in.
    #If directed = True, graph dict represents a directed graph
    graph = nx.MultiGraph()
    if directed:
        graph = nx.MultiDiGraph()
    for triplet in sum(graph_dict.values(), []):
        source, edge, target = triplet
        graph.add_node(source)
        graph.add_node(target)
        graph.add_edge(source, target, type_of_edge = edge)
    return graph

def convert_graph_df_to_nx_graph(graph_df, directed = False):
    #Converts graph_dict to a networkx graph object
    #graph_dict shuld be a default dict with default item being empty list, 
        #with key being the node and value being the list of (source, edge, target) it is involved in.
    #If directed = True, graph dict represents a directed graph
    graph = nx.MultiGraph()
    if directed:
        graph = nx.MultiDiGraph()
    return nx.from_pandas_edgelist(graph_df, source = 'source_item_id', target = 'target_item_id', edge_attr = True, create_using = graph)    
def convert_graph_df_to_graph_dict(graph_df):
    source_dict = defaultdict(list, {})
    graph_df.apply(lambda row:
                    source_dict[row['source_item_id']].append(tuple(row)), axis = 1)
    return source_dict


def get_all_k_degree_triplets(graph_dict, source_node, k):
    #Get all triplets from source that are k or fewer degrees away using BFS.     
    visited = set()
    queue = [source_node]
    triplets = []
    for i in range(k):
        new_queue = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                triplets += graph_dict[node]
                new_queue += [triplet[2] for triplet in graph_dict[node]]
            visited.add(node)
        queue = new_queue
    return triplets

def shortest_path(graph_input, source_node, target_node, directed = False):
    #Calculates shortest path between source node and target node and returns path as list
    #graph_dict shuld be a default dict with default item being empty list, 
        #with key being the node and value being the list of (source, edge, target) it is involved in.
    #If directed = True, graph dict represents a directed graph
    #If there is no path, return None
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)    
    try:
        shortest = nx.shortest_path(graph, source_node, target_node)
    except nx.NetworkXNoPath:
        shortest = None
    return shortest

def indegree_centrality(graph_input, directed = False):
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)
    if len(graph) == 1:
        return defaultdict(lambda: 0, {})
    return defaultdict(lambda: 0, nx.in_degree_centrality(graph))

def outdegree_centrality(graph_input, directed = False):
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)    
    if len(graph) == 1:
        return defaultdict(lambda: 0, {})
    return defaultdict(lambda: 0, nx.out_degree_centrality(graph))

def betweenness_centrality(graph_input, directed = False):
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)
    if len(graph) == 1:
        return defaultdict(lambda: 0, {})
    return defaultdict(lambda: 0, nx.betweenness_centrality(graph))

def eigenvector_centrality(graph_input, directed = False):
    #Still need to read more about this. Eigenvector centrality is usually only good for non-directed graph
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)
    if len(graph) == 1:
        return defaultdict(lambda: 0, {})
    return defaultdict(lambda: 0, nx.eigenvector_centrality(graph, max_iter = 1000))

def closeness_centrality(graph_input, directed = False):
    if type(graph_input) == dict:
        graph = convert_graph_dict_to_nx_graph(graph_input, directed)
    else:
        graph = convert_graph_df_to_nx_graph(graph_input, directed)
    if len(graph) == 1:
        return defaultdict(lambda: 0, {})
    return defaultdict(lambda: 0, nx.closeness_centrality(graph))

