import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import pprint
from datetime import datetime
import copy

from mimiciii_teg.teg.eventgraphs import *
from mimiciii_teg.teg.build_graph import *
from mimiciii_teg.teg.paths import *

def visualize_centrality(A, events, patients, CENTRALITY_all, CENTRALITY_P, conf, join_rules, fname):
    '''
    Visualize 
        1) the graph 
        2) the graph with centrality
        3) the subgraph of events with higher centrality percentile placement
        4) the events with higher centrality percentile placement, color coded by their event type
    '''
    # no CENTRALITY visualization if CENTRALITY is None
    G = build_networkx_graph(A, events, patients, None, conf, join_rules)
    # visualize the graph 
    visualize_graph(G, fname=fname + "_Graph")
    G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
    # visualize the graph with centrality
    visualize_graph(G, fname=fname + "_Graph_CENTRALITY")
    V_s_r = []
    for e in events:
        if 'Admission' in e['type'] or 'PI Stage' in e['type']:
            V_s_r.append(e['i'])
    # visualize the subgraph of nodes with non-zero centrality
    V_nz = [v for v in CENTRALITY_all if v > 0]
    visualize_graph(G, V=V_nz, fname=fname + "_Graph_NZ_CENTRALITY")
    # visualize the subgraph of events with higher centrality percentile placement
    V_P = list(CENTRALITY_P.keys())
    visualize_vertex_subgraph(G, V=V_P, fname=fname + '_Subgraph_V_P')
    #Visualize the vertices, color coded by their event type
    attrs = dict([(e['i'], e['type']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    visualize_vertices(G, list(CENTRALITY_P.keys()), fname + "_V_P")

def visualize_centrality_ET(A, events, patients, CENTRALITY_all, CENTRALITY_P, ET_CENTRALITY_P, conf, join_rules, fname):
    '''
    Visualize the followings filtered by the most influential event types 
        1) the graph 
        2) the graph with centrality
        3) the subgraph of the most influential vertices
        4) the most influential events, color-coded by their event types
    '''
    # no CENTRALITY visualization if CENTRALITY is None
    G = build_networkx_graph(A, events, patients, None, conf, join_rules)
    # visualize the graph 
    V_s_r = []
    for e in events:
        if 'Admission' in e['type'] or 'PI Stage' in e['type']:
            V_s_r.append(e['i'])
    V_ET = [e['i'] for e in events if e['type'] in ET_CENTRALITY_P]
    visualize_graph(G, V=V_ET, fname=fname + "_Graph_ET_P")
    G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
    # visualize the graph with centrality
    visualize_graph(G, V=V_ET, fname=fname + "_Graph_CENTRALITY_ET_P")
    # visualize the subgraph of nodes with non-zero centrality
    V_nz_ET = [v for v in V_ET if CENTRALITY_all[v] > 0]
    visualize_graph(G, V=V_nz_ET, fname=fname + "_Graph_NZ_CENTRALITY_ET_P")
    # visualize the subgraph of events with higher centrality percentile placement
    V_P_ET =[v for v in V_ET if v in CENTRALITY_P]
    visualize_vertex_subgraph(G, V=V_P_ET, fname=fname + "_Subgraph_V_P_ET_P")
    #Visualize the vertices, color coded by their event type
    attrs = dict([(e['i'], e['type']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    visualize_vertices(G, V=V_P_ET, fname=fname + "_V_P_ET_P")


def visualize_SCP(patients, events, A, CENTRALITY_all, CENTRALITY_P, v_paths, paths, conf, join_rules, fname):
    '''
    Visualize Shortest Centrality Paths (SCP).
    Visualize
        1) the most influential SCP of patients 
        2) SCP of the most influential vertices
        3) the most influential SCP
    '''

    n = len(events)
    if not conf['vis']:
        return
    G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
    # Visualize Patient SCP 
    for n_paths in conf['n_patient_paths']:
        patient_paths, patient_paths_list = get_patient_SCP(events, CENTRALITY_all, paths, n_paths)
        visualize_paths(G, patient_paths_list, fname + f"_{n_paths}_patient_SCP")
        patient_paths, patient_paths_list = get_patient_SP(A, events, CENTRALITY_all, paths, n_paths)
        visualize_paths(G, patient_paths_list, fname + f"_{n_paths}_patient_SP")
    # Visualize SCP of vertices with higher centrality percentile placement
    v_paths_P = dict([(i, v_paths[i]) for i in CENTRALITY_P])
    visualize_V_paths(G, v_paths, fname+"_SCP")
    visualize_V_paths(G, v_paths_P, fname + "_SCP_P")
    G_tmp = copy.deepcopy(G)
    visualize_graph(G_tmp, v_paths=v_paths_P, fname=fname + "_Graph_SCP_P")
    G_tmp = copy.deepcopy(G)
    visualize_graph(G_tmp, v_paths=v_paths, fname=fname + "_Graph_SCP")
    # Visualize SCP of vertices with higher centrality percentile placement
    if conf['path_percentile']:
        paths_P_P = get_influential_SCP(events, conf, CENTRALITY_all, paths, CENTRALITY_P)
        visualize_paths(G, paths_P_P, fname + "_SCP_Path_P_P")
        G_tmp = copy.deepcopy(G)
        visualize_graph(G_tmp, paths=paths_P_P, fname=fname + "_Graph_SCP_Path_P_P")
    '''
    if conf['vis'] and n > 2000:
        # when a graph is too large for visualization
        # use only shortest path subgraph
        A = A.toarray()
        A = dok_matrix(A[np.array(V, dtype=int)][:, np.array(V, dtype=int)])
        CENTRALITY_all = dict([(i, CENTRALITY_all[v]) for i, v in enumerate(V)])
        events = [events[v] for v in V]
        paths_new = dict()
        for v in v_paths:
            if v not in V:
                continue
            if v not in paths_new:
                paths_new[v] = []
            for p in v_paths[v]:
                paths_new[v].append([V.index(i) for i in p])
        paths = paths_new
        CENTRALITY_P = dict([(V.index(i), v) for i, v in CENTRALITY_P.items()])
        V = range(len(V))
        G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
        fname += "_Q" + str(len(conf['quantiles']))
        paths_P = dict([(i, v_paths[i]) for i in CENTRALITY_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_CENTRALITY(CENTRALITY_all, CENTRALITY_P, paths_P, conf['path_percentile'])
            visualize_V_paths(G, list(CENTRALITY_P.keys()), paths_P_P, fname+"CENTRALITY_P_Path_P_P")
            visualize_graph(G, list(CENTRALITY_P.keys()), paths_P_P, fname+"all_CENTRALITY_P_Path_P_P")
        else:
            visualize_graph(G, list(CENTRALITY_P.keys()), paths_P, fname+"all_CENTRALITY_P")
        visualize_V_paths(G, V, v_paths, fname+"SP_all")
        visualize_V_paths(G, list(CENTRALITY_P.keys()), paths_P, fname+"CENTRALITY_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(CENTRALITY_P.keys()), fname+"V_percentile")
    '''

def visualize_SCP_ET(patients,
                     events, 
                     A, 
                     CENTRALITY_all,
                     CENTRALITY_P,
                     ET_CENTRALITY_P,
                     v_paths,
                     paths,
                     conf,
                     join_rules, 
                     fname):
    '''
    Visualize Shortest Centrality Paths (SCP) filtered by central/influential event types.
    Visualize
        1) the most influential SCP of patients filtered by the most influential event types 
        2) SCP of the most influential vertices filtered by the most influential event types
        3) the most influential SCP filtered by the most influential event types
    '''

    n = len(events)
    if not conf['vis']:
        return
    G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
    # the most influential SCP of patients filtered by the most influential event types 
    for n_paths in conf['n_patient_paths']:
        patient_paths, path_list = get_patient_SCP(events,
                                                    CENTRALITY_all,
                                                    paths,
                                                    n_paths)
        path_list = filter_paths_by_ET(events, ET_CENTRALITY_P, path_list) 
        G_tmp = copy.deepcopy(G)
        visualize_paths(G_tmp, path_list, fname + f"_{n_paths}_patient_SCP_ET_P")
        patient_paths, paths_list = get_patient_SP(A, events, CENTRALITY_all, paths, n_paths)
        path_list = filter_paths_by_ET(events, ET_CENTRALITY_P, path_list) 
        G_tmp = copy.deepcopy(G)
        visualize_paths(G_tmp, path_list, fname + f"_{n_paths}_patient_SP_ET_P")
    # SCP of the most influential vertices filtered by the most influential event types
    v_paths_ET_P = filter_V_paths_by_ET(events, ET_CENTRALITY_P, v_paths)
    G_tmp = copy.deepcopy(G)
    visualize_V_paths(G_tmp, v_paths_ET_P, fname+"_SCP_ET_P")
    G_tmp = copy.deepcopy(G)
    visualize_graph(G_tmp, v_paths=v_paths_ET_P, fname=fname + "_Graph_SCP_ET_P")

    V_P_ET_P = [i for i in CENTRALITY_P if events[i]['type'] in ET_CENTRALITY_P] 
    v_paths_P_ET_P = dict([(i, v_paths[i]) for i in V_P_ET_P])
    v_paths_P_ET_P = filter_V_paths_by_ET(events, ET_CENTRALITY_P, v_paths_P_ET_P)
    G_tmp = copy.deepcopy(G)
    visualize_V_paths(G_tmp, v_paths_P_ET_P, fname + "_SCP_P_ET_P")
    G_tmp = copy.deepcopy(G)
    visualize_graph(G_tmp, v_paths=v_paths_P_ET_P, fname=fname + "_Graph_SCP_P_ET_P")

    # the most influential SCP filtered by the most influential event types
    if conf['path_percentile']:
        paths_P_P = get_influential_SCP(events, conf, CENTRALITY_all, paths, CENTRALITY_P)
        paths_P_P = filter_paths_by_ET(events, ET_CENTRALITY_P, paths_P_P)
        G_tmp = copy.deepcopy(G)
        visualize_paths(G_tmp, paths_P_P, fname + "_SCP_Path_P_P_ET_P")
        G_tmp = copy.deepcopy(G)
        visualize_graph(G_tmp, paths=paths_P_P, fname=fname + "_Graph_SCP_Path_P_P_ET_P")
        

def visualize_graph(G, V = None, v_paths=None, fname='Graph', paths=None):
    '''
    Visualize the graph. If paths are given, visualize the paths
    as black edges.
    '''
    if paths:
        for path in paths:
            i = path[0]
            for j in path[1:]:
                # an edge may not exist for paths filtered by event types
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=3, color='grey')
                else:
                    G[i][j]['color']='black'
                i = j

    if v_paths:
        for v in v_paths:
            for path in v_paths[v]:
                i = path[0]
                for j in path[1:]:
                    # an edge may not exist for paths filtered by event types
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=3, color='grey')
                    else:
                        G[i][j]['color']='black'
                    i = j
    '''
    unique_paths = set()
    for v in V:
        for path in paths[v]:
            unique_paths.add(tuple(path))
    for path in unique_paths:
        i = path[0]
        for j in path[1:]:
            G[i][j]['color']='black'
            i = j
    '''
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    if V is not None:
        g.from_nx(G.subgraph(V), show_edge_weights=False)
    else:
        g.from_nx(G, show_edge_weights=True)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')


def visualize_vertices(G, V, fname):
    '''
    Visualize the vertices, color coded by their event type
    '''
    G.remove_edges_from(list(G.edges()))
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.subgraph(V), show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')


def visualize_vertex_subgraph(G, V, fname):
    '''
    Visualize the subgraph of events with given centrality
    '''
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.subgraph(V), show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')


def visualize_V_paths(G, v_paths, fname):
    '''

    '''
    edges = set()
    for v in v_paths.keys():
        for path in v_paths[v]:
            i = path[0]
            for j in path[1:]:
                # an edge may not exist for paths filtered by event types
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=3, color='grey')
                edges.add((i,j))
                i = j
    edges = list(edges)
    '''
    unique_paths = set()
    for v in V:
        for path in paths[v]:
            unique_paths.add(tuple(path))
    for path in unique_paths:
        i = path[0]
        for j in path[1:]:
            CENTRALITY_edges.append((i,j))
            i = j
    '''
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.edge_subgraph(edges), show_edge_weights=True)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')


def visualize_paths(G, paths, fname):
    '''
    Visualize paths
    '''
    edges = set()
    for path in paths:
        i = path[0]
        for j in path[1:]:
            # an edge may not exist for paths filtered by event types
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=3, color='grey')
            edges.add((i,j))
            i = j
    edges = list(edges)
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.edge_subgraph(edges), show_edge_weights=True)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')
