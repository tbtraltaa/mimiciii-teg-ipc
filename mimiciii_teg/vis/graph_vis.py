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

def simple_visualization(A, events, patients, CENTRALITY_all, CENTRALITY_P, conf, join_rules, fname):
    # no CENTRALITY visualization if CENTRALITY is None
    G = build_networkx_graph(A, events, patients, None, conf, join_rules)
    visualize_graph(G, fname = fname + "-Graph-No-CENTRALITY")
    G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
    visualize_graph(G, fname = fname + "-Graph")
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.subgraph(list(CENTRALITY_P.keys())), show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    g.save_graph(fname + '_Simple_CENTRALITY_P_' + str(datetime.now()) + '.html')
    attrs = dict([(e['i'], e['type']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    #visualize_vertices(G, list(CENTRALITY_P.keys()), fname + "_Simple_V_CENTRALITY_P_")


def visualize(patients, events, A, V, CENTRALITY_all, CENTRALITY_P, v_paths, paths, conf, join_rules, fname):
    n = len(events)
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
            visualize_SP_tree(G, list(CENTRALITY_P.keys()), paths_P_P, fname+"CENTRALITY_P_Path_P_P")
            visualize_graph(G, list(CENTRALITY_P.keys()), paths_P_P, fname+"all_CENTRALITY_P_Path_P_P")
        else:
            visualize_graph(G, list(CENTRALITY_P.keys()), paths_P, fname+"all_CENTRALITY_P")
        visualize_SP_tree(G, V, v_paths, fname+"SP_all")
        visualize_SP_tree(G, list(CENTRALITY_P.keys()), paths_P, fname+"CENTRALITY_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(CENTRALITY_P.keys()), fname+"V_percentile")
    '''
    if conf['vis']:
        # no CENTRALITY visualization if CENTRALITY is None
        G = build_networkx_graph(A, events, patients, None, conf, join_rules)
        visualize_graph(G, fname = fname + "-Graph-No-CENTRALITY")
        G = build_networkx_graph(A, events, patients, CENTRALITY_all, conf, join_rules)
        visualize_graph(G, fname = fname + "-Graph")
        for n_paths in conf['n_patient_paths']:
            patient_paths, patient_paths_list = get_patient_CENTRALITY_paths(events, CENTRALITY_all, paths, n_paths)
            visualize_paths(G, patient_paths_list, fname + f"{n_paths}_patient_CENTRALITY_paths")
            #patient_paths, patient_paths_list = get_patient_shortest_paths(A, events, CENTRALITY_all, paths, n_paths)
            #visualize_paths(G, patient_paths_list, fname + f"{n_paths}_patient_shortest_paths")
        paths_P = dict([(i, v_paths[i]) for i in CENTRALITY_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_CENTRALITY(CENTRALITY_all, CENTRALITY_P, paths_P, conf['path_percentile'])
            visualize_SP_tree(G, list(CENTRALITY_P.keys()), paths_P_P, fname + "SP_Tree_CENTRALITY_P_Path_P_P")
            G_tmp = copy.deepcopy(G)
            visualize_graph(G_tmp, list(CENTRALITY_P.keys()), paths_P_P, fname + "Graph_CENTRALITY_P_Path_P_P")
        visualize_SP_tree(G, V, v_paths, fname+"SP_Tree_all")
        visualize_SP_tree(G, list(CENTRALITY_P.keys()), paths_P, fname + "SP_Tree_CENTRALITY_P_Path_P")
        G_tmp = copy.deepcopy(G)
        visualize_graph(G_tmp, list(CENTRALITY_P.keys()), paths_P, fname + "Graph_CENTRALITY_P_Path_P")
        G_tmp = copy.deepcopy(G)
        visualize_graph(G_tmp, V, v_paths, fname + "Graph_SP_Tree")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(CENTRALITY_P.keys()), fname + "Vertices_CENTRALITY_P")

def visualize_SP_tree(G, V, paths, fname):
    CENTRALITY_edges = list()
    for v in V:
        for path in paths[v]:
            i = path[0]
            for j in path[1:]:
                CENTRALITY_edges.append((i,j))
                i = j
    CENTRALITY_edges = list(set(CENTRALITY_edges))
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
    g.from_nx(G.edge_subgraph(CENTRALITY_edges), show_edge_weights=True)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')


def visualize_paths(G, paths, fname):
    edges = set()
    for path in paths:
        i = path[0]
        for j in path[1:]:
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

def visualize_vertices(G, V, fname):
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


def visualize_graph(G, V = None, paths = None, fname = 'Graph'):
    if V and paths:
        for v in V:
            for path in paths[v]:
                i = path[0]
                for j in path[1:]:
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
    g.from_nx(G, show_edge_weights=True)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    # g.show("mimic.html")
    g.save_graph(fname + '_' + str(datetime.now()) + '.html')
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15, seed=4572321)
    patient_i = dict([(id_, i) for i, id_ in enumerate(patients.keys())])
    node_color = [patient_i[e['id']] for e in events]
    node_size = [v*20000 for v in CENTRALITY.values()]
    nx.draw_networkx(
        G,
        pos=pos,
        arrows=True,
        with_labels=True,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title("Percolation Centrality", font)
    # Change font color for legend
    font["color"] = "r"
    ax.text(
        0.80,
        0.06,
        "node size = percolation centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()
    '''
