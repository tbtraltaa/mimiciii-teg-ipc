import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import pprint

from teg.eventgraphs import *
from teg.build_graph import *
from teg.paths import *

def simple_visualization(A, events, patients, PC_all, PC_P, conf, join_rules, fname):
    G = build_networkx_graph(A, events, patients, PC_all, conf, join_rules)
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.subgraph(list(PC_P.keys())), show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    g.save_graph(fname + 'PC_P.html')
    attrs = dict([(e['i'], e['type']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    visualize_vertices(G, list(PC_P.keys()), fname+"V_PC_P")


def visualize(patients, events, A, V, PC_all, PC_P, v_paths, paths, conf, join_rules, fname):
    n = len(events)
    if conf['vis'] and n > 100:
        # when a graph is too large for visualization
        # use only shortest path subgraph
        A = A.toarray()
        A = dok_matrix(A[np.array(V, dtype=int)][:, np.array(V, dtype=int)])
        PC_all = dict([(i, PC_all[v]) for i, v in enumerate(V)])
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
        PC_P = dict([(V.index(i), v) for i, v in PC_P.items()])
        V = range(len(V))
        G = build_networkx_graph(A, events, patients, PC_all, conf, join_rules)
        fname += "_Q" + str(len(conf['quantiles']))
        paths_P = dict([(i, v_paths[i]) for i in PC_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_PC(PC_all, PC_P, paths_P, conf['path_percentile'])
            visualize_SP_tree(G, list(PC_P.keys()), paths_P_P, fname+"PC_P_Path_P_P")
            visualize_graph(G, list(PC_P.keys()), paths_P_P, fname+"all_PC_P_Path_P_P")
        else:
            visualize_graph(G, list(PC_P.keys()), paths_P, fname+"all_PC_P")
        visualize_SP_tree(G, V, v_paths, fname+"SP_all")
        visualize_SP_tree(G, list(PC_P.keys()), paths_P, fname+"PC_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(PC_P.keys()), fname+"V_percentile")
    elif conf['vis']:
        G = build_networkx_graph(A, events, patients, PC_all, conf, join_rules)
        fname += "_Q" + str(len(conf['quantiles']))
        paths_P = dict([(i, v_paths[i]) for i in PC_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_PC(PC_all, PC_P, paths_P, conf['path_percentile'])
            visualize_SP_tree(G, list(PC_P.keys()), paths_P_P, fname+"PC_P_Path_P_P")
            visualize_graph(G, list(PC_P.keys()), paths_P_P, fname+"all_PC_P_Path_P_P")
        else:
            visualize_graph(G, list(PC_P.keys()), paths_P, fname+"all_PC_P")
        visualize_SP_tree(G, V, v_paths, fname+"SP_all")
        visualize_SP_tree(G, list(PC_P.keys()), paths_P, fname+"PC_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(PC_P.keys()), fname+"V_PC_P")

def visualize_patients(events, patients, PC_all):
    pass

def visualize_SP_tree(G, V, paths, fname):
    PC_edges = list()
    for v in V:
        for path in paths[v]:
            i = path[0]
            for j in path[1:]:
                PC_edges.append((i,j))
                i = j
    PC_edges = list(set(PC_edges))
    '''
    unique_paths = set()
    for v in V:
        for path in paths[v]:
            unique_paths.add(tuple(path))
    for path in unique_paths:
        i = path[0]
        for j in path[1:]:
            PC_edges.append((i,j))
            i = j
    '''
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.from_nx(G.edge_subgraph(PC_edges), show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()

    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(fname + '.html')

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
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(fname + '.html')


def visualize_graph(G, V, paths, fname):
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
    g.from_nx(G, show_edge_weights=False)
    g.repulsion()
    #g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(fname + '.html')
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15, seed=4572321)
    patient_i = dict([(id_, i) for i, id_ in enumerate(patients.keys())])
    node_color = [patient_i[e['id']] for e in events]
    node_size = [v*20000 for v in PC.values()]
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
