import sys
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
import pprint
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../src')
from mimic_events import *
from eventgraphs import *
from percolation import percolation_centrality_with_target, PC_with_target


# Event graph configuration
# t_max = [<delta days>, <delta hours>]
join_rules_LF = {
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": timedelta(days=1, hours=0),
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'age_similarity': 3,  # years
    'icustays-los': 3,  # los similarity in days
    'transfer-los': 3,  # los similarity in days
    'duration_similarity': timedelta(days=2),
    'sequential_join': True
}

conf_LF = {
    'max_hours': 720,
    'max_age': 89,
    'min_age': 15,
    'starttime': '2143-01-07',
    'endtime': '2143-01-14',
    'min_missing_percent': 0,
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'label': True,
    'PI_states': {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'duration': True
}


def eventgraph_mimiciii(event_list, join_rules, conf, file_name, vis=True):
    patients, all_events = mimic_events(event_list, join_rules, conf)
    n = len(all_events)
    A = build_eventgraph(patients, all_events, join_rules)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # G.remove_nodes_from([n for n, d in G.degree if d == 0])
    if conf['label']:
        attrs = dict([(e['i'], 'circle') for e in all_events])
        nx.set_node_attributes(G, attrs, 'shape')
        attrs = dict([(e['i'], e['type']) for e in all_events])
        nx.set_node_attributes(G, attrs, 'label')

    # attrs = dict([(e['i'], e['type']) if 'PI' in e['type'] \
    #        else (e['i'], e['id']) for e in all_events])
    attrs = dict([(e['i'], e['id']) for e in all_events])
    nx.set_node_attributes(G, attrs, 'group')
    states = np.zeros(n)
    for e in all_events:
        states[e['i']] = e['pi_state']
    PC, paths = PC_with_target(G, states=states, weight='weight')
    shapes = dict([(i, 'text') if PC[v] == 0.0 else (i, 'circle') for i, v in enumerate(PC)])
    shapes = dict([(i, shape) if all_events[i]['type']!= 'PI stage' else (i, 'box') for i, shape in shapes.items()])
    nx.set_node_attributes(G, shapes, 'shape')
    nx.set_node_attributes(G, PC, 'size')
    nx.set_node_attributes(G, PC, 'value')
    attrs = dict([(e['i'], "\n".join([str(k) + ": " + str(v)
        for k, v in e.items()]) + "\nPC: " + str(PC[e['i']])) for e in all_events])
    nx.set_node_attributes(G, attrs, 'title')
    #attrs = nx.betweenness_centrality(G, weight='weight', normalized=True)
    attrs = dict([(key, {'value': val, 'title': val})
                 for key, val in A.items()])
    nx.set_edge_attributes(G, attrs)
    PC_sorted = [[k, v, all_events[k]] for k, v in sorted(PC.items(), key=lambda x: x[1], reverse=True) if v > 0]
    pprint.pprint(PC_sorted[:5])
    PC_top = dict()
    num = 100 if len(PC_sorted) > 100 else len(PC_sorted)
    for info in PC_sorted[:num]:
        if info[2]['type'] not in PC_top:
            PC_top[info[2]['type']] = [1, info[1]]
        else:
            PC_top[info[2]['type']][0] += 1
            PC_top[info[2]['type']][1] = info[1]
    pprint.pprint(PC_top)
    #vis = False
    if vis:
        visualize_graph(G, all_events, patients, PC, paths, file_name)


def visualize_graph(G, all_events, patients, PC, paths, file_name):
    PC_related_nodes = set()
    for v, v_paths in paths.items():
        for path in v_paths:
            i = path[0]
            PC_related_nodes.add(i)
            for j in path[1:]:
                PC_related_nodes.add(j)
                G[i][j]['color']='black'
                i = j
    '''
    g = Network(
        directed=True,
        height=1000,
        neighborhood_highlight=True,
        select_menu=True)
    g.hrepulsion()
    g.from_nx(G, show_edge_weights=False)
    # g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons(filter_=['physics'])
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name + '.html')
    '''
    g = Network(
        directed=True,
        height=1000,
        neighborhood_highlight=True,
        select_menu=True)
    g.hrepulsion()
    g.from_nx(G.subgraph(PC_related_nodes), show_edge_weights=False)
    # g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons(filter_=['physics'])
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name + 'PC_related.html')

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15, seed=4572321)
    patient_i = dict([(id_, i) for i, id_ in enumerate(patients.keys())])
    node_color = [patient_i[e['id']] for e in all_events]
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

if __name__ == "__main__":
    fname_keys = [
        'max_hours',
        'starttime',
        'endtime',
        'min_missing_percent',
        'vitals_X_mean',
        'vitals_agg',
        'label',
        't_min',
        't_max',
        'sequential_join']
    fname_LF = 'output/mimic_icu_LF'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf_LF.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules_LF.items()
                                    if k in fname_keys])

    # eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_HF, conf_HF, fname_HF)
    eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_LF, conf_LF, fname_LF)
