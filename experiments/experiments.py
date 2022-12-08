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
from eventgraphs import *
from queries_mimic_extract import *
from queries_PI import *
from queries import *
from percolation import percolation_centrality_with_target


# Event graph configuration
# t_max = [<delta days>, <delta hours>]
join_rules_LF = {
    "t_min": timedelta(days=0, hours=0, minutes=0),
    "t_max": timedelta(days=1, hours=0),
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'age_similarity': 3,  # years
    'icustays-los': 3,  # los similarity in days
    'transfer-los': 3,  # los similarity in days
    'sequential_join': True
}

join_rules_HF = {
    't_min': timedelta(days=0, hours=0, minutes=0),
    't_max': timedelta(days=0, hours=2),
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'age_similarity': 3,  # years
    'icustays-los': 3,  # los similarity in days
    'transfer-los': 3,  # los similarity in days
    'sequential_join': False
}
conf_LF = {
    'max_hours': 168,
    'max_age': 89,
    'min_age': 15,
    'starttime': '2143-01-01',
    'endtime': '2143-01-14',
    'min_missing_percent': 80,
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': False,
    'label': True,
    'PI_states': {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1}
}
conf_HF = {
    'max_hours': 24,
    'max_age': 89,
    'min_age': 15,
    'starttime': '2143-01-01',
    'endtime': '2143-01-02',
    'min_missing_percent': 0,
    'vitals_agg': 'daily',
    'vitals_X_mean': True,
    'interventions': True
}


def eventgraph_mimiciii(event_list, join_rules, conf, file_name):
    conn = get_db_connection()
    patients = get_patient_demography(conn, conf)
    all_events = list()
    n = 0
    for event_name in PI_EVENTS:
        events = get_unique_PI_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    '''
    events = get_events_interventions(conn, conf)
    for i, e in enumerate(events):
        e['i'] = i + n
    n += len(events)
    print('Interventions', len(events))
    all_events += events
    '''
    if conf['vitals_X_mean']:
        events = get_events_vitals_X_mean(conn, conf)
    else:
        events = get_events_vitals_X(conn, conf)
    for i, e in enumerate(events):
        e['i'] = i + n
    n += len(events)
    print('Vitals', len(events))
    all_events += events
    for event_name in event_list:
        events = get_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    print("Total events: ", n)
    print("Total patients:", len(patients))
    A = build_eventgraph(patients, all_events, join_rules)
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t']))
    stage = 0
    for i, e in enumerate(sorted_events):
        if e['type'] == 'PI stage' and e['PI_stage'] != stage:
            stage = e['PI_stage']
        else:
            all_events[e['i']]['PI_stage'] = stage
            all_events[e['i']]['PI_state'] = conf['PI_states'][stage]
        if i + 1 < n:
            if e['id'] != sorted_events[i + 1]['id']:
                stage = 0
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
        states[e['i']] = e['PI_state']
    PC = percolation_centrality_with_target(G, states=states, weight='weight')
    nx.set_node_attributes(G, PC, 'size')
    nx.set_node_attributes(G, PC, 'value')
    attrs = dict([(e['i'], "\n".join([str(k) + ": " + str(v)
        for k, v in e.items()]) + "\nPC: " + PC[e[i]]) for e in all_events])
    nx.set_node_attributes(G, attrs, 'title')
    #attrs = nx.betweenness_centrality(G, weight='weight', normalized=True)
    attrs = dict([(key, {'value': val, 'title': val})
                 for key, val in A.items()])
    nx.set_edge_attributes(G, attrs)
    g = Network(
        directed=True,
        height=1000,
        neighborhood_highlight=True,
        select_menu=True)
    g.hrepulsion()
    # g.barnes_hut()
    g.from_nx(G, show_edge_weights=False)
    g.toggle_physics(True)
    g.show_buttons(filter_=['physics'])
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name)

    fig, ax = plt.subplots(figsize=(20, 15))
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    patient_i = {id_: i for i, id_ in enumerate(patients.keys())}
    node_color = [patient_i[e['id']] for e in enumerate(all_events)]
    node_size = [v*20000 for v in PC.values()]
    nx.draw_networkx(
        H,
        pos=pos,
        arrow=True,
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
    PC_sorted = [[k, v, all_events[k]] for k, v in sorted(PC.items(), key=lambda x: x[1], reverse=True)]
    pprint.pprint(PC_sorted)
    PC_top = dict()
    for info in PC_sorted[:40]:
        if info[2]['type'] not in PC_top:
            PC_top[infor[2]['type']] = 1
        else:
            PC_top[infor[2]['type']] += 1

    pprint.pprint(PC_top)

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
    fname_HF = 'output/mimic_icu_HF'
    fname_HF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf_HF.items() if k in fname_keys])
    fname_HF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules_HF.items()
                                    if k in fname_keys]) + '.html'
    fname_LF = 'output/mimic_icu_LF'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf_LF.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules_LF.items()
                                    if k in fname_keys]) + 'Position_excluded.html'

    # eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_HF, conf_HF, fname_HF)
    eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_LF, conf_LF, fname_LF)
