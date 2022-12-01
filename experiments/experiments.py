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


# Event graph configuration
# t_max = [<delta days>, <delta hours>]
join_rules_LF = {
    "t_min": timedelta(days=0, hours=1, minutes=0),
    "t_max": timedelta(days=2, hours=0),
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
    't_min': timedelta(days=0, hours=1, minutes=0),
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
    'endtime': '2143-02-01',
    'min_missing_percent': 80,
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': False,
    'label': True,
    'PI_states': {0: 1, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0}
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
    for event_name in PI_EVENT_NAMES:
        events = get_unique_PI_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    events = get_events_interventions(conn, conf)
    for i, e in enumerate(events):
        e['i'] = i + n
    n += len(events)
    print('Interventions', len(events))
    all_events += events
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
    attrs = dict([(e['i'], "\n".join([str(k) + ":" + str(v)
                 for k, v in e.items()])) for e in all_events])
    nx.set_node_attributes(G, attrs, 'title')
    attrs = nx.betweenness_centrality(G, weight='weight', normalized=True)
    nx.set_node_attributes(G, attrs, 'size')
    nx.set_node_attributes(G, attrs, 'value')
    attrs = dict([(key, {'value': val, 'title': val})
                 for key, val in A.items()])
    nx.set_edge_attributes(G, attrs)

    g = Network(
        directed=True,
        height=1100,
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
                                    if k in fname_keys]) + 'PI_groups.html'

    # eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_HF, conf_HF, fname_HF)
    eventgraph_mimiciii(LOW_FREQ_EVENTS, join_rules_LF, conf_LF, fname_LF)
