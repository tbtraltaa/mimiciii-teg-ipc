import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import pprint
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.append('../src')

from queries import *
from eventgraphs import *


# Event graph configuration
# t_max = [<delta days>, <delta hours>] 
join_rules_LF = {
                "t_min": timedelta(days=0, hours=0, minutes=5),
                "t_max": timedelta(days=7, hours=0),
                "t1_min": timedelta(days=0, hours=0, minutes=5),
                "t1_max": timedelta(days=7, hours=0),
                'w_e_max':0.2,
                'join_by_subject': True,
                'age_similarity': 2 #years
            }

join_rules_HF = {
                "t_max": timedelta(days=0, hours=1),
                "t_min": timedelta(days=0, hours=0, minutes=5),
                "t1_min": timedelta(days=0, hours=0, minutes=5),
                "t1_max": timedelta(days=3, hours=0),
                'w_e_max':0.2,
                'join_by_subject': True,
                'age_similarity': 2 #years
            }

def eventgraph_mimiciii(starttime, endtime, event_list, join_rules, file_name):
    conn = get_db_connection()
    patients = get_patient_demography(conn, starttime, endtime)
    all_events = list()
    n = 0
    for event_name in event_list:
        events = get_events(conn, event_name, starttime, endtime)
        for i, e in enumerate(events):
            e['i'] = i + n 
        n += len(events)
        print(event_name, len(events))
        all_events += events 
    print("Total events: ", n)
    print("Total patients:", len(patients))
    A = build_eventgraph(patients, all_events, join_rules)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    G.remove_nodes_from([n for n, d in G.degree if d == 0])
    attrs = dict([(e['i'], {'title': "\n".join([str(k) + ":" + str(v) for k, v in e.items()])}) for e in all_events])
    nx.set_node_attributes(G, attrs)
    attrs = dict([(key, {'value': val, 'title': val}) for key, val in A.items()])
    nx.set_edge_attributes(G, attrs)
    g = Network(directed=True)
    g.hrepulsion()
    g.from_nx(G, show_edge_weights=False)
    g.toggle_physics(True)
    g.show_buttons(filter_=['nodes', 'physics'])
    print(nx.is_directed_acyclic_graph(G))
    """
    g.set_options(
    '''
    const options = {
  "nodes": {
    "borderWidth": null,
    "borderWidthSelected": null,
    "opacity": null,
    "size": null
  },
  "physics": {
    "minVelocity": 0.75
  }
}
    ''')
    """
    #g.show("mimic.html")
    g.save_graph(file_name)


if __name__ == "__main__":
    starttime = '2143-01-01'
    endtime = '2143-01-14'
    file_name_HF = 'output/mimic_HF_1D.html'
    file_name_LF = 'output/mimic_LF_2W_longer.html'
    #eventgraph_mimiciii(starttime, endtime, HIGH_FREQ_EVENTS, join_rules_HF, file_name_HF)
    eventgraph_mimiciii(starttime, endtime, LOW_FREQ_EVENTS, join_rules_LF, file_name_LF)

