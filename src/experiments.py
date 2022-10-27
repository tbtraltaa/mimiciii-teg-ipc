import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import pprint
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

from queries import *
from eventgraphs import *


# Event graph configuration
# t_max = [<delta days>, <delta hours>] 
join_rules = {
                "t_max": timedelta(days=1, hours=0),
                "t_min": timedelta(days=0, hours=0.25),
                'w_e_max':0.2,
                'join_by_subject': True,
                'age_similarity': 2 #years
            }


def eventgraph_mimiciii(starttime, endtime):
    conn = get_db_connection()
    patients = get_patient_demography(conn, starttime, endtime)
    events_list = list()
    n = 0
    for event_name in EVENTS:
        events = get_events(conn, event_name, starttime, endtime)
        for i, e in enumerate(events):
            e['i'] = i + n 
        n += len(events)
        print(event_name, len(events))
        events_list += events 
    #pprint.pprint(events_list)
    print("Total events: ", n)
    print("Total patients:", len(patients))
    A = build_eventgraph(patients, events_list, join_rules)
    G = nx.from_numpy_array(A)
    attrs = dict([(e['i'], {'title': "\n".join([str(k) + ":" + str(v) for k, v in e.items()])}) for e in events_list])
    #pprint.pprint(attrs)
    nx.set_node_attributes(G, attrs)

    g = Network(height=800, width=800, directed=True)
    g.barnes_hut()
    g.from_nx(G)
    g.toggle_physics(True)
    #g.show("mimic.html")
    g.save_graph("output/mimic_debug.html")


if __name__ == "__main__":
    starttime = '2143-01-01'
    endtime = '2143-01-07'
    eventgraph_mimiciii(starttime, endtime)

