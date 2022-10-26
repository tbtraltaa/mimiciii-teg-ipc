import networkx as nx
import matplotlib.pyplot as plt
import pprint
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

from queries import *
from eventgraphs import *


# Event graph configuration
# t_max = [<delta days>, <delta hours>] 
join_rule = {
                "t_max": timedelta(days=1, hours=0),
                'event_diff_max':0.4,
                'join_by_subject': True
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
    exit()
    A = build_eventgraph(patients, events_list, join_rule)
    G = nx.from_numpy_array(A)
    nx.draw(G)
    plt.show()


if __name__ == "__main__":
    starttime = '2143-01-01'
    endtime = '2143-01-02'
    eventgraph_mimiciii(starttime, endtime)

