import numpy as np
from datetime import timedelta
from scipy.sparse import dok_matrix
import pprint
import networkx as nx

DEFAULT_EVENT_DIFF = 1


def event_difference(e1, e2):
    n = len(e1) - 4
    i = float(0)
    for (k1, v1), (k2, v2) in zip(e1.items(), e2.items()):
        if k1 == 'id' or k1 == 'type' or k1 == 't' or k1 == 'i':
            continue
        elif isinstance(v1, (int, float)) and not isinstance(v1, bool):
            n -= 1
        elif v1 == v2:
            i += 1
    return 1 - i/n

def subject_difference(s1, s2):
    n = len(s1)
    i = float(0)
    for (k1, v1), (k2, v2) in zip(s1.items(), s2.items()):
        if isinstance(v1, (int, float)) and not isinstance(v1, bool):
            n -= 1
        elif v1 == v2:
            i += 1
    return 1 - i/n

def join_by_subject(e1, e2, join_rule):
        '''
        Returns the weight for events of the same subject
        '''
        #time difference
        w_t = (e2['t'] - e1['t']).total_seconds()/join_rule['t_max'].total_seconds()
        if e1['type'] == e2['type']:
            w = w_t + event_difference(e1, e2)
        else:
            w = w_t + DEFAULT_EVENT_DIFF
        return w

    
def join_events(s1, s2, e1, e2, join_rule):
    w = None
    w_t = (e2['t'] - e1['t']).total_seconds()/join_rule['t_max'].total_seconds()
    w_e = event_difference(e1, e2)
    if w_e < join_rule['event_diff_max']:
        w = w_t + w_e
        w += subject_difference(s1, s2) 
        return True, w
    return False, w


def build_eventgraph(subjects, events, join_rule):
    """
    Builds event graphs
    An event = [{id: <id>, event_type: <event_type>, time: <time>, **<event_attributes>}]
    An entity = {id: **<entity_attributes>}
    """
    # number of events
    n = len(events)
    # adjacency matrix
    A = dok_matrix((n, n), dtype=float) 
    for i, e1 in enumerate(events):
        j = i + 1
        for e2 in events[i+1:]:
            join = False
            if e2['t'] > e1['t'] + join_rule['t_max']:
                break
            elif e1['type'] == e2['type'] and e2['t'] >= e1['t'] + join_rule['t_min']:
                s1 = subjects[e1['id']]
                s2 = subjects[e2['id']]
                join, weight = join_events(s1, s2, e1, e2, join_rule)
                if join:
                    A[i, j] = weight
            j += 1
    if join_rule['join_by_subject']:
        tmp = dict.fromkeys(subjects.keys(), [])
        for e in events:
            tmp[e['id']].append(e)
        patient_events = dict([(k, sorted(v, key=lambda x: x['t'])) for k, v in tmp.items() if len(v) !=0])
        for k, v in patient_events.items():
            for i, e1 in enumerate(v):
                for e2 in v[i+1:]:
                    if e2['t'] > e1['t'] + join_rule['t_max']:
                        break
                    elif e2['t'] >= e1['t'] + join_rule['t_min']:
                        A[e1['i'], e2['i']] = join_by_subject(e1, e2, join_rule)
    return A
