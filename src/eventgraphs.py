import numpy as np
from datetime import timedelta
from scipy.sparse import dok_matrix
import pprint
import networkx as nx

from schemas import *

DEFAULT_EVENT_DIFF = 1


def event_difference(e1, e2):
    n = len(e1) - 4
    i = float(0)
    for (k1, v1), (k2, v2) in zip(e1.items(), e2.items()):
        if k1 == 'id' or k1 == 'type' or k1 == 't' or k1 == 'i':
            continue
        elif e1['type'] + '-' + k1 in FLOAT_COLS:
            n -= 1
        elif v1 == v2:
            i += 1
    return 1 - i/n

def subject_difference(s1, s2, join_rules):
    n = len(s1) - 1 #dob
    i = float(0)
    for (k1, v1), (k2, v2) in zip(s1.items(), s2.items()):
        if k1 == 'dob':
            continue
        elif k1 == 'age' and abs(v1 - v2) <= join_rules['age_similarity']:
            i += 1
        elif v1 == v2:
            i += 1
    return 1 - i/n

def weight_same_subject(e1, e2, join_rules):
        '''
        Returns the weight for events of the same subject
        '''
        #time difference
        w_t = (e2['t'] - e1['t']).total_seconds()/join_rules['t_max'].total_seconds()
        if e1['type'] == e2['type']:
            w_e = event_difference(e1, e2)
        else:
            w_e = DEFAULT_EVENT_DIFF
        return w_t, w_e, 0

    
def weight(s1, s2, e1, e2, join_rules):
    w_t = (e2['t'] - e1['t']).total_seconds()/join_rules['t_max'].total_seconds()
    w_e = event_difference(e1, e2)
    w_s = subject_difference(s1, s2, join_rules) 
    return w_t, w_e, w_s


def build_eventgraph(subjects, events, join_rules):
    """
    Builds event graphs
    An event = [{id: <id>, event_type: <event_type>, time: <time>, **<event_attributes>}]
    An entity = {id: **<entity_attributes>}
    """
    # number of events
    n = len(events)
    # adjacency matrix
    A = dok_matrix((n, n), dtype=float) 
    c1 = 0
    for i, e1 in enumerate(events):
        j = i + 1
        for e2 in events[i+1:]:
            if e1['type'] != e2['type']:
                break
            if e2['t'] > e1['t'] + join_rules['t_max']:
                break
            elif e2['t'] >= e1['t'] + join_rules['t_min']:
                s1 = subjects[e1['id']]
                s2 = subjects[e2['id']]
                w_t, w_e, w_s = weight(s1, s2, e1, e2, join_rules)
                if w_e <= join_rules['w_e_max']:
                    A[i, j] = w_t + w_e + w_s
                    c1 += 1
            j += 1
    if join_rules['join_by_subject']:
        events.sort(key=lambda x: x['t'])
        subject_events = dict()
        for e in events:
            if e['id'] not in subject_events:
                subject_events[e['id']] = []
            subject_events[e['id']].append(e)
        #print(patient_events)
        c2 = 0
        for k, v in subject_events.items():
            for i, e1 in enumerate(v):
                for e2 in v[i+1:]:
                    if e2['t'] > e1['t'] + join_rules['t_max']:
                        break
                    elif e2['t'] >= e1['t'] + join_rules['t_min']:
                        w_t, w_e, w_s = weight_same_subject(e1, e2, join_rules)
                        A[e1['i'], e2['i']] = w_t + w_e + w_s
                        c2 += 1
        print(c1, c2)
    return A