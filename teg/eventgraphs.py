import copy
import numpy as np
from datetime import timedelta
import itertools
from scipy.sparse import dok_matrix
import pprint
import networkx as nx

from teg.schemas import *
from teg.schemas_PI import *
from teg.schemas_chart_events import *


def event_difference(e1, e2, join_rules):
    # events of different types
    if e1['type'] != e2['type']:
        return join_rules['w_e_default'], {}
    # events of same types
    n = len(e1) - len(join_rules['IDs'])
    i = float(0)
    I = dict() #intersection
    for k1 in e1:
        if k1 in join_rules['IDs']:
            continue
        elif k1 == 'duration' and e2[k1] - e1[k1] <= join_rules['duration_similarity']:
            i += 1
        elif f"{e1['parent_type']}-{k1}" in NUMERIC_COLS and not join_rules['include_numeric']:
            n -= 1
        elif e1['parent_type'] in CHART_EVENTS_NUMERIC and not join_rules['include_numeric']:
            n -= 1
        elif e1['parent_type'] in PI_EVENTS_NUMERIC and not join_rules['include_numeric']:
            n -= 1
        elif k1 in IGNORE_COLS or f"{e1['parent_type']}-{k1}" in IGNORE_COLS:
            n -= 1
        # ignore quantile intervals of numeric columns: parent_type-col-I
        elif f"{e1['parent_type']}-{k1[:-2]}" in NUMERIC_COLS and k1[-2:] == '-I':
            n -= 1
        elif e1[k1] == e2[k1]:
            I[k1] = e1[k1]
            i += 1
    return 1 - i / n, I


def subject_difference(s1, s2, join_rules):
    n = len(s1)
    i = float(0)
    I = dict() # intersection
    for k1 in s1:
        if k1 in PATIENT_ATTRS_EXCLUDED:
            n -= 1
            continue
        elif s1[k1] == s2[k1]:
            i += 1
            I[k1] = s1[k1]
    return 1 - i / n, I


def weight(s1, s2, e1, e2, join_rules, t_max):
    '''
    Returns the weight for an edge (an event connection)
    '''
    # time difference
    # TODO incorporate t_min
    w_t = (e2['t'] - e1['t']).total_seconds() / \
        t_max.total_seconds()
    w_e, I_e = event_difference(e1, e2, join_rules)
    w_s, I_s = subject_difference(s1, s2, join_rules)
    return w_t, w_e, w_s, I_e, I_s


def build_eventgraph(subjects, events, join_rules):
    """
    Builds event graphs
    An event:
    [{id: <id>, event_type: <event_type>, time: <time>, **<event_attributes>}]
    """
    events = copy.deepcopy(events)
    # number of events
    n = len(events)
    # adjacency matrix
    A = dok_matrix((n, n), dtype=float)
    c1 = 0
    for i, e1 in enumerate(events):
        j = i + 1
        for e2 in events[i + 1:]:
            if 'pi_stage' not in e1 or 'pi_stage' not in e2:
                print('e1', e1)
                print('e2', e2)
            if e1['type'] != e2['type']:
                break
            # Prevents over counting which happens if events with max PI stage connect with each other
            elif e1['pi_stage'] == e2['pi_stage'] and e1['pi_stage'] == join_rules['max_pi_stage']:
                break
            # get t_max by event type
            vals = [v for  k, v in join_rules['t_max'].items() if e1['type'] in k]
            if vals:
                t_max = vals[0]
            else:
                t_max = join_rules['t_max']['other']
            if e2['t'] > e1['t'] + t_max:
                break
            elif e2['t'] >= e1['t'] + join_rules['t_min'] \
                    and e1['id'] != e2['id']:
                s1 = subjects[e1['id']]
                s2 = subjects[e2['id']]
                w_t, w_e, w_s, I_e, I_s = weight(s1, s2, e1, e2, join_rules, t_max)
                if w_e <= join_rules['w_e_max']:
                    A[i, j] = w_t + w_e + w_s
                    c1 += 1
            j += 1
    c2 = 0
    if join_rules['join_by_subject'] and join_rules['sequential_join']:
        subject_events = dict()
        for e in events:
            if e['id'] not in subject_events:
                subject_events[e['id']] = []
            subject_events[e['id']].append(e)
        for s_id, s_events in subject_events.items():
            s_events_dict = dict()
            for e in s_events:
                if e['t'] not in s_events_dict:
                    s_events_dict[e['t']] = []
                s_events_dict[e['t']].append(e)
            times = sorted(s_events_dict.keys())
            for i, t in enumerate(times[:-1]):
                for e1 in s_events_dict[t]:
                    for e2 in s_events_dict[times[i + 1]]:
                        t_max = get_t_max(e1, e2, join_rules)
                        s1 = subjects[e1['id']]
                        s2 = subjects[e2['id']]
                        w_t, w_e, w_s, I_e, I_s = weight(s1, s2, e1, e2, join_rules, t_max)
                        A[e1['i'], e2['i']] = w_t + w_e + w_s
                        c2 += 1
            '''
            e1 = events[0]
            for e2 in events[1:]:
                if e2['id'] != e1['id']:
                    e1 = e2
                    continue
                w_t, w_e, w_s = weight_same_subject(e1, e2, join_rules)
                A[e1['i'], e2['i']] = w_t + w_e + w_s
                c2 += 1
                e1 = e2
            '''
        if join_rules['join_by_subject'] and not join_rules['sequential_join']:
            events.sort(key=lambda x: (x['id'], x['t']))
            for i, e1 in enumerate(events):
                for e2 in events[i + 1:]:
                    t_max = get_t_max(e1, e2, join_rules)
                    if e2['t'] < e1['t'] + join_rules['t_min']:
                        continue
                    elif e2['t'] > e1['t'] + t_max:
                        break
                    elif e2['id'] != e1['id']:
                        break
                    s1 = subjects[e1['id']]
                    s2 = subjects[e2['id']]
                    w_t, w_e, w_s, I_e, I_s = weight(s1, s2, e1, e2, join_rules, t_max)
                    A[e1['i'], e2['i']] = w_t + w_e + w_s
                    c2 += 1
        print("==========================================================")
        print("TEG construction")
        print("==========================================================")
        print("Number of events", len(events))
        print("Inter-patient event connections: ", c1)
        print("Same-patient event connections: ", c2)
        print("Total connections: ", c1 + c2)
        if c1 == 0:
            return A, False
    return A, True

def get_t_max(e1, e2, join_rules):
    if e1['type'] == e2['type']:
        vals = [v for  k, v in join_rules['t_max'].items() if e1['type'] in k]
        if vals:
            t_max = vals[0]
        else:
            t_max = join_rules['t_max']['other']
    else:
        t_max = join_rules['t_max']['diff_type_same_patient']
    return t_max

