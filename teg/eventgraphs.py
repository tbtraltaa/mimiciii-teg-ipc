import copy
import numpy as np
from datetime import timedelta
import itertools
from scipy.sparse import dok_matrix
import pprint
import networkx as nx

from teg.schemas import *


def event_difference(e1, e2, join_rules):
    n = len(e1) - len(join_rules['IDs'])
    i = float(0)
    for k1 in e1:
        if k1 in join_rules['IDs']:
            continue
        elif k1 == 'duration' and e2[k1] - e1[k1] <= join_rules['duration_similarity']:
            i += 1
        elif e1['type'] + '-' + k1 in NUMERIC_COLS and not join_rules['include_numeric']:
            n -= 1
        elif k1 in IGNORE_COLS:
            n -= 1
        elif e1[k1] == e2[k1]:
            i += 1
        '''
        elif e1['type'] + '-' + k1 in join_rules:
            if abs(e1[k1] - e2[k1]) <= join_rules[e1['type'] + '-' + k1]:
                i += 1
        '''
    return 1 - i / n


def subject_difference(s1, s2, join_rules):
    n = len(s1) - 1  # dob
    i = float(0)
    for k1 in s1:
        if k1 == 'dob':
            continue
        elif k1 == 'age' and abs(s1[k1] - s2[k1]) <= join_rules['age_similarity']:
            i += 1
        elif s1[k1] == s2[k1]:
            i += 1
    return 1 - i / n


def weight_same_subject(e1, e2, join_rules, t_max):
    '''
    Returns the weight for events of the same subject
    '''
    # time difference
    # TODO incorporate t_min
    w_t = (e2['t'] - e1['t']).total_seconds() / \
        t_max.total_seconds()
    if e1['type'] == e2['type']:
        w_e = event_difference(e1, e2, join_rules)
    else:
        w_e = join_rules['w_e_default']
    return w_t, w_e, 0


def weight(s1, s2, e1, e2, join_rules, t_max):
    w_t = (e2['t'] - e1['t']).total_seconds() / \
        t_max.total_seconds()
    w_e = event_difference(e1, e2, join_rules)
    w_s = subject_difference(s1, s2, join_rules)
    return w_t, w_e, w_s


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
            if e1['type'] != e2['type']:
                break
            # Prevents over counting which happens if events with max PI stage connect with each other
            elif e1['pi_stage'] == e2['pi_stage'] and e1['pi_stage'] == join_rules['max_pi_stage']:
                break
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
                w_t, w_e, w_s = weight(s1, s2, e1, e2, join_rules, t_max)
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
                        if e1['type'] == e2['type']:
                            vals = [v for  k, v in join_rules['t_max'].items() if e1['type'] in k]
                            if vals:
                                t_max = vals[0]
                            else:
                                t_max = join_rules['t_max']['other']
                        else:
                            t_max = join_rules['t_max']['diff_type_same_patient']
                        w_t, w_e, w_s = weight_same_subject(e1, e2, join_rules, t_max)
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
                    if e1['type'] == e2['type']:
                        vals = [v for  k, v in join_rules['t_max'].items() if e1['type'] in k]
                        if vals:
                            t_max = vals[0]
                        else:
                            t_max = join_rules['t_max']['other']
                    else:
                        t_max = join_rules['t_max']['diff_type_same_patient']
                    if e2['t'] < e1['t'] + join_rules['t_min']:
                        continue
                    elif e2['t'] > e1['t'] + t_max:
                        break
                    elif e2['id'] != e1['id']:
                        break
                    w_t, w_e, w_s = weight_same_subject(e1, e2, join_rules, t_max)
                    A[e1['i'], e2['i']] = w_t + w_e + w_s
                    c2 += 1
        print("Edges connecting events of different patients: ", c1)
        print("Edges connecting events of the same patients: ", c2)
    return A
