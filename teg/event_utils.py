from itertools import groupby
import copy
import pprint
from datetime import timedelta

def get_top_events(events, PC_P, conf, I = []):
    max_n = conf['PC_percentile_max_n']
    if not max_n:
        max_n = len(PC_P)
    PC_P = sorted(PC_P.items(), key=lambda x: x[1], reverse = True)
    top_events = []
    count = 0
    for i, val in PC_P:
        if events[i]['type'] not in I:
            top_events.append(events[i])
            count += 1
        if count == max_n:
            break
    return top_events

def intersection_and_differences(A, B):
    I = A & B
    A_minus_B  = list(A - I)
    B_minus_A = list(B - I)
    I = list(I)
    #print("A - B: ")
    #print(A_minus_B)
    #print("B - A")
    #print(B_minus_A)
    print("I")
    pprint.pprint(I)
    return A_minus_B, B_minus_A, I

def get_event_types(events, PC):
    etypes = set()
    for i, val in PC.items():
            etypes.add(events[i]['type'])
    return etypes

def group_events_by_parent_type1(events):
    events_grouped = dict()
    for key, val in groupby(events, key=lambda x: x['parent_type']):
        e_list = list(val)
        events_grouped[key] = sorted(e_list, key=lambda x: (x['type'], x['t']))
    return events_grouped

def group_events_by_parent_type(events):
    events_grouped = dict()
    events = sorted(events, key=lambda x: (x['type'], x['t']))
    for e in events:
        if e['parent_type'] not in events_grouped:
            events_grouped[e['parent_type']] = [e]
        else:
            events_grouped[e['parent_type']].append(e)
    return events_grouped

def group_events_by_patient(events):
    events_grouped = dict()
    events = sorted(events, key=lambda x: x['t'])
    for e in events:
        if e['id'] not in events_grouped:
            events_grouped[e['id']] = [e]
        else:
            events_grouped[e['id']].append(e)
    return events_grouped

def sort_and_index_events(events):
    sorted_events = sorted(events, key=lambda x: (x['type'], x['t']))
    #index events
    for i in range(len(sorted_events)):
        sorted_events[i]['i'] = i
    return sorted_events

def remove_event_type(events, types):
    '''
    Events are assumed to be ordered by type and time
    '''
    events_copy = copy.copy(events)
    indices = list()
    # iterate from the last
    for i in range(len(events)-1, -1, -1):
        if events[i]['type'] in types:
            indices.append(i)
    for i in indices:
        del events_copy[i]
    # reindex events
    for i in range(len(events_copy)):
        events_copy[i]['i'] = i
    return events_copy

def remove_events_after_t(events, t):
    remove_indices = []
    for i, e in enumerate(events):
        if e['t'] > t[e['hadm_id']]:
            remove_indices.append(i)
    for i in range(len(remove_indices)-1, -1, -1):
        del events[i]
    return events


def get_patient_Braden_Scores(braden_events):
    '''
    Returns a dictionary of patient Braden Scores
    with time points
    '''
    patient_events = group_events_by_patient(braden_events)
    patient_BS = dict()
    for p_id in patient_events:
        patient_BS[p_id] = {'t': [], 'BS': []}
        for e in patient_events[p_id]:
            patient_BS[p_id]['t'].append(e['t'])
            patient_BS[p_id]['BS'].append(int(e['value']))
    return patient_BS


def get_patient_max_Braden_Scores(braden_events, time_unit = timedelta(days=1, hours=0)):
    '''
    Return maximum Braden Score per hour for patients
    '''
    patient_events = group_events_by_patient(braden_events)
    patient_BS = dict()
    for p_id in patient_events:
        h_prev = -1
        max_PC = 0
        patient_BS[p_id] = {'t': [], 'BS': []}
        for e in patient_events[p_id]:
            # hour
            h = e['t'].total_seconds() // time_unit.total_seconds()
            val = int(e['value'])
            if val > 0 and h > h_prev:
                patient_BS[p_id]['t'].append(h)
                patient_BS[p_id]['BS'].append(val)
                h_prev = h
                max_PC = val
            elif val > 0  and h == h_prev and val > max_PC:
                patient_BS[p_id]['BS'][-1] = val
                max_PC = val
    return patient_BS
