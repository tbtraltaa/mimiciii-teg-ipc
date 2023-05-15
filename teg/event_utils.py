from itertools import groupby
import copy
import pprint

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

def group_events_by_parent_type(events):
    events_grouped = dict()
    for key, val in groupby(events, key=lambda x: x['parent_type']):
        e_list = list(val)
        events_grouped[key] = sorted(e_list, key=lambda x: (x['type'], x['t']))
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


