import copy
import pprint

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
        print(key, len(list(val)))
        events_grouped[key] = sorted(list(val), key=lambda x: (x['type'], x['t']))
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


