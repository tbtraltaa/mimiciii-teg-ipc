import sys
import pandas as pd
import numpy as np
import pprint
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from teg.eventgraphs import *
from teg.queries_mimic_extract import *
from teg.queries_PI import *
from teg.queries import *

def mimic_events(event_list, join_rules, conf):
    conn = get_db_connection()
    patients = get_patient_demography(conn, conf)
    all_events = list()
    n = 0
    for event_name in PI_EVENTS:
        events = get_unique_PI_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    events = get_events_interventions(conn, conf)
    for i, e in enumerate(events):
        e['i'] = i + n
    n += len(events)
    print('Interventions', len(events))
    all_events += events
    if conf['vitals_X_mean']:
        events = get_events_vitals_X_mean(conn, conf)
    else:
        events = get_events_vitals_X(conn, conf)
    for i, e in enumerate(events):
        e['i'] = i + n
    n += len(events)
    print('Vitals', len(events))
    all_events += events
    for event_name in event_list:
        events = get_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    print("Total events: ", n)
    print("Total patients:", len(patients))
    # Add stage to all events
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t']))
    stage = 0
    for i, e in enumerate(sorted_events):
        if e['type'] == 'PI stage' and e['pi_value'] != stage:
            stage = e['pi_value']
        else:
            all_events[e['i']]['pi_value'] = stage
            all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
        if i + 1 < n:
            if e['id'] != sorted_events[i + 1]['id']:
                stage = 0
    return patients, all_events
