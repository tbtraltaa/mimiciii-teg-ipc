import sys
import pandas as pd
import numpy as np
import pprint
from datetime import timedelta
import copy
import warnings
warnings.filterwarnings('ignore')

from teg.schemas import *
from teg.schemas_PI import *
from teg.schemas_chart_events import *
from teg.eventgraphs import *
from teg.queries_mimic_extract import *
from teg.queries_chart_events import *
from teg.queries import *

def mimic_events(event_list, join_rules, conf):
    conn = get_db_connection()
    patients = get_patient_demography(conn, conf)
    all_events = list()
    n = 0
    for event_name in PI_EVENTS:
        if not conf['include_numeric'] and event_name in PI_EVENTS_NUMERIC:
            continue
        events = get_unique_chart_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    for event_name in CHART_EVENTS:
        if not conf['include_numeric'] and event_name in CHART_EVENTS_NUMERIC:
            continue
        events = get_unique_chart_events(conn, event_name, conf)
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
    for event_key in event_list:
        event_name, table, time_col, main_attr = EVENTS[event_key]
        events = get_events(conn, event_key, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    # Add stage to all events
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t']))
    min_stage = min(conf['PI_states'].keys())
    max_stage = max(conf['PI_states'].keys())
    exclude_indices = []
    stage = 0
    prev_stage = 0
    for i, e in enumerate(sorted_events):
        # PI stage
        if e['type'] == 'PI stage':
            stage = e['pi_stage']
        # PI related events before stage I is considered as stage 1
        include = True
        if stage < min_stage or stage > max_stage:
            include = False
            exclude_indices.append(e['i'])
        # exclude events after max stage
        if stage == max_stage and stage == prev_stage:
            include= False
            exclude_indices.append(e['i'])
        if include:
            if stage == 0 and e['type'] != 'PI stage' and 'PI' in e['type']:
                stage = 1
            all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
            all_events[e['i']]['pi_stage'] = stage
        prev_stage = stage
        if i + 1 < n and e['id'] != sorted_events[i + 1]['id']:
            stage = 0
            prev_stage = 0
    for i in sorted(exclude_indices, reverse=True):
        del all_events[i]
    # join repeated admissions to the first admission as a duplicate
    p_events = sorted(all_events, key=lambda x: (x['subject_id'], x['datetime']))
    e1 = None
    t_0 = None
    idd = None
    adm_num = 1
    sub_adm_events = []
    duplicate = False
    for e2 in p_events:
        if e1 is not None:
            if e1['subject_id'] != e2['subject_id']:
                t_0 = None
                idd = None
                adm_num = 1
                duplicate = False
        # first admission of a patient
        if e2['type'] == 'admissions' and t_0 is None and idd is None:
            t_0 = e2['datetime']
            idd = e2['id']
            e2['adm_num'] = adm_num
            e1 = e2
            duplicate = False
        # subsequent admission of a patient
        elif e2['type'] == 'admissions' and t_0 is not None and idd is not None:
            e_tmp = copy.deepcopy(e2)
            e_tmp['id'] = idd
            e_tmp['t'] = e_tmp['datetime'] - t_0
            adm_num += 1
            e_tmp['adm_num'] = adm_num
            sub_adm_events.append(e_tmp)
            e1 = e2
            duplicate = True
        # events after a subsequent admission
        elif duplicate and t_0 is not None and idd is not None:
            e_tmp = copy.deepcopy(e2)
            e_tmp['id'] = idd
            e_tmp['t'] = e_tmp['datetime'] - t_0
            e_tmp['adm_num'] = adm_num
            sub_adm_events.append(e_tmp)
            e1 = e2
        # events during the first admission
        else:
            e2['adm_num'] = adm_num

    print("Subsequent admission events: ", len(sub_adm_events))
    all_events += sub_adm_events
    all_events.sort(key=lambda x: (x['type'], x['t']))
    for i in range(len(all_events)):
        all_events[i]['i'] = i
    print("Total events: ", n)
    print("Total patients:", len(patients))
    return patients, all_events
