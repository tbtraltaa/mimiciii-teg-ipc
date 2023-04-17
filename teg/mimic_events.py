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
        events = get_chart_events(conn, event_name, conf)
        for i, e in enumerate(events):
            e['i'] = i + n
        n += len(events)
        print(event_name, len(events))
        all_events += events
    for event_name in CHART_EVENTS:
        if not conf['include_numeric'] and event_name in CHART_EVENTS_NUMERIC:
            continue
        events = get_chart_events(conn, event_name, conf)
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
        print(event_key, len(events))
        all_events += events
    # Add stage to all events
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t']))
    min_stage = min(conf['PI_states'].keys())
    max_stage = max(conf['PI_states'].keys())
    stage = 0
    PI = False
    non_PI_ids = []
    non_PI_events = []
    exclude_ids = []
    exclude_indices = []
    id_excluded = False
    for i, e in enumerate(sorted_events):
        # consider events till the first PI stage
        if not PI:
            # PI stage
            if 'PI Stage' in e['type']:
                stage = e['pi_stage']
            # exclude a patient who had higher or lower stage than our focus.
            if stage < min_stage or stage > max_stage:
                exclude_ids.append(e['id'])
                id_excluded = True
            elif stage == max_stage:
                all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
                all_events[e['i']]['pi_stage'] = stage
                PI = True
            # PI related events before stage I is considered as stage I
            elif min_stage <= stage and stage < max_stage:
                if stage == 0 and \
                    e['type'] != 'PI Stage' and \
                    'PI' in e['type'] and \
                    conf['PI_as_stage'] and \
                    max_stage == 1:
                    stage = 1
                    PI = True
                all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
                all_events[e['i']]['pi_stage'] = stage
        # later all events belonging to excluded ids
        # then no need to exclude those events here
        elif PI and not id_excluded:
            # exlude events after maximum PI stage event
            exclude_indices.append(e['i'])
        if i + 1 < n and e['id'] != sorted_events[i + 1]['id']:
            if not PI:
                non_PI_ids.append(e['id'])
            PI = False
            stage = 0
            id_excluded = False

    for e in all_events:
        if e['id'] in exclude_ids:
            exclude_indices.append(e['i'])

    if conf['PI_only']:
        for e in all_events:
            if e['id'] in non_PI_ids:
                non_PI_events.append(e['i'])
    for i in sorted(set(exclude_indices + non_PI_events), reverse=True):
        del all_events[i]

    if conf['subsequent_adm']:
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
            if 'Admissions' in e2['type'] and t_0 is None and idd is None:
                t_0 = e2['datetime']
                idd = e2['id']
                e2['adm_num'] = adm_num
                e1 = e2
                duplicate = False
            # subsequent admission of a patient
            elif 'Admissions' in e2['type'] and t_0 is not None and idd is not None:
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
    all_events = sorted(all_events, key=lambda x: (x['type'], x['t']))
    for i in range(len(all_events)):
        all_events[i]['i'] = i
    #print('Total patients', len(patients))
    #print("Total events: ", n)
    print('Total patients', len(set([e['subject_id'] for e in all_events])))
    print('Total admissions', len(set([e['id'] for e in all_events])))
    print("Total events: ", len(all_events))
    return patients, all_events
