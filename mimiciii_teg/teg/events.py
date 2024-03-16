import sys
import pandas as pd
import numpy as np
from itertools import groupby
import pprint
from datetime import timedelta
import copy
import warnings
warnings.filterwarnings('ignore')

from mimiciii_teg.schemas.schemas import *
from mimiciii_teg.schemas.schemas_PI import *
from mimiciii_teg.schemas.schemas_chart_events import *
from mimiciii_teg.teg.eventgraphs import *
from mimiciii_teg.queries.queries_mimic_extract import *
from mimiciii_teg.queries.queries_chart_events import *
from mimiciii_teg.queries.queries import *
from mimiciii_teg.utils.event_utils import remove_events_by_id, group_events_by_parent_type

def events(conn, event_list, conf, hadms=()):
    print("==========================================================")
    print('Events from MIMIC-III')
    print("==========================================================")
    all_events = list()
    for event_name in event_list:
        if event_name in EVENTS:
            event_key = event_name
            if 'CV' in event_key and conf['dbsource'] == 'metavision':
                continue
            elif 'MV' in event_key and conf['dbsource'] == 'carevue':
                continue
            event_name, table, time_col, main_attr = EVENTS[event_key]
            events = get_events(conn, event_key, conf, hadms)
            print(event_key, len(events))
            if len(events) > 0:
                all_events += events
        elif event_name in PI_EVENTS:
            if not conf['include_numeric'] and event_name in PI_EVENTS_NUMERIC:
                continue
            events = get_chart_events(conn, event_name, conf, hadms)
            print(event_name, len(events))
            if len(events) > 0:
                all_events += events
        elif event_name in CHART_EVENTS:
            if not conf['include_numeric'] and event_name in CHART_EVENTS_NUMERIC:
                continue
            events = get_chart_events(conn, event_name, conf, hadms)
            print(event_name, len(events))
            if len(events) > 0:
                all_events += events
        elif event_name == 'Intervention': 
            events = get_events_interventions(conn, conf, hadms)
            print('Interventions', len(events))
            if len(events) > 0:
                all_events += events
        elif event_name == 'Vitals/Labs':
            if conf['vitals_X_mean']:
                events = get_events_vitals_X_mean(conn, conf, hadms)
            else:
                events = get_events_vitals_X(conn, conf, hadms)
            print('Vitals', len(events))
            if len(events) > 0:
                all_events += events
    '''
    remove_ids = []
    # remove patients with events no more than 2
    for key, val in groupby(all_events, key=lambda x: x['id']):
        if len(list(val)) <= 2:
            remove_ids.append(key)
    all_events = remove_events_by_id(all_events, remove_ids)
    '''
    for i, e in enumerate(all_events):
        e['i'] = i 
        e['j'] = i
    return all_events

def process_events_PI(all_events, conf):
    # Add stage to all events
    n = len(all_events)
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t'], -x['pi_stage']))
    min_stage = min(conf['PI_states'].keys())
    max_stage = max(conf['PI_states'].keys())
    # PI stage
    stage = min_stage
    # percolation state
    if min_stage in conf['PI_states']:
        min_state = conf['PI_states'][min_stage]
    else:
        print("Set the percolation states correctly.")
    state = min_state
    is_PI = False
    is_id_excluded = False
    non_PI_ids = []
    excluded_ids = []
    excluded_indices = []
    two_event_PI_ids = []
    subjects = []
    hadm_stage_t = dict()
    # exclude patients who had admission and PI Stage directly
    patient_events = 0
    for i, e in enumerate(sorted_events):
        patient_events += 1 
        # take only first admission
        # exclude subsequent admissions
        if 'Admissions' in e['type'] and e['subject_id'] in subjects and conf['first_hadm']:
            excluded_ids.append(e['id'])
            is_id_excluded = True
        # consider events till the first is_PI stage
        if not is_PI and not is_id_excluded:
            # PI stage
            if 'PI Stage' in e['type']:
                stage = e['pi_stage']
                if stage in conf['PI_states']:
                    state = conf['PI_states'][stage]
            # exclude a patient who had higher or lower stage than our focus.
            if stage < min_stage or stage > max_stage:
                excluded_ids.append(e['id'])
                is_id_excluded = True
                PI = False
            # maximum PI stage
            elif stage == max_stage:
                all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
                all_events[e['i']]['pi_stage'] = stage
                is_PI = True
                hadm_stage_t[e['hadm_id']] = e['t']
            # PI-related events before Stage I is considered as Stage I
            # if conf['PI_as_stage']
            elif min_stage == stage and \
                stage == 0 and \
                'PI Stage' not in e['type'] and \
                'PI' in e['type'] and \
                conf['PI_as_stage'] and \
                max_stage == 1:
                    stage = 1
                    is_PI = True
                    hadm_stage_t[e['hadm_id']] = e['t']
                    state = conf['PI_states'][stage]
                    all_events[e['i']]['pi_state'] = state
                    all_events[e['i']]['pi_stage'] = stage
            # if it is minimum stage
            elif stage == min_stage:
                all_events[e['i']]['pi_state'] = state
                all_events[e['i']]['pi_stage'] = stage
            # if it is a mid stage
            elif min_stage < stage and stage < max_stage and conf['PI_exclude_mid_stages']:
                    excluded_indices.append(e['i'])
                    stage = min_stage
                    patient_events -= 1
            elif min_stage < stage and stage < max_stage and not conf['PI_exclude_mid_stages']:
                all_events[e['i']]['pi_state'] = state
                all_events[e['i']]['pi_stage'] = stage
        # later all events belonging to excluded ids
        # then no need to exclude those events here
        elif is_PI and not is_id_excluded:
            patient_events -= 1
            # exlude events after maximum PI stage event
            excluded_indices.append(e['i'])
        # last event of a patient
        if i + 1 < n and e['id'] != sorted_events[i + 1]['id']:
            if not is_PI and not is_id_excluded:
                non_PI_ids.append(e['id'])
                print('Non PI', e['id'])
            elif is_PI and not is_id_excluded and patient_events == 2: # Admissions and Stage
                two_event_PI_ids.append(e['id'])
                del hadm_stage_t[e['hadm_id']]
            stage = min_stage
            state = min_state 
            is_PI = False
            is_id_excluded = False
            patient_events = 0
        # last event
        if i + 1 == n and not is_PI and not is_id_excluded:
            non_PI_ids.append(e['id'])
            print('Non PI', e['id'])
        elif i + 1 == n and is_PI and not is_id_excluded and patient_events == 2: # Admissions and Stage
            two_event_PI_ids.append(e['id'])
            del hadm_stage_t[e['hadm_id']]
    print("Excluded admissions", len(excluded_ids))
    print("Non PI admissions", len(non_PI_ids))
    print("Two event PI admissions (Admissions and PI Stage)", len(two_event_PI_ids))
    for e in all_events:
        if e['id'] in excluded_ids:
            excluded_indices.append(e['i'])
        elif conf['PI_only'] and e['id'] in non_PI_ids:
            excluded_indices.append(e['i'])
        elif e['id'] in two_event_PI_ids:
            excluded_indices.append(e['i'])
    for i in sorted(set(excluded_indices), reverse=True):
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
    print("==========================================================")
    print("Events in TEG for PI:")
    print("==========================================================")
    grouped_events = group_events_by_parent_type(all_events)
    for key, val in grouped_events.items():
        print(key, len(val))
    subject_ids = set([e['subject_id'] for e in all_events])
    print('Total patients', len(subject_ids))
    print('Total admissions', len(set([e['id'] for e in all_events])))
    print("Total events: ", len(all_events))
    for i in range(len(all_events)):
        all_events[i]['i'] = i
        all_events[i]['j'] = i
    return all_events, hadm_stage_t

def process_events_NPI(all_events, NPI_t, conf):
    # Add stage to all events
    n = len(all_events)
    sorted_events = sorted(all_events, key=lambda x: (x['id'], x['t']))
    min_stage = min(conf['PI_states'].keys())
    max_stage = max(conf['PI_states'].keys())
    # PI stage
    stage = min_stage
    # percolation state
    if min_stage in conf['PI_states']:
        min_state = conf['PI_states'][min_stage]
    else:
        print("Set the percolation states correctly.")
    state = min_state
    is_marker = False
    excluded_ids = []
    excluded_indices = []
    is_id_excluded = False
    t_marker = NPI_t[sorted_events[0]['hadm_id']]
    for i, e in enumerate(sorted_events):
        # No PI Stage event for NPI patient events
        if 'PI Stage' in e['type']:
            pprint.pprint(e)
        # consider events till the first is_PI stage
        if not is_marker:
            # is_PI stage
            if e['t'] > t_marker:
                stage = max_stage
                all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
                all_events[e['i']]['pi_stage'] = stage
                all_events[e['i']]['type'] = 'Marker'
                all_events[e['i']]['event_type'] = 'Marker'
                all_events[e['i']]['parent_type'] = 'Marker'
                all_events[e['i']]['t'] = t_marker
                is_marker = True
            else:
                all_events[e['i']]['pi_state'] = conf['PI_states'][stage]
                all_events[e['i']]['pi_stage'] = stage
        # later all events belonging to excluded ids
        # then no need to exclude those events here
        else:
            # exlude events after Marker event
            excluded_indices.append(e['i'])
        # last event of a patient
        if i + 1 < n and e['id'] != sorted_events[i + 1]['id']:
            if not is_marker:
                all_events[e['i']]['pi_state'] = conf['PI_states'][max_stage]
                all_events[e['i']]['pi_stage'] = max_stage
                all_events[e['i']]['type'] = 'Marker'
                all_events[e['i']]['event_type'] = 'Marker'
                all_events[e['i']]['parent_type'] = 'Marker'
                all_events[e['i']]['t'] = t_marker
            is_marker = False
            stage = min_stage
            t_marker = NPI_t[sorted_events[i + 1]['hadm_id']]
        # last event
        if i + 1 == n and not is_marker:
            all_events[e['i']]['pi_state'] = conf['PI_states'][max_stage]
            all_events[e['i']]['pi_stage'] = max_stage
            all_events[e['i']]['type'] = 'Marker'
            all_events[e['i']]['event_type'] = 'Marker'
            all_events[e['i']]['parent_type'] = 'Marker'
            all_events[e['i']]['t'] = t_marker

    for i in sorted(set(excluded_indices), reverse=True):
        del all_events[i]
    all_events = sorted(all_events, key=lambda x: (x['type'], x['t']))
    for i in range(len(all_events)):
        all_events[i]['i'] = i
    print("==========================================================")
    print("Events in TEG for NPI:")
    print("==========================================================")
    grouped_events = group_events_by_parent_type(all_events)
    for key, val in grouped_events.items():
        print(key, len(list(val)))
    subject_ids = set([e['subject_id'] for e in all_events])
    print('Total patients', len(subject_ids))
    print('Total admissions', len(set([e['id'] for e in all_events])))
    print("Total events: ", len(all_events))
    return all_events
