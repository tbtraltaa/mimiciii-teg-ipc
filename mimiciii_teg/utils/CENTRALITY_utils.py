import numpy as np
import pandas as pd
from datetime import timedelta

from mimiciii_teg.utils.event_utils import group_events_by_patient


def process_CENTRALITY_values(events, CENTRALITY_values, conf):
    '''
    Returns dictionaries of all CENTRALITY values, non-zero CENTRALITY values,
    and CENTRALITY values above the percentile 
    '''
    CENTRALITY_nz = dict()
    CENTRALITY_all = dict()
    max_CENTRALITY = float(max(CENTRALITY_values))
    min_CENTRALITY = float(min(CENTRALITY_values))
    CENTRALITY_vals = []
    for i, val in enumerate(CENTRALITY_values):
        v = float(val) / max_CENTRALITY if conf['scale_CENTRALITY'] else float(val)
        CENTRALITY_all[i] = v
        if val > 0:
            CENTRALITY_nz[i] = v
    CENTRALITY_nz_vals = list(CENTRALITY_nz.values())
    P_min = np.percentile(CENTRALITY_nz_vals, conf['P'][0])
    P_max = np.percentile(CENTRALITY_nz_vals, conf['P'][1])
    print("Nonzero CENTRALITY", len(CENTRALITY_nz))
    print("Min, Max CENTRALITY", min_CENTRALITY, max_CENTRALITY)
    if conf['scale_CENTRALITY']:
        print("Min, Max CENTRALITY scaled", min(CENTRALITY_nz_vals), max(CENTRALITY_nz_vals))
    print("Percentile", P_min, P_max)
    CENTRALITY_P = dict([(i, v) for i, v in CENTRALITY_nz.items() if v >= P_min and v <= P_max])
    print("Nodes above percentile", len(CENTRALITY_P))
    if conf['P_remove']:
        P_min_remove = np.percentile(CENTRALITY_nz_vals, conf['P_remove'][0])
        P_max_remove = np.percentile(CENTRALITY_nz_vals, conf['P_remove'][1])
        print(f"Percentile {conf['P_remove']}" , P_min_remove, P_max_remove)
        CENTRALITY_remove = dict([(i, v) for i, v in CENTRALITY_nz.items() if v >= P_min_remove and v <= P_max_remove])
        print(f"Nodes below percentile {conf['P_remove']}", len(CENTRALITY_remove))
    results = dict()
    results['CENTRALITY_all'] = CENTRALITY_all
    results['CENTRALITY_nz'] = CENTRALITY_nz
    results['CENTRALITY_P'] = CENTRALITY_P
    results['P'] = [P_min, P_max]
    if conf['P_remove']:
        results['CENTRALITY_remove'] = CENTRALITY_remove
        results['P_remove'] = [P_min_remove, P_max_remove]
        results['CENTRALITY_remove_ET'], results['CENTRALITY_remove_freq'],  results['CENTRALITY_remove_avg'] = \
            get_event_type_CENTRALITY(events, CENTRALITY_remove)
    results['CENTRALITY_P_ET'], results['CENTRALITY_P_ET_freq'], results['CENTRALITY_P_ET_avg'] = \
        get_event_type_CENTRALITY(events, CENTRALITY_P)
    return results


def process_event_type_CENTRALITY(events, CENTRALITY_values, conf):
    '''
    Returns dictionaries of all CENTRALITY values, non-zero CENTRALITY values,
    and CENTRALITY values above the percentile 
    '''
    # ET stands for Event Type
    ET_CENTRALITY = {}
    ET_CENTRALITY_freq= {}
    max_CENTRALITY = float(max(CENTRALITY_values))
    for e in events:
        val = CENTRALITY_values[e['i']]
        if val == 0:
            continue
        v = float(val) / max_CENTRALITY if conf['scale_CENTRALITY'] else float(val)
        if e['type'] not in ET_CENTRALITY:
            # value and freq
            ET_CENTRALITY[e['type']] = v
            ET_CENTRALITY_freq[e['type']] = 1 
        else:
            ET_CENTRALITY[e['type']] += v
            ET_CENTRALITY_freq[e['type']] += 1
    vals = list(ET_CENTRALITY.values())
    P_min = np.percentile(vals, conf['P'][0])
    P_max = np.percentile(vals, conf['P'][1])
    print("None zero event type CENTRALITY", len(vals))
    print("Min, Max event type CENTRALITY ", min(vals), max(vals))
    print("Percentile", P_min, P_max)
    ET_CENTRALITY_P = dict([(et, v) for et, v in ET_CENTRALITY.items() \
            if v >= P_min and v <= P_max and ET_CENTRALITY_freq[et] >= conf['ET_CENTRALITY_min_freq']])
    ET_CENTRALITY_P_freq = dict([(et, ET_CENTRALITY_freq[et]) for et, v in ET_CENTRALITY_P.items()])
    print("Number of event types CENTRALITY above percentile", len(ET_CENTRALITY_P))
    ET_CENTRALITY_avg = dict([(t, v/ET_CENTRALITY_freq[t]) for t, v in ET_CENTRALITY.items() if ET_CENTRALITY_freq[t] >= conf['ET_CENTRALITY_min_freq']])
    vals_avg = list(ET_CENTRALITY_avg.values())
    P_min_avg = np.percentile(vals_avg, conf['P'][0])
    P_max_avg = np.percentile(vals_avg, conf['P'][1])
    ET_CENTRALITY_P_avg = dict([(t, v) for t, v in ET_CENTRALITY_avg.items() if v >= P_min_avg and v <= P_max_avg])
    print("Average Event CENTRALITY Percentile", P_min_avg, P_max_avg)
    print("Number of average Event CENTRALITY above Percentile", len(ET_CENTRALITY_P_avg))
    if conf['P_remove']:
        P_min_remove = np.percentile(vals, conf['P_remove'][0])
        P_max_remove = np.percentile(vals, conf['P_remove'][1])
        ET_CENTRALITY_remove = dict([(et, v) for et, v in ET_CENTRALITY.items() \
                if v >= P_min_remove and v <= P_max_remove])
        ET_CENTRALITY_freq_remove= dict([(et, ET_CENTRALITY_freq[et]) for et, v in ET_CENTRALITY_remove.items()])
        print(f"Number of event types CENTRALITY below percentile {conf['P_remove']}", len(ET_CENTRALITY_remove))
        ET_CENTRALITY_avg_remove = dict([(t, v/ET_CENTRALITY_freq[t]) for t, v in ET_CENTRALITY_remove.items() if ET_CENTRALITY_freq[t] >= conf['ET_CENTRALITY_min_freq']])
    results = dict()
    results['ET_CENTRALITY'] = ET_CENTRALITY
    results['ET_CENTRALITY_freq'] = ET_CENTRALITY_freq
    results['ET_CENTRALITY_P'] = ET_CENTRALITY_P
    results['ET_CENTRALITY_P_freq'] = ET_CENTRALITY_P_freq
    results['ET_CENTRALITY_avg'] = ET_CENTRALITY_avg
    results['ET_CENTRALITY_P_avg'] = ET_CENTRALITY_P_avg
    results['ET_P'] = [P_min, P_max]
    results['ET_P_avg'] = [P_min_avg, P_max_avg]
    if conf['P_remove']:
        results['ET_CENTRALITY_remove'] = ET_CENTRALITY_remove
        results['ET_CENTRALITY_freq_remove'] = ET_CENTRALITY_freq_remove
        results['ET_CENTRALITY_avg_remove'] = ET_CENTRALITY_avg_remove
    return results


def get_patient_CENTRALITY(events, CENTRALITY):
    '''
    Return CENTRALITY values with time points
    '''
    patient_events = group_events_by_patient(events)
    patient_CENTRALITY = {}
    for p_id in patient_events:
        patient_CENTRALITY[p_id] = {'t': [], 'CENTRALITY': []}
        for e in patient_events[p_id]:
            if CENTRALITY[e['i']] > 0:
                patient_CENTRALITY[p_id]['t'].append(e['t'])
                patient_CENTRALITY[p_id]['CENTRALITY'].append(CENTRALITY[e['i']])
    return patient_CENTRALITY

def get_patient_CENTRALITY_total(events, CENTRALITY, conf):
    '''
    Return CENTRALITY values with time points
    '''
    patient_CENTRALITY = {}
    for e in events:
        if CENTRALITY[e['i']] == 0:
            continue
        if e['id'] not in patient_CENTRALITY:
            patient_CENTRALITY[e['id']] = CENTRALITY[e['i']]
        else:
            patient_CENTRALITY[e['id']] += CENTRALITY[e['i']]
    vals = list(patient_CENTRALITY.values())
    P_min = np.percentile(vals, conf['P_patients'][0])
    P_max = np.percentile(vals, conf['P_patients'][1])
    patient_CENTRALITY_P = dict([(k, v) for k, v in patient_CENTRALITY.items() if v >= P_min and v <= P_max])
    print("Patient CENTRALITY Percentile", P_min, P_max)
    print("Number of patient CENTRALITY above percentile", len(patient_CENTRALITY_P))
    results = dict()
    results['patient_CENTRALITY_total'] = patient_CENTRALITY
    results['patient_CENTRALITY_P'] = patient_CENTRALITY_P
    results['PCENTRALITY_P'] = [P_min, P_max]
    return results

def get_patient_max_CENTRALITY(events, CENTRALITY, time_unit = timedelta(days=1, hours=0)):
    '''
    Return maximum CENTRALITY value per hour
    '''
    patient_events = group_events_by_patient(events)
    patient_CENTRALITY = {}
    for p_id in patient_events:
        h_prev = -1
        max_CENTRALITY = 0
        patient_CENTRALITY[p_id] = {'t': [], 'CENTRALITY': []}
        for e in patient_events[p_id]:
            # hour
            h = e['t'].total_seconds()//time_unit.total_seconds()
            if CENTRALITY[e['i']] > 0 and h > h_prev:
                patient_CENTRALITY[p_id]['t'].append(h)
                patient_CENTRALITY[p_id]['CENTRALITY'].append(CENTRALITY[e['i']])
                h_prev = h
                max_CENTRALITY = CENTRALITY[e['i']]
            elif CENTRALITY[e['i']] > 0 and h == h_prev and CENTRALITY[e['i']] > max_CENTRALITY:
                patient_CENTRALITY[p_id]['CENTRALITY'][-1] = CENTRALITY[e['i']]
                max_CENTRALITY = CENTRALITY[e['i']]
    return patient_CENTRALITY


def get_event_type_CENTRALITY(events, CENTRALITY):
    CENTRALITY_ET = {}
    CENTRALITY_ET_freq = {}
    for i in CENTRALITY:
        if CENTRALITY[i] == 0:
            continue
        if events[i]['type'] not in CENTRALITY_ET:
            CENTRALITY_ET[events[i]['type']] = CENTRALITY[i]
            CENTRALITY_ET_freq[events[i]['type']] = 1
        else:
            CENTRALITY_ET[events[i]['type']] += CENTRALITY[i]
            CENTRALITY_ET_freq[events[i]['type']] += 1
    CENTRALITY_ET_avg = dict([(t, v/CENTRALITY_ET_freq[t]) for t, v in CENTRALITY_ET.items()])
    return CENTRALITY_ET, CENTRALITY_ET_freq, CENTRALITY_ET_avg


def get_event_type_df(PI_types, NPI_types, I, i, PI_results, NPI_results, conf):
    if conf['iter_type'] == 'event_CENTRALITY':
        PI_ET = PI_results['CENTRALITY_P_ET']
        PI_ET_ALL = PI_results['ET_CENTRALITY']
        PI_ET_freq = PI_results['CENTRALITY_P_ET_freq']
        NPI_ET = NPI_results['CENTRALITY_P_ET']
        NPI_ET_ALL = NPI_results['ET_CENTRALITY']
        NPI_ET_freq = NPI_results['CENTRALITY_P_ET_freq']
        PI_ET_avg = PI_results['CENTRALITY_P_ET_avg'] 
        NPI_ET_avg = NPI_results['CENTRALITY_P_ET_avg'] 
    elif conf['iter_type'] == 'event_type_CENTRALITY':
        PI_ET = PI_results['ET_CENTRALITY_P']
        PI_ET_ALL = PI_results['ET_CENTRALITY']
        PI_ET_freq = PI_results['ET_CENTRALITY_freq']
        NPI_ET = NPI_results['ET_CENTRALITY_P']
        NPI_ET_ALL = NPI_results['ET_CENTRALITY']
        NPI_ET_freq = NPI_results['ET_CENTRALITY_freq']
        PI_ET_avg = PI_results['ET_CENTRALITY_avg'] 
        NPI_ET_avg = NPI_results['ET_CENTRALITY_avg'] 
    elif conf['iter_type'] == 'average_event_CENTRALITY':
        PI_ET = PI_results['ET_CENTRALITY']
        PI_ET_ALL = PI_results['ET_CENTRALITY']
        PI_ET_freq = PI_results['ET_CENTRALITY_freq']
        NPI_ET = NPI_results['ET_CENTRALITY']
        NPI_ET_ALL = NPI_results['ET_CENTRALITY']
        NPI_ET_freq = NPI_results['ET_CENTRALITY_freq']
        PI_ET_avg = PI_results['ET_CENTRALITY_avg'] 
        NPI_ET_avg = NPI_results['ET_CENTRALITY_avg'] 
    df = pd.DataFrame(columns=['Iteration',
                               'Event Type',
                               'Result',
                               'Event Type CENTRALITY (PI)',
                               'Event Type CENTRALITY (NPI)',
                               'Event Count (PI)',
                               'Event Count (NPI)',
                               'Avg Event CENTRALITY (PI)',
                               'Avg Event CENTRALITY (NPI)',
                               ])
    for t in PI_types:
        df = df.append({'Event Type': t,
                        'Result': 'PI',
                        'Event Type CENTRALITY (PI)': PI_ET[t],
                        'Event Type CENTRALITY (NPI)': NPI_ET_ALL[t] if t in NPI_ET_ALL else np.nan,
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t],
                        'Event Count (NPI)': NPI_ET_freq[t] if t in NPI_ET_freq else np.nan,
                        'Avg Event CENTRALITY (PI)': PI_ET_avg[t],
                        'Avg Event CENTRALITY (NPI)': NPI_ET_avg[t] if t in NPI_ET_avg else np.nan,
                        }, 
                       ignore_index=True)
    for t in NPI_types:
        df = df.append({'Event Type': t,
                        'Result': 'NPI',
                        'Event Type CENTRALITY (PI)': PI_ET_ALL[t] if t in PI_ET_ALL else np.nan,
                        'Event Type CENTRALITY (NPI)': NPI_ET[t],
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t] if t in PI_ET_freq else np.nan,
                        'Event Count (NPI)': NPI_ET_freq[t],
                        'Avg Event CENTRALITY (PI)': PI_ET_avg[t] if t in PI_ET_avg else np.nan,
                        'Avg Event CENTRALITY (NPI)': NPI_ET_avg[t],
                        }, 
                       ignore_index=True)
    for t in I:
        df = df.append({'Event Type': t,
                        'Result': 'PI and NPI',
                        'Event Type CENTRALITY (PI)': PI_ET[t],
                        'Event Type CENTRALITY (NPI)': NPI_ET[t],
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t],
                        'Event Count (NPI)': NPI_ET_freq[t],
                        'Avg Event CENTRALITY (PI)': PI_ET_avg[t],
                        'Avg Event CENTRALITY (NPI)': NPI_ET_avg[t],
                        }, ignore_index=True)
    if conf['iter_type'] == 'average_event_CENTRALITY':
        df = df.sort_values(by=['Iteration',
                                'Event Type',
                                'Result',
                                'Avg Event CENTRALITY (PI)',
                                'Avg Event CENTRALITY (NPI)',
                                'Event Type CENTRALITY (PI)',
                                'Event Type CENTRALITY (NPI)'],
                            ascending=[False, True, True, False, True, False, True],
                            ignore_index=True)
    else:
        df = df.sort_values(by=['Iteration',
                                'Event Type',
                                'Result',
                                'Event Type CENTRALITY (PI)',
                                'Event Type CENTRALITY (NPI)',
                                'Avg Event CENTRALITY (PI)',
                                'Avg Event CENTRALITY (NPI)'],
                            ascending=[False, True, True, False, True, False, True],
                            ignore_index=True)
    return df
