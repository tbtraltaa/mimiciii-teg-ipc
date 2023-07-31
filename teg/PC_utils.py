import numpy as np
import pandas as pd
from datetime import timedelta

from teg.event_utils import group_events_by_patient


def process_PC_values(PC_values, conf):
    '''
    Returns dictionaries of all PC values, non-zero PC values,
    and PC values above the percentile 
    '''
    PC_nz = dict()
    PC_all = dict()
    max_PC = float(max(PC_values))
    min_PC = float(min(PC_values))
    PC_vals = []
    for i, val in enumerate(PC_values):
        v = float(val) / max_PC if conf['scale_PC'] else float(val)
        PC_all[i] = v
        if val > 0:
            PC_nz[i] = v
    PC_nz_vals = list(PC_nz.values())
    P_min = np.percentile(PC_nz_vals, conf['PC_percentile'][0])
    P_max = np.percentile(PC_nz_vals, conf['PC_percentile'][1])
    print("Nonzero PC", len(PC_nz))
    print("Min, Max PC", min_PC, max_PC)
    if conf['scale_PC']:
        print("Min, Max PC scaled", min(PC_nz_vals), max(PC_nz_vals))
    print("Percentile", P_min, P_max)
    PC_P = dict([(i, v) for i, v in PC_nz.items() if v >= P_min and v <= P_max])
    print("Nodes above percentile", len(PC_P))
    return PC_all, PC_nz, PC_P


def process_event_type_PC(events, PC_values, conf):
    '''
    Returns dictionaries of all PC values, non-zero PC values,
    and PC values above the percentile 
    '''
    # ET stands for Event Type
    ET_PC = {}
    ET_PC_freq= {}
    max_PC = float(max(PC_values))
    for e in events:
        val = PC_values[e['i']]
        if val == 0:
            continue
        v = float(val) / max_PC if conf['scale_PC'] else float(val)
        if e['type'] not in ET_PC:
            # value and freq
            ET_PC[e['type']] = v
            ET_PC_freq[e['type']] = 1 
        else:
            ET_PC[e['type']] += v
            ET_PC_freq[e['type']] += 1
    vals = list(ET_PC.values())
    P_min = np.percentile(vals, conf['PC_percentile'][0])
    P_max = np.percentile(vals, conf['PC_percentile'][1])
    print("None zero event type PC", len(vals))
    print("Min, Max event type PC ", min(vals), max(vals))
    print("Percentile", P_min, P_max)
    ET_PC_P = dict([(et, v) for et, v in ET_PC.items() \
            if v >= P_min and v <= P_max and ET_PC_freq[et] >= conf['ET_PC_min_freq']])
    ET_PC_P_freq = dict([(et, ET_PC_freq[et]) for et, v in ET_PC_P.items()])
    print("Event types PC above percentile", len(ET_PC_P))
    ET_PC_avg = dict([(t, v/ET_PC_freq[t]) for t, v in ET_PC.items() if ET_PC_freq[t] >= conf['ET_PC_min_freq']])
    vals = list(ET_PC_avg.values())
    P_min = np.percentile(vals, conf['PC_percentile'][0])
    P_max = np.percentile(vals, conf['PC_percentile'][1])
    ET_PC_P_avg = dict([(t, v) for t, v in ET_PC_avg.items() if v >= P_min and v <= P_max])
    print("Average Event PC above percentile", len(ET_PC_P_avg))
    return ET_PC, ET_PC_freq, ET_PC_P, ET_PC_P_freq, ET_PC_avg, ET_PC_P_avg


def get_patient_PC(events, PC):
    '''
    Return PC values with time points
    '''
    patient_events = group_events_by_patient(events)
    patient_PC = {}
    for p_id in patient_events:
        patient_PC[p_id] = {'t': [], 'PC': []}
        for e in patient_events[p_id]:
            if PC[e['i']] > 0:
                patient_PC[p_id]['t'].append(e['t'])
                patient_PC[p_id]['PC'].append(PC[e['i']])
    return patient_PC

def get_patient_PC_total(events, PC):
    '''
    Return PC values with time points
    '''
    patient_PC = {}
    for e in events:
        if e['id'] not in patient_PC:
            patient_PC[e['id']] = PC[e['i']]
        else:
            patient_PC[e['id']] += PC[e['i']]
    return patient_PC

def get_patient_max_PC(events, PC, time_unit = timedelta(days=1, hours=0)):
    '''
    Return maximum PC value per hour
    '''
    patient_events = group_events_by_patient(events)
    patient_PC = {}
    for p_id in patient_events:
        h_prev = -1
        max_PC = 0
        patient_PC[p_id] = {'t': [], 'PC': []}
        for e in patient_events[p_id]:
            # hour
            h = e['t'].total_seconds()//time_unit.total_seconds()
            if PC[e['i']] > 0 and h > h_prev:
                patient_PC[p_id]['t'].append(h)
                patient_PC[p_id]['PC'].append(PC[e['i']])
                h_prev = h
                max_PC = PC[e['i']]
            elif PC[e['i']] > 0 and h == h_prev and PC[e['i']] > max_PC:
                patient_PC[p_id]['PC'][-1] = PC[e['i']]
                max_PC = PC[e['i']]
    return patient_PC


def get_event_type_PC(events, PC):
    ET_PC = {}
    ET_PC_freq = {}
    for i in PC:
        if PC[i] == 0:
            continue
        if events[i]['type'] not in ET_PC:
            ET_PC[events[i]['type']] = PC[i]
            ET_PC_freq[events[i]['type']] = 1
        else:
            ET_PC[events[i]['type']] += PC[i]
            ET_PC_freq[events[i]['type']] += 1
    ET_PC_AVG = dict([(t, v/ET_PC_freq[t]) for t, v in ET_PC.items()])
    return ET_PC, ET_PC_freq, ET_PC_AVG


def get_event_type_df(PI_types, NPI_types, I, i, PI_results, NPI_results, conf):
    if conf['iter_type'] == 'event_PC':
        PI_ET = PI_results['PC_P_ET']
        PI_ET_freq = PI_results['PC_P_ET_freq']
        NPI_ET = NPI_results['PC_P_ET']
        NPI_ET_freq = NPI_results['PC_P_ET_freq']
        PI_ET_avg = PI_results['PC_P_ET_avg'] 
        NPI_ET_avg = NPI_results['PC_P_ET_avg'] 
    elif conf['iter_type'] == 'event_type_PC':
        PI_ET = PI_results['ET_PC_P']
        PI_ET_freq = PI_results['ET_PC_P_freq']
        NPI_ET = NPI_results['ET_PC_P']
        NPI_ET_freq = NPI_results['ET_PC_P_freq']
        PI_ET_avg = PI_results['ET_PC_avg'] 
        NPI_ET_avg = NPI_results['ET_PC_avg'] 
    elif conf['iter_type'] == 'average_event_PC':
        PI_ET = PI_results['ET_PC']
        PI_ET_freq = PI_results['ET_PC_freq']
        NPI_ET = NPI_results['ET_PC']
        NPI_ET_freq = NPI_results['ET_PC_freq']
        PI_ET_avg = PI_results['ET_PC_avg'] 
        NPI_ET_avg = NPI_results['ET_PC_avg'] 
    df = pd.DataFrame(columns=['Iteration',
                               'Event Type',
                               'Result',
                               'Event Type PC (PI)',
                               'Event Type PC (NPI)',
                               'Event Count (PI)',
                               'Event Count (NPI)',
                               'Avg Event PC (PI)',
                               'Avg Event PC (NPI)',
                               ])
    for t in PI_types:
        df = df.append({'Event Type': t,
                        'Result': 'PI',
                        'Event Type PC (PI)': PI_ET[t],
                        'Event Type PC (NPI)': NPI_ET[t] if t in NPI_ET else np.nan,
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t],
                        'Event Count (NPI)': NPI_ET_freq[t] if t in NPI_ET_freq else np.nan,
                        'Avg Event PC (PI)': PI_ET_avg[t],
                        'Avg Event PC (NPI)': NPI_ET_avg[t] if t in NPI_ET_avg else np.nan,
                        }, 
                       ignore_index=True)
    for t in NPI_types:
        df = df.append({'Event Type': t,
                        'Result': 'NPI',
                        'Event Type PC (PI)': PI_ET[t] if t in PI_ET else np.nan,
                        'Event Type PC (NPI)': NPI_ET[t],
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t] if t in PI_ET_freq else np.nan,
                        'Event Count (NPI)': NPI_ET_freq[t],
                        'Avg Event PC (PI)': PI_ET_avg[t] if t in PI_ET_avg else np.nan,
                        'Avg Event PC (NPI)': NPI_ET_avg[t],
                        }, 
                       ignore_index=True)
    for t in I:
        df = df.append({'Event Type': t,
                        'Result': 'PI and NPI',
                        'Event Type PC (PI)': PI_ET[t],
                        'Event Type PC (NPI)': NPI_ET[t],
                        'Iteration': i,
                        'Event Count (PI)': PI_ET_freq[t],
                        'Event Count (NPI)': NPI_ET_freq[t],
                        'Avg Event PC (PI)': PI_ET_avg[t],
                        'Avg Event PC (NPI)': NPI_ET_avg[t],
                        }, ignore_index=True)
    if conf['iter_type'] == 'average_event_PC':
        df = df.sort_values(by=['Iteration',
                                'Event Type',
                                'Result',
                                'Avg Event PC (PI)',
                                'Avg Event PC (NPI)',
                                'Event Type PC (PI)',
                                'Event Type PC (NPI)'],
                            ascending=[False, True, True, False, True, False, True],
                            ignore_index=True)
    else:
        df = df.sort_values(by=['Iteration',
                                'Event Type',
                                'Result',
                                'Event Type PC (PI)',
                                'Event Type PC (NPI)',
                                'Avg Event PC (PI)',
                                'Avg Event PC (NPI)'],
                            ascending=[False, True, True, False, True, False, True],
                            ignore_index=True)
    return df
