import numpy as np
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
    print("Min, Max PC scaled", min(PC_nz_vals), max(PC_nz_vals))
    print("Percentile", P_min, P_max)
    PC_P = dict([(i, v) for i, v in PC_nz.items() if v >= P_min and v <= P_max])
    print("Nodes above percentile", len(PC_P))
    return PC_all, PC_nz, PC_P

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
