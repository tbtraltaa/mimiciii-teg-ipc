import numpy as np
import pprint

from mimiciii_teg.utils.event_utils import group_events_by_patient 

def filter_V_paths_by_ET(events, ET_CENTRALITY, v_paths):
    '''
    Filter paths by event types
    '''
    v_paths_ET = dict()
    for v in v_paths:
        if events[v]['type'] not in ET_CENTRALITY:
            continue
        v_paths_ET[v] = []
        for path in v_paths[v]:
            count = 0
            p = []
            for i in path:
                if 'Admission' in events[i]['type'] or \
                    'PI Stage' in events[i]['type'] or \
                    'Marker' in events[i]['type']:
                    p.append(i)
                elif events[i]['type'] in ET_CENTRALITY:
                    count += 1
                    p.append(i)
            if count > 0:
                v_paths_ET[v].append(p)
    return v_paths_ET


def filter_paths_by_ET(events, ET_CENTRALITY, paths):
    '''
    Filter paths by event types
    '''
    paths_ET = []
    for path in paths:
        p = []
        count = 0
        for v in path:
            if 'Admission' in events[v]['type'] or \
                'PI Stage' in events[v]['type'] or \
                'Marker' in events[v]['type']:
                p.append(v)
            elif events[v]['type'] in ET_CENTRALITY:
                count += 1
                p.append(v)
        if count > 0:
            paths_ET.append(p)
    return paths_ET


def get_patient_SCP(events, CENTRALITY_all, paths, n):
    '''
    Returns n most influential Shortest Central Paths (SCP) of patients
    '''
    events_grouped = group_events_by_patient(events)
    patient_paths = dict()
    k = 0
    for p_id in events_grouped:
        s = events_grouped[p_id][0]['i']
        t = events_grouped[p_id][-1]['i']
        if (s, t) not in paths:
            print(f'{s, t} path missing')
            k += 1
        elif (s, t) in paths:
            patient_paths[p_id] = []
            for path in paths[(s, t)]:
                val = get_path_centrality(events, CENTRALITY_all, path)
                if val:
                    patient_paths[p_id].append({'CENTRALITY': val, 'path': path})
    print(f'{k} (s, t) SCP path missing')
    path_list = []
    for p_id in patient_paths:
        patient_paths[p_id] = sorted(patient_paths[p_id], key = lambda x: x['CENTRALITY'], reverse=True)[:n]
        path_list += [p['path'] for p in patient_paths[p_id]]
    return patient_paths, path_list
    

def get_patient_SP(A, events, CENTRALITY_all, paths, n):
    '''
    Returns n Shortest Centrality Paths (SCP) of patients
    '''
    events_grouped = group_events_by_patient(events)
    patient_paths = dict()
    k = 0
    for p_id in events_grouped:
        s = events_grouped[p_id][0]['i']
        t = events_grouped[p_id][-1]['i']
        if (s, t) not in paths:
            print(f'{s, t} path missing')
            k += 1
        elif (s, t) in paths:
            patient_paths[p_id] = []
            for path in paths[(s, t)]:
                val = get_path_weight(A, events, path)
                if val:
                    patient_paths[p_id].append({'w': val, 'path': path})
    print(f'{k} (s, t) SP path missing')
    path_list = []
    for p_id in patient_paths:
        patient_paths[p_id] = sorted(patient_paths[p_id], key = lambda x: x['w'])[: n]
        path_list += [p['path'] for p in patient_paths[p_id]]
    return patient_paths, paths

def get_path_centrality(events, CENTRALITY, path):
    val = 1
    count = 0
    for i, v in enumerate(path):
        if 'Admission' in events[v]['type'] or \
            'PI Stage' in events[v]['type'] or \
            'Marker' in events[v]['type']:
            continue
        elif i == 0 and CENTRALITY[v] == 0:
            continue
        elif i == len(path) - 1 and CENTRALITY[v] == 0:
            continue
        elif CENTRALITY[v] == 0:
            val = 0
            break
        else:
            val += np.log10(CENTRALITY[v])
            count += 1
    if count > 0 and val != 0:
        return val
    return None

def get_path_weight(A, events, path):
    path_weight = 0
    count = 0
    i = path[0]
    if 'Admission' not in events[i]['type'] and \
        'PI Stage' not in events[i]['type'] and \
        'Marker' not in events[i]['type']:
        count += 1
    for j in path[1:]:
        if 'Admission' not in events[j]['type'] and \
            'PI Stage' not in events[j]['type'] and \
            'Marker' not in events[j]['type']:
            count += 1
        path_weight += A[i, j]
        i = j
    if count > 0:
        return path_weight
    return None

def get_paths_centrality(events, conf, CENTRALITY, paths):
    '''
    Returns the Shortest Centrality Paths (SCPs) and their path centrality.
    '''
    C = []
    SCP = []
    for (s, t) in paths:
        for path in paths[(s, t)]:
            val = get_path_centrality(events, CENTRALITY, path)
            if val:
                C.append(val)
                SCP.append(path)
    P_min = np.percentile(C, conf['path_percentile'][0])
    P_max = np.percentile(C, conf['path_percentile'][1])
    return SCP, C, [P_min, P_max]


def get_influential_SCP(events, conf, CENTRALITY, paths,  CENTRALITY_P = None):
    '''
    Returns the most influential Shortest Centrality Paths (SCPs).
    If the most influential vertices are given, returns the most influential SCPs 
    containing the most influential events.
    '''
    SCP, C, P = get_paths_centrality(events, conf, CENTRALITY, paths)
    SCP_P = [SCP[i] for i, val in enumerate(C) if P[0] <= val <= P[1]] 

    if CENTRALITY_P:
        del_paths = []
        for i, path in enumerate(SCP_P):
            contains_v = False
            for v in  CENTRALITY_P:
                if v in path:
                    contains_v = True
                    break
            if not contains_v:
                del_paths.append(i)
        for i in sorted(del_paths, reverse=True):
            del SCP_P[i]
        print(f'Deleted {len(del_paths)} SCP paths which do not contain any of the most influential vertices.')
    return SCP_P


def get_total_path_CENTRALITY(CENTRALITY, CENTRALITY_P, paths, P=[0, 100]):
    '''
    Returns total CENTRALITY values of paths for each vertex
    '''
    total_path_CENTRALITY = {}
    for v, v_paths in paths.items():
        total_path_CENTRALITY[v] = 0
        for path in v_paths:
            C = 0
            for p in path:
                C += np.log10(CENTRALITY[p])
            total_path_CENTRALITY[v] += C
    return total_path_CENTRALITY

def analyze_paths(events, CENTRALITY, V, paths):
    for i in V:
        for path in paths[i]:
            u = path[0]
            for v in path[1:]:
                if events[u]['type'] == events[v]['type']:
                    pass
                else:
                    pass

def CENTRALITY_paths(D, pred, states):
    n = D.nrows
    S = 0.0
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        S_exclude_v = S - S_sv[v] - S_vt[v]
        v_paths[v] = []
        for s in D.extract_col(v).indices:
            for t in D.extract_row(v).indices:
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s, t] == D[s, v] + D[v, t]:
                    if (s, t) not in paths:
                        paths[(s,t)] = st_paths(pred[s], s, t)
                        for p in paths[(s, t)]:
                            V.update(p)
                    for p in paths[(s, t)]:
                        if v in p:
                            v_paths[v].append(p)
    return list(V), v_paths, paths

def CENTRALITY_paths_v1(D, pred, states):
    n = D.nrows
    S = 0.0
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    v_paths = dict()
    V = set()
    for v in range(n):
        S_exclude_v = S - S_sv[v] - S_vt[v]
        v_paths[v] = []
        for s in D.extract_col(v).indices:
            for t in D.extract_row(v).indices:
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s, t] == D[s, v] + D[v, t]:
                    sv_paths = st_paths(pred[s], s, v)
                    vt_paths = st_paths(pred[s], v, t)
                    for p1 in sv_paths:
                        for p2 in vt_paths:
                            v_paths[v].append(p1 + p2[1:])
                            V.update(p1 + p2[1:])
    return list(V), v_paths


def st_paths(pred, s, t):
    stack = [[t, 0]]
    top = 0
    paths = []
    while top >= 0:
        node, i = stack[top]
        if node == s:
            paths.append([p for p, n in reversed(stack[:top+1])])
        if node not in pred:
            continue
        if len(pred[node]) > i:
            top += 1
            if top == len(stack):
                stack.append([pred[node][i],0])
            else:
                stack[top] = [pred[node][i],0]
        else:
            stack[top-1][1] += 1
            top -= 1
    return paths
