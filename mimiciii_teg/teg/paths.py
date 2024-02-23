import numpy as np
import pprint

from mimiciii_teg.utils.event_utils import group_events_by_patient 

def get_patient_PC_paths(events, PC_all, paths, n):
    events_grouped = group_events_by_patient(events)
    patient_paths = dict()
    for p_id in events_grouped:
        s = events_grouped[p_id][0]['i']
        t = events_grouped[p_id][-1]['i']
        if (s, t) not in paths:
            pass
            # TODO: to fix this bug
            #print('st_path missing')
            #print('s', events_grouped[p_id][0])
            #print('t', events_grouped[p_id][-1])
            #pprint.pprint(events_grouped[p_id])
        else:
            st_paths = paths[(s, t)]
            patient_paths[p_id] = []
            for p in st_paths:
                path_PC = 0
                for v in p:
                    path_PC += PC_all[v]
                patient_paths[p_id].append({'PC': path_PC, 'path': p})
    paths = []
    for p_id in patient_paths:
        patient_paths[p_id] = sorted(patient_paths[p_id], key = lambda x: x['PC'], reverse=True)[:n]
        paths += [p['path'] for p in patient_paths[p_id]]
    return patient_paths, paths
    
def get_patient_shortest_paths(A, events, PC_all, paths, n):
    events_grouped = group_events_by_patient(events)
    patient_paths = dict()
    for p_id in events_grouped:
        s = events_grouped[p_id][0]['i']
        t = events_grouped[p_id][-1]['i']
        if (s, t) not in paths:
            pass
            #print('st_path missing')
            #print('s', events_grouped[p_id][0])
            #print('t', events_grouped[p_id][-1])
            #pprint.pprint(events_grouped[p_id])
        else:
            st_paths = paths[(s, t)]
            patient_paths[p_id] = []
            for p in st_paths:
                path_weight = 0
                i = p[0]
                for j in p[1:]:
                    path_weight += A[i, j]
                    i = j
                patient_paths[p_id].append({'w': path_weight, 'path': p})
    paths = []
    for p_id in patient_paths:
        patient_paths[p_id] = sorted(patient_paths[p_id], key = lambda x: x['w'])[: n]
        paths += [p['path'] for p in patient_paths[p_id]]
    return patient_paths, paths

def get_paths_by_PC(PC, PC_P, paths, P=[0, 100]):
    '''
    Computes total PC values for paths and
    returns paths with total PC values above the given percentile
    '''
    paths_PC = {}
    vals = []
    for v, v_paths in paths.items():
        paths_PC[v] = []
        for path in v_paths:
            summ = 0
            for p in path:
                summ += PC[p]
            paths_PC[v].append([summ, path])
            vals.append(summ)
    P_min = np.percentile(vals, P[0])
    P_max = np.percentile(vals, P[1])
    paths_P = {}
    for v, v_paths in paths_PC.items():
        if v not in PC_P:
            continue
        paths_P[v] = []
        for summ, path in v_paths:
            if summ >= P_min and summ <= P_max:
                paths_P[v].append(path)
    return paths_P

def get_total_path_PC(PC, PC_P, paths, P=[0, 100]):
    '''
    Returns total PC values of paths for each vertex
    '''
    total_path_PC = {}
    for v, v_paths in paths.items():
        total_path_PC[v] = 0
        for path in v_paths:
            summ = 0
            for p in path:
                summ += PC[p]
            total_path_PC[v] += summ
    return total_path_PC

def analyze_paths(events, PC, V, paths):
    for i in V:
        for path in paths[i]:
            u = path[0]
            for v in path[1:]:
                if events[u]['type'] == events[v]['type']:
                    pass
                else:
                    pass

def PC_paths(D, pred, states):
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

def PC_paths_v1(D, pred, states):
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
