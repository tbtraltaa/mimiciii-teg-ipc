import numpy as np

def get_paths_by_PC(PC, PC_P, paths, P=[0, 100]):
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
