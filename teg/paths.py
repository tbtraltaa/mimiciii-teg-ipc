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

def analyze_paths(events, PC, V, paths):
    for i in V:
        for path in paths[i]:
            u = path[0]
            for v in path[1:]:
                if events[u]['type'] == events[v]['type']:
                    pass
                else:
                    pass
