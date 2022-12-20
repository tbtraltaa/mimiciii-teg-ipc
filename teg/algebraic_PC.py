import numpy as np
import pprint
import networkx as nx


def algebraic_PC_with_target(A, states=None, weight=None):
    PC = np.zeros(shape=(n,1), dtype=float)
    n = A.shape[0]
    S = 0.0
    targets = dict()
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        if S_vt[v] != 0:
            targets[v] = list(np.argwhere(deltas > 0))
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    paths = dict()
    D = dict()
    v_set = set()
    for s in targets:
        for t in targets[s]:
            try:
                length = nx.shortest_path_length(G, source=s, target=t, weight='weight')
            except nx.NetworkXNoPath:
                continue
            st_paths = nx.all_shortest_paths(G, source=s, target=v, weight='weight')
            for path in st_paths:
                for v in path:
                    v_set.add(v)
            D[(s, t)] = [length, st_paths]
    for v in v_set:
        S_exclude_v = S - S_sv[v] - S_vt[v]
        if S_exclude_v == 0:
            continue
        for s in targets:
            for t in targets[s]:
                if s != v and t != v and s != t and D[s, t][0] == D[s, v][0] + D[v, t][0]:
                    w = (states[t] - states[s]) / S_exclude_v 
                    sigma_v_st = float(len(D[s, v][1]) * len(D[v, t][1]))
                    sigma_st = len(D[s, t][1])
                    if sigma_st == 0:
                        continue
                    if v not in paths:
                        paths[v] = []
                    PC[v] += sigma_v_st / sigma_st * w
                    for p1 in D[s, v][1]:
                        for p2 in D[v, t][1]:
                            paths[v].append(p1 + p2[1:])
        
    for v in PC:
        PC[v] *= 1.0 / (n - 2)
    return PC, paths
