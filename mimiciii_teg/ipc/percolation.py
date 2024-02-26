import numpy as np
import pprint
import networkx as nx

def PC_with_target_path_nx(G, states=None, weight=None):
'''
Inverse percolation centrality algorithm using Networkx
'''
    #PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    PC = np.zeros(n)
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in D:
            if v not in D[s][0]:
                continue
            for t in D[s][0]:
                if t not in D[v][0]:
                    continue
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    if (s, t) not in paths:
                        paths[(s,t)] = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                    sigma_st = len(paths[(s,t)])
                    if sigma_st == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    w = delta/S_exclude_v 
                    if v not in v_paths:
                        v_paths[v] = []
                    PC[v] += sigma_v_st / sigma_st * w
                    for p in paths[(s, t)]:
                        if v in p:
                            v_paths[v].append(p)
                            V.update(p)
    return PC, V, v_paths, paths

def PC_with_target_nx(G, states=None, weight=None):
    #PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    PC = np.zeros(n)
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    paths = dict()
    for v in range(n):
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in D:
            if v not in D[s][0]:
                continue
            for t in D[s][0]:
                if t not in D[v][0]:
                    continue
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    if (s, t) not in paths:
                        paths[(s,t)] = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                    sigma_st = len(paths[(s,t)])
                    if sigma_st == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    w = delta/S_exclude_v 
                    PC[v] += sigma_v_st / sigma_st * w
    return PC


def PC_with_target_v1(G, states=None, weight=None):
    #PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    PC = np.zeros(n)
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in D:
            if v not in D[s][0]:
                continue
            for t in D[s][0]:
                if t not in D[v][0]:
                    continue
                delta = states[t] - states[s]
                if delta <= 0:
                    continue
                if (s, t) not in paths:
                    paths[(s,t)] = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    w = float(delta) 
                    w /= S_exclude_v 
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    sigma_st = len(paths[(s, t)])
                    if sigma_st > 0:
                        if v not in paths:
                            paths[v] = []
                        PC[v] += sigma_v_st / sigma_st * w
                        for p1 in sv_paths:
                            for p2 in vt_paths:
                                v_paths[v].append(p1 + p2[1:])
                                V.update(p1 + p2[1:])
        
    return PC, V, v_paths, paths


def PC_with_target_v2(G, states=None, weight=None):
    PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    S = 0.0
    targets = dict()
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        if S_vt[v] != 0:
            targets[v] = [i for i, val in enumerate(deltas) if val > 0]
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
                st_paths = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
            except nx.NetworkXNoPath:
                continue
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
                if (s, v) not in D:
                    try:
                        sv_len = nx.shortest_path_length(G, source=s, target=v, weight='weight')
                        sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    except nx.NetworkXNoPath:
                        continue
                else:
                    sv_len = D[s, v][0]
                    sv_paths = D[s, v][1]
                if (v, t) not in D:
                    try:
                        vt_len = nx.shortest_path_length(G, source=v, target=t, weight='weight')
                        vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    except nx.NetworkXNoPath:
                        continue
                else:
                    vt_len = D[v, t][0]
                    vt_paths = D[v, t][1]
                if s != v and t != v and s != t and D[s, t][0] == sv_len + vt_len:
                    w = (states[t] - states[s]) / S_exclude_v 
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    sigma_st = len(D[s, t][1])
                    if sigma_st == 0:
                        continue
                    if v not in paths:
                        paths[v] = []
                    PC[v] += sigma_v_st / sigma_st * w
                    for p1 in sv_paths:
                        for p2 in vt_paths:
                            paths[v].append(p1 + p2[1:])
    return PC, v_set, paths
