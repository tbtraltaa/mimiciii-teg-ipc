import numpy as np
import pprint
import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic as dijkstra,
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic as shortest_path,
)

'''
def percolation_centrality_with_target(G, states=None, weight=None):
    PC = dict.fromkeys(G, 0.0)

    n = G.number_of_nodes()
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    paths = dict()
    for v in range(n):
        if v not in paths:
            paths[v] = []
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in range(n):
            if s not in D or v not in D[s][0]:
                continue
            for t in range(n):
                if t not in D[s][0] or t not in D[v][0]:
                    continue
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    delta = states[t] - states[s]
                    if delta <= 0:
                        continue
                    w = float(delta)
                    w /= S_exclude_v 
                    sigma_v_st = float(len(D[s][1][v]) * len(D[v][1][t]))
                    sigma_st = len(D[s][1][t])
                    PC[v] += sigma_v_st / sigma_st * w
        
    for v in PC:
        PC[v] *= 1.0 / (n - 2)
    return PC, None
'''

def PC_with_target(G, states=None, weight=None):
    PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
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
        for s in range(n):
            if s not in D or v not in D[s][0]:
                continue
            for t in range(n):
                if t not in D[s][0] or t not in D[v][0]:
                    continue
                st_paths = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    delta = states[t] - states[s]
                    if delta <= 0:
                        continue
                    w = float(delta) if delta > 0 else float(0.0) 
                    w /= S_exclude_v 
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    sigma_st = len(st_paths)
                    if sigma_st > 0:
                        if v not in paths:
                            paths[v] = []
                        PC[v] += sigma_v_st / sigma_st * w
                        for p1 in sv_paths:
                            for p2 in vt_paths:
                                paths[v].append(p1 + p2[1:])
        
    for v in PC:
        PC[v] *= 1.0 / (n - 2)
    return PC, paths

def percolation_centrality_with_target(G, states=None, weight=None):
    PC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
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
        for s in range(n):
            if s not in D or v not in D[s][0]:
                continue
            for t in range(n):
                if t not in D[s][0] or t not in D[v][0]:
                    continue
                st_paths = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    delta = states[t] - states[s]
                    if delta <= 0:
                        continue
                    w = float(delta) if delta > 0 else float(0.0) 
                    w /= S_exclude_v 
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    sigma_st = len(st_paths)
                    sigma_v_st1 = float(len(D[s][1][v]) * len(D[v][1][t]))
                    sigma_st1 = len(D[s][1][t])
                    print(sigma_v_st == sigma_v_st1, sigma_st == sigma_st1)
                    print(st_paths, D[s][1][t])
                    exit()
                    if sigma_st > 0:
                        if v not in paths:
                            paths[v] = []
                        PC[v] += sigma_v_st / sigma_st * w
                        for p1 in sv_paths:
                            for p2 in vt_paths:
                                paths[v].append(p1 + p2[1:])
        
    for v in PC:
        PC[v] *= 1.0 / (n - 2)
    return PC, paths
