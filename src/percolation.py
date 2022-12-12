import numpy as np
import pprint
import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic as dijkstra,
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic as shortest_path,
)

def percolation_centrality_with_target(G, states=None, weight=None):
    PC = dict.fromkeys(G, 0.0)

    n = G.number_of_nodes()
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
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
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    delta = states[t] - states[s]
                    w = float(delta) if delta > 0 else float(0.0) 
                    w /= S_exclude_v 
                    sigma_v_st = float(len(D[s][1][v]) * len(D[v][1][t]))
                    sigma_st = len(D[s][1][t])
                    PC[v] += sigma_v_st / sigma_st * w
        
    for v in PC:
        PC[v] *= 1.0 / (n - 2)
    return PC
