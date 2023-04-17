import pandas as pd
import numpy as np
import pprint
from datetime import timedelta
import time
import warnings
warnings.filterwarnings('ignore')


from teg.schemas import *
from teg.mimic_events import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target
from teg.apercolation import *
from teg.graph_vis import *
from teg.paths import *
from teg.plot import *

# Experiment configuration
conf = {
    'duration': False,
    'max_hours': 240,
    #'max_hours': 168,
    'min_age': 15,
    'max_age': 89,
    'age_interval': 5, # in years, for patients
    'starttime': False,
    #'starttime': '2143-01-14',
    #'endtime': '2143-01-21',
    #'endtime': '2143-02-14',
    'endtime': False,
    'min_missing_percent': 40, # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 1: 1},
    'PC_percentile': [97, 100],
    'path_percentile': [95, 100],
    'PI_only_sql': 'one', # PI patients slow to query chartevents
    'PI_only': False, # Delete non PI patients after querying all events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'scale_PC': True, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': False, # Use a list of vitals related to PI
    'skip_repeat': True,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 60],
    'input_percentile': [40, 100],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': False,
    'vis': False
}
# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=3, hours=0),
    'Discharges': timedelta(days=3, hours=0),
    'Icu In': timedelta(days=3, hours=0),
    'Icu_Out': timedelta(days=3, hours=0),
    'Callout': timedelta(days=3, hours=0),
    'Transfer In': timedelta(days=3, hours=0),
    'Transfer Out': timedelta(days=3, hours=0),
    'CPT': timedelta(days=3, hours=0),
    'Presc': timedelta(days=3, hours=0),
    'Services': timedelta(days=3, hours=0),
    'other': timedelta(days=3, hours=0),
    'diff_type_same_patient': timedelta(days=1, hours=0),
    'PI': timedelta(days=3, hours=0),
    'Braden': timedelta(days=3, hours=0),
    'Input': timedelta(days=3, hours=0),
}

join_rules = {
    'IDs': EVENT_IDs if not conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': conf['duration'],
    'duration_similarity': timedelta(days=2),
    'sequential_join': True,
    'max_pi_stage':1,
    'include_numeric': True
}


def eventgraph_mimiciii(event_list, join_rules, conf, file_name):
    patients, events = mimic_events(event_list, join_rules, conf)
    n = len(events)
    A = build_eventgraph(patients, events, join_rules)
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    print(states)
    print(np.nonzero(states))
    start = time.time()
    PC_values = algebraic_PC(A, states=states)
    print("Time for PC without paths", float(time.time() - start)/60.0)
    '''
    start = time.time()
    P, pred, D = algebraic_PC_with_pred(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)
    start = time.time()
    V, v_paths = PC_paths(D, pred, states)
    print("Compute paths", float(time.time() - start)/60.0)
    start = time.time()
    start = time.time()
    PC_values, V, v_paths = algebraic_PC_with_paths_v1(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)
    PC_values, V, paths, all_paths = algebraic_PC_with_paths(A, states=states)
    print('Algebraic PC time', float(time.time() - start)/60.0)
    # Check if algebraic PC match PC from networkx
    print(PC_values)
    b = np.sort(np.nonzero(PC_values)[0])
    print(b)
    '''
    '''
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print(nx.is_directed_acyclic_graph(G))
    start = time.time()
    PC, V, paths, all_paths = PC_with_target(G, states=states, weight='weight')
    PC_values = np.array(list(PC.values()))
    print(PC)
    print('PC time based on networkx', float(time.time() - start)/60.0)
    '''
    '''
    # Check if algebraic PC matches PC computed using NetworkX
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print(nx.is_directed_acyclic_graph(G))
    start = time.time()
    PC, V_nx, paths_nx, all_paths_nx = PC_with_target(G, states=states, weight='weight')
    print(PC)
    print('PC time based on networkx', float(time.time() - start)/60.0)
    a = np.sort(np.nonzero(list(PC.values()))[0])
    print(a)
    print('Algebraic PC nodes match that from networkx:', np.all(a==b))
    print('PC values match:', np.all(list(PC.values())==PC_values))
    print('V match:', np.all(sorted(V)==sorted(V_nx)))
    print('v_paths match:', paths==paths_nx)
    print('all_paths match:', all_paths==all_paths_nx)
    '''
    PC = dict()
    PC_all = dict()
    max_PC = float(max(PC_values))
    min_PC = float(min(PC_values))
    PC_vals = []
    for i, val in enumerate(PC_values):
        v = float(val) / max_PC if conf['scale_PC'] else float(val)
        PC_vals.append(v)
        PC_all[i] = v
        if val > 0:
            PC[i] = v
    '''
    PC_sorted = [[k, v, events[k]] for k, v in sorted(PC.items(), key=lambda x: x[1], reverse=True)]
    pprint.pprint(PC_sorted[:11])
    PC_top = dict()
    num = 100 if len(PC_sorted) > 100 else len(PC_sorted)
    for i, info in enumerate(PC_sorted):
        if info[2]['type'] not in PC_top:
            PC_top[info[2]['type']] = [1, info[1]]
        else:
            PC_top[info[2]['type']][0] += 1
            PC_top[info[2]['type']][1] += info[1]
    pprint.pprint([e for e in events if e['type']=='admissions' or e['type'] =='discharges'])
    pprint.pprint(PC_top, sort_dicts=False)
    '''
    PC_nz = list(PC.values())
    P_min = np.percentile(PC_nz, conf['PC_percentile'][0])
    P_max = np.percentile(PC_nz, conf['PC_percentile'][1])
    print("Nonzero PC", len(PC_nz))
    print("Min, Max PC", min_PC, max_PC)
    print("Min, Max PC scaled", min(PC_nz), max(PC_nz))
    print("Percentile", P_min, P_max)
    PC_P = dict([(i, v) for i, v in PC.items() if v >= P_min and v <= P_max])
    print("Nodes above percentile", len(PC_P))
    plot_PC(events, PC, conf, nbins=30)
    plot_PC(events, PC_P, conf, conf['PC_percentile'], nbins=10)
    if conf['vis'] and n > 500:
        # when a graph is too large for visualization
        # use only shortest path subgraph
        A = A.toarray()
        A = dok_matrix(A[np.array(V, dtype=int)][:, np.array(V, dtype=int)])
        PC_all = dict([(i, PC_all[v]) for i, v in enumerate(V)])
        events = [events[v] for v in V]
        paths_new = dict()
        for v in paths:
            if v not in V:
                continue
            if v not in paths_new:
                paths_new[v] = []
            for p in paths[v]:
                paths_new[v].append([V.index(i) for i in p])
        paths = paths_new
        PC_P = dict([(V.index(i), v) for i, v in PC_P.items()])
        V = range(len(V))
        G = build_networkx_graph(A, events, patients, PC_all, conf, join_rules)
        file_name += "_Q" + str(len(conf['quantiles']))
        paths_P = dict([(i, paths[i]) for i in PC_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_PC(PC_all, PC_P, paths_P, conf['path_percentile'])
            visualize_SP_tree(G, list(PC_P.keys()), paths_P_P, file_name+"PC_P_Path_P_P")
            visualize_graph(G, list(PC_P.keys()), paths_P_P, file_name+"all_PC_P_Path_P_P")
        else:
            visualize_graph(G, list(PC_P.keys()), paths_P, file_name+"all_PC_P")
        visualize_SP_tree(G, V, paths, file_name+"SP_all")
        visualize_SP_tree(G, list(PC_P.keys()), paths_P, file_name+"PC_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(PC_P.keys()), file_name+"V_percentile")
    elif conf['vis']:
        G = build_networkx_graph(A, events, patients, PC_all, conf, join_rules)
        file_name += "_Q" + str(len(conf['quantiles']))
        paths_P = dict([(i, paths[i]) for i in PC_P])
        if conf['path_percentile']:
            paths_P_P = get_paths_by_PC(PC_all, PC_P, paths_P, conf['path_percentile'])
            visualize_SP_tree(G, list(PC_P.keys()), paths_P_P, file_name+"PC_P_Path_P_P")
            visualize_graph(G, list(PC_P.keys()), paths_P_P, file_name+"all_PC_P_Path_P_P")
        else:
            visualize_graph(G, list(PC_P.keys()), paths_P, file_name+"all_PC_P")
        visualize_SP_tree(G, V, paths, file_name+"SP_all")
        visualize_SP_tree(G, list(PC_P.keys()), paths_P, file_name+"PC_P")
        attrs = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(PC_P.keys()), file_name+"V_PC_P")


if __name__ == "__main__":
    fname_keys = [
        'max_hours',
        'starttime',
        'endtime',
        'PC_percentile',
        'drug_percentile',
        'input_percentile',
        'skip_repeat',
        'PI_only_sql',
        'hadm_limit', 
        'min_missing_percent']
    fname_LF = 'output/TEG'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules.items()
                                    if k in fname_keys])

    eventgraph_mimiciii(EVENTS, join_rules, conf, fname_LF)
