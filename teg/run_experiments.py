import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, datetime
import time
import warnings
warnings.filterwarnings('ignore')


from teg.schemas import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target_path_nx
from teg.apercolation import *
from teg.event_utils import *
from teg.PC_utils import *
from teg.graph_vis import *
from teg.build_graph import *
from teg.paths import *
from teg.plot import *
from teg.psm import *
from teg.pca import *


def run_experiments(admissions, events, conf, join_rules, fname, title=''):
    n = len(events)
    A, interconnection = build_eventgraph(admissions, events, join_rules)
    if interconnection == 0:
        return None, None, None, None
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    if not conf['PC_path']:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    else:
        start = time.time()
        #PC_values, pred, D = algebraic_PC_with_pred(A, states=states)
        #print("Time for PC with pred", float(time.time() - start)/60.0)
        #V, v_paths, paths = PC_paths(D, pred, states)
        PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
        print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
    if max(PC_values) == 0:
        return None, None, None, None
    PC_all, PC_nz, PC_P = process_PC_values(PC_values, conf) 
    patient_PC = get_patient_max_PC(events, PC_all, conf['PC_time_unit'])
    if conf['plot']:
        plot_PC(events, PC_nz, conf, nbins=30, title=title, fname=f"{fname}_nz")
        plot_PC(events, PC_P, conf, conf['PC_percentile'], nbins=10, title=title, fname=f"{fname}_P")
    if conf['vis'] and conf['PC_path']:
        simple_visualization(A, events, admissions, PC_all, PC_P, conf, join_rules, fname)
        visualize(admissions, events, A, V, PC_all, PC_P, v_paths, paths, conf, join_rules, fname+'_Paths_')
    elif conf['vis']:
        simple_visualization(A, events, admissions, PC_all, PC_P, conf, join_rules, fname)
    return PC_all, PC_nz, PC_P, patient_PC


def run_iterations(PI_admissions, NPI_admissions, PI_events, NPI_events, conf, join_rules, fname, title='', 
                   vis_last_iter = False, PC_path_last_iter = False):
    I = []
    #for i in range(conf['iterations']):
    i = 0
    while True:

        print("==========================================================")
        print(f"Iteration: {i + 1}")
        print("==========================================================")
        if len(I) != 0:
            PI_events = remove_event_type(PI_events, I) 
            NPI_events = remove_event_type(NPI_events, I) 
        PI_PC_all, PI_PC_nz, PI_PC_P, PI_patient_PC = run_experiments(PI_admissions,
                                                       PI_events,
                                                       conf,
                                                       join_rules,
                                                       fname + f'_PI_{i}', 'PI: ' + title)
        if PI_PC_P is None:
            return PI_events, NPI_events, PI_patient_PC, None
        PI_etypes_P = get_event_types(PI_events, PI_PC_P)

        NPI_PC_all, NPI_PC_nz, NPI_PC_P, NPI_patient_PC = run_experiments(NPI_admissions,
                                                          NPI_events,
                                                          conf,
                                                          join_rules,
                                                          fname + f'_NPI_{i}', 'NPI: ' + title)
        if NPI_PC_P is None:
            return PI_events, NPI_events, None, None
        NPI_etypes_P = get_event_types(NPI_events, NPI_PC_P)
        PI_types, NPI_types, I = intersection_and_differences(PI_etypes_P, NPI_etypes_P)
        if I == [] and vis_last_iter and PC_path_last_iter:
            conf['vis'] = True
            conf['plot'] = True
            vis_last_iter = False
            conf['PC_path'] = True
            PC_path_last_iter = False
        elif I == [] and vis_last_iter and not PC_path_last_iter:
            conf['vis'] = True
            conf['plot'] = True
            vis_last_iter = False
        elif I == [] and not vis_last_iter and not PC_path_last_iter:
            break
        i += 1
    if conf['PC_P_events']:
        PI_events = get_top_events(PI_events, PI_PC_P, conf, I)
        NPI_events = get_top_events(NPI_events, NPI_PC_P, conf, I)
    return PI_events, NPI_events, PI_patient_PC, NPI_patient_PC


def check_PC_values(A, states):
    start = time.time()
    PC_values = algebraic_PC(A, states=states)
    print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    start = time.time()
    b = np.sort(np.nonzero(PC_values)[0])
    print(b)

    # Check if algebraic PC matches PC computed using NetworkX
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    #print(nx.is_directed_acyclic_graph(G))
    start = time.time()
    PC, V_nx, v_paths_nx, paths_nx = PC_with_target_path_nx(G, states=states, weight='weight')
    print(PC)
    print('PC time based on networkx', float(time.time() - start)/60.0)
    a = np.sort(np.nonzero(list(PC.values()))[0])
    print(a)
    print('Algebraic PC nodes match that from networkx:', np.all(a==b))
    print('PC values match:', np.all(list(PC.values())==PC_values))
    print('V match:', np.all(sorted(V)==sorted(V_nx)))
    print('v_paths match:',v_paths==paths_nx)
    print('paths match:', paths==paths_nx)
    
    start = time.time()
    P, pred, D = algebraic_PC_with_pred(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)
    start = time.time()
    V, v_paths, paths = PC_paths(D, pred, states)
    print("Compute paths", float(time.time() - start)/60.0)

    start = time.time()
    PC_values, V, v_paths = algebraic_PC_with_paths_v1(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)

    start = time.time()
    PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
    print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
