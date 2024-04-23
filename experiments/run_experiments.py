import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, datetime
import time
import warnings
warnings.filterwarnings('ignore')

from mimiciii_teg.schemas.schemas import *
from mimiciii_teg.teg.eventgraphs import *
from mimiciii_teg.centrality.IPC import IPC_dense, IPC_sparse
from mimiciii_teg.centrality.IPC_nx import IPC_nx, IPC_with_paths_nx
from mimiciii_teg.centrality.algebraic_IPC import algebraic_IPC, algebraic_IPC_with_paths
from mimiciii_teg.utils.event_utils import *
from mimiciii_teg.utils.CENTRALITY_utils import *
from mimiciii_teg.utils.utils import *
from mimiciii_teg.vis.graph_vis import *
from mimiciii_teg.teg.build_graph import *
from mimiciii_teg.teg.paths import *
from mimiciii_teg.vis.plot_centrality import *
from mimiciii_teg.vis.plot_path_centrality import plot_path_CENTRALITY
from mimiciii_teg.vis.plot_events import *
from mimiciii_teg.utils.psm import *


def run_experiments(admissions, events, conf, join_rules, fname):
    '''
    Run a centrality experiment
    '''
    n = len(events)
    A, interconnection = build_eventgraph(admissions, events, join_rules)
    if interconnection == 0:
        pprint.pprint(events)
        return None
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    if 'check_IPC_values' in conf:
        if conf['check_IPC_values']:
            check_IPC_values(A, states)
    if not conf['CENTRALITY_path'] or n > conf['max_n_for_ICP_paths']:
        '''
            start = time.time()
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            CENTRALITY_values = IPC_nx(G, x=states)
            print("Time for IPC without paths (NetworkX)", float(time.time() - start)/60.0, 'min' )
        '''
        start = time.time()
        CENTRALITY_values = algebraic_IPC(A, x=states)
        print("Time for IPC without paths ", float(time.time() - start)/60.0, 'min' )
    elif conf['CENTRALITY_path']:
        '''
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start = time.time()
            CENTRALITY_values, v_paths, paths = IPC_with_paths_nx(G, x=states)
            print('IPC time with paths (NetworkX))', float(time.time() - start)/60.0, 'min')
        '''
        start = time.time()
        #CENTRALITY_values, pred, D = algebraic_CENTRALITY_with_pred(A, states=states)
        #print("Time for CENTRALITY with pred", float(time.time() - start)/60.0)
        #V, v_paths, paths = CENTRALITY_paths(D, pred, states)
        CENTRALITY_values, v_paths, paths = algebraic_IPC_with_paths(A, x=states)
        print('Algebraic IPC time with paths', float(time.time() - start)/60.0, 'min')
    if max(CENTRALITY_values) == 0:
        return None
    CENTRALITY_R = process_CENTRALITY_values(events, CENTRALITY_values, conf) 
    ET_CENTRALITY_R = process_event_type_CENTRALITY(events, CENTRALITY_values, conf) 
    PCENTRALITY_R = get_patient_CENTRALITY_total(events, CENTRALITY_R['CENTRALITY_all'], conf)
    PCENTRALITY_R['patient_CENTRALITY'] = get_patient_max_CENTRALITY(events, CENTRALITY_R['CENTRALITY_all'], conf['CENTRALITY_time_unit'])
    if conf['vis']:
        plot_CENTRALITY(events, CENTRALITY_R['CENTRALITY_nz'], conf, conf['P'], \
                        CENTRALITY_R['P'], nbins=30, fname=f"{fname}_nz")
        plot_CENTRALITY(events, CENTRALITY_R['CENTRALITY_P'], conf, conf['P'], nbins=10, fname=f"{fname}_P")
        plot_event_type_CENTRALITY(ET_CENTRALITY_R['ET_CENTRALITY'],
                           ET_CENTRALITY_R['ET_CENTRALITY_freq'],
                           ET_CENTRALITY_R['ET_CENTRALITY_avg'],
                           conf,
                           conf['ET_P'],
                           ET_CENTRALITY_R['ET_P'],
                           fname=f"{fname}_event_type")
        plot_event_type_CENTRALITY(ET_CENTRALITY_R['ET_CENTRALITY_P'], 
                           ET_CENTRALITY_R['ET_CENTRALITY_P_freq'],
                           ET_CENTRALITY_R['ET_CENTRALITY_P_avg'],
                           conf, 
                           conf['ET_P'],
                           fname=f"{fname}_event_type_P")
        visualize_centrality(A,
                             events,
                             admissions, 
                             CENTRALITY_R['CENTRALITY_all'],
                             CENTRALITY_R['CENTRALITY_P'],
                             conf,
                             join_rules,
                             fname)
        visualize_centrality_ET(A,
                                events,
                                admissions, 
                                CENTRALITY_R['CENTRALITY_all'],
                                CENTRALITY_R['CENTRALITY_P'],
                                ET_CENTRALITY_R['ET_CENTRALITY_P'],
                                conf,
                                join_rules,
                                fname)
        if conf['CENTRALITY_path'] and n <= 5000:
            plot_path_CENTRALITY(events, conf, CENTRALITY_R['CENTRALITY_all'], paths, fname)
            # SCP stands for Shortest Centrality Paths
            visualize_SCP(admissions,
                      events,
                      A,
                      CENTRALITY_R['CENTRALITY_all'],
                      CENTRALITY_R['CENTRALITY_P'],
                      v_paths,
                      paths,
                      conf,
                      join_rules,
                      fname)

            visualize_SCP_ET(admissions,
                      events,
                      A,
                      CENTRALITY_R['CENTRALITY_all'],
                      CENTRALITY_R['CENTRALITY_P'],
                      ET_CENTRALITY_R['ET_CENTRALITY_P'],
                      v_paths,
                      paths,
                      conf,
                      join_rules,
                      fname)
    results = {}
    results.update(CENTRALITY_R)
    results.update(ET_CENTRALITY_R)
    results.update(PCENTRALITY_R)
    return results


def run_iterations(PI_admissions, NPI_admissions, PI_events, NPI_events, conf, join_rules, fname, 
                   vis_last_iter = False, CENTRALITY_path_last_iter = False):
    I = []
    #for i in range(conf['iterations']):
    i = 0
    df = None
    last_iter = False
    while True:

        print("==========================================================")
        print(f"Iteration: {i + 1}")
        print("==========================================================")
        if len(I) != 0:
            PI_events = remove_event_types(PI_events, I) 
            NPI_events = remove_event_types(NPI_events, I) 
        if conf['P_remove'] and last_iter:
            print('PI remove', PI_remove)
            print('NPI remove', NPI_remove)
            PI_events = remove_event_types(PI_events, PI_remove) 
            NPI_events = remove_event_types(NPI_events, NPI_remove) 
        if i == 0:
            plot_types(PI_events,
                             NPI_events,
                             conf,
                             fname = f"{fname}_Types_{i+1}")
            plot_event_types(PI_events,
                             NPI_events,
                             conf,
                             fname = f"{fname}_event_types_{i+1}")
            plot_event_parent_types(PI_events,
                                    NPI_events,
                                    fname = f"{fname}_Categories_{i+1}")
        PI_results = run_experiments(PI_admissions,
                                       PI_events,
                                       conf,
                                       join_rules,
                                       fname + f'_PI_{i+1}')
        NPI_results = run_experiments(NPI_admissions,
                                          NPI_events,
                                          conf,
                                          join_rules,
                                          fname + f'_NPI_{i+1}')
        if NPI_results is None or PI_results is None:
            return PI_events, NPI_events, PI_results, NPI_results
        if conf['iter_type'] == 'event_type_CENTRALITY':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['ET_CENTRALITY_P'],
                                        NPI_results['ET_CENTRALITY_P'])
            if conf['P_remove']:
                PI_remove = PI_results['ET_CENTRALITY_remove']
                NPI_remove = NPI_results['ET_CENTRALITY_remove']
        elif conf['iter_type'] == 'event_CENTRALITY':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['CENTRALITY_P_ET'],
                                        NPI_results['CENTRALITY_P_ET'])
            if conf['P_remove']:
                PI_remove = PI_results['CENTRALITY_remove_ET']
                NPI_remove = NPI_results['CENTRALITY_remove_ET']
        elif conf['iter_type'] == 'average_event_CENTRALITY':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['ET_CENTRALITY_P_avg'],
                                        NPI_results['ET_CENTRALITY_P_avg'])
            if conf['P_remove']:
                PI_remove = PI_results['ET_CENTRALITY_avg_remove']
                NPI_remove = NPI_results['ET_CENTRALITY_avg_remove']

        print("==========================================================")
        print(f"Intersections")
        print("==========================================================")
        pprint.pprint(I)
        if df is None and not last_iter:
            df = get_event_type_df(PI_types, 
                                   NPI_types, 
                                   I, 
                                   i + 1, 
                                   PI_results,
                                   NPI_results, 
                                   conf)
        elif df is not None and not last_iter:
            df_i = get_event_type_df(PI_types, 
                                     NPI_types,
                                     I, 
                                     i + 1,
                                     PI_results,
                                     NPI_results,
                                     conf)
            df = pd.concat([df_i, df], ignore_index=True) 
        if last_iter:
            plot_types(PI_events,
                             NPI_events,
                             conf,
                             fname = f"{fname}_Types_{i+1}")
            plot_event_types(PI_events,
                             NPI_events,
                             conf,
                             fname = f"{fname}_event_types_{i+1}")
            plot_event_parent_types(PI_events,
                                    NPI_events,
                                    fname = f"{fname}_Categories_{i+1}")
        if len(I) == 0 and vis_last_iter and CENTRALITY_path_last_iter and not last_iter:
            conf['vis'] = True
            vis_last_iter = False
            conf['CENTRALITY_path'] = True
            CENTRALITY_path_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and vis_last_iter and not CENTRALITY_path_last_iter and not last_iter:
            conf['vis'] = True
            conf['CENTRALITY_path'] = False
            vis_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and not vis_last_iter and CENTRALITY_path_last_iter and not last_iter:
            conf['CENTRALITY_path'] = True
            CENTRALITY_path_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and not last_iter:
            conf['vis'] = False
            conf['CENTRALITY_path'] = False
            last_iter = True
            i -= 1
        elif last_iter:
            break
        i += 1
    if conf['CENTRALITY_P_events']:
        PI_events = get_top_events(PI_events, PI_results['CENTRALITY_P'], conf, I)
        NPI_events = get_top_events(NPI_events, NPI_results['CENTRALITY_P'], conf, I)
    df.to_csv(f"{fname}_results.csv")
    return PI_events, NPI_events, PI_results, NPI_results


def check_IPC_values(A, x):
    C_paths, v_paths, paths = algebraic_IPC_with_paths(A, x)
    C = algebraic_IPC(A, x)
    C_dense = IPC_dense(A, x)
    C_sparse= IPC_sparse(A, x)
    print('C', C)
    print('C_dense', C_dense)
    print('C_sparse', C_sparse)
    print('C matches C_paths', np.allclose(C, C_paths))
    print('C matches C_sparse:', np.allclose(C, C_sparse))
    print('C matches C_dense:', np.allclose(C, C_dense))
    print('C_dense matches C_sparse:', np.allclose(C_sparse, C_dense))
    # Check if algebraic IPC matches IPC computed using NetworkX
    # NetworkX has some numerical instability and doesn't satisfy Bellman criterion
    # of sigma_sr == sigma_sv * sigma_vr
    #G = nx.from_numpy_array(A.todense(), create_using=nx.DiGraph)
    #C_nx_paths, v_paths_nx, paths_nx = IPC_with_paths_nx(G, x=x)
    #C_nx = IPC_nx(G, x)
    #print('C matches C_nx', np.allclose(C, C_nx))
    #print('C_nx', C_nx)
    #print('C_nx matches C_nx_paths:', np.allclose(C_nx_paths, C_nx))
    #print('C_nx matches C_sparse:', np.allclose(C_sparse, C_nx))
    #print('C_nx matches C_dense:', np.allclose(C_dense, C_nx))
