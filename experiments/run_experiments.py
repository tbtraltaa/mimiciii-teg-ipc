import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, datetime
import time
import warnings
warnings.filterwarnings('ignore')

'''
from teg.schemas import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target_path_nx
from teg.apercolation import *
from teg.event_utils import *
from teg.PC_utils import *
from teg.utils import *
from teg.graph_vis import *
from teg.build_graph import *
from teg.paths import *
from teg.plot import *
from teg.plot_events import *
from teg.psm import *
from teg.pca import *
'''

from mimiciii_teg.schemas.schemas import *
from mimiciii_teg.teg.eventgraphs import *
from mimiciii_teg.ipc.percolation import PC_with_target_path_nx
from mimiciii_teg.ipc.apercolation import *
from mimiciii_teg.utils.event_utils import *
from mimiciii_teg.utils.PC_utils import *
from mimiciii_teg.utils.utils import *
from mimiciii_teg.vis.graph_vis import *
from mimiciii_teg.teg.build_graph import *
from mimiciii_teg.teg.paths import *
from mimiciii_teg.vis.plot import *
from mimiciii_teg.vis.plot_events import *
from mimiciii_teg.utils.psm import *


def run_experiments(admissions, events, conf, join_rules, fname, title=''):
    n = len(events)
    A, interconnection = build_eventgraph(admissions, events, join_rules)
    if interconnection == 0:
        pprint.pprint(events)
        return None
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    if not conf['PC_path'] or n > 5000:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    elif conf['PC_path']:
        start = time.time()
        #PC_values, pred, D = algebraic_PC_with_pred(A, states=states)
        #print("Time for PC with pred", float(time.time() - start)/60.0)
        #V, v_paths, paths = PC_paths(D, pred, states)
        PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
        print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
    if max(PC_values) == 0:
        return None
    PC_R = process_PC_values(events, PC_values, conf) 
    ET_PC_R = process_event_type_PC(events, PC_values, conf) 
    PPC_R = get_patient_PC_total(events, PC_R['PC_all'], conf)
    PPC_R['patient_PC'] = get_patient_max_PC(events, PC_R['PC_all'], conf['PC_time_unit'])
    if conf['vis']:
        plot_PC(events, PC_R['PC_nz'], conf, nbins=30, title=title, fname=f"{fname}_nz")
        plot_PC(events, PC_R['PC_P'], conf, conf['P'], nbins=10, title=title, fname=f"{fname}_P")
        plot_event_type_PC(ET_PC_R['ET_PC'],
                           ET_PC_R['ET_PC_freq'],
                           ET_PC_R['ET_PC_avg'],
                           conf,
                           title=title,
                           fname=f"{fname}_event_type")
        plot_event_type_PC(ET_PC_R['ET_PC_P'], 
                           ET_PC_R['ET_PC_P_freq'],
                           ET_PC_R['ET_PC_P_avg'],
                           conf, 
                           conf['P'],
                           title=title,
                           fname=f"{fname}_event_type_P")
        if conf['PC_path'] and n <= 5000:
            simple_visualization(A, 
                                 events, 
                                 admissions,
                                 PC_R['PC_all'],
                                 PC_R['PC_P'],
                                 conf,
                                 join_rules,
                                 fname)
            visualize(admissions,
                      events,
                      A,
                      V, 
                      PC_R['PC_all'],
                      PC_R['PC_P'],
                      v_paths,
                      paths,
                      conf,
                      join_rules,
                      fname)
        else:
            simple_visualization(A,
                                 events,
                                 admissions, 
                                 PC_R['PC_all'],
                                 PC_R['PC_P'],
                                 conf,
                                 join_rules,
                                 fname)
    results = {}
    results.update(PC_R)
    results.update(ET_PC_R)
    results.update(PPC_R)
    return results


def run_iterations(PI_admissions, NPI_admissions, PI_events, NPI_events, conf, join_rules, fname, title='', 
                   vis_last_iter = False, PC_path_last_iter = False):
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
            title = ""
            plot_types(PI_events,
                             NPI_events,
                             conf,
                             title = title,
                             fname = f"{fname}_Types_{i+1}")
            plot_event_types(PI_events,
                             NPI_events,
                             conf,
                             title = title,
                             fname = f"{fname}_event_types_{i+1}")
            plot_event_parent_types(PI_events,
                                    NPI_events,
                                    title = title,
                                    fname = f"{fname}_Categories_{i+1}")
        else:
            title = f"Iteration {i + 1}"
        PI_results = run_experiments(PI_admissions,
                                       PI_events,
                                       conf,
                                       join_rules,
                                       fname + f'_PI_{i+1}', 'PI: ' + title)
        NPI_results = run_experiments(NPI_admissions,
                                          NPI_events,
                                          conf,
                                          join_rules,
                                          fname + f'_NPI_{i+1}', 'NPI: ' + title)
        if NPI_results is None or PI_results is None:
            return PI_events, NPI_events, PI_results, NPI_results
        if conf['iter_type'] == 'event_type_PC':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['ET_PC_P'],
                                        NPI_results['ET_PC_P'])
            if conf['P_remove']:
                PI_remove = PI_results['ET_PC_remove']
                NPI_remove = NPI_results['ET_PC_remove']
        elif conf['iter_type'] == 'event_PC':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['PC_P_ET'],
                                        NPI_results['PC_P_ET'])
            if conf['P_remove']:
                PI_remove = PI_results['PC_remove_ET']
                NPI_remove = NPI_results['PC_remove_ET']
        elif conf['iter_type'] == 'average_event_PC':
            PI_types, NPI_types, I = dict_intersection_and_differences(
                                        PI_results['ET_PC_P_avg'],
                                        NPI_results['ET_PC_P_avg'])
            if conf['P_remove']:
                PI_remove = PI_results['ET_PC_avg_remove']
                NPI_remove = NPI_results['ET_PC_avg_remove']

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
                             title = title,
                             fname = f"{fname}_Types_{i+1}")
            plot_event_types(PI_events,
                             NPI_events,
                             conf,
                             title = title,
                             fname = f"{fname}_event_types_{i+1}")
            plot_event_parent_types(PI_events,
                                    NPI_events,
                                    title = title,
                                    fname = f"{fname}_Categories_{i+1}")
        if len(I) == 0 and vis_last_iter and PC_path_last_iter and not last_iter:
            conf['vis'] = True
            vis_last_iter = False
            conf['PC_path'] = True
            PC_path_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and vis_last_iter and not PC_path_last_iter and not last_iter:
            conf['vis'] = True
            conf['PC_path'] = False
            vis_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and not vis_last_iter and PC_path_last_iter and not last_iter:
            conf['PC_path'] = True
            PC_path_last_iter = False
            i -= 1
            last_iter = True
        elif len(I) == 0 and not last_iter:
            conf['vis'] = False
            conf['PC_path'] = False
            last_iter = True
            i -= 1
        elif last_iter:
            break
        i += 1
    if conf['PC_P_events']:
        PI_events = get_top_events(PI_events, PI_results['PC_P'], conf, I)
        NPI_events = get_top_events(NPI_events, NPI_results['PC_P'], conf, I)
    df.to_csv(f"{fname}_results.csv")
    return PI_events, NPI_events, PI_results, NPI_results


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
