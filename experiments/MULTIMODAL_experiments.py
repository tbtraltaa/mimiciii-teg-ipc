import os
import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import time
import warnings
warnings.filterwarnings('ignore')

from pygraphblas import *
options_set(nthreads=12)


from mimiciii_teg.queries.admissions import admissions
from mimiciii_teg.schemas.event_setup import *
from mimiciii_teg.schemas.PI_risk_factors import PI_VITALS, PI_VITALS_TOP_20
from mimiciii_teg.teg.events import *
from mimiciii_teg.vis.plot import *
from mimiciii_teg.vis.plot_patients import *
from run_experiments import *
from mimiciii_teg.queries.queries import get_db_connection
from mimiciii_teg.schemas.schemas import *

# Experiment configuration
# P = {<EVENT TYPE>: 
P = {
    'Admissions': [95, 100],
    'Discharges': [95, 100],
    'ICU In': [95, 100],
    'ICU Out': [95, 100],
    'Callout': [95, 100],
    'Transfer In': [95, 100],
    'Transfer Out': [95, 100],
    'CPT': [95, 100],
    'Presc Start': [95, 100],
    'Presc End': [95, 100],
    'Input': [95, 100],
    'Services': [95, 100],
    'PI': [95, 100],
    'Braden': [95, 100],
    'Intervention': [95, 100],
    'Vitals/Labs': [95, 100]
}

M_conf = {
    'modality': 'event_type',
    'include_chronic_illness': True,
    'patient_history': timedelta(weeks=24), # 6 months
    'admission_type': 'EMERGENCY',
    'duration': False,
    'max_hours': 336,
    'min_los_hours': 24,
    #'max_hours': 168,
    'min_age': 15,
    'max_age': 89,
    'age_interval': 5, # in years, for patients
    'starttime': False,
    #'starttime': '2143-01-14',
    #'endtime': '2143-01-21',
    #'endtime': '2143-02-14',
    'endtime': False,
    'missing_percent': [0, 100], # for mimic extract
    'PI_exclude_mid_stages': True,
    'CENTRALITY_time_unit': timedelta(days=0, hours=1), # maximum CENTRALITY per time unit
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 2: 1},
    'PI_exclude_mid_stages': True,
    'PI_daily_max_stage': True,
    'CENTRALITY_time_unit': timedelta(days=0, hours=1), # maximum CENTRALITY per time unit
    'max_pi_stage':2,
    'P': [90, 100],
    'P_results': [90, 100],
    'P_remove': [0, 20],
    'P_patients': [60, 100],
    'ET_CENTRALITY_min_freq': 0,
    'CENTRALITY_path': False,
    'P_max_n': False,
    'path_percentile': [95, 100],
    'PI_sql': 'one', #multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_only': False, # Delete non PI patients after querying all events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_CENTRALITY': False, # scale by max_CENTRALITY
    'Top_n_CENTRALITY': 20,
    'PI_vitals': PI_VITALS, # Use a list of vitals related to PI
    'skip_repeat': False,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'quantile_round': 2,
    'drug_percentile': [40, 70],
    'input_percentile': [20, 90],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': False,
    'min_n': 200,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'n_patient_paths': [1, 3, 5], # n highest CENTRALITY paths of a patient
    'vis': False,
    'plot': False,
    'plot_types_n': 30,
    'CENTRALITY_BS_nnz': 0, # in percentage
    'first_hadm': True,
    'dbsource': 'carevue', # carevue or False
    'iterations': 10,
    'iter_type': 'event_type_CENTRALITY', # event_type_CENTRALITY or event_CENTRALITY
    'CENTRALITY_P_events': False,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        #'admission_type',
        'insurance',
        'los',
        'age',
        'oasis'],
    'max_node_size': 300
}

#for key in CHRONIC_ILLNESS:
#    conf['psm_features'].append(key)

# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=2, hours=0),
    'Discharges': timedelta(days=2, hours=0),
    'ICU In': timedelta(days=2, hours=0),
    'ICU Out': timedelta(days=2, hours=0),
    'Callout': timedelta(days=2, hours=0),
    'Transfer In': timedelta(days=2, hours=0),
    'Transfer Out': timedelta(days=2, hours=0),
    'CPT': timedelta(days=2, hours=0),
    'Presc': timedelta(days=2, hours=0),
    'Services': timedelta(days=2, hours=0),
    'other': timedelta(days=2, hours=0),
    'diff_type_same_patient': timedelta(days=2, hours=0),
    'PI': timedelta(days=2, hours=0),
    'Braden': timedelta(days=2, hours=0),
    'Input': timedelta(days=2, hours=0),
    'Intervention': timedelta(days=2, hours=0),
    'Vitals/Labs': timedelta(days=2, hours=0),
}

M_join_rules = {
    'IDs': EVENT_IDs if not M_conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': M_conf['duration'],
    'duration_similarity': timedelta(days=2),
    'sequential_join': True,
    'max_pi_state':1,
    'max_pi_stage':2,
    'include_numeric': True
}

M_fname = 'output/Multimodal'

def MULTIMODAL_TEG_CENTRALITY_PI_ONLY(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    PI_df, PI_admissions = get_patient_demography(conn, conf) 
    print('Admissions', len(PI_admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    PI_events = events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    PI_hadms = tuple(PI_hadm_stage_t.keys())
    PI_events_grouped = group_events_by_parent_type(PI_events)
    PI_events_P = []
    max_pi_stage = f"PI Stage {conf['max_pi_stage']}"
    conf['vis'] = True
    conf['plot'] = True
    conf['CENTRALITY_path'] = True
    for parent_type in PI_events_grouped:
        if max_pi_stage in parent_type:
            continue
        print('==================================================================')
        print('Parent_type: ', parent_type)
        __type = parent_type.split('/')[0]
        pi_count = len(PI_events_grouped[parent_type])
        if pi_count > conf['min_n']:
            pi_events = PI_events_grouped[parent_type] + PI_events_grouped[max_pi_stage]
            # sort by type, t and reindex
            pi_events = sort_and_index_events(pi_events)
            # choose CENTRALITY percentile interval based on parent type
            for name in P:
                if name in parent_type:
                    conf['P'] = P[name]
            results = run_experiments(PI_admissions,
                                       pi_events,
                                       conf,
                                       join_rules,
                                       fname + f'_{__type}', f'PI: {__type}: ')
            if results['CENTRALITY_P'] is None:
                PI_events_P += PI_events_grouped[parent_type]
            elif conf['CENTRALITY_P_events']:
                pi_events = get_top_events(pi_events, results['CENTRALITY_P'], conf)
                PI_events_P += [e for e in pi_events if e['parent_type'] == parent_type]
            else:
                PI_events_P += [e for e in pi_events if e['type'] in results['ET_CENTRALITY_P']]
        else:
            PI_events_P += PI_events_grouped[parent_type]
    PI_events_P += PI_events_grouped[max_pi_stage]
    # sort by type, t and reindex
    PI_events_P = sort_and_index_events(PI_events_P)
    conf['CENTRALITY_path'] = True
    results = run_experiments(PI_admissions, PI_events_P, conf, join_rules, f'{fname}_results', 'All types')
    plot_CENTRALITY_and_BS(conn, conf, results['patient_CENTRALITY'], PI_hadms, PI_hadm_stage_t)


def MULTIMODAL_TEG_CENTRALITY_PI_NPI(conn, r, join_rules, conf, fname):
    print(conf)
    if conf['modality'] == 'parent_type':
        PI_events_grouped = group_events_by_parent_type(r['PI_events'])
        NPI_events_grouped = group_events_by_parent_type(['NPI_events'])
    elif conf['modality'] == 'event_type':
        PI_events_grouped = group_events_by_type(r['PI_events'])
        NPI_events_grouped = group_events_by_type(r['NPI_events'])
    print('PI groups', list(PI_events_grouped.keys()))
    print('NPI groups', list(NPI_events_grouped.keys()))
    PI_events_P = []
    NPI_events_P = []
    max_pi_stage = f"PI Stage {conf['max_pi_stage']}"
    for _type in PI_events_grouped:
        conf['vis'] = False
        conf['plot'] = False
        if _type not in NPI_events_grouped:
            continue
        if max_pi_stage in _type:
            continue
        if 'Admissions' in _type:
            PI_events_P += PI_events_grouped[_type]
            NPI_events_P += NPI_events_grouped[_type]
            continue
        if 'Braden' in _type: #'Braden Score' not in _type:
            #pprint.pprint(PI_events_grouped[_type])
            continue
        if 'CPT' in _type:
            #pprint.pprint(PI_events_grouped[_type])
            continue
        print('==================================================================')
        print('_type: ', _type)
        pi_count = len(PI_events_grouped[_type])
        npi_count = len(NPI_events_grouped[_type])
        print(pi_count, npi_count)
        __type = _type.replace('/', '_')
        if pi_count > conf['min_n'] and npi_count > conf['min_n']:
            pi_events = PI_events_grouped[_type] + PI_events_grouped[max_pi_stage]
            # sort by type, t and reindex
            pi_events = sort_and_index_events(pi_events)
            npi_events = NPI_events_grouped[_type] + NPI_events_grouped['Marker']
            # sort by type, t and reindex
            npi_events = sort_and_index_events(npi_events)
            '''
            # choose CENTRALITY percentile interval based on parent type
            for name in P:
                if name in _type:
                    conf['P'] = P[name]
            '''
            # PCENTRALITY Patient CENTRALITY
            pi_events, npi_events, PI_results, NPI_results = run_iterations(\
                                                       r['PI_admissions'],
                                                       r['NPI_admissions'],
                                                       pi_events,
                                                       npi_events,
                                                       conf,
                                                       join_rules,
                                                       fname + f'_{__type}',
                                                       _type,
                                                       vis_last_iter = False,
                                                       CENTRALITY_path_last_iter = False)
            if conf['modality'] == 'parent_type':
                PI_events_P += [e for e in pi_events if e['parent_type'] == _type]
                NPI_events_P += [e for e in npi_events if e['parent_type'] == _type]
            elif conf['modality'] == 'event_type':
                PI_events_P += [e for e in pi_events if e['event_type'] == _type]
                NPI_events_P += [e for e in npi_events if e['event_type'] == _type]
            if PI_results is None or NPI_results is None:
                continue
            plot_PI_NPI(PI_results, NPI_results, conf, nbins=30,
                        title=f'{_type}', fname=f"{fname}_{__type}_PI_NPI_Results")
            plot_PI_NPI(PI_results, NPI_results, conf, conf['P'],
                        nbins=10, title=f'{_type}', fname=f"{fname}_{__type}_PI_NPI_P_Results")
        else:
            PI_events_P += PI_events_grouped[_type]
            NPI_events_P += NPI_events_grouped[_type]
    PI_events_P += PI_events_grouped[max_pi_stage]
    # sort by type, t and reindex
    PI_events_P = sort_and_index_events(PI_events_P)

    NPI_events_P = NPI_events_P + NPI_events_grouped['Marker']
    # sort by type, t and reindex
    NPI_events_P = sort_and_index_events(NPI_events_P)
    '''
    if conf['CENTRALITY_P_events']:
        conf['CENTRALITY_path'] = True
    '''
    conf['vis'] = False
    conf['plot'] = False
    conf['P'] = conf['P_results']
    pi_events_P, npi_events_P, PI_results, NPI_results = run_iterations(\
                                                r['PI_admissions'],
                                                r['NPI_admissions'],
                                                PI_events_P,
                                                NPI_events_P,
                                                conf,
                                                join_rules,
                                                fname + '_ALL',
                                                'All types',
                                                True, True)
    plot_PI_NPI(PI_results, NPI_results, conf, nbins=30,
                title='All types', fname=f"{fname}_PI_NPI_Results")
    plot_PI_NPI(PI_results, NPI_results, conf, conf['P'],
                nbins=10, title='All types', fname=f"{fname}_PI_NPI_P_Results")
    plot_CENTRALITY_and_BS(conn, conf, PI_results['patient_CENTRALITY'], \
                    r['PI_hadms'], r['PI_hadm_stage_t'], fname)
    plot_PI_NPI_patients(r['PI_admissions'],
                         r['NPI_admissions'],
                         r['PI_df'],
                         r['NPI_df'],
                         conf,
                         pi_events_P,
                         npi_events_P,
                         PI_results,
                         NPI_results,
                         title=f"{conf['P_patients']}",
                         fname=f'{fname}_Patients_P')
    plot_patients(r['PI_admissions'],
                    r['PI_df'],
                    conf,
                    pi_events,
                    PI_results,
                    title=f"{conf['P_patients']}",
                    fname=f"{fname}_PI_Patients_P",
                    c='blue')
    plot_patients(r['NPI_admissions'],
                  r['NPI_df'],
                  conf,
                  npi_events,
                  NPI_results,
                  title=f"{conf['P_patients']}",
                  fname=f"{fname}_NPI_Patients_P",
                  c='red')
    

if __name__ == "__main__":
    #fname = 'output/MULTIMODAL_PI_ONLY'
    #MULTIMODAL_TEG_CENTRALITY_PI_ONLY(PI_RISK_EVENTS, join_rules, conf, fname)
    conn = get_db_connection()
    mp = [[67, 100], [34, 66], [0, 33]]
    os.mkdir(M_fname)
    r = admissions(conn, PI_RISK_EVENTS, M_join_rules, M_conf, fname=f'{M_fname}/TEG')
    _r = copy.deepcopy(r)
    for i in mp:
        conf_tmp = copy.deepcopy(M_conf)
        os.mkdir(f'{M_fname}/Multimodal-{i[0]}-{i[1]}')
        fname = f'{M_fname}/Multimodal-{i[0]}-{i[1]}/Multimodal-{i[0]}-{i[1]}'
        conf_tmp['missing_percent'] = i
        _r['PI_events'] = remove_by_missing_percent(r['PI_events'], conf_tmp)
        _r['NPI_events'] = remove_by_missing_percent(r['NPI_events'], conf_tmp)
        MULTIMODAL_TEG_CENTRALITY_PI_NPI(conn, _r, M_join_rules, conf_tmp, fname)
