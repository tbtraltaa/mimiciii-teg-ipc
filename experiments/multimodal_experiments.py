import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import time
import warnings
warnings.filterwarnings('ignore')

from pygraphblas import *
options_set(nthreads=16)


from teg.schemas import *
from teg.event_setup import *
from teg.events import *
from teg.eventgraphs import *
from teg.apercolation import *
from teg.PC_utils import *
from teg.graph_vis import *
from teg.build_graph import *
from teg.paths import *
from teg.plot import *
from teg.psm import *
from teg.pca import *
from teg.event_utils import *
from teg.run_experiments import *
from teg.PI_risk_factors import *

# Experiment configuration
# PC_PERCENTILE = {<EVENT TYPE>: 
PC_PERCENTILE = {
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

conf = {
    'patient_history': timedelta(weeks=24), # 6 months
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
    'min_missing_percent': 0, # for mimic extract
    'PI_exclude_mid_stages': True,
    'PC_time_unit': timedelta(days=0, hours=1), # maximum PC per time unit
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 2: 1},
    'max_pi_stage':2,
    'PC_percentile': [97, 100],
    'PC_percentile_max_n': False,
    'path_percentile': [95, 100],
    'PI_sql': 'one', #multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_only': False, # Delete non PI patients after querying all events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_PC': True, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': True, # Use a list of vitals related to PI
    'skip_repeat': True,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 70],
    'input_percentile': [20, 90],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': 100,
    'min_n': 500,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'PC_path': False,
    'n_patient_paths': [1, 3, 5], # n highest PC paths of a patient
    'vis': False,
    'plot': False,
    'PC_BS_nnz': 0, # in percentage
    'first_hadm': True,
    'dbsource': 'carevue', # carevue or False
    'iterations': 10,
    'PC_P_events': True,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        'admission_type',
        'insurance',
        'los',
        'age',
        'oasis']
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
    'max_pi_state':1,
    'max_pi_stage':2,
    'include_numeric': True
}

def MULTIMODAL_TEG_PC_PI_ONLY(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    PI_df, PI_admissions = get_patient_demography(conn, conf) 
    print('Admissions', len(PI_admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    PI_events = events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    PI_events_grouped = group_events_by_parent_type(PI_events)
    PI_events_P = []
    max_pi_stage = f"PI Stage {conf['max_pi_stage']}"
    for parent_type in PI_events_grouped:
        if max_pi_stage in parent_type:
            continue
        print('==================================================================')
        print('Parent_type: ', parent_type)
        p_type = parent_type.split('/')[0]
        pi_count = len(PI_events_grouped[parent_type])
        if pi_count > conf['min_n']:
            pi_events = PI_events_grouped[parent_type] + PI_events_grouped[max_pi_stage]
            # sort by type, t and reindex
            pi_events = sort_and_index_events(pi_events)
            # choose PC percentile interval based on parent type
            for name in PC_PERCENTILE:
                if name in parent_type:
                    pass
                    #conf['PC_percentile'] = PC_PERCENTILE[name]
            PI_PC_all, PI_PC_nz, PI_PC_P, patient_PC = run_experiments(PI_admissions,
                                                           pi_events,
                                                           conf,
                                                           join_rules,
                                                           fname + f'_PI_ONLY_{p_type}', f'PI: {p_type}: ')
            if PI_PC_P is None:
                PI_events_P += PI_events_grouped[parent_type]
            else:
                pi_events = get_top_events(pi_events, PI_PC_P, conf)
                PI_events_P += [e for e in pi_events if e['parent_type'] == parent_type]
        else:
            PI_events_P += PI_events_grouped[parent_type]
    PI_events_P += PI_events_grouped[max_pi_stage]
    # sort by type, t and reindex
    PI_events_P = sort_and_index_events(PI_events_P)
    conf['PC_percentile'] = [90, 100]
    conf['PC_percentile_max_n'] = False
    conf['PC_path'] = False
    PI_PC_all, PI_PC_nz, PI_PC_P, patient_PC = \
        run_experiments(PI_admissions, PI_events_P, conf, join_rules, f'{fname}_PI_ONLY', 'All types')
    #braden_events = get_chart_events(conn, 'Braden Score', conf, PI_hadms)
    #braden_events = remove_events_after_t(braden_events, PI_hadm_stage_t)
    braden_events = [e for e in PI_events if 'Braden Score' in e['type']]
    patient_BS = get_patient_max_Braden_Scores(braden_events, conf['PC_time_unit'])
    plot_time_series(patient_PC, patient_BS, conf)
    plot_time_series_average(patient_PC, patient_BS, conf)
    conf['PC_BS_nnz'] = 0.05
    plot_time_series(patient_PC, patient_BS, conf)
    plot_time_series_average(patient_PC, patient_BS, conf)
    for idd in patient_PC:
        if idd in patient_BS:
            print(idd, len(patient_PC[idd]['PC']), len(patient_BS[idd]['BS']))
        else:
            print(idd, len(patient_PC[idd]['PC']))


def MULTIMODAL_TEG_PC_PI_NPI(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    # PI patients
    PI_df, PI_admissions = get_patient_demography(conn, conf) 
    print('PI Patients', len(PI_admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    PI_events = events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    # Remove invalid admissions
    PI_df = PI_df[PI_df['hadm_id'].isin(list(PI_hadm_stage_t.keys()))]
    PI_df = PI_df[conf['psm_features']]
    print(PI_df.columns)
    # Non PI patients
    conf['PI_sql'] = 'no_PI_events'
    conf['hadm_limit'] = conf['NPI_hadm_limit']
    NPI_df, NPI_admissions = get_patient_demography(conn, conf) 
    NPI_df = NPI_df[conf['psm_features']]
    print('Non PI Patients', len(NPI_admissions))
    # Check if PI and NPI admissions intersect
    int_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print('Intersection of PI and NPI patients', len(int_df))
    print(int_df)
    # Exclude the intersection if exists
    NPI_df = NPI_df.loc[~NPI_df['hadm_id'].isin(PI_df['hadm_id'])]
    int_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print(len(int_df))
    print(int_df)
    # Label admissions as PI or Non PI
    NPI_df['PI'] = 0
    PI_df['PI'] = 1
    df = pd.concat([PI_df, NPI_df])
    # Propensity Score matrching
    psm = get_psm(df, conf, fname)
    PI_hadms = tuple(psm.matched_ids['hadm_id'].tolist())
    if len(PI_hadm_stage_t) != len(PI_hadms):
        PI_events = events(conn, event_list, conf, PI_hadms)
        PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    PI_events_grouped = group_events_by_parent_type(PI_events)

    NPI_t = dict()
    NPI_hadms = list()
    PI_NPI_match = dict()
    for i, row in psm.matched_ids.iterrows():
        if row['hadm_id'] in PI_hadm_stage_t:
            NPI_t[row['matched_ID']] = PI_hadm_stage_t[row['hadm_id']]
            NPI_hadms.append(row['matched_ID'])
            PI_NPI_match[row['hadm_id']] = row['matched_ID']
    NPI_hadms = tuple(NPI_hadms)
    NPI_events = events(conn, event_list, conf, NPI_hadms)
    NPI_events = process_events_NPI(NPI_events, NPI_t, conf)
    NPI_events_grouped = group_events_by_parent_type(NPI_events)
    print('PI groups', list(PI_events_grouped.keys()))
    print('NPI groups', list(NPI_events_grouped.keys()))
    PI_events_P = []
    NPI_events_P = []
    max_pi_stage = f"PI Stage {conf['max_pi_stage']}"
    for parent_type in PI_events_grouped:
        conf['vis'] = False
        conf['plot'] = False
        if parent_type not in NPI_events_grouped:
            continue
        if max_pi_stage in parent_type:
            continue
        if 'Admissions' in parent_type:
            PI_events_P += PI_events_grouped[parent_type]
            NPI_events_P += NPI_events_grouped[parent_type]
            continue
        if 'Braden' in parent_type: #'Braden Score' not in parent_type:
            #pprint.pprint(PI_events_grouped[parent_type])
            continue
        if 'CPT' in parent_type:
            #pprint.pprint(PI_events_grouped[parent_type])
            continue
        print('==================================================================')
        print('Parent_type: ', parent_type)
        p_type = parent_type.split('/')[0]
        pi_count = len(PI_events_grouped[parent_type])
        npi_count = len(NPI_events_grouped[parent_type])
        if pi_count > conf['min_n'] and npi_count > conf['min_n']:
            pi_events = PI_events_grouped[parent_type] + PI_events_grouped[max_pi_stage]
            # sort by type, t and reindex
            pi_events = sort_and_index_events(pi_events)
            npi_events = NPI_events_grouped[parent_type] + NPI_events_grouped['Marker']
            # sort by type, t and reindex
            npi_events = sort_and_index_events(npi_events)
            # choose PC percentile interval based on parent type
            for name in PC_PERCENTILE:
                if name in parent_type:
                    conf['PC_percentile'] = PC_PERCENTILE[name]
            # PPC Patient PC
            pi_events, npi_events, pi_PPC, npi_PPC = run_iterations(PI_admissions,
                                                       NPI_admissions,
                                                       pi_events,
                                                       npi_events,
                                                       conf,
                                                       join_rules,
                                                       fname + f'_{p_type}',
                                                       p_type,
                                                       True)
            PI_events_P += [e for e in pi_events if e['parent_type'] == parent_type]
            NPI_events_P += [e for e in npi_events if e['parent_type'] == parent_type]
        else:
            PI_events_P += PI_events_grouped[parent_type]
            NPI_events_P += NPI_events_grouped[parent_type]
    PI_events_P += PI_events_grouped[max_pi_stage]
    # sort by type, t and reindex
    PI_events_P = sort_and_index_events(PI_events_P)

    NPI_events_P = NPI_events_P + NPI_events_grouped['Marker']
    # sort by type, t and reindex
    NPI_events_P = sort_and_index_events(NPI_events_P)
    conf['PC_percentile'] = [95, 100]
    conf['PC_percentile_max_n'] = False
    '''
    if conf['PC_P_events']:
        conf['PC_path'] = True
    '''
    #pprint.pprint(PI_events_grouped[max_pi_stage])
    #print('Number of Stage 2', len(PI_events_grouped[max_pi_stage]))

    conf['vis'] = False
    conf['plot'] = False
    pi_events_P, npi_events_P, pi_PPC, npi_PPC = run_iterations(PI_admissions,
                                                NPI_admissions,
                                                PI_events_P,
                                                NPI_events_P,
                                                conf,
                                                join_rules,
                                                fname + '_ALL_97_100',
                                                'All types',
                                                True, True)
    '''
    conf['PC_percentile'] = [95, 100]
    pi_events_P, npi_events_P, pi_PPC, npi_PPC = run_iterations(PI_admissions,
                                                    NPI_admissions,
                                                    PI_events_P,
                                                    NPI_events_P,
                                                    conf,
                                                    join_rules,
                                                    fname + '_ALL_95_100',
                                                    'All types',
                                                    True)
    conf['PC_percentile'] = [97, 100]
    pi_events_P, npi_events_P, pi_PPC, npi_PPC = run_iterations(PI_admissions,
                                                    NPI_admissions,
                                                    PI_events_P,
                                                    NPI_events_P,
                                                    conf,
                                                    join_rules,
                                                    fname + '_ALL_97_100',
                                                    'All types',
                                                    True)
    '''
    #braden_events = [e for e in PI_events if 'Braden Score' in e['type']]
    braden_events = get_chart_events(conn, 'Braden Score', conf, PI_hadms)
    print(len(braden_events))
    braden_events = remove_events_after_t(braden_events, PI_hadm_stage_t)
    #braden_events = [e for e in PI_events if 'Braden Score' in e['type']]
    patient_BS = get_patient_max_Braden_Scores(braden_events, conf['PC_time_unit'])
    plot_time_series(pi_PPC, patient_BS, conf)
    plot_time_series_average(pi_PPC, patient_BS, conf)
    conf['PC_BS_nnz'] = 0.05
    plot_time_series(pi_PPC, patient_BS, conf)
    plot_time_series_average(pi_PPC, patient_BS, conf)
    for idd in pi_PPC:
        if idd in patient_BS:
            print(idd, len(pi_PPC[idd]['PC']), len(patient_BS[idd]['BS']))
        else:
            print(idd, len(pi_PPC[idd]['PC']))

if __name__ == "__main__":
    fname = 'output/MULTIMODAL_PI_NPI_RISK'
    '''
    fname_keys = [
    ]
    fname += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf.items() if k in fname_keys])
    fname += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules.items()
                                    if k in fname_keys])
    '''
    MULTIMODAL_TEG_PC_PI_NPI(PI_RISK_EVENTS, join_rules, conf, fname)
    #MULTIMODAL_TEG_PC_PI_ONLY(EVENTS, join_rules, conf, fname)