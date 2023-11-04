import os
import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import warnings
warnings.filterwarnings('ignore')

from pygraphblas import *
options_set(nthreads=12)

from teg.admissions import admissions
from teg.event_setup import *
from teg.events import *
from teg.event_utils import remove_by_missing_percent
from teg.plot import *
from teg.plot_patients import *
from teg.run_experiments import *
from teg.queries import get_db_connection
from teg.schemas import *


# Experiment configuration
TEG_conf = {
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
    'missing_percent': [20, 40], # for mimic extract
    #'missing_percent': [0, 100], # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 2: 1},
    'PI_exclude_mid_stages': True,
    'PI_daily_max_stage': True,
    'PC_time_unit': timedelta(days=0, hours=1), # maximum PC per time unit
    'P': [90, 100], # PC percentile
    'P_results': [90, 100], # PC percentile
    'P_remove': False,
    'P_patients': [60, 100],
    'ET_PC_min_freq': 0,
    'PC_path': False,
    'P_max_n': False,
    'path_percentile': [80, 100],
    'PI_sql': 'one', #one, multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_PC': False, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': True, # Use a list of vitals related to PI
    'skip_repeat': False,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'quantile_round': 2,
    'drug_percentile': [40, 60],
    'input_percentile': [40, 80],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': 30,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'n_patient_paths': [1, 2, 3], # n highest PC paths of a patient
    'vis': False,
    'plot': False,
    'plot_types_n': 20,
    'PC_BS_nnz': 0, # in percentage
    'first_hadm': True,
    'dbsource': 'carevue', # carevue or False
    'iterations': 20,
    'iter_type': 'event_type_PC', # event_type_PC or event_PC
    'PC_P_events': False,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        #'admission_type',
        'insurance',
        'los',
        'age',
        'oasis'] 
}

# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=5, hours=0),
    'Discharges': timedelta(days=5, hours=0),
    'ICU In': timedelta(days=5, hours=0),
    'ICU Out': timedelta(days=5, hours=0),
    'Callout': timedelta(days=5, hours=0),
    'Transfer In': timedelta(days=5, hours=0),
    'Transfer Out': timedelta(days=5, hours=0),
    'CPT': timedelta(days=5, hours=0),
    'Presc': timedelta(days=5, hours=0),
    'Services': timedelta(days=5, hours=0),
    'other': timedelta(days=5, hours=0),
    'diff_type_same_patient': timedelta(days=5, hours=0),
    'PI': timedelta(days=5, hours=0),
    'Braden': timedelta(days=5, hours=0),
    'Input': timedelta(days=5, hours=0),
    'Intervention': timedelta(days=5, hours=0),
    'Vitals/Labs': timedelta(days=5, hours=0),
}

TEG_join_rules = {
    'IDs': EVENT_IDs if not TEG_conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': TEG_conf['duration'],
    'duration_similarity': timedelta(days=5),
    'sequential_join': True,
    'max_pi_state': 1,
    'max_pi_stage': 2,
    'include_numeric': True
}

TEG_fname = f'output/TEG-PI-ONLY'


def TEG_PC_PI_ONLY(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    PI_df, patients = get_patient_demography(conn, conf) 
    print('Patients', len(patients))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    all_events = events(conn, event_list, conf, PI_hadms)
    all_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    PI_hadms = tuple(list(PI_hadm_stage_t.keys()))
    conf['vis'] = True
    conf['plot'] = True
    conf['PC_path'] = True
    results = run_experiments(patients, all_events, conf, join_rules, fname)
    #plot_PC_and_BS(conn, conf, results['patient_PC'], PI_hadms, PI_hadm_stage_t)


if __name__ == "__main__":
    os.mkdir(TEG_fname)
    fname = f'{TEG_fname}/TEG-PI-ONLY'
    TEG_PC_PI_ONLY(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
