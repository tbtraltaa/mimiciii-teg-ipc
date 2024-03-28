import os
import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import warnings
warnings.filterwarnings('ignore')

from pygraphblas import *
options_set(nthreads=12)

from mimiciii_teg.queries.admissions import admissions
from mimiciii_teg.schemas.event_setup import *
from mimiciii_teg.teg.events import *
from mimiciii_teg.utils.event_utils import remove_by_missing_percent
from mimiciii_teg.vis.plot import *
from mimiciii_teg.vis.plot_patients import *
from mimiciii_teg.queries.queries import get_db_connection
from mimiciii_teg.schemas.schemas import *
from run_experiments import *
from MULTIMODAL_experiments import *
from TEG_experiments import *


TEG_M_fname = 'output/TEG_Multimodal'

def top_events_experiment():
    mp = [[0, 100]]
    remove = [False]
    conf_teg = copy.deepcopy(TEG_conf)
    conf_teg['P_remove'] = False
    conf_teg['missing_percent'] = [0, 100]
    TEG_CENTRALITY_PI_NPI(conn, r, TEG_join_rules, conf_teg, fname_teg)
    conf_teg = copy.deepcopy(TEG_conf)
    conf_multimodal = copy.deepcopy(M_conf)
    conf_multimodal['P_remove'] = False
    conf_multimodal['missing_percent'] = [0, 100]
    MULTIMODAL_TEG_CENTRALITY_PI_NPI(conn, r, M_join_rules, conf_multimodal, fname_multimodal)


if __name__ == "__main__":
    conn = get_db_connection()
    #mp = [[64, 100], [30, 70], [0, 36], [0, 100]]
    mp = [[80, 100], [60, 85], [40, 65], [20, 45], [0, 25], [0, 100]]
    remove = [False, False, False, False, False, False]
    P = [[95, 100], [95, 100], [95, 100], [95, 100], [97, 100], [97, 100]]
    os.mkdir(TEG_M_fname)
    r = admissions(conn, PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname=f'{TEG_M_fname}/TEG_M')
    _r = copy.deepcopy(r)
    j = 0
    for i, i_r in zip(mp, remove):
        conf_teg = copy.deepcopy(TEG_conf)
        #conf_multimodal = copy.deepcopy(M_conf)
        os.mkdir(f'{TEG_M_fname}/TEG-{i[0]}-{i[1]}')
        #os.mkdir(f'{TEG_M_fname}/Multimodal-{i[0]}-{i[1]}')
        fname_teg = f'{TEG_M_fname}/TEG-{i[0]}-{i[1]}/TEG-{i[0]}-{i[1]}'
        #fname_multimodal = f'{TEG_M_fname}/Multimodal-{i[0]}-{i[1]}/Multimodal-{i[0]}-{i[1]}'
        conf_teg['missing_percent'] = i
        #conf_multimodal['missing_percent'] = i
        conf_teg['P_remove'] = i_r
        conf_teg['P'] = P[j]
        conf_teg['ET_P'] = P[j]
        #conf_multimodal['P_remove'] = i_r
        _r['PI_events'] = remove_by_missing_percent(r['PI_events'], conf_teg)
        _r['NPI_events'] = remove_by_missing_percent(r['NPI_events'], conf_teg)
        TEG_CENTRALITY_PI_NPI(conn, _r, TEG_join_rules, conf_teg, fname_teg)
        #MULTIMODAL_TEG_CENTRALITY_PI_NPI(conn, _r, M_join_rules, conf_multimodal, fname_multimodal)
        j += 1
