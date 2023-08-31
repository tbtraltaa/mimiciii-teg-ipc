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
from MULTIMODAL_experiments import *
from TEG_experiments import *


TEG_M_fname = 'output/TEG_Multimodal'


if __name__ == "__main__":
    #fname = 'output/TEG-PI-ONLY'
    #TEG_PC_PI_ONLY(PI_RISK_EVENTS, join_rules, conf, fname)
    conn = get_db_connection()
    mp = [[67, 100], [34, 66], [0, 33]]
    remove = [False, [0, 20], [0, 20]]
    os.mkdir(TEG_M_fname)
    r = admissions(conn, PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname=f'{TEG_M_fname}/TEG_M')
    _r = copy.deepcopy(r)
    for i, i_r in zip(mp, remove):
        conf_teg = copy.deepcopy(TEG_conf)
        conf_multimodal = copy.deepcopy(M_conf)
        os.mkdir(f'{TEG_M_fname}/TEG-{i[0]}-{i[1]}')
        os.mkdir(f'{TEG_M_fname}/Multimodal-{i[0]}-{i[1]}')
        fname_teg = f'{TEG_M_fname}/TEG-{i[0]}-{i[1]}/TEG-{i[0]}-{i[1]}'
        fname_multimodal = f'{TEG_M_fname}/Multimodal-{i[0]}-{i[1]}/Multimodal-{i[0]}-{i[1]}'
        conf_teg['missing_percent'] = i
        conf_multimodal['missing_percent'] = i
        conf_teg['P_remove'] = i_r
        conf_multimodal['P_remove'] = i_r
        _r['PI_events'] = remove_by_missing_percent(r['PI_events'], conf_teg)
        _r['NPI_events'] = remove_by_missing_percent(r['NPI_events'], conf_teg)
        TEG_PC_PI_NPI(conn, _r, TEG_join_rules, conf_teg, fname_teg)
        MULTIMODAL_TEG_PC_PI_NPI(conn, _r, M_join_rules, conf_multimodal, fname_multimodal)
