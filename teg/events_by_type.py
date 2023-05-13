import sys
import pandas as pd
import numpy as np
from itertools import groupby
import pprint
from datetime import timedelta
import copy
import warnings
warnings.filterwarnings('ignore')

from teg.schemas import *
from teg.schemas_PI import *
from teg.schemas_chart_events import *
from teg.eventgraphs import *
from teg.queries_mimic_extract import *
from teg.queries_chart_events import *
from teg.queries import *

def events_by_type(conn, event_name, conf, hadms=()):
    if event_name in EVENTS:
        event_name, table, time_col, main_attr = EVENTS[event_key]
        events = get_events(conn, event_key, conf, hadms)
    if event_name in PI_EVENTS:
        if not conf['include_numeric'] and event_name in PI_EVENTS_NUMERIC:
            return []
        events = get_chart_events(conn, event_name, conf, hadms)
    if event_name in CHART_EVENTS:
        if not conf['include_numeric'] and event_name in CHART_EVENTS_NUMERIC:
            return []
        events = get_chart_events(conn, event_name, conf, hadms)
    if event_name == 'Interventions':
        events = get_events_interventions(conn, conf, hadms)
    if event_name == 'Vitals/Labs':
        if conf['vitals_X_mean']:
            events = get_events_vitals_X_mean(conn, conf, hadms)
        else:
            events = get_events_vitals_X(conn, conf, hadms)
    for i, e in enumerate(events):
        e['i'] = i
    return events
