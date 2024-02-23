import numpy as np
import pandas as pd
import sys
import pprint
from datetime import timedelta
import copy
import warnings

warnings.filterwarnings('ignore')


from mimiciii_teg.schemas.schemas_PI import *
from mimiciii_teg.schemas.schemas_chart_events import *
from mimiciii_teg.queries.queries_utils import *
from mimiciii_teg.utils.utils import get_quantile, get_quantile_interval
from mimiciii_teg.queries.queries import *

# an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]

def get_chart_events(conn, event_name, conf, hadms=()):
    '''
    Returns PI events within the given time window.
    '''
    schema = 'mimiciii'
    table = 'chartevents'
    time_col = 'charttime'
    labels = []
    ignored_values = []
    if event_name in PI_EVENTS_CV:
        label_CV, ignored_values_CV = PI_EVENTS_CV[event_name]
        labels.append(label_CV)
        ignored_values += ignored_values_CV
    if event_name in PI_EVENTS_MV:
        label_MV, ignored_values_MV = PI_EVENTS_MV[event_name]
        labels.append(label_MV)
        ignored_values += ignored_values_MV
    if event_name in CHART_EVENTS:
        labels = [CHART_EVENTS[event_name][0]]
    #if event_name == 'PI Stage':
    #    ignored_values += [k for k, v in PI_STAGE_MAP.items() if v not in conf['PI_states']]

    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id, "
    cols += f"tb.subject_id, tb.hadm_id, "
    cols += f"'{event_name}' as type, "
    cols += f"tb.{time_col} - a.admittime as t, "
    cols += f"tb.{time_col} as datetime, "
    cols += " 0 as pi_stage, "
    cols += f"EXTRACT(DAY FROM tb.{time_col} - a.admittime) as day,"
    if event_name in PI_EVENTS_NUMERIC or event_name in CHART_EVENTS_NUMERIC:
        cols += f"split_part(tb.value, ' ', 1) as value, " 
        if conf['unique_chartvalue_per_day_sql']:
            # Take max value events for a day
            # The number of events match the results from pandas, but
            # the event values could be different due to the casting.
            # TEG connections differed from using pandas.
            # Hence, not using this option
            cols += f''' row_number() over (partition by 
                tb.subject_id, tb.hadm_id,
                EXTRACT(DAY FROM tb.{time_col} - a.admittime)
                order by cast(split_part(tb.value, ' ', 1) as double precision) DESC), '''
    else:
        cols += f"tb.value as value, "
        if conf['unique_chartvalue_per_day_sql']:
            # Take unique value events for a day
            # Ignoring PI numbers
            cols += f''' row_number() over (partition by 
                tb.subject_id, tb.hadm_id, tb.value,
                EXTRACT(DAY FROM tb.{time_col} - a.admittime)
                order by tb.charttime), '''
    if event_name in PI_EVENTS:
        cols += f"regexp_replace(d.label, '\\D+', '', 'g') as pi_number, "
    cols += f"tb.icustay_id, d.dbsource "  # extra info
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    table += f' INNER JOIN {schema}.d_items d'
    table += f' ON tb.itemid = d.itemid'
    where = 'tb.hadm_id is NOT NULL'
    if len(hadms) != 0:
        where += f' AND tb.hadm_id IN {hadms}'
    else:
        if conf['has_icustay'] == 'True':
            table += f' INNER JOIN {schema}.icustays icu ON tb.hadm_id=icu.hadm_id'
            where += f''' AND icu.intime - a.admittime <= '{conf['max_hours']} hours'
                AND icu.intime >= a.admittime
                '''
        elif conf['has_icustay'] == 'False':
            table += f' LEFT JOIN {schema}.icustays icu ON tb.hadm_id=icu.hadm_id'
            where += f' AND icu.hadm_id IS NULL'
        # Exclude patients with PI staging within 24 hours after admission
        pi24_hadms = PI_hadms_24h(conn, conf)
        where += f' AND tb.hadm_id NOT IN {pi24_hadms}'
    where += f'''
        AND a.diagnosis != 'NEWBORN'
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) >= {conf['min_age']}
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) <= {conf['max_age']}
        AND tb.{time_col} - a.admittime <= '{conf['max_hours']} hours'
        AND a.dischtime - a.admittime >= '{conf['min_los_hours']} hours'
        '''
    if conf['starttime'] and conf['endtime']:
        where += f" AND tb.{time_col} >= '{conf['starttime']}'"
        where += f" AND tb.{time_col} <= '{conf['endtime']}'"
        where += f" AND a.admittime >= '{conf['starttime']}'"
        where += f" AND a.admittime < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND tb.{time_col} >= a.admittime'
    where += f' AND tb.{time_col} <= a.dischtime'
    where += f' AND tb.value is NOT NULL'
    where += f" AND coalesce(TRIM(tb.value), '') != ''"
    if event_name in PI_EVENTS_NUMERIC or event_name in CHART_EVENTS_NUMERIC:
        where += f'''
            AND split_part(TRIM(tb.value), ' ', 1) != '0'
            AND split_part(TRIM(tb.value), ' ', 1) != '0.0'
            AND split_part(TRIM(tb.value), ' ', 1) similar to '\+?\d*\.?\d*'
            ''' 
    where += f" AND (d.label similar to '{labels[0]}'"
    if len(labels) > 1:
        for label in labels[1:]:
            where += f" OR d.label SIMILAR TO '{label}'"
    where += ")"
    for value in ignored_values:
        where += f" AND TRIM(tb.value) not similar to '{value}'"
    order_by = f'ORDER BY t ASC'
    if conf['unique_chartvalue_per_day_sql']:
        query = f"SELECT * FROM (SELECT {cols} FROM {table} WHERE {where}) AS tmp"
        query += f" WHERE row_number=1 {order_by}"
        df = pd.read_sql_query(query, conn)
        df.drop(['row_number'], axis=1, inplace=True)
    else:
        query = f"SELECT {cols} FROM {table} WHERE {where} {order_by}"
        df = pd.read_sql_query(query, conn)

    UOM = ""
    if event_name in PI_EVENTS_NUMERIC or event_name in CHART_EVENTS_NUMERIC:
        if event_name in PI_EVENTS_NUMERIC:
            UOM = 'cm'
            if PI_EVENTS_NUMERIC[event_name]['dtype']:
                df['value'] = df['value'].astype(PI_EVENTS_NUMERIC[event_name]['dtype'])
            df = df[df['value'] > 0]
            Q  = query_quantiles(conn, conf['quantiles'], event_name, **PI_EVENTS_NUMERIC[event_name]) 
        else:
            UOM = ""
            if CHART_EVENTS_NUMERIC[event_name]['dtype']:
                df['value'] = df['value'].astype(CHART_EVENTS_NUMERIC[event_name]['dtype'])
                df = df[df['value'] > 0]
            Q  = query_quantiles(conn, conf['quantiles'], event_name, **CHART_EVENTS_NUMERIC[event_name]) 
        if conf['unique_chartvalue_per_day']:
            # gets the maximum value per day
            df['row_number'] = df.sort_values(['value', 't'], ascending=[False, True]) \
                 .groupby(['id', 'day']) \
                 .cumcount() + 1
            df = df[df['row_number'] == 1]
            df.drop(['row_number'], axis=1, inplace=True)
        df['numeric_value'] = df['value']
        df['value'] = df['value'].apply(lambda x: get_quantile(x, Q))
        df['Q_I'] = df['value'].apply(lambda x: get_quantile_interval(x, Q))
        df.drop(['day'], axis=1, inplace=True)
    elif conf['unique_chartvalue_per_day']:
        df['row_number'] = df.sort_values(['t'], ascending=[True]) \
             .groupby(['id', 'day', 'value']) \
             .cumcount() + 1
        df = df[df['row_number'] == 1]
        df.drop(['row_number'], axis=1, inplace=True)
        df.drop(['day'], axis=1, inplace=True)
    # value is used for event comparision
    # pi_stage is PI Stage of PI Stage event.
    #if event_name in PI_EVENTS_NUMERIC:
    #    df['value'] = df['value'].apply(lambda x: float(x.strip().split()[0]))
    if event_name == 'PI Stage':
        df['value'] = df['value'] \
            .apply(lambda x: PI_STAGE_MAP[x])
        df['pi_stage'] = df['value']
    elif event_name in PI_VALUE_MAP:
        df['value'] = df['value'] \
                                .apply(lambda x: PI_VALUE_MAP[event_name][x.strip()])
    elif event_name in CHART_VALUE_MAP:
        df['value'] = df['value'] \
                                .apply(lambda x: CHART_VALUE_MAP[event_name][x.strip()])
    else:
        df['value'] = df['value'].apply(lambda x: x.strip().title())
    if event_name == 'PI Skin Type':
        df[df['value'] == 'Subq Emphysema']['value'] = 'Sub Q Emphysema'
    if event_name == 'PI Stage':
        df['parent_type'] = df['type'] + ' ' + df['value'].astype(str)
        df['event_type'] = df['type'] + ' ' + df['value'].astype(str)
    else:
        df['parent_type'] = df['type']
        df['event_type'] = df['type']
    # Decoupling events by their value
    if UOM:
        df['type'] = df['type'] + ' ' + df['value'].astype(str) + ' ' + UOM
    else:
        df['type'] = df['type'] + ' ' + df['value'].astype(str)
    # each row is converted into a dictionary indexed by column names
    events = df.to_dict('records')
    # Taking max PI stage per day
    if 'PI Stage' in event_name and conf['PI_daily_max_stage']:
        # take maximum stage of a day
        sorted_events = sorted(events, key=lambda x: (x['id'], x['t'].days, -x['pi_stage']))
        id_ = None
        day = None
        events_daily_max = []
        for e in sorted_events:
            if e['id'] != id_ or e['t'].days!= day:
                events_daily_max.append(e)
                id_ = e['id']
                day = e['t'].days
        events = events_daily_max
    # old code. it assumes only querying PI events
    if conf['duration']:
        df['duration'] = timedelta(days=0)
        # each row is converted into a dictionary indexed by column names
        events = df.to_dict('records')
        # calculating event duration for each PI
        events.sort(key=lambda x: (x['id'], x['pi_number'], x['value'], x['t']))
        key = 'value'
        prev_val = None
        prev_t = np.inf
        prev_id = None
        prev_num = None
        start = None
        t_unit = timedelta(days=1)
        del_list = []
        for i, e in enumerate(events):
            if e[key] == prev_val and \
                e['t'].days - prev_t == 1 and \
                e['id'] == prev_id and \
                e['pi_number'] == prev_num:
                events[start]['duration'] += t_unit 
                prev_t = e['t'].days
                del_list.append(i)
            else:
                start = i
                prev_val = e[key]
                prev_t = e['t'].days
                prev_id = e['id']
                prev_num = e['pi_number']
        del_count=0
        for i in del_list:
            del events[i - del_count]
            del_count += 1
    events = sorted(events, key = lambda x: (x['type'], x['t']))
    return events
