import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings


warnings.filterwarnings('ignore')

from mimiciii_teg.schemas.schemas import *
from mimiciii_teg.schemas.schemas_PI import *
from mimiciii_teg.queries.queries_utils import *
from mimiciii_teg.utils.utils import *
from mimiciii_teg.schemas.PI_risk_factors import *

# Database configuration
username = 'postgres'
password = 'postgres'
dbname = 'mimic'
schema = 'mimiciii'


def get_db_connection():
    # Connect to MIMIC-III database
    return psycopg2.connect(dbname=dbname, user=username, password=password)

# a patient query: [<id>, **<demography_attributes>]
# a patient dict: {<id>: **<demography_attributes>}


def get_patient_demography(conn, conf, hadms=()):
    '''
    Returns patients with demography between the given time window.
    '''
    cols = f"CONCAT(p.subject_id, '-', a.hadm_id) as id, p.subject_id, a.hadm_id, "
    for table in PATIENTS:
        for col in PATIENTS[table]:
            cols += f'{table}.{col}, '
    cols += 'EXTRACT(DAY FROM a.dischtime - a.admittime) as los, '
    cols += 'EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) as age, '
    cols += 'a.admittime, o.oasis '
    table = f'''{schema}.admissions a INNER JOIN {schema}.patients p
            ON a.subject_id = p.subject_id
            INNER JOIN {schema}.oasis o
            ON a.hadm_id = o.hadm_id
            '''
    where = f''' a.diagnosis != 'NEWBORN'
        AND a.admission_type != 'NEWBORN'
        AND a.hadm_id is NOT NULL
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) >= {conf['min_age']}
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) <= {conf['max_age']}
        AND a.dischtime - a.admittime >= '{conf['min_los_hours']} hours'
        '''
    if conf['admission_type']:
        where += f" AND a.admission_type='{conf['admission_type']}'" 
    if conf['has_icustay'] == 'True':
        table += f' INNER JOIN {schema}.icustays icu ON a.hadm_id=icu.hadm_id'
        where += f''' AND icu.intime - a.admittime <= '{conf['max_hours']} hours'
            AND icu.intime >= a.admittime
            '''
    elif conf['has_icustay'] == 'False':
        table += f' LEFT JOIN {schema}.icustays icu ON a.hadm_id=icu.hadm_id'
        where += f' AND icu.hadm_id IS NULL'
    # survived patients
    where += ' AND a.hospital_expire_flag=0'
    limit = ""
    if hadms:
        where += f' AND a.hadm_id IN {hadms}'
        limit = ""
    else:
        if conf['PI_sql'] == 'PI_or_NPI':
            # Exclude patients with PI staging within 24 hours after admission
            pi24_hadms = PI_hadms_24h(conn, conf)
            where += f' AND a.hadm_id NOT IN {pi24_hadms}'
        elif conf['PI_sql'] == 'one_or_multiple':
            pi24_hadms = PI_hadms_24h(conn, conf)
            where += f' AND a.hadm_id NOT IN {pi24_hadms}'
            ignored_values_stage = []
            label_CV_stage, ignored_values_CV_stage = PI_EVENTS_CV['PI Stage']
            ignored_values_stage += ignored_values_CV_stage
            label_MV_stage, ignored_values_MV_stage = PI_EVENTS_MV['PI Stage']
            ignored_values_stage += ignored_values_MV_stage
            pi_where = f't1.value is NOT NULL AND t1.itemid in {PI_STAGE_ITEMIDS}'
            for value in ignored_values_stage:
                pi_where += f" AND t1.value not similar to '{value}'"
            pi_where += " AND ("
            for i, value in enumerate(STAGE_PI_MAP[max(conf['PI_states'].keys())]):
                if i == 0:
                    pi_where += f" t1.value similar to '{value}'"
                else:
                    pi_where += f" OR t1.value similar to '{value}'"
            pi_where += ")"
            if conf['starttime'] and conf['endtime']:
                pi_where += f" AND t1.charttime >= '{conf['starttime']}'"
                pi_where += f" AND t1.charttime <= '{conf['endtime']}'"
            pi_where += f" AND t1.charttime - t2.admittime <='{conf['max_hours']} hours'"
            pi = f'''(SELECT DISTINCT t1.hadm_id, t2.admittime from {schema}.chartevents t1
                    INNER JOIN {schema}.admissions t2
                    ON t1.hadm_id=t2.hadm_id
                '''
            if conf['dbsource']:
                pi += f' INNER JOIN {schema}.d_items d ON t1.itemid=d.itemid'
                pi_where += f" AND d.dbsource='{conf['dbsource']}'"
            pi += f' WHERE {pi_where}'
            # Take more recent admissions
            pi += f" ORDER BY t2.admittime {conf['hadm_order']}"
            pi+= ") as pi"
            table += f' INNER JOIN {pi} ON a.hadm_id=pi.hadm_id'
        elif conf['PI_sql'] == 'one' or conf['PI_sql'] == 'multiple':
            pi24_hadms = PI_hadms_24h(conn, conf)
            where += f' AND a.hadm_id NOT IN {pi24_hadms}'
            ignored_values_stage = []
            label_CV_stage, ignored_values_CV_stage = PI_EVENTS_CV['PI Stage']
            ignored_values_stage += ignored_values_CV_stage
            label_MV_stage, ignored_values_MV_stage = PI_EVENTS_MV['PI Stage']
            ignored_values_stage += ignored_values_MV_stage
            pi_where = f'tb1.value is NOT NULL AND tb1.itemid in {PI_STAGE_ITEMIDS}'
            for value in ignored_values_stage:
                pi_where += f" AND tb1.value not similar to '{value}'"
            pi_where += " AND ("
            for i, value in enumerate(STAGE_PI_MAP[max(conf['PI_states'].keys())]):
                if i == 0:
                    pi_where += f" tb1.value similar to '{value}'"
                else:
                    pi_where += f" OR tb1.value similar to '{value}'"
            pi_where += ")"
            if conf['starttime'] and conf['endtime']:
                pi_where += f" AND tb1.charttime >= '{conf['starttime']}'"
                pi_where += f" AND tb1.charttime <= '{conf['endtime']}'"
            # number of PIs within max_hours from admittime
            pi_where += f" AND tb1.charttime - tb2.admittime <='{conf['max_hours']} hours'"
            pi = f'''
                (SELECT t2.hadm_id
                    FROM (SELECT t1.hadm_id, count(*)
                        FROM (SELECT distinct tb1.itemid, tb1.hadm_id
                            FROM {schema}.chartevents tb1 
                            INNER JOIN {schema}.admissions tb2
                            ON tb1.hadm_id=tb2.hadm_id
                            '''
            if conf['dbsource']:
                pi += f' INNER JOIN {schema}.d_items d ON tb1.itemid=d.itemid'
                pi_where += f" AND d.dbsource='{conf['dbsource']}'"
            pi += f'''
                            WHERE {pi_where} ) as t1
                        GROUP BY t1.hadm_id) as t2
                    INNER JOIN {schema}.admissions t3
                    ON t2.hadm_id=t3.hadm_id
                '''
            if conf['PI_sql'] == 'one':
                pi += ' WHERE t2.count = 1'
            elif conf['PI_sql'] == 'multiple':
                pi += ' WHERE t2.count > 1'
            pi += f" ORDER BY t3.admittime {conf['hadm_order']}"
            pi += ") as pi"
            table += f' INNER JOIN {pi} ON a.hadm_id=pi.hadm_id'
        elif conf['PI_sql'] == 'no_PI_stages':
            if conf['dbsource']:
                npi =  f'''
                    (SELECT DISTINCT c.hadm_id
                    FROM {schema}.chartevents c
                    INNER JOIN {schema}.d_items d
                    ON c.itemid=d.itemid
                    WHERE d.dbsource='{conf['dbsource']}')
                    as npi
                    '''
            table += f' INNER JOIN {npi} ON a.hadm_id=npi.hadm_id'
            pi_stage_hadms = PI_stage_hadms(conn, conf)
            print('All PI Stage Admissions in MIMIC-III', len(pi_stage_hadms))
            where += f' AND a.hadm_id NOT IN {pi_stage_hadms}'
        # check if majority of patients have some PI related events
        elif conf['PI_sql'] == 'no_PI_events':
            if conf['dbsource']:
                npi =  f'''
                    (SELECT DISTINCT c.hadm_id
                    FROM {schema}.chartevents c
                    INNER JOIN {schema}.d_items d
                    ON c.itemid=d.itemid
                    WHERE d.dbsource='{conf['dbsource']}')
                    as npi
                    '''
            table += f' INNER JOIN {npi} ON a.hadm_id=npi.hadm_id'
            pi_event_hadms = PI_event_hadms(conn, conf)
            print('All Admissions with PI-related event', len(pi_event_hadms))
            where += f' AND a.hadm_id NOT IN {pi_event_hadms}'
        # patients, admitted in the hospital within the time window
        # Some events occur before the admisson date, but have the correct hadm_id.
        # Those events are ignored.
        if conf['starttime'] and conf['endtime']:
            where += f" AND a.admittime < '{conf['endtime']}'"
            where += f" AND a.admittime >= '{conf['starttime']}'"
        if conf['hadm_limit']:
            limit = f" LIMIT {conf['hadm_limit']}"
    # order admissions by admittime
    query = f"SELECT {cols} FROM {table} WHERE {where} ORDER BY a.admittime {limit}"
    df = pd.read_sql_query(query, conn)
    print("Admissions including subsequent admissions", len(df))
    if conf['first_hadm']:
        # drop subsequent admissions
        df.drop_duplicates('subject_id', inplace=True)
    print("First admissions", len(df))
    #if hadms:
    if conf['include_chronic_illness']:
        df = add_chronic_illness(conn, df, conf)
        #print('Stroke', df['Stroke'].eq(1))
    df.drop('admittime', axis=1, inplace=True)
    df.drop('subject_id', axis=1, inplace=True)
    # compute the age interval
    dff = df.copy()
    df.drop('los', axis=1, inplace=True)
    df.drop('hadm_id', axis=1, inplace=True)
    df.drop('oasis', axis=1, inplace=True)
    left = df['age'] // conf['age_interval'] * conf['age_interval'] 
    right  = left + conf['age_interval']
    df['age'] = left.astype(str) + '-' + right.astype(str)
    # creates a dictionary with <id> as a key.
    df_dict = dict([(k, v)
              for k, v in zip(df.id, df.iloc[:, 1:].to_dict('records'))])
    print("Admissions", len(df))
    return dff, df_dict

def PI_hadms_24h(conn, conf):
    # Patients with PI staging within 24 hours after admission
    pi_where = f't1.value is NOT NULL AND t1.itemid in {PI_STAGE_ITEMIDS}'
    if conf['starttime'] and conf['endtime']:
        pi_where += f" AND t1.charttime >= '{conf['starttime']}'"
        pi_where += f" AND t1.charttime <= '{conf['endtime']}'"
    pi_where += f" AND t1.charttime - t2.admittime <='24 hours'"
    pi = f'''SELECT DISTINCT t1.hadm_id from {schema}.chartevents t1
            INNER JOIN {schema}.admissions t2
            ON t1.hadm_id=t2.hadm_id
        '''
    if conf['dbsource']:
        pi += f' INNER JOIN {schema}.d_items d ON t1.itemid=d.itemid'
        pi_where += f" AND d.dbsource='{conf['dbsource']}'"
    pi += f' WHERE {pi_where}'
    pi24_hadms = pd.read_sql_query(pi, conn)
    return tuple(pi24_hadms['hadm_id'].tolist())

def PI_stage_hadms(conn, conf):
    # Patients with PI stage anytime after admission
    pi_where = f't1.value is NOT NULL AND t1.itemid in {PI_STAGE_ITEMIDS}'
    pi = f'''SELECT DISTINCT t1.hadm_id, t2.admittime FROM {schema}.chartevents t1
            INNER JOIN {schema}.admissions t2
            ON t1.hadm_id=t2.hadm_id
        '''

    if conf['dbsource']:
        pi += f' INNER JOIN {schema}.d_items d ON t1.itemid=d.itemid'
        pi_where += f" AND d.dbsource='{conf['dbsource']}'"
    pi += f' WHERE {pi_where}'
    pi_stage_hadms = pd.read_sql_query(pi, conn)
    return tuple(pi_stage_hadms['hadm_id'].tolist())

def PI_event_hadms(conn, conf):
    # Patients with PI events including PI Stage after admission
    pi_where = f't1.value is NOT NULL'
    for value in PI_ITEM_LABELS:
        pi_where += f"  OR d.label similar to '{value}'"
    pi = f'''SELECT DISTINCT t1.hadm_id from {schema}.chartevents t1
            INNER JOIN {schema}.d_items d
            ON t1.itemid=d.itemid
        '''
    if conf['dbsource']:
        pi_where += f" AND d.dbsource='{conf['dbsource']}'"
    pi += f' WHERE {pi_where}'
    pi_event_hadms = pd.read_sql_query(pi, conn)
    return tuple(pi_event_hadms['hadm_id'].tolist())

def PI_first_stage(conn, conf):
    # Patients with PI stage anytime after admission
    pi_where = f't1.value is NOT NULL AND t1.itemid in {PI_STAGE_ITEMIDS}'
    pi = f'''SELECT DISTINCT t1.hadm_id, t2.admittime FROM {schema}.chartevents t1
            INNER JOIN {schema}.admissions t2
            ON t1.hadm_id=t2.hadm_id
        '''
    if conf['dbsource']:
        pi += f' INNER JOIN {schema}.d_items d ON t1.itemid=d.itemid'
        pi_where += f" AND d.dbsource='{conf['dbsource']}'"
    pi += f' WHERE {pi_where}'
    pi_stage_hadms = pd.read_sql_query(pi, conn)
    return tuple(pi_stage_hadms['hadm_id'].tolist())

# an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]
def get_events(conn, event_key, conf, hadms=(), fname='output/'):
    '''
    Returns events within the given time window.
    '''
    event_name, table, time_col, main_attr = EVENTS[event_key]
    t_table = 'tb'  # time table
    if event_key in EVENT_COLS_INCLUDE:
        cols = 'tb.' + ', tb.'.join(sorted(EVENT_COLS_INCLUDE[event_key]))
    else:
        all_cols = list(
            pd.read_sql_query(
                f'SELECT * FROM {schema}.{table} WHERE false',
                conn).astype(str))
        all_cols.sort()
        all_cols.remove('subject_id')
        all_cols.remove('hadm_id')
        all_cols.remove('row_id')
        if time_col in all_cols:
            all_cols.remove(time_col)
        else:
            t_table = 'a'
        cols = 'tb.' + ', tb.'.join(all_cols)
        if event_name in EVENT_COLS_EXCLUDE:
            cols = [col for col in all_cols if col not in
                    EVENT_COLS_EXCLUDE[event_name]]
            cols = 'tb.' + ', tb.'.join(cols)
    if main_attr:
        ID_cols = f'''
            CONCAT(tb.subject_id, '-', tb.hadm_id) as id,
            tb.subject_id as subject_id, tb.hadm_id as hadm_id,
            RTRIM(INITCAP({main_attr}), '.') as item_col,
            {t_table}.{time_col} - a.admittime as t,
            {t_table}.{time_col} as datetime,
            '''
        # Not to initcap care unit and services abbreviations
        initcap = True
        for e in LOGISTIC_EVENTS:
            if e in event_name:
                initcap = False
        if initcap:
            ID_cols += f"CONCAT('{event_name}', '-', RTRIM(INITCAP({main_attr}), '.')) as type,"
        else:
            ID_cols += f"CONCAT('{event_name}', '-', RTRIM({main_attr}, '.')) as type,"
    else:
        ID_cols = f'''
            CONCAT(tb.subject_id, '-', tb.hadm_id) as id,
            tb.subject_id as subject_id, tb.hadm_id as hadm_id,
            '{event_name}' as type,
            tb.hadm_id as item_col,
            {t_table}.{time_col} - a.admittime as t,
            {t_table}.{time_col} as datetime,
            '''

    ID_cols += f"'{event_name}' as parent_type, "
    cols = ID_cols + cols
    cols += f", 0 as pi_stage "
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    where = ' tb.hadm_id is NOT NULL'
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
    if event_name == 'Input':
        table += f''' INNER JOIN {schema}.d_items d
            ON tb.itemid = d.itemid'''
    where += f'''
        AND a.hospital_expire_flag=0
        AND a.diagnosis != 'NEWBORN'
        AND a.admission_type != 'NEWBORN'
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) >= {conf['min_age']}
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) <= {conf['max_age']}
        AND {t_table}.{time_col} - a.admittime <= '{conf['max_hours']} hours'
        AND a.dischtime - a.admittime >= '{conf['min_los_hours']} hours'
        '''
    if conf['admission_type']:
        where += f" AND a.admission_type = '{conf['admission_type']}'"
    if conf['starttime'] and conf['endtime']:
        where += f" AND {t_table}.{time_col} >= '{conf['starttime']}'"
        where += f" AND {t_table}.{time_col} <= '{conf['endtime']}'"
        where += f" AND a.admittime >= '{conf['starttime']}'"
        where += f" AND a.admittime < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND {t_table}.{time_col} is NOT NULL'
    where += f' AND {t_table}.{time_col} >= a.admittime'
    where += f' AND {t_table}.{time_col} <= a.dischtime'
    order_by = f'ORDER BY t ASC'
    if event_name == 'Transfer In' or event_name == 'Transfer Out':
        # transfers contain admissions and discharge, we exclude them
        # we only take transfers with previous and current care unit
        where += ' AND tb.curr_careunit IS NOT NULL'
        where += ' AND tb.prev_careunit IS NOT NULL'
        where += " AND tb.eventtype='transfer'"
    if 'Presc' in event_name:
        if conf['drug_percentile'] != [0, 100]:
            drugs = prescription_filter(conn, conf, fname)
            where += f" AND RTRIM(INITCAP(tb.drug), '.') IN {drugs} "
        where += f'''
            AND tb.drug IS NOT NULL
            AND tb.dose_val_rx is NOT NULL
            AND tb.dose_unit_rx is NOT NULL
            AND split_part(TRIM(tb.dose_val_rx), ' ', 1) != '0'
            AND split_part(TRIM(tb.dose_val_rx), ' ', 1) != '0.0'
            AND split_part(TRIM(tb.dose_val_rx), '-', 1) != '0'
            AND split_part(TRIM(tb.dose_val_rx), '-', 1) != '0.0'
            AND (split_part(TRIM(tb.dose_val_rx), ' ', 1) similar to '\+?\d*\.?\d*'
            OR split_part(TRIM(tb.dose_val_rx), '-', 1) similar to '\+?\d*\.?\d*')
            '''
    if 'Input' in event_name:
        if conf['input_percentile'] != [0, 100]:
            inputs = input_filter(conn, conf, fname)
            where += f''' 
                AND RTRIM(INITCAP(d.label), '.') in {inputs}
                '''
        where += f''' 
            AND tb.amount is NOT NULL
            AND tb.amountuom is NOT NULL
            AND tb.amount > 0
            '''
        if event_key == 'Input MV':
            where += "AND TRIM(tb.statusdescription) = 'FinishedRunning'"
        elif event_key == 'Input CV':
            where += "AND TRIM(tb.stopped) != 'D/C''d'"
            where += "AND TRIM(tb.stopped) != 'Stopped'"

    query = f"SELECT {cols} FROM {table} WHERE {where} {order_by}"
    df = pd.read_sql_query(query, conn)
    high_freq = False
    for name in HIGH_FREQ_EVENTS:
        if name in event_name:
            high_freq = False
    if high_freq:
        df['event_type'] = df['type']
    else:
        df['event_type'] = df['parent_type']
    for col in df.columns:
        uom_col = None
        if f'{event_name}-{col}' not in NUMERIC_COLS:
            continue
        args = NUMERIC_COLS[f'{event_name}-{col}']
        if event_name == 'Input':
            uom_col = 'amountuom'
        elif args['uom_col']:
            uom_col = args['uom_col']
        if not conf['include_numeric'] and uom_col:
            df[col] = df[col].astype(str) + ' ' + df[uom_col]
            df = df.drop([uom_col], axis=1)
            continue
        if not conf['include_numeric']:
            continue
        if event_name == 'Input':
            Q  = query_quantiles_Input(conn, conf['quantiles'], args) 
        else:
            Q  = query_quantiles(conn, conf, conf['quantiles'], event_name, **args) 
        # compute percentiles for numeric values with uom
        if uom_col:
            df['numeric_value'] = df[col]
            # due to apply error in pandas, a loop is used
            vals = list()
            intervals = list()
            for idx, row in df.iterrows():
                val, I = get_quantile_uom(row, 'item_col', col, uom_col, Q, conf)
                vals.append(val)
                intervals.append(I)
            df[col] = vals
            df[col + '-I'] = intervals
            # add uom to numeric values
            # drop unit of measure column
            df[col] = df[col].astype(str) + ' ' + df[uom_col]
            df = df.drop([uom_col], axis=1)
        # compute percentiles for numeric values without uom
        else:
            df['numeric_value'] = df[col]
            df[col + '-I'] = df[col].apply(lambda x: get_quantile_interval(x, Q, conf))
            df[col] = df[col].apply(lambda x: get_quantile(x, Q, conf))
        # include percentiles in event type
        df['type'] = df['type'] + ' ' + df[col]
    df = df.drop(['item_col'], axis=1)
    if conf['duration']:
        df['duration'] = timedelta(days=0)
    # each row is converted into a dictionary indexed by column names
    events = df.to_dict('records')
    # delete the same cpt code events of same patients happening within the same day
    # aggregate the same cpt code events of duration is true
    if 'CPT' in event_name:
        events.sort(key=lambda x: (x['id'],
                                   x['cpt_cd'],
                                   x['t']))
        key = 'cpt_cd'
        prev_val = None
        prev_t = np.inf
        prev_id = None
        start = None
        t_unit = timedelta(days=1)
        del_list = []
        for i, e in enumerate(events):
            # the same cpt event happening next day
            if e[key] == prev_val and \
                e['t'].days - prev_t == 1 and \
                e['id'] == prev_id:
                if conf['duration']:
                    events[start]['duration'] += t_unit 
                    del_list.append(i)
                prev_t = e['t'].days
            # the same cpt event happening same day
            elif e[key] == prev_val and \
                e['t'].days - prev_t == 0 and \
                e['id'] == prev_id:
                del_list.append(i)
            else:
                start = i
                prev_val = e[key]
                prev_t = e['t'].days
                prev_id = e['id']
        for i in sorted(del_list, reverse=True):
            del events[i]
    events.sort(key = lambda x: (x['type'], x['t']))
    return events


def get_icustays(conn, conf, hadms=()):
    table = 'icustays'
    time_col = 'intime'
    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id"
    cols += f",tb.subject_id, tb.hadm_id"
    cols += f", tb.{time_col} - a.admittime as t, tb.icustay_id"
    cols += f", tb.{time_col} as datetime "
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    where = 'a.hadm_id is NOT NULL'
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
        AND a.hospital_expire_flag=0
        AND a.diagnosis != 'NEWBORN'
        AND a.admission_type != 'NEWBORN'
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) >= {conf['min_age']}
        AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) <= {conf['max_age']}
        AND tb.{time_col} - a.admittime <= '{conf['max_hours']} hours'
        AND a.dischtime - a.admittime >= '{conf['min_los_hours']} hours'
        '''
    if conf['admission_type']:
        where += f" AND a.admission_type='{conf['admission_type']}'"
    if conf['starttime'] and conf['endtime']:
        where += f" AND tb.{time_col} >= '{conf['starttime']}'"
        where += f" AND tb.{time_col} < '{conf['endtime']}'"
        where += f" AND a.admittime >= '{conf['starttime']}'"
        where += f" AND a.admittime < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND tb.{time_col} >= a.admittime'
    where += f' AND tb.{time_col} <= a.dischtime'
    order_by = f'ORDER BY t ASC'
    query = f"SELECT {cols} FROM {table} WHERE {where}"
    df = pd.read_sql_query(query, conn)
    return dict([(k, v) for k, v in zip(df.icustay_id, df.to_dict('records'))])


if __name__ == '__main__':
    conn = get_db_connection()
    get_patient_demography(conn, '2143-01-01', '2143-02-01')
    get_events(conn, 'microbiologyevents', '2143-01-01', '2143-02-01')
