import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from teg.queries import *
from teg.PI_risk_factors import *
LEVEL2 = '../data/all_hourly_data.h5'
#LEVEL2 = 'data/all_hourly_data.h5'



def get_vitals(conf):
    if conf['vitals_X_mean'] and conf['vitals_agg'] == 'hourly':
        vitals = pd.read_hdf(LEVEL2, 'vitals_labs_mean')
        return vitals.droplevel('Aggregation Function', axis=1)
    elif not conf['vitals_X_mean'] and conf['vitals_agg'] == 'hourly':
        return pd.read_hdf(LEVEL2, 'vitals_labs')
    elif conf['vitals_X_mean'] and conf['vitals_agg'] == 'daily':
        vitals = pd.read_hdf(LEVEL2, 'vitals_labs_mean')
        vitals = vitals.droplevel('Aggregation Function', axis=1)
        return get_daily_vitals_X_mean(vitals)
    elif not conf['vitals_X_mean'] and conf['vitals_agg'] == 'daily':
        vitals = pd.read_hdf(LEVEL2, 'vitals_labs')
        return get_daily_vitals_X(vitals)


def get_interventions(conf):
    interventions = pd.read_hdf(LEVEL2, 'interventions')
    if conf['vitals_agg'] == 'daily':
        df = interventions.index.to_frame()
        df['days_in'] = df['hours_in'] // 24
        interventions.index = pd.MultiIndex.from_frame(df)
        interventions = interventions.groupby(
            ['subject_id', 'hadm_id', 'icustay_id', 'days_in']) \
            .sum(numeric_only=True)
    return interventions


def get_events_interventions(conn, conf):
    if not conf['interventions']:
        return []
    events = []
    icustays = get_icustays(conn, conf)
    icustays_ids = list(icustays.keys())
    interventions = get_interventions(conf)
    skip_count = 0
    event_idx = 0
    zero_duration = timedelta(days=0)
    if conf['vitals_agg'] == 'daily':
        time_unit = timedelta(days=1)
    elif conf['vitals_agg'] == 'hourly':
        time_unit = timedelta(hours=1)
    interventions = interventions.loc[(
        slice(None), slice(None), icustays_ids, slice(None)), :]
    for subject_id, hadm_id, icustay_id, h in interventions.index:
        if h == 0:
            prev = dict()
        e = icustays[icustay_id]
        if conf['vitals_agg'] == 'daily':
            time = timedelta(days=h + 1)
        elif conf['vitals_agg'] == 'hourly':
            time = timedelta(hours=h)
        if e['t'] + time > timedelta(hours=conf['max_hours']):
            continue
        for i, col in enumerate(interventions.columns):
            count = interventions.loc[(subject_id,
                                     hadm_id,
                                     icustay_id,
                                     h), col]
            val = 1 if count >= 1 else 0
            if val == 0:
                continue
            if not conf['skip_repeat']:
                events.append({'id': e['id'],
                               'type': 'intervention-' + col,
                               't': e['t'] + time,
                               'icu-time': time,
                               'intervention-count': count,
                               'intervention': 1})
                continue
            if i not in prev:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               'type': 'intervention-' + col,
                               't': e['t'] + time,
                               'icu-time': time,
                               'intervention-count': count,
                               'intervention': 1,
                               'duration': zero_duration})
                event_idx += 1
            elif conf['duration'] and prev[i][0] == val and prev[i][1] == h - 1:
                events[prev[i][2]]['duration'] += time_unit
                events[prev[i][2]]['intervention-count'] += count
            elif conf['duration'] and prev[i][0] == val and prev[i][1] != h - 1:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               'type': 'intervention-' + col,
                               't': e['t'] + time,
                               'icu-time': time,
                               'intervention-count': count,
                               'intervention': 1,
                               'duration': zero_duration})
                event_idx += 1
            elif val != prev[i][0]:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               'type': 'intervention-' + col,
                               't': e['t'] + time,
                               'icu-time': time,
                               'intervention-count': count,
                               'intervention': 1,
                               'duration': zero_duration})
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1

    events.sort(key=lambda x: (x['type'], x['t']))
    print("Interventions skipped: ", skip_count)
    return events


def get_events_vitals_X_mean(conn, conf):
    icustays = get_icustays(conn, conf)
    icustays_ids = list(icustays.keys())
    vitals = get_vitals(conf)
    vitals_stats = get_stats_vitals_X_mean(vitals)
    icu_events = []
    vitals = vitals.loc[(slice(None), slice(
        None), icustays_ids, slice(None)), :]
    skip_count = 0
    event_idx = 0
    zero_duration = timedelta(days=0)
    if conf['vitals_agg'] == 'daily':
        time_unit = timedelta(days=1)
    elif conf['vitals_agg'] == 'hourly':
        time_unit = timedelta(hours=1)
    if conf['PI_vitals']:
        vitals_included = PI_VITALS
    else:
        vitals_included = vitals.columns.levels[0]
    for subject_id, hadm_id, icustay_id, h in vitals.index:
        e = icustays[icustay_id]
        if conf['vitals_agg'] == 'daily':
            time = timedelta(days=h + 1)
            time_unit = timedelta(days=1)
        elif conf['vitals_agg'] == 'hourly':
            time = timedelta(hours=h)
            time_unit = timedelta(hours=1)
        if e['t'] + time > timedelta(hours=conf['max_hours']):
            continue
        if h == 0:
            prev = dict()
        for i, col in enumerate(vitals_included):
            print(['%s'%col for col in vitals.columns])
            val = vitals.loc[(subject_id, hadm_id, icustay_id, h), col]
            if pd.isnull(val) or \
                vitals_stats.loc[col, 'missing percent'] \
                    < conf['min_missing_percent']:
                continue
            stat = vitals_stats.loc[col]
            Q = get_quartile(val, stat['Q1'], stat['mean'], stat['Q3'])
            if not conf['skip_repeat']:
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-mean': val,
                                   'Q': Q})
                continue

            if i not in prev:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif conf['duration'] and prev[i][0] == Q and prev[i][1] == h - 1:
                icu_events[prev[i][2]]['duration'] += time_unit
            elif conf['duration'] and prev[i][0] == Q and prev[i][1] != h - 1:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif Q != prev[i][0]:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1
    icu_events.sort(key=lambda x: (x['type'], x['t']))
    print("Vitals skipped: ", skip_count)
    return icu_events


def get_events_vitals_X(conn, conf):
    icustays = get_icustays(conn, conf)
    icustays_ids = list(icustays.keys())
    vitals = get_vitals(conf)
    Q1, Q2, Q3 = get_quartiles_vitals_X(vitals)
    print(Q1)
    print(Q2)
    print(Q3)
    missing_percents = get_missing_percents_vitals_X(vitals)
    icu_events = []
    vitals = vitals.loc[(slice(None), slice(
        None), icustays_ids, slice(None)), :]
    print(vitals.index)
    skip_count = 0
    event_idx = 0
    zero_duration = timedelta(days=0)
    if conf['vitals_agg'] == 'daily':
        time_unit = timedelta(days=1)
    elif conf['vitals_agg'] == 'hourly':
        time_unit = timedelta(hours=1)
    if conf['PI_vitals']:
        vitals_included = PI_VITALS
    else:
        vitals_included = vitals.columns.levels[0]
    for subject_id, hadm_id, icustay_id, h in vitals.index:
        e = icustays[icustay_id]
        if conf['vitals_agg'] == 'daily':
            time = timedelta(days=h + 1)
        elif conf['vitals_agg'] == 'hourly':
            time = timedelta(hours=h)
        if e['t'] + time > timedelta(hours=conf['max_hours']):
            continue
            # print(vitals.columns.levels[0])
        if h == 0:
            prev = dict()
        for i, col in enumerate(vitals_included):
            count, mean, std = vitals.loc[(
                subject_id, hadm_id, icustay_id, h), (col, slice(None))]
            if count == 0 or \
                missing_percents.loc[col, 'missing percent'] \
                    < conf['min_missing_percent']:
                continue
            count_Q = get_quartile(count,
                                   Q1.loc[(col, 'count')],
                                   Q2.loc[(col, 'count')],
                                   Q3.loc[(col, 'count')])
            mean_Q = get_quartile(mean,
                                  Q1.loc[(col, 'mean')],
                                  Q2.loc[(col, 'mean')],
                                  Q3.loc[(col, 'mean')])
            # Todo: std is NaN sometimes
            std_Q = get_quartile(std,
                                 Q1.loc[(col, 'std')],
                                 Q2.loc[(col, 'std')],
                                 Q3.loc[(col, 'std')])

            if not conf['skip_repeat']:
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q})
                continue
            if i not in prev:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif conf['duration'] and prev[i][1] == mean_Q and prev[i][3] == h - 1:
                icu_events[prev[i][4]]['duration'] += time_unit
                icu_events[prev[i][4]]['vitals-count'] += count
            elif conf['duration'] and prev[i][1] == mean_Q and prev[i][1] != h - 1:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif mean_Q != prev[i][1]:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   'type': col,
                                   't': e['t'] + time,
                                   'icu-time': time,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'duration': zero_duration})
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1
    icu_events.sort(key=lambda x: (x['type'], x['t']))
    print('Vitals skipped: ', skip_count)
    return icu_events


def get_quartile(vital, Q1, Q2, Q3):
    if vital <= Q1:
        vital = 1
    elif vital > Q1 and vital <= Q2:
        vital = 2
    elif vital > Q2 and vital <= Q3:
        vital = 3
    else:
        vital = 4
    return vital


def get_stats_vitals_X_mean(vitals):
    vitals_mean = pd.DataFrame(
        vitals.mean(
            numeric_only=True),
        columns=['mean'])
    vitals_std = pd.DataFrame(vitals.std(numeric_only=True), columns=['std'])
    vitals_Q1 = pd.DataFrame(
        vitals_mean['mean'] -
        vitals_std['std'] *
        0.675,
        columns=['Q1'])
    vitals_Q3 = pd.DataFrame(
        vitals_mean['mean'] +
        vitals_std['std'] *
        0.675,
        columns=['Q3'])
    vitals_missing = pd.DataFrame(
        vitals.isnull().sum() / vitals.shape[0] * 100,
        columns=['missing percent'])

    vitals_stats = pd.concat([vitals_mean, vitals_std, vitals_missing], axis=1)
    vitals_stats = pd.concat([vitals_stats, vitals_Q1, vitals_Q3], axis=1)
    # vitals_stats.index = vitals_stats.index.droplevel(1)
    vitals_stats.sort_values(
        by='missing percent',
        ascending=True,
        inplace=True)
    #vitals_states.to_csv('vital_stats.csv', sep=',')
    return vitals_stats


def get_quartiles_vitals_X(vitals):
    vitals_mean = vitals.mean(numeric_only=True)
    vitals_std = vitals.std(numeric_only=True)
    vitals_Q1 = vitals_mean - vitals_std * 0.675
    vitals_Q3 = vitals_mean + vitals_std * 0.675
    return vitals_Q1, vitals_mean, vitals_Q3


def get_missing_percents_vitals_X(vitals):
    df = pd.DataFrame((vitals.loc[:, (slice(None), 'count')] == 0).sum(
    ) / vitals.shape[0] * 100, columns=['missing percent'])
    df = df.droplevel('Aggregation Function', axis=0)
    df.sort_values(by='missing percent', ascending=True, inplace=True)
    return df


def get_daily_vitals_X_mean(vitals):
    df = vitals.index.to_frame()
    df['days_in'] = df['hours_in'] // 24
    vitals.index = pd.MultiIndex.from_frame(df)
    daily_vitals = vitals.groupby(
        ['subject_id', 'hadm_id', 'icustay_id', 'days_in']) \
        .mean(numeric_only=True)
    return daily_vitals


def get_daily_vitals_X(vitals):
    df = vitals.index.to_frame()
    df['days_in'] = df['hours_in'] // 24
    vitals.index = pd.MultiIndex.from_frame(df)
    daily_vitals = vitals.groupby(
        ['subject_id', 'hadm_id', 'icustay_id', 'days_in']) \
        .mean(numeric_only=True)
    daily_vitals.loc[:, (slice(None), 'count')] = vitals.groupby(
        ['subject_id', 'hadm_id', 'icustay_id', 'days_in']) \
        .sum().loc[:, (slice(None), 'count')]
    return daily_vitals


if __name__ == '__main__':
    LEVEL2 = '../experiments/data/all_hourly_data.h5'
    vitals = pd.read_hdf(LEVEL2, 'vitals_labs')
    # vitals = vitals.droplevel('Aggregation Function', axis=1)
    print(get_daily_vitals_X(vitals))
    get_missing_percents_X(get_daily_vitals_X(vitals))
