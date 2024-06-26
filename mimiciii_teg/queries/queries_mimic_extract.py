import os.path

import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from mimiciii_teg.queries.queries import *
from mimiciii_teg.utils.utils import *
from mimiciii_teg.schemas.PI_risk_factors import *
LEVEL2 = '../data/all_hourly_data.h5'



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

def get_missing_percents_interventions(X = None):
    fname = f'''data/intervention_stats'''
    if os.path.exists(f'{fname}.h5'):
        return pd.read_hdf(f'{fname}.h5') 
    df = pd.DataFrame((X == 0).sum(
    ) / X.shape[0] * 100, columns=['missing percent'])
    df.sort_values(by='missing percent', ascending=True, inplace=True)
    if not os.path.exists(f'{fname}.h5'):
        df.to_csv(f'{fname}.csv', encoding='UTF-8')
        df.to_hdf(f'{fname}.h5', key='df', mode='w', encoding='UTF-8')
    return df


def get_events_interventions(conn, conf, hadms=None):
    if not conf['interventions']:
        return []
    events = []
    icustays = get_icustays(conn, conf, hadms)
    icustays_ids = list(icustays.keys())
    interventions = get_interventions(conf)
    missing_percents = get_missing_percents_interventions(interventions)
    skip_count = 0
    event_idx = 0
    zero_duration = timedelta(days=0)
    if conf['vitals_agg'] == 'daily':
        time_unit = timedelta(days=1)
    elif conf['vitals_agg'] == 'hourly':
        time_unit = timedelta(hours=1)
    interventions = interventions.loc[(
        slice(None), slice(None), icustays_ids, slice(None)), :]
    excluded = set()
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
            mp = missing_percents.loc[col, 'missing percent']
            val = 1 if count >= 1 else 0
            if val == 0:
                continue
            elif mp < conf['missing_percent'][0] or  mp > conf['missing_percent'][1]:
                excluded.add(col)
                continue
            if not conf['skip_repeat_intervention']:
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               #'event_type': 'Intervention',
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time, # additional info
                               'count': count, # additional info
                               'intervention': 1,
                               'pi_stage': 0})
                continue
            if i not in prev and conf['duration']:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               #'event_type': 'Intervention',
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time,
                               'count': count,
                               'intervention': 1,
                               'duration': zero_duration,
                               'pi_stage': 0})
                event_idx += 1
            elif i not in prev and not conf['duration']:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               #'event_type': 'Intervention',
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time,
                               'count': count,
                               'intervention': 1,
                               'pi_stage': 0})
                event_idx += 1
            elif conf['duration'] and prev[i][0] == val and prev[i][1] == h - 1:
                events[prev[i][2]]['duration'] += time_unit
                events[prev[i][2]]['count'] += count
            elif conf['duration'] and prev[i][0] == val and prev[i][1] != h - 1:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               #'event_type': 'Intervention',
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time,
                               'count': count,
                               'intervention': 1,
                               'pi_stage': 0,
                               'duration': zero_duration})
                event_idx += 1
            elif val != prev[i][0] and conf['duration']:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               #'event_type': 'Intervention',
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time,
                               'count': count,
                               'intervention': 1,
                               'pi_stage': 0,
                               'duration': zero_duration})
                event_idx += 1
            elif val != prev[i][0] and not conf['duration']:
                prev[i] = [val, h, event_idx, count]
                events.append({'id': e['id'],
                               #'type': 'Intervention-' + col,
                               'type': col.title(),
                               'event_type': col.title(),
                               'parent_type': 'Intervention',
                               't': e['t'] + time,
                               'datetime': e['datetime'] + time,
                               'subject_id': subject_id,
                               'hadm_id': hadm_id,
                               'icu-time': time,
                               'count': count,
                               'intervention': 1,
                               'pi_stage': 0})
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1

    events.sort(key=lambda x: (x['type'], x['t']))
    print("Interventions skipped: ", skip_count)
    return events


def get_events_vitals_X_mean(conn, conf, hadms=None, fname='output/'):
    icustays = get_icustays(conn, conf, hadms)
    icustays_ids = list(icustays.keys())
    vitals = get_vitals(conf)
    vitals_stats = get_stats_vitals_X_mean(vitals)
    vitals_nan = vitals.replace(0, np.NaN)
    fname = f"data/mimic-extract-Q-{len(conf['quantiles'])}.h5"
    if os.path.exists(fname):
        Qs = pd.read_hdf(fname) 
    else:
        Qs = vitals_nan.quantile(conf['quantiles'], numeric_only=True)
        Qs.to_hdf(fname, key='df', mode='w', encoding='UTF-8')
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
        if len(conf['PI_vitals']) == 1:
            col = conf['PI_vitals'][0]
            fname = f"data/mimic-extract-{col}"
            plt.clf()
            plt.cla()
            plt.style.use('default')
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(10, 8))
            plt.hist(vitals_nan.loc[:, col], bins=100, rwidth=0.7)
            i = 1
            print("Percentiles")
            for q_val in Qs.loc[:, (col, 'mean')]:
                print(i, q_val)
                if i == 1 or i == len(conf['quantiles']):
                    i += 1
                    continue
                # draw the percentile line
                plt.axvline(q_val, color='red', linestyle='dashed', linewidth=1)
                i += 1
            plt.xlabel("Value (mg/dL)")
            plt.ylabel("Frequency")
            plt.ylim(0, 16000)
            plt.xlim(0, 5)
            plt.savefig(f"{fname}")
            plt.clf()
            plt.cla()
            plt.style.use('default')
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(10, 8))
            plt.hist(vitals_nan.loc[:, col], bins=100, rwidth=0.7)
            i = 1
            for q_val in Qs.loc[:, (col, 'mean')]:
                if i == 1 or i == len(conf['quantiles']):
                    i += 1
                    continue
                # draw the percentile line
                plt.axvline(q_val, color='red', linestyle='dashed', linewidth=1)
                i += 1
            plt.ylabel("Frequency")
            #plt.ylim(0, 1000000)
            plt.xlabel("Value (mg/dL)")
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("Frequency")
            plt.savefig(f"{fname}_log")
    if conf['PI_vitals']:
        vitals_included = conf['PI_vitals']
    else:
        vitals_included = vitals.columns
    excluded = set()
    for subject_id, hadm_id, icustay_id, h in vitals.index:
        # icu event
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
            val = vitals.loc[(subject_id, hadm_id, icustay_id, h), col]
            mp = vitals_stats.loc[col, 'missing percent']
            if pd.isnull(val):
                continue
            elif mp < conf['missing_percent'][0] or mp > conf['missing_percent'][1]:
                excluded.add(col)
                continue
            Q, Q_I = get_quantile_mimic_extract(val, Qs.loc[:, col], conf)
            if not conf['skip_repeat']:
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I
                                   })
                continue

            if i not in prev and conf['duration']:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif i not in prev and not conf['duration']:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I,
                                   })
                event_idx += 1
            elif conf['duration'] and prev[i][0] == Q and prev[i][1] == h - 1:
                icu_events[prev[i][2]]['duration'] += time_unit
            elif conf['duration'] and prev[i][0] == Q and prev[i][1] != h - 1:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif Q != prev[i][0] and conf['duration']:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif Q != prev[i][0] and not conf['duration']:
                prev[i] = [Q, h, event_idx]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-mean': val,
                                   'Q': Q,
                                   'Q_I': Q_I,
                                   })
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1
    icu_events.sort(key=lambda x: (x['type'], x['t']))
    print("Vitals skipped: ", skip_count)
    print("Excluded by missing percent: ", excluded )
    return icu_events


def get_events_vitals_X(conn, conf, hadms=None):
    icustays = get_icustays(conn, conf, hadms)
    icustays_ids = list(icustays.keys())
    vitals = get_vitals(conf)
    vitals_nan = vitals.replace(0, np.NaN)
    fname = f"data/mimic-extract-Q-{len(conf['quantiles'])}.h5"
    if os.path.exists(fname):
        Qs = pd.read_hdf(fname) 
    else:
        Qs = vitals_nan.quantile(conf['quantiles'], numeric_only=True)
        Qs.to_hdf(fname, key='df', mode='w', encoding='UTF-8')
    missing_percents = get_missing_percents_vitals_X(vitals)
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
        if len(conf['PI_vitals']) == 1:
            col = conf['PI_vitals'][0]
            fname = f"data/mimic-extract-{col}-dist"
            plt.clf()
            plt.cla()
            plt.style.use('default')
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(10, 8))
            plt.hist(vitals_nan.loc[:, (col, 'mean')], bins=100, rwidth=0.7)
            i = 1
            print("Percentiles")
            for q_val in Qs.loc[:, (col, 'mean')]:
                print(i, q_val)
                if i == 1 or i == len(conf['quantiles']):
                    i += 1
                    continue
                # draw the percentile line
                plt.axvline(q_val, color='red', linestyle='dashed', linewidth=1)
                i += 1
            plt.xlabel("Value (mg/dL)")
            plt.ylabel("Frequency")
            plt.ylim(0, 16000)
            plt.xlim(0, 5)
            plt.savefig(f"{fname}")
            plt.clf()
            plt.cla()
            plt.style.use('default')
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(10, 8))
            plt.hist(vitals_nan.loc[:, (col, 'mean')], bins=100, rwidth=0.7)
            #plt.ylim(0, 1000000)
            plt.xlabel("Value (mg/dL)")
            i = 1
            for q_val in Qs.loc[:, (col, 'mean')]:
                if i == 1 or i == len(conf['quantiles']):
                    i += 1
                    continue
                # draw the percentile line
                plt.axvline(q_val, color='red', linestyle='dashed', linewidth=1)
                i += 1
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("Frequency")
            plt.savefig(f"{fname}_log")
    if conf['PI_vitals']:
        vitals_included = conf['PI_vitals']
    else:
        vitals_included = vitals.columns.levels[0]
    excluded = set()
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
            mp = missing_percents.loc[col, 'missing percent']
            count, mean, std = vitals.loc[(
                subject_id, hadm_id, icustay_id, h), (col, slice(None))]
            if count == 0:
                continue
            elif mp < conf['missing_percent'][0] or  mp > conf['missing_percent'][1]:
                excluded.add(col)
                continue
            count_Q, count_I = get_quantile_mimic_extract(count, Qs.loc[:, (col, 'count')], conf)
            mean_Q, mean_I = get_quantile_mimic_extract(mean, Qs.loc[:, (col, 'mean')], conf)
            # Todo: std is NaN sometimes
            std_Q, std_I = get_quantile_mimic_extract(std, Qs.loc[:, (col, 'std')], conf)
            if not conf['skip_repeat']:
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   })
                continue
            if i not in prev and conf['duration']:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif i not in prev and not conf['duration']:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   })
                event_idx += 1
            elif conf['duration'] and prev[i][1] == mean_Q and prev[i][3] == h - 1:
                icu_events[prev[i][4]]['duration'] += time_unit
                icu_events[prev[i][4]]['vitals-count'] += count
            elif conf['duration'] and prev[i][1] == mean_Q and prev[i][1] != h - 1:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif mean_Q != prev[i][1] and conf['duration']:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   'duration': zero_duration})
                event_idx += 1
            elif mean_Q != prev[i][1] and not conf['duration']:
                prev[i] = [count_Q, mean_Q, std_Q, h, event_idx, count]
                icu_events.append({'id': e['id'],
                                   #'type': 'Vitals/Labs-' + col + f' {mean_Q}',
                                   #'event_type': 'Vitals/Labs-' + col,
                                   'type': col.title() + f' {mean_Q}',
                                   'event_type': col.title(),
                                   'parent_type': 'Vitals/Labs',
                                   't': e['t'] + time,
                                   'datetime': e['datetime'] + time,
                                   'subject_id': subject_id,
                                   'hadm_id': hadm_id,
                                   'icu-time': time,
                                   'pi_stage': 0,
                                   'vitals-count': count,
                                   'vitals-mean': mean,
                                   'vitals-std': std,
                                   'count_Q': count_Q,
                                   'mean_Q': mean_Q,
                                   'std_Q': std_Q,
                                   'count_I': count_I,
                                   'mean_I': mean_I,
                                   'std_I': std_I,
                                   })
                event_idx += 1
            elif not conf['duration']:
                skip_count += 1
    icu_events.sort(key=lambda x: (x['type'], x['t']))
    print('Vitals skipped: ', skip_count)
    print("Excluded by missing percent: ", excluded )
    return icu_events


def get_stats_vitals_X_mean(vitals = None):
    fname = f'''data/vital_stats_X_mean'''
    if os.path.exists(f'{fname}.h5'):
        return pd.read_hdf(fname) 
    vitals_mean = pd.DataFrame(
        vitals.mean(
            numeric_only=True),
        columns=['mean'])
    vitals_std = pd.DataFrame(vitals.std(numeric_only=True), columns=['std'])
    vitals_missing = pd.DataFrame(
        vitals.isnull().sum() / vitals.shape[0] * 100,
        columns=['missing percent'])

    vitals_stats = pd.concat([vitals_mean, vitals_std, vitals_missing], axis=1)
    # vitals_stats.index = vitals_stats.index.droplevel(1)
    vitals_stats.sort_values(
        by='missing percent',
        ascending=True,
        inplace=True)
    #vitals_states.to_csv('vital_stats.csv', sep=',')
    if not os.path.exists(f'{fname}.h5'):
        vitals_stats.to_csv(f'{fname}.csv', encoding='UTF-8')
        vitals_stats.to_hdf(f'{fname}.h5', key='df', mode='w', encoding='UTF-8')
    return vitals_stats

def get_missing_percents_vitals_X(vitals = None):
    fname = f'''data/vital_stats_X'''
    if os.path.exists(f'{fname}.h5'):
        return pd.read_hdf(f'{fname}.h5') 
    df = pd.DataFrame((vitals.loc[:, (slice(None), 'count')] == 0).sum(
    ) / vitals.shape[0] * 100, columns=['missing percent'])
    df = df.droplevel('Aggregation Function', axis=0)
    df.sort_values(by='missing percent', ascending=True, inplace=True)
    if not os.path.exists(f'{fname}.h5'):
        df.to_csv(f'{fname}.csv', encoding='UTF-8')
        df.to_hdf(f'{fname}.h5', key='df', mode='w', encoding='UTF-8')
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
