import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from scipy.signal import savgol_filter
from scipy import stats

from mimiciii_teg.utils.event_utils import *
from mimiciii_teg.utils.CENTRALITY_utils import *
from mimiciii_teg.queries.queries_chart_events import get_chart_events

def plot_event_type_CENTRALITY(CENTRALITY, CENTRALITY_freq, CENTRALITY_avg, conf, percentile='', nbins = 30, title = '', fname='ET_Figure'):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(CENTRALITY)
    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution " + str(percentile))
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Event Type Centrality Values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution After Log Transformation " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Event Type Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution " + str(percentile))
    res = stats.relfreq(list(CENTRALITY.values()), numbins=30)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,
                                 res.frequency.size)
    plt.bar(x, res.frequency, width=res.binsize)
    plt.xlabel("Event Type Centrality Values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Relative Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_0")
    plt.clf()
    plt.cla()


    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_sorted = dict(sorted(CENTRALITY.items(), key=lambda x: x[1]))
    top_n = conf['Top_n_CENTRALITY'] if len(CENTRALITY) > conf['Top_n_CENTRALITY'] else len(CENTRALITY)
    vals = []
    labels = []
    for event_type in list(CENTRALITY_sorted.keys())[n-top_n:]:
        labels.append(event_type)
        vals.append(CENTRALITY_sorted[event_type])
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top Event Type Centrality" + str(percentile))
    else:
        plt.title(f"Top Event Cype Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_freq = dict(sorted(CENTRALITY_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_freq), 2)
    plt.barh(y_pos, list(CENTRALITY_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_freq.keys()), fontsize=14)
    if title:
        plt.title(f"{title}: Event Frequency " + str(percentile))
    else:
        plt.title("Event Frequency " + str(percentile))
    plt.xlabel("Event Frequency")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)
    plt.barh(y_pos, list(CENTRALITY_sorted.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_sorted.keys()), fontsize=14)
    if title:
        plt.title(f"{title}: Event Type Centrality " + str(percentile))
    else:
        plt.title("Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    CENTRALITY_avg = dict(sorted(CENTRALITY_avg.items(), key=lambda x: x[1]))
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 2*len(CENTRALITY_avg), 2)
    plt.barh(y_pos, list(CENTRALITY_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_avg.keys()), fontsize=14)
    if title:
        plt.title(f"{title}: Average Event Centrality " + str(percentile))
    else:
        plt.title("Average Event Centrality " + str(percentile))
    plt.xlabel("Average Event Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()

def plot_CENTRALITY(events, CENTRALITY, conf, percentile='', nbins=30, title='', fname='Figure'):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(CENTRALITY)
    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Centrality Value Distribution " + str(percentile))
    else:
        plt.title(f"Centrality Value Distribution " + str(percentile))
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Centrality Value Distribution After Log Transformation " + str(percentile))
    else:
        plt.title(f"Centrality Value Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_sorted = dict(sorted(CENTRALITY.items(), key=lambda x: x[1]))
    top_n = conf['Top_n_CENTRALITY'] if len(CENTRALITY) > conf['Top_n_CENTRALITY'] else len(CENTRALITY)
    vals = []
    labels = []
    for i in list(CENTRALITY_sorted.keys())[n-top_n:]:
        labels.append(events[i]['type'])
        vals.append(CENTRALITY_sorted[i])
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top Centrality Events " + str(percentile))
    else:
        plt.title(f"Top Centrality Events " + str(percentile))
    plt.xlabel("Centrality Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    CENTRALITY_ET, CENTRALITY_ET_freq, CENTRALITY_ET_avg = get_event_type_CENTRALITY(events, CENTRALITY)
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_ET_freq = dict(sorted(CENTRALITY_ET_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_ET_freq), 2)
    plt.barh(y_pos, list(CENTRALITY_ET_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET_freq.keys()))
    if title:
        plt.title(f"{title}: Event Frequency " + str(percentile))
    else:
        plt.title("Event Frequency " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_ET = dict(sorted(CENTRALITY_ET.items(), key=lambda x: x[1]))
    plt.barh(y_pos, list(CENTRALITY_ET.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET.keys()))
    if title:
        plt.title(f"{title}: Event Type Centrality " + str(percentile))
    else:
        plt.title("Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_ET_avg = dict(sorted(CENTRALITY_ET_avg.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_ET_avg), 2)
    plt.barh(y_pos, list(CENTRALITY_ET_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET_avg.keys()))
    if title:
        plt.title(f"{title}: Average Event Centrality " + str(percentile))
    else:
        plt.title("Average Event Centrality " + str(percentile))
    plt.xlabel("Average Event Centrality")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()


def plot_CENTRALITY_by_parent_type(events, CENTRALITY, conf, percentile='', nbins=30, title=''):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Centrality Value Distribution " + str(percentile))
    else:
        plt.title(f"Centrality Value Distribution " + str(percentile))
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero Centrality values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Centrality Value Distribution After Log Transformation " + str(percentile))
    else:
        plt.title(f"Centrality Value Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_sorted = dict(sorted(CENTRALITY.items(), key=lambda x: x[1], reverse=True))
    top_n = conf['Top_n_CENTRALITY'] if len(CENTRALITY) > conf['Top_n_CENTRALITY'] else len(CENTRALITY)
    vals = []
    labels = []
    for i in list(CENTRALITY_sorted.keys())[:top_n]:
        labels = [events[i]['parent_type']] + labels
        vals = [CENTRALITY_sorted[i]] + vals
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top Centrality Events " + str(percentile))
    else:
        plt.title(f"Top Centrality Events " + str(percentile))
    plt.xlabel("Centrality Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_n = len(CENTRALITY)
    CENTRALITY_freq = dict()
    CENTRALITY_sum = dict()
    CENTRALITY_total = 0
    for i, val in CENTRALITY.items():
        etype = events[i]['parent_type']
        CENTRALITY_total += val
        if etype not in CENTRALITY_freq:
            CENTRALITY_freq[etype] = 1
            CENTRALITY_sum[etype] = val
        else:
            CENTRALITY_freq[etype] += 1
            CENTRALITY_sum[etype] += val
    CENTRALITY_freq = dict(sorted(CENTRALITY_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_freq), 2)
    plt.barh(y_pos, list(CENTRALITY_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_freq.keys()))
    if title:
        plt.title(f"{title}: Centrality Event Type Distribution " + str(percentile))
    else:
        plt.title("Centrality Event Type Distribution " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    CENTRALITY_p = dict()
    for k, f in CENTRALITY_freq.items():
        CENTRALITY_p[k] = CENTRALITY_sum[k]/CENTRALITY_total
    CENTRALITY_sum = dict(sorted(CENTRALITY_sum.items(), key=lambda x: x[1]))
    CENTRALITY_p = dict(sorted(CENTRALITY_p.items(), key=lambda x: x[1]))
    #plt.bar(CENTRALITY_w.keys(), CENTRALITY_w.values(), width=0.5, align='center')
    y_pos  =  range(0, 2*len(CENTRALITY_sum), 2)
    plt.barh(y_pos, list(CENTRALITY_sum.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_sum.keys()))
    if title:
        plt.title(f"{title}: Sum of Centrality Values" + str(percentile))
    else:
        plt.title("Sum of Centrality Values" + str(percentile))
    plt.xlabel("Sum of Centrality Values")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    plt.barh(y_pos, list(CENTRALITY_p.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_p.keys()))
    if title:
        plt.title(f"{title}: Portion Of Total Sum Of Centrality Values " + str(percentile))
    else:
        plt.title("Portion of Total Sum of Centrality Values" + str(percentile))
    plt.xlabel("Portion of Total Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()


def plot_CENTRALITY_and_BS(conn, conf, patient_CENTRALITY, PI_hadms, PI_hadm_stage_t, fname):
    braden_events = get_chart_events(conn, 'Braden Score', conf, PI_hadms)
    print('Braden Scale Events: ', len(braden_events))
    braden_events = remove_events_after_t(braden_events, PI_hadm_stage_t)
    #braden_events = [e for e in PI_events if 'Braden Score' in e['type']]
    patient_BS = get_patient_max_Braden_Scores(braden_events, conf['CENTRALITY_time_unit'])
    plot_time_series(patient_CENTRALITY, patient_BS, conf, f'{fname}_points')
    plot_time_series_average(patient_CENTRALITY, patient_BS, conf, f'{fname}_avg')
    '''
    for idd in patient_CENTRALITY:
        if idd in patient_BS:
            print(idd, len(patient_CENTRALITY[idd]['CENTRALITY']), len(patient_BS[idd]['BS']))
        else:
            print(idd, len(patient_CENTRALITY[idd]['CENTRALITY']))
    '''


def plot_time_series(patient_CENTRALITY, patient_BS, conf, fname, patients_NPI_CENTRALITY = None, PI_NPI_match = None):
    #extract color palette, the palette can be changed
    colors = list(sns.color_palette(palette='viridis', n_colors=len(patient_CENTRALITY)).as_hex())
    
    plt.style.use('default')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for p_id, color in zip(patient_CENTRALITY, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_CENTRALITY[p_id]['t'],
                y = patient_CENTRALITY[p_id]['CENTRALITY'],
                name = p_id,
                mode='lines+markers',
                line_color = color,
                fill=None),
            secondary_y=False)
        if p_id in patient_BS:
            fig.add_trace(
                go.Scatter(
                    x = patient_BS[p_id]['t'],
                    y = patient_BS[p_id]['BS'],
                    name = p_id,
                    mode='lines+markers',
                    line= dict(color=color, dash='dot'),
                    fill=None),
                secondary_y=True)
        if PI_NPI_match is None:
            continue
        if p_id.split('-')[1] in PI_NPI_match:
            print('Match')
            npi_hadm_id = PI_NPI_match[p_id.split('-')[1]]
            npi_id = [idd for idd in patients_NPI_CENTRALITY if npi_hadm_id in idd][0]
            fig.add_trace(
                go.Scatter(
                    x = patients_NPI_CENTRALITY[npi_id]['t'],
                    y = patients_NPI_CENTRALITY[npi_id]['CENTRALITY'],
                    name = p_id,
                    mode='lines+markers',
                    line= dict(color=color, dash='dash'),
                    fill=None),
                    secondary_y=False)


    # label x-axes
    fig.update_xaxes(title_text = f"Time After Admission (Time Unit: {str(conf['CENTRALITY_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "Centrality Calue", secondary_y=False)
    fig.update_yaxes(title_text = "Braden Scale", secondary_y=True)
    fig.write_html(f"{fname}_CENTRALITY_BS.html")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for p_id, color in zip(patient_CENTRALITY, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_CENTRALITY[p_id]['t'],
                y = patient_CENTRALITY[p_id]['CENTRALITY'],
                name = p_id,
                mode='markers',
                line_color = color,
                fill=None),
            secondary_y=False)
        if p_id in patient_BS:
            fig.add_trace(
                go.Scatter(
                    x = patient_BS[p_id]['t'],
                    y = patient_BS[p_id]['BS'],
                    name = p_id,
                    mode='markers',
                    line= dict(color=color, dash='dot'),
                    fill=None),
                secondary_y=True)
        if PI_NPI_match is None:
            continue
        if p_id.split('-')[1] in PI_NPI_match:
            print('Match')
            npi_hadm_id = PI_NPI_match[p_id.split('-')[1]]
            npi_id = [idd for idd in patients_NPI_CENTRALITY if npi_hadm_id in idd][0]
            fig.add_trace(
                go.Scatter(
                    x = patients_NPI_CENTRALITY[npi_id]['t'],
                    y = patients_NPI_CENTRALITY[npi_id]['CENTRALITY'],
                    name = p_id,
                    mode='lines+markers',
                    line= dict(color=color, dash='dash'),
                    fill=None),
                    secondary_y=False)

    plt.style.use('default')
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for p_id, color in zip(patient_CENTRALITY, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_CENTRALITY[p_id]['t'],
                y = patient_CENTRALITY[p_id]['CENTRALITY'],
                name = p_id,
                mode='markers',
                line_color = color,
                fill=None))
    # label x-axes
    fig.update_xaxes(title_text = f"Time After Admission (Time Unit: {str(conf['CENTRALITY_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "Centrality Value")
    fig.write_html(f"{fname}_CENTRALITY.html")

    plt.style.use('default')
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for p_id, color in zip(patient_CENTRALITY, colors):
        if p_id in patient_BS:
            fig.add_trace(
                go.Scatter(
                    x = patient_BS[p_id]['t'],
                    y = patient_BS[p_id]['BS'],
                    name = p_id,
                    mode='markers',
                    line= dict(color=color, dash='dot'),
                    fill=None))

    # label x-axes
    fig.update_xaxes(title_text = f"Time After Admission (Time Unit: {str(conf['CENTRALITY_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "Braden Scale")
    fig.write_html(f"{fname}_BS.html")

def plot_time_series_average(patient_CENTRALITY, patient_BS, conf, fname):
    #extract color palette, the palette can be changed
    colors = list(sns.color_palette(palette='viridis', n_colors=2).as_hex())
    
    plt.style.use('default')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = int(conf['max_hours'] * timedelta(hours=1).total_seconds()/conf['CENTRALITY_time_unit'].total_seconds())
    x = [i for i in range(0, n + 1)]
    nnz1 = np.zeros(n + 1)
    y1_sum = np.zeros(n + 1)
    nnz2 = np.zeros(n + 1)
    y2_sum = np.zeros(n + 1)
    for p_id in patient_CENTRALITY:
        if len(patient_CENTRALITY[p_id]['CENTRALITY']) == 0:
            continue
        # get interpolated values
        # values beyond given time period are considered 0
        y1 = np.interp(x, patient_CENTRALITY[p_id]['t'], patient_CENTRALITY[p_id]['CENTRALITY'], right = 0, left = 0)
        # counts number of nonzero CENTRALITY values
        nnz1 += y1 != 0
        # accumulates total CENTRALITY values all patients per hour
        y1_sum += y1
        if p_id in patient_BS:
            if len(patient_BS[p_id]['BS']) == 0:
                continue
            # get interpolated values
            # values beyond given time period are considered 0
            y2 = np.interp(x, patient_BS[p_id]['t'], patient_BS[p_id]['BS'], right = 0, left = 0)
            # counts number of nonzero CENTRALITY values
            nnz2 += y2 != 0
            # accumulates total CENTRALITY values all patients per hour
            y2_sum += y2
    # average value per hour
    y1_avg = y1_sum / nnz1
    y2_avg = y2_sum / nnz2
    n_p = len(patient_CENTRALITY)
    x_CENTRALITY = x
    x_BS = x
    '''
    x_CENTRALITY = [i for i in x if float(nnz1[i])/n_p >= conf['CENTRALITY_BS_nnz']]
    x_BS = [ i for i in x if float(nnz2[i])/n_p >= conf['CENTRALITY_BS_nnz']]
    y1_avg = [y1_avg[i] for i in x_CENTRALITY]
    y2_avg = [y2_avg[i] for i in x_BS]
    '''
    fig.add_trace(
        go.Scatter(
            x = x_CENTRALITY,
            y = y1_avg,
            name = 'Average Centrality Values',
            #mode='lines+markers',
            mode='markers',
            line_color = colors[0],
            fill=None),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x = x_CENTRALITY,
            y = savgol_filter(y1_avg, 51, 4),
            name = 'Smoothed Average Centrality Values',
            #mode='lines+markers',
            mode='lines',
            line_color = colors[0],
            fill=None),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x = x_BS,
            y = y2_avg,
            name = 'Average Braden Scale',
            mode='markers',
            fill=None),
        secondary_y=True)

    fig.add_trace(
        go.Scatter(
            x = x_BS,
            y = savgol_filter(y2_avg, 51, 4),
            name = 'Smoothed Average Braden Scale',
            mode='lines',
            fill=None),
        secondary_y=True)

    # label x-axes
    fig.update_xaxes(title_text=f"Time After Admission (Time Unit: {str(conf['CENTRALITY_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text="Average Centrality Value", secondary_y=False)
    fig.update_yaxes(title_text="Average Braden Scale", secondary_y=True)
    fig.write_html(f"{fname}_CENTRALITY_BS_smooth.html")

def plot_PI_NPI(PI_R, NPI_R, conf, percentile='', nbins = 30, title = '', fname='ET_PI_NPI'):
    if PI_R is None or NPI_R is None:
        return
    PI_c = 'red'
    NPI_c = 'blue'
    if not percentile:
        PI_CENTRALITY = PI_R['ET_CENTRALITY']
        PI_CENTRALITY_freq = PI_R['ET_CENTRALITY_freq']
        PI_CENTRALITY_avg = PI_R['ET_CENTRALITY_avg']
        NPI_CENTRALITY = NPI_R['ET_CENTRALITY']
        NPI_CENTRALITY_freq = NPI_R['ET_CENTRALITY_freq']
        NPI_CENTRALITY_avg = NPI_R['ET_CENTRALITY_avg']
    else:
        PI_CENTRALITY = PI_R['ET_CENTRALITY_P']
        PI_CENTRALITY_freq = PI_R['ET_CENTRALITY_P_freq']
        PI_CENTRALITY_avg = PI_R['ET_CENTRALITY_P_avg']
        NPI_CENTRALITY = NPI_R['ET_CENTRALITY_P']
        NPI_CENTRALITY_freq = NPI_R['ET_CENTRALITY_P_freq']
        NPI_CENTRALITY_avg = NPI_R['ET_CENTRALITY_P_avg']

    PI_n = len(PI_CENTRALITY)
    NPI_n = len(NPI_CENTRALITY)
    max_n = PI_n if PI_n > NPI_n else NPI_n
    PI_l = 'PI'
    NPI_l = 'NPI'

    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution " + str(percentile))
    res = stats.relfreq(list(PI_CENTRALITY.values()), numbins=30)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,
                                 res.frequency.size)
    plt.bar(x, res.frequency, width=res.binsize, color=PI_c, label=PI_l)
    res = stats.relfreq(list(NPI_CENTRALITY.values()), numbins=30)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,
                                 res.frequency.size)
    plt.bar(x, res.frequency, width=res.binsize, color=NPI_c, label=NPI_l)
    plt.xlabel("Event Type Centrality Values")
    plt.ylabel("Relative Frequency")
    if percentile == '':
        plt.axvline(PI_R['ET_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['ET_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)
        #plt.xscale("log")
        #plt.yscale("log")
    else:
        plt.axvline(PI_R['ET_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['ET_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_0")

    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution " + str(percentile))
    plt.hist(list(PI_CENTRALITY.values()), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(list(NPI_CENTRALITY.values()), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Event Type Centrality Values")
    if percentile == '':
        plt.axvline(PI_R['ET_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['ET_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)
        #plt.xscale("log")
        #plt.yscale("log")
    else:
        plt.axvline(PI_R['ET_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['ET_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)

    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Event Type Centrality Distribution After Log Transformation " + str(percentile))
    else:
        plt.title(f"Event Type Centrality Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(PI_CENTRALITY.values())), bins=20, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(np.log10(list(NPI_CENTRALITY.values())), bins=20, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.axvline(np.log10(PI_R['ET_P'][0]), color=PI_c, linestyle='dashed', linewidth=1)
    plt.axvline(np.log10(NPI_R['ET_P'][0]), color=NPI_c, linestyle='dashed', linewidth=1)
    plt.xlabel("Event Type Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()


    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    PI_CENTRALITY = dict(sorted(PI_CENTRALITY.items(), key=lambda x: x[1], reverse=True))
    NPI_CENTRALITY = dict(sorted(NPI_CENTRALITY.items(), key=lambda x: x[1], reverse=True))
    PI_top_n = conf['Top_n_CENTRALITY'] if PI_n > conf['Top_n_CENTRALITY'] else PI_n
    NPI_top_n = conf['Top_n_CENTRALITY'] if NPI_n > conf['Top_n_CENTRALITY'] else NPI_n
    max_top_n = PI_top_n if PI_top_n > NPI_top_n else NPI_top_n
    PI_minor = True if PI_top_n <= NPI_top_n else False
    vals = []
    labels = []
    for event_type in list(NPI_CENTRALITY.keys())[:NPI_top_n]:
        labels.append(event_type)
        vals.append(NPI_CENTRALITY[event_type])
    y_pos  =  range(8 * (max_top_n + 1)-4, 0, -8)
    for i in range(len(y_pos) - len(vals)):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor= not PI_minor)
    vals = []
    labels = []
    for event_type in list(PI_CENTRALITY.keys())[:PI_top_n]:
        labels.append(event_type)
        vals.append(PI_CENTRALITY[event_type])
    y_pos  =  range(8 * (max_top_n + 1), 0, -8)
    for i in range(len(y_pos) - len(vals)):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Top Event Type Centrality " + str(percentile))
    else:
        plt.title(f"Top Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    PI_minor = True if PI_n <= NPI_n else False
    NPI_CENTRALITY_freq = dict(sorted(NPI_CENTRALITY_freq.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8 * (max_n + 1) - 4, 0, -8)
    vals = list(NPI_CENTRALITY_freq.values())
    labels = list(NPI_CENTRALITY_freq.keys())
    for i in range(len(y_pos) - NPI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)
    PI_CENTRALITY_freq = dict(sorted(PI_CENTRALITY_freq.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8 * (max_n + 1), 0, -8)
    vals = list(PI_CENTRALITY_freq.values())
    labels = list(PI_CENTRALITY_freq.keys())
    for i in range(len(y_pos) - PI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)

    if title:
        plt.title(f"{title}: Event Frequency " + str(percentile))
    else:
        plt.title("Event Frequency " + str(percentile))
    plt.xlabel("Event Frequency")
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    y_pos  =  range(8 * (max_n + 1) -4, 0, -8)
    vals = list(NPI_CENTRALITY.values())
    labels = list(NPI_CENTRALITY.keys())
    for i in range(len(y_pos) - NPI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)
    y_pos  =  range(8 * (max_n + 1), 0, -8)
    vals = list(PI_CENTRALITY.values())
    labels = list(PI_CENTRALITY.keys())
    for i in range(len(y_pos) - PI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Event Type Centrality " + str(percentile))
    else:
        plt.title("Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    PI_n_avg = len(PI_CENTRALITY_avg)
    NPI_n_avg = len(NPI_CENTRALITY_avg)
    max_n_avg = PI_n_avg if PI_n_avg > NPI_n_avg else NPI_n_avg
    PI_minor = True if PI_n_avg <= NPI_n_avg else False
    plt.figure(figsize=(14, 8))
    NPI_CENTRALITY_avg = dict(sorted(NPI_CENTRALITY_avg.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8* (max_n_avg + 1) - 4, 0, -8)
    vals = list(NPI_CENTRALITY_avg.values())
    labels = list(NPI_CENTRALITY_avg.keys())
    for i in range(len(y_pos) - NPI_n_avg):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)

    PI_CENTRALITY_avg = dict(sorted(PI_CENTRALITY_avg.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8* (max_n_avg + 1), 0, -8)
    vals = list(PI_CENTRALITY_avg.values())
    labels = list(PI_CENTRALITY_avg.keys())
    for i in range(len(y_pos) - PI_n_avg):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Average Event Centrality " + str(percentile))
    else:
        plt.title("Average Event Centrality " + str(percentile))
    plt.xlabel("Average event CENTRALITY")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()

    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if percentile == '':
        PI_PCENTRALITY = PI_R['patient_CENTRALITY_total']
        NPI_PCENTRALITY = NPI_R['patient_CENTRALITY_total']
    else:
        PI_PCENTRALITY = PI_R['patient_CENTRALITY_P']
        NPI_PCENTRALITY = NPI_R['patient_CENTRALITY_P']
    if title:
        plt.title(f"{title}: Patient CENTRALITY " + str(percentile))
    else:
        plt.title(f"Patient CENTRALITY " + str(percentile))
    plt.hist(list(PI_PCENTRALITY.values()), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(list(NPI_PCENTRALITY.values()), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Patient CENTRALITY")
    if percentile == '':
        plt.axvline(PI_R['PCENTRALITY_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['PCENTRALITY_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.axvline(PI_R['PCENTRALITY_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['PCENTRALITY_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)

    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_7")
    plt.clf()
    plt.cla()

    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Patient Centrality Log Base 10 " + str(percentile))
    else:
        plt.title(f"Patient Centrality Log Base 10 " + str(percentile))
    plt.hist(np.log10(list(PI_PCENTRALITY.values())), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(np.log10(list(NPI_PCENTRALITY.values())), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Patient Centrality")
    plt.axvline(np.log10(PI_R['PCENTRALITY_P'][0]), color=PI_c, linestyle='dashed', linewidth=1)
    plt.axvline(np.log10(NPI_R['PCENTRALITY_P'][0]), color=NPI_c, linestyle='dashed', linewidth=1)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_8")
    plt.clf()
    plt.cla()
