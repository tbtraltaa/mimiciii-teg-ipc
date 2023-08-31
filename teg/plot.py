import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from scipy.signal import savgol_filter

from teg.event_utils import *
from teg.PC_utils import *
from teg.queries_chart_events import get_chart_events

def plot_event_type_PC(PC, PC_freq, PC_avg, conf, percentile='', nbins = 30, title = '', fname='ET_Figure'):
    #PC_t = dict([(events[i]['type'], v) for i, v in PC.items()])
    #df = pd.DataFrame({'type': list(PC_type.keys()), 'val': list(PC_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(PC)
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event type PC distribution " + str(percentile))
    else:
        plt.title(f"Event type PC distribution " + str(percentile))
    plt.hist(list(PC.values()), bins=30, rwidth=0.7)
    plt.xlabel("Event type PC values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(PC.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Event type PC distribution after Log transformation " + str(percentile))
    else:
        plt.title(f"Event type PC distribution after Log transformation " + str(percentile))

    plt.hist(np.log10(list(PC.values())), bins=20, rwidth=0.7)
    plt.xlabel("Event type PC values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()


    plt.figure(figsize=(14, 8))
    PC_sorted = dict(sorted(PC.items(), key=lambda x: x[1]))
    top_n = conf['Top_n_PC'] if len(PC) > conf['Top_n_PC'] else len(PC)
    vals = []
    labels = []
    for event_type in list(PC_sorted.keys())[n-top_n:]:
        labels.append(event_type)
        vals.append(PC_sorted[event_type])
    y_pos  =  range(0, 2*len(PC_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top event type PC " + str(percentile))
    else:
        plt.title(f"Top event type PC " + str(percentile))
    plt.xlabel("Event type PC")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_freq = dict(sorted(PC_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(PC_freq), 2)
    plt.barh(y_pos, list(PC_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_freq.keys()))
    if title:
        plt.title(f"{title}: Event frequency " + str(percentile))
    else:
        plt.title("Event frequency " + str(percentile))
    plt.xlabel("Event frequency")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 2*len(PC_sorted), 2)
    plt.barh(y_pos, list(PC_sorted.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_sorted.keys()))
    if title:
        plt.title(f"{title}: Event type PC " + str(percentile))
    else:
        plt.title("Event type PC " + str(percentile))
    plt.xlabel("Event type PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    PC_avg = dict(sorted(PC_avg.items(), key=lambda x: x[1]))
    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 2*len(PC_avg), 2)
    plt.barh(y_pos, list(PC_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_avg.keys()))
    if title:
        plt.title(f"{title}: Average event PC " + str(percentile))
    else:
        plt.title("Average event PC " + str(percentile))
    plt.xlabel("Average event PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()

def plot_PC(events, PC, conf, percentile='', nbins=30, title='', fname='Figure'):
    #PC_t = dict([(events[i]['type'], v) for i, v in PC.items()])
    #df = pd.DataFrame({'type': list(PC_type.keys()), 'val': list(PC_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(PC)
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: PC value distribution " + str(percentile))
    else:
        plt.title(f"PC value distribution " + str(percentile))
    plt.hist(list(PC.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(PC.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: PC value distribution after Log transformation " + str(percentile))
    else:
        plt.title(f"PC value distribution after Log transformation " + str(percentile))

    plt.hist(np.log10(list(PC.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_sorted = dict(sorted(PC.items(), key=lambda x: x[1]))
    top_n = conf['Top_n_PC'] if len(PC) > conf['Top_n_PC'] else len(PC)
    vals = []
    labels = []
    for i in list(PC_sorted.keys())[n-top_n:]:
        labels.append(events[i]['type'])
        vals.append(PC_sorted[i])
    y_pos  =  range(0, 2*len(PC_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top PC events " + str(percentile))
    else:
        plt.title(f"Top PC events " + str(percentile))
    plt.xlabel("PC Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    PC_ET, PC_ET_freq, PC_ET_avg = get_event_type_PC(events, PC)
    plt.figure(figsize=(14, 8))
    PC_ET_freq = dict(sorted(PC_ET_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(PC_ET_freq), 2)
    plt.barh(y_pos, list(PC_ET_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_ET_freq.keys()))
    if title:
        plt.title(f"{title}: Event frequency " + str(percentile))
    else:
        plt.title("Event frequency " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_ET = dict(sorted(PC_ET.items(), key=lambda x: x[1]))
    plt.barh(y_pos, list(PC_ET.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_ET.keys()))
    if title:
        plt.title(f"{title}: Event type PC " + str(percentile))
    else:
        plt.title("Event type PC " + str(percentile))
    plt.xlabel("Event type PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_ET_avg = dict(sorted(PC_ET_avg.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(PC_ET_avg), 2)
    plt.barh(y_pos, list(PC_ET_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_ET_avg.keys()))
    if title:
        plt.title(f"{title}: Average event PC " + str(percentile))
    else:
        plt.title("Average event PC " + str(percentile))
    plt.xlabel("Average event PC")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()


def plot_PC_by_parent_type(events, PC, conf, percentile='', nbins=30, title=''):
    #PC_t = dict([(events[i]['type'], v) for i, v in PC.items()])
    #df = pd.DataFrame({'type': list(PC_type.keys()), 'val': list(PC_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: PC value distribution " + str(percentile))
    else:
        plt.title(f"PC value distribution " + str(percentile))
    plt.hist(list(PC.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(PC.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: PC value distribution after Log transformation " + str(percentile))
    else:
        plt.title(f"PC value distribution after Log transformation " + str(percentile))

    plt.hist(np.log10(list(PC.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(14, 8))
    PC_sorted = dict(sorted(PC.items(), key=lambda x: x[1], reverse=True))
    top_n = conf['Top_n_PC'] if len(PC) > conf['Top_n_PC'] else len(PC)
    vals = []
    labels = []
    for i in list(PC_sorted.keys())[:top_n]:
        labels = [events[i]['parent_type']] + labels
        vals = [PC_sorted[i]] + vals
    y_pos  =  range(0, 2*len(PC_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    if title:
        plt.title(f"{title}: Top PC Events " + str(percentile))
    else:
        plt.title(f"Top PC Events " + str(percentile))
    plt.xlabel("PC Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    PC_n = len(PC)
    PC_freq = dict()
    PC_sum = dict()
    PC_total = 0
    for i, val in PC.items():
        etype = events[i]['parent_type']
        PC_total += val
        if etype not in PC_freq:
            PC_freq[etype] = 1
            PC_sum[etype] = val
        else:
            PC_freq[etype] += 1
            PC_sum[etype] += val
    PC_freq = dict(sorted(PC_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(PC_freq), 2)
    plt.barh(y_pos, list(PC_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_freq.keys()))
    if title:
        plt.title(f"{title}: PC Event Type Distribution " + str(percentile))
    else:
        plt.title("PC Event Type Distribution " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(14, 8))
    PC_p = dict()
    for k, f in PC_freq.items():
        PC_p[k] = PC_sum[k]/PC_total
    PC_sum = dict(sorted(PC_sum.items(), key=lambda x: x[1]))
    PC_p = dict(sorted(PC_p.items(), key=lambda x: x[1]))
    #plt.bar(PC_w.keys(), PC_w.values(), width=0.5, align='center')
    y_pos  =  range(0, 2*len(PC_sum), 2)
    plt.barh(y_pos, list(PC_sum.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_sum.keys()))
    if title:
        plt.title(f"{title}: Sum of PC values" + str(percentile))
    else:
        plt.title("Sum of PC values" + str(percentile))
    plt.xlabel("Sum of PC values")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    plt.barh(y_pos, list(PC_p.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_p.keys()))
    if title:
        plt.title(f"{title}: Portion of total sum of PC values " + str(percentile))
    else:
        plt.title("Portion of total sum of PC values" + str(percentile))
    plt.xlabel("Portion of total PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()


def plot_PC_and_BS(conn, conf, patient_PC, PI_hadms, PI_hadm_stage_t, fname):
    braden_events = get_chart_events(conn, 'Braden Score', conf, PI_hadms)
    print('Braden Scale events: ', len(braden_events))
    braden_events = remove_events_after_t(braden_events, PI_hadm_stage_t)
    #braden_events = [e for e in PI_events if 'Braden Score' in e['type']]
    patient_BS = get_patient_max_Braden_Scores(braden_events, conf['PC_time_unit'])
    plot_time_series(patient_PC, patient_BS, conf, f'{fname}_points')
    plot_time_series_average(patient_PC, patient_BS, conf, f'{fname}_avg')
    '''
    for idd in patient_PC:
        if idd in patient_BS:
            print(idd, len(patient_PC[idd]['PC']), len(patient_BS[idd]['BS']))
        else:
            print(idd, len(patient_PC[idd]['PC']))
    '''


def plot_time_series(patient_PC, patient_BS, conf, fname, patients_NPI_PC = None, PI_NPI_match = None):
    #extract color palette, the palette can be changed
    colors = list(sns.color_palette(palette='viridis', n_colors=len(patient_PC)).as_hex())
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for p_id, color in zip(patient_PC, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_PC[p_id]['t'],
                y = patient_PC[p_id]['PC'],
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
            npi_id = [idd for idd in patients_NPI_PC if npi_hadm_id in idd][0]
            fig.add_trace(
                go.Scatter(
                    x = patients_NPI_PC[npi_id]['t'],
                    y = patients_NPI_PC[npi_id]['PC'],
                    name = p_id,
                    mode='lines+markers',
                    line= dict(color=color, dash='dash'),
                    fill=None),
                    secondary_y=False)


    # label x-axes
    fig.update_xaxes(title_text = f"Time after admission (time unit: {str(conf['PC_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "PC value", secondary_y=False)
    fig.update_yaxes(title_text = "Braden Scale", secondary_y=True)
    fig.write_html(f"{fname}_PC_BS.html")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for p_id, color in zip(patient_PC, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_PC[p_id]['t'],
                y = patient_PC[p_id]['PC'],
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
            npi_id = [idd for idd in patients_NPI_PC if npi_hadm_id in idd][0]
            fig.add_trace(
                go.Scatter(
                    x = patients_NPI_PC[npi_id]['t'],
                    y = patients_NPI_PC[npi_id]['PC'],
                    name = p_id,
                    mode='lines+markers',
                    line= dict(color=color, dash='dash'),
                    fill=None),
                    secondary_y=False)

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for p_id, color in zip(patient_PC, colors):
        fig.add_trace(
            go.Scatter(
                x = patient_PC[p_id]['t'],
                y = patient_PC[p_id]['PC'],
                name = p_id,
                mode='markers',
                line_color = color,
                fill=None))
    # label x-axes
    fig.update_xaxes(title_text = f"Time after admission (time unit: {str(conf['PC_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "PC value")
    fig.write_html(f"{fname}_PC.html")

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for p_id, color in zip(patient_PC, colors):
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
    fig.update_xaxes(title_text = f"Time after admission (time unit: {str(conf['PC_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text = "Braden Scale")
    fig.write_html(f"{fname}_BS.html")

def plot_time_series_average(patient_PC, patient_BS, conf, fname):
    #extract color palette, the palette can be changed
    colors = list(sns.color_palette(palette='viridis', n_colors=2).as_hex())
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = int(conf['max_hours'] * timedelta(hours=1).total_seconds()/conf['PC_time_unit'].total_seconds())
    x = [i for i in range(0, n + 1)]
    nnz1 = np.zeros(n + 1)
    y1_sum = np.zeros(n + 1)
    nnz2 = np.zeros(n + 1)
    y2_sum = np.zeros(n + 1)
    for p_id in patient_PC:
        if len(patient_PC[p_id]['PC']) == 0:
            continue
        # get interpolated values
        # values beyond given time period are considered 0
        y1 = np.interp(x, patient_PC[p_id]['t'], patient_PC[p_id]['PC'], right = 0, left = 0)
        # counts number of nonzero PC values
        nnz1 += y1 != 0
        # accumulates total PC values all patients per hour
        y1_sum += y1
        if p_id in patient_BS:
            if len(patient_BS[p_id]['BS']) == 0:
                continue
            # get interpolated values
            # values beyond given time period are considered 0
            y2 = np.interp(x, patient_BS[p_id]['t'], patient_BS[p_id]['BS'], right = 0, left = 0)
            # counts number of nonzero PC values
            nnz2 += y2 != 0
            # accumulates total PC values all patients per hour
            y2_sum += y2
    # average value per hour
    y1_avg = y1_sum / nnz1
    y2_avg = y2_sum / nnz2
    n_p = len(patient_PC)
    x_PC = x
    x_BS = x
    '''
    x_PC = [i for i in x if float(nnz1[i])/n_p >= conf['PC_BS_nnz']]
    x_BS = [ i for i in x if float(nnz2[i])/n_p >= conf['PC_BS_nnz']]
    y1_avg = [y1_avg[i] for i in x_PC]
    y2_avg = [y2_avg[i] for i in x_BS]
    '''
    fig.add_trace(
        go.Scatter(
            x = x_PC,
            y = y1_avg,
            name = 'Average PC values',
            #mode='lines+markers',
            mode='markers',
            line_color = colors[0],
            fill=None),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x = x_PC,
            y = savgol_filter(y1_avg, 51, 4),
            name = 'Smoothed Average PC values',
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
    fig.update_xaxes(title_text=f"Time after admission (time unit: {str(conf['PC_time_unit'])})")
    # label y-axes
    fig.update_yaxes(title_text="Average PC value", secondary_y=False)
    fig.update_yaxes(title_text="Average Braden Scale", secondary_y=True)
    fig.write_html(f"{fname}_PC_BS_smooth.html")

def plot_PI_NPI(PI_R, NPI_R, conf, percentile='', nbins = 30, title = '', fname='ET_PI_NPI'):
    if PI_R is None or NPI_R is None:
        return
    PI_c = 'red'
    NPI_c = 'blue'
    if not percentile:
        PI_PC = PI_R['ET_PC']
        PI_PC_freq = PI_R['ET_PC_freq']
        PI_PC_avg = PI_R['ET_PC_avg']
        NPI_PC = NPI_R['ET_PC']
        NPI_PC_freq = NPI_R['ET_PC_freq']
        NPI_PC_avg = NPI_R['ET_PC_avg']
    else:
        PI_PC = PI_R['ET_PC_P']
        PI_PC_freq = PI_R['ET_PC_P_freq']
        PI_PC_avg = PI_R['ET_PC_P_avg']
        NPI_PC = NPI_R['ET_PC_P']
        NPI_PC_freq = NPI_R['ET_PC_P_freq']
        NPI_PC_avg = NPI_R['ET_PC_P_avg']

    PI_n = len(PI_PC)
    NPI_n = len(NPI_PC)
    max_n = PI_n if PI_n > NPI_n else NPI_n
    PI_l = 'PI'
    NPI_l = 'NPI'
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Event type PC distribution " + str(percentile))
    else:
        plt.title(f"Event type PC distribution " + str(percentile))
    plt.hist(list(PI_PC.values()), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(list(NPI_PC.values()), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Event type PC values")
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
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    #df['val_log10'] = np.log10(list(PC.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    if title:
        plt.title(f"{title}: Event type PC distribution after Log transformation " + str(percentile))
    else:
        plt.title(f"Event type PC distribution after Log transformation " + str(percentile))

    plt.hist(np.log10(list(PI_PC.values())), bins=20, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(np.log10(list(NPI_PC.values())), bins=20, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.axvline(np.log10(PI_R['ET_P'][0]), color=PI_c, linestyle='dashed', linewidth=1)
    plt.axvline(np.log10(NPI_R['ET_P'][0]), color=NPI_c, linestyle='dashed', linewidth=1)
    plt.xlabel("Event type PC values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()


    plt.figure(figsize=(14, 8))
    PI_PC = dict(sorted(PI_PC.items(), key=lambda x: x[1], reverse=True))
    NPI_PC = dict(sorted(NPI_PC.items(), key=lambda x: x[1], reverse=True))
    PI_top_n = conf['Top_n_PC'] if PI_n > conf['Top_n_PC'] else PI_n
    NPI_top_n = conf['Top_n_PC'] if NPI_n > conf['Top_n_PC'] else NPI_n
    max_top_n = PI_top_n if PI_top_n > NPI_top_n else NPI_top_n
    PI_minor = True if PI_top_n <= NPI_top_n else False
    vals = []
    labels = []
    for event_type in list(NPI_PC.keys())[:NPI_top_n]:
        labels.append(event_type)
        vals.append(NPI_PC[event_type])
    y_pos  =  range(8 * (max_top_n + 1)-4, 0, -8)
    for i in range(len(y_pos) - len(vals)):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor= not PI_minor)
    vals = []
    labels = []
    for event_type in list(PI_PC.keys())[:PI_top_n]:
        labels.append(event_type)
        vals.append(PI_PC[event_type])
    y_pos  =  range(8 * (max_top_n + 1), 0, -8)
    for i in range(len(y_pos) - len(vals)):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Top event type PC " + str(percentile))
    else:
        plt.title(f"Top event type PC " + str(percentile))
    plt.xlabel("Event type PC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PI_minor = True if PI_n <= NPI_n else False
    NPI_PC_freq = dict(sorted(NPI_PC_freq.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8 * (max_n + 1) - 4, 0, -8)
    vals = list(NPI_PC_freq.values())
    labels = list(NPI_PC_freq.keys())
    for i in range(len(y_pos) - NPI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)
    PI_PC_freq = dict(sorted(PI_PC_freq.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8 * (max_n + 1), 0, -8)
    vals = list(PI_PC_freq.values())
    labels = list(PI_PC_freq.keys())
    for i in range(len(y_pos) - PI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)

    if title:
        plt.title(f"{title}: Event frequency " + str(percentile))
    else:
        plt.title("Event frequency " + str(percentile))
    plt.xlabel("Event frequency")
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    y_pos  =  range(8 * (max_n + 1) -4, 0, -8)
    vals = list(NPI_PC.values())
    labels = list(NPI_PC.keys())
    for i in range(len(y_pos) - NPI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)
    y_pos  =  range(8 * (max_n + 1), 0, -8)
    vals = list(PI_PC.values())
    labels = list(PI_PC.keys())
    for i in range(len(y_pos) - PI_n):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Event type PC " + str(percentile))
    else:
        plt.title("Event type PC " + str(percentile))
    plt.xlabel("Event type PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    PI_n_avg = len(PI_PC_avg)
    NPI_n_avg = len(NPI_PC_avg)
    max_n_avg = PI_n_avg if PI_n_avg > NPI_n_avg else NPI_n_avg
    PI_minor = True if PI_n_avg <= NPI_n_avg else False
    plt.figure(figsize=(14, 8))
    NPI_PC_avg = dict(sorted(NPI_PC_avg.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8* (max_n_avg + 1) - 4, 0, -8)
    vals = list(NPI_PC_avg.values())
    labels = list(NPI_PC_avg.keys())
    for i in range(len(y_pos) - NPI_n_avg):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=NPI_c, label=NPI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=NPI_c, minor=not PI_minor)

    PI_PC_avg = dict(sorted(PI_PC_avg.items(), key=lambda x: x[1], reverse=True))
    y_pos  =  range(8* (max_n_avg + 1), 0, -8)
    vals = list(PI_PC_avg.values())
    labels = list(PI_PC_avg.keys())
    for i in range(len(y_pos) - PI_n_avg):
        vals.append(0)
        labels.append('')
    plt.barh(y_pos, vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=labels, fontsize=10, color=PI_c, minor=PI_minor)
    if title:
        plt.title(f"{title}: Average event PC " + str(percentile))
    else:
        plt.title("Average event PC " + str(percentile))
    plt.xlabel("Average event PC")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()

    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if percentile == '':
        PI_PPC = PI_R['patient_PC_total']
        NPI_PPC = NPI_R['patient_PC_total']
    else:
        PI_PPC = PI_R['patient_PC_P']
        NPI_PPC = NPI_R['patient_PC_P']
    if title:
        plt.title(f"{title}: Patient PC " + str(percentile))
    else:
        plt.title(f"Patient PC " + str(percentile))
    plt.hist(list(PI_PPC.values()), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(list(NPI_PPC.values()), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Patient PC")
    if percentile == '':
        plt.axvline(PI_R['PPC_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['PPC_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.axvline(PI_R['PPC_P'][0], color=PI_c, linestyle='dashed', linewidth=1)
        plt.axvline(NPI_R['PPC_P'][0], color=NPI_c, linestyle='dashed', linewidth=1)

    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_7")
    plt.clf()
    plt.cla()

    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: Patient PC Log base 10 " + str(percentile))
    else:
        plt.title(f"Patient PC Log base 10 " + str(percentile))
    plt.hist(np.log10(list(PI_PPC.values())), bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(np.log10(list(NPI_PPC.values())), bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Patient PC")
    plt.axvline(np.log10(PI_R['PPC_P'][0]), color=PI_c, linestyle='dashed', linewidth=1)
    plt.axvline(np.log10(NPI_R['PPC_P'][0]), color=NPI_c, linestyle='dashed', linewidth=1)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_8")
    plt.clf()
    plt.cla()
