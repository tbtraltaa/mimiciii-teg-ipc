import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from scipy.signal import savgol_filter

from mimiciii_teg.utils.event_utils import *
from mimiciii_teg.utils.CENTRALITY_utils import *
from mimiciii_teg.queries.queries_chart_events import get_chart_events


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
    plt.rcParams['font.size'] = 14
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
    plt.rcParams['font.size'] = 14
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
    plt.rcParams['font.size'] = 14
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
    plt.rcParams['font.size'] = 14
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
