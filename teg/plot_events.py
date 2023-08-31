import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_events_by_parent_type(events, title = '', fname = 'Events'):
    parent_type = dict()
    for e in events:
        if e['parent_type'] not in parent_type:
            parent_type[e['parent_type']] = 1
        else:
            parent_type[e['parent_type']] += 1
    
    parent_type = dict(sorted(parent_type.items(), key=lambda x: x[1]))
    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 2*len(parent_type), 2)
    plt.barh(y_pos, list(parent_type.values()), align='center')
    plt.yticks(y_pos, labels=list(parent_type.keys()))
    plt.title(f"Event categories")
    plt.xlabel("Event count")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}")
    plt.clf()
    plt.cla()

def plot_event_parent_types(PI_events, NPI_events, title = '', fname = 'PI_NPI_Events'):
    PI_c = 'red'
    NPI_c = 'blue'
    PI_l = 'PI'
    NPI_l = 'NPI'
    parent_type = dict()
    for e in PI_events:
        if e['parent_type'] not in parent_type:
            parent_type[e['parent_type']] = [1, 0]
        else:
            parent_type[e['parent_type']][0] += 1
    for e in NPI_events:
        if e['parent_type'] not in parent_type:
            parent_type[e['parent_type']] = [0, 1]
        else:
            parent_type[e['parent_type']][1] += 1
    parent_type = dict(sorted(parent_type.items(), key=lambda x: x[1][0]))
    plt.figure(figsize=(14, 8))
    y_pos  =  range(0, 4 * len(parent_type), 4)
    PI_vals = [val[0] for val in parent_type.values()]
    NPI_vals = [val[1] for val in parent_type.values()]
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    y_pos  =  range(2, 4 * len(parent_type) + 2, 4)
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, labels=list(parent_type.keys()))
    plt.title(f"Event categories {title}")
    plt.xlabel("Event count")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}")
    plt.clf()
    plt.cla()

def plot_types(PI_events, NPI_events, conf, title = '', fname = 'PI_NPI_Events'):
    PI_c = 'red'
    NPI_c = 'blue'
    PI_l = 'PI'
    NPI_l = 'NPI'
    types = dict()
    for e in PI_events:
        if 'PI Stage' in e['type'] or 'Marker' in e['type'] or 'Admissions' in e['type']:
            continue
        if e['type'] not in types:
            types[e['type']] = [1, 0]
        else:
            types[e['type']][0] += 1
    for e in NPI_events:
        if 'PI Stage' in e['type'] or 'Marker' in e['type'] or 'Admissions' in e['type']:
            continue
        if e['type'] not in types:
            types[e['type']] = [0, 1]
        else:
            types[e['type']][1] += 1
    types = dict(sorted(types.items(), key=lambda x: x[1][0], reverse=True))
    plt.figure(figsize=(14, 8))
    n = min(len(types), conf['plot_types_n'])
    y_pos  =  range(8, 8 * (n + 1),  8)[:n]
    PI_vals = [val[0] for val in types.values()][:n]
    PI_vals.reverse()
    NPI_vals = [val[1] for val in types.values()][:n]
    NPI_vals.reverse()
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=list(types.keys())[:n])
    y_pos  =  range(4, 8 * (n + 1) + 4, 8)[:n]
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    plt.title(f" Event types {title}")
    plt.xlabel("Event count")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}")
    plt.clf()
    plt.cla()

def plot_event_types(PI_events, NPI_events, conf, title = '', fname = 'PI_NPI_Events'):
    PI_c = 'red'
    NPI_c = 'blue'
    PI_l = 'PI'
    NPI_l = 'NPI'
    types = dict()
    for e in PI_events:
        if 'PI Stage' in e['event_type'] or 'Marker' in e['event_type'] or 'Admissions' in e['event_type']:
            continue
        if e['event_type'] not in types:
            types[e['event_type']] = [1, 0]
        else:
            types[e['event_type']][0] += 1
    for e in NPI_events:
        if 'PI Stage' in e['event_type'] or 'Marker' in e['event_type'] or 'Admissions' in e['event_type']:
            continue
        if e['event_type'] not in types:
            types[e['event_type']] = [0, 1]
        else:
            types[e['event_type']][1] += 1
    types = dict(sorted(types.items(), key=lambda x: x[1][0], reverse=True))
    plt.figure(figsize=(14, 8))
    n = min(len(types), conf['plot_types_n'])
    y_pos  =  range(8, 8 * (n + 1),  8)[:n]
    PI_vals = [val[0] for val in types.values()][:n]
    PI_vals.reverse()
    NPI_vals = [val[1] for val in types.values()][:n]
    NPI_vals.reverse()
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=list(types.keys())[:n])
    y_pos  =  range(4, 8 * (n + 1) + 4, 8)[:n]
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    plt.title(f" Event types {title}")
    plt.xlabel("Event count")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}")
    plt.clf()
    plt.cla()
