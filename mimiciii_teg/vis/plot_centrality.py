import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14
from scipy import stats

from mimiciii_teg.utils.CENTRALITY_utils import get_event_type_CENTRALITY

def plot_event_type_CENTRALITY(CENTRALITY, CENTRALITY_freq, CENTRALITY_avg, conf, percentile='', \
                               P=None, nbins=30, fname='ET_Figure'):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(CENTRALITY)
    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Event Type Centrality Distribution " + str(percentile), fontsize=14)
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Event Type Centrality Values", fontsize=14)
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(False)
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    plt.title(f"Event Type Centrality Distribution After Log Transformation " + str(percentile), fontsize=14)

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Event Type Centrality Values", fontsize=14)
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.title(f"Top Event Type Centrality " + str(percentile))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_sorted = dict(sorted(CENTRALITY.items(), key=lambda x: x[1]))
    top_n = len(CENTRALITY)
    vals = []
    labels = []
    for event_type in list(CENTRALITY_sorted.keys())[n-top_n:]:
        labels.append(event_type)
        vals.append(CENTRALITY_sorted[event_type])
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    if P:
        # draw the percentile line
        plt.axvline(P[0], color='red', linestyle='dashed', linewidth=1, label=f'{percentile[0]}th Percentile')
        plt.legend()
    plt.yticks(y_pos, labels=labels, fontsize=14)
    plt.title(f"Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_33")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_freq = dict(sorted(CENTRALITY_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_freq), 2)
    plt.barh(y_pos, list(CENTRALITY_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_freq.keys()), fontsize=14)
    plt.title("Event Frequency " + str(percentile))
    plt.xlabel("Event Frequency")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    # shortening the verticle space for example experiment
    #plt.figure(figsize = (10, 3))
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)
    plt.barh(y_pos, list(CENTRALITY_sorted.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_sorted.keys()))
    plt.title("Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    CENTRALITY_avg = dict(sorted(CENTRALITY_avg.items(), key=lambda x: x[1]))
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    y_pos  =  range(0, 2*len(CENTRALITY_avg), 2)
    plt.barh(y_pos, list(CENTRALITY_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_avg.keys()), fontsize=14)
    plt.title("Average Event Centrality " + str(percentile))
    plt.xlabel("Average Event Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()

def plot_CENTRALITY(events, CENTRALITY, conf, percentile='', P=None, nbins=30, fname='Figure'):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    n = len(CENTRALITY)
    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Centrality Value Distribution " + str(percentile))
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_1")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    plt.title(f"Centrality Value Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_2")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.title(f"Top Centrality Events " + str(percentile))
    plt.xlabel("Centrality Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_sorted = dict(sorted(CENTRALITY.items(), key=lambda x: x[1]))
    top_n = len(CENTRALITY)
    vals = []
    labels = []
    for i in list(CENTRALITY_sorted.keys())[n-top_n:]:
        labels.append(events[i]['type'])
        vals.append(CENTRALITY_sorted[i])
    y_pos  =  range(0, 2*len(CENTRALITY_sorted), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    if P:
        # draw the percentile line
        plt.axvline(P[0], color='red', linestyle='dashed', linewidth=1,
                    label=f'{percentile[0]}th Percentile')
        plt.legend()
    plt.yticks(y_pos, labels=labels, fontsize=14)
    plt.title(f"Event Centrality " + str(percentile))
    plt.xlabel("Centrality Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_33")
    plt.clf()
    plt.cla()

    CENTRALITY_ET, CENTRALITY_ET_freq, CENTRALITY_ET_avg = get_event_type_CENTRALITY(events, CENTRALITY)
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_ET_freq = dict(sorted(CENTRALITY_ET_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_ET_freq), 2)
    plt.barh(y_pos, list(CENTRALITY_ET_freq.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET_freq.keys()))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_ET = dict(sorted(CENTRALITY_ET.items(), key=lambda x: x[1]))
    plt.barh(y_pos, list(CENTRALITY_ET.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET.keys()))
    plt.title("Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_ET_avg = dict(sorted(CENTRALITY_ET_avg.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(CENTRALITY_ET_avg), 2)
    plt.barh(y_pos, list(CENTRALITY_ET_avg.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_ET_avg.keys()))
    plt.title("Average Event Centrality " + str(percentile))
    plt.xlabel("Average Event Centrality")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_6")
    plt.clf()
    plt.cla()


def plot_CENTRALITY_by_parent_type(events, CENTRALITY, conf, percentile=''):
    #CENTRALITY_t = dict([(events[i]['type'], v) for i, v in CENTRALITY.items()])
    #df = pd.DataFrame({'type': list(CENTRALITY_type.keys()), 'val': list(CENTRALITY_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Centrality Value Distribution " + str(percentile))
    plt.hist(list(CENTRALITY.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero Centrality values")
    if not percentile:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    plt.title(f"Centrality Value Distribution After Log Transformation " + str(percentile))

    plt.hist(np.log10(list(CENTRALITY.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero Centrality Values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.title(f"Top Centrality Events " + str(percentile))
    plt.xlabel("Centrality Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.title("Centrality Event Type Distribution " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    CENTRALITY_p = dict()
    for k, f in CENTRALITY_freq.items():
        CENTRALITY_p[k] = CENTRALITY_sum[k]/CENTRALITY_total
    CENTRALITY_sum = dict(sorted(CENTRALITY_sum.items(), key=lambda x: x[1]))
    CENTRALITY_p = dict(sorted(CENTRALITY_p.items(), key=lambda x: x[1]))
    #plt.bar(CENTRALITY_w.keys(), CENTRALITY_w.values(), width=0.5, align='center')
    y_pos  =  range(0, 2*len(CENTRALITY_sum), 2)
    plt.barh(y_pos, list(CENTRALITY_sum.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_sum.keys()))
    plt.title("Sum of Centrality Values" + str(percentile))
    plt.xlabel("Sum of Centrality Values")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.barh(y_pos, list(CENTRALITY_p.values()), align='center')
    plt.yticks(y_pos, labels=list(CENTRALITY_p.keys()))
    plt.title("Portion of Total Sum of Centrality Values" + str(percentile))
    plt.xlabel("Portion of Total Centrality")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()
