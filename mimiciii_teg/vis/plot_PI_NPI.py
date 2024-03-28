import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14
from scipy import stats

def plot_PI_NPI(PI_R, NPI_R, conf, percentile='', fname='ET_PI_NPI'):
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    #df['val_log10'] = np.log10(list(CENTRALITY.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.title(f"Top Event Type Centrality " + str(percentile))
    plt.xlabel("Event Type Centrality")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
    plt.figure(figsize = (12, 8))
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    if percentile == '':
        PI_PCENTRALITY = PI_R['patient_CENTRALITY_total']
        NPI_PCENTRALITY = NPI_R['patient_CENTRALITY_total']
    else:
        PI_PCENTRALITY = PI_R['patient_CENTRALITY_P']
        NPI_PCENTRALITY = NPI_R['patient_CENTRALITY_P']
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
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
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
