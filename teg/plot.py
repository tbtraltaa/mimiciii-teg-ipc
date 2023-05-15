import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_PC(events, PC, conf, percentile='', nbins=30, title='', fname='Figure'):
    #PC_t = dict([(events[i]['type'], v) for i, v in PC.items()])
    #df = pd.DataFrame({'type': list(PC_type.keys()), 'val': list(PC_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    if title:
        plt.title(f"{title}: PC value distribution " + str(percentile))
    else:
        plt.title(f"PC value distribution " + str(percentile))
    plt.hist(list(PC.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
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

    '''
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        PC_tt = dict()
        for t, v in PC_t.items():
            if t not in PC_t:
                PC_tt[t] = [v]
            else:
                PC_tt[t].append(v)
        ys = []
        cmap = plt.get_cmap('inferno')
        for i, t in enumerate(PC_tt):
            hist, bins = np.histogram(PC_tt[t], bins=nbins)
            xs = (bins[:-1] + bins[1:])/2
            ys.append(i * 10)
            ax.bar(xs, hist, zs=i*10, zdir='y', color=cmap(i), alpha=0.8)
        ax.set_xlabel('PC Value')
        ax.set_ylabel('Event types')
        ax.set_zlabel('Frequency')
        ax.set_yticks(ys, labels=list(PC_tt.keys()), fontsize=14)
        plt.show()
    '''

    plt.figure(figsize=(14, 8))
    PC_sorted = dict(sorted(PC.items(), key=lambda x: x[1], reverse=True))
    top_n = conf['Top_n_PC'] if len(PC) > conf['Top_n_PC'] else len(PC)
    vals = []
    labels = []
    for i in list(PC_sorted.keys())[:top_n]:
        labels = [events[i]['type']] + labels
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
    plt.savefig(f"{fname}_3")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_n = len(PC)
    PC_freq = dict()
    PC_sum = dict()
    PC_total = 0
    for i, val in PC.items():
        etype = events[i]['type']
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
    plt.savefig(f"{fname}_4")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    PC_w = dict()
    PC_p = dict()
    for k, f in PC_freq.items():
        #PC_w[k] = PC_sum[k]* f/PC_n
        PC_w[k] = (PC_sum[k]/f) * f/PC_n
        PC_p[k] = PC_sum[k]/PC_total
    PC_w = dict(sorted(PC_w.items(), key=lambda x: x[1]))
    PC_p = dict(sorted(PC_p.items(), key=lambda x: x[1]))
    #plt.bar(PC_w.keys(), PC_w.values(), width=0.5, align='center')
    y_pos  =  range(0, 2*len(PC_w), 2)
    plt.barh(y_pos, list(PC_w.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_w.keys()))
    if title:
        plt.title(f"{title}: Average PC weighted by frequency " + str(percentile))
    else:
        plt.title("Average PC weighted by frequency " + str(percentile))
    plt.xlabel("Average PC weighted by frequency")
    #plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_5")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8))
    plt.barh(y_pos, list(PC_p.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_p.keys()))
    if title:
        plt.title(f"{title}: Portion of total PC " + str(percentile))
    else:
        plt.title("Portion of total PC " + str(percentile))
    plt.xlabel("Portion of total PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
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
