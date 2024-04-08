import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14

from mimiciii_teg.teg.paths import get_paths_centrality


def plot_path_CENTRALITY(events, conf, CENTRALITY, paths, fname='Path_Figure'):
    SCP, C, P = get_paths_centrality(events, conf, CENTRALITY, paths)
    plt.clf()
    plt.cla()
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Path Centrality Distribution " + str(conf['path_percentile']), fontsize=14)
    plt.hist(C, bins=30, rwidth=0.7)
    plt.xlabel("Log Centrality Values of Paths", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    # draw the percentile line
    plt.axvline(P[0], color='red', linestyle='dashed', linewidth=1,
                label=f"{conf['path_percentile'][0]}th Percentile")
    plt.legend()
    plt.savefig(f"{fname}_Path_Centrality")
    plt.clf()
    plt.cla()

    '''
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Path Centrality Distribution (log_10)" + str(conf['path_percentile']),
              fontsize=14)

    plt.hist(np.log10(C), bins=30, rwidth=0.7)
    plt.xlabel("Path Centrality Values", fontsize=14)
    #plt.yscale("log")
    plt.ylabel("Frequency")
    # draw the percentile line
    plt.axvline(np.log10(P[0]), color='red', linestyle='dashed', linewidth=1,
                label=f"{conf['path_percentile'][0]}th Percentile")
    plt.legend()
    plt.savefig(f"{fname}_Path_Centrality_log_10")
    plt.clf()
    plt.cla()
    '''

    C_P = [math.exp(val) for i, val in enumerate(C) if P[0] <= val <= P[1]] 
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Path Centrality Distribution " + str(conf['path_percentile']), fontsize=14)
    plt.hist(C_P, bins=30, rwidth=0.7)
    plt.xlabel("Centrality Values of Paths", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    # draw the percentile line
    plt.axvline(math.exp(P[0]), color='red', linestyle='dashed', linewidth=1,
                label=f"{conf['path_percentile'][0]}th Percentile")
    plt.legend()
    plt.savefig(f"{fname}_Path_Centrality_P")
    plt.clf()
    plt.cla()

    '''
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (12, 8))
    plt.title(f"Path Centrality Distribution (log_10) " + str(conf['path_percentile']),
              fontsize=14)

    plt.hist(np.log10(C_P), bins=30, rwidth=0.7)
    plt.xlabel("Path Centrality (log_10)", fontsize=14)
    #plt.yscale("log")
    plt.ylabel("Frequency")
    # draw the percentile line
    plt.axvline(np.log10(P[0]), color='red', linestyle='dashed', linewidth=1,
                label=f"{conf['path_percentile'][0]}th Percentile")
    plt.legend()
    plt.savefig(f"{fname}_Path_Centrality_P_log_10")
    plt.clf()
    plt.cla()
    '''
