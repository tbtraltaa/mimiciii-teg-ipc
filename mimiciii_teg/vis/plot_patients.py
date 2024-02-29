import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14

from mimiciii_teg.schemas.PI_risk_factors import CHRONIC_ILLNESS


def plot_PI_NPI_patients(PI_patients,
                         NPI_patients,
                         PI_df,
                         NPI_df,
                         conf,
                         PI_results=None,
                         NPI_results=None,
                         title = '',
                         fname = 'Patients'):
    pi_patients = dict()
    npi_patients = dict()
    print('PI_df', PI_df)
    if PI_results and NPI_results:
        for idd in PI_results['patient_CENTRALITY_P']:
            pi_patients[idd] = PI_patients[idd]
        for idd in NPI_results['patient_CENTRALITY_P']:
            npi_patients[idd] = NPI_patients[idd]
        print('pi_patients', pi_patients)
        pi_df = PI_df[PI_df['id'].isin(pi_patients)] 
        npi_df = NPI_df[NPI_df['id'].isin(npi_patients)] 
    else:
        pi_patients = PI_patients
        npi_patients = NPI_patients
        pi_df = PI_df
        npi_df = NPI_df
    PI_c = 'red'
    NPI_c = 'blue'
    PI_l = 'PI'
    NPI_l = 'NPI'
    patient_dict = dict()
    chronic_dict = dict()
    unknown_ethnicities = ['unknown', 'patient declined to answer', 'unable to obtain', 'other']
    unknown_religions = ['other', 'not specified', 'unobtainable']
    for (pi_id, pi_vals), (npi_id, npi_vals) in zip(pi_patients.items(), npi_patients.items()): 
        for key, val in pi_vals.items():
            if conf['admission_type'] and 'admission_type' in key:
                continue
            elif 'age' in key:
                continue
            if key in CHRONIC_ILLNESS:
                if key not in chronic_dict:
                    chronic_dict[key] = [0, 0]
                if key in chronic_dict and val == 1:
                    chronic_dict[key][0] += 1
                continue
            if val is None:
                continue
            if key.lower() == 'ethnicity':
                if '/' in val:
                    val = val.split('/')[0].strip()
                elif '-' in val:
                    val = val.split('-')[0].strip()
                if val.lower() in unknown_ethnicities:
                    val = 'Unknown'
                elif val.lower() == 'hispanic or latino':
                    val = 'HISPANIC'
            if key.lower() == 'religion' and val.lower() in unknown_religions:
                val = 'Unknown'
            k = f"{key}-{val}"
            k = k.strip()
            if k in patient_dict:
                patient_dict[k][0] += 1
            else:
                patient_dict[k] = [1, 0]
        for key, val in npi_vals.items():
            if conf['admission_type'] and 'admission_type' in key:
                continue
            elif 'age' in key:
                continue
            if key in CHRONIC_ILLNESS:
                if key in chronic_dict and val == 1:
                    chronic_dict[key][1] += 1
                elif key not in chronic_dict and val == 1:
                    chronic_dict[key] = [0, 1]
                continue
            if val is None:
                continue
            if key.lower() == 'ethnicity':
                if '/' in val:
                    val = val.split('/')[0].strip()
                elif '-' in val:
                    val = val.split('-')[0].strip()
                if val.lower() in unknown_ethnicities:
                    val = 'Unknown'
                elif val.lower() == 'hispanic or latino':
                    val = 'HISPANIC'
            if key.lower() == 'religion' and val.lower() in unknown_religions:
                val = 'Unknown'
            k = f"{key}-{val}"
            k = k.strip()
            if k in patient_dict:
                patient_dict[k][1] += 1
            else:
                patient_dict[k] = [0, 1]
    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' in k.lower() or 'ethnicity' in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1][0]))
    chronic_dict = dict(sorted(chronic_dict.items(), key=lambda x: x[1][0]))
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    PI_vals = [val[0] for val in tmp.values()]
    NPI_vals = [val[1] for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=labels)
    y_pos  =  range(0, 4 * len(tmp), 4)
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    y_pos  =  range(1, 4*len(tmp) + 1, 4)
    if title:
        plt.title(f"Patient Attributes, Patient Centrality {title}")
    else:
        plt.title(f"Patient Attributes")
    plt.xlabel("Frequency")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_attrs1")
    plt.clf()
    plt.cla()

    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' not in k.lower() and 'ethnicity' not in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1][0]))
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    PI_vals = [val[0] for val in tmp.values()]
    NPI_vals = [val[1] for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=labels)
    y_pos  =  range(0, 4 * len(tmp), 4)
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    y_pos  =  range(1, 4*len(tmp) + 1, 4)
    if title:
        plt.title(f"Patient Attributes, Patient Centrality {title}")
    else:
        plt.title(f"Patient Attributes")
    plt.xlabel("Frequency")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}_attrs2")
    plt.clf()
    plt.cla()

    if conf['include_chronic_illness']:
        plt.style.use('default')
        plt.rcParams['font.size'] = 14
        plt.figure(figsize = (10, 8))
        y_pos  =  range(2, 4*len(chronic_dict) + 2, 4)
        PI_vals = [val[0] for val in chronic_dict.values()]
        NPI_vals = [val[1] for val in chronic_dict.values()]
        labels = [l.title() for l in chronic_dict.keys()]
        plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
        plt.yticks(y_pos, fontsize=10, labels=labels)
        y_pos  =  range(0, 4*len(chronic_dict), 4)
        plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
        if title:
            plt.title(f"Chronic Illness, Patient Centrality {title}")
        else:
            plt.title(f"Chronic Illness")
        plt.xlabel("Frequency")
        plt.legend()
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.grid(False)
        plt.savefig(f"{fname}_chronic")
        plt.clf()
        plt.cla()


    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    if title:
        plt.title(f"Age, Patient Centrality {title}")
    else:
        plt.title(f"Age")
    plt.hist(pi_df['age'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['age'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_age")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    if title:
        plt.title(f"Length Of Stay, Patient Centrality {title}")
    else:
        plt.title(f"Length Of Stay")
    plt.hist(pi_df['los'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['los'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Length Of Stay In Days")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_los")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    if title:
        plt.title(f" OASIS, Patient Centrality {title}")
    else:
        plt.title(f"OASIS")
    plt.hist(pi_df['oasis'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['oasis'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("OASIS")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{fname}_oasis")
    plt.clf()
    plt.cla()

