import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
sns.set(rc={'figure.figsize':(10,8)}, font_scale = 1.3)
#plt.style.use('default')
#plt.rcParams['font.size'] = 12

from mimiciii_teg.queries.queries import *

def get_psm(df, fname):
    dff = pd.DataFrame()
    for col in df.columns:
        if col == 'PI' or col == 'hadm_id':
            dff[col] = df[col]
        elif col == 'los' or col == 'oasis':
            dff[col.upper()] = df[col]
        else:
            dff[col.title()] = df[col]
    category_cols = [] 
    for col in dff.columns:
        if not is_numeric_dtype(dff.dtypes[col]) and dff[col].nunique() > 2:
            category_cols.append(col)
        elif not is_numeric_dtype(dff.dtypes[col]):
            dff[col] = dff[col].astype('category').cat.codes
    dff = pd.get_dummies(dff, columns = category_cols, dtype=int)
    print('Category columns', category_cols)
    psm = PsmPy(dff, treatment='PI', indx='hadm_id', exclude = [])
    psm.logistic_ps(balance = True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.02, drop_unmatched=True)
    psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity score', names = ['PI', 'NPI'], colors=['#E69F00', '#56B4E9'])
    plt.savefig(fname + '_propensity_score')
    plt.clf()
    plt.cla()

    psm.effect_size_plot(title='Standardized Mean differences across covariates before and after matching')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(left=0.3)
    plt.savefig(fname + '_mean_diff')
    plt.clf()
    plt.cla()
    print(psm.effect_size)
    return psm 
