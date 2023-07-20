import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
#sns.set(rc={'figure.figsize':(10,8)}, font_scale = 1.3)

from teg.queries import *

def get_psm(df, conf, fname):
    category_cols = [] 
    for col in conf['psm_features']:
        if not is_numeric_dtype(df.dtypes[col]) and df[col].nunique() > 2:
            category_cols.append(col)
        elif not is_numeric_dtype(df.dtypes[col]):
            df[col] = df[col].astype('category').cat.codes
    print('category cols', category_cols)
    df = pd.get_dummies(df, columns = category_cols, dtype=int)
    psm = PsmPy(df, treatment='PI', indx='hadm_id', exclude = [])
    psm.logistic_ps(balance = True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.02, drop_unmatched=True)
    psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity score', names = ['PI', 'NPI'], colors=['#E69F00', '#56B4E9'])
    plt.savefig(fname + '_propensity_score')
    plt.show()
    plt.clf()
    plt.cla()
    psm.effect_size_plot(title='Standardized Mean differences across covariates before and after matching')
    plt.savefig(fname + '_mean_diff')
    plt.show()
    plt.clf()
    plt.cla()
    print(psm.matched_ids)
    print(psm.df_matched)
    print(psm.effect_size)
    return psm 
