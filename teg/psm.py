import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
#sns.set(rc={'figure.figsize':(10,8)}, font_scale = 1.3)

from teg.queries import *

def get_psm(df, conf):
    for col in conf['psm_features']:
        if not is_numeric_dtype(df.dtypes[col]):
            df[col] = df[col].astype('category').cat.codes
    psm = PsmPy(df, treatment='PI', indx='hadm_id', exclude = [])
    psm.logistic_ps(balance = True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
    psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity score', names = ['PI', 'NPI'], colors=['#E69F00', '#56B4E9'])
    plt.show()
    psm.effect_size_plot(title='Standardized Mean differences across covariates before and after matching')
    plt.show()
    print(psm.matched_ids)
    print(psm.df_matched)
    print(psm.effect_size)
    return psm 
