# coding=utf-8
"""
Create plots for supplementary analysis:

•  testing the influence of the break on model decoding performance
•  testing for the influence of auto-correlation (block-CV, block-permutation)
•  testing linear decoding model (logistic regression)

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2021
"""

#%% Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy.stats import norm

#%% Set paths & vars >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

# Path to tables
p2results = os.path.join(".", "Results", "Tables")  # asserts being on root level ./NeVRo/

# Models
models = ["CSP", "LogReg"]  # latter: (linear) logistic regression

# Conditions
conditions = ["nomov", "mov"]

# Data span
span = ["SBA", "SA"]

#%% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

pass

#%% __main__ >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# TODO add one-sided ttest
if __name__ == "__main__":

    for cond in conditions:
        p2tab = os.path.join(p2results, f"results_across_methods_{cond}_supplementary_analysis.csv")

        table = pd.read_csv(p2tab, index_col="Subject")

        # Create figure
        fig, axs = plt.subplots(2, 2, num=f"{cond}", figsize=(6, 6), sharex=True, sharey=True)

        ax_ctn = 0
        for sp in span:
            for model in models:

                ax = axs.ravel()[ax_ctn]

                data_col = f"{sp}_BLOCK_{model.upper()}_auc"
                perm_data_col = data_col.replace("_auc", "_perm_auc")

                # h_perm = sns.histplot(table[perm_data_col], binwidth=.02, kde=True, binrange=(.35, .75),
                #                       color="blue", ax=ax)
                #
                # h = sns.histplot(table[data_col], binwidth=.02, kde=True, binrange=(.35, .75),
                #                  color="orange" if ax_ctn%2 == 0 else "lightgreen", ax=ax)

                h_perm = sns.distplot(table[perm_data_col], bins=15, kde=False, hist=True, fit=norm,
                                      color="blue",
                                      hist_kws=dict(range=(.35, .75)), fit_kws=dict(color="blue"),
                                      ax=ax)

                h = sns.distplot(table[data_col], bins=15, kde=False, hist=True, fit=norm,
                                 hist_kws=dict(range=(.35, .75)),
                                 color="orange" if ax_ctn % 2 == 0 else "lightgreen",
                                 fit_kws=dict(color="orange" if ax_ctn % 2 == 0 else "lightgreen"), ax=ax)

                ax.vlines(x=table[perm_data_col].mean(), ymin=0, ymax=h_perm.get_ylim()[-1], ls="dotted",
                          color="blue")

                ax.vlines(x=table[data_col].mean(), ymin=0, ymax=h.get_ylim()[-1], ls="dotted",
                          color="orange" if ax_ctn % 2 == 0 else "lightgreen")

                # Set labels & titles
                ax.set_xlabel("AUC")
                ax.set_ylabel(f"{sp}\nCount")
                if ax_ctn < 2:
                    ax.set_title(model)

                ax_ctn += 1
