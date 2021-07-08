# coding=utf-8
"""
Create plots for supplementary analysis:

•  testing the influence of the break on model decoding performance
•  testing for the influence of auto-correlation (block-CV, block-permutation)
•  testing linear decoding model (logistic regression)

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2021
"""

# %% Import

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy.stats import norm, ttest_ind

# %% Set paths & vars >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Path to tables
p2results = os.path.join(".", "Results", "Tables")  # asserts being on root level ./NeVRo/

# Models
models = ["CSP", "LogReg"]  # latter: (linear) logistic regression

# Conditions
conditions = ["nomov", "mov"]

# Data span
span = ["SBA", "SA"]

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

pass

# %% __main__ >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

if __name__ == "__main__":

    for cond in conditions:
        p2tab = os.path.join(p2results, f"results_across_methods_{cond}_supplementary_analysis.csv")

        table = pd.read_csv(p2tab, index_col="Subject")

        # Create figure
        fig, axs = plt.subplots(2, 2, num=f"{cond}", figsize=(10, 6), sharex=True, sharey=False)

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
                                      ax=ax, label="" if ax_ctn > 0 else "permuted")

                h = sns.distplot(table[data_col], bins=15, kde=False, hist=True, fit=norm,
                                 hist_kws=dict(range=(.35, .75)),
                                 color="orange" if ax_ctn % 2 == 0 else "lightgreen",
                                 fit_kws=dict(color="orange" if ax_ctn % 2 == 0 else "lightgreen"),
                                 ax=ax, label="" if ax_ctn > 1 else "original")

                # Add means
                ax.vlines(x=table[perm_data_col].mean(), ymin=0, ymax=h_perm.get_ylim()[-1], ls="dotted",
                          color="blue")

                ax.vlines(x=table[data_col].mean(), ymin=0, ymax=h.get_ylim()[-1], ls="dotted",
                          color="orange" if ax_ctn % 2 == 0 else "lightgreen")

                # Add stats: one-sided ttest
                t, p = ttest_ind(a=table[data_col], b=table[perm_data_col])  # results of two-sided
                p /= 2  # one-sided

                print(f"{cond}-condition: t_{model}={t:.3f}, p_{model}={p:.3f}")
                m = "ns" if p > .05 else "*" if (.05 >= p > .01) else "**" if (.01 >= p > .001) else "***"

                # Indicate (non-)significance in plot
                x_pos = (table[perm_data_col].mean() + table[data_col].mean()) / 2
                y_pos = max(h_perm.get_ylim()[-1], h.get_ylim()[-1]) + .5
                ax.annotate(m, xy=(x_pos, y_pos + 1.), ha="center", va="bottom")
                ax.annotate('',
                            xy=(table[perm_data_col].mean(), y_pos),
                            xytext=(table[data_col].mean(), y_pos),
                            arrowprops={'connectionstyle': 'bar', 'arrowstyle': '-',
                                        'shrinkA': 1, 'shrinkB': 1, 'linewidth': 1})

                # Set labels & titles
                ax.set_ylim(0, round(ax.get_ylim()[-1]) + 5)
                if ax_ctn % 2 == 0:
                    ax.set_ylabel(f"{sp}\nCount")
                if ax_ctn < 2:
                    ax.legend()
                    ax.set_title(model)
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("AUC")

                ax_ctn += 1

        plt.tight_layout()

        # # Save plot
        p2save = "./Results/Plots/Supplementary_Analysis/"
        os.makedirs(p2save, exist_ok=True)
        for fm in ['png', 'pdf']:
            plt.savefig(fname=os.path.join(p2save, f"SA-vs-SBA_classification_{cond}.{fm}"), dpi=300,
                        format=fm)
