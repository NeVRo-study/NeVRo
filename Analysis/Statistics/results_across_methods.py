# coding=utf-8
"""

•  Export table of results across methods as pdf
•  Export best hyperparameter sets per subject per condition

Both table exports are aimed for publication

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

#%% Import
import pandas as pd
from utils import *

#%% Set paths & vars >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

# Path to tables
p2results = os.path.join(".", "Results", "Tables")  # asserts being on root level ./NeVRo/

# Conditions
conditions = ["nomov", "mov"]

# Which tables to process (True: still to do; False: done)
to_process = {"across_method": False,
              "LSTM": False,
              "alpha_peak": False}


#%% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

def table2tex2pdf(path_to_table, preamble=False):

    # Export as Latex (*.tex)
    table.to_latex(path_to_table + ".tex", index=False)

    # Export to pdf via pandoc (shell)
    preamb = ("--include-in-header=" + os.path.join(p2results, "preamble.tex")) if preamble else ''
    os.system(f"pandoc {path_to_table + '.tex'} {preamb} -o {path_to_table + '.pdf'}")


#%% Read and export >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

if __name__ == "__main__":

    for cond in conditions:

        if to_process["across_method"]:

            # Load table with results across methods
            p2tab = os.path.join(p2results, f"results_across_methods_{cond}")
            table = pd.read_csv(p2tab + ".csv")
            table = table.dropna().round(3)

            # Change order of columns to: SPoC, CSP, LSTM
            table = table[table.columns[[0, -3, -2, -1, 1, 2]]]

            # Renamce SPoC
            table.rename(columns=dict(zip(table.columns,
                                          [col.replace("SPOC", "SPoC") for col in table.columns])),
                         inplace=True)

            # Table to pdf
            table2tex2pdf(path_to_table=p2tab)

        # %% LSTM Best HP Sets >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

        if to_process["LSTM"]:

            # # Load LSTM table
            p2tabdir = os.path.join(p2results,
                                    f"LSTM/Random_Search_Tables/{cond}/1_narrow_search/classification/"
                                    f"per_subject/")
            p2tab = find(fname="Best_2_HPsets_over", folder=p2tabdir, typ="file", exclusive=False,
                         fullname=False, verbose=True)
            p2tab = [p for p in p2tab if "unique" not in p][0]
            # e.g. ".../Best_2_HPsets_over_26_Subjects_mean_acc_0.590_Narrow_Search_Table_nomov_BiCl.csv"
            table = pd.read_csv(p2tab, sep=";")
            # print(table.columns)
            table.drop(columns=['round', 'cond', 'seed', 'task', 'shuffle', 'repet_scalar', 's_fold',
                                'balanced_cv', 'batch_size', 'successive', 'successive_mode', 'rand_batch',
                                'plot',   'filetype', 'hilbert_power', 'band_pass', 'hrcomp', 'fixncol',
                                'summaries', 'path_specificities', 'mean_val_acc', 'meanline_acc'],
                       inplace=True)

            table.drop(index=range(1, len(table)+1, 2), inplace=True)  # second of each subject is worse
            table.sort_values(by="subject", inplace=True)  # sort by subject
            table.subject = [f"NVR_{s(sub)}" for sub in table.subject]  # change subj. nr to e.g. "NVR_S12"

            # Renamce columns
            # print(table.columns)
            table.rename(columns=dict(zip(table.columns,
                                          ['Subject', 'LSTM', 'FC', 'l.rate',
                                           'reg.', 'reg. strength',
                                           'activ.func.', 'components', 'mean accuracy'])),
                         inplace=True)

            # Round
            table['l.rate'] = [f"{lr:.0e}".replace("e-0", "e-") for lr in table['l.rate']]
            table['reg. strength'] = table['reg. strength'].round(3)

            # Table path
            p2tab = os.path.join(p2results, f"LSTM_best_hyperparmeters_per_subject_{cond}")

            # Table to pdf
            table2tex2pdf(path_to_table=p2tab)

    # %% Alpha peaks >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

    if to_process["alpha_peak"]:

        p2tab = find(fname="alphapeaks_FOOOF_fres024_813.csv", folder=p2results, typ="file",
                     exclusive=True, fullname=True, abs_path=True, verbose=True)
        # same table as in "./Data/EEG/07_SSD/alphapeaks/"
        assert os.path.isfile(p2tab), f"alpha-peak table '{p2tab}' not found!"

        # Load & Round
        table = pd.read_csv(p2tab).round(3)

        # Renamce SPoC
        table.rename(columns=dict(zip(table.columns,
                                      ['Subject', 'resting state', 'nomov', 'mov'])),
                     inplace=True)

        # Table to pdf
        table2tex2pdf(path_to_table=p2tab[:-4], preamble=True)  # [:-4]: remove .csv here

    end()
# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
