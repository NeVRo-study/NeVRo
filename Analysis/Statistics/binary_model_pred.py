"""
Analysis on binary predictions of LSTM and CSP

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

# %% Import
from load_data import *

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

# # Set variables
save_plot_individual_cm = True
verbose = False

# # Fixed vars
classes = ["low", "high"]
models = ["LSTM", "CSP"]
conditions = ["nomov", "mov"]
# Set Paths
wdic_results = f"../../../Results/"
p2_predplots = f"../../../Results/Plots/ConfusionMatrix/"


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Define plotting function

def plot_cm(cm, model_name, condi, title, path2save=None):
    """Create plot of confusion matrix and save if path is given."""
    # Prepare matrix
    df_confmat = pd.DataFrame(data=cm, columns=classes, index=classes)
    df_confmat.index.name = 'Subjective arousal rating'
    df_confmat.columns.name = f'Predicted arousal rating'

    # Plot
    fig = plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.7)  # for label size
    ax = sns.heatmap(df_confmat,
                     cmap="RdBu_r",  # or: "Blues"
                     annot=True,
                     vmin=.1, vmax=.9,  # equal col-bar for each plot
                     annot_kws={"size": 24})  # "ha": 'center', "va": 'center'})
    # ax.set_ylim([0, 2])  # switch on if labelling is off, OR downgrade matplotlib==3.1.0
    ax.set_title(f"{title} Confusion Matrix of {model_name} in {condi}-condition")

    if path2save:
        fig.savefig(path2save + f"{title}_{model_name}_ConfusionMatrix_{condi}.png")
        plt.close()


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Main()
if __name__ == "__main__":

    for cond in conditions:  # Conditions

        # Read results per method/model:
        p2results = f"../../../Results/Tables/results_across_methods_{cond}.csv"

        # Get subjects per condition
        result_tab = pd.read_csv(p2results, sep=",", index_col="Subject").dropna()  # drop NaNs
        subjects = [int(sub.lstrip("NVR_S")) for sub in result_tab.index.to_list()]

        for model in models:

            wdic_val_pred = wdic_results + f"Tables/{model}/{cond}/"  # set path
            model_confmats = np.zeros(4 * len(subjects)).reshape((-1, 2, 2))  # init list of all conf.mats

            for sidx, subject in enumerate(subjects):

                # # Load full rating vector (true y)
                true_rating = load_rating_files(subjects=subject, condition=cond,
                                                bins=True)[str(subject)]["SBA"][cond]
                # Alternatively:
                # true_rating = pd.read_csv(wdic_val_pred + f"targetTable{model}_{cond}.csv",
                #                           header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()
                true_rating[np.where(true_rating == 0)] = np.nan
                true_y = true_rating[~np.isnan(true_rating)]  # remove NaNs

                # # Load model prediction (predicted y)
                # Get concatenated validation predictions for given subject and model
                model_pred = pd.read_csv(wdic_val_pred + f"predictionTable{model}_{cond}.csv",
                                         header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()
                pred_y = model_pred[~np.isnan(model_pred)]  # remove NaNs
                if model == "CSP":
                    pred_y[pred_y == 0] = -1  # due to different coding

                # Test two arrays:
                assert np.all(np.isnan(model_pred) == np.isnan(true_rating)) & (
                        len(pred_y) == len(true_y)), "model prediction doesn't fit to true rating"

                # # Print classification report of subject
                if verbose:
                    cprint(f"Classification report – {s(subject)} in {cond}-condition", fm="bo")
                    cprint(classification_report(true_y, pred_y, target_names=classes), 'y')

                # # Create confusion matrix
                confmat = confusion_matrix(true_y, pred_y, normalize='true')  # normalize=None
                model_confmats[sidx] = confmat

                # # Plot individual confusion matrix
                if save_plot_individual_cm:
                    # Create subfolder for indvidiual plots
                    p2_predplots_individ = p2_predplots + f"{cond}/"
                    if not os.path.exists(p2_predplots_individ):
                        os.makedirs(p2_predplots_individ)

                    # Create plot and save
                    plot_cm(cm=confmat, model_name=model, condi=cond, title=s(subject),
                            path2save=p2_predplots_individ)

            # # Plot mean confusion matrix per model per condition
            mean_model_confmats = np.mean(model_confmats, axis=0)
            plot_cm(cm=mean_model_confmats, model_name=model, condi=cond, title="Average",
                    path2save=p2_predplots)

    end()

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  End
