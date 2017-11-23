"""
After we run the gridsearch on S36, we find now the best hyperparamter from the processed files.
"""

import os
import subprocess

wdic = "./LSTM/S36/already_plotted/"
# wdic = "./LSTM/S02/already_plotted/"
wdic_plot = "../../Results/Plots/LSTM/Hyperparameter_Search/"

acc_name_list = []
acc_list = []

for file_name in os.listdir(wdic):
    if "txt" in file_name:
        with open(wdic+file_name) as f:
            for line_terminated in f:
                line = line_terminated.rstrip("\n")
                if "mean(Accuracy)" in line:
                    accuracy = line.split(" ")[1]
                    # print(accuracy)

                    # Fill in lists
                    acc_list.append(float(accuracy))
                    acc_name_list.append(file_name)

acc_list_sorted, acc_name_list_sorted = map(list, zip(*sorted(zip(acc_list, acc_name_list))[::-1]))

for i, j in zip(acc_list_sorted[:5], acc_name_list_sorted[:5]):
    print(i, "\t\t\t", j)

for file_n in acc_name_list_sorted[:10]:
    for plot_file in os.listdir(wdic_plot):
        try:
            identifier = file_n.split("folds_")[1].split(".")[0]
        except IndexError:
            identifier = file_n.split("_S36")[0]

        if identifier in plot_file and "_all_train_val_" in plot_file:
            current_plot_file = wdic_plot+plot_file
            subprocess.Popen(["open", current_plot_file])  # 'open' only works for Mac
