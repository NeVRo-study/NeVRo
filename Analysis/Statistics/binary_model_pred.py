"""
Analysis on binary predictions of LSTM and CSP

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

# %% Import
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

# Create confusion matrix
confmat = confusion_matrix(y_true, y_pred, normalize='true')  # normalize=None

# # Plot confusion matrix
df_confmat = pd.DataFrame(data=confmat, columns=classes, index=classes)
df_confmat.index.name = 'TrueValues'
df_confmat.columns.name = f'Predicted {target.upper()}'
fig = plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_confmat, cmap="Blues", annot=True,
                 annot_kws={"size": 16})  # "ha": 'center', "va": 'center'})
ax.set_ylim([0, 2])  # because labelling is off otherwise, OR downgrade matplotlib==3.1.0
fig.savefig(p2_predplots + f"{model.name}_confusion-matrix.png")
plt.close()

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  End