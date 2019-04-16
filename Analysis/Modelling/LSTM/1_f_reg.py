# coding=utf-8
"""
Regress out 1/f
"""

from meta_functions import *
from load_data import get_filename

path_data = set_path2data()
p2ssd = path_data + "EEG/07_SSD/"
p2ssdnomov = p2ssd + "nomov/SBA/broadband/"

sub = 14  # rnd subject

sub_ssd = get_filename(subject=sub, filetype="SSD", band_pass=False, cond="nomov", sba=True,
                       check_existence=True)

if os.path.exists(sub_ssd):
    if np.genfromtxt(sub_ssd, delimiter="\t").ndim > 1:
        n_comp = np.genfromtxt(sub_ssd, delimiter="\t").shape[0]
    else:
        n_comp = 1

if os.path.isfile(sub_ssd):
    # rows = components, columns value per timestep
    # first column: Nr. of component, last column is empty
    sub_df = np.genfromtxt(sub_ssd, delimiter="\t")[:, 1:-1].transpose()

print("subject SSD df.shape:", sub_df.shape)
print("First 5 rows:\n", sub_df[:5, :])
