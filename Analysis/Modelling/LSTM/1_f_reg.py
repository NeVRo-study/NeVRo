# coding=utf-8
"""
Regress out 1/f
"""

from meta_functions import *
from load_data import get_filename
import matplotlib.pyplot as plt

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


# Plot power spectral density (Welch)
freq_peak = []
min_max_Pxx_den = 99

fig, ax = plt.subplots(1, 1)

for ch in range(sub_df.shape[1]):
    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    ax.semilogy(f, Pxx_den)
    plt.ylim([-0.1e-5, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    max_Pxx_den = max(Pxx_den)
    freq_peak.append(f[np.where(Pxx_den == max_Pxx_den)])
    min_max_Pxx_den = max_Pxx_den if max_Pxx_den < min_max_Pxx_den else min_max_Pxx_den

plt.vlines(np.mean(freq_peak), ymin=-0.1e-5, ymax=min_max_Pxx_den, linestyles="dashed", alpha=.2)
xt = ax.get_xticks()
xt = np.append(xt, np.mean(freq_peak))
xtl = xt.tolist()
xtl[-1] = str(np.round(np.mean(freq_peak), 1))
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
plt.xlim([-1, 130])
plt.show()
