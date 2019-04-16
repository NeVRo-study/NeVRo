# coding=utf-8
"""
Regress out 1/f
"""

from meta_functions import *
from load_data import get_filename
import matplotlib.pyplot as plt

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# TODO overwrite current working dir path in load_data

path_data = set_path2data()
# path_data = "../../../Data/"
p2ssd = path_data + "EEG/07_SSD/"
p2ssdnomov = p2ssd + "nomov/SBA/broadband/"

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


sub = 14  # rnd subject

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
new_ssd = False

# Case1: post-SSD remaining comps after rejecting
# sub_ssd = get_filename(subject=sub, filetype="SSD", band_pass=False, cond="nomov", sba=True,
#                        check_existence=True)

# # Case2: pre-SSD ICA comps
# sub_ssd = path_data + "EEG/NVR_S14_rejcomp_testfile.txt"

# # Case3: post-SSD comps befor rejecting
# sub_ssd = path_data + "EEG/NVR_S02_1_SSD_filt_cmp.csv"  # 3.1.
# sub_ssd = path_data + "EEG/NVR_S03_1_SSD_filt_cmp.csv"  # 3.2.

# # Case4: post-SSD comps befor rejecting
sub_ssd = path_data + "EEG/ssdcomps.csv"
new_ssd = True

if os.path.isfile(sub_ssd):
    # rows = components, columns value per timestep
    # first column: Nr. of component, last column is empty
    if not new_ssd:
        sub_df = np.genfromtxt(sub_ssd, delimiter="\t")[:, 1:-1].transpose()
    else:
        sub_df = np.genfromtxt(sub_ssd, delimiter=",").transpose()

    n_comp = sub_df.shape[1]

    print("subject SSD df.shape:", sub_df.shape)
    print("First 5 rows:\n", sub_df[:5, :])

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


# Plot power spectral density (Welch)
freq_peak = []
min_max_Pxx_den = 99

fig = plt.figure()
ax = plt.subplot(1, 2, 1)
# for ch in range(1, sub_df.shape[1]):
for ch in range(sub_df.shape[1]):
    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    ax.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    max_Pxx_den = max(Pxx_den)
    freq_peak.append(f[np.where(Pxx_den == max_Pxx_den)])
    min_max_Pxx_den = max_Pxx_den if max_Pxx_den < min_max_Pxx_den else min_max_Pxx_den

plt.vlines(np.mean(freq_peak), ymin=-0.01, ymax=min_max_Pxx_den, linestyles="dashed", alpha=.2)
xt = ax.get_xticks()
xt = np.append(xt, np.mean(freq_peak))
xtl = xt.tolist()
xtl[-1] = str(np.round(np.mean(freq_peak), 1))
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
ax.set_xlim([-1, 130])
ax.set_title("plt.semilogy(f ,Pxx_den)")

# Alternative: plt.psd
ax2 = plt.subplot(1, 2, 2)
# for ch in range(1, sub_df.shape[1]):
for ch in range(sub_df.shape[1]):
    ax2.psd(x=sub_df[:, ch], Fs=250.)
ax2.vlines(np.mean(freq_peak), ymin=-121, ymax=np.log(min_max_Pxx_den), linestyles="dashed", alpha=.2)
xt2 = ax2.get_xticks()
xt2 = np.append(xt2, np.mean(freq_peak))
xtl2 = xt2.tolist()
xtl2[-1] = str(np.round(np.mean(freq_peak), 1))
ax2.set_xticks(xt2)
ax2.set_xticklabels(xtl2)
ax2.set_xlim([-1, 130])
ax2.set_title("plt.psd()")

plt.show()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Detrend
# for ch in range(1, sub_df.shape[1]):
fig3 = plt.figure()
for ch in range(sub_df.shape[1]):
    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    ax3 = plt.subplot(2, 1, 1)
    ax3.semilogy(f, Pxx_den)  # == plt.plot(f, np.log(Pxx_den))
    ax3.set_title("semilogy(f, Pxx_den)")

    ax4 = plt.subplot(2, 1, 2)
    ax4.plot(f, np.exp(np.log(Pxx_den)))  # == plt.plot(f, Pxx_den)
    ax4.set_title("plot(f,  np.exp(np.log(Pxx_den))")
    # plt.plot(f, Pxx_den)

ax3.vlines(np.mean(freq_peak), ymin=- 1e-1, ymax=min_max_Pxx_den, linestyles="dashed", alpha=.2)
ax4.vlines(np.mean(freq_peak), ymin=0, ymax=min_max_Pxx_den, linestyles="dashed", alpha=.2)
xt4 = ax4.get_xticks()
xt4 = np.append(xt4, np.mean(freq_peak))
xtl4 = xt4.tolist()
xtl4[-1] = str(np.round(np.mean(freq_peak), 1))
ax3.set_xticks(xt4)
ax3.set_xticklabels(xtl4)
ax4.set_xticks(xt4)
ax4.set_xticklabels(xtl4)
ax3.set_xlim([-1, 130])
ax4.set_xlim([-1, 130])

plt.tight_layout()
plt.show()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
fig4 = plt.figure()
model = np.polyfit(f, np.log(Pxx_den), 1)
predicted = np.polyval(model, f)

model2 = np.polyfit(f, np.log(Pxx_den), 2)
predicted2 = np.polyval(model2, f)

model3 = np.polyfit(f, np.log(Pxx_den), 3)
predicted3 = np.polyval(model3, f)

ax5 = plt.subplot(2, 1, 1)
plt.plot(f, np.log(Pxx_den), label='data')
plt.plot(predicted, alpha=.8, linestyle=":", label='poly_1/linear')
plt.plot(predicted2, alpha=.8, linestyle=":", label='poly_2')
plt.plot(predicted3, alpha=.8, linestyle=":", label='poly_3')
ax5.set_title("Original")
plt.legend()
plt.tight_layout()

ax6 = plt.subplot(2, 1, 2)
plt.plot(f, np.log(Pxx_den), linestyle=":", label='data')
plt.plot(f, np.log(Pxx_den) - predicted, label='poly_1/linear')
plt.plot(f, np.log(Pxx_den) - predicted2, label='poly_2')
plt.plot(f, np.log(Pxx_den) - predicted3, label='poly_3')
plt.plot(f, np.exp(np.log(Pxx_den)), label='exp')
ax6.set_title("Detrend")
plt.legend()
plt.tight_layout()

plt.show()