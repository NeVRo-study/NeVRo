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
p2ssd = path_data + "EEG/07_SSD/"  # TEMP
p2ssdnomov = p2ssd + "nomov/SBA/broadband/"  # TEMP

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

sub = 5  # rnd subject

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Case1: post-SSD remaining comps after rejecting
new_ssd = False
# sub_ssd = get_filename(subject=sub, filetype="SSD", band_pass=False, cond="nomov", sba=True,
#                        check_existence=True)

# # Case2: post-SSD comps befor rejecting
sub_ssd = path_data + "EEG/ssdcomps_test.csv"  # subject 5
new_ssd = True

if os.path.isfile(sub_ssd):
    # rows = components, columns value per timestep
    # first column: Nr. of component, last column is empty
    if not new_ssd:
        sub_df = np.genfromtxt(sub_ssd, delimiter="\t")[:, 1:-1].transpose()
    else:
        sub_df = np.genfromtxt(sub_ssd, delimiter=",").transpose()

    sub_df = sub_df[:, int(sub_df.shape[1]/2):]  # Take subset (half)

    n_comp = sub_df.shape[1]

    print("subject SSD df.shape:", sub_df.shape)
    print("First 5 rows:\n", sub_df[:5, :])

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

tight_range = True  # Freq-range 0-50 Hz

# Get alpha peak information for subject
tab_alpha_peaks = np.genfromtxt(p2ssd + "alphaPeaks.csv", delimiter=",", names=True)
sub_apeak = tab_alpha_peaks["nomov"][tab_alpha_peaks["ID"] == sub].item()


# # Plot power spectral density (Welch)
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

# # subplot per comps
# for ch in range(1, sub_df.shape[1]):

rpl = int(n_comp/2) if n_comp % 2 == 0 else int(n_comp/2) + 1

opt = [0, 1]
for o in opt:
    freq_peak = []
    min_max_Pxx_den = 99

    figs = plt.figure(figsize=[14, 10])

    for ch in range(n_comp):
        axs = figs.add_subplot(rpl, 2, ch + 1)

        f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                           nfft=None,
                           detrend='constant', return_onesided=True, scaling='density', axis=-1,
                           average='mean')

        if tight_range:
            Pxx_den = Pxx_den[f <= 50]
            f = f[f <= 50]

        max_Pxx_den = max(Pxx_den)
        freq_peak.append(f[np.where(Pxx_den == max_Pxx_den)])
        min_max_Pxx_den = max_Pxx_den if max_Pxx_den < min_max_Pxx_den else min_max_Pxx_den

        if o == 0:
            axs.semilogy(f, Pxx_den)
        else:
            axs.plot(f, Pxx_den)
        axs.set_xlabel('frequency [Hz]')
        axs.set_ylabel('PSD [V**2/Hz]')

        axs.vlines(f[np.where(Pxx_den == max_Pxx_den)], ymin=-0.01, ymax=min_max_Pxx_den,
                   linestyles="dashed", alpha=.2)
        xt = axs.get_xticks()
        xt = np.append(xt, f[np.where(Pxx_den == max_Pxx_den)][0])
        xtl = xt.tolist()
        xtl[-1] = str(np.round(f[np.where(Pxx_den == max_Pxx_den)][0], 1))
        axs.set_xticks(xt)
        axs.set_xticklabels(xtl)

        axs.set_title("SSD_comp_{}".format(ch+1))

        axs.set_xlim([-1, 51 if tight_range else 130])
        if o == 1:
            axs.set_ylim([0, 1])

    plt.tight_layout()

# # Alternative with plt.pds()
# figs = plt.figure(figsize=[14, 10])
# for ch in range(n_comp):
#     axs = figs.add_subplot(rpl, 2, ch+1)
#     axs.psd(x=sub_df[:, ch], Fs=250.)
#     axs.set_title("SSD_comp_{}".format(ch+1))
#     axs.set_xlim([-1, 130])
# plt.tight_layout()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Detrend
# for ch in range(1, sub_df.shape[1]):
fig3 = plt.figure()
for ch in range(sub_df.shape[1]):
    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    if tight_range:
        Pxx_den = Pxx_den[f <= 50]
        f = f[f <= 50]

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
ax3.set_xlim([-1, 51 if tight_range else 130])
ax4.set_xlim([-1, 51 if tight_range else 130])

plt.tight_layout()
plt.show()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

for ch in range(sub_df.shape[1]):

    fig4 = plt.figure()

    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    if tight_range:
        Pxx_den = Pxx_den[f <= 50]
        f = f[f <= 50]

    # TODO consider average polyfit across all normalized comps
    model = np.polyfit(f, np.log(Pxx_den), 1)
    predicted = np.polyval(model, f)

    model2 = np.polyfit(f, np.log(Pxx_den), 2)
    predicted2 = np.polyval(model2, f)

    model3 = np.polyfit(f, np.log(Pxx_den), 3)
    predicted3 = np.polyval(model3, f)

    np.polyfit(f, np.log(Pxx_den), 3)

    ax5 = plt.subplot(2, 1, 1)
    plt.plot(f, np.log(Pxx_den), label='data')
    plt.plot(predicted, alpha=.8, linestyle=":", label='poly_1/linear')
    plt.plot(predicted2, alpha=.8, linestyle=":", label='poly_2')
    plt.plot(predicted3, alpha=.8, linestyle=":", label='poly_3')
    ax5.set_title("Original SSD comp{}".format(ch+1))
    plt.legend()
    plt.tight_layout()

    ax6 = plt.subplot(2, 1, 2)
    plt.plot(f, np.log(Pxx_den), linestyle=":", label='data')
    plt.plot(f, np.log(Pxx_den) - predicted, label='poly_1/linear')
    plt.plot(f, np.log(Pxx_den) - predicted2, label='poly_2')
    plt.plot(f, np.log(Pxx_den) - predicted3, label='poly_3')
    # plt.plot(f, np.exp(np.log(Pxx_den)), label='exp')
    ax6.set_title("Detrend")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()

# Smaller freq-window (0-50Hz)  + Leave alpha-out
for ch in range(sub_df.shape[1]):

    fig5 = plt.figure()

    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    # Freq.-Range (0-50Hz)
    f_small = f[f <= 50]
    Pxx_den_small = Pxx_den[f <= 50]

    # Leave alpha out
    f_small_alphout = f_small[~((sub_apeak+4 > f_small) & (f_small > sub_apeak-4))]
    Pxx_den_small_alphout = Pxx_den_small[~((sub_apeak+4 > f_small) & (f_small > sub_apeak-4))]

    # Fit polynomial(3)
    model3_small = np.polyfit(f_small, np.log(Pxx_den_small), 3)
    predicted3_small = np.polyval(model3_small, f_small)

    # Fit polynomial(3) to alpha out data
    model3_small_alphout = np.polyfit(f_small_alphout, np.log(Pxx_den_small_alphout), deg=3)
    predicted3_small_alphout = np.polyval(model3_small_alphout, f_small)  # predict on whole! freq-range

    ax7 = plt.subplot(2, 1, 1)
    plt.plot(f_small, np.log(Pxx_den_small), label='data (f<=50)')
    plt.plot(predicted3_small, alpha=.8, linestyle=":", label='poly3')
    plt.plot(predicted3_small_alphout, alpha=.8, linestyle=":", label='poly3_alpha-out')

    ax7.vlines(sub_apeak,
               ymin=ax7.get_ylim()[0], ymax=np.log(Pxx_den_small)[np.argmin(np.abs(f_small - sub_apeak))],
               linestyles="dashed", alpha=0.2)  # ymax=np.polyval(model3_small, sub_apeak)

    ax7.set_title("Original SSD comp{}".format(ch+1))
    plt.legend()
    plt.tight_layout()

    ax8 = plt.subplot(2, 1, 2)
    plt.plot(f_small, np.log(Pxx_den_small), linestyle=":", label='data (f<=50)')
    plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small, label='poly_3')
    plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small_alphout, label='poly3_alpha-out')
    ax8.vlines(sub_apeak,
               ymin=ax7.get_ylim()[0], ymax=np.log(Pxx_den_small)[np.argmin(np.abs(f_small - sub_apeak))],
               linestyles="dashed", alpha=0.2)  # ymax=np.polyval(model3_small, sub_apeak)
    # plt.plot(f, np.exp(np.log(Pxx_den)), label='exp')
    ax8.set_title("Detrend")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Define component selection criterion

figs6 = plt.figure(figsize=[14, 10])

for ch in range(n_comp):

    axs = figs6.add_subplot(rpl, 2, ch + 1)

    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    if tight_range:
        # Freq.-Range (0-50Hz)
        Pxx_den = Pxx_den[f <= 50]
        f = f[f <= 50]

    # Leave alpha out
    f_alphout = f[~((sub_apeak + 4 > f) & (f > sub_apeak - 4))]
    Pxx_den_alphout = Pxx_den[~((sub_apeak + 4 > f) & (f > sub_apeak - 4))]

    # Fit polynomial(3) to alpha out data
    model3_alphout = np.polyfit(f_alphout, np.log(Pxx_den_alphout), deg=3)
    predicted3_alphout = np.polyval(model3_alphout, f)  # predict on whole! freq-range

    log_Pxx_den_detrend = np.log(Pxx_den) - predicted3_alphout

    f_apeak = f[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]
    log_Pxx_den_apeak = log_Pxx_den_detrend[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]

    axs.plot(f, log_Pxx_den_detrend)
    axs.plot(f_apeak, log_Pxx_den_apeak)
    axs.set_title("SSD comp{}".format(ch+1))
    axs.vlines(sub_apeak,
               ymin=min(log_Pxx_den_detrend),
               ymax=log_Pxx_den_detrend[np.argmin(np.abs(f - sub_apeak))],
               linestyles="dashed", alpha=.2)
    plt.hlines(y=0, xmin=0, xmax=50, alpha=.4, linestyles=":")  # Zero-line

    log_Pxx_den_apeak_stand = log_Pxx_den_apeak - np.linspace(log_Pxx_den_apeak[0], log_Pxx_den_apeak[-1],
                                                              num=len(log_Pxx_den_apeak))

    axs.plot(f_apeak, log_Pxx_den_apeak_stand)
    # axs.fill_between(f_apeak, np.array(log_Pxx_den_apeak_stand+.5), alpha=.2)

plt.tight_layout()

