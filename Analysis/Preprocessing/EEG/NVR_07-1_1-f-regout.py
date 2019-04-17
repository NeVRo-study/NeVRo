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

    sub_df = sub_df[:, :int(sub_df.shape[1]/2)]  # Take subset (first half)
    # sub_df = sub_df[:, int(sub_df.shape[1]/2):]  # Take subset (second half)

    n_comp = sub_df.shape[1]

    print("subject SSD df.shape:", sub_df.shape)
    print("First 5 rows:\n", sub_df[:5, :])

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

tight_range = True  # Freq-range 0-50 Hz

# Get alpha peak information for subject
tab_alpha_peaks = np.genfromtxt(p2ssd + "alphaPeaks.csv", delimiter=",", names=True)
sub_apeak = tab_alpha_peaks["nomov"][tab_alpha_peaks["ID"] == sub].item()


# # Plot power spectral density (Welch)
min_apeak = 99  # init
psd_alternative = False  # True: plt.psd() [is equivalent]

fig = plt.figure()
ax = plt.subplot(1, 2 if psd_alternative else 1, 1)
# for ch in range(1, sub_df.shape[1]):
for ch in range(sub_df.shape[1]):
    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    if tight_range:
        Pxx_den = Pxx_den[f <= 50]
        f = f[f <= 50]

    ax.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    Pxx_den_apeak = Pxx_den[np.argmin(np.abs(f - sub_apeak))]
    min_apeak = Pxx_den_apeak if Pxx_den_apeak < min_apeak else min_apeak

plt.vlines(sub_apeak, ymin=-0.01, ymax=min_apeak, linestyles="dashed", alpha=.2)
xt = ax.get_xticks()
xt = np.append(xt, sub_apeak)
xtl = xt.tolist()
xtl[-1] = str(np.round(sub_apeak, 1))
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
ax.set_xlim([-1, 51 if tight_range else 130])
ax.set_title("plt.semilogy(f, Pxx_den)")

# Alternative: plt.psd
if psd_alternative:
    ax2 = plt.subplot(1, 2, 2)
    # for ch in range(1, sub_df.shape[1]):
    for ch in range(sub_df.shape[1]):
        ax2.psd(x=sub_df[:, ch], Fs=250.)
    ax2.vlines(sub_apeak, ymin=-121, ymax=np.log(min_apeak), linestyles="dashed", alpha=.2)
    xt2 = ax2.get_xticks()
    xt2 = np.append(xt2, sub_apeak)
    xtl2 = xt2.tolist()
    xtl2[-1] = str(np.round(sub_apeak, 1))
    ax2.set_xticks(xt2)
    ax2.set_xticklabels(xtl2)
    ax2.set_xlim([-1, 51 if tight_range else 130])
    ax2.set_title("plt.psd()")

plt.show()

# # Subplot per component
# for ch in range(1, sub_df.shape[1]):

rpl = int(n_comp/2) if n_comp % 2 == 0 else int(n_comp/2) + 1

opt = [0, 1]
for o in opt:

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

        Pxx_den_apeak = Pxx_den[np.argmin(np.abs(f - sub_apeak))]

        if o == 0:
            axs.semilogy(f, Pxx_den)
        else:
            axs.plot(f, Pxx_den)
        axs.set_xlabel('frequency [Hz]')
        axs.set_ylabel('PSD [V**2/Hz]')

        axs.vlines(sub_apeak, ymin=-0.01, ymax=Pxx_den_apeak,
                   linestyles="dashed", alpha=.2)
        xt = axs.get_xticks()
        xt = np.append(xt, sub_apeak)
        xtl = xt.tolist()
        xtl[-1] = str(np.round(sub_apeak, 1))
        axs.set_xticks(xt)
        axs.set_xticklabels(xtl)

        axs.set_title("{}SSD_comp_{}".format("semilog " if o == 0 else "", ch+1))

        axs.set_xlim([-1, 51 if tight_range else 130])
        # if o == 1:
        #     axs.set_ylim([0, 1])

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

    Pxx_den_apeak = Pxx_den[np.argmin(np.abs(f - sub_apeak))]

    ax3 = plt.subplot(2, 1, 1)
    ax3.semilogy(f, Pxx_den)  # == plt.plot(f, np.log(Pxx_den))
    ax3.set_title("semilogy(f, Pxx_den)")

    ax4 = plt.subplot(2, 1, 2)
    ax4.plot(f, np.exp(np.log(Pxx_den)))  # == plt.plot(f, Pxx_den)
    ax4.set_title("plot(f,  np.exp(np.log(Pxx_den))")
    # plt.plot(f, Pxx_den)

ax3.vlines(sub_apeak, ymin=- 1e-1, ymax=Pxx_den_apeak, linestyles="dashed", alpha=.2)
ax4.vlines(sub_apeak, ymin=0, ymax=Pxx_den_apeak, linestyles="dashed", alpha=.2)
xt4 = ax4.get_xticks()
xt4 = np.append(xt4, sub_apeak)
xtl4 = xt4.tolist()
xtl4[-1] = str(np.round(sub_apeak, 1))
ax3.set_xticks(xt4)
ax3.set_xticklabels(xtl4)
ax4.set_xticks(xt4)
ax4.set_xticklabels(xtl4)
ax3.set_xlim([-1, 51 if tight_range else 130])
ax4.set_xlim([-1, 51 if tight_range else 130])

plt.tight_layout()
plt.show()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

figs4 = plt.figure(figsize=[14, 10])

for ch in range(sub_df.shape[1]):

    axs = figs4.add_subplot(rpl, 2, ch + 1)

    f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                       nfft=None,
                       detrend='constant', return_onesided=True, scaling='density', axis=-1,
                       average='mean')

    if tight_range:
        Pxx_den = Pxx_den[f <= 50]
        f = f[f <= 50]

    Pxx_den_apeak = np.log(Pxx_den)[np.argmin(np.abs(f - sub_apeak))]

    # TODO consider average polyfit across all normalized comps
    model = np.polyfit(f, np.log(Pxx_den), 1)
    predicted = np.polyval(model, f)

    model2 = np.polyfit(f, np.log(Pxx_den), 2)
    predicted2 = np.polyval(model2, f)

    model3 = np.polyfit(f, np.log(Pxx_den), 3)
    predicted3 = np.polyval(model3, f)

    np.polyfit(f, np.log(Pxx_den), 3)

    # Plot
    axs.plot(f, np.log(Pxx_den), linestyle="-.", label='data')

    axs.plot(predicted, alpha=.8, linestyle=":", c="g", label='poly_1/linear')
    axs.plot(predicted2, alpha=.8, linestyle=":", c="y", label='poly_2')
    axs.plot(predicted3, alpha=.8, linestyle=":", c="m", label='poly_3')
    axs.set_title("Detrend SSD comp{}".format(ch+1))

    axs.plot(f, np.log(Pxx_den) - predicted, c="g", label='poly_1/linear')
    axs.plot(f, np.log(Pxx_den) - predicted2, c="y", label='poly_2')
    axs.plot(f, np.log(Pxx_den) - predicted3, c="m", label='poly_3')

    # TODO adapt
    axs.vlines(sub_apeak, ymin=min(np.log(Pxx_den)), ymax=Pxx_den_apeak, linestyles="dashed", alpha=.2)

    if ch == 0:
        axs.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Smaller freq-window (0-50Hz)  + Leave alpha-out

figs5 = plt.figure(figsize=[14, 10])

for ch in range(sub_df.shape[1]):

    axs = figs5.add_subplot(rpl, 2, ch + 1)

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

    plt.plot(f_small, np.log(Pxx_den_small), linestyle="-.", label='data (f<=50)')
    plt.plot(predicted3_small, alpha=.8, linestyle=":", c="m", label='poly3')
    plt.plot(predicted3_small_alphout, alpha=.8, c="g", linestyle=":", label='poly3_alpha-out')

    plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small, c="m", label='poly_3')
    plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small_alphout, c="g", label='poly3_alpha-out')
    axs.vlines(sub_apeak,
               ymin=axs.get_ylim()[0], ymax=np.log(Pxx_den_small)[np.argmin(np.abs(f_small - sub_apeak))],
               linestyles="dashed", alpha=0.2)  # ymax=np.polyval(model3_small, sub_apeak)
    axs.set_title("Detrend SSD comp{}".format(ch+1))
    if ch == 0:
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

