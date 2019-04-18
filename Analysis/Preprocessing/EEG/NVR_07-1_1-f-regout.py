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
p2ssd = path_data + "EEG/07_SSD/"  # TODO remove after testphase
p2ssdnomov = p2ssd + "nomov/SBA/broadband/"  # TODO remove after testphase

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Set hyperparameters
new_ssd = True  # TODO remove after testphase
tight_range = True  # Freq-range 0-50 Hz
subjects = np.arange(1, 45+1)
bpass = False
condition = "nomov"

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Create table for selected components
# Colums: ID | selected components  | number of selected components | number of all SSD components
col_names = ["ID", "selected_comps", "n_sel_comps", "n_all_comps"]
tab_select_ssd = np.zeros(shape=(subjects[-1], 4))  # Init table
tab_select_ssd.fill(np.nan)  # convert entries to nan
tab_select_ssd[:, 0] = subjects

# col_names = [(cnam, "<f8") for cnam in ["ID", "selected_comps", "n_sel_comps", "n_all_comps"]]
# tab_select_ssd.dtype = col_names
# tab_select_ssd["ID"] = subjects.reshape((len(subjects), 1))
# print(tab_select_ssd["ID"])
# print(tab_select_ssd)
# np.savetxt("test.csv", tab_select_ssd, delimiter=",", fmt="%s",
#            header=",".join(tab_select_ssd.dtype.names))
# np.genfromtxt("test.csv")  # comes without col names


# # Test Case: post-SSD comps before rejecting # TODO remove after testphase
# sub = 5
# sub_ssd = path_data + "EEG/ssdcomps_test.csv"  # subject 5
# new_ssd = False

# # Full Case: fresh SSDs components
for sub in subjects:
    try:
        sub_ssd = get_filename(subject=sub, filetype="SSD", band_pass=bpass, cond=condition, sba=True,
                               check_existence=True)
    except FileExistsError:
        continue  # if file doesn't exist, continue with next subject
    print(sub_ssd)

    if os.path.isfile(sub_ssd):  # TODO remove after testphase, redundant due to check_existence=True
        # rows = components, columns value per timestep
        # first column: Nr. of component, last column is empty
        if not new_ssd:  # TODO remove after testphase
            sub_df = np.genfromtxt(sub_ssd, delimiter="\t")[:, 1:-1].transpose()
        else:
            sub_df = np.genfromtxt(sub_ssd, delimiter=",").transpose()

        # number of components
        n_comp = sub_df.shape[1]

        # TODO remove after testphase
        n_comp = int(n_comp / 2)  # half
        sub_df = sub_df[:, :n_comp]  # Take subset (first half)
        # sub_df = sub_df[:, n_comp:]  # Take subset (second half)

        print("{} SSD df.shape: {}".format(s(sub), sub_df.shape))
        print("First 5 rows:\n", sub_df[:5, :])

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # Get alpha peak information for given subject
    tab_alpha_peaks = np.genfromtxt(p2ssd + "alphaPeaks.csv", delimiter=",", names=True)
    sub_apeak = tab_alpha_peaks[condition][tab_alpha_peaks["ID"] == sub].item()

    # # Plot power spectral density (Welch)
    min_apeak = 99  # init
    psd_alternative = False  # True: plt.psd() [is equivalent]

    fig = plt.figure()
    ax = plt.subplot(1, 2 if psd_alternative else 1, 1)
    # for ch in range(1, n_comp):
    for ch in range(n_comp):
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
    ax.set_title("{} | {} | {} | plt.semilogy(f, Pxx_den)".format(s(sub), condition,
                                                                  "narrowband" if bpass else "broadband"))

    # Alternative: plt.psd
    if psd_alternative:
        ax2 = plt.subplot(1, 2, 2)
        # for ch in range(1, n_comp):
        for ch in range(n_comp):
            ax2.psd(x=sub_df[:, ch], Fs=250.)
        ax2.vlines(sub_apeak, ymin=-121, ymax=np.log(min_apeak), linestyles="dashed", alpha=.2)
        xt2 = ax2.get_xticks()
        xt2 = np.append(xt2, sub_apeak)
        xtl2 = xt2.tolist()
        xtl2[-1] = str(np.round(sub_apeak, 1))
        ax2.set_xticks(xt2)
        ax2.set_xticklabels(xtl2)
        ax2.set_xlim([-1, 51 if tight_range else 130])
        ax2.set_title("S{} | {} | {} |plt.psd()".format(str(sub).zfill(2), condition,
                                                        "narrowband" if bpass else "broadband"))

    plt.tight_layout()
    plt.show()

    # # Subplot per component
    # for ch in range(1, n_comp):

    rpl = int(n_comp/2) if n_comp % 2 == 0 else int(n_comp/2) + 1

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

        axs.semilogy(f, Pxx_den)
        # axs.psd(x=sub_df[:, ch], Fs=250.)  # Alternative

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

        axs.set_title("{} | SSD component {}".format(s(sub), ch+1))

        axs.set_xlim([-1, 51 if tight_range else 130])

        plt.tight_layout()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # Detrend

    figs3 = plt.figure(figsize=[14, 10])

    for ch in range(n_comp):

        axs = figs3.add_subplot(rpl, 2, ch + 1)

        f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                           nfft=None,
                           detrend='constant', return_onesided=True, scaling='density', axis=-1,
                           average='mean')

        if tight_range:
            Pxx_den = Pxx_den[f <= 50]
            f = f[f <= 50]

        Pxx_den_apeak = np.log(Pxx_den)[np.argmin(np.abs(f - sub_apeak))]

        # Linear fit / poly(1)
        # model = np.polyfit(f, np.log(Pxx_den), 1)
        # predicted = np.polyval(model, f)

        # Quadratic fit / poly(2)
        model2 = np.polyfit(f, np.log(Pxx_den), 2)
        predicted2 = np.polyval(model2, f)

        # Cubic fit / poly(3)
        model3 = np.polyfit(f, np.log(Pxx_den), 3)
        predicted3 = np.polyval(model3, f)

        # Plot
        axs.plot(f, np.log(Pxx_den), linestyle="-.", label='data')

        # axs.plot(predicted, alpha=.8, linestyle=":", c="g", label='poly_1/linear')
        axs.plot(predicted2, alpha=.8, linestyle=":", c="y", label='poly_2')
        axs.plot(predicted3, alpha=.8, linestyle=":", c="m", label='poly_3')
        axs.set_title("Detrend SSD comp{}".format(ch+1))

        # axs.plot(f, np.log(Pxx_den) - predicted, c="g", label='poly_1/linear')
        axs.plot(f, np.log(Pxx_den) - predicted2, c="y", label='detrend/poly_2')
        axs.plot(f, np.log(Pxx_den) - predicted3, c="m", label='detrend/poly_3')

        # Add subject's alpha peak
        axs.vlines(sub_apeak, ymin=np.min([np.log(Pxx_den),
                                           np.log(Pxx_den) - predicted2,
                                           np.log(Pxx_den) - predicted3]),
                   ymax=Pxx_den_apeak, linestyles="dashed",
                   alpha=.2)

        if ch == 0:
            axs.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    # Smaller freq-window (0-50Hz)  + Leave alpha-out

    figs4 = plt.figure(figsize=[14, 10])

    for ch in range(n_comp):

        axs = figs4.add_subplot(rpl, 2, ch + 1)

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
        predicted3_small_alphout = np.polyval(model3_small_alphout, f_small)  # pred on full! freq-range

        plt.plot(f_small, np.log(Pxx_den_small), linestyle="-.", label='data (f<=50)')
        plt.plot(predicted3_small, alpha=.8, linestyle=":", c="m", label='poly3')
        plt.plot(predicted3_small_alphout, alpha=.8, c="g", linestyle=":", label='poly3_alpha-out')

        plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small, c="m",
                 label='detrend/poly_3')
        plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small_alphout, c="g",
                 label='detrend/poly3_alpha-out')
        axs.vlines(sub_apeak,
                   ymin=axs.get_ylim()[0],
                   ymax=np.log(Pxx_den_small)[np.argmin(np.abs(f_small - sub_apeak))],
                   linestyles="dashed", alpha=0.2)  # ymax=np.polyval(model3_small, sub_apeak)
        axs.set_title("Detrend SSD comp{}".format(ch+1))
        if ch == 0:
            plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # Define component selection criterion:
    # If bump around alpha peak is above zero + small error term: select component

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

        # Select
        # TODO write in table

        log_Pxx_den_detrend_above_zero = log_Pxx_den_detrend[log_Pxx_den_detrend > 0]
        f_above_zero = f[log_Pxx_den_detrend > 0]
        # log_Pxx_den_detrend_above_zero /= max(log_Pxx_den_detrend_above_zero)  # normalize
        # axs.plot(f_above_zero, log_Pxx_den_detrend_above_zero)
        min(log_Pxx_den_detrend_above_zero)

        error_term = 0.01  # TODO define more systematically
        if np.any(log_Pxx_den_apeak > 0 + error_term):
            # write ch (component) as selected
            selected = True
            pass
        else:
            # Throw ch (component) out
            selected = False
            pass

        axs.plot(f, log_Pxx_den_detrend)
        axs.plot(f_apeak, log_Pxx_den_apeak, c="g" if selected else "r")
        axs.set_title("{} | {} | {} | SSD comp{}".format(s(sub), condition,
                                                         "narrowband" if bpass else "broadband", ch+1))
        axs.vlines(sub_apeak,
                   ymin=min(log_Pxx_den_detrend),
                   ymax=log_Pxx_den_detrend[np.argmin(np.abs(f - sub_apeak))],
                   linestyles="dashed", alpha=.2)
        plt.hlines(y=0, xmin=0, xmax=50, alpha=.4, linestyles=":")  # Zero-line

        log_Pxx_den_apeak_stand = log_Pxx_den_apeak - np.linspace(log_Pxx_den_apeak[0],
                                                                  log_Pxx_den_apeak[-1],
                                                                  num=len(log_Pxx_den_apeak))

        axs.plot(f_apeak, log_Pxx_den_apeak_stand, c="g" if selected else "r", alpha=.2)

        # np.var()
        # axs.fill_between(f_apeak, np.array(log_Pxx_den_apeak_stand+.5), alpha=.2)

    plt.tight_layout()


# Save Table of selected components
np.savetxt(fname="SSD_selected_components.csv", X=tab_select_ssd, header=",".join(col_names),
           delimiter=",")
