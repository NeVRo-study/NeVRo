# coding=utf-8
"""
Selection of SSD components

1) Regress out 1/f-curve per component
2) Criterion 1: Check whether bump around given alpha peak is greater than zero
3) Criterion 2: then check whether flanks outside of alpha area is smaller
4) Select and write in table & save plots of components

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2019
"""

from meta_functions import *
from load_data import get_filename
import matplotlib.pyplot as plt

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Set paths
path_data = set_path2data()  # path_data = "../../../Data/"
p2ssd = path_data + "EEG/07_SSD/"

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Set hyperparameters
sanity_check = False  # plot additional infos (see below)
tight_range = True  # Freq-range 0 – max_range Hz
max_range = 40  # 40 Hz: ignores the line-noise related bump in data | 20 Hz: Low-Pass | 130 Hz: ~max
save_plots = False
subjects = np.arange(1, 45+1)  # ALL
# subjects = np.arange(1, 20+1)  # subjects = np.arange(21, 45+1)  # subsets
# subjects = np.array([6, 15, 18, 21, 22, 26, 27, 31, 35])  # subset: check selections
subjects = np.array([7, 14, 15, 21, 25])  # subset: check alpha peak info
# subjects = np.array([8])  # subset: single subject: 6, 11
condition = "nomov"
if save_plots:
    plt_folder = p2ssd + "{0}/selection_plots_{0}/".format(condition)
    if not os.path.exists(plt_folder):
        os.mkdir(plt_folder)

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Create table for selected components
# Colums: ID | selected components  | number of selected components | number of all SSD components
tab_select_name = p2ssd + "{0}/SSD_selected_components_{0}.csv".format(condition)
col_names = ["ID", "selected_comps", "n_sel_comps", "n_all_comps"]

if os.path.isfile(tab_select_name):
    tab_select_ssd = np.genfromtxt(tab_select_name, delimiter=";", dtype='<U{}'.format(
        len(",".join(str(x) for x in np.arange(1, 25+1)))))  # == '<Uxx' needed if 25 comps selected
else:
    tab_select_ssd = np.zeros(shape=(subjects[-1], 4), dtype='<U{}'.format(
        len(",".join(str(x) for x in np.arange(1, 25+1)))))  # Init table
    tab_select_ssd.fill(np.nan)  # convert entries to nan
    tab_select_ssd[:, 0] = subjects

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Run: Select SSDs components per subject, plot & save in table
for sub in subjects:
    try:
        sub_ssd = get_filename(subject=sub, filetype="SSD", band_pass=False, cond=condition, sba=True,
                               check_existence=True)
    except FileExistsError:
        cprint("No SSD data for {} in {} condition!".format(s(sub), condition), "r")
        continue  # if file doesn't exist, continue with next subject

    # columns: components; rows: value per timestep
    sub_df = np.genfromtxt(sub_ssd, delimiter=",").transpose()

    # number of components
    n_comp = sub_df.shape[1]

    # # Get alpha peak information for given subject
    tab_alpha_peaks = np.genfromtxt(p2ssd + "alphaPeaks.csv", delimiter=",", names=True)
    sub_apeak = tab_alpha_peaks[condition][tab_alpha_peaks["ID"] == sub].item()
    if sub_apeak == 0.:
        cprint("No alpha peak information for {} in {} condition!".format(s(sub), condition), "r")
        continue  # No alpha peak information go to next subject

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
            Pxx_den = Pxx_den[f <= max_range]
            f = f[f <= max_range]

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
    ax.set_xlim([-1, max_range+1 if tight_range else 130])
    ax.set_title("{} | {} | plt.semilogy(f, Pxx_den)".format(s(sub), condition))

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
        ax2.set_xlim([-1, max_range+1 if tight_range else 130])
        ax2.set_title("S{} | {} | plt.psd()".format(str(sub).zfill(2), condition))

    plt.tight_layout()
    plt.show()
    if save_plots:
        plt.savefig(fname=plt_folder + "{}_SSD_powerspec.png".format(s(sub)))
        plt.close()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # Detrend
    # Subplot per component

    # Define figure grid-size: with rpl x cpl cells
    rpl = 1
    cpl = 1
    while (rpl*cpl) < n_comp:
        if rpl == cpl:
            rpl += 1
        else:
            cpl += 1

    if sanity_check:
        figs2 = plt.figure(figsize=[14, 10])

        for ch in range(n_comp):

            axs = figs2.add_subplot(rpl, cpl, ch + 1)

            f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                               nfft=None,
                               detrend='constant', return_onesided=True, scaling='density', axis=-1,
                               average='mean')

            if tight_range:
                Pxx_den = Pxx_den[f <= max_range]
                f = f[f <= max_range]

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
            axs.plot(predicted2, alpha=.8, linestyle=":", c="y", label='poly2')
            axs.plot(predicted3, alpha=.8, linestyle=":", c="m", label='poly3')
            axs.set_title("S{} | {} | Detrend SSD comp{}".format(s(sub), condition, ch+1))

            # axs.plot(f, np.log(Pxx_den) - predicted, c="g", label='poly_1/linear')
            axs.plot(f, np.log(Pxx_den) - predicted2, c="y", label='detrend-p2')
            axs.plot(f, np.log(Pxx_den) - predicted3, c="m", label='detrend-p3')

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
        # plt.savefig(fname=p2ssd + "{}/selection_plots/{}_SSD_Detrend_polies.png".format(condition,
        #                                                                                 s(sub)))
        if save_plots:
            plt.close()

    # Smaller freq-window (0-max_range Hz)  + Leave alpha-out

    if sanity_check:
        figs3 = plt.figure(figsize=[14, 10])

        for ch in range(n_comp):

            axs = figs3.add_subplot(rpl, cpl, ch + 1)

            f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                               nfft=None,
                               detrend='constant', return_onesided=True, scaling='density', axis=-1,
                               average='mean')

            # Freq.-Range (0-max_range Hz)
            f_small = f[f <= max_range]
            Pxx_den_small = Pxx_den[f <= max_range]

            # Leave alpha out
            f_small_alphout = f_small[~((sub_apeak+4 > f_small) & (f_small > sub_apeak-4))]
            Pxx_den_small_alphout = Pxx_den_small[~((sub_apeak+4 > f_small) & (f_small > sub_apeak-4))]

            # Remove leading ascent for poly-fit
            lead_peak_idx = np.where(Pxx_den_small_alphout == np.max(
                Pxx_den_small_alphout[f_small_alphout < 5]))[0][0]
            f_small_alphout = f_small_alphout[lead_peak_idx:]
            Pxx_den_small_alphout = Pxx_den_small_alphout[lead_peak_idx:]

            # Fit polynomial(3)
            model3_small = np.polyfit(f_small, np.log(Pxx_den_small), 3)
            predicted3_small = np.polyval(model3_small, f_small)  # pred on full! freq-range

            # Fit polynomial(3) to alpha out data
            model3_small_alphout = np.polyfit(f_small_alphout, np.log(Pxx_den_small_alphout), deg=3)
            predicted3_small_alphout = np.polyval(model3_small_alphout, f_small)

            # Plot
            plt.plot(f_small, np.log(Pxx_den_small), linestyle="-.",
                     label='data (f<={})'.format(max_range))
            plt.plot(predicted3_small, alpha=.8, linestyle=":", c="m", label='poly3')
            plt.plot(predicted3_small_alphout, alpha=.8, c="g", linestyle=":", label='poly3_alpha-out')
            plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small, c="m",
                     label='detrend-p3')
            plt.plot(f_small, np.log(Pxx_den_small) - predicted3_small_alphout, c="g",
                     label='detrend-p3_a-out')

            axs.vlines(sub_apeak,
                       ymin=axs.get_ylim()[0],
                       ymax=np.log(Pxx_den_small)[np.argmin(np.abs(f_small - sub_apeak))],
                       linestyles="dashed", alpha=0.2)  # ymax=np.polyval(model3_small, sub_apeak)

            axs.set_title("S{} | {} | Detrend SSD comp{}".format(s(sub), condition, ch + 1))

            if ch == 0:
                plt.legend(loc='upper right')
            plt.tight_layout()

        plt.show()
        if save_plots:
            plt.close()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # Define component selection criterion:
    # If bump around alpha peak is above zero + small error term: select component

    figs4 = plt.figure(figsize=[14, 10])

    selected_comps = []

    for ch in range(n_comp):

        axs = figs4.add_subplot(rpl, cpl, ch + 1)

        f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann", nperseg=None, noverlap=None,
                           nfft=None,
                           detrend='constant', return_onesided=True, scaling='density', axis=-1,
                           average='mean')

        if tight_range:
            # Freq.-Range (0-max_range Hz)
            Pxx_den = Pxx_den[f <= max_range]
            f = f[f <= max_range]

        # Leave alpha out
        f_alphout = f[~((sub_apeak + 4 > f) & (f > sub_apeak - 4))]
        Pxx_den_alphout = Pxx_den[~((sub_apeak + 4 > f) & (f > sub_apeak - 4))]

        # Remove leading ascent for poly-fit
        lead_peak_idx = np.where(Pxx_den_alphout == np.max(
            Pxx_den_alphout[f_alphout < 5]))[0][0]
        f_alphout = f_alphout[lead_peak_idx:]
        Pxx_den_alphout = Pxx_den_alphout[lead_peak_idx:]

        # Fit polynomial(3) to alpha-out data
        model3_alphout = np.polyfit(f_alphout, np.log(Pxx_den_alphout), deg=3)
        predicted3_alphout = np.polyval(model3_alphout, f)  # predict on whole! freq-range

        log_Pxx_den_detrend = np.log(Pxx_den) - predicted3_alphout

        # Define area around given alpha-peak
        f_apeak = f[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]
        log_Pxx_den_apeak = log_Pxx_den_detrend[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]

        # Define adjacent area
        log_Pxx_den_apeak_flank_left = log_Pxx_den_detrend[(f <= sub_apeak - 4)][-2:]
        log_Pxx_den_apeak_flank_right = log_Pxx_den_detrend[(f >= sub_apeak + 4)][:2]

        # # # Select
        # # Criterion: alpha-peak above zero-line + small error term
        error_term = 0.15  # TODO Could be defined more systematically
        selected = False
        if np.any(log_Pxx_den_apeak > 0 + error_term):
            # # Additional criterion: peak in area > adjacent areas
            c = 0.01  # difference-constant
            if np.max(log_Pxx_den_apeak) - np.mean(log_Pxx_den_apeak_flank_left) > c and \
                    np.max(log_Pxx_den_apeak) - np.mean(log_Pxx_den_apeak_flank_right) > c:
                # could be c * np.max(FLANK...)
                selected = True  # write ch (component) as selected
                selected_comps.append(ch+1)  # range(1, ...)

        axs.plot(f, np.log(Pxx_den), c="m", alpha=.3)  # Original
        axs.plot(f, predicted3_alphout, c="m", linestyle=":", alpha=.3)  # polyfit
        axs.plot(f, log_Pxx_den_detrend, c="b", alpha=.8)  # Detrend
        axs.plot(f_apeak, log_Pxx_den_apeak, c="g" if selected else "r")

        axs.plot(f[(f <= sub_apeak - 4)][-2:], log_Pxx_den_apeak_flank_left, c="y")
        axs.plot(f[(f >= sub_apeak + 4)][:2], log_Pxx_den_apeak_flank_right, c="y")

        axs.set_title("{} | {} | SSD comp{}".format(s(sub), condition, ch+1))
        axs.vlines(sub_apeak,
                   ymin=min(log_Pxx_den_detrend),
                   ymax=log_Pxx_den_detrend[np.argmin(np.abs(f - sub_apeak))],
                   linestyles="dashed", alpha=.2)
        plt.hlines(y=0, xmin=0, xmax=max_range, alpha=.4, linestyles=":")  # Zero-line

        log_Pxx_den_apeak_stand = log_Pxx_den_apeak - np.linspace(log_Pxx_den_apeak[0],
                                                                  log_Pxx_den_apeak[-1],
                                                                  num=len(log_Pxx_den_apeak))

        axs.plot(f_apeak, log_Pxx_den_apeak_stand, c="g" if selected else "r", alpha=.2)

        # np.var()
        # axs.fill_between(f_apeak, np.array(log_Pxx_den_apeak_stand+.5), alpha=.2)

    plt.tight_layout()
    plt.show()
    if save_plots:
        plt.savefig(fname=plt_folder + "{}_SSD_selection.png".format(s(sub)))
        plt.close()

    # Write selected SSD components in table
    tab_select_ssd[np.where(tab_select_ssd[:, 0] == str(sub)), 1] = ",".join(
        str(x) for x in selected_comps)
    tab_select_ssd[tab_select_ssd[:, 0] == str(sub), 2] = len(selected_comps)
    tab_select_ssd[tab_select_ssd[:, 0] == str(sub), 3] = n_comp


# Save Table of selected components
np.savetxt(fname=tab_select_name, X=tab_select_ssd, header=";".join(col_names), delimiter=";", fmt='%s')
cprint("{}able saved. End.".format("Plots and t" if save_plots else "T"), "b")
