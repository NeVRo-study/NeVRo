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
import pandas as pd
from scipy.signal import welch
from scipy.optimize import curve_fit

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Set paths
path_data = set_path2data()  # path_data = "../../../Data/"
p2ssd = path_data + "EEG/07_SSD/"

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Set hyperparameters
sanity_check = False  # plot additional infos (see below)
max_range = 40  # freq max for plots, 130 Hz: ~max
ffit_max = 20  # freq. max for fit; 40 Hz: ignores the line-noise related bump in data | 20 Hz: Low-Pass
assert max_range >= ffit_max, "max_range must be >= ffit_max"
save_plots = False
f_res_fac = 5  # sets nperseg= f_res_fac*250 in scipy.welch(), Default=256
poly_fit = False  # False: Uses 1/f-fit
n_subs = 45  # number of all subjects
subjects = np.arange(1, n_subs+1)  # ALL
# subjects = np.arange(1, 20+1)  # subjects = np.arange(21, 45+1)  # subsets
# subjects = np.array([5, 6, 15, 18, 21, 22, 26, 27, 31, 35])  # subset: check selections
# subjects = np.array([7, 14, 15, 21, 25])  # subset: check alpha peak info
# subjects = np.array([7, 13, 21, 22])  # subset: zero-line treshold
# subjects = np.array([3])  # subset: single subject: 5,6,8,11
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
    tab_select_ssd = np.zeros(shape=(n_subs, 4), dtype='<U{}'.format(
        len(",".join(str(x) for x in np.arange(1, 25+1)))))  # Init table
    tab_select_ssd.fill(np.nan)  # convert entries to nan
    tab_select_ssd[:, 0] = np.arange(1, n_subs+1)

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
    tab_alpha_peaks = pd.read_csv(p2ssd + "alphapeaks.csv", index_col=0)
    sub_apeak = tab_alpha_peaks.loc["NVR_{}".format(s(sub)), condition]

    if pd.isna(sub_apeak):
        cprint("No alpha peak information for {} in {} condition!".format(s(sub), condition), "r")
        continue  # No alpha peak information go to next subject

    # # Plot power spectral density (Welch)
    min_apeak = 99  # init
    psd_alternative = False  # True: plt.psd() [is equivalent]
    maxPxx_den = 0

    fig = plt.figure()
    ax = plt.subplot(1, 2 if psd_alternative else 1, 1)

    # for ch in range(1, n_comp):
    for ch in range(n_comp):
        f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann",
                           nperseg=f_res_fac*250, noverlap=None,
                           nfft=None,
                           detrend='constant', return_onesided=True, scaling='density', axis=-1,
                           average='mean')

        # Adapt range
        Pxx_den = Pxx_den[f <= max_range]
        f = f[f <= max_range]
        maxPxx_den = max(Pxx_den) if maxPxx_den < max(Pxx_den) else maxPxx_den

        ax.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        Pxx_den_apeak = Pxx_den[np.argmin(np.abs(f - sub_apeak))]
        min_apeak = Pxx_den_apeak if Pxx_den_apeak < min_apeak else min_apeak

    ax.vlines(sub_apeak, ymin=-0.01, ymax=min_apeak, linestyles="dashed", alpha=.2)
    ax.annotate('alpha peak: {:.2f}'.format(sub_apeak), xy=(sub_apeak, maxPxx_den),
                ha='left', va='top',
                bbox=dict(boxstyle='round', fc='w'))
    xt = ax.get_xticks()
    xt = np.append(xt, sub_apeak)
    xtl = xt.tolist()
    xtl[-1] = str(np.round(sub_apeak, 1))
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl)
    ax.set_xlim([-1, max_range+1])
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
        ax2.set_xlim([-1, max_range+1])
        ax2.set_title("S{} | {} | plt.psd()".format(s(sub), condition))

    fig.tight_layout()
    fig.show()
    if save_plots:
        fig.savefig(fname=plt_folder + "{}_SSD_powerspec.png".format(s(sub)))
        plt.close(fig)

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # # Detrend
    # # Subplot per component

    # # Define figure grid-size: with rpl x cpl cells
    rpl = 1
    cpl = 1
    while (rpl*cpl) < n_comp:
        if rpl == cpl:
            rpl += 1
        else:
            cpl += 1

    # # Test different polynomial fits (order: 2, 3)
    if sanity_check:
        figs2 = plt.figure(figsize=[14, 10])
        # figs22 = plt.figure(figsize=[14, 10])  # TEST

        for ch in range(n_comp):

            axs = figs2.add_subplot(rpl, cpl, ch + 1)

            f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann",
                               nperseg=f_res_fac*250, noverlap=None,
                               nfft=None,
                               detrend='constant', return_onesided=True, scaling='density', axis=-1,
                               average='mean')

            # Adapt range (0-max_range Hz)
            Pxx_den = Pxx_den[f <= max_range]
            f = f[f <= max_range]

            Pxx_den_apeak = np.log(Pxx_den)[np.argmin(np.abs(f - sub_apeak))]

            # Linear fit / poly(1)
            # model = np.polyfit(f, np.log(Pxx_den), 1)
            # predicted = np.polyval(model, f)

            # Quadratic fit / poly(2)
            model2 = np.polyfit(f, np.log(Pxx_den), 2)  # fit on whole range
            predicted2 = np.polyval(model2, f)

            # Cubic fit / poly(3)
            model3 = np.polyfit(f, np.log(Pxx_den), 3)
            predicted3 = np.polyval(model3, f)

            # Fit 1/(a*f**b): Find optimal a, b params
            def f1_ab(fr, a, b):
                return np.log(1/(a*fr**b))  # in log scale

            modelb_opt_param, modelb_cov_param = curve_fit(f=f1_ab,
                                                           xdata=f[1:],  # f > 0 values, due to 1/f
                                                           ydata=np.log(Pxx_den)[1:])

            predicted4 = f1_ab(fr=f[1:], a=modelb_opt_param[0], b=modelb_opt_param[1])

            # Plot
            axs.plot(f, np.log(Pxx_den), linestyle="-.", label='data')

            # axs.plot(predicted, alpha=.8, linestyle=":", c="g", label='poly_1/linear')
            axs.plot(f, predicted2, alpha=.8, linestyle=":", c="y", label='poly2')
            axs.plot(f, predicted3, alpha=.8, linestyle=":", c="m", label='poly3')
            axs.plot(f[1:], predicted4, alpha=.8, linestyle=":", c="g", label='1/af**b')
            axs.set_title("S{} | {} | Detrend SSD comp{}".format(s(sub), condition, ch+1))

            # axs.plot(f, np.log(Pxx_den) - predicted, c="g", label='poly_1/linear')
            axs.plot(f, np.log(Pxx_den) - predicted2, c="y", label='detrend-p2')
            axs.plot(f, np.log(Pxx_den) - predicted3, c="m", label='detrend-p3')
            axs.plot(f[1:], np.log(Pxx_den)[1:] - predicted4, c="g", label='detrend-1/f')

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

    # # Test Smaller freq-window (0-max_range Hz) + Leave alpha-out
    if sanity_check:
        figs3 = plt.figure(figsize=[14, 10])

        for ch in range(n_comp):

            axs = figs3.add_subplot(rpl, cpl, ch + 1)

            f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann",
                               nperseg=f_res_fac*250, noverlap=None,
                               nfft=None,
                               detrend='constant', return_onesided=True, scaling='density', axis=-1,
                               average='mean')

            # Adapt Freq.-range (0-max_range Hz)
            Pxx_den = Pxx_den[f <= max_range]
            f = f[f <= max_range]

            # Adapt data to fit (0-fit_max Hz)
            f_fit = f[f <= ffit_max]
            Pxx_den_fit = Pxx_den[f <= ffit_max]

            # Leave alpha out
            f_alphout_fit = f_fit[~((sub_apeak+4 > f_fit) & (f_fit > sub_apeak-4))]
            Pxx_den_alphout_fit = Pxx_den_fit[~((sub_apeak+4 > f_fit) & (f_fit > sub_apeak-4))]

            # Remove leading ascent for fit-functions
            lead_peak_idx = np.where(Pxx_den_alphout_fit == np.max(
                Pxx_den_alphout_fit[f_alphout_fit < 5]))[0][0]
            _f_fit = f_fit[lead_peak_idx:]
            _f_alphout_fit = f_alphout_fit[lead_peak_idx:]
            _Pxx_den_fit = Pxx_den_fit[lead_peak_idx:]
            _Pxx_den_alphout_fit = Pxx_den_alphout_fit[lead_peak_idx:]

            # Fit polynomial(3)
            model3_small = np.polyfit(f_fit, np.log(Pxx_den_fit), 3)
            predicted3_small = np.polyval(model3_small, f)  # pred on full! freq-range

            # Fit polynomial(3) to alpha out data
            model3_small_alphout = np.polyfit(f_alphout_fit, np.log(Pxx_den_alphout_fit), deg=3)
            predicted3_small_alphout = np.polyval(model3_small_alphout, f)

            # Fit 1/(a*f**b): Find optimal a, b params
            def f1_ab(fr, a, b):
                return np.log(1 / (a * fr ** b))  # in log scale


            modelf_opt_param, modelf_cov_param = curve_fit(f=f1_ab,
                                                           xdata=_f_fit,
                                                           ydata=np.log(_Pxx_den_fit))

            modelfao_opt_param, modelfao_cov_param = curve_fit(f=f1_ab,
                                                               xdata=_f_alphout_fit,
                                                               ydata=np.log(_Pxx_den_alphout_fit))

            predicted4 = f1_ab(fr=f, a=modelf_opt_param[0], b=modelf_opt_param[1])
            predicted4ao = f1_ab(fr=f, a=modelfao_opt_param[0], b=modelfao_opt_param[1])

            # Plot
            plt.plot(f, np.log(Pxx_den), linestyle="-.", label='data (f<={})'.format(max_range))
            # plt.plot(predicted3_small, alpha=.8, linestyle=":", c="m", label='poly3')
            # plt.plot(predicted3_small_alphout, alpha=.8, c="g", linestyle=":", label='poly3_alpha-out')
            plt.plot(f, predicted4, alpha=.8, color='orange', linestyle=":", label='1/fit')
            plt.plot(f, predicted4ao, alpha=.8, c="y", linestyle=":", label='1/fit_alpha-out')

            # plt.plot(f_fit, np.log(Pxx_den_fit) - predicted3_small, c="m",
            #          label='detrend-p3')
            # plt.plot(f_fit, np.log(Pxx_den_fit) - predicted3_small_alphout, c="g",
            #          label='detrend-p3_a-out')
            plt.plot(f, np.log(Pxx_den) - predicted4, color='orange', label='detrend-1/fit')
            plt.plot(f, np.log(Pxx_den) - predicted4ao, c="y", label='detrend-1/fit_a-out')

            axs.vlines(sub_apeak,
                       ymin=axs.get_ylim()[0],
                       ymax=np.log(Pxx_den_fit)[np.argmin(np.abs(f_fit - sub_apeak))],
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
    figs5 = plt.figure(figsize=[14, 10])

    selected_comps = []

    for ch in range(n_comp):

        axs = figs4.add_subplot(rpl, cpl, ch + 1)

        f, Pxx_den = welch(x=sub_df[:, ch], fs=250.0, window="hann",
                           nperseg=f_res_fac*250, noverlap=None,
                           nfft=None,
                           detrend='constant',  # detrend=False: no major difference to 'constant'
                           return_onesided=True, scaling='density', axis=-1,
                           average='mean')

        # Adapt Freq.-Range (0-max_range Hz)
        Pxx_den = Pxx_den[f <= max_range]
        f = f[f <= max_range]
        # f[0] += 0.0001  #  stability term

        # Adapt data to fit (0-fit_max Hz)
        f_fit = f[f <= ffit_max]
        Pxx_den_fit = Pxx_den[f <= ffit_max]

        # Leave alpha out
        f_alphout_fit = f_fit[~((sub_apeak + 4 > f_fit) & (f_fit > sub_apeak - 4))]
        Pxx_den_alphout_fit = Pxx_den_fit[~((sub_apeak + 4 > f_fit) & (f_fit > sub_apeak - 4))]

        # Remove leading ascent for poly-fit
        lead_peak_idx = np.where(Pxx_den_alphout_fit == np.max(Pxx_den_alphout_fit[
                                                                   f_alphout_fit < 5]))[0][0]

        _f_alphout_fit = f_alphout_fit[lead_peak_idx:]
        _Pxx_den_alphout_fit = Pxx_den_alphout_fit[lead_peak_idx:]

        if poly_fit:
            # Fit polynomial(3) to alpha-out data
            model3_alphout = np.polyfit(f_alphout_fit, np.log(Pxx_den_alphout_fit), deg=3)
            predicted = np.polyval(model3_alphout, f)  # predict on whole! freq-range

        else:
            # Fit 1/(a*f**b): Find optimal a, b params
            def f1_ab(fr, a, b):
                return np.log(1 / (a * fr ** b))  # in log scale

            modelfao_opt_param, modelfao_cov_param = curve_fit(f=f1_ab,
                                                               xdata=_f_alphout_fit,
                                                               ydata=np.log(_Pxx_den_alphout_fit))

            predicted = f1_ab(fr=f, a=modelfao_opt_param[0], b=modelfao_opt_param[1])

        log_Pxx_den_detrend = np.log(Pxx_den) - predicted

        # Define area around given alpha-peak
        f_apeak = f[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]
        log_Pxx_den_apeak = log_Pxx_den_detrend[((f > sub_apeak - 4) & (sub_apeak + 4 > f))]

        # Define adjacent area
        log_Pxx_den_apeak_flank_left = log_Pxx_den_detrend[(f <= sub_apeak - 4)][-2*f_res_fac:]
        log_Pxx_den_apeak_flank_right = log_Pxx_den_detrend[(f >= sub_apeak + 4)][:2*f_res_fac]

        # # # Select
        # # Criterion: alpha-peak above zero-line + small error term
        error_term = 0.35  # TODO Could be defined more systematically

        selected = False
        if np.any(log_Pxx_den_apeak > 0 + error_term):
            # # Additional criterion: peak in area > adjacent areas
            cSD = 1.45  # TODO check = 1.45 whether to conservative

            # Z-Score decision area:
            left_pxx_right = np.append(log_Pxx_den_apeak_flank_left,
                                       np.append(log_Pxx_den_apeak, log_Pxx_den_apeak_flank_right))
            z_left_pxx_right = z_score(left_pxx_right)

            lb = len(log_Pxx_den_apeak_flank_left)
            rb = lb + len(log_Pxx_den_apeak)

            # Crop into single flanks and alpha area, respectively
            z_log_Pxx_den_apeak_flank_left = z_left_pxx_right[:lb]
            z_log_Pxx_den_apeak = z_left_pxx_right[lb:rb]
            z_log_Pxx_den_apeak_flank_right = z_left_pxx_right[rb:]

            # could be c * np.max(FLANK...)
            # if np.max(log_Pxx_den_apeak) - np.mean(log_Pxx_den_apeak_flank_left) > cSD and \
            #         np.max(log_Pxx_den_apeak) - np.mean(log_Pxx_den_apeak_flank_right) > cSD:

            # Select comp if alpha area c*standard deviation (SD) higher than of flanks
            ldist = np.max(z_log_Pxx_den_apeak) - np.mean(z_log_Pxx_den_apeak_flank_left)
            rdist = np.max(z_log_Pxx_den_apeak) - np.mean(z_log_Pxx_den_apeak_flank_right)
            pdist = min(ldist, rdist)
            if pdist >= cSD:
                selected = True  # write ch (component) as selected
                selected_comps.append(ch + 1)  # range(1, ...)

            # Plot z-Transform Comp >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            axs2 = figs5.add_subplot(rpl, cpl, ch + 1)

            axs2.plot(f[(f <= sub_apeak - 4)][-2 * f_res_fac:], z_log_Pxx_den_apeak_flank_left, c="y")
            axs2.plot(f[((f > sub_apeak - 4) & (sub_apeak + 4 > f))], z_log_Pxx_den_apeak,
                      c="g" if selected else "r")
            axs2.plot(f[(f >= sub_apeak + 4)][:2 * f_res_fac], z_log_Pxx_den_apeak_flank_right, c="y")
            axs2.hlines(y=0, xmin=0, xmax=max_range, alpha=.4, linestyles=":")  # Zero-line

            axs2.hlines(y=np.mean(z_log_Pxx_den_apeak_flank_left),
                        xmin=0, xmax=f[(f <= sub_apeak - 4)][-2 * f_res_fac:][-1]+1,
                        alpha=.4, linestyles=":", color="y")  # Mean-line, left
            axs2.hlines(y=np.max(z_log_Pxx_den_apeak),
                        xmin=f[(f <= sub_apeak - 4)][-2 * f_res_fac:][-1]-1,
                        xmax=f[(f >= sub_apeak + 4)][:2 * f_res_fac][0]+1,
                        alpha=.4, linestyles=":",
                        color="g" if selected else "r")  # max/peak-line, alpha area
            axs2.hlines(y=np.mean(z_log_Pxx_den_apeak_flank_right),
                        xmin=f[(f >= sub_apeak + 4)][:2 * f_res_fac][0]-1,
                        xmax=max_range,
                        alpha=.4, linestyles=":", color="y")  # Mean-line, right
            axs2.vlines(sub_apeak,
                        ymin=min(z_left_pxx_right), ymax=max(z_left_pxx_right),
                        linestyles="dashed", alpha=.2)

            axs2.vlines(f[(f >= sub_apeak + 4)][:2 * f_res_fac][0]+1 if rdist < ldist else
                        f[(f <= sub_apeak - 4)][-2 * f_res_fac:][-1]-1,
                        ymin=max(z_log_Pxx_den_apeak)-pdist, ymax=max(z_log_Pxx_den_apeak),
                        linestyles=":", alpha=.5, color="g" if selected else "r",
                        label="diff: {}".format(np.round(pdist, 2)))

            axs2.set_xlim([int(sub_apeak-7) if int(sub_apeak-7) > 0 else 0, int(sub_apeak+7)])
            axs2.set_ylim([-3.0, 3.0])
            axs2.set_title("{} | {} | comp{} | z-alpha".format(s(sub), condition, ch + 1),
                           color="g" if selected else "r")
            axs2.legend(handlelength=0, handletextpad=0, loc='lower right')
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        axs.plot(f, np.log(Pxx_den), c="m", alpha=.3)  # Original

        axs.plot(f, predicted, c="m", linestyle=":", alpha=.3)  # poly or 1/f fit
        axs.plot(f, log_Pxx_den_detrend, c="b", alpha=.8)  # Detrend
        axs.plot(f_apeak, log_Pxx_den_apeak, c="g" if selected else "r")

        axs.plot(f[(f <= sub_apeak - 4)][-2*f_res_fac:], log_Pxx_den_apeak_flank_left, c="y")
        axs.plot(f[(f >= sub_apeak + 4)][:2*f_res_fac], log_Pxx_den_apeak_flank_right, c="y")

        axs.set_title("{} | {} | SSD comp{}".format(s(sub), condition, ch+1),
                      color="g" if selected else "r")
        axs.vlines(sub_apeak,
                   ymin=min(log_Pxx_den_detrend[1:]),  # first value: -inf
                   ymax=log_Pxx_den_detrend[np.argmin(np.abs(f - sub_apeak))],
                   linestyles="dashed", alpha=.2)
        axs.hlines(y=0, xmin=0, xmax=max_range, alpha=.4, linestyles=":")  # Zero-line

        axs.hlines(y=max(log_Pxx_den_apeak), xmin=0, xmax=f[(f >= sub_apeak + 4)][:2*f_res_fac][0]+1,
                   linestyles="-.", alpha=.3,
                   color="black", label="max: {}".format(np.round(max(log_Pxx_den_apeak), 3)))

        log_Pxx_den_apeak_stand = log_Pxx_den_apeak - np.linspace(log_Pxx_den_apeak[0],
                                                                  log_Pxx_den_apeak[-1],
                                                                  num=len(log_Pxx_den_apeak))

        axs.plot(f_apeak, log_Pxx_den_apeak_stand, c="g" if selected else "r", alpha=.2)

        axs.set_xlim(2, max_range)
        axs.set_ylim(-2, 2)

        axs.legend(handlelength=0, handletextpad=0, loc='lower right')

        # np.var()
        # axs.fill_between(f_apeak, np.array(log_Pxx_den_apeak_stand+.5), alpha=.2)

    figs4.tight_layout()
    figs4.show()
    figs5.tight_layout()
    figs5.show()

    if save_plots:
        figs4.savefig(fname=plt_folder + "{}_SSD_selection.png".format(s(sub)))
        figs5.savefig(fname=plt_folder + "{}_SSD_selection_flank-criterion.png".format(s(sub)))
        plt.close(fig=figs4)
        plt.close(fig=figs5)

    # Write selected SSD components in table
    tab_select_ssd[np.where(tab_select_ssd[:, 0] == str(sub)), 1] = ",".join(
        str(x) for x in selected_comps)
    tab_select_ssd[tab_select_ssd[:, 0] == str(sub), 2] = len(selected_comps)
    tab_select_ssd[tab_select_ssd[:, 0] == str(sub), 3] = n_comp


# Save Table of selected components
if save_plots:
    np.savetxt(fname=tab_select_name, X=tab_select_ssd, header=";".join(col_names), delimiter=";",
               fmt='%s')
    cprint("{}able saved. End.".format("Plots and t" if save_plots else "T"), "b")
