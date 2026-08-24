import numpy as np
import matplotlib.pyplot as plt

import main_config as conf
import data_processor as proc


def plot_orig_sections(acc_sampling_periods: list, acc_tsvs: list, acc_labels: list,
                       dec_sampling_periods: list, dec_tsvs: list, dec_labels: list, speed_unit: str = 'km/h'):
    if speed_unit == 'm/s':
        coeff_v = 1
    elif speed_unit == 'km/h':
        coeff_v = 3.6
    else:
        coeff_v = 1

    fig, axs = plt.subplots(
        np.max([len(acc_sampling_periods), len(dec_sampling_periods)]), 2, sharex='col')

    labels = [acc_labels, dec_labels]
    for i, tsvs in enumerate([acc_tsvs, dec_tsvs]):
        for j, tv in enumerate(tsvs):
            ax = axs[j][i]

            t, v = tv
            ax.plot(t, coeff_v * v, color='g', label=labels[i][j])
            ax.grid()

            label_loc = 'lower right' if i == 0 else 'upper right'
            ax.legend(loc=label_loc)

            text_ypos = .7 if i == 0 else .2
            ax.text(.05, text_ypos, get_id_number(i, j),
                    fontsize=13, fontweight='bold', transform=ax.transAxes)

    axs[0][0].set_title('Acceleration sections')
    axs[0][1].set_title('Deceleration sections')
    axs[-1][0].set_xlabel('Time [s]')
    axs[-1][1].set_xlabel('Time [s]')
    fig.text(.02, .5, f'Speed [{speed_unit}]', va='center', rotation='vertical')
    fig.text(.512, .5, f'Speed [{speed_unit}]', va='center', rotation='vertical')

    for axind, ax in enumerate(axs.flatten()):
        pos_o = ax.get_position()
        pos_n = [pos_o.x0 + (axind % 2) * .07 - .065,
                 pos_o.y0 - (axind // 2) * .015 + .043,
                 1.2 * pos_o.width, 1.2 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(32 / 2.54, 22 / 2.54)  # :(

    return


def plot_centre_freqs_by_acceleration_time(
        acc_result_omegas: np.ndarray, acc_secs: list, acc_sampling_periods: list,
        dec_result_omegas: np.ndarray, dec_secs: list, dec_sampling_periods: list):

    fig, axs = plt.subplots(1, 2)

    for i, ax in enumerate(axs):
        if i == 0:
            result_omegas = acc_result_omegas
            sampling_periods = acc_sampling_periods
            secs = acc_secs
        else:
            result_omegas = dec_result_omegas
            sampling_periods = dec_sampling_periods
            secs = dec_secs

        t_diffs = [s[1] - s[0] for s in secs]
        omegas_to_plot = result_omegas.transpose()
        ymax = sampling_periods[0]/np.min(omegas_to_plot)
        yaxis_diff = .02 * ymax
        ax.set_ylim([-2 * yaxis_diff, ymax + 3 * yaxis_diff])

        for mode_ind in range(omegas_to_plot.shape[0]):
            iter_times = [sampling_periods[i]/o for i, o in enumerate(omegas_to_plot[mode_ind])]
            ax.plot(t_diffs, iter_times, label=f'Mode {mode_ind}', marker='o', linestyle=' ')

            if mode_ind == 0:
                for j, t_diff in enumerate(t_diffs):
                    ax.text(t_diff, iter_times[j] + yaxis_diff, get_id_number(i, j),
                            fontweight='bold', ha='center')

        ax.set_title(f'{"Ac" if i == 0 else "De"}celeration results')
        # ax.set_xlim([0, len(iterations))
        ax.set_xlabel('Acceleration time [s]')
        ax.set_ylabel(f'Inverse centre frequency of all modes [s]')
        # ax.legend(bbox_to_anchor=(1, .5), loc="center left")
        ax.legend()
        ax.grid()

        pos_o = ax.get_position()
        pos_n = [pos_o.x0 + [-.045, .015][i], pos_o.y0 - .02, 1.15 * pos_o.width, 1.1 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(35 / 2.54, 16 / 2.54)  # :(
    plt.show()
    return


def get_id_number(i: int, j: int):
    return f'{"A" if i == 0 else "D"}/{j + 1}'
