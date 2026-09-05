import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import inflect
from datetime import datetime

import line_fitter as lf
import main_config as conf
import colours as col


if conf.TARGET == 'paper':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 11


def plot_orig_sections(acc_sampling_periods: list, acc_tsvs: list, acc_labels: list,
                       dec_sampling_periods: list, dec_tsvs: list, dec_labels: list,
                       idx_range: list, top_speed: float, show_number_texts: bool, speed_unit: str = 'km/h'):
    if speed_unit == 'm/s':
        coeff_v = 1
    elif speed_unit == 'km/h':
        coeff_v = 3.6
    else:
        coeff_v = 1

    fig, axs = plt.subplots(
        np.max([len(acc_sampling_periods), len(dec_sampling_periods)]), 2, sharex='col')

    labels = [acc_labels, dec_labels]
    for i, labs in enumerate(labels):
        labels[i] = [l.replace(
            'Tien', 'A').replace(
            'Karesz', 'B'
        ) for l in labs]


    for i, tsvs in enumerate([acc_tsvs, dec_tsvs]):
        for j, tv in enumerate(tsvs):
            ax = axs[j][i]

            t, v = tv
            ax.plot(t, coeff_v * v, color='g', label=labels[i][j])
            ax.grid()

            # label_loc = 'lower right' if i == 0 else 'upper right'
            label_loc = 'center right'
            # ax.legend(loc=label_loc)

            if show_number_texts:
                text_ypos = .7 if i == 0 else .2
                ax.text(.05, text_ypos, get_id_number(i, j, idx_range),
                        fontsize=13, fontweight='bold', transform=ax.transAxes)

            ax.set_ylim([0, top_speed * coeff_v])

    axs[0][0].set_title(f'Acceleration sections ({idx_range[0]}-{idx_range[1]-1})')
    axs[0][1].set_title(f'Deceleration sections ({idx_range[0]}-{idx_range[1]-1})')
    axs[-1][0].set_xlabel('Time [s]')
    axs[-1][1].set_xlabel('Time [s]')
    fig.text(.02, .5, f'Speed [{speed_unit}]', va='center', rotation='vertical')
    fig.text(.512, .5, f'Speed [{speed_unit}]', va='center', rotation='vertical')

    for axind, ax in enumerate(axs.flatten()):
        pos_o = ax.get_position()
        pos_n = [pos_o.x0 + (axind % 2) * .07 - .065,
                 pos_o.y0 - (axind // 2) * .0075 + .073,
                 1.2 * pos_o.width, 1.07 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(32 / 2.54, 22 / 2.54)  # :(

    return fig


def plot_centre_freqs_by_acceleration_time(
        acc_result_omegas: np.ndarray, acc_secs: list, acc_sampling_periods: list,
        dec_result_omegas: np.ndarray, dec_secs: list, dec_sampling_periods: list,
        max_num_modes: int, show_number_texts: bool):

    fig, axs = plt.subplots(1, 2)

    for i, ax in enumerate(axs):
        if i == 0:
            result_omegas = np.array(acc_result_omegas + [np.nan] * (max_num_modes - len(acc_result_omegas)))
            sampling_periods = acc_sampling_periods
            secs = acc_secs
        else:
            result_omegas = np.array(dec_result_omegas + [np.nan] * (max_num_modes - len(dec_result_omegas)))
            sampling_periods = dec_sampling_periods
            secs = dec_secs

        if len(secs) == 0:
            continue

        t_diffs = [s[1] - s[0] for s in secs]
        omegas_to_plot = result_omegas.transpose()
        ymax = sampling_periods[0]/np.min(omegas_to_plot)

        yaxis_diff = .02 * ymax
        ax.set_ylim([-2 * yaxis_diff, ymax + 3 * yaxis_diff])

        for mode_ind in range(omegas_to_plot.shape[0]):
            iter_times = [sampling_periods[i]/o for i, o in enumerate(omegas_to_plot[mode_ind])]
            ax.plot(t_diffs, iter_times, label=f'Mode {mode_ind}', marker='o', linestyle=' ')

            if mode_ind == 0 and show_number_texts:
                for j, t_diff in enumerate(t_diffs):
                    ax.text(t_diff, iter_times[j] + yaxis_diff, get_id_number(i, j),
                            fontweight='bold', ha='center')

        ax.set_title(f'{"Ac" if i == 0 else "De"}celeration results')
        # ax.set_xlim([0, len(iterations))
        ax.set_xlabel(r'$\tau_\mathrm{acc}$, acceleration time [s]')
        ax.set_ylabel(r'$T_{k_i}$, inverse centre frequency of modes [s]')
        # ax.legend(bbox_to_anchor=(1, .5), loc="center left")
        ax.legend()
        ax.grid()

        pos_o = ax.get_position()
        pos_n = [pos_o.x0 + [-.045, .015][i], pos_o.y0 - .02, 1.15 * pos_o.width, 1.1 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(35 / 2.54, 16 / 2.54)  # :(
    return fig


def plot_highest_energy_mode_centre_frequencies(
        acc_result_omegas: np.ndarray, acc_mode_energies: list, acc_secs: list, acc_sampling_periods: list,
        dec_result_omegas: np.ndarray, dec_mode_energies: list, dec_secs: list, dec_sampling_periods: list,
        max_num_modes: int, show_number_texts: bool, which_orders: list, plot_lines: bool = True,
        plot_as: list = None, combine_acc_dec: bool = False, custom_x_y_max = None, target: str = 'view',
        ratio_to_remove: float = 0.0, noisy_sampling_period: float = 0.2, y_axis_scale: str = 'lin',
        make_data_better: bool = False
):

    num_axs = 1 if combine_acc_dec else 1
    fig, axs = plt.subplots(1, num_axs)

    if ratio_to_remove > 0 and (not make_data_better):
        acc_sampling_periods = [noisy_sampling_period for _ in range(len(acc_sampling_periods))]
        dec_sampling_periods = [noisy_sampling_period for _ in range(len(dec_sampling_periods))]

    if combine_acc_dec:
        axs = [axs, axs]

    marker_size = 5 if y_axis_scale == 'lin' else 1.5

    plots = list()
    axs_details_already_set = False
    for axi, ax in enumerate(axs):
        if axi == 0:
            result_omegas = np.array(acc_result_omegas + [np.nan] * (max_num_modes - len(acc_result_omegas)))
            mode_energies = acc_mode_energies
            sampling_periods = acc_sampling_periods
            secs = acc_secs
        else:
            result_omegas = np.array(dec_result_omegas + [np.nan] * (max_num_modes - len(dec_result_omegas)))
            mode_energies = dec_mode_energies
            sampling_periods = dec_sampling_periods
            secs = dec_secs

        if len(mode_energies) == 0:
            continue

        if not combine_acc_dec:
            plots = list()
        ymaxs = list()
        yaxis_diffs = list()
        p = inflect.engine()
        t_diffs = [s[1] - s[0] for s in secs]
        max_mode_indices = np.array([-1] * result_omegas.shape[0])

        which_orders.sort(reverse=True)
        for oi, which_order in enumerate(which_orders):
            for exp_ind in range(result_omegas.shape[0]):
                # get the mode index of the nth-highest energy mode
                max_mode_indices[exp_ind] = np.argsort(mode_energies[exp_ind])[-which_order]

            omegas_to_plot = np.take_along_axis(result_omegas, max_mode_indices[:, None], axis=1).squeeze()

            ymax = np.max(np.array(sampling_periods)/omegas_to_plot)
            ymaxs.append(ymax)
            yaxis_diff = .02 * ymax
            yaxis_diffs.append(yaxis_diff)

            iter_times = [sampling_periods[i] / o for i, o in enumerate(omegas_to_plot)]
            ordinal_str = str(p.ordinal(which_order))
            if combine_acc_dec and axi == 1:
                    ax.plot(t_diffs, iter_times, color=col.get_mode_colour(which_order), marker='o', ms=marker_size,
                            linestyle=' ', label=f'{ordinal_str[0].upper()}{ordinal_str[1:]}-highest-energy modes')
            else:
                plots.append(
                    ax.plot(t_diffs, iter_times, color=col.get_mode_colour(which_order), marker='o', ms=marker_size,
                            linestyle=' ', label=f'{ordinal_str[0].upper()}{ordinal_str[1:]}-highest-energy modes')[0])

            if show_number_texts:
                for j, t_diff in enumerate(t_diffs):
                    ax.text(t_diff, iter_times[j] + yaxis_diff, get_id_number(axi, j),
                            fontweight='bold', ha='center')

        if plot_lines:
            if combine_acc_dec:
                if not axs_details_already_set:
                    plots.reverse()
                    plots.append(plot_lines_axs(ax, which_orders, plot_as))

                    lines = ax.get_lines()
                    rc = lf.RatiosCalculator()
                    for line in lines:
                        label = line.get_label()
                        x_t_diffs = line.get_xdata()
                        y_iter_times = line.get_ydata()
                        rc.add_line(x_t_diffs, y_iter_times, label)
                    rc.calculate_print_ratios()
            else:
                plots.reverse()
                plots.append(plot_lines_axs(ax, which_orders, plot_as))

        if y_axis_scale == 'log':
            ax.set_yscale('log')
            ax.set_yticks([.55, .7, 1., 2., 3., 4., 5.])
            ax.set_yticklabels([str(yt) for yt in ax.get_yticks()])
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            y_min = 0.45
        else:
            y_min = 0.0

        if custom_x_y_max is not None:
            x_max = custom_x_y_max[0]
            y_max = custom_x_y_max[1]
        else:
            x_max = 1.06 * np.max(t_diffs)
            y_max = 1.06 * np.max(ymaxs)

        if len(conf.ORDERS) == 1:
            y_axis_title = r'$T_{k_' + str(conf.ORDERS[0]) + r'}$, mode period [s]'
        else:
            y_axis_title = r'$T_{k_i}$, mode period [s]'

        if combine_acc_dec:
            if not axs_details_already_set:
                ax.set_xlim([0, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_xlabel(r'$\tau_\mathrm{acc}$, acceleration time [s]')
                ax.legend(plots, [p.get_label() for p in plots], loc='upper left')
                ax.grid()

                pos_o = ax.get_position()
                if target == 'view':
                    ax.set_ylabel(y_axis_title)
                    pos_n = [pos_o.x0 - .03, pos_o.y0 - .02, 1.14 * pos_o.width, 1.15 * pos_o.height]
                    fig.set_size_inches(22 / 2.54, 15 / 2.54)  # :(
                elif target == 'paper':
                    ax.set_ylabel(y_axis_title)
                    pos_n = [pos_o.x0 - .03, pos_o.y0 + .06, 1.15 * pos_o.width, 1.05 * pos_o.height]
                    fig.set_size_inches(16 / 2.54, 7 / 2.54)  # :(
                else:
                    pos_n = pos_o
                    fig.set_size_inches(10 / 2.54, 10 / 2.54)  # :(
                ax.set_position(pos_n)

                axs_details_already_set = True

        else:
            ax.set_title(f'{"Ac" if axi == 0 else "De"}celeration results')
            ax.set_xlim([0, x_max])
            ax.set_ylim([0, y_max])
            ax.set_xlabel(r'$\tau_\mathrm{acc}$, acceleration time [s]')
            ax.set_ylabel(y_axis_title)
            # ax.legend(bbox_to_anchor=(1, .5), loc="center left")
            ax.legend(plots, [p.get_label() for p in plots], loc='upper right')
            ax.grid()

            pos_o = ax.get_position()
            pos_n = [pos_o.x0 + [-.045, .015][axi], pos_o.y0 - .02, 1.15 * pos_o.width, 1.1 * pos_o.height]
            ax.set_position(pos_n)

            fig.set_size_inches(35 / 2.54, 16 / 2.54)  # :(

    return fig


def show():
    plt.show()
    return


def get_id_number(i: int, j: int, idx_range: list = None):
    if idx_range is None:
        plot_index = j
    else:
        plot_index = j + idx_range[0]
    return f'{"A" if i == 0 else "D"}/{plot_index}'


def plot_lines_axs(ax: plt.Axes, which_orders: list, plot_as: list = None):
    if plot_as is None:
        a_s = np.array([])
        for which_order in which_orders:
            a_s_temp = lf.get_a_s_by_order(which_order)
            if a_s_temp is not None:
                a_s = np.append(a_s, a_s_temp)
        a_s = np.unique(a_s)
    else:
        a_s = plot_as

    if len(a_s) == 1:
        if len(conf.ORDERS) == 1:
            label_str = r'$T_{k_' + str(conf.ORDERS[0]) + r'} = ' + get_a_str(a_s[0]) + r'\cdot \tau_\mathrm{acc}$'
        else:
            label_str = r'$T_{k_i} = ' + get_a_str(a_s[0]) + r'\cdot \tau_\mathrm{acc}$'
    else:
        label_str = r'$T_{k_i} = c \cdot \tau_\mathrm{acc}, \: c \in \{ ' + \
                ', '.join([get_a_str(a) for a in a_s]) + ' \}$'
    plot = None
    for a in a_s:
        line_x, line_y = lf.get_line(0, ax.get_xlim()[1], a)
        plot = ax.plot(line_x, line_y, color='k', linestyle='--', alpha=.5, label=label_str)[0]
    return plot


def get_a_str(a: float):
    if int(a) == a:
        return str(int(a))
    elif a == .5:
        return '1/2'
    elif a == 2/3:
        return '2/3'
    elif a == 2/5:
        return '2/5'
    else:
        return str(round(a, 2))


def plot_zero_top_speed_sections(
        t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections,
        all_sections, measurement_name, speed_unit: str = 'km/h',
        specific_start_index: int = None, specific_end_index: int = None):

    if speed_unit == 'm/s':
        coeff_v = 1
    elif speed_unit == 'km/h':
        coeff_v = 3.6
    else:
        coeff_v = 1

    print(f'Showing >{measurement_name}<')

    fig, axs = plt.subplots(2, 1, sharex='all')

    axs[0].plot(t, coeff_v * v, color='g', label='Original speed data', linewidth=2)
    axs[0].axhline(coeff_v * speed_tolerance, color='k', linestyle='--',
                   label=r'Range of speed tolerance ($\epsilon_v$)', alpha=.7)
    axs[0].axhline(coeff_v * (top_speed - speed_tolerance), color='k', linestyle='--', alpha=.7)
    axs[0].axhline(coeff_v * top_speed, color='k', label=r'Top speed ($V_\text{max}$)', alpha=.7)
    axs[0].axhline(coeff_v * (top_speed + speed_tolerance), color='k', linestyle='--', alpha=.7)
    for i, s in enumerate (zero_sections):
        labels = ['Beginning of a Zero section', 'End of a Zero section'] if i == 0 else [None, None]
        axs[0].axvline(t[s[0]], color='orange', linestyle='--', label=labels[0], alpha=.7)
        axs[0].axvline(t[s[1]], color='orange', label=labels[1], alpha=.7)
    for i, s in enumerate (top_speed_sections):
        labels = ['Beginning of a TopSpeed section', 'End of a TopSpeed section'] if i == 0 else [None, None]
        axs[0].axvline(t[s[0]], color='r', linestyle='--', label=labels[0], alpha=.5)
        axs[0].axvline(t[s[1]], color='r', label=labels[1], alpha=.5)
    # for sec in all_sections:
    #     text_x = t[(sec[0] + sec[1]) // 2]
    #     text_y = get_text_y(sec[2], coeff_v * speed_tolerance)
    #     axs[0].text(text_x, text_y, get_pretty_text(sec[2]), ha='center')

    if specific_start_index is not None and specific_end_index is not None:
        axs[0].set_xlim(t[specific_start_index], t[specific_end_index])

    axs[0].set_ylabel(f'Speed [km/h]')
    axs[0].set_ylim(conf.PLOT_Y_LIMS)
    axs[0].legend(loc='upper center', facecolor='#e3e3e3', framealpha=.9)

    type_labels = list()
    for sec in all_sections:
        type_labels += [sec[2]] * (sec[1] - sec[0] + 1)
        if sec[2] in conf.PLOT_HIGHLIGHT:
            for ax in axs:
                ax.axvspan(t[sec[0]], t[sec[1]], color='lightblue', alpha=.3)
    get_type_id_vectorised = np.vectorize(get_type_id)
    type_ids = get_type_id_vectorised(type_labels)

    axs[1].plot(t, type_ids, marker='o', color='k', linestyle='')
    axs[1].set_yticks(range(len(conf.PLOT_TYPES)))
    axs[1].set_yticklabels([get_pretty_text(t) for t in conf.PLOT_TYPES])

    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Section type')

    for i, ax in enumerate(axs):
        ax.grid()

        pos_o = ax.get_position()
        if i == 0:
            pos_n = [pos_o.x0 - .019, pos_o.y0 - .19, 1.145 * pos_o.width, 1.85 * pos_o.height]
        else:
            pos_n = [pos_o.x0 - .019, pos_o.y0 - .04, 1.145 * pos_o.width, .7 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(24 / 2.54, 16 / 2.54)  # :(

    plt.show()
    return

def get_text_y(name, speed_tolerance):
    if name =='zero':
        return 2 * speed_tolerance
    if name == 'acc':
        return 4 * speed_tolerance
    if name == 'nothing':
        return 6 * speed_tolerance
    if name == 'dec':
        return 8 * speed_tolerance
    if name == 'top_speed':
        return 10 * speed_tolerance
    return 12 * speed_tolerance


def get_pretty_text(text: str):
    if text == 'zero':
        return 'Zero'
    if text == 'top_speed':
        return 'TopSpeed'
    if text == 'acc':
        return r'$\bf{Acc}$'
    if text == 'dec':
        return r'$\bf{Dec}$'
    if text == 'nothing':
        return 'None'
    return text

def get_type_id(text: str) -> float:
    if text in conf.PLOT_TYPES:
        return conf.PLOT_TYPES.index(text)
    else:
        return -1


def tmp_plot_1(t, v):
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, v, color='g', label='orig')
    return ax

def tmp_plot_2(t, v, ax):
    ax.plot(t, v, color='orange', label='mod')
    ax.grid()
    ax.legend()
    plt.savefig(f'plots/tmp_fig_{datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")}.pdf')
    return
