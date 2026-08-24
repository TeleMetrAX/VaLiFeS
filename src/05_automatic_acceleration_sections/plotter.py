import matplotlib.pyplot as plt
import numpy as np

import main_config as conf


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
    axs[0].legend(loc='center left', bbox_to_anchor=(1.01, .5))
    
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
            pos_n = [pos_o.x0 - .02, pos_o.y0 - .06, .77 * pos_o.width, 1.4 * pos_o.height]
        else:
            pos_n = [pos_o.x0 - .02, pos_o.y0 + .04, .77 * pos_o.width, .8 * pos_o.height]
        ax.set_position(pos_n)

    fig.set_size_inches(25 / 2.54, 9 / 2.54)  # :(

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
