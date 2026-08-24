import numpy as np
import inflect

import main_config as conf
import sim_config as sconf
import data_processor as proc

SIM_TYPES = {
    0: 'LINEAR',
    1: 'LINEAR',
    2: 'LINEAR',
    3: 'COS',
    4: 'LOGISTIC'
}


def print_simulation_details(
        acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels, acc_secs,
        dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies, dec_labels, dec_secs,
        max_num_modes):
    print()

    print(f'Simulation type: {sconf.SIMULATION_TYPE} ({SIM_TYPES[sconf.SIMULATION_TYPE]}) \n'
          f'Ratio to remove: {np.round(sconf.RATIO_TO_REMOVE * 100)}% \n'
          f'Use interpolation and smoothing: {conf.MAKE_DATA_BETTER} \n')

    max_accdur_freqs, max_accdur_inv_freqs, max_accdur = \
        proc.get_highest_acctime_frequencies_times(
            np.array(acc_result_omegas + dec_result_omegas),
            np.array(acc_secs + dec_secs), sconf.SAMPLING_PERIOD)
    print(f'Sampling period: {sconf.SAMPLING_PERIOD} s')

    ####################################
    print()

    print(f'Highest acceleration duration: {np.round(max_accdur, 1)} s')
    print('Real-world mode cent. frequencies of that acc. sect.: ')
    for i in range(len(max_accdur_freqs)):
        print(f'\t{np.round(max_accdur_freqs[i], 4)} Hz <-> '
              f'{np.round(max_accdur_inv_freqs[i], 1)} s')

    ####################################
    print()

    act_per_exp_num_points_ratios = []
    for i, (t, v) in enumerate(acc_tsvs):
        exp_num_points = (t[-1] - t[0]) / sconf.SAMPLING_PERIOD + 1
        act_num_points = len(t)
        act_per_exp_num_points_ratios.append(act_num_points / exp_num_points)

    print(f'Average(no. of data points/expected no. based on {sconf.SAMPLING_PERIOD} s): '
          f'{np.round(100.0*float(np.average(act_per_exp_num_points_ratios)), 2)} %')
    print(f'Expected slope of the Tk(tau_acc) curve: '
          f'{np.round(2.0 * float(np.average(act_per_exp_num_points_ratios)), 3)}')

    ####################################
    print()

    avg_diffs = []
    for i, (t, v) in enumerate(acc_tsvs):
        t_diff = np.diff(t)
        avg_diffs.append(np.average(t_diff))

    quasi_sampling_period = np.average(avg_diffs)

    print(f'Average difference between timesteps: {quasi_sampling_period} s')

    ####################################
    print()

    num_negative_timesteps = 0
    for t, v in acc_tsvs:
        for i, ts in enumerate(t):
            if i > 0 and ts < t[i - 1]:
                num_negative_timesteps += 1
    print(f'Number of negative timesteps (next timestamp is smaller than the previous)\n'
          f'in all acc. sections: {num_negative_timesteps}' + ('' if num_negative_timesteps>0 else ' (great!)'))

    ####################################
    print()

    ord_eng = inflect.engine()
    result_omegas = np.array(acc_result_omegas + dec_result_omegas)
    mode_energies = np.array(acc_mode_energies + dec_mode_energies)
    max_mode_indices = np.array([-1] * result_omegas.shape[0])
    big_total_345 = 0
    all_total_345 = 0
    for oi, which_order in enumerate(conf.ORDERS):
        for exp_ind in range(result_omegas.shape[0]):
            max_mode_indices[exp_ind] = np.argsort(mode_energies[exp_ind])[-which_order]
        omegas = np.take_along_axis(result_omegas, max_mode_indices[:, None], axis=1).squeeze()
        iter_times = [quasi_sampling_period / o for i, o in enumerate(omegas)]
        num_big = sum([1 for t in iter_times if t >= conf.SMALL_TK_THRESHOLD])

        if which_order in [3, 4, 5]:
            big_total_345 += num_big
            all_total_345 += result_omegas.shape[0]

        ordinal_str = str(ord_eng.ordinal(which_order))
        print(f'{ordinal_str[0].upper()}{ordinal_str[1:]}-highest-energy modes:\n'
              f'{num_big}, so {np.round(num_big/result_omegas.shape[0]*100, 1)}% of inv.'
              f'cent. freqs. are larger than {conf.SMALL_TK_THRESHOLD} s, \n'
              f'and {result_omegas.shape[0] - num_big}, so '
              f'{np.round((1-num_big/result_omegas.shape[0])*100, 1)}% are smaller')
    print('(small Tki is equivalent to high-freq. noisy data)')

    print()
    if all_total_345 > 0:
        print(f'Over 3rd, 4th, and 5th highest-energy modes combined: '
              f'{big_total_345} out of {all_total_345} sections, so '
              f'{np.round(big_total_345/all_total_345*100, 1)}% of inv. cent. freqs. are '
              f'larger than {conf.SMALL_TK_THRESHOLD} s')

    print('\n'*2)


