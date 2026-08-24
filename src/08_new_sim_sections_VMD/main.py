import numpy as np
import pandas as pd
import pickle

import main_config as conf
import sim_config as sconf
import data_generator as gen
import importer as imp
import data_processor as proc
import vmd
import plotter as plo
import status_bar as sb
import data_simulator_new_simple as sim
import saver as sav
import details_printer as det


if __name__ == '__main__':

    rest_str = '_restored' if conf.MAKE_DATA_BETTER else ''
    old_tmp_res_path = conf.OLD_RESULTS_PATH.format(sconf.SIMULATION_TYPE, sconf.RATIO_TO_REMOVE, rest_str)

    if not conf.USE_OLD:
        # initialise
        max_num_modes = 0
        acc_result_omegas = list()
        dec_result_omegas = list()
        acc_mode_energies = list()
        dec_mode_energies = list()
        acc_sampling_periods = list()
        dec_sampling_periods = list()
        acc_tsvs = list()
        dec_tsvs = list()
        acc_labels = list()
        dec_labels = list()
        acc_secs = list()
        dec_secs = list()

        for sim_ind in range(conf.NUM_SIMULATIONS):
            measurement_name = f'SIM{sim_ind + 1}/{conf.NUM_SIMULATIONS}'

            if conf.SIMULATION_MODE == 'old':
                data, data_noise = gen.get_random_velo_data(
                    length=conf.SIM_LENGTH, status=conf.SIM_STATUS,
                    acc=(0.2, 1.3), phase=10, sd=0.3, noise_per_sec=8, rate=14)

                data[conf.GEN_DATA_T_COL] = pd.to_datetime(data[conf.GEN_DATA_T_COL], format='ISO8601')
                data[conf.GEN_DATA_T_COL] = (data[conf.GEN_DATA_T_COL] - data[conf.GEN_DATA_T_COL][0]).dt.total_seconds()

                t_complete = data[conf.GEN_DATA_T_COL].to_numpy()
                v_complete = data[conf.GEN_DATA_V_COL].to_numpy()

                sections_df = proc.infer_acceleration_deceleration_sections(
                    t_complete, v_complete, measurement_name, conf.SPEED_TOLERANCE,
                    conf.TOP_SPEED, conf.SECTIONS_DF_COLUMNS)

            elif conf.SIMULATION_MODE == 'new':
                t_complete, v_complete, acc_type = sim.get_simulated_data()
                sections_df = pd.DataFrame(columns=conf.SECTIONS_DF_COLUMNS)
                sections_df.loc[0] = [0, len(t_complete)-1, acc_type]

            else:
                raise Exception(f'Unknown simulation mode: {conf.SIMULATION_MODE}')

            u = u_hat = omegas = np.array([])

            if conf.DEBUG:
                print('#'*50)
                print(f'Processing simulation {sim_ind+1}/{conf.NUM_SIMULATIONS}')
                print('#'*50)

            sections_df.reset_index(drop=True, inplace=True)

            for sec_ind, section in sections_df.iterrows():
                if conf.DEBUG:
                    print(f'Begun section {sec_ind+1}/{len(sections_df)}...')

                section_type = section[conf.SECTIONS_DF_COLUMNS[2]]
                if section_type not in [conf.ACC_TYPE, conf.DEC_TYPE]:
                    raise ValueError(f'Current section type \'{section_type}\' '
                                     f'is not in the list of expected types [{conf.ACC_TYPE}, {conf.DEC_TYPE}]')

                indices = [section[conf.SECTIONS_DF_COLUMNS[0]],
                           section[conf.SECTIONS_DF_COLUMNS[1]]]
                secs = [t_complete[indices[0]], t_complete[indices[1]]]
                t, v = imp.get_section_by_indices(t_complete, v_complete, indices)

                if conf.DEBUG:
                    print('\tSection data extracted.')

                if conf.SIMULATION_MODE == 'old':
                    sampling_period, t, v = \
                        proc.infer_sampling_period(t, v, conf.SAMPLING_PERIOD_TOLERANCE, conf.PRECISION)
                    if conf.DEBUG:
                        print('\tSampling period inferred.')

                    t, v = proc.smooth_timeseries(t, v, rate=1/sampling_period)
                    if conf.DEBUG:
                        print('\tTimeseries smoothed.')

                elif conf.SIMULATION_MODE == 'new':
                    sampling_period = sconf.SAMPLING_PERIOD

                    if conf.MAKE_DATA_BETTER:
                        try:
                            sampling_period, t, v = proc.infer_sampling_period(
                                t, v, conf.SAMPLING_PERIOD_TOLERANCE, conf.PRECISION, sconf.SAMPLING_PERIOD)
                            t, v = proc.smooth_timeseries(t, v, rate=1 / sampling_period)
                        except:
                            continue


                # save data
                label = (f'{measurement_name}:\n{round(secs[0], 1)} s - {round(secs[1], 1)} s')

                if section_type == conf.ACC_TYPE:
                    acc_sampling_periods.append(sampling_period)
                    acc_tsvs.append([t, v])
                    acc_labels.append(label)
                    acc_secs.append(secs)
                else:
                    dec_sampling_periods.append(sampling_period)
                    dec_tsvs.append([t, v])
                    dec_labels.append(label)
                    dec_secs.append(secs)

                if conf.DEBUG:
                    print('\tData saved.')

                # run VMD
                u, u_hat, omegas = vmd.decompose_orig(
                    v, conf.ALPHA, conf.K, conf.TAU, conf.DC, conf.INIT, conf.TOL)
                curr_omegas = omegas[-1]
                max_num_modes = max(conf.K, max_num_modes)
                
                if conf.DEBUG:
                    print('\tVMD done.')

                # calculate mode energies
                mode_energies = list()
                for mode in u:
                    mode_energies.append(proc.get_mode_energy(mode))

                if conf.DEBUG:
                    print('\tCalculated mode energies.')

                # save centre frequencies
                if section_type == conf.ACC_TYPE:
                    acc_result_omegas.append(curr_omegas)
                    acc_mode_energies.append(mode_energies)
                else:
                    dec_result_omegas.append(curr_omegas)
                    dec_mode_energies.append(mode_energies)

                if conf.SIMULATION_MODE == 'old':
                    sb.draw_status_bar(sec_ind + 1, len(sections_df), text=measurement_name, on=conf.STATUS_BAR_ON)

                if conf.DEBUG:
                    print(f'\tSection named \'{measurement_name}\' {sim_ind+1}/{conf.NUM_SIMULATIONS} done.')

            if conf.STATUS_BAR_ON:
                if conf.SIMULATION_MODE == 'old':
                    print()
                elif conf.SIMULATION_MODE == 'new':
                    sb.draw_status_bar(sim_ind + 1, conf.NUM_SIMULATIONS, text=measurement_name, on=conf.STATUS_BAR_ON)


        # save results
        if conf.STATUS_BAR_ON:
            print()
            print('Experiment done, saving results...')
        with open(old_tmp_res_path, 'wb') as write_file:
            pickle.dump([acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels,
                         acc_secs, dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies,
                         dec_labels, dec_secs, max_num_modes], write_file)

    else:
        # load old results from file
        print('Loading previous results from file...')
        with open(old_tmp_res_path, 'rb') as read_file:
            (acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels, acc_secs,
             dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies, dec_labels, dec_secs,
             max_num_modes) = pickle.load(read_file)

    if conf.PRINT_DETAILS and conf.SIMULATION_MODE == 'new':
        det.print_simulation_details(
            acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels, acc_secs,
            dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies, dec_labels, dec_secs,
            max_num_modes)

    figs = list()
    fig_names = list()
    if conf.TO_PLOT == 'all':
        # plot all section in chunks defined by conf.LIMIT_PLOT1
        for start_ind in range(0, max(len(acc_labels), len(dec_labels)), conf.LIMIT_PLOT1):
            end_ind = start_ind + conf.LIMIT_PLOT1 - 1 + 1
            a_beg = start_ind if len(acc_labels) > start_ind else len(acc_labels)
            a_end = end_ind if len(acc_labels) > end_ind else len(acc_labels)
            d_beg = start_ind if len(dec_labels) > start_ind else len(dec_labels)
            d_end = end_ind if len(dec_labels) > end_ind else len(dec_labels)

            idx_range = [start_ind, end_ind]
            figs.append(plo.plot_orig_sections(
                acc_sampling_periods[a_beg:a_end], acc_tsvs[a_beg:a_end], acc_labels[a_beg:a_end],
                dec_sampling_periods[d_beg:d_end], dec_tsvs[d_beg:d_end], dec_labels[d_beg:d_end],
                idx_range, conf.TOP_SPEED, conf.SHOW_PLOT_NUMBER_TEXTS))
            fig_names.append(f'sections_{start_ind}_{end_ind-1}')

        figs.append(plo.plot_centre_freqs_by_acceleration_time(
            acc_result_omegas, acc_secs, acc_sampling_periods,
            dec_result_omegas, dec_secs, dec_sampling_periods,
            max_num_modes, conf.SHOW_PLOT_NUMBER_TEXTS))
        fig_names.append('centre_freqs')

    elif conf.TO_PLOT == 'highests':
        figs.append(plo.plot_highest_energy_mode_centre_frequencies(
            acc_result_omegas, acc_mode_energies, acc_secs, acc_sampling_periods,
            dec_result_omegas, dec_mode_energies, dec_secs, dec_sampling_periods,
            max_num_modes, conf.SHOW_PLOT_NUMBER_TEXTS, conf.ORDERS,
            conf.PLOT_LINES, conf.PLOT_AS, conf.COMBINE_ACC_DEC, conf.CUSTOM_X_Y_MAX, conf.TARGET,
            sconf.RATIO_TO_REMOVE, sconf.NOISY_SAMPLING_PERIOD, conf.PLOT_YAXIS_SCALE, conf.MAKE_DATA_BETTER
        ))
        order_str = '_'.join([str(o) for o in conf.ORDERS])
        fig_names.append(f'centre_freqs_{order_str}_highest_energy_modes')

    if conf.MODE == 'save':
        for i, fig in enumerate(figs):
            if fig is not None:
                plot_filename = '_'.join(old_tmp_res_path[:-7].replace('..', '').split('/'))
                if conf.SAVE_FORMAT == 'png':
                    sav.save_fig_png(fig, f'{conf.SAVE_FOLDER}/{plot_filename}.png', dpi=300)
                elif conf.SAVE_FORMAT == 'pdf':
                    sav.save_fig_pdf(fig, f'{conf.SAVE_FOLDER}/{plot_filename}.pdf', max_points=5000, dpi=300)
    elif conf.MODE == 'show':
        plo.show()
