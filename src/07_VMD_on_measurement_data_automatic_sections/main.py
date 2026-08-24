import numpy as np
import glob
import pickle

import main_config as conf
import importer as imp
import data_processor as proc
import vmd
import plotter as plo
import status_bar as sb


if __name__ == '__main__':
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

        sections_paths = [s.replace('\\', '/') for s in glob.glob(f'{conf.SECTIONS_FOLDER}/*.csv')]

        for path_ind, sections_path in enumerate(sections_paths):
            # read sections data
            sections_df = imp.read_sections_data(sections_path, conf.SECTIONS_DF_COLUMNS)
            data_filename = sections_path.split('/')[-1]

            u = u_hat = omegas = np.array([])

            for sec_ind, section in sections_df.iterrows():
                if conf.DEBUG:
                    print(f'Begun section named \'{data_filename[:-4]}\' {path_ind+1}/{len(sections_paths)}...')

                section_type = section[conf.SECTIONS_DF_COLUMNS[2]]
                if section_type not in [conf.ACC_TYPE, conf.DEC_TYPE]:
                    raise ValueError(f'Current section type \'{section_type}\' '
                                     f'is not in the list of expected types [{conf.ACC_TYPE}, {conf.DEC_TYPE}]')

                # get data
                input_path = f'{conf.MEASUREMENTS_FOLDER}/{data_filename}'
                t, v, units, measurement_name = \
                    imp.read_import_measurement_data(input_path, False)

                if conf.DEBUG:
                    print('\tMeasurement data read.')

                indices = [section[conf.SECTIONS_DF_COLUMNS[0]],
                           section[conf.SECTIONS_DF_COLUMNS[1]]]
                secs = [t[indices[0]], t[indices[1]]]
                t, v = imp.get_section_by_indices(t, v, indices)

                if conf.DEBUG:
                    print('\tSection data extracted.')

                if conf.MAKE_DATA_BETTER:
                    sampling_period, t, v = \
                        proc.infer_sampling_period(t, v, conf.SAMPLING_PERIOD_TOLERANCE, conf.PRECISION)
                    t, v = proc.smooth_timeseries(t, v, rate=1 / sampling_period)
                    if conf.DEBUG:
                        print('\tSampling period inferred, timeseries interpolated and smoothed.')
                else:
                    sampling_period, _, _ = \
                        proc.infer_sampling_period(t, v, conf.SAMPLING_PERIOD_TOLERANCE, conf.PRECISION)
                    if conf.DEBUG:
                        print('\tSampling period inferred.')

                # print(f'Sampling period: {round(sampling_period, 3)} s')

                # save data
                label = (f'{measurement_name}:\n{round(secs[0], 1)} s - {round(secs[1], 1)} s '
                         f'({imp.get_driver(data_filename)})')

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

                sb.draw_status_bar(sec_ind + 1, len(sections_df),
                                   text=f'data file {path_ind+1}/{len(sections_paths)} named {data_filename[:-4]}',
                                   on=conf.STATUS_BAR_ON)

                if conf.DEBUG:
                    print(f'\tSection named \'{data_filename[:-4]}\' {path_ind+1}/{len(sections_paths)} done.')

            if conf.STATUS_BAR_ON:
                print()

        # save results
        if conf.STATUS_BAR_ON:
            print('Experiment done, saving results...')
        interp_str = 'interp' if conf.MAKE_DATA_BETTER else 'orig'
        with open(conf.OLD_RESULTS_PATH.format(interp_str), 'wb') as write_file:
            pickle.dump([acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels,
                         acc_secs, dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies,
                         dec_labels, dec_secs, max_num_modes], write_file)

    else:
        # load old results from file
        print('Loading previous results from file...')
        interp_str = 'interp' if conf.MAKE_DATA_BETTER else 'orig'
        with open(conf.OLD_RESULTS_PATH.format(interp_str), 'rb') as read_file:
            (acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_mode_energies, acc_labels, acc_secs,
             dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_mode_energies, dec_labels, dec_secs,
             max_num_modes) = pickle.load(read_file)

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
            conf.PLOT_LINES, conf.PLOT_AS, conf.COMBINE_ACC_DEC, conf.CUSTOM_X_Y_MAX,
            conf.TARGET, 0, 0, conf.YAXIS_SCALE))
        order_str = '_'.join([str(o) for o in conf.ORDERS])
        fig_names.append(f'centre_freqs_{order_str}_highest_energy_modes')

    if conf.MODE == 'save':
        for i, fig in enumerate(figs):
            if fig is not None:
                fig.savefig(f'{conf.SAVE_FOLDER}/{fig_names[i]}.png')
    elif conf.MODE == 'show':
        plo.show()
