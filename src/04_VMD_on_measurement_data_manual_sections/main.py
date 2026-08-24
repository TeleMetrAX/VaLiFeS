import numpy as np
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
        u = u_hat = omegas = acc_result_omegas = dec_result_omegas = np.array([])
        max_num_modes = 0
        iter_omegas = list()
        acc_sampling_periods = list()
        dec_sampling_periods = list()
        acc_tsvs = list()
        dec_tsvs = list()
        acc_labels = list()
        dec_labels = list()
        acc_secs = list()
        dec_secs = list()

        # read sections data
        acc_sections_data, dec_sections_data = imp.read_sections_data(
            conf.SECTIONS_JSON_PATH, conf.SECTIONS_DATA_COLUMNS)

        for sec_ind, sections_data in enumerate([acc_sections_data, dec_sections_data]):
            u = u_hat = omegas = np.array([])
            max_num_modes = 0
            iter_omegas = list()

            for j, section in enumerate(sections_data):
                # get data
                input_path = \
                    f'{conf.CSV_DATA_FOLDER}/{section[conf.SECTIONS_DATA_IDS[0]]}.csv'
                t, v, units, measurement_name = \
                    imp.read_import_measurement_data(input_path, False)

                secs = [section[conf.SECTIONS_DATA_IDS[1]],
                        section[conf.SECTIONS_DATA_IDS[2]]]
                t, v = imp.get_section_by_secs(t, v, secs)

                sampling_period, t, v = \
                    proc.infer_sampling_period(t, v, conf.SAMPLING_PERIOD_TOLERANCE)

                # save data
                label = f'{measurement_name}:\n{secs[0]} s - {secs[1]} s ({section[conf.SECTIONS_DATA_IDS[4]]})'
                if sec_ind == 0:
                    acc_sampling_periods.append(sampling_period)
                    acc_tsvs.append([t, v])
                    acc_labels.append(label)
                    acc_secs.append(secs)
                else:
                    dec_sampling_periods.append(sampling_period)
                    dec_tsvs.append([t, v])
                    dec_labels.append(label)
                    dec_secs.append(secs)

                # run VMD
                u, u_hat, omegas = vmd.decompose_orig(v, conf.ALPHA, conf.K, conf.TAU, conf.DC, conf.INIT, conf.TOL)
                curr_omegas = omegas[-1]
                max_num_modes = conf.K
                

                # save centre frequencies
                iter_omegas.append(curr_omegas)

                sb.draw_status_bar(sec_ind * len(acc_sections_data) + j + 1,
                                   len(acc_sections_data) + len(dec_sections_data), on=conf.STATUS_BAR_ON)

            # create numpy array out of list with mismatched shapes
            if sec_ind == 0:
                acc_result_omegas = np.zeros([len(acc_sections_data), max_num_modes])
                for i, omega in enumerate(iter_omegas):
                    acc_result_omegas[i] = list(omega) + [np.NaN] * (max_num_modes - len(omega))
            else:
                dec_result_omegas = np.zeros([len(dec_sections_data), max_num_modes])
                for i, omega in enumerate(iter_omegas):
                    dec_result_omegas[i] = list(omega) + [np.NaN] * (max_num_modes - len(omega))

        # save results
        sb.draw_status_bar(len(acc_sections_data) + len(dec_sections_data),
                           len(acc_sections_data) + len(dec_sections_data),
                           'Experiment done, saving results...', on=conf.STATUS_BAR_ON)
        with open(conf.OLD_RESULTS_PATH, 'wb') as write_file:
            pickle.dump([acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_labels, acc_secs,
                         dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_labels, dec_secs], write_file)

    else:
        # load old results from file
        print('Loading previous results from file...')
        with open(conf.OLD_RESULTS_PATH, 'rb') as read_file:
            (acc_sampling_periods, acc_tsvs, acc_result_omegas, acc_labels, acc_secs,
             dec_sampling_periods, dec_tsvs, dec_result_omegas, dec_labels, dec_secs) = pickle.load(read_file)

    plo.plot_orig_sections(
        acc_sampling_periods, acc_tsvs, acc_labels,
        dec_sampling_periods, dec_tsvs, dec_labels)

    plo.plot_centre_freqs_by_acceleration_time(
        acc_result_omegas, acc_secs, acc_sampling_periods,
        dec_result_omegas, dec_secs, dec_sampling_periods)
