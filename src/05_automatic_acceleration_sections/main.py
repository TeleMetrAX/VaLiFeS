import glob

import main_config as conf
import importer as imp
import data_processor as proc


if __name__ == '__main__':
    measurement_paths = glob.glob(f'{conf.MEASUREMENTS_FOLDER}/*.csv')

    if conf.SPECIFIC_MEASUREMENT_INDEX is not None:
        measurement_paths = [measurement_paths[conf.SPECIFIC_MEASUREMENT_INDEX]]

    for measurement_path in measurement_paths[conf.START_INDEX:]:
        t, v, units, measurement_name =\
            imp.read_import_measurement_data(measurement_path, False)

        acc_dec_sections_df = proc.infer_acceleration_deceleration_sections(
            t, v, measurement_name, conf.SPEED_TOLERANCE, conf.TOP_SPEED,
            conf.SECTIONS_DF_COLUMNS, conf.SPECIFIC_START_INDEX, conf.SPECIFIC_END_INDEX)

        if conf.ACTUALLY_SAVE:
            filename = measurement_path.replace('\\', '/').split('/')[-1][:-4]
            acc_dec_sections_df.to_csv(f'{conf.RESULTS_FOLDER}/{filename}.csv', index=False)
