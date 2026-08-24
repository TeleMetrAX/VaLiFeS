import numpy as np
import pandas as pd


def read_import_measurement_data(
        measurement_path: str, start_at_zero_seconds: bool, speed_unit: str = 'm/s'):
    """
    :param measurement_path:
    :param start_at_zero_seconds
    :param speed_unit: 'm/s' | 'km/h'
    :return: t (time data), v (speed data), units ('m/s' | 'km/h'), measurement_name
    """
    measurement_name = measurement_path.replace(
        '\\', '/').split('/')[-1][:-4].replace('_', '/')
    measurement_df = pd.read_csv(measurement_path, low_memory=False)
    units = measurement_df.iloc[0, :].to_numpy()
    t = measurement_df.iloc[1:, 0].to_numpy(dtype=float)
    v = measurement_df.iloc[1:, 1].to_numpy(dtype=float)
    if speed_unit == 'km/h':
        v *= 3.6
        units[1] = 'km/h'
    if start_at_zero_seconds:
        t = t - t[0]
    return t, v, units, measurement_name


def get_section_by_secs(t: np.ndarray, v: np.ndarray, secs: list,
                        start_at_zero_seconds: bool = True) -> (np.ndarray, np.ndarray):
    beg_ind = (t < secs[0]).nonzero()[0][-1]
    end_ind = (t < secs[1]).nonzero()[0][-1]
    t_section = t[beg_ind:end_ind+1]
    v_section = v[beg_ind:end_ind+1]

    if start_at_zero_seconds:
        t_section = t_section - t_section[0]

    return t_section, v_section


def get_section_by_indices(t: np.ndarray, v: np.ndarray, indices: list,
                           start_at_zero_seconds: bool = True) -> (np.ndarray, np.ndarray):
    beg_ind = indices[0]
    end_ind = indices[1]
    t_section = t[beg_ind:end_ind+1]
    v_section = v[beg_ind:end_ind+1]

    if start_at_zero_seconds:
        t_section = t_section - t_section[0]

    return t_section, v_section


def read_sections_data(sections_path: str, expected_columns: list):
    df = pd.read_csv(sections_path)
    if list(df.columns) == expected_columns:
        return df
    else:
        raise ValueError(f'Columns of sections data ({df.columns}) and '
                         f'expected columns ({expected_columns}) don\'t match!')

def get_driver(data_filename: str):
    parts = data_filename.split('.')[0].split('_')
    folder = int(parts[0])
    file = int(parts[1])
    if 1 <= folder <= 12:
        return 'Tien I.'
    elif 13 <= folder <= 30:
        return 'Karesz I.'
    elif 31 <= folder <= 40:
        return 'Tien II.'
    elif 41 <= folder <= 46:
        return 'Tien III.'
