import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import plotter as plo


def infer_sampling_period(
        t: np.ndarray, v: np.ndarray, tolerance: float = 1e-6,
        precision: int = 6, target_sampling_period: float = None) -> (float, np.ndarray, np.ndarray):
    """
    :param t: time data [s]
    :param v: speed data [no matter the unit]
    :param tolerance: allowed tolerance from the average of time jumps [s]
    :param precision: number of significant decimal places in the sampling period [s]
    :param target_sampling_period: creates new time array with this sampling period, if specified [s]
    :return: sampling period [s]
    """
    # calculate sampling period
    most_common_time_jump, bads = get_sampling_period_values(t, tolerance, precision)
    if np.sum(bads) == 0:
        return float(most_common_time_jump), t, v

    # attempt filling in holes via interpolation
    # (base * (np.array(x) / base).round()).round(prec)
    diffs_round = (tolerance * (np.diff(t) / tolerance).round()).round(precision)
    unique_diffs, diffs_counts = np.unique(diffs_round, return_counts=True)
    step = unique_diffs[np.argmax(diffs_counts)]
    if step == 0:
        step = unique_diffs[np.argsort(diffs_counts)[-2]]
    if target_sampling_period is not None:
        new_t = np.arange(t.min(), t.max() + target_sampling_period, target_sampling_period)
    else:
        new_t = np.arange(t.min(), t.max() + step, step)
    new_v = np.interp(new_t, t, v)

    most_common_time_jump, bads = get_sampling_period_values(new_t, tolerance, precision)

    if np.sum(bads) == 0:
        if target_sampling_period is not None:
            return target_sampling_period, new_t, new_v
        else:
            return float(most_common_time_jump), new_t, new_v

    # if it still fails,
    raise ValueError(f'Non-constant sampling period in acceleration data!\n'
                     f'{np.sum(bads)} elements contain a timestamp larger than expected by other jumps\n'
                     f'(sampling period := average jump = {most_common_time_jump} s).\n'
                     f'Allowed tolerance from the average is {tolerance} s.\n'
                     f'Filling in missing data by average could not fix the issue either.')


def get_sampling_period_values(t: np.ndarray, tolerance: float, precision: int):
    diffs_round = (tolerance * (np.diff(t) / tolerance).round()).round(precision)
    unique_diffs, diffs_counts = np.unique(diffs_round, return_counts=True)
    most_common_time_jump = unique_diffs[np.argmax(diffs_counts)]
    if most_common_time_jump == 0:
        most_common_time_jump = unique_diffs[np.argsort(diffs_counts)[-2]]
    diffs = np.abs(np.diff(t) - most_common_time_jump * np.ones(t.shape[0] - 1))
    bads = diffs > tolerance
    return most_common_time_jump, bads


def get_acceleration_from_speed(v: np.ndarray, sampling_period: float):
    delta_v = v[1:] - v[:-1]
    return delta_v / sampling_period


def get_moving_average(data: np.ndarray, window: int = 3):
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def infer_acceleration_deceleration_sections(
        t, v, measurement_name: str, speed_tolerance: float,
        top_speed: float, sections_df_columns: list) -> pd.DataFrame:

    zeroes = (v < speed_tolerance)
    top_speeds = np.logical_and(v > (top_speed - speed_tolerance), v < (top_speed + speed_tolerance))

    # plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, [], [], [])

    zero_sections = get_zero_top_speed_sections(zeroes)
    top_speed_sections = get_zero_top_speed_sections(top_speeds)

    # plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections, [])

    z_ind = 0
    ts_ind = 0
    current_section = [-1, -1, 'nothing']
    all_sections = list()
    if len(zero_sections) > 0 and len(top_speed_sections) > 0:
        while current_section[1] < len(v) - 1:
            curr_beg = current_section[1] + 1
            if z_ind < zero_sections.shape[0] and ts_ind < top_speed_sections.shape[0]:
                # a zero section has just begun
                if zero_sections[z_ind][0] == curr_beg:
                    curr_end = zero_sections[z_ind][1]
                    curr_type = 'zero'
                    z_ind += 1
                # a top speed section has just begun
                elif top_speed_sections[ts_ind][0] == curr_beg:
                        curr_end = top_speed_sections[ts_ind][1]
                        curr_type = 'top_speed'
                        ts_ind += 1
                # a zero section has just ended
                elif zero_sections[z_ind-1][1] + 1 == curr_beg:
                    following_top_speed_section = [s for s in top_speed_sections if s[0] > curr_beg][0]
                    if zero_sections[z_ind][0] < following_top_speed_section[0]:
                        curr_end = zero_sections[z_ind][0] - 1
                        curr_type = 'nothing'
                    else:
                        curr_end = following_top_speed_section[0] - 1
                        curr_type = 'acc'
                # a top speed section has just ended
                elif top_speed_sections[ts_ind-1][1] + 1 == curr_beg:
                    following_zero_section = [s for s in zero_sections if s[0] > curr_beg][0]
                    if top_speed_sections[ts_ind][0] < following_zero_section[0]:
                        curr_end = top_speed_sections[ts_ind][0] - 1
                        curr_type = 'nothing'
                    else:
                        curr_end = following_zero_section[0] - 1
                        curr_type = 'dec'
                # we are at the beginning in the nothing range
                elif zero_sections[z_ind][0] > curr_beg and top_speed_sections[ts_ind][0] > curr_beg:
                    curr_end = np.min([zero_sections[z_ind][0], top_speed_sections[ts_ind][0]]) - 1
                    curr_type = 'nothing'
                else:
                    raise NotImplementedError('There should be no other case (both indices small)')

            # ts_ind is too big: no more top_speed sections
            elif z_ind < zero_sections.shape[0]:
                # a zero section has just begun
                if zero_sections[z_ind][0] == curr_beg:
                    curr_end = zero_sections[z_ind][1]
                    curr_type = 'zero'
                    z_ind += 1
                # a zero section has just ended
                elif zero_sections[z_ind - 1][1] + 1 == curr_beg:
                    curr_end = zero_sections[z_ind][0] - 1
                    curr_type = 'nothing'
                # a top speed section has just ended
                elif top_speed_sections[ts_ind - 1][1] + 1 == curr_beg:
                    following_zero_section = [s for s in zero_sections if s[0] > curr_beg][0]
                    curr_end = following_zero_section[0] - 1
                    curr_type = 'dec'
                else:
                    raise NotImplementedError('There should be no other case (z_ind small)')

            # z_ind is too big: no more zero sections
            elif ts_ind < top_speed_sections.shape[0]:
                # a top speed section has just begun
                if top_speed_sections[ts_ind][0] == curr_beg:
                    curr_end = top_speed_sections[ts_ind][1]
                    curr_type = 'top_speed'
                    ts_ind += 1
                # a zero section has just ended
                elif zero_sections[z_ind - 1][1] + 1 == curr_beg:
                    following_top_speed_section = [s for s in top_speed_sections if s[0] > curr_beg][0]
                    curr_end = following_top_speed_section[0] - 1
                    curr_type = 'acc'
                # a top speed section has just ended
                elif top_speed_sections[ts_ind - 1][1] + 1 == curr_beg:
                    curr_end = top_speed_sections[ts_ind][0] - 1
                    curr_type = 'nothing'
                else:
                    raise NotImplementedError('There should be no other case (ts_ind small)')

            elif z_ind == zero_sections.shape[0] and ts_ind == top_speed_sections.shape[0]:
                curr_end = len(v) - 1
                curr_type = 'nothing'
            else:
                print('z_ind', z_ind)
                print('zero_sections.shape[0]', zero_sections.shape[0])
                print('ts_ind', ts_ind)
                print('top_speed_sections.shape[0]', top_speed_sections.shape[0])
                # plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections,
                #                              all_sections)
                raise NotImplementedError('There should be no other case (main loop)')

            current_section = [curr_beg, curr_end, curr_type]
            all_sections.append(current_section)

        # plo.plot_zero_top_speed_sections(
        #     t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections,
        #     all_sections, measurement_name, 'km/h')

        # extract only acceleration and deceleration sections
        all_sections_df = pd.DataFrame(all_sections, columns=sections_df_columns)
        return all_sections_df.loc[all_sections_df.iloc[:, 2].isin(['acc', 'dec'])]

    else:
        return pd.DataFrame(columns=sections_df_columns)


def get_zero_top_speed_sections(boolean_arr: np.ndarray):
    section_begs = np.logical_and(boolean_arr[1:], np.logical_not(boolean_arr[:-1])).nonzero()[0] + 1
    section_ends = np.logical_and(np.logical_not(boolean_arr[1:]), boolean_arr[:-1]).nonzero()[0]

    # add very first and very last indicies to begin and finish off sections
    if boolean_arr[0]:
        section_begs = np.insert(section_begs, 0, 0)
    if boolean_arr[-1]:
        section_ends = np.insert(section_ends, len(section_ends), len(boolean_arr)-1)

    # remove a section end if the data begins with neither a zero, nor a top speed
    if len(section_begs) > 0 and len(section_ends) > 0:
        if section_begs[0] > section_ends[0]:
            section_ends = section_ends[1:]

    if len(section_begs) > len(section_ends):
        section_ends = np.array(list(section_ends) + [np.NaN] * (len(section_begs) - len(section_ends)))

    sections = np.array([section_begs, section_ends]).transpose()

    return sections


# def plot_zero_top_speed_sections(
#         t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections, all_sections):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(t, v, color='g')
#     ax.axhline(speed_tolerance, color='k', linestyle='--')
#     ax.axhline(top_speed - speed_tolerance, color='k', linestyle='--')
#     ax.axhline(top_speed, color='k')
#     ax.axhline(top_speed + speed_tolerance, color='k', linestyle='--')
#     for s in zero_sections:
#         ax.axvline(t[s[0]], color='orange', linestyle='--')
#         ax.axvline(t[s[1]], color='orange')
#     for s in top_speed_sections:
#         ax.axvline(t[s[0]], color='r', linestyle='--')
#         ax.axvline(t[s[1]], color='r')
#     for sec in all_sections:
#         text_x = t[(sec[0] + sec[1]) // 2]
#         text_y = get_text_y(sec[2], speed_tolerance)
#         ax.text(text_x, text_y, sec[2], ha='center')
#
#     ax.grid()
#     plt.show()
#     return


def get_text_y(name, speed_tolerance):
    if name == 'nothing':
        return speed_tolerance
    if name in ['zero', 'top_speed']:
        return 2 * speed_tolerance
    if name in ['acc', 'dec']:
        return 3 * speed_tolerance
    return 4 * speed_tolerance


def smooth_timeseries(t, v, bandwidth_seconds=5, rate=14, degree=2):
    """
    Smooth the 'Speed' column using local polynomial regression.

    Parameters:
    - t, v: Numpy arrays with time [s] and speeds.
    - bandwidth_seconds: Window size in seconds (default 5).
    - rate: Samples per second (default 14 Hz).
    - degree: Degree of the polynomial (default 2).

    Returns:
    - Numpy arrays t, v (times and speeds).
    """
    smoothed = np.empty_like(v)

    window_half_size = int((bandwidth_seconds * rate) // 2)
    n_points = len(t)

    for i in range(n_points):
        # Determine the window indices
        start = max(0, i - window_half_size)
        end = min(n_points - 1, i + window_half_size)
        window_indices = slice(start, end + 1)

        # window_times = t[window_indices]
        window_times = t[window_indices].reshape(-1, 1)
        window_speeds = v[window_indices]

        if len(window_times) > 1:
        # if len(window_times) >= degree + 1:
            # # Fit polynomial
            # coeffs = np.polyfit(window_times, window_speeds, deg=degree)

            # Fit linear regression
            model = LinearRegression()
            model.fit(window_times, window_speeds)
            # Predict current time
            smoothed[i] = model.predict([[t[i]]])[0]
        else:
            # Not enough points; use original value
            smoothed[i] = v[i]

    return t, smoothed


def get_mode_energy(mode: np.ndarray) -> float:
    return np.sum(np.square(np.abs(mode)))


def get_highest_acctime_frequencies_times(
        result_omegas: np.ndarray, secs: np.ndarray, sampling_period: float):
    acc_durations = [s[1] - s[0] for s in secs]
    max_accdur_index = np.argmax(acc_durations)
    max_accdur_omegas = result_omegas[max_accdur_index]
    real_center_freqs = np.sort([o / sampling_period for o in max_accdur_omegas])
    real_inv_cfs = 1 / real_center_freqs
    return real_center_freqs, real_inv_cfs, acc_durations[max_accdur_index]
