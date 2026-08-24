import numpy as np
import pandas as pd

import plotter as plo


def infer_sampling_period(
        t: np.ndarray, v: np.ndarray, tolerance: float = 1e-6) -> (float, np.ndarray, np.ndarray):
    """
    :param t: time data [s]
    :param v: speed data [no matter the unit]
    :param tolerance: allowed tolerance from the average of time jumps [s]
    :return: sampling period [s]
    """
    # calculate sampling period
    avg_time_jump, bads = get_sampling_period_values(t, tolerance)
    if np.sum(bads) == 0:
        return float(avg_time_jump), t, v

    # attempt filling in holes via interpolation
    step = np.median(np.diff(t))
    new_t = np.arange(t.min(), t.max() + step, step)
    new_v = np.interp(new_t, t, v)

    avg_time_jump, bads = get_sampling_period_values(new_t, tolerance)

    if np.sum(bads) == 0:
        return float(avg_time_jump), new_t, new_v

    # if it still fails,
    raise ValueError(f'Non-constant sampling period in acceleration data!\n'
                     f'{np.sum(bads)} elements contain a timestamp larger than expected by other jumps\n'
                     f'(sampling period := average jump = {avg_time_jump} s).\n'
                     f'Allowed tolerance from the average is {tolerance} s.\n'
                     f'Filling in missing data by average could not fix the issue either.')


def get_sampling_period_values(t: np.ndarray, tolerance: float):
    time_jumps = t[1:] - t[:-1]
    avg_time_jump = np.mean(time_jumps)
    diffs = np.abs(time_jumps - avg_time_jump * np.ones(t.shape[0] - 1))
    bads = diffs > tolerance
    return avg_time_jump, bads


def get_acceleration_from_speed(v: np.ndarray, sampling_period: float):
    delta_v = v[1:] - v[:-1]
    return delta_v / sampling_period


def get_moving_average(data: np.ndarray, window: int = 3):
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def infer_acceleration_deceleration_sections(
        t, v, measurement_name: str, speed_tolerance: float,
        top_speed: float, sections_df_columns: list,
        specific_start_index: int = None, specific_end_index: int = None) -> pd.DataFrame:

    zeroes = (v < speed_tolerance)
    top_speeds = np.logical_and(v > (top_speed - speed_tolerance), v < (top_speed + speed_tolerance))

    # plo.plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, [], [], [])

    zero_sections = get_zero_top_speed_sections(zeroes)
    top_speed_sections = get_zero_top_speed_sections(top_speeds)

    # plo.plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections, [])

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
                plo.plot_zero_top_speed_sections(t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections,
                                             all_sections, 'debugging')
                raise NotImplementedError('There should be no other case (main loop)')

            current_section = [curr_beg, curr_end, curr_type]
            all_sections.append(current_section)

        plo.plot_zero_top_speed_sections(
            t, v, top_speed, speed_tolerance, zero_sections, top_speed_sections, all_sections,
            measurement_name, 'km/h', specific_start_index, specific_end_index)

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
        section_ends = np.array(list(section_ends) + [np.nan] * (len(section_begs) - len(section_ends)))

    sections = np.array([section_begs, section_ends]).transpose()

    return sections
