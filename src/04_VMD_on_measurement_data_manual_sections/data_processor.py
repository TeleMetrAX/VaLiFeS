import numpy as np


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
