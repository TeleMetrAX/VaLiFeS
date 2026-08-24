import numpy as np

from generate_noise_bootstrap import get_noise
import main_config as conf
import sim_config as sconf


def get_simulated_data():
    t, v, acc_type = simulate_data()
    if t.size == 0 or v.size == 0:
        raise ValueError("Simulation failed to produce valid data.")

    t, v = remove_points(t, v, sconf.RATIO_TO_REMOVE)

    if sconf.USE_NOISE:
        if sconf.NOISE_METHOD == 'bootstrap':
            n = get_noise(len(v), conf.NOISE_CSV_FILE, conf.NOISE_RANDOM_SEED)
            v = v + n
        elif sconf.NOISE_METHOD == 'gauss':
            t, v = add_noise(t, v, sconf.NOISE_MU, sconf.NOISE_SIGMA)
        else:
            pass
        

    return t, v, acc_type


def simulate_data() -> (np.ndarray, np.ndarray, str):
    # constant acceleration
    if sconf.SIMULATION_TYPE in [0, 1, 2]:
        acc_type = ''
        a_const = 0
        v_0 = 0
        duration = 0
        good_duration = False
        while not good_duration:
            a_const = get_random_constant_acceleration(sconf.SIMULATION_TYPE, sconf.A_MAX, sconf.A_MIN)
            v_0 = sconf.V_MIN if a_const > 0 else sconf.V_MAX
            acc_type = conf.ACC_TYPE if a_const > 0 else conf.DEC_TYPE
            duration = np.abs((sconf.V_MAX - sconf.V_MIN) / a_const)
            if sconf.D_MIN <= duration <= sconf.D_MAX:
                good_duration = True
        t = generate_time_random_sampling_period(
            duration, sconf.SAMPLING_PERIOD, sconf.USE_RANDOM_SAMPLING_PERIOD,
            sconf.SAMPLING_PERIOD_MU, sconf.SAMPLING_PERIOD_SIGMA)
        v = v_0 + a_const * t
        return t, v, acc_type

    # sine acceleration
    elif sconf.SIMULATION_TYPE == 3:
        t = None
        v = None
        good = False
        while not good:
            duration = np.random.uniform(sconf.D_MIN, sconf.D_MAX)
            t = generate_time_random_sampling_period(
                duration, sconf.SAMPLING_PERIOD, sconf.USE_RANDOM_SAMPLING_PERIOD,
                sconf.SAMPLING_PERIOD_MU, sconf.SAMPLING_PERIOD_SIGMA)

            v = (sconf.V_MAX - sconf.V_MIN) / 2 * (1 - np.cos(np.pi * t / duration)) + sconf.V_MIN
            a = (sconf.V_MAX - sconf.V_MIN) / 2 * (np.pi / duration) * np.sin(np.pi * t / duration)
            if np.all(a <= sconf.A_MAX) and np.all(a >= sconf.A_MIN):
                good = True
        return t, v, conf.ACC_TYPE

    # logistic acceleration
    elif sconf.SIMULATION_TYPE == 4:
        t = None
        v = None
        good = False
        while not good:
            duration = np.random.uniform(sconf.D_MIN, sconf.D_MAX)
            t = generate_time_random_sampling_period(
                duration, sconf.SAMPLING_PERIOD, sconf.USE_RANDOM_SAMPLING_PERIOD,
                sconf.SAMPLING_PERIOD_MU, sconf.SAMPLING_PERIOD_SIGMA)

            L = sconf.V_MAX - sconf.V_MIN
            x_0 = duration / 2
            k = 10 / duration

            v = L / (1 + np.exp(-k * (t - x_0))) + sconf.V_MIN
            a = (L * k * np.exp(-k * (t - x_0))) / ((1 + np.exp(-k * (t - x_0)))**2)
            if np.all(a <= sconf.A_MAX) and np.all(a >= sconf.A_MIN):
                good = True
        return t, v, conf.ACC_TYPE

    return np.array([]), np.array([]), ''


def get_random_constant_acceleration(simulation_type, a_max, a_min):
    if simulation_type == 0:
        return np.random.uniform(0, a_max)
    elif simulation_type == 1:
        return np.random.uniform(a_min, 0)
    elif simulation_type == 2:
        return np.random.uniform(a_min, a_max)
    else:
        raise ValueError(f"Invalid simulation type: {simulation_type}. Must be 0, 1, or 2.")


def remove_points(t: np.ndarray, v: np.ndarray, ratio_to_remove: float) -> (np.ndarray, np.ndarray):
    if not (0 <= ratio_to_remove < 1):
        raise ValueError("ratio_to_remove must be in the range [0, 1).")

    num_to_remove = int(len(t) * ratio_to_remove)
    poi_indices = np.random.choice(len(t), size=num_to_remove, replace=False)

    mask = np.ones(len(t), dtype=bool)
    mask[poi_indices] = False

    return t[mask], v[mask]


def add_noise(t, v, noise_mu: float, noise_sigma: float) -> (np.ndarray, np.ndarray):
    noise = np.random.normal(noise_mu, noise_sigma, size=v.shape)
    v_noisy = v + noise
    v_noisy = np.clip(v_noisy, sconf.V_MIN, sconf.V_MAX)
    return t, v_noisy


def add_acceleration_noise_to_speed(
        t: np.ndarray, v: np.ndarray, a_noise_mu: float, a_noise_sigma: float) -> (np.ndarray, np.ndarray):
    a_noise = np.random.normal(a_noise_mu, a_noise_sigma, size=v.shape)
    v_noisy = np.copy(v)
    for i in range(1, len(t)):
        delta_t = t[i] - t[i - 1]
        v_noisy[i] += a_noise[i] * delta_t
    v_noisy = np.clip(v_noisy, sconf.V_MIN, sconf.V_MAX - 1e-6)
    return t, v_noisy


def generate_time_random_sampling_period(
        duration: float, nonrandom_sampling_period: float, use_random: bool,
        sampling_period_mu: float, sampling_period_sigma: float) -> np.ndarray:
    if use_random:
        t = np.array([0.0])
        while t[-1] < duration:
            sampling_period = np.random.normal(sampling_period_mu, sampling_period_sigma)
            if sampling_period <= 0.0:
                continue
            next_time = t[-1] + sampling_period
            if next_time > duration:
                break
            t = np.append(t, next_time)
        return t
    else:
        return np.arange(0, duration, nonrandom_sampling_period)
