import numpy as np
import pandas as pd


def get_random_velo_data(length, status, acc, phase, sd, rate=14, noise_per_sec=8, control=15,
                         initial_warmup_min=7, warmup_acc_sigma=0.1):
    """
    Generates velocity time series with initial warm-up period and proportional state allocation.
    :param length: Total running time of the car (minutes)
    :param status: A list or tuple of three probabilities for selecting the car's status
              ("increasing", "decreasing", or "stable"). For example, [0.4, 0.3, 0.3].
              The status is chosen every 'phase' seconds.
    :param phase: duration for that the car guarantees not change status [seconds]
    :param acc: A two-item tuple representing the parameters (mean, sigma) for the lognormal
           distribution used to sample the magnitude of the acceleration [m/s^2].
           When the car is decreasing, the sampled value will be minus absolute values.
    :param rate: Sampling rate in Hz.
    :param sd: standard deviation of white noise (Gaussian noise)
    :param control: control parameter to get synthetic data being close form to real data (default is 15, get from experiments)
    :param noise_per_sec: number of white noises per second (Hz)
    (we do not need add noise to every data points as it's not looked naturally, preferably around 50% of data points per second is enough)

    New Parameters:

    :param initial_warmup_min: Initial warm-up duration in minutes (0 by default)
    :param warmup_acc_sigma: Acceleration noise during warm-up (controls start smoothness)
    """
    total_time_sec = length * 60
    initial_warmup_sec = initial_warmup_min * 60

    # Validate input constraints
    if initial_warmup_sec > status[2] * total_time_sec:
        raise ValueError(f"Initial warm-up exceeds {status[2]} percent - stable time of total time")

    # Calculate remaining time allocation
    remaining_time_sec = total_time_sec - initial_warmup_sec
    required_stable = status[2] * total_time_sec - initial_warmup_sec
    required_inc = status[0] * total_time_sec
    required_dec = status[1] * total_time_sec

    # Adjusted probabilities for remaining time
    prob_inc = required_inc / remaining_time_sec
    prob_dec = required_dec / remaining_time_sec
    prob_stable = required_stable / remaining_time_sec
    adj_probs = [prob_inc, prob_dec, prob_stable]

    # Initialize data structures
    dt = 1.0 / rate
    total_samples = int(total_time_sec * rate)
    velocities = np.zeros(total_samples)
    current_velocity = 0.0
    states = ["increasing", "decreasing", "stable"]
    index = 0

    # Handle warm-up period
    if initial_warmup_min > 0:
        warmup_samples = int(initial_warmup_sec * rate)
        a_warmup = np.random.normal(0, warmup_acc_sigma, warmup_samples)
        # / control

        for j in range(warmup_samples):
            current_velocity += a_warmup[j] * dt
            current_velocity = np.clip(current_velocity, 0,
                                       14)  # control maximum velocity does not exceed 14 m/s (52km/h)
            velocities[index + j] = current_velocity
        index += warmup_samples

    # Main processing loop for remaining time
    period_samples = int(phase * rate)
    while index < total_samples:
        block_length = min(period_samples, total_samples - index)
        current_status = np.random.choice(states, p=adj_probs if initial_warmup_min else status)

        if current_status == "increasing":
            a = np.random.lognormal(*acc, block_length) / control
        elif current_status == "decreasing":
            a = -np.random.lognormal(*acc, block_length) / control
        else:
            a = np.zeros(block_length)

        for j in range(block_length):
            current_velocity += a[j] * dt
            current_velocity = np.clip(current_velocity, 0, 14)
            velocities[index + j] = current_velocity
        index += block_length

    # Generate timestamps and noise
    t = pd.date_range(start=pd.Timestamp.now().floor('s'), periods=len(velocities), freq=f"{1000 // rate}ms")
    noise = np.zeros_like(velocities)

    for sec in range(int(np.ceil(len(velocities) / rate))):
        start = sec * rate
        end = start + rate
        positions = np.random.choice(rate, noise_per_sec, replace=False) + start
        positions = positions[positions < len(velocities)]
        noise[positions] = np.random.normal(0, sd, len(positions))

    speed_noise = np.clip(velocities + noise, 0, None)

    data = pd.DataFrame({'timestamps': t, 'Speed': velocities})
    noisy_df = pd.DataFrame({'timestamps': t, 'Speed': speed_noise})
    return data, noisy_df
