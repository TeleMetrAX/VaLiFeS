import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length


def get_noise(length: int, noise_csv_path: str = 'speed_last_5min_5Hz.csv', random_seed: int = 42) -> np.ndarray:
    """
    Stationary Bootstrap using the `arch` library
    ==============================================

    Generates bootstrap replicates of a GPS speed noise time series
    using the Stationary Bootstrap method (Politis & Romano, 1994),
    implemented via the `arch` package.

    Input:
        length: the number of timesteps in the time series.
        noise_csv_path: the path to the file containing the original noise time series.
            CSV with columns: timestamp_s, Speed.

    Output:
        noise_ts: noise time series.
    """
    
    # ─── 1. Load the data ────────────────────────────────────────────────────
    df = pd.read_csv(noise_csv_path)
    speed = df["Speed"].values
    n_source = len(speed)

    # ─── 2. Estimate the optimal average block length ────────────────────────
    opt_bl = optimal_block_length(speed)
    avg_block_length = max(1, int(round(opt_bl["stationary"].values[0])))
    
    # ─── 3. Run the Stationary Bootstrap ─────────────────────────────────────
    bs = StationaryBootstrap(avg_block_length, speed, seed=random_seed)
    
    # Calculate how many bootstrap iterations we need to cover 'length'
    # math.ceil(length / n_source)
    iterations_needed = -(-length // n_source) 
    
    # Collect samples until we meet or exceed the requested length
    samples = []
    gen = bs.bootstrap(iterations_needed)
    for _ in range(iterations_needed):
        resampled_tuple, _ = next(gen)
        samples.append(resampled_tuple[0].flatten())
    
    # Concatenate and truncate to the exact requested length
    bootstrap_vec = np.concatenate(samples)[:length]

    return bootstrap_vec
