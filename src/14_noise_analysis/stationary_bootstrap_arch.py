"""
Stationary Bootstrap using the `arch` library
==============================================

Generates bootstrap replicates of a GPS speed noise time series
using the Stationary Bootstrap method (Politis & Romano, 1994),
implemented via the `arch` package.

Requirements:
    pip install numpy pandas arch

Input:
    speed_last_5min_5Hz.csv  –  CSV with columns: timestamp_s, Speed

Output:
    stationary_bootstrap_arch_output/bootstrap_noise.csv
        – Each column is one bootstrap replicate of the noise time series.
          Column names: replicate_0, replicate_1, ...
          The original timestamps are included as the first column.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from arch.bootstrap import StationaryBootstrap, optimal_block_length

# ─── Configuration ────────────────────────────────────────────────────────

CSV_FILE    = Path("speed_last_5min_5Hz.csv")   # Input data file
OUTPUT_DIR  = Path("stationary_bootstrap_arch_output")
N_BOOTSTRAP = 1         # Number of bootstrap replicates to generate
RANDOM_SEED = 42         # For reproducibility

# ─── 1. Load the data ────────────────────────────────────────────────────

df = pd.read_csv(CSV_FILE)
timestamps = df["timestamp_s"].values          # Time axis (seconds)
speed      = df["Speed"].values                # Noise time series
n_samples  = len(speed)

print(f"Loaded {n_samples} samples from {CSV_FILE.name}")
print(f"  Duration : {timestamps[-1]:.1f} s")
print(f"  Mean     : {speed.mean():.6f} m/s")
print(f"  Std      : {speed.std(ddof=1):.6f} m/s")

# ─── 2. Estimate the optimal average block length ────────────────────────
#
# The `arch` library provides `optimal_block_length` which implements
# the Politis & White (2004) / Patton, Politis & White (2009) algorithm.
# It returns a DataFrame with columns "stationary" and "circular" —
# we use the "stationary" column for the Stationary Bootstrap.

opt_bl = optimal_block_length(speed)
avg_block_length = opt_bl["stationary"].values[0]

# Round to nearest integer (must be >= 1)
avg_block_length = max(1, int(round(avg_block_length)))

print(f"\nOptimal average block length: {avg_block_length} samples")

# ─── 3. Run the Stationary Bootstrap ─────────────────────────────────────
#
# `StationaryBootstrap(block_length, data, seed=...)` creates a bootstrap
# generator.  Calling `.bootstrap(n)` yields n resampled datasets.
# Each yielded item is a tuple: (data_tuple, index_array).
#   - data_tuple[0] contains the resampled array (same length as input).

bs = StationaryBootstrap(avg_block_length, speed, seed=RANDOM_SEED)

# Pre-allocate the output matrix: rows = samples, columns = replicates
boot_matrix = np.empty((n_samples, N_BOOTSTRAP))

for i, (resampled_data, _) in enumerate(bs.bootstrap(N_BOOTSTRAP)):
    boot_matrix[:, i] = resampled_data[0].flatten()

print(f"Generated {N_BOOTSTRAP} bootstrap replicates of length {n_samples}")

# ─── 4. Save to CSV ──────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build a DataFrame: first column is timestamps, then one column per replicate
col_names = [f"replicate_{i}" for i in range(N_BOOTSTRAP)]
out_df = pd.DataFrame(boot_matrix, columns=col_names)
out_df.insert(0, "timestamp_s", timestamps)

output_path = OUTPUT_DIR / "bootstrap_noise.csv"
out_df.to_csv(output_path, index=False)

print(f"\nSaved to {output_path}")
print("Done.")
