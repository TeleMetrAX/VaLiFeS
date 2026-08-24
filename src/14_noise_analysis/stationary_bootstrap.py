"""
Stationary Bootstrap (Politis & Romano, 1994) for GPS Speed Noise
===================================================================

Implements the nonparametric stationary bootstrap on the
noise data.  It generates bootstrap replicates and validates the
simulation's reliability against the real data using:

  1. Basic statistics comparison (mean, std, skewness, kurtosis)
  2. Distribution overlays & KS / Anderson-Darling tests
  3. Autocorrelation function (ACF) comparison
  4. Power Spectral Density (PSD) comparison
  5. Allan Deviation (OADEV) comparison
  6. Stationarity tests on bootstrap replicates
  7. Bootstrap confidence intervals for summary statistics

"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for reliable saving
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import (ks_2samp, anderson_ksamp, skew, kurtosis,
                         shapiro, normaltest)
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf, adfuller, kpss
from pathlib import Path
import warnings
import time as timer

warnings.filterwarnings('ignore')

# ── Optional: allantools ──────────────────────────────────────────────────
try:
    import allantools
    HAS_ALLANTOOLS = True
except ImportError:
    HAS_ALLANTOOLS = False
    print("WARNING: allantools not installed – Allan Deviation comparison skipped.")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CSV_FILE       = Path("speed_last_5min_5Hz.csv")
OUTPUT_FOLDER  = Path("stationary_bootstrap_output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

FS              = 5.0          # Sampling rate (Hz)
DT              = 1.0 / FS     # 0.2 s
N_BOOTSTRAP     = 10000        # Number of bootstrap replicates
RANDOM_SEED     = 42           
CONFIDENCE      = 0.95         # For confidence intervals

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD REAL DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  STATIONARY BOOTSTRAP  –  GPS Speed Noise Simulation Validation")
print("=" * 72)

df = pd.read_csv(CSV_FILE)
time_s = df['timestamp_s'].values
speed  = df['Speed'].values
N      = len(speed)

print(f"\n  Real data loaded    : {CSV_FILE.name}")
print(f"  Samples             : {N}")
print(f"  Duration            : {time_s[-1]:.1f} s  ({time_s[-1]/60:.2f} min)")
print(f"  Sampling rate       : {FS} Hz")
print(f"  Mean speed          : {np.mean(speed):.6f} m/s")
print(f"  Std  speed          : {np.std(speed, ddof=1):.6f} m/s")

# ── Stationarity tests on real data ───────────────────────────────────────
print("\n" + "─" * 72)
print("  STATIONARITY TESTS ON REAL DATA (prerequisite check)")
print("─" * 72)

# ADF test (H0: unit root / non-stationary)
adf_real_stat, adf_real_p, adf_real_lags, _, adf_real_cv, _ = adfuller(speed, autolag='AIC')
print(f"  ADF test statistic  : {adf_real_stat:.4f}")
print(f"  ADF p-value         : {adf_real_p:.4e}")
print(f"  ADF lags used       : {adf_real_lags}")
print(f"  ADF critical values : {adf_real_cv}")
print(f"  → {'STATIONARY' if adf_real_p < 0.05 else 'NON-STATIONARY'} (reject H0 at 5%)")

# KPSS test (H0: stationary)
kpss_real_stat, kpss_real_p, kpss_real_lags, kpss_real_cv = kpss(speed, regression='c', nlags='auto')
print(f"\n  KPSS test statistic : {kpss_real_stat:.4f}")
print(f"  KPSS p-value        : {kpss_real_p:.4f}")
print(f"  KPSS lags used      : {kpss_real_lags}")
print(f"  KPSS critical values: {kpss_real_cv}")
print(f"  → {'STATIONARY' if kpss_real_p > 0.05 else 'NON-STATIONARY'} (fail to reject H0 at 5%)")

if adf_real_p < 0.05 and kpss_real_p > 0.05:
    real_stationarity_verdict = "STATIONARY (both ADF and KPSS agree)"
elif adf_real_p >= 0.05 and kpss_real_p <= 0.05:
    real_stationarity_verdict = "NON-STATIONARY (both ADF and KPSS agree)"
elif adf_real_p < 0.05 and kpss_real_p <= 0.05:
    real_stationarity_verdict = "TREND-STATIONARY (ADF rejects unit root, KPSS rejects level-stationarity)"
else:
    real_stationarity_verdict = "INCONCLUSIVE (ADF fails to reject unit root, KPSS fails to reject stationarity)"

print(f"\n  Combined verdict    : {real_stationarity_verdict}")

# ══════════════════════════════════════════════════════════════════════════
# 2. OPTIMAL BLOCK LENGTH SELECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 1: OPTIMAL BLOCK LENGTH SELECTION")
print("─" * 72)


def estimate_block_length_stable_ci_crossing(x, n_consecutive=3, confidence=0.95):
    """
    Estimate optimal block length L as the lag where the ACF *stably*
    crosses inside the confidence interval.

    "Stably" means the ACF stays within ± CI for at least `n_consecutive`
    consecutive lags — not just a single transient dip.  This captures
    the full correlation length of the series.

    Parameters
    ----------
    x : array-like
        Stationary time series.
    n_consecutive : int
        Number of consecutive lags that must all be inside the CI
        before declaring a stable crossing (default: 3).
    confidence : float
        Confidence level for the white-noise CI bands (default: 0.95).

    Returns
    -------
    L_opt : int
        Recommended average block length (≥ 1).
    p_opt : float
        Corresponding geometric parameter p = 1 / L_opt.
    crossing_lag : int
        The lag at which the ACF first stably enters the CI.
    acf_vals : ndarray
        The computed ACF values (for diagnostic plotting).
    ci_threshold : float
        The CI threshold used (z / √n).
    """
    from scipy.stats import norm as norm_dist

    n = len(x)
    max_lag = min(n // 2, 200)
    acf_vals = acf(x, nlags=max_lag, fft=True)

    z = norm_dist.ppf(0.5 + confidence / 2.0)   # e.g. 1.96 for 95%
    ci_threshold = z / np.sqrt(n)

    # Walk through lags and find first run of n_consecutive inside CI
    crossing_lag = max_lag  # fallback: never crossed stably
    run = 0
    for k in range(1, len(acf_vals)):
        if abs(acf_vals[k]) <= ci_threshold:
            run += 1
            if run >= n_consecutive:
                crossing_lag = k - n_consecutive + 1  # start of the run
                break
        else:
            run = 0

    L_opt = max(1, crossing_lag)
    # Clamp to sensible range
    L_opt = min(L_opt, n // 5)
    p_opt = 1.0 / L_opt

    return L_opt, p_opt, crossing_lag, acf_vals, ci_threshold


L_opt, p_opt, acf_crossing_lag, acf_full, ci_thresh = \
    estimate_block_length_stable_ci_crossing(speed, n_consecutive=3, confidence=0.95)

print(f"  95% CI threshold       : ±{ci_thresh:.4f}")
print(f"  ACF stable crossing    : lag {acf_crossing_lag}  "
      f"({acf_crossing_lag * DT:.2f} s)")
print(f"  → Block length L       : {L_opt}")
print(f"  → Geometric param p    : {p_opt:.4f}")
print(f"  → Avg block duration   : {L_opt * DT:.2f} s")

# ══════════════════════════════════════════════════════════════════════════
# 3. STATIONARY BOOTSTRAP ALGORITHM
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 2: STATIONARY BOOTSTRAP (Politis & Romano, 1994)")
print("─" * 72)


def stationary_bootstrap(data, p, n_boot, rng):
    """
    Generate n_boot bootstrap replicates of *data* using the stationary
    bootstrap of Politis & Romano (1994).

    Parameters
    ----------
    data  : 1-D array of length n.
    p     : Geometric probability parameter (1/avg_block_length).
    n_boot: Number of bootstrap replicates.
    rng   : numpy.random.Generator instance.

    Returns
    -------
    boot_matrix : (n_boot, n) array of bootstrap replicates.
    """
    n = len(data)
    boot_matrix = np.empty((n_boot, n), dtype=data.dtype)

    for b in range(n_boot):
        # Precompute all random numbers for this replicate (vectorised)
        starts = rng.integers(0, n, size=n)         # uniform start indices
        coins  = rng.random(size=n)                  # U(0,1) coin flips

        j = starts[0]                                # first position always new block
        boot_matrix[b, 0] = data[j]

        for t in range(1, n):
            if coins[t] < p:
                # Start a new block
                j = starts[t]
            else:
                # Continue current block (wrap around circularly)
                j = (j + 1) % n
            boot_matrix[b, t] = data[j]

    return boot_matrix


rng = np.random.default_rng(RANDOM_SEED)

t0 = timer.time()
boot_matrix = stationary_bootstrap(speed, p_opt, N_BOOTSTRAP, rng)
elapsed = timer.time() - t0

print(f"  Generated {N_BOOTSTRAP} bootstrap replicates of length {N}")
print(f"  Avg block length        : {1/p_opt:.1f} samples ({1/p_opt*DT:.2f} s)")
print(f"  Elapsed time            : {elapsed:.2f} s")

# ══════════════════════════════════════════════════════════════════════════
# 4. SUMMARY STATISTICS OF BOOTSTRAP REPLICATES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 3: BOOTSTRAP SUMMARY STATISTICS")
print("─" * 72)

# Compute statistics for each replicate
boot_means = np.mean(boot_matrix, axis=1)
boot_stds  = np.std(boot_matrix, axis=1, ddof=1)
boot_skews = np.array([skew(row) for row in boot_matrix])
boot_kurts = np.array([kurtosis(row) for row in boot_matrix])
boot_mins  = np.min(boot_matrix, axis=1)
boot_maxs  = np.max(boot_matrix, axis=1)

# Real data statistics
real_mean = np.mean(speed)
real_std  = np.std(speed, ddof=1)
real_skew = skew(speed)
real_kurt = kurtosis(speed)
real_min  = np.min(speed)
real_max  = np.max(speed)

alpha = 1.0 - CONFIDENCE
ci_lo = alpha / 2.0 * 100
ci_hi = (1.0 - alpha / 2.0) * 100


def ci_str(real_val, boot_vals):
    lo, hi = np.percentile(boot_vals, [ci_lo, ci_hi])
    inside = lo <= real_val <= hi
    return (f"Real={real_val:>10.6f}   Boot={np.mean(boot_vals):>10.6f}  "
            f"95%CI=[{lo:.6f}, {hi:.6f}]  "
            f"{'✓ INSIDE' if inside else '✗ OUTSIDE'}")


print(f"\n  Statistic           Real value   Boot mean     95% CI              Status")
print(f"  {'─'*78}")
print(f"  Mean      : {ci_str(real_mean, boot_means)}")
print(f"  Std Dev   : {ci_str(real_std,  boot_stds)}")
print(f"  Skewness  : {ci_str(real_skew, boot_skews)}")
print(f"  Kurtosis  : {ci_str(real_kurt, boot_kurts)}")
print(f"  Min       : {ci_str(real_min,  boot_mins)}")
print(f"  Max       : {ci_str(real_max,  boot_maxs)}")

# ══════════════════════════════════════════════════════════════════════════
# 5. DISTRIBUTION COMPARISON  (KS test + Anderson-Darling)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 4: DISTRIBUTION COMPARISON")
print("─" * 72)

# KS test: real vs. pooled bootstrap
# Use a random subset of bootstrap for pooled comparison (avoid memory blow-up)
n_pool = min(50, N_BOOTSTRAP)
pool_idx = rng.choice(N_BOOTSTRAP, size=n_pool, replace=False)
boot_pool = boot_matrix[pool_idx].flatten()

ks_stat, ks_p = ks_2samp(speed, boot_pool)
print(f"  KS test (real vs pooled bootstrap):")
print(f"    D = {ks_stat:.6f},  p = {ks_p:.4e}")
print(f"    → {'SAME distribution (fail to reject H0)' if ks_p > 0.05 else 'DIFFERENT distribution (reject H0)'}")

# Per-replicate KS tests (subsample for speed when N_BOOTSTRAP is large)
n_ks_test = N_BOOTSTRAP
ks_idx = rng.choice(N_BOOTSTRAP, size=n_ks_test, replace=False)
ks_stats_per = np.array([ks_2samp(speed, boot_matrix[i])[0]
                         for i in ks_idx])
ks_p_per     = np.array([ks_2samp(speed, boot_matrix[i])[1]
                         for i in ks_idx])
pct_pass = 100.0 * np.mean(ks_p_per > 0.05)
print(f"\n  Per-replicate KS tests (H0: same distribution):")
print(f"    Mean D          : {np.mean(ks_stats_per):.6f}")
print(f"    Median p-value  : {np.median(ks_p_per):.4e}")
print(f"    % not rejected  : {pct_pass:.1f}%  (expect ~95% under H0)")

# ══════════════════════════════════════════════════════════════════════════
# 6. ACF COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 5: AUTOCORRELATION FUNCTION (ACF) COMPARISON")
print("─" * 72)

max_lags = min(60, N // 2 - 1)
acf_real = acf(speed, nlags=max_lags, fft=True)

# Compute ACF for each bootstrap replicate
acf_boot_all = np.array([acf(boot_matrix[i], nlags=max_lags, fft=True)
                         for i in range(N_BOOTSTRAP)])
acf_boot_mean = np.mean(acf_boot_all, axis=0)
acf_boot_lo   = np.percentile(acf_boot_all, ci_lo, axis=0)
acf_boot_hi   = np.percentile(acf_boot_all, ci_hi, axis=0)

# Check how well bootstrap ACF covers the real ACF
lags_arr = np.arange(max_lags + 1)
n_inside_acf = np.sum((acf_real >= acf_boot_lo) & (acf_real <= acf_boot_hi))
pct_inside_acf = 100.0 * n_inside_acf / (max_lags + 1)

print(f"  ACF lags analysed   : 0 – {max_lags}")
print(f"  Real ACF inside CI  : {n_inside_acf}/{max_lags+1} ({pct_inside_acf:.1f}%)")

# Lag-1 autocorrelation comparison
real_acf1 = acf_real[1]
boot_acf1_vals = acf_boot_all[:, 1]
lo1, hi1 = np.percentile(boot_acf1_vals, [ci_lo, ci_hi])
print(f"  Lag-1 ACF  real     : {real_acf1:.4f}")
print(f"  Lag-1 ACF  boot CI  : [{lo1:.4f}, {hi1:.4f}]"
      f"  {'✓' if lo1 <= real_acf1 <= hi1 else '✗'}")

# ══════════════════════════════════════════════════════════════════════════
# 7. PSD COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 6: POWER SPECTRAL DENSITY (PSD) COMPARISON")
print("─" * 72)

nperseg = min(256, N // 2)
freqs_real, psd_real = welch(speed, fs=FS, nperseg=nperseg, window='hann',
                             scaling='density')
# Remove DC
freqs_real = freqs_real[1:]
psd_real   = psd_real[1:]

# Compute PSD for bootstrap replicates (use a representative subset for speed)
n_psd_boot = min(200, N_BOOTSTRAP)
psd_boot_all = np.empty((n_psd_boot, len(freqs_real)))
for i in range(n_psd_boot):
    _, psd_b = welch(boot_matrix[i], fs=FS, nperseg=nperseg, window='hann',
                     scaling='density')
    psd_boot_all[i] = psd_b[1:]

psd_boot_mean = np.mean(psd_boot_all, axis=0)
psd_boot_lo   = np.percentile(psd_boot_all, ci_lo, axis=0)
psd_boot_hi   = np.percentile(psd_boot_all, ci_hi, axis=0)

n_inside_psd = np.sum((psd_real >= psd_boot_lo) & (psd_real <= psd_boot_hi))
pct_inside_psd = 100.0 * n_inside_psd / len(freqs_real)

# Spectral slope comparison
log_f = np.log10(freqs_real)
log_psd_real = np.log10(psd_real)

def lin(x, a, b):
    return a * x + b

popt_real, _ = curve_fit(lin, log_f, log_psd_real)
alpha_real = popt_real[0]

boot_alphas = []
for i in range(n_psd_boot):
    popt_b, _ = curve_fit(lin, log_f, np.log10(psd_boot_all[i]))
    boot_alphas.append(popt_b[0])
boot_alphas = np.array(boot_alphas)

alpha_boot_mean = np.mean(boot_alphas)
alpha_boot_lo, alpha_boot_hi = np.percentile(boot_alphas, [ci_lo, ci_hi])

print(f"  Real PSD inside CI  : {n_inside_psd}/{len(freqs_real)} ({pct_inside_psd:.1f}%)")
print(f"  Real PSD slope α    : {alpha_real:.3f}")
print(f"  Boot PSD slope α    : {alpha_boot_mean:.3f}  CI=[{alpha_boot_lo:.3f}, {alpha_boot_hi:.3f}]"
      f"  {'✓' if alpha_boot_lo <= alpha_real <= alpha_boot_hi else '✗'}")

# ══════════════════════════════════════════════════════════════════════════
# 8. ALLAN DEVIATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════
if HAS_ALLANTOOLS:
    print("\n" + "─" * 72)
    print("  STEP 7: ALLAN DEVIATION (OADEV) COMPARISON")
    print("─" * 72)

    taus_real, ad_real, _, _ = allantools.oadev(speed, rate=FS, data_type='freq')

    n_ad_boot = min(200, N_BOOTSTRAP)
    ad_boot_all = []
    for i in range(n_ad_boot):
        taus_b, ad_b, _, _ = allantools.oadev(boot_matrix[i], rate=FS,
                                               data_type='freq')
        # Interpolate onto real taus for comparison
        ad_interp = np.interp(taus_real, taus_b, ad_b)
        ad_boot_all.append(ad_interp)

    ad_boot_all  = np.array(ad_boot_all)
    ad_boot_mean = np.mean(ad_boot_all, axis=0)
    ad_boot_lo   = np.percentile(ad_boot_all, ci_lo, axis=0)
    ad_boot_hi   = np.percentile(ad_boot_all, ci_hi, axis=0)

    n_inside_ad = np.sum((ad_real >= ad_boot_lo) & (ad_real <= ad_boot_hi))
    pct_inside_ad = 100.0 * n_inside_ad / len(taus_real)

    # Bias instability comparison
    bi_real = np.min(ad_real)
    bi_boot_vals = np.min(ad_boot_all, axis=1)
    bi_lo, bi_hi = np.percentile(bi_boot_vals, [ci_lo, ci_hi])

    print(f"  Real OADEV inside CI    : {n_inside_ad}/{len(taus_real)} ({pct_inside_ad:.1f}%)")
    print(f"  Bias Instab (real)      : {bi_real:.6f} m/s")
    print(f"  Bias Instab (boot) CI   : [{bi_lo:.6f}, {bi_hi:.6f}]"
          f"  {'✓' if bi_lo <= bi_real <= bi_hi else '✗'}")

# ══════════════════════════════════════════════════════════════════════════
# 9. STATIONARITY TESTS ON BOOTSTRAP REPLICATES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 8: STATIONARITY TESTS ON BOOTSTRAP REPLICATES")
print("─" * 72)

n_stat_test = min(200, N_BOOTSTRAP)
adf_pass = 0
kpss_pass = 0

for i in range(n_stat_test):
    # ADF (H0: unit root)
    _, adf_p, *_ = adfuller(boot_matrix[i], autolag='AIC')
    if adf_p < 0.05:
        adf_pass += 1
    # KPSS (H0: stationary)
    _, kpss_p_val, *_ = kpss(boot_matrix[i], regression='c', nlags='auto')
    if kpss_p_val > 0.05:
        kpss_pass += 1

pct_adf  = 100.0 * adf_pass / n_stat_test
pct_kpss = 100.0 * kpss_pass / n_stat_test

print(f"  Tested {n_stat_test} replicates:")
print(f"    ADF  → stationary  : {adf_pass}/{n_stat_test} ({pct_adf:.1f}%)")
print(f"    KPSS → stationary  : {kpss_pass}/{n_stat_test} ({pct_kpss:.1f}%)")
print(f"    Both → stationary  : expectation > 90% for valid bootstrap")

# ══════════════════════════════════════════════════════════════════════════
# 10. PLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 9: GENERATING PLOTS")
print("─" * 72)

# ── 10a. Summary statistics bootstrap distributions ───────────────────────
fig_stats, axes = plt.subplots(2, 3, figsize=(18, 10))
stat_data = [
    (boot_means, real_mean, 'Mean (m/s)'),
    (boot_stds,  real_std,  'Std Dev (m/s)'),
    (boot_skews, real_skew, 'Skewness'),
    (boot_kurts, real_kurt, 'Excess Kurtosis'),
    (boot_mins,  real_min,  'Minimum (m/s)'),
    (boot_maxs,  real_max,  'Maximum (m/s)'),
]

for ax, (bvals, rval, label) in zip(axes.flat, stat_data):
    ax.hist(bvals, bins=40, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label='Bootstrap')
    ax.axvline(rval, color='red', lw=2, ls='--', label=f'Real = {rval:.5f}')
    lo, hi = np.percentile(bvals, [ci_lo, ci_hi])
    ax.axvspan(lo, hi, alpha=0.15, color='orange', label='95% CI')
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# fig_stats.suptitle(f"Stationary Bootstrap ({N_BOOTSTRAP} replicates, L={L_opt}) – "
#                    "Summary Statistics", fontsize=14, fontweight='bold')
fig_stats.tight_layout(rect=[0, 0, 1, 0.96])
fig_stats.savefig(OUTPUT_FOLDER / "01_bootstrap_statistics.png", dpi=200,
                  bbox_inches='tight')
fig_stats.savefig(OUTPUT_FOLDER / "01_bootstrap_statistics.pdf", dpi=200,
                  bbox_inches='tight')
print("  → 01_bootstrap_statistics saved.")

# ── 10b. Distribution overlay (histogram) ─────────────────────────────────
fig_dist, ax_d = plt.subplots(figsize=(12, 6))
ax_d.hist(speed, bins=50, density=True, alpha=0.6, color='steelblue',
          edgecolor='white', label='Real data')
# Overlay 5 random bootstrap replicates
for i in range(5):
    idx = rng.integers(0, N_BOOTSTRAP)
    ax_d.hist(boot_matrix[idx], bins=50, density=True, alpha=0.15,
              color='orange', edgecolor='none')
ax_d.hist(boot_matrix[0], bins=50, density=True, alpha=0.15,
          color='orange', edgecolor='none', label='Boot replicates (5 shown)')
ax_d.set_xlabel("Speed (m/s)", fontsize=12)
ax_d.set_ylabel("Probability Density", fontsize=12)
ax_d.set_title("Distribution Comparison: Real vs Bootstrap", fontsize=13,
               fontweight='bold')
ax_d.legend(fontsize=10)
ax_d.grid(alpha=0.3)
fig_dist.tight_layout()
fig_dist.savefig(OUTPUT_FOLDER / "02_distribution_overlay.png", dpi=200,
                 bbox_inches='tight')
fig_dist.savefig(OUTPUT_FOLDER / "02_distribution_overlay.pdf", dpi=200,
                 bbox_inches='tight')
print("  → 02_distribution_overlay saved.")

# ── 10c. Time-series examples ─────────────────────────────────────────────
fig_ts, axes_ts = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=True)
axes_ts[0].plot(time_s, speed, lw=0.6, color='steelblue')
axes_ts[0].set_title("Real Data", fontsize=12, fontweight='bold')
axes_ts[0].set_ylabel("Speed (m/s)")
axes_ts[0].grid(alpha=0.3)

for i, ax in enumerate(axes_ts[1:], 1):
    idx = rng.integers(0, N_BOOTSTRAP)
    ax.plot(time_s, boot_matrix[idx], lw=0.6, color='darkorange')
    ax.set_title(f"Bootstrap Replicate #{idx}", fontsize=12)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(alpha=0.3)

axes_ts[-1].set_xlabel("Time (s)", fontsize=12)
fig_ts.suptitle("Time Series Comparison: Real vs Bootstrap Replicates",
                fontsize=14, fontweight='bold')
fig_ts.tight_layout(rect=[0, 0, 1, 0.96])
fig_ts.savefig(OUTPUT_FOLDER / "03_timeseries_comparison.png", dpi=200,
               bbox_inches='tight')
fig_ts.savefig(OUTPUT_FOLDER / "03_timeseries_comparison.pdf", dpi=200,
               bbox_inches='tight')
print("  → 03_timeseries_comparison saved.")

# ── 10d. ACF comparison ───────────────────────────────────────────────────
fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
ax_acf.fill_between(lags_arr, acf_boot_lo, acf_boot_hi, alpha=0.3,
                    color='orange', label='Bootstrap 95% CI')
ax_acf.plot(lags_arr, acf_boot_mean, 'o-', color='darkorange', markersize=2,
            lw=1.5, label='Bootstrap mean ACF')
ax_acf.plot(lags_arr, acf_real, 's-', color='steelblue', markersize=3,
            lw=1.5, label='Real ACF')
ci_wn = 1.96 / np.sqrt(N)
ax_acf.axhline(ci_wn, color='gray', ls=':', lw=1, alpha=0.5)
ax_acf.axhline(-ci_wn, color='gray', ls=':', lw=1, alpha=0.5,
               label=f'White noise CI (±{ci_wn:.4f})')
ax_acf.set_xlabel("Lag (samples, Δt = 0.2 s)", fontsize=12)
ax_acf.set_ylabel("ACF", fontsize=12)
ax_acf.set_title(f"ACF Comparison – Real vs Bootstrap  "
                 f"({pct_inside_acf:.0f}% of real ACF inside boot CI)",
                 fontsize=13, fontweight='bold')
ax_acf.legend(fontsize=10)
ax_acf.grid(alpha=0.3)
fig_acf.tight_layout()
fig_acf.savefig(OUTPUT_FOLDER / "04_acf_comparison.png", dpi=200,
                bbox_inches='tight')
fig_acf.savefig(OUTPUT_FOLDER / "04_acf_comparison.pdf", dpi=200,
                bbox_inches='tight')
print("  → 04_acf_comparison saved.")

# ── 10e. PSD comparison ──────────────────────────────────────────────────
fig_psd, ax_psd = plt.subplots(figsize=(12, 7))
ax_psd.fill_between(freqs_real, psd_boot_lo, psd_boot_hi, alpha=0.3,
                    color='orange', label='Bootstrap 95% CI')
ax_psd.loglog(freqs_real, psd_boot_mean, '-', color='darkorange', lw=1.5,
              label='Bootstrap mean PSD')
ax_psd.loglog(freqs_real, psd_real, '-', color='steelblue', lw=1.5,
              label='Real PSD')
ax_psd.set_xlabel("Frequency (Hz)", fontsize=12)
ax_psd.set_ylabel("PSD  ((m/s)²/Hz)", fontsize=12)
ax_psd.set_title(f"PSD Comparison – Real (α={alpha_real:.2f}) vs Bootstrap "
                 f"(α={alpha_boot_mean:.2f})  –  "
                 f"{pct_inside_psd:.0f}% inside CI",
                 fontsize=13, fontweight='bold')
ax_psd.legend(fontsize=10)
ax_psd.grid(True, which='both', alpha=0.3)
fig_psd.tight_layout()
fig_psd.savefig(OUTPUT_FOLDER / "05_psd_comparison.png", dpi=200,
                bbox_inches='tight')
fig_psd.savefig(OUTPUT_FOLDER / "05_psd_comparison.pdf", dpi=200,
                bbox_inches='tight')
print("  → 05_psd_comparison saved.")

# ── 10f. Allan Deviation comparison ──────────────────────────────────────
if HAS_ALLANTOOLS:
    fig_ad, ax_ad = plt.subplots(figsize=(12, 7))
    ax_ad.fill_between(taus_real, ad_boot_lo, ad_boot_hi, alpha=0.3,
                       color='orange', label='Bootstrap 95% CI')
    ax_ad.loglog(taus_real, ad_boot_mean, '-', color='darkorange', lw=1.5,
                 label='Bootstrap mean OADEV')
    ax_ad.loglog(taus_real, ad_real, '-', color='steelblue', lw=1.5,
                 label='Real OADEV')
    ax_ad.set_xlabel("Averaging Time τ (s)", fontsize=12)
    ax_ad.set_ylabel("Allan Deviation (m/s)", fontsize=12)
    ax_ad.set_title(f"Allan Deviation Comparison – "
                    f"{pct_inside_ad:.0f}% inside CI",
                    fontsize=13, fontweight='bold')
    ax_ad.legend(fontsize=10)
    ax_ad.grid(True, which='both', alpha=0.3)
    fig_ad.tight_layout()
    fig_ad.savefig(OUTPUT_FOLDER / "06_allan_deviation_comparison.png",
                   dpi=200, bbox_inches='tight')
    fig_ad.savefig(OUTPUT_FOLDER / "06_allan_deviation_comparison.pdf",
                   dpi=200, bbox_inches='tight')
    print("  → 06_allan_deviation_comparison saved.")

# ── 10g. KS test p-value distribution ────────────────────────────────────
fig_ks, ax_ks = plt.subplots(figsize=(10, 5))
ax_ks.hist(ks_p_per, bins=50, density=True, alpha=0.7, color='steelblue',
           edgecolor='white')
ax_ks.axvline(0.05, color='red', ls='--', lw=2, label='α = 0.05 threshold')
ax_ks.set_xlabel("KS Test p-value", fontsize=12)
ax_ks.set_ylabel("Density", fontsize=12)
ax_ks.set_title(f"KS Test p-values (Real vs Each Replicate) – "
                f"{pct_pass:.1f}% pass at α=0.05",
                fontsize=13, fontweight='bold')
ax_ks.legend(fontsize=10)
ax_ks.grid(alpha=0.3)
fig_ks.tight_layout()
fig_ks.savefig(OUTPUT_FOLDER / "07_ks_pvalue_distribution.png", dpi=200,
               bbox_inches='tight')
fig_ks.savefig(OUTPUT_FOLDER / "07_ks_pvalue_distribution.pdf", dpi=200,
               bbox_inches='tight')
print("  → 07_ks_pvalue_distribution saved.")

# ── 10h. Comprehensive dashboard ─────────────────────────────────────────
fig_dash, axes_d = plt.subplots(2, 2, figsize=(16, 10))

# ACF
axes_d[0, 0].fill_between(lags_arr, acf_boot_lo, acf_boot_hi, alpha=0.3,
                           color='orange')
axes_d[0, 0].plot(lags_arr, acf_real, 's-', color='steelblue', markersize=2,
                  lw=1.5, label='Real')
axes_d[0, 0].plot(lags_arr, acf_boot_mean, '-', color='darkorange', lw=1.5,
                  label='Boot mean')
axes_d[0, 0].set_title(f"ACF ({pct_inside_acf:.0f}% inside CI)",
                       fontweight='bold')
axes_d[0, 0].set_xlabel("Lag")
axes_d[0, 0].legend(fontsize=8)
axes_d[0, 0].grid(alpha=0.3)

# PSD with fitted slope lines
axes_d[0, 1].fill_between(freqs_real, psd_boot_lo, psd_boot_hi, alpha=0.3,
                           color='orange')
axes_d[0, 1].loglog(freqs_real, psd_real, '-', color='steelblue', lw=1.5,
                    label='Real')
axes_d[0, 1].loglog(freqs_real, psd_boot_mean, '-', color='darkorange',
                    lw=1.5, label='Boot mean')
# Fit and plot slope lines
popt_real_dash, _ = curve_fit(lin, log_f, np.log10(psd_real))
popt_boot_dash, _ = curve_fit(lin, log_f, np.log10(psd_boot_mean))
axes_d[0, 1].loglog(freqs_real, 10**lin(log_f, *popt_real_dash), '--',
                    color='steelblue', lw=2, alpha=0.7,
                    label=f'Real fit (α={popt_real_dash[0]:.2f})')
axes_d[0, 1].loglog(freqs_real, 10**lin(log_f, *popt_boot_dash), '--',
                    color='darkorange', lw=2, alpha=0.7,
                    label=f'Boot fit (α={popt_boot_dash[0]:.2f})')
axes_d[0, 1].set_title("PSD", fontweight='bold')
axes_d[0, 1].set_xlabel("Frequency (Hz)")
axes_d[0, 1].set_ylabel("PSD ((m/s)²/Hz)")
axes_d[0, 1].legend(fontsize=8)
axes_d[0, 1].grid(True, which='both', alpha=0.3)

# Allan Deviation
if HAS_ALLANTOOLS:
    axes_d[1, 0].fill_between(taus_real, ad_boot_lo, ad_boot_hi, alpha=0.3,
                               color='orange')
    axes_d[1, 0].loglog(taus_real, ad_real, '-', color='steelblue', lw=1.5,
                        label='Real')
    axes_d[1, 0].loglog(taus_real, ad_boot_mean, '-', color='darkorange',
                        lw=1.5, label='Boot mean')
    axes_d[1, 0].set_title(f"OADEV ({pct_inside_ad:.0f}% inside CI)",
                           fontweight='bold')
    axes_d[1, 0].set_xlabel("τ (s)")
    axes_d[1, 0].set_ylabel("ADEV (m/s)")
    axes_d[1, 0].legend(fontsize=8)
    axes_d[1, 0].grid(True, which='both', alpha=0.3)
else:
    axes_d[1, 0].text(0.5, 0.5, 'allantools\nnot installed',
                      ha='center', va='center', fontsize=14,
                      transform=axes_d[1, 0].transAxes)
    axes_d[1, 0].set_title("OADEV (skipped)")

# KS p-value distribution
axes_d[1, 1].hist(ks_p_per, bins=50, density=True, alpha=0.7,
                  color='steelblue', edgecolor='white')
axes_d[1, 1].axvline(0.05, color='red', ls='--', lw=2,
                     label='α = 0.05 threshold')
axes_d[1, 1].set_title(f"KS p-values ({pct_pass:.1f}% pass at α=0.05)",
                       fontweight='bold')
axes_d[1, 1].set_xlabel("KS Test p-value")
axes_d[1, 1].set_ylabel("Density")
axes_d[1, 1].legend(fontsize=8)
axes_d[1, 1].grid(alpha=0.3)

# fig_dash.suptitle(f"Stationary Bootstrap Validation Dashboard  "
#                   f"(B={N_BOOTSTRAP}, L={L_opt})",
#                   fontsize=15, fontweight='bold')
fig_dash.tight_layout(rect=[0, 0, 1, 0.96])
fig_dash.savefig(OUTPUT_FOLDER / "08_dashboard.png", dpi=200,
                 bbox_inches='tight')
fig_dash.savefig(OUTPUT_FOLDER / "08_dashboard.pdf", dpi=200,
                 bbox_inches='tight')
print("  → 08_dashboard saved.")

plt.close('all')

# ══════════════════════════════════════════════════════════════════════════
# 11. EXPORT DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 10: EXPORTING DATA")
print("─" * 72)

# Export bootstrap summary statistics
stats_df = pd.DataFrame({
    'replicate': np.arange(N_BOOTSTRAP),
    'mean': boot_means,
    'std': boot_stds,
    'skewness': boot_skews,
    'kurtosis': boot_kurts,
    'min': boot_mins,
    'max': boot_maxs,
})
stats_df.to_csv(OUTPUT_FOLDER / "bootstrap_statistics.csv", index=False)

# Export ACF comparison
acf_df = pd.DataFrame({
    'lag': lags_arr,
    'acf_real': acf_real,
    'acf_boot_mean': acf_boot_mean,
    'acf_boot_2.5pct': acf_boot_lo,
    'acf_boot_97.5pct': acf_boot_hi,
})
acf_df.to_csv(OUTPUT_FOLDER / "acf_comparison.csv", index=False)

# Export PSD comparison
psd_df = pd.DataFrame({
    'frequency_Hz': freqs_real,
    'psd_real': psd_real,
    'psd_boot_mean': psd_boot_mean,
    'psd_boot_2.5pct': psd_boot_lo,
    'psd_boot_97.5pct': psd_boot_hi,
})
psd_df.to_csv(OUTPUT_FOLDER / "psd_comparison.csv", index=False)

# Export Allan Deviation comparison
if HAS_ALLANTOOLS:
    ad_df = pd.DataFrame({
        'tau_s': taus_real,
        'oadev_real': ad_real,
        'oadev_boot_mean': ad_boot_mean,
        'oadev_boot_2.5pct': ad_boot_lo,
        'oadev_boot_97.5pct': ad_boot_hi,
    })
    ad_df.to_csv(OUTPUT_FOLDER / "allan_deviation_comparison.csv", index=False)

# Export KS test results
ks_df = pd.DataFrame({
    'replicate': ks_idx,
    'ks_statistic': ks_stats_per,
    'ks_pvalue': ks_p_per,
})
ks_df.to_csv(OUTPUT_FOLDER / "ks_test_results.csv", index=False)

print("  → CSV data exported to:", OUTPUT_FOLDER)

# ══════════════════════════════════════════════════════════════════════════
# 12. COMPREHENSIVE TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STEP 11: WRITING REPORT")
print("─" * 72)

report = []
report.append("=" * 72)
report.append("  STATIONARY BOOTSTRAP VALIDATION REPORT")
report.append("  Politis & Romano (1994) – GPS Speed Noise")
report.append("=" * 72)
report.append(f"\n  Data file       : {CSV_FILE.name}")
report.append(f"  Samples         : {N}")
report.append(f"  Duration        : {time_s[-1]:.1f} s")
report.append(f"  Sampling rate   : {FS} Hz")
report.append(f"  Bootstrap reps  : {N_BOOTSTRAP}")
report.append(f"  Avg block length: L = {L_opt}  (p = {p_opt:.4f})")
report.append(f"  Random seed     : {RANDOM_SEED}")

report.append(f"\n{'─'*72}")
report.append("  SUMMARY STATISTICS")
report.append(f"{'─'*72}")
report.append(f"  {'Statistic':<14} {'Real':>12} {'Boot mean':>12} "
              f"{'95% CI lo':>12} {'95% CI hi':>12} {'Status':>10}")
for label, rval, bvals in [
    ('Mean',     real_mean, boot_means),
    ('Std Dev',  real_std,  boot_stds),
    ('Skewness', real_skew, boot_skews),
    ('Kurtosis', real_kurt, boot_kurts),
    ('Min',      real_min,  boot_mins),
    ('Max',      real_max,  boot_maxs),
]:
    lo, hi = np.percentile(bvals, [ci_lo, ci_hi])
    inside = "PASS" if lo <= rval <= hi else "FAIL"
    report.append(f"  {label:<14} {rval:>12.6f} {np.mean(bvals):>12.6f} "
                  f"{lo:>12.6f} {hi:>12.6f} {inside:>10}")

report.append(f"\n{'─'*72}")
report.append("  DISTRIBUTION TESTS")
report.append(f"{'─'*72}")
report.append(f"  KS test (pooled): D={ks_stat:.6f}, p={ks_p:.4e} "
              f"→ {'PASS' if ks_p > 0.05 else 'FAIL'}")
report.append(f"  Per-replicate KS pass rate: {pct_pass:.1f}%")

report.append(f"\n{'─'*72}")
report.append("  DEPENDENCE STRUCTURE")
report.append(f"{'─'*72}")
report.append(f"  ACF coverage (real inside boot CI): {pct_inside_acf:.1f}%")
report.append(f"  Lag-1 ACF  real={real_acf1:.4f}  boot CI=[{lo1:.4f}, {hi1:.4f}]")

report.append(f"\n{'─'*72}")
report.append("  SPECTRAL ANALYSIS")
report.append(f"{'─'*72}")
report.append(f"  PSD coverage: {pct_inside_psd:.1f}%")
report.append(f"  PSD slope real={alpha_real:.3f}  "
              f"boot={alpha_boot_mean:.3f} CI=[{alpha_boot_lo:.3f}, {alpha_boot_hi:.3f}]")

if HAS_ALLANTOOLS:
    report.append(f"\n{'─'*72}")
    report.append("  ALLAN DEVIATION")
    report.append(f"{'─'*72}")
    report.append(f"  OADEV coverage: {pct_inside_ad:.1f}%")
    report.append(f"  Bias Instab real={bi_real:.6f}  "
                  f"boot CI=[{bi_lo:.6f}, {bi_hi:.6f}]")

report.append(f"\n{'─'*72}")
report.append("  STATIONARITY OF REAL DATA")
report.append(f"{'─'*72}")
report.append(f"  ADF  test: stat={adf_real_stat:.4f}, p={adf_real_p:.4e} "
              f"→ {'STATIONARY' if adf_real_p < 0.05 else 'NON-STATIONARY'}")
report.append(f"  KPSS test: stat={kpss_real_stat:.4f}, p={kpss_real_p:.4f} "
              f"→ {'STATIONARY' if kpss_real_p > 0.05 else 'NON-STATIONARY'}")
report.append(f"  Verdict  : {real_stationarity_verdict}")

report.append(f"\n{'─'*72}")
report.append("  STATIONARITY OF BOOTSTRAP REPLICATES")
report.append(f"{'─'*72}")
report.append(f"  ADF  stationary: {pct_adf:.1f}%")
report.append(f"  KPSS stationary: {pct_kpss:.1f}%")

# Overall verdict
report.append(f"\n{'='*72}")
report.append("  OVERALL RELIABILITY VERDICT")
report.append(f"{'='*72}")

checks = []
checks.append(('Real data ADF stationary',  adf_real_p < 0.05))
checks.append(('Real data KPSS stationary', kpss_real_p > 0.05))
checks.append(('Mean inside 95% CI',
               np.percentile(boot_means, ci_lo) <= real_mean <= np.percentile(boot_means, ci_hi)))
checks.append(('Std Dev inside 95% CI',
               np.percentile(boot_stds, ci_lo) <= real_std <= np.percentile(boot_stds, ci_hi)))
checks.append(('KS test (pooled) pass', ks_p > 0.05))
checks.append(('Per-replicate KS ≥ 80%', pct_pass >= 80))
checks.append(('ACF coverage ≥ 80%', pct_inside_acf >= 80))
checks.append(('PSD slope inside CI',
               alpha_boot_lo <= alpha_real <= alpha_boot_hi))
if HAS_ALLANTOOLS:
    checks.append(('OADEV coverage ≥ 70%', pct_inside_ad >= 70))
checks.append(('Boot ADF stationary ≥ 90%', pct_adf >= 90))
checks.append(('Boot KPSS stationary ≥ 90%', pct_kpss >= 90))

n_pass = sum(v for _, v in checks)
n_total = len(checks)

for label, passed in checks:
    report.append(f"  {'✓' if passed else '✗'} {label}")

report.append(f"\n  Score: {n_pass}/{n_total} checks passed")

if n_pass == n_total:
    verdict = "EXCELLENT – Bootstrap simulation is fully reliable."
elif n_pass >= n_total - 1:
    verdict = "GOOD – Bootstrap simulation is reliable (minor deviation in one metric)."
elif n_pass >= n_total - 2:
    verdict = "ACCEPTABLE – Bootstrap simulation is mostly reliable."
else:
    verdict = "POOR – Bootstrap simulation shows significant discrepancies."

report.append(f"  → {verdict}")
report.append(f"\n{'='*72}")

report_text = "\n".join(report)

# Print to console
print(report_text)

# Save to file
report_path = OUTPUT_FOLDER / "stationary_bootstrap_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\n  → Report saved to: {report_path}")

print("\n" + "=" * 72)
print("  STATIONARY BOOTSTRAP ANALYSIS COMPLETE")
print("=" * 72)
