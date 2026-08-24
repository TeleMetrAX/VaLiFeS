import numpy as np
from vmdpy import VMD

# VMD default parameters
# bandwidth constraint (the higher, the smaller bandwidth)
ALPHA = 100
# number of modes
K = 12
# noise-tolerance (Lagrangian multiplier)
TAU = 0.0
# DC part: fix w0 = 0
DC = False
# initialize omegas
INIT = 1
TOL = 1e-7

# VMD parameters
VMD_SUM_FROM = 0


# Run VMD
def decompose(values, alpha=ALPHA, k=K, tau=TAU, dc=DC, init=INIT, tol=TOL, vmd_sum_from=VMD_SUM_FROM):
    values = np.array(values)
    u, u_hat, omega = VMD(values, alpha, tau, k, dc, init, tol)
    usum = sum(u[vmd_sum_from:])

    if values.shape[0] - usum.shape[0] > 0:
        usum = np.append(usum, usum[-1] * (values.shape[0] - usum.shape[0]))

    data = values.reshape(values.shape[0]) - usum

    time = [t for t in range(len(data))]

    data = np.transpose(np.array([time, data]))
    usum = np.transpose(np.array([time, usum]))

    return data, omega, usum


def decompose_orig(values, alpha, k, tau, dc, init, tol):
    return VMD(values, alpha, tau, k, dc, init, tol)


def get_highest_energy_mode_omega(curr_omegas: np.ndarray, u: np.ndarray):
    mode_energies = np.zeros(u.shape[0])
    for k, mode in enumerate(u):
        mode_energies[k] = np.sum(np.square(np.abs(mode)))
    max_mode_index = np.argmax(mode_energies)
    return curr_omegas[max_mode_index]
