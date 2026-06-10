import argparse
from warnings import simplefilter

import numpy as np
import scipy.stats as sstat
import sklearn.gaussian_process as GP
from sklearn.exceptions import ConvergenceWarning


# ----------------------------
# Helpers for normalization
# ----------------------------
def uni_to_raw(x_d, bb):
    return bb[0, :] + x_d * (bb[1, :] - bb[0, :])


def raw_to_uni(raw_d, bb):
    return (raw_d - bb[0, :] ) / (bb[1, :] - bb[0, :])


def all_u2r(x_nd, bb):
    return np.asarray([uni_to_raw(x_nd[n, :], bb) for n in range(x_nd.shape[0])])


def all_r2u(raw_config_nd, bb):
    return np.asarray([raw_to_uni(raw_config_nd[n, :], bb) for n in range(raw_config_nd.shape[0])])


# ----------------------------
# Uniform grid with random jitter
# ----------------------------
def grid_uniform(space_d):
    grid_d = list(np.meshgrid(*space_d, indexing="ij"))
    for d in range(len(grid_d)):
        diff = np.diff(sorted(list(set(grid_d[d].ravel()))))
        if diff.size == 0:
            continue
        interval = np.mean(diff)
        if np.isfinite(interval) and interval > 0:
            grid_d[d] += np.random.uniform(high=interval, size=grid_d[d].shape)
    return np.c_[[g.ravel() for g in grid_d]].T


def fixed_grid_uniform(num_d, l_fix=None, low=0.0, high=1.0):
    if l_fix is None:
        l_fix = []
    space_d = [np.linspace(low, high, num + 1)[:num] for num in num_d]
    for d, val in l_fix:
        space_d[d] = np.asarray([val])
    return grid_uniform(space_d)


# ----------------------------
# Data loading
# ----------------------------
def load_csv_numeric(csv_file):
    """Load a CSV file with one target column followed by design variables.

    The function supports both headered CSV files, such as
    Urbach,Temp,RF,Ar,O2, and headerless numeric CSV files.
    Missing target values can be written as nan.
    """
    try:
        data = np.genfromtxt(csv_file, delimiter=",", comments="#", names=True, dtype=float)
        if data.dtype.names is None:
            raise ValueError("No header detected")
        arr = np.column_stack([data[name] for name in data.dtype.names])
    except Exception:
        arr = np.genfromtxt(csv_file, delimiter=",", comments="#", dtype=float)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    if arr.shape[1] < 2:
        raise ValueError("The CSV must contain one target column and at least one design variable.")

    return arr


# ----------------------------
# Minimal random-mean-prior BO
# ----------------------------
class RndMeanPriorBO:
    def __init__(self, kernel, alpha=1e-2):
        self.init_kernel_ = kernel
        self.alpha_ = alpha
        self.gp_ = None
        self.last_prior_ = None

    def fit(self, X_nd, Y_n):
        Y = Y_n.copy()
        finite = np.isfinite(Y)
        floor = np.nanmin(Y) if finite.any() else 0.0
        Y[np.isnan(Y)] = floor
        self.gp_ = GP.GaussianProcessRegressor(
            kernel=self.init_kernel_,
            n_restarts_optimizer=1,
            normalize_y=False,
            alpha=self.alpha_,
        )
        self.gp_.fit(X_nd, Y)
        return self

    def _pred_prior(self, X_nd, num_trials=1):
        m_ni = np.zeros((X_nd.shape[0], num_trials))
        s_ni = np.zeros_like(m_ni)
        u_i = np.random.uniform(size=num_trials)
        ceil = np.nanmax(self.gp_.y_train_)
        floor = np.nanmin(self.gp_.y_train_)
        prior_i = floor + u_i * (ceil - floor)

        for i, prior in enumerate(prior_i):
            gp = GP.GaussianProcessRegressor(
                kernel=self.gp_.kernel_,
                optimizer=None,
                normalize_y=False,
                alpha=self.alpha_,
            )
            gp.fit(self.gp_.X_train_, self.gp_.y_train_ - prior)
            m, s = gp.predict(X_nd, return_std=True)
            m_ni[:, i] = m + prior
            s_ni[:, i] = s
        return np.squeeze(m_ni), np.squeeze(s_ni), prior_i

    def acquisition(self, X_nd, return_ms=False, num_trials=1):
        if self.gp_ is None:
            rand = np.random.uniform(size=(X_nd.shape[0],))
            return (rand, None, None) if return_ms else rand

        m_ni, s_ni, prior_i = self._pred_prior(X_nd, num_trials=num_trials)
        self.last_prior_ = float(np.mean(prior_i))

        # Expected improvement for maximization of the internally signed target.
        diff_best_ni = m_ni - self.gp_.y_train_.max()
        z_ni = np.divide(diff_best_ni, s_ni, out=np.zeros_like(diff_best_ni), where=s_ni > 0)
        cdf_ni = sstat.norm.cdf(z_ni)
        pdf_ni = sstat.norm.pdf(z_ni)
        ei_ni = np.fmax(0, diff_best_ni * cdf_ni + s_ni * pdf_ni)

        if return_ms:
            return ei_ni, m_ni, s_ni
        return ei_ni


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal single-file Bayesian optimization script for the paper dataset. "
            "Expected improvement is used."
        )
    )
    parser.add_argument("csv_file", help="CSV file: first column=y, remaining columns=x")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=25,
        help=(
            "Number of grid divisions per dimension for candidate search. "
            "The total number of candidates is grid_size^4. Default: 25."
        ),
    )
    parser.add_argument("--top", type=int, default=10, help="Number of candidate points to print")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    all_csv = load_csv_numeric(args.csv_file)

    rawy_n = all_csv[:, 0]
    raw_config_nd = all_csv[:, 1:]
    D = raw_config_nd.shape[1]

    if D != 4:
        raise ValueError(f"This script expects 4 design variables: T, PRF, FAr, FO2; found D={D}.")

    # Search ranges used in the manuscript BO campaign:
    # T in [30, 700] degC, PRF in [50, 150] W,
    # FAr in [10, 50] sccm, FO2 in [0, 10] sccm.
    xbb = np.array([
        [30.0, 50.0, 10.0, 0.0],
        [700.0, 150.0, 50.0, 10.0],
    ])

    finite_y = rawy_n[np.isfinite(rawy_n)]
    if finite_y.size == 0:
        raise ValueError("No finite y values found in the CSV.")

    # Automatic replacement for: --yrange min(data):max(data)
    ybb = np.array([[finite_y.min()], [finite_y.max()]])

    # Equivalent to minimization of the original target.
    sign = -1.0

    X_nd = all_r2u(raw_config_nd, xbb)
    Y_n = raw_to_uni(rawy_n, ybb) * sign

    ker_signal = GP.kernels.Product(
        GP.kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e2)),
        GP.kernels.Matern(
            length_scale=np.ones(D),
            length_scale_bounds=(1e-1, 10.0),
            nu=2.5,
        ),
    )
    ker = GP.kernels.Sum(ker_signal, GP.kernels.WhiteKernel(1e-2, noise_level_bounds=(1e-3, 1e-1)))

    simplefilter("ignore", category=ConvergenceWarning)
    bo = RndMeanPriorBO(ker, alpha=1e-2)
    bo.fit(X_nd, Y_n)

    grid_X_nd = fixed_grid_uniform(np.asarray((args.grid_size,) * D), l_fix=[])
    acq_n, m_n, s_n = bo.acquisition(grid_X_nd, return_ms=True, num_trials=1)

    idx_n = np.argsort(acq_n)[::-1]

    if bo.last_prior_ is not None:
        prior_raw = uni_to_raw(np.array([bo.last_prior_ * sign]), ybb)[0]
        print(f"prior = {prior_raw:1.4g}")

    print(f"y-range (auto) = {ybb[0,0]:.6g}:{ybb[1,0]:.6g}")
    print("search range = T[30,700], PRF[50,150], FAr[10,50], FO2[0,10]")
    print(f"grid size = {args.grid_size} per dimension ({args.grid_size ** D} candidates)")
    print("---- one-shot grid search results ----")
    print("acq".ljust(11) + "mean+/-std".ljust(16) + "input value (T, PRF, FAr, FO2)")

    n_show = min(args.top, len(idx_n))
    for j in range(n_show):
        idx = idx_n[j]
        acq_val = acq_n[idx] * (ybb[1, 0] - ybb[0, 0])
        mean_val = m_n[idx] * sign * (ybb[1, 0] - ybb[0, 0]) + ybb[0, 0]
        std_val = s_n[idx] * (ybb[1, 0] - ybb[0, 0])
        x_raw = uni_to_raw(grid_X_nd[idx, :], xbb)
        x_str = "(" + ", ".join(f"{v:.4g}" for v in x_raw) + ")"
        print(f"{acq_val:.4g}".ljust(11) + f"{mean_val:.4g}+/-{std_val:.4g}".ljust(16) + x_str)


if __name__ == "__main__":
    main()
