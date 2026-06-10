\
"""
Interpretable_analysis_fixed.py

Purpose
-------
Predict Urbach energy (Eu) from process parameters (Temp, RF, Ar, O2) and provide
interpretable analyses:
- Model comparison (Linear / Quadratic / Random Forest)
- Partial Dependence Plots (1D and 2D)
- Additive approximation quality (RF vs additive PDP model)
- Pairwise interaction strength (Friedman H^2)
- RF decomposition: additive (1D) and additive + Temp–O2 interaction (2D residual)

Key change vs. the original:
- This version NEVER blocks on plt.show() by default.
  It saves figures to a folder and continues, so Step 5 prints always appear.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --- Matplotlib: default to non-blocking, headless-friendly behavior ---
import matplotlib
HEADLESS = True  # Set False if you want interactive windows
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import partial_dependence

def _pd_grid_and_avg(pd_result):
    """Compatibility helper for sklearn.inspection.partial_dependence.

    Depending on scikit-learn version, the returned object may expose the grid as
    'grid_values' or 'values'. This function returns (grid_list, avg_list) where:
      - grid_list is a list of 1D arrays (one per feature dimension)
      - avg_list is a list/array where avg_list[0] is the average PDP array
    """
    # dict-like
    if isinstance(pd_result, dict):
        grid = pd_result.get("grid_values", None)
        if grid is None:
            grid = pd_result.get("values", None)
        avg = pd_result.get("average", None)
        if grid is None or avg is None:
            raise KeyError(f"partial_dependence result keys: {list(pd_result.keys())}")
        return grid, avg

    # attribute-like (Bunch)
    grid = getattr(pd_result, "grid_values", None)
    if grid is None:
        grid = getattr(pd_result, "values", None)
    avg = getattr(pd_result, "average", None)
    if grid is None or avg is None:
        raise KeyError("partial_dependence result has neither grid_values nor values / average")
    return grid, avg



# =========
# Settings
# =========
CSV_PATH = "BOdata_growth.csv"  # Put the CSV next to this script, or change the path.

TARGET_COL = "Urbach"
FEATURE_COLS = ["Temp", "RF", "Ar", "O2"]  # fixed order (important)
RANDOM_STATE = 0

# Where to save figures
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Parameter grid for RF tuning (can be heavy).
RF_PARAM_GRID = {
    "n_estimators": [150, 300],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.8],
}

# Fast/demo grid (much lighter)
RF_PARAM_GRID_FAST = {
    "n_estimators": [80, 160],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}


def save_fig(name: str) -> None:
    """Save current figure to FIG_DIR and close it (non-blocking)."""
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[figure saved] {path}")


# ============================
# 0) Load data & preprocessing
# ============================
def load_data(csv_path: str = CSV_PATH):
    """Load CSV; fill NaN in Urbach with its maximum value; return df, X, y."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    max_urbach = df[TARGET_COL].max(skipna=True)
    df_imputed = df.copy()
    df_imputed[TARGET_COL] = df_imputed[TARGET_COL].fillna(max_urbach)

    X = df_imputed[FEATURE_COLS].to_numpy()
    y = df_imputed[TARGET_COL].to_numpy()
    return df_imputed, X, y


# ============================
# 1) RF tuning with K-fold CV
# ============================
def tune_rf_with_cv(X, y, param_grid, n_splits: int = 5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for n_est in param_grid["n_estimators"]:
        for md in param_grid["max_depth"]:
            for leaf in param_grid["min_samples_leaf"]:
                for mf in param_grid["max_features"]:
                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=md,
                        min_samples_leaf=leaf,
                        max_features=mf,
                        random_state=RANDOM_STATE,
                        n_jobs=1 if HEADLESS else -1,  # stable in constrained envs
                    )
                    scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
                    results.append(
                        {
                            "n_estimators": n_est,
                            "max_depth": md,
                            "min_samples_leaf": leaf,
                            "max_features": mf,
                            "cv_mean_R2": float(scores.mean()),
                            "cv_std_R2": float(scores.std(ddof=1)),
                        }
                    )

    res_df = pd.DataFrame(results).sort_values("cv_mean_R2", ascending=False)

    print("=== RF CV results (top 5) ===")
    print(res_df.head(5).to_string(index=False))

    best = res_df.iloc[0]

    # IMPORTANT: DataFrame stores None as NaN. Convert back.
    best_max_depth = best["max_depth"]
    if pd.isna(best_max_depth):
        best_max_depth = None

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": best_max_depth,
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
        "random_state": RANDOM_STATE,
        "n_jobs": 1 if HEADLESS else -1,
    }

    print("\nBest RF params:")
    print(best_params)

    best_rf = RandomForestRegressor(**best_params).fit(X, y)
    return best_rf, res_df


# ============================
# 2) Accuracy comparison
# ============================
def model_comparison(X, y, feature_names, rf_model, save_parity_csv: bool = True):
    """Compare Linear, Quadratic, and tuned RF; save figures & optional CSV."""
    # Linear
    lin1 = LinearRegression().fit(X, y)
    y_pred_lin1 = lin1.predict(X)
    R2_lin1 = r2_score(y, y_pred_lin1)

    # Quadratic
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    lin2 = LinearRegression().fit(X_poly, y)
    y_pred_lin2 = lin2.predict(X_poly)
    R2_lin2 = r2_score(y, y_pred_lin2)

    # RF
    rf = rf_model
    y_pred_rf = rf.predict(X)
    R2_rf = r2_score(y, y_pred_rf)

    print("=== Model comparison (R² on full data) ===")
    print(f"Linear (1st)   : {R2_lin1:.4f}")
    print(f"Quadratic (2nd): {R2_lin2:.4f}")
    print(f"Random Forest  : {R2_rf:.4f}")

    # R2 bar chart
    plt.figure(figsize=(5, 3))
    plt.bar(["Linear", "Quadratic", "RF (tuned)"], [R2_lin1, R2_lin2, R2_rf])
    plt.ylabel("R²")
    plt.ylim(0, 1.0)
    plt.title("Model comparison (Urbach)")
    save_fig("01_r2_comparison.png")

    # Parity plots
    def parity_plot(y_true, y_pred, title, filename):
        plt.figure(figsize=(4, 4))
        plt.scatter(y_true, y_pred)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("Measured Eu (meV)")
        plt.ylabel("Predicted Eu (meV)")
        plt.title(title)
        save_fig(filename)

    parity_plot(y, y_pred_lin1, f"Linear parity (R²={R2_lin1:.3f})", "02_parity_linear.png")
    parity_plot(y, y_pred_lin2, f"Quadratic parity (R²={R2_lin2:.3f})", "03_parity_quadratic.png")
    parity_plot(y, y_pred_rf,  f"RF parity (R²={R2_rf:.3f})", "04_parity_rf.png")

    # Save parity CSV
    if save_parity_csv:
        df_parity = pd.DataFrame(
            {
                "Temp": X[:, 0],
                "RF": X[:, 1],
                "Ar": X[:, 2],
                "O2": X[:, 3],
                "Urbach_true": y,
                "Urbach_pred_linear": y_pred_lin1,
                "Urbach_pred_quadratic": y_pred_lin2,
                "Urbach_pred_rf": y_pred_rf,
            }
        )
        out = "parity_scatter_linear_quadratic_rf.csv"
        df_parity.to_csv(out, index=False)
        print(f"Saved parity data to '{out}'.")

    # RF feature importance
    plt.figure(figsize=(5, 3))
    plt.bar(feature_names, rf.feature_importances_)
    plt.ylabel("Feature importance")
    plt.title("RF Feature Importances (tuned RF)")
    save_fig("05_rf_feature_importance.png")

    print("\nRF feature importances (tuned RF):")
    for name, val in zip(feature_names, rf.feature_importances_):
        print(f"  {name}: {val:.4f}")

    # RF 1D PDP (manual)
    for i, name in enumerate(feature_names):
        grid, vals = compute_1d_pdp(rf, X, i, grid_resolution=60)
        plt.figure(figsize=(4.5, 3))
        plt.plot(grid, vals)
        plt.xlabel(name)
        plt.ylabel("Partial dependence (Eu, meV)")
        plt.title(f"PDP: Eu vs {name}")
        save_fig(f"06_pdp_1d_{i}_{name}.png")

    return {"lin1_R2": R2_lin1, "lin2_R2": R2_lin2, "rf_R2": R2_rf}


# ======================
# Helper: 1D / 2D PDP
# ======================
def compute_1d_pdp(model, X, feature_index, grid_resolution=40):
    X_arr = X.copy()
    col = X_arr[:, feature_index]
    grid = np.linspace(col.min(), col.max(), grid_resolution)
    pdp_vals = []
    for g in grid:
        X_mod = X_arr.copy()
        X_mod[:, feature_index] = g
        pdp_vals.append(model.predict(X_mod).mean())
    return grid, np.array(pdp_vals)


def compute_2d_pdp(model, X, i, j, grid_resolution=20):
    X_arr = X.copy()
    xi = X_arr[:, i]
    xj = X_arr[:, j]
    grid_i = np.linspace(xi.min(), xi.max(), grid_resolution)
    grid_j = np.linspace(xj.min(), xj.max(), grid_resolution)
    F = np.zeros((grid_resolution, grid_resolution))
    for a, gi in enumerate(grid_i):
        X_mod = X_arr.copy()
        X_mod[:, i] = gi
        for b, gj in enumerate(grid_j):
            X_mod[:, j] = gj
            F[a, b] = model.predict(X_mod).mean()
    return grid_i, grid_j, F


def manual_2d_pdp_array(model, X_arr, i, j, grid_resolution=15):
    """2D PDP helper for Friedman H² (returns F, grid_i, grid_j)."""
    grid_i, grid_j, F = compute_2d_pdp(model, X_arr, i, j, grid_resolution=grid_resolution)
    return F, grid_i, grid_j


def friedman_H2(model, X_arr, i, j, grid_resolution=15):
    F, gi, gj = manual_2d_pdp_array(model, X_arr, i, j, grid_resolution)
    F_mean = F.mean()
    f_i = F.mean(axis=1, keepdims=True)
    f_j = F.mean(axis=0, keepdims=True)
    F_int = F - f_i - f_j + F_mean
    num = np.sum(F_int ** 2)
    den = np.sum((F - F_mean) ** 2)
    H2 = num / den if den > 0 else 0.0
    return float(np.sqrt(H2)), float(H2)


def interp1d_clipped(x_grid, y_grid, x):
    xg = np.asarray(x_grid)
    yg = np.asarray(y_grid)
    xx = np.asarray(x)
    xx = np.clip(xx, xg.min(), xg.max())
    return np.interp(xx, xg, yg)


def bilinear_interp(xg, yg, zg, x, y):
    xg = np.asarray(xg)
    yg = np.asarray(yg)
    zg = np.asarray(zg)
    x = np.clip(np.asarray(x), xg.min(), xg.max())
    y = np.clip(np.asarray(y), yg.min(), yg.max())

    nx = len(xg)
    ny = len(yg)
    ix = np.searchsorted(xg, x, side="right") - 1
    iy = np.searchsorted(yg, y, side="right") - 1
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)

    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]
    tx = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (y - y0) / (y1 - y0), 0.0)

    z00 = zg[ix, iy]
    z10 = zg[ix + 1, iy]
    z01 = zg[ix, iy + 1]
    z11 = zg[ix + 1, iy + 1]

    z0 = z00 * (1 - tx) + z10 * tx
    z1 = z01 * (1 - tx) + z11 * tx
    return z0 * (1 - ty) + z1 * ty


# ============================
# 3) 2D PDP + additive model + Friedman H²
# ============================
def analyze_2d_pdp_and_interactions(rf, X, feature_names):
    y_rf = rf.predict(X)

    # 2D PDP surfaces: Temp vs others
    pairs_for_surface = [(0, 1), (0, 2), (0, 3)]  # Temp-RF, Temp-Ar, Temp-O2
    for (i, j) in pairs_for_surface:
        grid_i, grid_j, F = compute_2d_pdp(rf, X, i, j, grid_resolution=20)
        XI, XJ = np.meshgrid(grid_i, grid_j, indexing="ij")

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(XI, XJ, F, linewidth=0, antialiased=True)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_zlabel("PDP Eu (meV)")
        ax.set_title(f"2D PDP: {feature_names[i]}–{feature_names[j]}")
        save_fig(f"07_pdp_2d_{feature_names[i]}_{feature_names[j]}.png")

    # Pairwise Friedman H^2 interaction matrix
    H2_mat = np.zeros((len(feature_names), len(feature_names)))
    pairs_list = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            H, H2 = friedman_H2(rf, X, i, j, grid_resolution=15)
            H2_mat[i, j] = H2
            H2_mat[j, i] = H2
            pairs_list.append((feature_names[i], feature_names[j], H2))

    print("Pairwise H² (i<j):")
    for name_i, name_j, H2 in pairs_list:
        print(f"{name_i}–{name_j}: H²={H2:.4f}")

    plt.figure(figsize=(5, 4))
    im = plt.imshow(H2_mat)
    plt.xticks(range(len(feature_names)), feature_names)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.colorbar(im, label="H²")
    plt.title("Friedman H² interaction matrix (tuned RF)")
    save_fig("09_friedman_h2_matrix.png")


# ============================
# 4) Temp–O2 local maps (470–550°C & O2=0–5)
# ============================
def analyze_T_O2_optimal_region(rf, X):
    idx_T = 0
    idx_O2 = 3

    grid_T_1d, f_T = compute_1d_pdp(rf, X, feature_index=idx_T, grid_resolution=80)
    grid_O2_1d, f_O2 = compute_1d_pdp(rf, X, feature_index=idx_O2, grid_resolution=80)

    grid_T = np.linspace(470, 550, 40)
    grid_O2 = np.arange(0, 6, 1)

    X_base = X.copy()

    F_TO = np.zeros((len(grid_T), len(grid_O2)))
    for a, T_val in enumerate(grid_T):
        for b, O2_val in enumerate(grid_O2):
            X_mod = X_base.copy()
            X_mod[:, idx_T] = T_val
            X_mod[:, idx_O2] = O2_val
            F_TO[a, b] = rf.predict(X_mod).mean()

    FT = interp1d_clipped(grid_T_1d, f_T, grid_T)
    FO2 = interp1d_clipped(grid_O2_1d, f_O2, grid_O2)
    F_mean = F_TO.mean()
    F_int = F_TO - FT[:, None] - FO2[None, :] + F_mean

    threshold = np.percentile(F_TO, 10)
    low_mask = F_TO <= threshold

    # Eu(T,O2) PDP
    plt.figure(figsize=(6, 4))
    plt.imshow(
        F_TO.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[grid_T.min(), grid_T.max(), grid_O2.min() - 0.5, grid_O2.max() + 0.5],
        vmin=F_TO.min(),
        vmax=F_TO.max(),
    )
    plt.xlabel("Temp (°C)")
    plt.ylabel("O$_2$ flow (sccm)")
    plt.yticks(grid_O2)
    plt.title("Eu(T, O$_2$) PDP (470–550 °C)")
    plt.colorbar(label="Eu (meV)")
    plt.contour(grid_T, grid_O2, low_mask.T, levels=[0.5], colors="white", linewidths=1.5)
    save_fig("10_temp_o2_pdp_region.png")

    # Interaction component
    plt.figure(figsize=(6, 4))
    plt.imshow(
        F_int.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[grid_T.min(), grid_T.max(), grid_O2.min() - 0.5, grid_O2.max() + 0.5],
        vmin=F_int.min(),
        vmax=F_int.max(),
    )
    plt.xlabel("Temp (°C)")
    plt.ylabel("O$_2$ flow (sccm)")
    plt.yticks(grid_O2)
    plt.title("Interaction component F_int(T, O$_2$)")
    plt.colorbar(label="interaction (meV)")
    plt.contour(grid_T, grid_O2, low_mask.T, levels=[0.5], colors="white", linewidths=1.5)
    save_fig("11_temp_o2_interaction_region.png")


# ============================
# 5) Decompose RF: additive (1D) + Temp–O2 interaction (2D residual)
# ============================
def decompose_rf_with_1d_and_TO2_interaction(rf, X, y):
    """
    Approximate RF prediction as:
      Eu ≈ c0 + Σ_j g_j(x_j)              (additive only)
      Eu ≈ c0 + Σ_j g_j(x_j) + g_int(T,O2) (additive + Temp–O2 interaction)
    and report R² values.
    """
    yhat_rf = rf.predict(X)

    PERCENTILES = (0.0, 1.0)
    GRID_1D = 50
    GRID_2D = 30

    # 1D PDPs
    pdp_1d = {}
    for j, col in enumerate(FEATURE_COLS):
        pd = partial_dependence(
            rf,
            X=X,
            features=[j],
            kind="average",
            percentiles=PERCENTILES,
            grid_resolution=GRID_1D,
        )
        grid_list, avg_list = _pd_grid_and_avg(pd)
        grid = np.asarray(grid_list[0])
        avg = np.asarray(avg_list[0])
        pdp_1d[col] = (grid, avg)

    # Additive (1D) approximation
    sum_1d = np.zeros_like(yhat_rf)
    for j, col in enumerate(FEATURE_COLS):
        grid, avg = pdp_1d[col]
        sum_1d += interp1d_clipped(grid, avg, X[:, j])

    c0_add = float(np.mean(yhat_rf - sum_1d))
    y_add = c0_add + sum_1d

    r2_rf_meas = r2_score(y, yhat_rf)
    r2_add_vs_rf = r2_score(yhat_rf, y_add)
    r2_add_vs_meas = r2_score(y, y_add)

    # 2D PDP for (Temp, O2), and extract interaction residual
    idx_T = FEATURE_COLS.index("Temp")
    idx_O2 = FEATURE_COLS.index("O2")

    pd2 = partial_dependence(
        rf,
        X=X,
        features=[(idx_T, idx_O2)],
        kind="average",
        percentiles=PERCENTILES,
        grid_resolution=GRID_2D,
    )
    grid_list_2d, avg_list_2d = _pd_grid_and_avg(pd2)
    T_grid, O2_grid = [np.asarray(v) for v in grid_list_2d]
    PDP2D = np.asarray(avg_list_2d[0])

    T1_grid, T1_avg = pdp_1d["Temp"]
    O21_grid, O21_avg = pdp_1d["O2"]

    gT_on_Tgrid = interp1d_clipped(T1_grid, T1_avg, T_grid)
    gO2_on_O2grid = interp1d_clipped(O21_grid, O21_avg, O2_grid)

    raw_int = PDP2D - gT_on_Tgrid[:, None] - gO2_on_O2grid[None, :]
    c1 = -float(np.mean(raw_int))  # center the interaction term
    g_int_grid = raw_int + c1

    g_int_samples = bilinear_interp(T_grid, O2_grid, g_int_grid, X[:, idx_T], X[:, idx_O2])

    c0_total = float(np.mean(yhat_rf - (sum_1d + g_int_samples)))
    y_total = c0_total + sum_1d + g_int_samples

    r2_total_vs_rf = r2_score(yhat_rf, y_total)
    r2_total_vs_meas = r2_score(y, y_total)

    print("=== Full RF ===")
    print(f"R2 (RF prediction vs measured Eu): {r2_rf_meas:.4f}")
    print("\n=== Additive (1D PDP sum + const) ===")
    print(f"R2 (additive approx vs full RF predictions): {r2_add_vs_rf:.4f}")
    print(f"R2 (additive approx vs measured Eu):        {r2_add_vs_meas:.4f}")
    print("\n=== Additive + Temp–O2 interaction (2D PDP residual) ===")
    print(f"R2 (total approx vs full RF predictions):   {r2_total_vs_rf:.4f}")
    print(f"R2 (total approx vs measured Eu):          {r2_total_vs_meas:.4f}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    print("=== Step 0: Load data ===")
    df, X, y = load_data(CSV_PATH)
    print(f"Loaded {len(df)} rows. Features={FEATURE_COLS}, Target={TARGET_COL}")

    # Choose FAST mode if you want guaranteed quick completion.
    # Set to False to use the full (heavier) grid.
    FAST_MODE = False

    print("\n=== Step 1: Tune Random Forest (CV) ===")
    param_grid = RF_PARAM_GRID_FAST if FAST_MODE else RF_PARAM_GRID
    best_rf, cv_results = tune_rf_with_cv(X, y, param_grid=param_grid, n_splits=5)

    print("\n=== Step 2: Model comparison + outputs ===")
    model_comparison(X, y, FEATURE_COLS, rf_model=best_rf, save_parity_csv=True)

    print("\n=== Step 3: 2D PDP + additive model + Friedman H² ===")
    analyze_2d_pdp_and_interactions(best_rf, X, FEATURE_COLS)

    print("\n=== Step 4: Temp–O2 local maps (saved, non-blocking) ===")
    analyze_T_O2_optimal_region(best_rf, X)

    print("\n=== Step 5: Decomposition R² report (this should ALWAYS print) ===")
    decompose_rf_with_1d_and_TO2_interaction(best_rf, X, y)

    print("\nDone. Figures are saved under:", FIG_DIR)
