# estimate_Eg_Eu.py
# -------------------------------------
# CSVの透過スペクトルから
#  1) Tauc (direct allowed) で Eg を推定
#  2) Urbach で Eu を推定
#
# 2025-12-26: Tauc 側も Urbach と同じ alpha(1/cm) = -ln(T)/d を使うように修正
#            （onset検出の閾値は従来の -ln(T) 基準のまま使えるよう、自動で 1/d スケール）

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def linear_fit(x, y):
    """y = m x + b を最小二乗でフィットし (m,b,r2) を返す"""
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m * x + b
    r2 = r2_score(y, yhat)
    return m, b, r2


def detect_absorption_onset(E, alpha, alpha_thresh=1e-3, dalpha_thresh=1e-3):
    """吸収の立ち上がりインデックスを返す。条件を満たさない場合は0。
    条件: alpha > alpha_thresh かつ d(alpha)/dE > dalpha_thresh
    """
    dalpha_dE = np.gradient(alpha, E)
    mask = (alpha > alpha_thresh) & (dalpha_dE > dalpha_thresh)
    return int(np.argmax(mask)) if mask.any() else 0


def find_first_linear_segment(E, y, min_width_eV=0.20, r2_threshold=0.995, start_idx=0):
    """start_idx以降で、最初に見つかるR2>=閾値の線形区間を返す。
    戻り: (m, b, i0, j0)
    """
    n = len(E)
    for i0 in range(start_idx, n - 2):
        for j0 in range(i0 + 2, n):
            width = E[j0] - E[i0]
            if width < min_width_eV:
                continue
            x = E[i0:j0 + 1]
            yy = y[i0:j0 + 1]
            m, b, r2 = linear_fit(x, yy)
            if r2 >= r2_threshold and m > 0:
                return m, b, i0, j0
    raise RuntimeError("No linear segment found for Eg with given criteria.")


def pick_urbach_window(E, ln_a, Eg, end_offset=0.03, win_width=0.08,
                       r2_threshold=0.995, min_pts=8, expand_max=0.40):
    """Urbach windowを Eg の少し下に取り、
    ln(alpha) vs E を直線フィットして Eu = 1/slope を求める。
    window幅が足りない場合 expand_max まで広げる。
    戻り: (m, b, startE, endE, r2, npts)
    """
    endE0 = Eg - end_offset
    # E配列は昇順想定
    idx_end = np.searchsorted(E, endE0)
    if idx_end <= 0:
        raise RuntimeError("Eg too low or energy grid invalid for Urbach window end.")
    # 初期 window: [endE0 - win_width, endE0]
    width = win_width
    while width <= expand_max:
        startE0 = endE0 - width
        idx_start = np.searchsorted(E, startE0)
        if idx_end - idx_start >= min_pts:
            x = E[idx_start:idx_end]
            y = ln_a[idx_start:idx_end]
            m, b, r2 = linear_fit(x, y)
            if r2 >= r2_threshold and m > 0:
                return m, b, startE0, endE0, r2, (idx_end - idx_start)
        width += win_width
    raise RuntimeError("No suitable Urbach window found within expand_max.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--energy-col', default='energy', help='Photon energy column (eV)')
    parser.add_argument('--trans-col',   default='Tr_film', help='Transmittance column')
    parser.add_argument('--thickness-nm', type=float, default=55, help='Film thickness (nm)')

    # Eg(Tauc) settings
    parser.add_argument('--eg-min-energy', type=float, default=5.1, help='Start searching Eg from this energy (eV)')
    parser.add_argument('--min-width-eg',  type=float, default=0.20, help='Min width (eV) for Eg linear region')
    parser.add_argument('--r2-eg',         type=float, default=0.995, help='R^2 threshold for Eg fit')
    parser.add_argument('--alpha-thresh',  type=float, default=1e-3,
                        help='Onset detection threshold for -ln(T) (dimensionless). If Tauc uses thickness-based alpha, this is auto-scaled by 1/d')
    parser.add_argument('--dalpha-thresh', type=float, default=1e-3,
                        help='Onset detection threshold for d(-ln(T))/dE. If Tauc uses thickness-based alpha, this is auto-scaled by 1/d')

    # Eu(Urbach) settings
    parser.add_argument('--end-offset',    type=float, default=0.03, help='How far below Eg the Eu window ends (eV)')
    parser.add_argument('--win-width-eu',  type=float, default=0.08, help='Initial window width for Eu (eV)')
    parser.add_argument('--r2-eu',         type=float, default=0.995, help='R^2 threshold for Eu fit')
    parser.add_argument('--eu-expand-max', type=float, default=0.40, help='Max expansion (eV) if not enough points')

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.energy_col not in df.columns:
        raise KeyError(f"Energy column '{args.energy_col}' not found in CSV.")
    if args.trans_col not in df.columns:
        raise KeyError(f"Transmittance column '{args.trans_col}' not found in CSV.")

    E = df[args.energy_col].values.astype(float)
    T_raw = df[args.trans_col].values.astype(float)

    # Ensure sorted by energy
    sort_idx = np.argsort(E)
    E = E[sort_idx]
    T_raw = T_raw[sort_idx]

    if np.any(np.isnan(E)) or np.any(np.isnan(T_raw)):
        raise ValueError("NaN detected in input columns.")

    # Normalize transmittance to max=1 and clip
    if T_raw.max() == 0:
        raise ValueError('Transmittance column is all zeros or invalid.')
    T = T_raw / T_raw.max()
    T = np.clip(T, 0.0, 1.0)  # ensure never >1

    # Thickness-based absorption coefficient (ignore reflection): alpha = -ln(T)/d
    d_cm = args.thickness_nm * 1e-7  # nm -> cm
    alpha_rel = -np.log(T + EPS)     # dimensionless (-ln T)
    alpha = alpha_rel / d_cm         # 1/cm

    # Tauc (direct allowed) using alpha [1/cm]
    y_tauc = (alpha * E) ** 2        # (eV/cm)^2

    # Detect onset (keep CLI thresholds compatible with the old -ln(T) convention)
    alpha_thresh_cm = args.alpha_thresh / d_cm
    dalpha_thresh_cm = args.dalpha_thresh / d_cm
    onset_idx = detect_absorption_onset(E, alpha, alpha_thresh_cm, dalpha_thresh_cm)

    # Ensure start index respects eg_min_energy
    idx_min = np.searchsorted(E, args.eg_min_energy)
    start_idx = max(onset_idx, idx_min)

    # Eg fit
    mE, bE, i0, j0 = find_first_linear_segment(
        E, y_tauc,
        min_width_eV=args.min_width_eg,
        r2_threshold=args.r2_eg,
        start_idx=start_idx
    )
    Eg = -bE / mE

    # Urbach analysis uses the same thickness-based alpha [1/cm]
    ln_a = np.log(alpha + EPS)

    # Urbach fit (auto expand)
    mU, bU, sE, eE, r2_u, npts = pick_urbach_window(
        E, ln_a, Eg,
        end_offset=args.end_offset,
        win_width=args.win_width_eu,
        r2_threshold=args.r2_eu,
        expand_max=args.eu_expand_max
    )
    Eu = 1.0 / mU  # because ln(alpha) = (1/Eu) E + const

    print(f"Estimated Eg = {Eg:.4f} eV")
    print(f"Estimated Eu = {Eu:.4f} eV  (R2={r2_u:.5f}, npts={npts}, window=[{sE:.3f}, {eE:.3f}] eV)")

    # ---------------- Plotting ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=False)

    # Tauc plot
    ax1.plot(E, y_tauc, '.', ms=3, label='Tauc data')
    x_fit = E[i0:j0 + 1]
    ax1.plot(x_fit, mE * x_fit + bE, label='Linear fit')
    ax1.axvline(Eg, ls='--', label=f'Eg = {Eg:.2f} eV')
    ax1.set_xlabel('Photon Energy (eV)')
    ax1.set_ylabel('$(\\alpha h\\nu)^2$ (eV$^2$ cm$^{-2}$)')
    ax1.set_title('Tauc Plot using $\\alpha$ (1/cm) (first linear region)')
    ax1.legend()
    ax1.set_xlim(4.0, E.max())  # plot range from 4 eV
    ax1.set_ylim(0, None)       # y >= 0

    # Urbach plot (ln alpha vs E)
    ax2.plot(E, ln_a, '.', ms=3, label='ln(alpha)')
    # plot fit line in the selected window
    idx_s = np.searchsorted(E, sE)
    idx_e = np.searchsorted(E, eE)
    x_u = E[idx_s:idx_e]
    ax2.plot(x_u, mU * x_u + bU, label=f'Urbach fit (Eu={Eu:.3f} eV)')
    ax2.axvline(Eg, ls='--', label=f'Eg = {Eg:.2f} eV')
    ax2.set_xlabel('Photon Energy (eV)')
    ax2.set_ylabel('ln(α)  [α in 1/cm]')
    ax2.set_title('Urbach Plot')
    ax2.legend()
    ax2.set_xlim(E.min(), E.max())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
