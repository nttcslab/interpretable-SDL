README.txt — How to run the interpretable RF analysis
====================================================

This README explains how to run the provided Python script to:
- Tune a Random Forest model (CV)
- Compare Linear / Quadratic / Random Forest models
- Output parity CSV
- Save figures (non-blocking; no plt.show())
- Print:
    - RF feature importances
    - Pairwise Friedman H² interaction scores
    - R² for additive decomposition (1D PDP) and additive+Temp–O2 interaction

Target script
-------------
- Interpretable_analysis.py

Input data
----------
- BOdata_growth.csv

Expected CSV columns
--------------------
Required feature columns (order is fixed in the script):
- Temp
- RF
- Ar
- O2

Target column:
- Urbach

Notes:
- If Urbach has NaN, the script fills NaN with the maximum Urbach value in the CSV.

Requirements
------------
- Python 3.9+ (recommended: 3.10 or 3.11)
- Packages:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

Setup (recommended: virtual environment)
----------------------------------------
1) Create and activate a virtual environment.

   macOS / Linux:
     python3 -m venv .venv
     source .venv/bin/activate

   Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1

2) Install dependencies.

   pip install -U pip
   pip install numpy pandas matplotlib scikit-learn


How to run
----------
1) Put these files in the same folder:
   - Interpretable_analysis.py
   - BOdata_growth.csv

2) Run:

   python Interpretable_analysis.py


Outputs
-------
A) Console output (examples)
- Best RF params chosen by CV
- Model comparison R² (Linear / Quadratic / RF)
- RF feature importances (printed)
- Pairwise H² interaction scores (printed)
- Decomposition R² report (printed):
    === Full RF ===
    === Additive (1D PDP sum + const) ===
    === Additive + Temp–O2 interaction (2D PDP residual) ===

B) Files created
- parity_scatter_linear_quadratic_rf.csv
  Columns:
    Temp, RF, Ar, O2,
    Urbach_true,
    Urbach_pred_linear,
    Urbach_pred_quadratic,
    Urbach_pred_rf

C) Figures
- A folder named 'figures' is created (if not present).
- Plots are saved as PNG into 'figures/' (no GUI required).
  Examples:
    01_r2_comparison.png
    02_parity_linear.png
    03_parity_quadratic.png
    04_parity_rf.png
    05_rf_feature_importance.png
    06_pdp_1d_*.png
    07_pdp_2d_*.png
    09_friedman_h2_matrix.png
    10_temp_o2_pdp_region.png
    11_temp_o2_interaction_region.png


Performance / reproducibility notes
-----------------------------------
- RF tuning (CV) can be slow if the parameter grid is large.
- The script uses FAST_MODE=True by default (smaller grid).
  If you want the full (heavier) grid, set:
    FAST_MODE = False
- Reproducibility:
  - CV splits and RF training use RANDOM_STATE=0.
  - n_jobs is set conservatively for stable runs.
  - Results can still vary slightly if you change OS / BLAS / scikit-learn versions.

