readme.txt — Running Transmit_analyses_20251226.py on W20250001– CSVs
============================================================================

This README explains how to run the script:
  Transmit_analyses_20251226.py

The script estimates:
  1) Optical bandgap Eg by Tauc plot (direct allowed): (α hν)^2 vs hν
  2) Urbach energy Eu from the Urbach tail: ln(α) vs hν

It reads a single CSV per run and prints Eg/Eu, and shows plots (Tauc + Urbach).

Input data set
--------------
You have multiple spectra files:
  W20250001_energy_Tr.csv
  W20250002_energy_Tr.csv
  ...

(Each file is processed independently; you can run the script in a loop.)

What the script expects in each CSV
-----------------------------------
Default column names (can be changed via CLI options):
  - Photon energy column:  energy   (eV)
  - Transmittance column:  Tr_film  (unitless, 0–1 recommended)

Important notes:
  - The script sorts rows by energy automatically.
  - It normalizes transmittance so that max(T)=1, then clips to [0,1].
  - Absorption coefficient is computed as:
        α (1/cm) = -ln(T) / d
    where d is thickness in cm (thickness-nm × 1e-7).

Requirements
------------
- Python 3.9+ (recommended: 3.10/3.11)
- Packages:
  - numpy
  - pandas
  - matplotlib

Setup (recommended: virtual environment)
----------------------------------------
macOS / Linux:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install numpy pandas matplotlib

Windows (PowerShell):
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -U pip
  pip install numpy pandas matplotlib


Run for one file
----------------
Example (defaults: energy column 'energy', trans column 'Tr_film', thickness 55 nm):

  python Transmit_analyses_20251226.py --csv W20250001_energy_Tr.csv

If your column names are different:

  python Transmit_analyses_20251226.py --csv W20250001_energy_Tr.csv \
    --energy-col energy \
    --trans-col Tr_film

If film thickness is not 55 nm, set it:

  python Transmit_analyses_20251226.py --csv W20250001_energy_Tr.csv --thickness-nm 55


Tips:
- The script shows plots (matplotlib). You must close the figure window to proceed to the next file.
- If you want a non-interactive batch (no GUI), modify the script to save figures instead of plt.show().


Key CLI options (most useful)
-----------------------------
--csv             (required) path to the CSV
--energy-col      photon energy column name (default: energy)
--trans-col       transmittance column name (default: Tr_film)
--thickness-nm    film thickness in nm (default: 55)

Eg (Tauc) related:
--eg-min-energy   start searching Eg above this energy (default: 5.1 eV)
--min-width-eg    minimum linear region width (default: 0.20 eV)
--r2-eg           R^2 threshold for Tauc linear segment (default: 0.995)
--alpha-thresh    onset threshold (legacy -ln(T) convention; auto-scaled internally)
--dalpha-thresh   onset slope threshold (legacy -ln(T) convention; auto-scaled internally)

Eu (Urbach) related:
--end-offset      window end is (Eg - end_offset) (default: 0.03 eV)
--win-width-eu    initial Urbach window width (default: 0.08 eV)
--r2-eu           R^2 threshold for Urbach linear fit (default: 0.995)
--eu-expand-max   maximum expansion if window fails (default: 0.40 eV)


Saving results (recommended)
----------------------------
By default the script prints results like:
  Estimated Eg = X.XXXX eV
  Estimated Eu = X.XXXX eV (R2=..., npts=..., window=[..., ...] eV)

If you want to save Eg/Eu for all 66 files into one summary CSV, use a wrapper
script that parses stdout (or modify the script to output a CSV).


Troubleshooting
---------------
- KeyError: column not found
  -> Use --energy-col / --trans-col to match your CSV headers.

- NaN detected
  -> Ensure 'energy' and 'Tr_film' columns contain valid numbers (no blanks).

- Transmittance is all zeros / invalid
  -> The script checks T_raw.max()==0 and errors out.

- No linear segment found for Eg
  -> Try lowering --r2-eg (e.g., 0.990) or reducing --min-width-eg,
     or adjusting --eg-min-energy.

- No suitable Urbach window found
  -> Try lowering --r2-eu slightly or increase --eu-expand-max.
