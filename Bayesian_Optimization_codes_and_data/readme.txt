BO.py

Usage:
  python "BO.py" "BO_data.csv"

Input CSV format:
  The input CSV file must contain five numeric columns:
    column 1: target value y, Urbach energy in meV
    column 2: T, substrate temperature in degC
    column 3: PRF, RF power in W
    column 4: FAr, Ar flow rate in sccm
    column 5: FO2, O2 flow rate in sccm

  The provided example file is BO_data.csv. A header line such as
  Urbach,Temp,RF,Ar,O2 is accepted. Missing target values can be written as nan.

Optimization setting:
  - Optimization mode: minimization of Urbach energy
  - Search range, matching the manuscript BO campaign:
      T   in [30, 700] degC
      PRF in [50, 150] W
      FAr in [10, 50] sccm
      FO2 in [0, 10] sccm
  - Default candidate grid size: 25 points per dimension
      The total number of candidate points is grid_size^4.
      A different value can be specified with --grid-size.
  - y-range: automatically determined from the minimum and maximum finite y values in the input data
  - Acquisition function: expected improvement
  - Gaussian process kernel:
      ConstantKernel * Matern(nu = 2.5) + WhiteKernel
  - NaN target values are handled by worst-value imputation after normalization/sign conversion.

Output:
  The script prints recommended candidate points and related optimization information to the console.
  Candidate input values are shown as (T, PRF, FAr, FO2).

Requirements:
  - Python 3
  - numpy
  - scipy
  - scikit-learn

Examples:
  python "BO(2).py" "BO_data(3).csv"
  python "BO(2).py" "BO_data(3).csv" --seed 0
  python "BO(2).py" "BO_data(3).csv" --seed 0 --grid-size 30 --top 20

Notes:
  - A grid size of 50 in four dimensions generates 50^4 = 6,250,000 candidate points and may require substantial memory and runtime.
  - The default grid size is therefore set to 25 for a lightweight reproducible example, while preserving the manuscript search ranges.
