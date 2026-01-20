# Numerical Methods Project

This project implements a comprehensive suite of numerical methods for solving Ordinary Differential Equations (ODEs), Numerical Integration, Interpolation, and Regression. It is designed for high-precision calculations, offering detailed step-by-step reporting, automatic error estimation, and visualization.

## Structure

The project is organized into logical sub-modules within `methods/` for core algorithms and `runners/` for execution scripts:

- **`methods/`**: Core implementations.
  - **`ODE/`**: Solvers for Ordinary Differential Equations.
    - Euler methods (Explicit, Implicit)
    - Runge-Kutta methods (RK2, RK3, RK4)
    - Adams-Bashforth and Adams-Moulton methods (Predictor-Corrector)
    - Finite Difference Method (FDM) for Boundary Value Problems
  - **`Integration/`**: Numerical Integration for **Discrete Data Points**.
    - Trapezoidal Rule
    - Simpson's 1/3 Rule
    - Newton-Cotes
  - **`Function_Integration/`**: Numerical Integration for **Mathematical Functions**.
    - Solvers for defined functions $f(x)$ with automatic convergence checking.
  - **`Interpolation/`**:
    - Spline Interpolation (Natural Cubic Spline)
  - **`Regression/`**:
    - Least Squares Regression (Generic basis functions)

- **`runners/`**: Scripts to execute the methods and generate reports/plots.
  - **`ODE/`**: Runners for 1D/2D ODEs and FDM.
  - **`Integration/`**: Runners for integration from CSV data.
  - **`Function_Integration/`**: Runners for function-based integration.
  - **`Interpolation/`**: Runners for interpolation tasks.
  - **`Regression/`**: Runners for curve fitting and regression.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the scripts located in the `runners/` directory. Results (CSV, Text Reports, Plots) will be generated in the `output/` directory.

### 1. Ordinary Differential Equations (ODE)

Solve Initial Value Problems (IVP) or Boundary Value Problems (BVP).

**Examples:**
```bash
# Rate 4 Runge-Kutta for 1D ODE
python runners/ODE/1D_RK4.py

# System of ODEs (2D) - e.g., Predator-Prey models
python runners/ODE/2D_RK4.py

# Finite Difference Method for BVP
python runners/ODE/FiniteDifferenceMethod.py
```

### 2. Numerical Integration (Data Mode)

Calculate integrals when you only have discrete data points (e.g., experimental data in a CSV file).

**Examples:**
```bash
# Simpson's Rule from Data
python runners/Integration/Simpson.py

# Trapezoidal Rule from Data
python runners/Integration/Trapezoidal.py
```

### 3. Numerical Integration (Function Mode)

Integrate defined mathematical functions $f(x)$ (e.g., $e^x, \sin(x)$) with automatic error control.

**Examples:**
```bash
# Simpson's Rule for f(x)
python runners/Function_Integration/Simpson.py

# Newton-Cotes via Function
python runners/Function_Integration/NewtonCotes.py
```

### 4. Interpolation

Construct new data points within the range of a discrete set of known data points.

**Examples:**
```bash
# Cubic Spline Interpolation
python runners/Interpolation/Spline.py
```

### 5. Regression

Find the best-fitting curve through a set of data points using the method of least squares.

**Examples:**
```bash
# Least Squares Regression (Generic)
python runners/Regression/LeastSquares.py
```

## Key Features

*   **Automatic Convergence**: Ideally refines grids or steps until estimated errors meet a specified `EPSILON`.
*   **Runge Error Estimation**: Uses Runge's principle for robust error checking.
*   **Detailed Reporting**: Generates rich `.txt` reports with intermediate steps and `.csv` files for raw data.
*   **Visualization**: Automatically generates Matplotlib graphs for visual verification of results.
*   **Polished Output**: Uses `rich` for beautiful, readable terminal output.
