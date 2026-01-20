# Numerical Methods for ODEs & Integration

This project implements various numerical methods for solving Ordinary Differential Equations (ODEs) and Numerical Integration. It allows for high-precision calculations, detailed step-by-step reporting, and automatic error estimation.

## Structure

The project is organized into two main directories:

- **`methods/`**: Core implementations of numerical algorithms.
  - **ODE Solvers**:
    - Euler methods (Explicit, Implicit)
    - Runge-Kutta methods (RK2, RK3, RK4)
    - Adams-Bashforth and Adams-Moulton methods (multi-step)
  - **Numerical Integration**:
    - Trapezoidal Rule
    - Simpson's 1/3 Rule
    - Newton-Cotes (Degree 4, 5, 6)

- **`runners/`**: Scripts to execute the methods and visualize results.
  - `Run_ODE_*.py`: Runners for 1D and 2D ODE problems.
  - `Run_Integration_*.py`: Runners for Integration problems.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Numerical Integration

To run integration methods, execute the corresponding runner script. The scripts read data from `20251GKtest2.csv` (by default) and perform integration on a specified function `g(f(x), x)`.

**Available Runners:**

- **Trapezoidal Rule**:
  ```bash
  python runners/Run_Integration_Trapezoidal.py
  ```
- **Simpson's 1/3 Rule**:
  ```bash
  python runners/Run_Integration_Simpson.py
  ```
- **Newton-Cotes (General)**:
  ```bash
  python runners/Run_Integration_NewtonCotes.py
  ```

**Key Features:**

*   **Detailed Table**: Displays the first 6 and last 6 rows of the calculation table for the *first iteration*, showing $x_i, f(x_i), g(x_i), Raw~Weight~C_i, Term$.
*   **Automatic Convergence**: Iteratively doubles the number of intervals $N$ until the error meets `EPSILON`.
*   **Smart N Adjustment**: Automatically adjusts $N$ to meet method constraints (e.g., $N$ must be even for Simpson, divisible by $degree$ for Newton-Cotes) and warns the user if interpolation is required.
*   **Formula Display**: Shows the explicit global formula used (e.g., $\sum C_i g(x_i)$) and Big-O error estimates.

### 2. ODE Solvers

Run individual runner scripts to solve specific ODE problems:

- **1D ODE**: `python runners/Run_ODE_1D_RK4.py`
- **2D System**: `python runners/Run_ODE_2D_RK4.py`

The scripts utilize `rich` for beautiful terminal output and generate report files (`.txt`, `.csv`) in the `output/` directory.
