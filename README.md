# Numerical Methods for ODEs

This project implements various numerical methods for solving Ordinary Differential Equations (ODEs), including 1D and 2D problems. It includes core algorithm implementations and runner scripts for visualization and reporting.

## Structure

The project is organized into two main directories:

- **`methods/`**: Contains the core implementations of numerical algorithms.
  - Euler methods (Explicit, Implicit)
  - Runge-Kutta methods (RK2, RK3, RK4)
  - Adams-Bashforth and Adams-Moulton methods (multi-step)

- **`runners/`**: Contains scripts to execute the methods, visualize results, and generate reports.
  - `Run_ODE_1D_*.py`: Runners for 1D ODE problems.
  - `Run_ODE_2D_*.py`: Runners for 2D ODE problems (systems of equations).

## Installation

1.  Clone the repository (if applicable).
2.  Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run individual runner scripts to solve specific ODE problems. For example, to run the 4th-order Runge-Kutta method for a 1D ODE:

```bash
python runners/Run_ODE_1D_RK4.py
```

Or for a 2D system:

```bash
python runners/Run_ODE_2D_RK4.py
```

The scripts will typically output a table of results to the console (using `rich`) and may generate plots using `matplotlib`.
