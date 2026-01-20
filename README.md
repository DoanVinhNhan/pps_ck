# Numerical Methods for ODEs & Integration

This project implements various numerical methods for solving Ordinary Differential Equations (ODEs) and Numerical Integration. It allows for high-precision calculations, detailed step-by-step reporting, and automatic error estimation.

## Structure

The project is now organized into logical sub-modules:

- **`methods/`**: Core implementations of numerical algorithms.
  - **`ODE/`**: Solvers for Ordinary Differential Equations.
    - Euler methods (Explicit, Implicit)
    - Runge-Kutta methods (RK2, RK3, RK4)
    - Adams-Bashforth and Adams-Moulton methods
    - Finite Difference Method (FDM) for BVPs
  - **`Integration/`**: Numerical Integration for **Discrete Data Points**.
    - Trapezoidal Rule
    - Simpson's 1/3 Rule
    - Newton-Cotes
  - **`Function_Integration/`**: Numerical Integration for **Mathematical Functions**.
    - Trapezoidal Rule (Function Mode)
    - Simpson's 1/3 Rule (Function Mode)
    - Newton-Cotes (Function Mode)

- **`runners/`**: Scripts to execute the methods and visualize results.
  - **`ODE/`**: Runners for ODE problems (1D, 2D, FDM).
  - **`Integration/`**: Runners for integration from CSV data.
  - **`Function_Integration/`**: Runners for integration of direct functions $f(x)$.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Ordinary Differential Equations (ODE)

Run runners located in `runners/ODE/` to solve 1D or 2D ODE problems or BVPs.

**Examples:**
```bash
# Rate 4 Runge-Kutta for 1D ODE
python runners/ODE/1D_RK4.py

# System of ODEs (2D)
python runners/ODE/2D_RK4.py

# Finite Difference Method
python runners/ODE/FiniteDifferenceMethod.py
```

### 2. Numerical Integration (Data Mode)

Use these runners when you have discrete data points (e.g., from a CSV file).

**Examples:**
```bash
# Simpson's Rule from Data
python runners/Integration/Simpson.py

# Trapezoidal Rule from Data
python runners/Integration/Trapezoidal.py
```

### 3. Numerical Integration (Function Mode)

Use these runners when you want to integrate a defined mathematical function $f(x)$ (e.g., $e^x$, $\sin(x)$).

**Examples:**
```bash
# Simpson's Rule for f(x)
python runners/Function_Integration/Simpson.py

# Newton-Cotes for f(x)
python runners/Function_Integration/NewtonCotes.py
```

**Key Features:**

*   **Detailed Table**: Displays calculation details for the first iteration (Weights $C_i$, Terms).
*   **Automatic Convergence**: Iteratively refines the grid (doubling $N$) until the estimated error meets `EPSILON`.
*   **Runge Error Estimation**: Uses Runge's principle to estimate error between iterations.
*   **Rich Output**: Beautiful terminal output with tables and panels.
*   **Reports**: Automatically saves results to `.txt` and `.csv` files in `output/` directory.
