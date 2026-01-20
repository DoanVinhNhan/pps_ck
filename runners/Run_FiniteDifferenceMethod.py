import numpy as np
import pandas as pd
import sys
import os
import inspect

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods.FiniteDifferenceMethod import solve_bvp_fdm

console = Console(record=True)

# --- Input Parameters ---
SELECTED_CASE = 2  # 1: Simple Poisson, 2: Euler-Cauchy (Variable Coeffs)

# --- Problem Definitions ---

# Case 1: Simple Poisson (u'' = -1)
def p1(x): return 1.0
def q1(x): return 0.0
def f1(x): return 1.0 
def exact1(x): return x * (1 - x) / 2.0

# Case 2: Euler-Cauchy with Source (Variable Coefficients)
# Eq: [x u']' - (1/x)u = -4x^2
def p2(x): return x
def q2(x): return 1.0/x
def f2(x): return -4.0 * x**2
def exact2(x): return x/6.0 + 4.0/(3.0*x) + x**3/2.0

def get_problem_setup(case_id):
    """
    Trả về cấu hình bài toán: (p, q, f, a, b, N, bc_a, bc_b, exact_func, desc)
    """
    if case_id == 1:
        # Simple Poisson: u'' = -1, u(0)=0, u(1)=0
        return (
            p1, q1, f1, 
            0.0, 1.0, 10,
            (1.0, 0.0, 0.0), # u(0)=0
            (1.0, 0.0, 0.0), # u(1)=0
            exact1,
            "Simple Poisson: u'' = -1, u(0)=u(1)=0"
        )
    elif case_id == 2:
        # Complex Euler-Cauchy: u(1)=2, u(2)=5
        return (
            p2, q2, f2,
            1.0, 2.0, 20,
            (1.0, 0.0, 2.0),
            (1.0, 0.0, 5.0),
            exact2,
            "Variable Coeffs (Euler-Cauchy): [x u']' - (1/x)u = 4x^2"
        )
    else:
        # Default to Case 1
        return get_problem_setup(1)

# --- Execution ---

def run():
    # 1. Configuration
    method_name = "Finite_Difference_Method"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get Problem Setup based on Global Parameter
    p_func, q_func, f_func, A, B, N, bc_a, bc_b, exact_func, case_desc = get_problem_setup(SELECTED_CASE)
    
    # 3. Print Info
    info = (
        f"Phương pháp: Finite Difference Method (FDM)\n"
        f"Bài toán (Case {SELECTED_CASE}): {case_desc}\n"
        f"Miền: [{A}, {B}]\n"
        f"Số khoảng N: {N}\n"
    )
    console.print(Panel(info, title="Finite Difference Method - Parameters", border_style="cyan"))
    
    # 4. Solve
    result = solve_bvp_fdm(p_func, q_func, f_func, A, B, N, bc_a, bc_b)
    
    x_nodes = result['x_nodes']
    u_values = result['u_values']
    logs = result['log']
    
    # 5. Exact Solution
    exact_values = None
    if exact_func is not None:
        exact_values = exact_func(x_nodes)
    
    # 6. Display Table
    table = Table(title=f"Kết quả FDM (Case {SELECTED_CASE})")
    table.add_column("i", justify="center")
    table.add_column("x_i", justify="right")
    table.add_column("u_FDM", justify="right", style="green")
    if exact_values is not None:
        table.add_column("u_Exact", justify="right", style="blue")
        table.add_column("Error", justify="right", style="red")
        
    for i in range(len(x_nodes)):
        row_data = [str(i), f"{x_nodes[i]:.4f}", f"{u_values[i]:.6f}"]
        if exact_values is not None:
            exact = exact_values[i]
            error = abs(u_values[i] - exact)
            row_data.append(f"{exact:.6f}")
            row_data.append(f"{error:.2e}")
        table.add_row(*row_data)
        
    console.print(table)
    
    # Print Logs
    if logs:
        console.print(Panel("\n".join(logs), title="Computation Log", border_style="yellow"))
        
    # 7. Save
    df = pd.DataFrame({
        "x": x_nodes,
        "u_FDM": u_values
    })
    if exact_values is not None:
        df["u_Exact"] = exact_values
        df["Error"] = np.abs(u_values - exact_values)
        
    csv_path = os.path.join(output_dir, "FDM_results.csv")
    df.to_csv(csv_path, index=False)
    console.print(f"[bold blue]Kết quả đã lưu tại: {csv_path}[/bold blue]")
    
    report_path = os.path.join(output_dir, "FDM_report.txt")
    console.save_text(report_path)

if __name__ == "__main__":
    run()
