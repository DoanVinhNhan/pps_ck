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

# --- Problem Definitions ---

# Example Problem: u'' + p(x)u' - q(x)u = -f(x)
# Let's solve: -u'' + u = 0  => u'' - u = 0 (p=-1, q=1, f=0? No wait, check form)
# Method Form: [p(x)u']' - q(x)u = -f(x)
# If p(x) = 1, then u'' - q(x)u = -f(x)

# Let's pick a problem with a known solution for verification.
# Problem: u'' = -1, u(0)=0, u(1)=0
# Solution: u(x) = x(1-x)/2
# Transform to [p u'] - q u = -f
# p(x) = 1 => u''
# q(x) = 0
# f(x) = -1 (since -f(x) = u'' => -(-1) = 1? No u'' = -1 corresponds to -f(x) = -1 => f(x) = 1)
# Checking: [1 * u']' - 0*u = -1 => u'' = -1. Correct. So f(x) = 1.

def p_func(x): return 1.0
def q_func(x): return 0.0 # Simple Poisson equation
def f_func(x): return 1.0 # u'' = -1

# Domain
A = 0.0
B = 1.0
N_DEFAULT = 10

# --- Boundary Condition Cases ---
# Format: (alpha, beta, gamma) => alpha*u + beta*u' = gamma

def get_boundary_conditions(case_id):
    """
    Trả về điều kiện biên dựa trên case_id đã chọn.
    """
    if case_id == 1:
        # Case 1: Dirichlet thuần (u(0)=0, u(1)=0)
        # 1*u + 0*u' = 0
        return (
            (1.0, 0.0, 0.0), # Left
            (1.0, 0.0, 0.0), # Right
            "Dirichlet (u(0)=0, u(1)=0)"
        )
    elif case_id == 2:
        # Case 2: Mixed (u(0)=0, u'(1)=0) - Sợi dây cố định 1 đầu, tự do 1 đầu?
        # Left: 1*u + 0*u' = 0
        # Right: 0*u + 1*u' = 0
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            "Mixed (u(0)=0, u'(1)=0)"
        )
    elif case_id == 3:
        # Case 3: Robin (u(0) - u'(0) = 0, u(1) + u'(1) = 0)
        # Left: 1*u - 1*u' = 0
        # Right: 1*u + 1*u' = 0
        return (
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            "Robin (u-u'=0 @0, u+u'=0 @1)"
        )
    else:
        # Default to Case 1
        return (
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "Dirichlet (Default)"
        )

# --- Execution ---

def run():
    # 1. Configuration
    method_name = "Finite_Difference_Method"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Select Case
    # Normally user input, here hardcoded or loop. Let's run Case 1 (Standard)
    # TODO: Allow interaction? Detailed Prompt said "THIẾT LẬP 1 TRƯỜNG HỢP (Loại điều kiện biên) LÀ 1 HÀM! RUNNER CHO PHÉP CHỌN"
    # I will simulate selection by running a default, but structuring it nicely.
    
    SELECTED_CASE = 1 
    bc_a, bc_b, case_desc = get_boundary_conditions(SELECTED_CASE)
    
    # 3. Print Info
    info = (
        f"Phương trình: [p(x)u']' - q(x)u = -f(x)\n"
        f"Với: p(x)=1, q(x)=0, f(x)=1 => u'' = -1\n"
        f"Miền: [{A}, {B}]\n"
        f"Số khoảng N: {N_DEFAULT}\n"
        f"Trường hợp biên: {case_desc}\n"
    )
    console.print(Panel(info, title="Finite Difference Method - Parameters", border_style="cyan"))
    
    # 4. Solve
    result = solve_bvp_fdm(p_func, q_func, f_func, A, B, N_DEFAULT, bc_a, bc_b)
    
    x_nodes = result['x_nodes']
    u_values = result['u_values']
    logs = result['log']
    
    # 5. Exact Solution for Case 1 (u''=-1, u(0)=0, u(1)=0) -> u = 0.5 * (x - x^2)
    # u = x(1-x)/2
    exact_values = None
    if SELECTED_CASE == 1:
        exact_values = (x_nodes * (1 - x_nodes)) / 2.0
    
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
