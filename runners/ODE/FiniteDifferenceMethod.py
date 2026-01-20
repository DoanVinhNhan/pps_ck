import numpy as np
import pandas as pd
import sys
import os
import inspect

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from methods.ODE.FiniteDifferenceMethod import solve_bvp_fdm

console = Console(record=True)

# --- Input parameters ---

def p_func(x): return 1.0 + x    # p(x) = 1 + x (Variable coefficient)
def q_func(x): return 1.0        # q(x) = 1
def f_func(x): return -x         # f(x) = -x (Equilibrium: [(1+x)u']' - u = x)

# Domain
A = 0.0
B = 1.0
N_DEFAULT = 20    # Increased N for better precision with variable coeffs
SELECTED_CASE = 3 # 1: Dirichlet thuần (u(0)=0, u(1)=0), 
                  # 2: Mixed (u(0)=0, u'(1)=0),
                  # 3: Robin (u(0) - u'(0) = 0, u(1) + u'(1) = 0)



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
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Select Case
    # Normally user input, here hardcoded or loop. Let's run Case 1 (Standard)
    # TODO: Allow interaction? Detailed Prompt said "THIẾT LẬP 1 TRƯỜNG HỢP (Loại điều kiện biên) LÀ 1 HÀM! RUNNER CHO PHÉP CHỌN"
    # I will simulate selection by running a default, but structuring it nicely.
    
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
    
    # 3.1 Info hội tụ
    # Note: result is not available yet, move this after result is calculated or use known info
    # Better to wait until result is available.
    
    # 4. Solve
    result = solve_bvp_fdm(p_func, q_func, f_func, A, B, N_DEFAULT, bc_a, bc_b)

    if "convergence_info" in result:
        c_info = result["convergence_info"]
        info_text = f"Method Name: {c_info.get('method_name', 'Unknown')}\n"
        info_text += f"Order: {c_info.get('approximation_order', 'Unknown')}\n"
        info_text += f"Stability Condition: {c_info.get('stability_condition', 'Unknown')}\n"
        info_text += f"Matrix Solver: {c_info.get('matrix_solver', 'Unknown')}"
        console.print(Panel(info_text, title="[bold magenta]Hội tụ & Ổn định[/bold magenta]", expand=False))
    
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
        row_data = [str(i), f"{x_nodes[i]:.6g}", f"{u_values[i]:.8g}"]
        if exact_values is not None:
            exact = exact_values[i]
            error = abs(u_values[i] - exact)
            row_data.append(f"{exact:.8g}")
            row_data.append(f"{error:.4g}")
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

    # 8. Plotting
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot FDM solution
        plt.plot(x_nodes, u_values, 'o-', label='FDM Solution', markersize=4)
        
        # Plot Exact solution if available
        if exact_values is not None:
            # Generate smoother x for exact solution if possible, 
            # but here we just use the same nodes for simplicity or generate fine mesh
            x_fine = np.linspace(A, B, 200)
            if SELECTED_CASE == 1:
                u_exact_fine = (x_fine * (1 - x_fine)) / 2.0
                plt.plot(x_fine, u_exact_fine, 'r--', label='Exact Solution', alpha=0.7)
            else:
                 # If we had exact solutions for other cases, we would plot them here
                 plt.plot(x_nodes, exact_values, 'r--', label='Exact Solution', alpha=0.7)

        plt.title(f"Finite Difference Method Solution (Case {SELECTED_CASE})")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        plot_path = os.path.join(output_dir, "FDM_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        console.print(f"[bold blue]Biểu đồ đã lưu tại: {plot_path}[/bold blue]")
        
    except ImportError:
        console.print("[bold red]Lỗi: Không tìm thấy thư viện matplotlib. Vui lòng cài đặt: pip install matplotlib[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Lỗi khi vẽ biểu đồ: {e}[/bold red]")

if __name__ == "__main__":
    run()
