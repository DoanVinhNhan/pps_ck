import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from scipy.interpolate import interp1d

import sys
import os
import inspect # Added for dynamic g display

# Add parent directory to sys.path to import methods
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods.Integration_Simpson import simpson_integration
from methods.utils import load_data

console = Console(record=True)

# --- Configuration ---
CSV_FILE = "20251GKtest2.csv"
INPUT_PATH = os.path.join(os.path.dirname(__file__), '..', CSV_FILE)
X_COL = "x_i"
Y_COL = "y_i"

# --- Input Parameters ---
a_manual = 1.0       # Cận dưới (None -> dùng min(x) từ file)
b_manual = 12.845    # Cận trên (None -> dùng max(x) từ file)

# Parameters for Convergence Loop
EPSILON = 1e-6
MAX_ITER = 15

# --- Define g(f, x) ---
def g(f, x):
    """
    Hàm cần tính tích phân.
    """
    return f

# --- Execution ---

def run():
    # 1. Load Data
    try:
        x_raw, y_raw = load_data(INPUT_PATH, X_COL, Y_COL)
    except Exception as e:
        console.print(f"[bold red]Failed to load data:[/bold red] {e}")
        return

    a = a_manual if a_manual is not None else np.min(x_raw)
    b = b_manual if b_manual is not None else np.max(x_raw)
    
    # Calculate initial N based on data density
    mask = (x_raw >= a - 1e-9) & (x_raw <= b + 1e-9)
    count = np.sum(mask)
    if count < 2: count = 2
    N_START = count - 1
    h_initial = (b - a) / N_START
    
    if N_START < 2: N_START = 2

    # 2. Output Directory
    method_name = "Integration_Simpson"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Print Problem Info
    info_text = (
        f"File dữ liệu: {CSV_FILE}\n"
        f"Cột x: {X_COL}, Cột y: {Y_COL}\n"
        f"Miền tích phân: [{a}, {b}]\n"
        f"Phương pháp: Simpson 1/3\n"
        f"Hàm g(f, x):\n{inspect.getsource(g).strip()}\n"
        f"Epsilon: {EPSILON}"
    )
    console.print(Panel(info_text, title="Đề bài", style="bold cyan", border_style="cyan"))

    # 4. Print Formula
    formula_text = (
        "Công thức Simpson 1/3:\n"
        "  I ≈ (h/3) * [ g(x_0) + 4*(Odd Terms) + 2*(Even Terms) + g(x_n) ]\n\n"
        "Sai số lý thuyết (Big O):\n"
        "  Error = O(h^4) (Chính xác bậc 4)\n"
        "  R_n ≈ - ((b-a) * h^4 / 180) * g^{(4)}(\\xi)"
    )
    console.print(Panel(formula_text, title="Công thức & Sai số", border_style="green"))

    # Explicit Formula (User Request)
    explicit_formula = (
        "Công thức tổng quát (Simpson's 1/3 Rule):\n"
        "I ≈ (h/3) * [g(x_0) + 4*sum(g_odd) + 2*sum(g_even) + g(x_N)]\n\n"
        "Sai số (Big O):\n"
        "  O(h^4) (Bậc chính xác 4)"
    )
    console.print(Panel(explicit_formula, title="Công thức Chính xác", border_style="cyan"))

    # 5. Call Integration Method (It handles the loop)
    result_dict = simpson_integration(x_raw, y_raw, a, b, g=g, epsilon=EPSILON)
    
    # Check for N adjustments in logs and notify user
    logs = result_dict.get('computation_process', [])
    warnings = [line for line in logs if "Điều chỉnh N" in line or "N ban đầu" in line or "Cảnh báo:" in line]
    if warnings:
        warning_msg = "\n".join(warnings)
        console.print(Panel(f"[bold yellow]Cảnh báo Điều chỉnh N:[/bold yellow]\n{warning_msg}", border_style="yellow"))
    
    # 6. Print Detailed Table (User Request: FIRST)
    detailed_table = result_dict['intermediate_values'].get('initial_detailed_table', [])
    common_factor = result_dict['intermediate_values'].get('initial_common_factor', 1.0)
    
    if detailed_table:
        N_first = len(detailed_table) - 1
        console.print(Panel(
            f"I ≈ (Common Factor) * Sum(C_i * g_i)\n"
            f"Common Factor = h/3 = {common_factor:.9f}", 
            title=f"Bảng chi tiết (Iteration 1, N={N_first})", 
            border_style="yellow"
        ))
        
        dt_table = Table()
        dt_table.add_column("i", justify="center")
        dt_table.add_column("x_i", justify="right")
        dt_table.add_column("f(x_i)", justify="right")
        dt_table.add_column("g(x_i)", justify="right")
        dt_table.add_column("C_i ", justify="right", style="cyan")
        dt_table.add_column("Term ", justify="right", style="green")
        
        show_full = len(detailed_table) < 20
        display_rows = detailed_table if show_full else detailed_table[:6] + detailed_table[-6:]
        
        for row in display_rows:
            dt_table.add_row(
                str(row['i']),
                f"{row['x']:.6f}",
                f"{row['f']:.6f}",
                f"{row['g']:.6f}",
                f"{row['C']:.1f}",
                f"{row['term']:.6f}"
            )
            
        if not show_full:
             console.print(Panel(f"Bảng quá dài ({len(detailed_table)} dòng). Hiển thị 6 dòng đầu và cuối.", style="yellow"))

        console.print(dt_table)

    # 7. Display Iteration History
    table = Table(title="Bảng lặp (Doubling N) - Calculated by Method")
    table.add_column("Iter", justify="right", style="cyan", no_wrap=True)
    table.add_column("N", justify="right", style="magenta")
    table.add_column("h", justify="right", style="green")
    table.add_column("Result (I)", justify="right", style="yellow")
    table.add_column("Error Est.", justify="right", style="red")
    
    history = result_dict['intermediate_values']['iteration_history']
    
    history_data = [] # For CSV (kept for later use)

    for step in history:
        iter_num = str(step['iter'])
        N_step = str(step['N'])
        h_step = f"{step['h']:.6f}"
        res_step = f"{step['result']:.8f}"
        err_step = f"{step['error']:.8e}" if step['error'] is not None else "-"
        
        table.add_row(iter_num, N_step, h_step, res_step, err_step)
        
        # Populate history_data for CSV export
        history_data.append({
            "Iter": step['iter'], "N": step['N'], "h": step['h'], "Result": step['result'], "Error": step['error'] if step['error'] is not None else np.nan
        })
        
    console.print(table)
    
    if history[-1]['error'] is not None and history[-1]['error'] < EPSILON:
        console.print(f"\n[bold green]Hội tụ sau {len(history)} vòng lặp.[/bold green]")
    else:
        console.print("\n[bold red]Dừng lại (có thể chưa hội tụ hoặc đạt giới hạn lặp).[/bold red]")
    
    # Final h
    h = result_dict.get('h', (b-a)/history[-1]['N'])

    # 8. Save Results
    txt_filename = os.path.join(output_dir, f"{method_name}_report.txt")
    console.save_text(txt_filename)
    console.print(f"[bold blue]Đã lưu báo cáo vào: {txt_filename}[/bold blue]")
    
    df_res = pd.DataFrame(history_data)
    csv_filename = os.path.join(output_dir, f"{method_name}_convergence.csv")
    df_res.to_csv(csv_filename, index=False)
    console.print(f"[bold blue]Đã lưu bảng lặp vào: {csv_filename}[/bold blue]")

if __name__ == "__main__":
    run()
