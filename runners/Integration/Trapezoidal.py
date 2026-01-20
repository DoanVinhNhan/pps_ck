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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from methods.Integration.Integration_Trapezoidal import trapezoidal_integration
from methods.utils import load_data

console = Console(record=True)

# --- Configuration ---
CSV_FILE = "20251GKtest2.csv"
INPUT_PATH = os.path.join(os.path.dirname(__file__), '../..', CSV_FILE)
X_COL = "x_i"
Y_COL = "y_i"

# --- Input Parameters ---
a_manual = 1.0       # Cận dưới (None -> dùng min(x) từ file)
b_manual = 12.845    # Cận trên (None -> dùng max(x) từ file)

# Parameters for Convergence Loop
EPSILON = None
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
    # Estimate h from data: (max - min) / (count - 1)
    mask = (x_raw >= a - 1e-9) & (x_raw <= b + 1e-9)
    count = np.sum(mask)
    if count < 2: count = 2
    N_START = count - 1
    h_initial = (b - a) / N_START

    if N_START < 1: N_START = 1

    # 2. Output Directory
    method_name = "Integration_Trapezoidal"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Print Problem Info
    info_text = (
        f"File dữ liệu: {CSV_FILE}\n"
        f"Cột x: {X_COL}, Cột y: {Y_COL}\n"
        f"Miền tích phân: [{a}, {b}]\n"
        f"Phương pháp: Hình thang (Trapezoidal)\n"
        f"Hàm g(f, x):\n{inspect.getsource(g).strip()}\n"
        f"Epsilon: {EPSILON}"
    )
    console.print(Panel(info_text, title="Đề bài", style="bold cyan", border_style="cyan"))

    # 4. Print Formula
    formula_text = (
        "Công thức Hình thang mở rộng:\n"
        "  I ≈ (h/2) * [ g(x_0) + 2*sum(g(x_i)) + g(x_n) ]\n\n"
        "Sai số lý thuyết (Big O):\n"
        "  Error = O(h^2) (Chính xác bậc 2)\n"
        "  R_n ≈ - ((b-a) * h^2 / 12) * g''(\\xi)\n"
        "Đánh giá sai số lưới thưa (khi không lặp):\n"
        "  Error ≈ |I_h - I_{kh}| / (k^2 - 1)"\

    )
    console.print(Panel(formula_text, title="Công thức & Sai số", border_style="green"))

    # Explicit Formula (User Request)
    explicit_formula = (
        "Công thức tổng quát (Trapezoidal Rule):\n"
        "I ≈ (h/2) * [g(x_0) + 2*g(x_1) + ... + 2*g(x_{N-1}) + g(x_N)]\n\n"
        "Sai số (Big O):\n"
        "  O(h^2) (Bậc chính xác 2)"
    )
    console.print(Panel(explicit_formula, title="Công thức Chính xác", border_style="cyan"))

    # 5. Call Integration Method (It handles the loop)
    # The method will start with N_START (derived from data) and double it until EPSILON is met.
    # Note: We pass N_initial indirectly via the data density or we might need to reset 'h_initial' if method calculates it.
    # Actually, the methods logic (as updated) calculates n_initial from data inside. 
    # So we just pass x_raw, y_raw (interpolated nodes? No, the method interpolates internally if we pass raw nodes? 
    # Wait, the method code uses its OWN interpolation to refine grid.
    # We should pass the RAW nodes to the method, and let IT handle grid refinement.
    # BUT, the runner currently does interpolation THEN passes nodes. 
    # The method signature is `(x_nodes, y_nodes, ...)`
    # The method uses these nodes to interpolate and generate its OWN grid.
    # So we should pass the RAW data.
    
    result_dict = trapezoidal_integration(x_raw, y_raw, a, b, g=g, epsilon=EPSILON)
    
    
    # 6. Print Detailed Table (User Request: FIRST)
    detailed_table = result_dict['intermediate_values'].get('initial_detailed_table', [])
    common_factor = result_dict['intermediate_values'].get('initial_common_factor', 1.0)
    
    if detailed_table:
        N_first = len(detailed_table) - 1
        console.print(Panel(
            f"I ≈ (Common Factor) * Sum(C_i * g_i)\n"
            f"Common Factor = h/2 = {common_factor:.11g}", 
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
        # Limit rows for display if too large
        # User requested: 6 first, 6 last
        show_full = len(detailed_table) < 20
        
        display_rows = detailed_table if show_full else detailed_table[:6] + detailed_table[-6:]
        
        for row in display_rows:
            dt_table.add_row(
                str(row['i']),
                f"{row['x']:.8g}",
                f"{row['f']:.8g}",
                f"{row['g']:.8g}",
                f"{row['C']:.6g}", # Raw weight is usually int/simple float
                f"{row['term']:.8g}"
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
    
    history_data = [] # For CSV

    for step in history:
        iteration = step['iter']
        N_step = step['N']
        h_step = step['h']
        I_val = step['result']
        err_val = step['error']
        
        iter_num = str(iteration)
        N_step_str = str(N_step)
        h_step_str = f"{h_step:.8g}"
        res_step_str = f"{I_val:.10g}"
        err_step_str = f"{err_val:.9g}" if err_val is not None else "-"
        
        table.add_row(iter_num, N_step_str, h_step_str, res_step_str, err_step_str)
        
        history_data.append({
            "Iter": iteration, "N": N_step, "h": h_step, "Result": I_val, "Error": err_val if err_val is not None else np.nan
        })

    
    if EPSILON is not None:
        console.print(table)
        
        if history[-1]['error'] is not None and history[-1]['error'] < EPSILON:
             console.print(f"\n[bold green]Hội tụ sau {len(history)} vòng lặp.[/bold green]")
        else:
             console.print("\n[bold red]Dừng lại (có thể chưa hội tụ hoặc đạt giới hạn lặp).[/bold red]")
    else:
        # Chế độ tính 1 lần (Sparse Grid) - extract info from logs
        console.print("\n[bold cyan]--- Kết quả Đánh giá Sai số Lưới thưa ---[/bold cyan]")
        logs = result_dict.get('computation_process', [])
        
        # Extract specific lines
        sparse_lines = [line for line in logs if "I_dense" in line or "I_sparse" in line or "Sai số ước lượng" in line or "lưới thưa k=" in line]
        
        if sparse_lines:
            sparse_text = "\n".join(sparse_lines).replace("-> ", "")
            console.print(Panel(sparse_text, title="Chi tiết Đánh giá", border_style="green"))
            
            # Highlight final result and error
            final_I = result_dict['result']
            final_err = result_dict['error_estimate']
            console.print(f"[bold yellow]Kết quả tích phân (I_h):[/bold yellow] {final_I:.11g}")
            console.print(f"[bold red]Sai số ước lượng:[/bold red] {final_err:.10g}")
        else:
             console.print("[yellow]Không tìm thấy thông tin đánh giá lưới thưa (có thể do N quá nhỏ hoặc không tìm được k).[/yellow]")
    
    # Final h
    h = result_dict.get('h', (b-a)/history[-1]['N'])
    
    # 6. Save Results
    # Save text report
    txt_filename = os.path.join(output_dir, f"{method_name}_report.txt")
    console.save_text(txt_filename)
    console.print(f"[bold blue]Đã lưu báo cáo vào: {txt_filename}[/bold blue]")
    
    # Save CSV history
    df_res = pd.DataFrame(history_data)
    csv_filename = os.path.join(output_dir, f"{method_name}_convergence.csv")
    df_res.to_csv(csv_filename, index=False)
    console.print(f"[bold blue]Đã lưu bảng lặp vào: {csv_filename}[/bold blue]")

if __name__ == "__main__":
    run()
