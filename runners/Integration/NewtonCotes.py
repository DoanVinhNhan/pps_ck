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

from methods.Integration.Integration_NewtonCotes import newton_cotes_integration
from methods.utils import load_data

console = Console(record=True)

# --- Configuration ---
CSV_FILE = "20251GKtest2.csv"
INPUT_PATH = os.path.join(os.path.dirname(__file__), '../..', CSV_FILE)
X_COL = "x_i"
Y_COL = "y_i"
n = 6

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
    mask = (x_raw >= a - 1e-9) & (x_raw <= b + 1e-9)
    count = np.sum(mask)
    if count < 2: count = 2
    N_START = count - 1
    h_initial = (b - a) / N_START

    if N_START < 4: N_START = 4 # Ensure minimum samples for NC

    # 2. Output Directory
    method_name = "Integration_NewtonCotes"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Print Problem Info
    info_text = (
        f"File dữ liệu: {CSV_FILE}\n"
        f"Cột x: {X_COL}, Cột y: {Y_COL}\n"
        f"Miền tích phân: [{a}, {b}]\n"
        f"Phương pháp: Newton-Cotes (General)\n"
        f"Hàm g(f, x):\n{inspect.getsource(g).strip()}\n"
        f"Epsilon: {EPSILON}"
    )
    console.print(Panel(info_text, title="Đề bài", style="bold cyan", border_style="cyan"))

    # 4. Print Formula
    # Since specific order isn't fixed here, we print general info or assume e.g. 3/8 rule or higher
    formula_text = (
        "Công thức Newton-Cotes:\n"
        "  Dạng tổng quát: I ≈ sum(C_i * g(x_i))\n"
        "  Bao gồm các phương pháp như Trapezoidal (n=1), Simpson (n=2), 3/8 Rule (n=3), vv.\n\n"
        "Sai số lý thuyết (Big O):\n"
        "  Phụ thuộc vào bậc n (số điểm).\n"
        "  Với n lẻ, bậc chính xác là n.\n"
        "  Với n chẵn, bậc chính xác là n+1.\n"
        "Đánh giá sai số lưới thưa (khi không lặp):\n"
        "  Error ≈ |I_h - I_{kh}| / (k^p - 1)\n"
        "  (p=6 cho n=4,5; p=8 cho n=6)"\

    )
    console.print(Panel(formula_text, title="Công thức & Sai số", border_style="green"))

    # Explicit Formula & Coefficients (User Request)
    # Using 'g' instead of 'y' as requested
    coefficients_info = (
        "Công thức Chính xác (Newton-Cotes):\n"
        "I ≈ C * h * sum(w_i * g_i)\n\n"
        "Các hệ số C_i (Weights) cho từng bậc n:\n\n"
        "1. n=4 (Boole's Rule):\n"
        "   Factor: 2h/45\n"
        "   Weights Pattern: [7, 32, 12, 32, 7]\n\n"
        "2. n=5:\n"
        "   Factor: 5h/288\n"
        "   Weights Pattern: [19, 75, 50, 50, 75, 19]\n\n"
        "3. n=6:\n"
        "   Factor: h/140\n"
        "   Weights Pattern: [41, 216, 27, 272, 27, 216, 41]\n\n"
        "Sai số (Big O):\n"
        "   - Với n lẻ (odd): Độ chính xác bậc n (O(h^{n+1}) local?)\n"
        "     User Rule: Bậc chính xác = n\n"
        "   - Với n chẵn (even): Độ chính xác bậc n+1\n"
        "     User Rule: Bậc chính xác = n+1"
    )
    console.print(Panel(coefficients_info, title="Hệ số Newton-Cotes (n=4,5,6)", border_style="cyan"))

    # 5. Call Integration Method (It handles the loop)
    result_dict = newton_cotes_integration(x_raw, y_raw, a=a, b=b, g=g, epsilon=EPSILON, degree=n)
    
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
            f"Common Factor = {common_factor:.9f}", 
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
    
    history_data = [] # For CSV

    for step in history:
        iter_num = str(step['iter'])
        N_step = str(step['N'])
        h_step = f"{step['h']:.6f}"
        res_step = f"{step['result']:.8f}"
        err_step = f"{step['error']:.8e}" if step['error'] is not None else "-"
        
        table.add_row(iter_num, N_step, h_step, res_step, err_step)
        
        history_data.append({
            "Iter": step['iter'], "N": step['N'], "h": step['h'], "Result": step['result'], "Error": step['error'] if step['error'] is not None else np.nan
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
            console.print(f"[bold yellow]Kết quả tích phân (I_h):[/bold yellow] {final_I:.9f}")
            console.print(f"[bold red]Sai số ước lượng:[/bold red] {final_err:.9e}")
        else:
             console.print("[yellow]Không tìm thấy thông tin đánh giá lưới thưa (có thể do N quá nhỏ hoặc không tìm được k).[/yellow]")
    
    # Final h
    h = result_dict.get('h', (b-a)/history[-1]['N'])

    # 6. Save Results
    txt_filename = os.path.join(output_dir, f"{method_name}_report.txt")
    console.save_text(txt_filename)
    console.print(f"[bold blue]Đã lưu báo cáo vào: {txt_filename}[/bold blue]")
    
    df_res = pd.DataFrame(history_data)
    csv_filename = os.path.join(output_dir, f"{method_name}_convergence.csv")
    df_res.to_csv(csv_filename, index=False)
    console.print(f"[bold blue]Đã lưu bảng lặp vào: {csv_filename}[/bold blue]")

if __name__ == "__main__":
    run()
