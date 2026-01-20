import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import os
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from methods.Function_Integration.Integration_Trapezoidal_Func import trapezoidal_integration_func

console = Console(record=True)

# --- Configuration ---
def f(x):
    return np.exp(x)

A = 0.0
B = 1.0
EPSILON = 1e-6

def run():
    method_name = "Integration_Trapezoidal_Func"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    info = (
        f"Phương pháp: Trapezoidal (Function Mode)\n"
        f"Hàm f(x):\n{inspect.getsource(f).strip()}\n"
        f"Miền tích phân: [{A}, {B}]\n"
        f"Epsilon: {EPSILON}"
    )
    console.print(Panel(info, title="Đề bài", style="bold cyan"))
    
    result_dict = trapezoidal_integration_func(f, A, B, epsilon=EPSILON, N_start=1)
    
    # Detailed Table
    if 'intermediate_values' in result_dict:
        detailed = result_dict['intermediate_values'].get('initial_detailed_table', [])
        common_f = result_dict['intermediate_values'].get('initial_common_factor', 0)
        
        if detailed:
            console.print(Panel(f"Bảng chi tiết (Lần lặp 1)\nCommon Factor = {common_f:.11g}", title="Chi tiết", border_style="yellow"))
            dt = Table()
            dt.add_column("i")
            dt.add_column("x")
            dt.add_column("f(x)")
            dt.add_column("Coeff")
            dt.add_column("Term")
            
            display = detailed if len(detailed) < 15 else detailed[:5] + detailed[-5:]
            for row in display:
                dt.add_row(str(row['i']), f"{row['x']:.8g}", f"{row['f']:.8g}", f"{row['C']:.6g}", f"{row['term']:.8g}")
            if len(detailed) >= 15:
                console.print(f"[yellow]... {len(detailed)-10} rows hidden ...[/yellow]")
            console.print(dt)
            
    # History
    hist = result_dict['intermediate_values'].get('iteration_history', [])
    t_hist = Table(title="Lịch sử Lặp")
    t_hist.add_column("Iter")
    t_hist.add_column("N")
    t_hist.add_column("h")
    t_hist.add_column("Result")
    t_hist.add_column("Error")
    
    hist_data = []
    for step in hist:
        err_str = f"{step['error']:.4g}" if step['error'] else "-"
        t_hist.add_row(str(step['iter']), str(step['N']), f"{step['h']:.8g}", f"{step['result']:.11g}", err_str)
        hist_data.append(step)
    console.print(t_hist)
    
    final_res = result_dict['result']
    final_err = result_dict['error_estimate']
    console.print(Panel(f"Kết quả cuối cùng: {final_res:.11g}\nSai số ước lượng: {final_err:.4g}", title="KẾT QUẢ", style="bold green"))
    
    df = pd.DataFrame(hist_data)
    csv_path = os.path.join(output_dir, "convergence.csv")
    df.to_csv(csv_path, index=False)
    console.save_text(os.path.join(output_dir, "report.txt"))
    console.print(f"Đã lưu kết quả tại: {output_dir}")

if __name__ == "__main__":
    run()
