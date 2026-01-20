import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import os
import inspect

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from methods.Function_Integration.Integration_Simpson_Func import simpson_integration_func

console = Console(record=True)

# --- Configuration ---
# Định nghĩa hàm f(x) cần tính tích phân
def f(x):
    # Ví dụ: f(x) = x * sin(x)
    # Hoặc f(x) = e^x
    return np.exp(x)

# Tham số tích phân
A = 0.0
B = 1.0

# Tham số sai số (Nếu None -> chạy 1 lần)
EPSILON = 1e-6
# EPSILON = None

def run():
    # 1. Output Setup
    method_name = "Integration_Simpson_Func"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Print Info
    info = (
        f"Phương pháp: Simpson 1/3 (Function Mode)\n"
        f"Hàm f(x):\n{inspect.getsource(f).strip()}\n"
        f"Miền tích phân: [{A}, {B}]\n"
        f"Epsilon: {EPSILON}"
    )
    console.print(Panel(info, title="Đề bài", style="bold cyan"))
    
    # 3. Calculate
    # result_dict = simpson_integration_func(f, A, B, epsilon=EPSILON, N_start=2)
    # Start with N=2 to ensure Simpson requirements
    result_dict = simpson_integration_func(f, A, B, epsilon=EPSILON, N_start=2)
    
    # 4. Display Initial Detailed Table
    if 'intermediate_values' in result_dict:
        detailed = result_dict['intermediate_values'].get('initial_detailed_table', [])
        common_f = result_dict['intermediate_values'].get('initial_common_factor', 0)
        
        if detailed:
            console.print(Panel(f"Bảng chi tiết (Lần lặp 1)\nCommon Factor = {common_f:.9f}", title="Chi tiết", border_style="yellow"))
            
            dt = Table()
            dt.add_column("i")
            dt.add_column("x")
            dt.add_column("f(x)")
            dt.add_column("Coeff")
            dt.add_column("Term")
            
            # Show limited rows
            display = detailed if len(detailed) < 15 else detailed[:5] + detailed[-5:]
            
            for row in display:
                dt.add_row(
                    str(row['i']),
                    f"{row['x']:.6f}",
                    f"{row['f']:.6f}",
                    f"{row['C']:.1f}",
                    f"{row['term']:.6f}"
                )
            if len(detailed) >= 15:
                console.print(f"[yellow]... {len(detailed)-10} rows hidden ...[/yellow]")
                
            console.print(dt)
            
    # 5. Iteration History
    hist = result_dict['intermediate_values'].get('iteration_history', [])
    t_hist = Table(title="Lịch sử Lặp")
    t_hist.add_column("Iter")
    t_hist.add_column("N")
    t_hist.add_column("h")
    t_hist.add_column("Result")
    t_hist.add_column("Error")
    
    hist_data = []
    
    for step in hist:
        err_str = f"{step['error']:.2e}" if step['error'] else "-"
        t_hist.add_row(
            str(step['iter']), str(step['N']), f"{step['h']:.6f}", f"{step['result']:.9f}", err_str
        )
        hist_data.append(step)
        
    console.print(t_hist)
    
    # 6. Final Result
    final_res = result_dict['result']
    final_err = result_dict['error_estimate']
    console.print(Panel(
        f"Kết quả cuối cùng: {final_res:.9f}\nSai số ước lượng: {final_err:.2e}",
        title="KẾT QUẢ", style="bold green"
    ))
    
    # 7. Save
    df = pd.DataFrame(hist_data)
    csv_path = os.path.join(output_dir, "convergence.csv")
    df.to_csv(csv_path, index=False)
    console.save_text(os.path.join(output_dir, "report.txt"))
    console.print(f"Đã lưu kết quả tại: {output_dir}")

if __name__ == "__main__":
    run()
