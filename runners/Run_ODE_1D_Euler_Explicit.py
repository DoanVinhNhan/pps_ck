import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import thuật toán từ file nguồn
# Import thuật toán từ file nguồn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.ODE_1D_Euler_Explicit import solve_ode_euler_forward_1d

# ==========================================
# 1. INPUT PARAMETERS
# ==========================================
def f_ode(t, x):
    """Hàm f(t, x) cho phương trình x' = f(t, x)"""
    # Ví dụ: x' = x - t^2 + 1
    return x - t**2 + 1

FUNC_STR = "x' = x - t^2 + 1"
T0 = 0.0      # Thời điểm bắt đầu
X0 = 0.5      # Giá trị ban đầu x(t0)
H = 0.2       # Bước nhảy
T_END = 2.0   # Thời điểm kết thúc

# ==========================================
# 2. EXECUTION
# ==========================================
result = solve_ode_euler_forward_1d(f_ode, T0, X0, H, T_END)
t_values = result['t']
x_values = result['x']

# ==========================================
# 2. MAIN RUNNER
# ==========================================
def main():
    console = Console(record=True)
    method_name = "ODE_1D_Euler_Explicit"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3.1. In đề bài
    console.print(Panel("[bold cyan]Đề bài[/bold cyan]", expand=False, border_style="cyan"))
    console.print(f"• Phương trình: [yellow]{FUNC_STR}[/yellow]")
    console.print(f"• Khoảng tính toán: [{T0}, {T_END}]")
    console.print(f"• Bước nhảy (h): {H}")
    console.print(f"• Điều kiện ban đầu: x({T0}) = {X0}")
    console.print("")

    # 3.2. Tiêu đề phương pháp
    console.print("[bold green]Áp dụng ODE_1D_Euler_Explicit Ta có:[/bold green]")

    # 3.3. Công thức
    console.print(Panel("x_{i+1} = x_i + h * f(t_i, x_i)", title="Công thức Euler Hiện", expand=False))
    console.print("")

    # 3.4. Bảng giá trị
    console.print("[bold magenta]Bảng giá trị[/bold magenta]")
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Iteration", justify="center", style="cyan")
    table.add_column("t (Time)", justify="right")
    table.add_column("x (Value)", justify="right", style="green")
    table.add_column("Slope k1=f(t,x)", justify="right", style="yellow")

    n = len(t_values)
    # Tính slope để hiển thị trong bảng
    slopes = [f_ode(t, x) for t, x in zip(t_values, x_values)]

    for i in range(n):
        # Logic hiển thị: 5 dòng đầu và 5 dòng cuối
        if n > 10 and 5 <= i < n - 5:
            if i == 5:
                table.add_row("...", "...", "...", "...")
            continue

        t_val = t_values[i]
        x_val = x_values[i]
        slope_val = slopes[i]
        
        table.add_row(
            str(i), 
            f"{t_val:.6f}", 
            f"{x_val:.6f}", 
            f"{slope_val:.6f}"
        )

    console.print(table)

    # ==========================================
    # 4. CSV OUTPUT
    # ==========================================
    # Mapping: t -> x (biến độc lập), x -> y (biến phụ thuộc) theo yêu cầu
    df = pd.DataFrame({
        'x': t_values,
        'y': x_values
    })
    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    df.to_csv(csv_filename, index=False)
    console.print(f"\n[bold green]Đã xuất file kết quả: {csv_filename}[/bold green]")

    # ==========================================
    # 5. GRAPH PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, x_values, marker='o', linestyle='-', color='b', label='Euler Explicit')
    plt.title(f"Giải phương trình vi phân bằng Euler hiện (h={H})")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    
    img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
    plt.savefig(img_filename)
    console.print(f"[bold green]Đã lưu đồ thị: {img_filename}[/bold green]")
    
    # Save Text Report
    txt_filename = os.path.join(output_dir, f"{method_name}.txt")
    console.save_text(txt_filename)
    console.print(f"[bold green]Đã lưu báo cáo text vào file: {txt_filename}[/bold green]")

    plt.show()

if __name__ == "__main__":
    main()