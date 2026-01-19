import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods.ODE_1D_Euler_Explicit import solve_ode_euler_forward_1d

# --- INPUT PARAMETERS ---
def f_ode(t, x):
    # Ví dụ: x' = x - t, với x(0) = 1
    return x - t

t_start = 0.0       # Thời điểm bắt đầu (t0)
x_start = 1.0       # Giá trị ban đầu (x0)
step_size = 0.1     # Bước nhảy (h)
t_end = 1.0         # Thời điểm kết thúc (T)

# --- EXECUTION ---
result = solve_ode_euler_forward_1d(f_ode, t_start, x_start, step_size, t_end)
t_values = result['t']
x_values = result['x']

# --- TERMINAL OUTPUT (RICH) ---
console = Console()

# 1. In tiêu đề Đề bài
input_desc = (
    f"Hàm số: x' = f(t, x) = x - t\n"
    f"Khoảng tính toán: [{t_start}, {t_end}]\n"
    f"Bước nhảy h: {step_size}\n"
    f"Điều kiện đầu: x({t_start}) = {x_start}"
)
console.print(Panel(input_desc, title="Đề bài", style="bold cyan"))

# 2. In tiêu đề Áp dụng
console.print("\n[bold green]Áp dụng ODE_1D_Euler_Explicit Ta có:[/bold green]")

# 3. In công thức
formula_text = "x_{i+1} = x_i + h * f(t_i, x_i)"
console.print(f"\n{formula_text}\n")

# 4. In Bảng giá trị
table = Table(title="Bảng giá trị")
table.add_column("Iteration", justify="center")
table.add_column("t (Biến độc lập)", justify="right")
table.add_column("x (Biến phụ thuộc)", justify="right")
table.add_column("f(t, x) (Slope)", justify="right")

num_points = len(t_values)

def get_row_data(idx):
    t_val = t_values[idx]
    x_val = x_values[idx]
    # Tính slope để hiển thị (trừ điểm cuối cùng không dùng để tính tiếp)
    slope = f_ode(t_val, x_val) if idx < num_points - 1 else float('nan')
    slope_str = f"{slope:.6f}" if idx < num_points - 1 else "-"
    return str(idx), f"{t_val:.6f}", f"{x_val:.6f}", slope_str

if num_points <= 10:
    for i in range(num_points):
        table.add_row(*get_row_data(i))
else:
    for i in range(5):
        table.add_row(*get_row_data(i))
    table.add_row("...", "...", "...", "...")
    for i in range(num_points - 5, num_points):
        table.add_row(*get_row_data(i))

console.print(table)

# --- CSV OUTPUT ---
# Mapping: t -> x (independent), x -> y (dependent)
df = pd.DataFrame({
    'x': t_values,
    'y': x_values
})
df.to_csv('ODE_1D_Euler_Explicit.csv', index=False)
console.print(f"\n[yellow]Đã xuất file CSV: ODE_1D_Euler_Explicit.csv[/yellow]")

# --- GRAPH PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, marker='o', linestyle='-', color='b', label='Euler Explicit')
plt.title(f"Nghiệm gần đúng ODE bằng phương pháp Euler Hiện (h={step_size})")
plt.xlabel("Thời gian (t)")
plt.ylabel("Nghiệm x(t)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('graph_ODE_1D_Euler_Explicit.png')
console.print(f"[yellow]Đã lưu đồ thị: graph_ODE_1D_Euler_Explicit.png[/yellow]")