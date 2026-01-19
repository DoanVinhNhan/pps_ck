import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.ODE_1D_RK4 import solve_ode_1d_rk4

# ==========================================
# 1. INPUT PARAMETERS
# ==========================================
def f(t, x):
    # Ví dụ: x' = x - t^2 + 1
    return x - t**2 + 1

t0 = 0.0        # Thời điểm bắt đầu
x0 = 0.5        # Giá trị ban đầu x(t0)
h = 0.1         # Bước nhảy
T = 2.0         # Thời điểm kết thúc

# ==========================================
# 2. EXECUTION
# ==========================================
result = solve_ode_1d_rk4(f, t0, x0, h, T)
t_values = result['t']
x_values = result['x']

# ==========================================
# 3. TERMINAL OUTPUT (RICH)
# ==========================================
console = Console()

# 3.1. In đề bài
input_desc = (
    f"Hàm số f(t, x): x - t^2 + 1\n"
    f"Khoảng [t0, T]: [{t0}, {T}]\n"
    f"Bước nhảy h: {h}\n"
    f"Giá trị đầu x0: {x0}"
)
console.print(Panel(input_desc, title="Đề bài", style="bold cyan"))

# 3.2. In công thức
console.print("\n[bold yellow]Áp dụng ODE_1D_RK4 Ta có:[/bold yellow]")
formula_text = (
    "k₁ = h * f(t_n, x_n)\n"
    "k₂ = h * f(t_n + 0.5h, x_n + 0.5k₁)\n"
    "k₃ = h * f(t_n + 0.5h, x_n + 0.5k₂)\n"
    "k₄ = h * f(t_n + h, x_n + k₃)\n"
    "x_{n+1} = x_n + (k₁ + 2k₂ + 2k₃ + k₄) / 6"
)
console.print(Panel(formula_text, style="italic green"))

# 3.3. In bảng giá trị
table = Table(title="Bảng giá trị")
table.add_column("Iteration", justify="center")
table.add_column("t", justify="right")
table.add_column("x", justify="right")
table.add_column("k1", justify="right")
table.add_column("k2", justify="right")
table.add_column("k3", justify="right")
table.add_column("k4", justify="right")

# Chuẩn bị dữ liệu bảng (tính lại k để hiển thị)
rows = []
num_points = len(t_values)

for i in range(num_points):
    t_curr = t_values[i]
    x_curr = x_values[i]
    
    if i < num_points - 1:
        # Tính lại các hệ số k chỉ để hiển thị
        k1 = h * f(t_curr, x_curr)
        k2 = h * f(t_curr + 0.5 * h, x_curr + 0.5 * k1)
        k3 = h * f(t_curr + 0.5 * h, x_curr + 0.5 * k2)
        k4 = h * f(t_curr + h, x_curr + k3)
        row_data = [
            str(i), f"{t_curr:.4f}", f"{x_curr:.6f}",
            f"{k1:.6f}", f"{k2:.6f}", f"{k3:.6f}", f"{k4:.6f}"
        ]
    else:
        # Dòng cuối cùng không tính k cho bước tiếp theo
        row_data = [str(i), f"{t_curr:.4f}", f"{x_curr:.6f}", "-", "-", "-", "-"]
    
    rows.append(row_data)

# Logic rút gọn bảng (5 đầu, 5 cuối)
if len(rows) > 10:
    for row in rows[:5]:
        table.add_row(*row)
    table.add_row("...", "...", "...", "...", "...", "...", "...")
    for row in rows[-5:]:
        table.add_row(*row)
else:
    for row in rows:
        table.add_row(*row)

console.print(table)

# ==========================================
# 4. SAVE CSV
# ==========================================
df = pd.DataFrame({
    'x': t_values,
    'y': x_values
})
df.to_csv("ODE_1D_RK4.csv", index=False)
console.print(f"\n[green]Đã xuất file kết quả: ODE_1D_RK4.csv[/green]")

# ==========================================
# 5. PLOT GRAPH
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, 'o-', label='RK4 Approximation', color='blue')
plt.title(f"Solution of ODE using RK4 Method (h={h})")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("graph_ODE_1D_RK4.png")
console.print(f"[green]Đã lưu đồ thị: graph_ODE_1D_RK4.png[/green]")
plt.show()