import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import algorithm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from methods.ODE.ODE_1D_ABs_AMs import solve_ode_1d_ab_am

# --- Input Parameters ---
def f(t, x):
    # Ví dụ: x' = x - t^2 + 1
    return x - t**2 + 1

t0 = 0.0        # Thời điểm bắt đầu
x0 = 0.5        # Giá trị ban đầu x(t0)
h = 0.1         # Bước nhảy
T = 2.0         # Thời điểm kết thúc

# --- Execution ---
t_values, x_values = solve_ode_1d_ab_am(f, t0, x0, h, T)

# --- Terminal Output ---
console = Console(record=True)
import os
method_name = "ODE_1D_ABs_AMs"
output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
os.makedirs(output_dir, exist_ok=True)

# 1. In tiêu đề Đề bài
input_info = f"Hàm số f(t, x): x - t^2 + 1\nKhoảng thời gian: [{t0}, {T}]\nBước nhảy h: {h}\nGiá trị ban đầu x({t0}): {x0}"
console.print(Panel(input_info, title="Đề bài", style="bold cyan", border_style="cyan"))

# 2. In tiêu đề Áp dụng
console.print("\n[bold yellow]Áp dụng ODE_1D_ABs_AMs Ta có:[/bold yellow]")

# 3. In Công thức
formula_text = (
    "Predictor (AB4):\n"
    "  x_{n+1}^* = x_n + (h/24) * (55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})\n\n"
    "Corrector (AM4):\n"
    "  x_{n+1} = x_n + (h/24) * (9f(t_{n+1}, x_{n+1}^*) + 19f_n - 5f_{n-1} + f_{n-2})"
)
console.print(Panel(formula_text, title="Công thức Adams-Bashforth-Moulton", border_style="green"))

# 4. In Bảng giá trị
table = Table(title="Bảng giá trị")
table.add_column("Iteration", justify="center", style="magenta")
table.add_column("t (Biến độc lập)", justify="right", style="cyan")
table.add_column("x (Nghiệm)", justify="right", style="green")

num_points = len(t_values)
for i in range(num_points):
    if i < 5 or i >= num_points - 5:
        table.add_row(
            str(i), 
            f"{t_values[i]:.4f}", 
            f"{x_values[i]:.6f}"
        )
    elif i == 5:
        table.add_row("...", "...", "...")

console.print(table)

# --- CSV Output ---
# Map t -> x, x -> y cho file CSV theo yêu cầu
df = pd.DataFrame({
    'x': t_values,
    'y': x_values
})
csv_filename = os.path.join(output_dir, f"{method_name}.csv")
df.to_csv(csv_filename, index=False)
console.print(f"\n[bold blue]Đã lưu kết quả vào file: {csv_filename}[/bold blue]")

# --- Graph Output ---
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, 'r-o', label='AB4-AM4 Solution', markersize=4)
plt.title(f"Nghiệm phương trình vi phân (Adams-Bashforth-Moulton)\nh={h}, [t0, T]=[{t0}, {T}]")
plt.xlabel("Thời gian (t)")
plt.ylabel("Giá trị nghiệm x(t)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
plt.savefig(img_filename)
console.print(f"[bold blue]Đã lưu đồ thị vào file: {img_filename}[/bold blue]")

# Save Text Report
txt_filename = os.path.join(output_dir, f"{method_name}.txt")
console.save_text(txt_filename)
console.print(f"[bold blue]Đã lưu báo cáo text vào file: {txt_filename}[/bold blue]")