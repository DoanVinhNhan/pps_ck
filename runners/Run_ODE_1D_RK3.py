import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import thuật toán từ file methods (Giả sử cấu trúc thư mục đã có)
# Import thuật toán từ file methods (Giả sử cấu trúc thư mục đã có)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.ODE_1D_RK3 import solve_ode_1d_rk3

# ==========================================
# 1. INPUT PARAMETERS
# ==========================================
def f(t, x):
    # Ví dụ: x' = x - t^2 + 1
    return x - t**2 + 1

t0 = 0.0        # Thời điểm bắt đầu
x0 = 0.5        # Giá trị ban đầu
h = 0.2         # Bước nhảy
T = 2.0         # Thời điểm kết thúc

# ==========================================
# 2. EXECUTION & CALCULATION
# ==========================================
# Gọi hàm giải thuật toán
result = solve_ode_1d_rk3(f, t0, x0, h, T)
t_res = result['t']
x_res = result['x']

# Tái tạo dữ liệu chi tiết cho bảng (tính lại k1, k2, k3 để hiển thị)
table_data = []
for i in range(len(t_res) - 1):
    ti = t_res[i]
    xi = x_res[i]
    hi = t_res[i+1] - ti # Bước nhảy thực tế (có thể thay đổi ở bước cuối)
    
    k1 = hi * f(ti, xi)
    k2 = hi * f(ti + hi/2, xi + k1/2)
    k3 = hi * f(ti + hi, xi - k1 + 2*k2)
    
    table_data.append({
        "iter": i + 1,
        "t": ti,
        "x": xi,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "x_next": x_res[i+1]
    })

# ==========================================
# 3. TERMINAL OUTPUT (RICH)
# ==========================================
console = Console(record=True)
import os
method_name = "ODE_1D_RK3"
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
os.makedirs(output_dir, exist_ok=True)

# 3.1. In Đề bài
input_info = f"""
Hàm số f(t, x): x - t^2 + 1
Khoảng thời gian: [{t0}, {T}]
Giá trị ban đầu x({t0}): {x0}
Bước nhảy h: {h}
"""
console.print(Panel(input_info.strip(), title="[bold cyan]Đề bài[/bold cyan]", expand=False))

# 3.2. In Tiêu đề áp dụng
console.print("\n[bold yellow]Áp dụng ODE_1D_RK3 Ta có:[/bold yellow]")

# 3.3. In Công thức
formula = """
k₁ = h * f(t, x)
k₂ = h * f(t + h/2, x + k₁/2)
k₃ = h * f(t + h, x - k₁ + 2k₂)
x_{n+1} = x_n + (1/6) * (k₁ + 4k₂ + k₃)
"""
console.print(Panel(formula.strip(), title="[bold green]Công thức RK3 (Kutta)[/bold green]", expand=False))

# 3.4. In Bảng giá trị
table = Table(title="Bảng giá trị chi tiết")
table.add_column("Iter", justify="center", style="cyan")
table.add_column("t", justify="right")
table.add_column("x", justify="right")
table.add_column("k1", justify="right", style="magenta")
table.add_column("k2", justify="right", style="magenta")
table.add_column("k3", justify="right", style="magenta")
table.add_column("x_next", justify="right", style="green")

# Logic hiển thị 5 dòng đầu và 5 dòng cuối
total_rows = len(table_data)
display_rows = []

if total_rows <= 10:
    display_rows = table_data
else:
    display_rows = table_data[:5] + [{"iter": "..."}] + table_data[-5:]

for row in display_rows:
    if row["iter"] == "...":
        table.add_row("...", "...", "...", "...", "...", "...", "...")
    else:
        table.add_row(
            str(row["iter"]),
            f"{row['t']:.4f}",
            f"{row['x']:.6f}",
            f"{row['k1']:.6f}",
            f"{row['k2']:.6f}",
            f"{row['k3']:.6f}",
            f"{row['x_next']:.6f}"
        )

console.print(table)

# ==========================================
# 4. CSV OUTPUT
# ==========================================
# Mapping: t -> x (biến độc lập), x -> y (biến phụ thuộc)
df = pd.DataFrame({
    'x': t_res,
    'y': x_res
})

csv_filename = os.path.join(output_dir, f"{method_name}.csv")
df.to_csv(csv_filename, index=False)
console.print(f"\n[bold green]Đã xuất file CSV: {csv_filename}[/bold green]")

# ==========================================
# 5. GRAPH PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(t_res, x_res, 'o-', label='RK3 Approximation', color='blue', markersize=4)
plt.title(f"Giải ODE bằng phương pháp RK3\n$x' = x - t^2 + 1, x({t0})={x0}, h={h}$")
plt.xlabel("Thời gian (t)")
plt.ylabel("Nghiệm x(t)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
plt.savefig(img_filename)
console.print(f"[bold green]Đã lưu đồ thị: {img_filename}[/bold green]")
plt.close()

# Save Text Report
txt_filename = os.path.join(output_dir, f"{method_name}.txt")
console.save_text(txt_filename)
console.print(f"[bold green]Đã lưu báo cáo text vào file: {txt_filename}[/bold green]")