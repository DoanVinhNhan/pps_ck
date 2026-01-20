import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ==========================================
# 1. INPUT PARAMETERS
# ==========================================
def f(t, x, y):
    # dx/dt = x - y + t
    return x - y + t

def g(t, x, y):
    # dy/dt = x + y
    return x + y

t0 = 0.0        # Thời điểm bắt đầu
x0 = 1.0        # Giá trị ban đầu x(t0)
y0 = 0.0        # Giá trị ban đầu y(t0)
h = 0.1         # Bước nhảy
T = 1.0         # Thời điểm kết thúc

# Import thuật toán
try:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from methods.ODE.ODE_2D_RK3 import ode_2d_rk3
except ImportError:
    # Fallback nếu không tìm thấy module (để code chạy được demo)
    def ode_2d_rk3(f, g, t0, x0, y0, h, T):
        t_values, x_values, y_values = [t0], [x0], [y0]
        t_curr, x_curr, y_curr = t0, x0, y0
        num_steps = int(np.round((T - t0) / h))
        for _ in range(num_steps):
            k1 = h * f(t_curr, x_curr, y_curr)
            l1 = h * g(t_curr, x_curr, y_curr)
            k2 = h * f(t_curr + h/2, x_curr + k1/2, y_curr + l1/2)
            l2 = h * g(t_curr + h/2, x_curr + k1/2, y_curr + l1/2)
            k3 = h * f(t_curr + h, x_curr - k1 + 2*k2, y_curr - l1 + 2*l2)
            l3 = h * g(t_curr + h, x_curr - k1 + 2*k2, y_curr - l1 + 2*l2)
            x_next = x_curr + (1.0/6.0) * (k1 + 4*k2 + k3)
            y_next = y_curr + (1.0/6.0) * (l1 + 4*l2 + l3)
            t_next = t_curr + h
            t_values.append(t_next); x_values.append(x_next); y_values.append(y_next)
            t_curr, x_curr, y_curr = t_next, x_next, y_next
        return {"t": t_values, "x": x_values, "y": y_values}

# ==========================================
# 2. EXECUTION & RICH OUTPUT
# ==========================================
console = Console(record=True)

# Define output directory
method_name = "ODE_2D_RK3"
output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
os.makedirs(output_dir, exist_ok=True)

# Chạy thuật toán
result = ode_2d_rk3(f, g, t0, x0, y0, h, T)
t_res = result["t"]
x_res = result["x"]
y_res = result["y"]
num_steps = len(t_res) - 1

# 2.1 In Đề bài
input_info = f"""
[bold]Hệ phương trình:[/bold]
  dx/dt = f(t, x, y)
  dy/dt = g(t, x, y)

[bold]Tham số:[/bold]
  Khoảng t: [{t0}, {T}]
  Bước h: {h}
  Initial x0: {x0}
  Initial y0: {y0}
"""
console.print(Panel(input_info, title="[bold blue]Đề bài[/bold blue]", expand=False))

# 2.2 In Công thức
console.print("\n[bold green]Áp dụng ODE_2D_RK3 Ta có:[/bold green]")
formula = """
k1 = h * f(t, x, y)
l1 = h * g(t, x, y)

k2 = h * f(t + h/2, x + k1/2, y + l1/2)
l2 = h * g(t + h/2, x + k1/2, y + l1/2)

k3 = h * f(t + h, x - k1 + 2*k2, y - l1 + 2*l2)
l3 = h * g(t + h, x - k1 + 2*k2, y - l1 + 2*l2)

x_next = x + (1/6)*(k1 + 4*k2 + k3)
y_next = y + (1/6)*(l1 + 4*l2 + l3)
"""
console.print(Panel(formula, title="Công thức RK3 (Kutta)", border_style="green"))

# 2.3 In Bảng giá trị
table = Table(title="Bảng giá trị Runge-Kutta 3")
cols = ["Iter", "t", "x", "y", "k1", "l1", "k2", "l2", "k3", "l3"]
for col in cols:
    table.add_column(col, justify="right")

# Logic in bảng (5 đầu, 5 cuối)
# Cần tính lại hệ số k, l để hiển thị vì hàm gốc không trả về
for i in range(num_steps):
    # Điều kiện in: 5 dòng đầu hoặc 5 dòng cuối
    if i < 5 or i >= num_steps - 5:
        ti, xi, yi = t_res[i], x_res[i], y_res[i]
        
        # Tính lại hệ số để hiển thị
        k1 = h * f(ti, xi, yi)
        l1 = h * g(ti, xi, yi)
        k2 = h * f(ti + h/2, xi + k1/2, yi + l1/2)
        l2 = h * g(ti + h/2, xi + k1/2, yi + l1/2)
        k3 = h * f(ti + h, xi - k1 + 2*k2, yi - l1 + 2*l2)
        l3 = h * g(ti + h, xi - k1 + 2*k2, yi - l1 + 2*l2)
        
        table.add_row(
            str(i),
            f"{ti:.4f}", f"{xi:.6f}", f"{yi:.6f}",
            f"{k1:.4f}", f"{l1:.4f}",
            f"{k2:.4f}", f"{l2:.4f}",
            f"{k3:.4f}", f"{l3:.4f}"
        )
    elif i == 5:
        table.add_row("...", "...", "...", "...", "...", "...", "...", "...", "...", "...")

# Thêm dòng kết quả cuối cùng (không có k, l)
table.add_row(
    str(num_steps), 
    f"{t_res[-1]:.4f}", f"{x_res[-1]:.6f}", f"{y_res[-1]:.6f}", 
    "-", "-", "-", "-", "-", "-"
)

console.print(table)

# ==========================================
# 3. EXPORT CSV
# ==========================================
df = pd.DataFrame({
    't': t_res,
    'x': x_res,
    'y': y_res
})

csv_filename = os.path.join(output_dir, f"{method_name}.csv")
df.to_csv(csv_filename, index=False)
console.print(f"\n[bold yellow]Đã xuất file CSV: {csv_filename}[/bold yellow]")

# Xuất Phase CSV (x, y)
df_phase = pd.DataFrame({'x': x_res, 'y': y_res})
csv_phase_filename = os.path.join(output_dir, f"{method_name}_Phase.csv")
df_phase.to_csv(csv_phase_filename, index=False)
console.print(f"[bold yellow]Đã xuất file CSV Phase: {csv_phase_filename}[/bold yellow]")

# ==========================================
# 4. PLOT GRAPH
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(t_res, x_res, 'b-o', label='x(t)')
plt.plot(t_res, y_res, 'r-s', label='y(t)')
plt.title(f'Giải hệ PTVP bằng RK3 (h={h})')
plt.xlabel('Thời gian (t)')
plt.ylabel('Giá trị nghiệm')
plt.legend()
plt.grid(True)
plt.grid(True)
img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
plt.savefig(img_filename)
console.print(f"[bold yellow]Đã lưu đồ thị: {img_filename}[/bold yellow]")

# Vẽ đồ thị Phase (y vs x)
plt.figure(figsize=(8, 8))
plt.plot(x_res, y_res, label='Phase Portrait', color='purple')
plt.title(f'Đồ thị Pha - RK3 (h={h})')
plt.xlabel('x (Nghiệm 1)')
plt.ylabel('y (Nghiệm 2)')
plt.grid(True)
plt.legend()
plt.tight_layout()
img_phase_filename = os.path.join(output_dir, f"graph_{method_name}_Phase.png")
plt.savefig(img_phase_filename)
console.print(f"[bold yellow]Đã lưu đồ thị Phase: {img_phase_filename}[/bold yellow]")

# Save Text Report
txt_filename = os.path.join(output_dir, f"{method_name}.txt")
console.save_text(txt_filename)
console.print(f"[bold yellow]Đã lưu báo cáo text vào file: {txt_filename}[/bold yellow]")
plt.show()