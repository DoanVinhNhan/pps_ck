import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ==========================================
# 1. INPUT PARAMETERS (ĐỀ BÀI)
# ==========================================
# Hệ phương trình: x' = f(t, x, y), y' = g(t, x, y)
# Ví dụ: Hệ dao động điều hòa x' = y, y' = -x
def f(t, x, y):
    return y

def g(t, x, y):
    return -x

t0 = 0.0        # Thời điểm đầu
x0 = 1.0        # Giá trị ban đầu x(t0)
y0 = 0.0        # Giá trị ban đầu y(t0)
h = 0.1         # Bước nhảy
T = 5.0         # Thời điểm kết thúc
s_steps = 4     # Số bước của phương pháp (2, 3, hoặc 4)

# ==========================================
# 2. IMPORT THUẬT TOÁN
# ==========================================
try:
  import sys
  import os
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

  from methods.ODE.ODE_2D_ABs_AMs import solve_ode_2d_ab_am
except ImportError:
    # Fallback nếu không tìm thấy module (để code chạy được trong môi trường test đơn lẻ)
    # Trong thực tế, phần này sẽ được import từ file methods/ODE_2D_ABs_AMs.py
    def solve_ode_2d_ab_am(f, g, t0, x0, y0, h, T, s=4):
        num_steps = int(np.ceil((T - t0) / h))
        t_values = np.zeros(num_steps + 1)
        x_values = np.zeros(num_steps + 1)
        y_values = np.zeros(num_steps + 1)
        t_values[0], x_values[0], y_values[0] = t0, x0, y0
        
        limit_rk4 = min(s - 1, num_steps)
        for i in range(limit_rk4):
            ti, xi, yi = t_values[i], x_values[i], y_values[i]
            k1_x, k1_y = h * f(ti, xi, yi), h * g(ti, xi, yi)
            k2_x, k2_y = h * f(ti + 0.5*h, xi + 0.5*k1_x, yi + 0.5*k1_y), h * g(ti + 0.5*h, xi + 0.5*k1_x, yi + 0.5*k1_y)
            k3_x, k3_y = h * f(ti + 0.5*h, xi + 0.5*k2_x, yi + 0.5*k2_y), h * g(ti + 0.5*h, xi + 0.5*k2_x, yi + 0.5*k2_y)
            k4_x, k4_y = h * f(ti + h, xi + k3_x, yi + k3_y), h * g(ti + h, xi + k3_x, yi + k3_y)
            x_values[i+1] = xi + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
            y_values[i+1] = yi + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
            t_values[i+1] = t0 + (i + 1) * h

        if s == 2:
            ab_coeffs, am_coeffs = np.array([-1, 3])/2.0, np.array([1, 1])/2.0
        elif s == 3:
            ab_coeffs, am_coeffs = np.array([5, -16, 23])/12.0, np.array([-1, 8, 5])/12.0
        elif s == 4:
            ab_coeffs, am_coeffs = np.array([-9, 37, -59, 55])/24.0, np.array([1, -5, 19, 9])/24.0
        else: raise ValueError("s must be 2, 3, or 4")

        for i in range(s - 1, num_steps):
            f_hist, g_hist = [], []
            for k in range(s):
                idx = i - (s - 1) + k
                f_hist.append(f(t_values[idx], x_values[idx], y_values[idx]))
                g_hist.append(g(t_values[idx], x_values[idx], y_values[idx]))
            
            x_pred = x_values[i] + h * np.dot(ab_coeffs, f_hist)
            y_pred = y_values[i] + h * np.dot(ab_coeffs, g_hist)
            t_next = t_values[i] + h
            
            f_next_pred, g_next_pred = f(t_next, x_pred, y_pred), g(t_next, x_pred, y_pred)
            f_hist_am, g_hist_am = np.append(f_hist[1:], f_next_pred), np.append(g_hist[1:], g_next_pred)
            
            x_values[i+1] = x_values[i] + h * np.dot(am_coeffs, f_hist_am)
            y_values[i+1] = y_values[i] + h * np.dot(am_coeffs, g_hist_am)
            t_values[i+1] = t_next

        return {"t": t_values.tolist(), "x": x_values.tolist(), "y": y_values.tolist()}

# ==========================================
# 3. EXECUTION & OUTPUT
# ==========================================
console = Console(record=True)

import os
# Define output directory
method_name = "ODE_2D_ABs_AMs"
output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
os.makedirs(output_dir, exist_ok=True)

# 3.1. In đề bài
console.print(Panel(Text(f"Hàm f(t,x,y): y\nHàm g(t,x,y): -x\nKhoảng t: [{t0}, {T}]\nBước nhảy h: {h}\nGiá trị đầu: x({t0})={x0}, y({t0})={y0}\nSố bước s: {s_steps}", justify="left"), title="Đề bài", style="bold cyan"))

# 3.2. Tính toán
result = solve_ode_2d_ab_am(f, g, t0, x0, y0, h, T, s=s_steps)

# 3.3. In công thức
console.print("\n[bold yellow]Áp dụng ODE_2D_ABs_AMs Ta có:[/bold yellow]")
if s_steps == 4:
    formula = (
        "Dự báo (Adams-Bashforth 4 step):\n"
        "  x_{n+1}^P = x_n + h/24 * (55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})\n"
        "Hiệu chỉnh (Adams-Moulton 4 step):\n"
        "  x_{n+1} = x_n + h/24 * (9f_{n+1}^P + 19f_n - 5f_{n-1} + f_{n-2})"
    )
elif s_steps == 3:
    formula = (
        "Dự báo (AB3): x_{n+1}^P = x_n + h/12 * (23f_n - 16f_{n-1} + 5f_{n-2})\n"
        "Hiệu chỉnh (AM3): x_{n+1} = x_n + h/12 * (5f_{n+1}^P + 8f_n - f_{n-1})"
    )
else:
    formula = (
        "Dự báo (AB2): x_{n+1}^P = x_n + h/2 * (3f_n - f_{n-1})\n"
        "Hiệu chỉnh (AM2): x_{n+1} = x_n + h/2 * (f_{n+1}^P + f_n)"
    )
console.print(Panel(formula, style="italic green"))

# 3.4. In bảng giá trị (Top 5 & Bottom 5)
table = Table(title="Bảng giá trị")
table.add_column("Iteration", justify="center")
table.add_column("t (Biến độc lập)", justify="right")
table.add_column("x (Nghiệm 1)", justify="right")
table.add_column("y (Nghiệm 2)", justify="right")

t_vals = result['t']
x_vals = result['x']
y_vals = result['y']
n = len(t_vals)

# Xác định các dòng cần in
indices = list(range(min(5, n)))
if n > 10:
    indices.append(-1) # Marker cho dòng "..."
    indices.extend(range(n - 5, n))
elif n > 5:
    indices.extend(range(5, n))

for i in indices:
    if i == -1:
        table.add_row("...", "...", "...", "...")
    else:
        table.add_row(
            str(i),
            f"{t_vals[i]:.4f}",
            f"{x_vals[i]:.6f}",
            f"{y_vals[i]:.6f}"
        )

console.print(table)

# 3.5. Xuất CSV
df = pd.DataFrame({
    't': t_vals,
    'x': x_vals,
    'y': y_vals
})
df.to_csv(os.path.join(output_dir, f"{method_name}.csv"), index=False)
console.print(f"\n[bold]Đã lưu file kết quả:[/bold] {method_name}.csv")

# Xuất Phase CSV (x, y)
df_phase = pd.DataFrame({'x': x_vals, 'y': y_vals})
df_phase.to_csv(os.path.join(output_dir, f"{method_name}_Phase.csv"), index=False)
console.print(f"[bold]Đã lưu file kết quả Phase:[/bold] {method_name}_Phase.csv")

# 3.6. Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='x(t)', marker='o', markersize=3)
plt.plot(t_vals, y_vals, label='y(t)', marker='x', markersize=3)
plt.title(f'Giải hệ PTVP bằng phương pháp Adams-Bashforth-Moulton (s={s_steps})')
plt.xlabel('Thời gian (t)')
plt.ylabel('Giá trị nghiệm')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"graph_{method_name}.png"))
console.print(f"[bold]Đã lưu đồ thị:[/bold] graph_{method_name}.png")

# Vẽ đồ thị Phase (y vs x)
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Phase Portrait (y vs x)', color='purple')
plt.title(f'Đồ thị Pha (Phase Portrait) - AB-AM (s={s_steps})')
plt.xlabel('x (Nghiệm 1)')
plt.ylabel('y (Nghiệm 2)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"graph_{method_name}_Phase.png"))
console.print(f"[bold]Đã lưu đồ thị Phase:[/bold] graph_{method_name}_Phase.png")

# Save Text Report
txt_filename = os.path.join(output_dir, f"{method_name}.txt")
console.save_text(txt_filename)
console.print(f"[bold]Đã lưu báo cáo text bao gồm bảng giá trị vào file: {txt_filename}[/bold]")

plt.show()