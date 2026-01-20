import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ==============================================================================
# 1. INPUT PARAMETERS (Thay đổi thông số bài toán tại đây)
# ==============================================================================

# Ví dụ: Mô hình Thú - Mồi (Lotka-Volterra) từ Slide 13
# x: Số lượng con mồi (n)
# y: Số lượng thú săn mồi (p)
# dx/dt = r*x*(1 - x/K) - a*x*y
# dy/dt = -mu*y + a*x*y

r = 1.0     # Tốc độ sinh trưởng của mồi
K = 10.0    # Sức chứa môi trường
a = 0.5     # Tỷ lệ săn mồi
mu = 0.5    # Tỷ lệ tử vong của thú săn mồi

def f(t, x, y):
    """Hàm f(t, x, y) tương ứng với dx/dt"""
    return r * x * (1 - x / K) - a * x * y

def g(t, x, y):
    """Hàm g(t, x, y) tương ứng với dy/dt"""
    return -mu * y + a * x * y

# Điều kiện ban đầu và lưới thời gian
t0 = 0.0        # Thời điểm bắt đầu
x0 = 2.0        # Số lượng mồi ban đầu
y0 = 1.0        # Số lượng thú săn mồi ban đầu
h = 0.1         # Bước nhảy thời gian
T = 20.0        # Thời điểm kết thúc

# ==============================================================================
# 2. IMPORT THUẬT TOÁN
# ==============================================================================
try:
    from methods.ODE.ODE_2D_Euler_Explicit import euler_forward_2d
except ImportError:
    # Fallback nếu không tìm thấy module (để code có thể chạy demo độc lập)
    def euler_forward_2d(f, g, t0, x0, y0, h, T):
        num_steps = int(np.ceil((T - t0) / h))
        t_values, x_values, y_values = [t0], [x0], [y0]
        t_curr, x_curr, y_curr = t0, x0, y0
        for i in range(num_steps):
            val_f = f(t_curr, x_curr, y_curr)
            val_g = g(t_curr, x_curr, y_curr)
            x_next = x_curr + h * val_f
            y_next = y_curr + h * val_g
            t_next = t0 + (i + 1) * h
            t_values.append(t_next)
            x_values.append(x_next)
            y_values.append(y_next)
            t_curr, x_curr, y_curr = t_next, x_next, y_next
        return {"t": t_values, "x": x_values, "y": y_values}

# ==============================================================================
# 3. EXECUTION & RICH OUTPUT
# ==============================================================================
console = Console(record=True)
import os
# Define output directory
method_name = "ODE_2D_Euler_Explicit"
output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
os.makedirs(output_dir, exist_ok=True)

# Thực thi thuật toán
result = euler_forward_2d(f, g, t0, x0, y0, h, T)
t_vals = result['t']
x_vals = result['x']
y_vals = result['y']

# --- 3.1. In tiêu đề Đề bài ---
console.print(Panel(Text("ĐỀ BÀI: Giải hệ PTVP (Lotka-Volterra) bằng Euler Hiện", style="bold magenta", justify="center")))
console.print(f"[bold]Khoảng thời gian:[/bold] [{t0}, {T}]")
console.print(f"[bold]Bước nhảy h:[/bold] {h}")
console.print(f"[bold]Giá trị ban đầu:[/bold] x({t0})={x0}, y({t0})={y0}")
console.print(f"[bold]Hệ phương trình:[/bold]")
console.print(f"  dx/dt = {r}x(1 - x/{K}) - {a}xy")
console.print(f"  dy/dt = -{mu}y + {a}xy")

# --- 3.2. In công thức ---
console.print("\n[bold cyan]Áp dụng ODE_2D_Euler_Explicit Ta có:[/bold cyan]")
formula_text = (
    "x_{n+1} = x_n + h * f(t_n, x_n, y_n)\n"
    "y_{n+1} = y_n + h * g(t_n, x_n, y_n)\n"
    "t_{n+1} = t_n + h"
)
console.print(Panel(formula_text, title="Công thức lặp", border_style="cyan"))

# --- 3.3. In bảng giá trị ---
console.print("\n[bold yellow]Bảng giá trị[/bold yellow]")
table = Table(show_header=True, header_style="bold blue", border_style="dim")
table.add_column("Iteration", justify="center")
table.add_column("t (Time)", justify="right")
table.add_column("x (Prey)", justify="right")
table.add_column("y (Predator)", justify="right")

n_points = len(t_vals)
# In 5 dòng đầu
for i in range(min(5, n_points)):
    table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")

# In dòng rút gọn nếu dữ liệu dài
if n_points > 10:
    table.add_row("...", "...", "...", "...")
    # In 5 dòng cuối
    for i in range(n_points - 5, n_points):
        table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")
elif n_points > 5:
    for i in range(5, n_points):
        table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")

console.print(table)

# ==============================================================================
# 4. EXPORT CSV
# ==============================================================================
# Yêu cầu: Chỉ có cột x (biến độc lập - ở đây là t) và y (biến phụ thuộc - ở đây là x và y)
df = pd.DataFrame({
    't': t_vals,
    'x': x_vals,
    'y': y_vals
})
csv_filename = os.path.join(output_dir, f"{method_name}.csv")
df.to_csv(csv_filename, index=False)
console.print(f"\n[green]✔ Đã xuất file kết quả: [bold]{csv_filename}[/bold][/green]")

# Xuất Phase CSV (x, y)
df_phase = pd.DataFrame({'x': x_vals, 'y': y_vals})
csv_phase_filename = os.path.join(output_dir, f"{method_name}_Phase.csv")
df_phase.to_csv(csv_phase_filename, index=False)
console.print(f"[green]✔ Đã xuất file kết quả Phase: [bold]{csv_phase_filename}[/bold][/green]")

# ==============================================================================
# 5. DRAW GRAPH
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='Con mồi (x)', color='blue', linewidth=2)
plt.plot(t_vals, y_vals, label='Thú săn mồi (y)', color='red', linestyle='--', linewidth=2)

plt.title(f"Giải hệ PTVP bằng Euler Hiện (h={h})")
plt.xlabel("Thời gian (t)")
plt.ylabel("Số lượng cá thể")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
plt.savefig(img_filename)
console.print(f"[green]✔ Đã lưu đồ thị: [bold]{img_filename}[/bold][/green]")

# Vẽ đồ thị Phase (y vs x)
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label='Phase Portrait (y vs x)', color='purple', linewidth=2)
plt.title(f'Đồ thị Pha (Lotka-Volterra) - Euler Explicit')
plt.xlabel('x (Prey)')
plt.ylabel('y (Predator)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
img_phase_filename = os.path.join(output_dir, f"graph_{method_name}_Phase.png")
plt.savefig(img_phase_filename)
console.print(f"[green]✔ Đã lưu đồ thị Phase: [bold]{img_phase_filename}[/bold][/green]")

# Save Text Report
txt_filename = os.path.join(output_dir, f"{method_name}.txt")
console.save_text(txt_filename)
console.print(f"[green]✔ Đã lưu báo cáo text vào file: {txt_filename}[/green]")

plt.show()