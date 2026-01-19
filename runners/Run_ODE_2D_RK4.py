import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
# Định nghĩa hệ phương trình vi phân:
# dx/dt = f(t, x, y)
# dy/dt = g(t, x, y)

def f(t, x, y):
    # Ví dụ: dx/dt = x - y + 2t
    return x - y + 2 * t

def g(t, x, y):
    # Ví dụ: dy/dt = x + y
    return x + y

# Tham số đầu vào
t0 = 0.0        # Thời điểm bắt đầu
x0 = 1.0        # Giá trị ban đầu x(t0)
y0 = 1.0        # Giá trị ban đầu y(t0)
h  = 0.1        # Bước nhảy thời gian
T  = 1.0        # Thời điểm kết thúc

# ==============================================================================
# 2. IMPORT ALGORITHM
# ==============================================================================
try:
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from methods.ODE_2D_RK4 import solve_ode_2d_rk4
except ImportError:
    # Fallback nếu không tìm thấy module (để code có thể chạy demo ngay lập tức)
    def solve_ode_2d_rk4(f, g, t0, x0, y0, h, T):
        num_steps = int(np.ceil((T - t0) / h))
        t_values = [t0]
        x_values = [x0]
        y_values = [y0]
        t, x, y = t0, x0, y0
        for _ in range(num_steps):
            k1_x = h * f(t, x, y)
            k1_y = h * g(t, x, y)
            k2_x = h * f(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y)
            k2_y = h * g(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y)
            k3_x = h * f(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y)
            k3_y = h * g(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y)
            k4_x = h * f(t + h, x + k3_x, y + k3_y)
            k4_y = h * g(t + h, x + k3_x, y + k3_y)
            x = x + (1.0 / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            y = y + (1.0 / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
            t = t + h
            t_values.append(t)
            x_values.append(x)
            y_values.append(y)
        return {"t": t_values, "x": x_values, "y": y_values}

# ==============================================================================
# 3. EXECUTION & OUTPUT
# ==============================================================================
def main():
    console = Console()

    # --- 3.1. In đề bài ---
    console.print(Panel(Text("Đề bài: Giải hệ PTVP bằng RK4", style="bold magenta"), expand=False))
    console.print(f"- Hàm f(t, x, y) (dx/dt): [green]x - y + 2t[/green]")
    console.print(f"- Hàm g(t, x, y) (dy/dt): [green]x + y[/green]")
    console.print(f"- Khoảng thời gian: [{t0}, {T}]")
    console.print(f"- Bước nhảy h: {h}")
    console.print(f"- Điều kiện đầu: x({t0})={x0}, y({t0})={y0}")
    console.print("-" * 40)

    # --- 3.2. Tính toán ---
    result = solve_ode_2d_rk4(f, g, t0, x0, y0, h, T)
    t_vals = result['t']
    x_vals = result['x']
    y_vals = result['y']

    # --- 3.3. In công thức ---
    console.print(Text("Áp dụng ODE_2D_RK4 Ta có:", style="bold cyan"))
    formula = (
        "k₁x = h.f(t, x, y);  k₁y = h.g(t, x, y)\n"
        "k₂x = h.f(t + h/2, x + k₁x/2, y + k₁y/2);  k₂y = h.g(t + h/2, x + k₁x/2, y + k₁y/2)\n"
        "k₃x = h.f(t + h/2, x + k₂x/2, y + k₂y/2);  k₃y = h.g(t + h/2, x + k₂x/2, y + k₂y/2)\n"
        "k₄x = h.f(t + h, x + k₃x, y + k₃y);  k₄y = h.g(t + h, x + k₃x, y + k₃y)\n"
        "x_next = x + ⅙(k₁x + 2k₂x + 2k₃x + k₄x)\n"
        "y_next = y + ⅙(k₁y + 2k₂y + 2k₃y + k₄y)"
    )
    console.print(Panel(formula, title="Công thức RK4 (Hệ 2 chiều)", border_style="blue"))

    # --- 3.4. In bảng giá trị ---
    console.print(Text("Bảng giá trị", style="bold yellow"))
    
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Iteration", justify="right")
    table.add_column("t (Time)", justify="right")
    table.add_column("x (Result)", justify="right")
    table.add_column("y (Result)", justify="right")

    n = len(t_vals)
    # Logic in 5 dòng đầu và 5 dòng cuối
    indices = list(range(n))
    if n > 10:
        display_indices = indices[:5] + [-1] + indices[-5:]
    else:
        display_indices = indices

    for i in display_indices:
        if i == -1:
            table.add_row("...", "...", "...", "...")
            continue
        
        table.add_row(
            str(i),
            f"{t_vals[i]:.4f}",
            f"{x_vals[i]:.6f}",
            f"{y_vals[i]:.6f}"
        )

    console.print(table)

    # --- 3.5. Xuất CSV ---
    # Yêu cầu: Chỉ có cột x (biến độc lập - ở đây là t) và y (biến phụ thuộc - ở đây là x và y)
    df = pd.DataFrame({
        'x': t_vals,  # Biến độc lập
        'y1': x_vals, # Nghiệm thứ nhất
        'y2': y_vals  # Nghiệm thứ hai
    })
    csv_filename = "ODE_2D_RK4.csv"
    df.to_csv(csv_filename, index=False)
    console.print(f"\n[bold green]Đã lưu kết quả vào file: {csv_filename}[/bold green]")

    # --- 3.6. Vẽ đồ thị ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, 'o-', label='x(t)', markersize=4)
    plt.plot(t_vals, y_vals, 's-', label='y(t)', markersize=4)
    
    plt.title(f"Giải hệ PTVP bằng RK4 (h={h})")
    plt.xlabel("Thời gian (t)")
    plt.ylabel("Giá trị nghiệm")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_filename = "graph_ODE_2D_RK4.png"
    plt.savefig(img_filename)
    plt.close()
    console.print(f"[bold green]Đã lưu đồ thị vào file: {img_filename}[/bold green]")

if __name__ == "__main__":
    main()