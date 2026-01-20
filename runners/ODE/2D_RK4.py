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
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from methods.ODE.ODE_2D_RK4 import solve_ode_2d_rk4
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
    console = Console(record=True)
    
    # Define output directory
    method_name = "ODE_2D_RK4"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

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
    table.add_column("k1_x", justify="right", style="cyan")
    table.add_column("k1_y", justify="right", style="cyan")
    table.add_column("k2_x", justify="right", style="magenta")
    table.add_column("k2_y", justify="right", style="magenta")
    table.add_column("k3_x", justify="right", style="green")
    table.add_column("k3_y", justify="right", style="green")
    table.add_column("k4_x", justify="right", style="yellow")
    table.add_column("k4_y", justify="right", style="yellow")

    n = len(t_vals)
    # Logic in 5 dòng đầu và 5 dòng cuối
    indices = list(range(n))
    if n > 10:
        display_indices = indices[:5] + [-1] + indices[-5:]
    else:
        display_indices = indices

    for i in display_indices:
        if i == -1:
            table.add_row("...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...")
            continue
        
        ti, xi, yi = t_vals[i], x_vals[i], y_vals[i]

        # Calculate k values for display (if not the last step)
        if i < n - 1:
             k1_x = h * f(ti, xi, yi)
             k1_y = h * g(ti, xi, yi)
             k2_x = h * f(ti + 0.5 * h, xi + 0.5 * k1_x, yi + 0.5 * k1_y)
             k2_y = h * g(ti + 0.5 * h, xi + 0.5 * k1_x, yi + 0.5 * k1_y)
             k3_x = h * f(ti + 0.5 * h, xi + 0.5 * k2_x, yi + 0.5 * k2_y)
             k3_y = h * g(ti + 0.5 * h, xi + 0.5 * k2_x, yi + 0.5 * k2_y)
             k4_x = h * f(ti + h, xi + k3_x, yi + k3_y)
             k4_y = h * g(ti + h, xi + k3_x, yi + k3_y)
             
             k_strs = [
                 f"{k1_x:.4f}", f"{k1_y:.4f}",
                 f"{k2_x:.4f}", f"{k2_y:.4f}",
                 f"{k3_x:.4f}", f"{k3_y:.4f}",
                 f"{k4_x:.4f}", f"{k4_y:.4f}"
             ]
        else:
             k_strs = ["-", "-", "-", "-", "-", "-", "-", "-"]

        table.add_row(
            str(i),
            f"{ti:.4f}",
            f"{xi:.6f}",
            f"{yi:.6f}",
            *k_strs
        )

    console.print(table)

    # --- 3.5. Xuất CSV ---
    # Yêu cầu: Chỉ có cột x (biến độc lập - ở đây là t) và y (biến phụ thuộc - ở đây là x và y)
    df = pd.DataFrame({
        't': t_vals,
        'x': x_vals,
        'y': y_vals
    })

    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    df.to_csv(csv_filename, index=False)
    console.print(f"\n[bold green]Đã lưu kết quả vào file: {csv_filename}[/bold green]")

    # Xuất Phase CSV (x, y)
    df_phase = pd.DataFrame({'x': x_vals, 'y': y_vals})
    csv_phase_filename = os.path.join(output_dir, f"{method_name}_Phase.csv")
    df_phase.to_csv(csv_phase_filename, index=False)
    console.print(f"[bold green]Đã lưu kết quả Phase vào file: {csv_phase_filename}[/bold green]")

    # --- 3.6. Vẽ đồ thị ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, 'o-', label='x(t)', markersize=4)
    plt.plot(t_vals, y_vals, 's-', label='y(t)', markersize=4)
    
    plt.title(f"Giải hệ PTVP bằng RK4 (h={h})")
    plt.xlabel("Thời gian (t)")
    plt.ylabel("Giá trị nghiệm")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
    plt.savefig(img_filename)
    plt.close()
    console.print(f"[bold green]Đã lưu đồ thị vào file: {img_filename}[/bold green]")

    # Vẽ đồ thị Phase (y vs x)
    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, label='Phase Portrait', color='purple')
    plt.title(f'Đồ thị Pha - RK4 (h={h})')
    plt.xlabel('x (Nghiệm 1)')
    plt.ylabel('y (Nghiệm 2)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    img_phase_filename = os.path.join(output_dir, f"graph_{method_name}_Phase.png")
    plt.savefig(img_phase_filename)
    console.print(f"[bold green]Đã lưu đồ thị Phase vào file: {img_phase_filename}[/bold green]")
    
    # Save Text Report
    txt_filename = os.path.join(output_dir, f"{method_name}.txt")
    console.save_text(txt_filename)
    console.print(f"[bold green]Đã lưu báo cáo text vào file: {txt_filename}[/bold green]")

if __name__ == "__main__":
    main()