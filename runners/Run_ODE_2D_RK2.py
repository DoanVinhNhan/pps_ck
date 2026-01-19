Dưới đây là file runner Python đáp ứng đầy đủ các yêu cầu của bạn.

```python
import numpy as np
import matplotlib.pyplot as plt
import csv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from methods.ODE_2D_RK2 import ode_2d_rk2

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
def f(t, x, y):
    """Hàm f(t, x, y) tương ứng với dx/dt"""
    return y  # Ví dụ: x' = y

def g(t, x, y):
    """Hàm g(t, x, y) tương ứng với dy/dt"""
    return -x # Ví dụ: y' = -x (Dao động điều hòa)

t0 = 0.0        # Thời điểm đầu
x0 = 1.0        # Giá trị ban đầu x(t0)
y0 = 0.0        # Giá trị ban đầu y(t0)
h  = 0.1        # Bước nhảy
T  = 2.0        # Thời điểm kết thúc

# ==============================================================================
# 2. MAIN RUNNER
# ==============================================================================
def run():
    console = Console()

    # --- In Đề bài ---
    input_info = f"""
    Hệ phương trình:
      dx/dt = f(t, x, y)
      dy/dt = g(t, x, y)
    
    Khoảng tính toán: [{t0}, {T}]
    Bước nhảy h: {h}
    Điều kiện đầu: t0 = {t0}, x0 = {x0}, y0 = {y0}
    """
    console.print(Panel(input_info, title="[bold cyan]Đề bài[/bold cyan]", expand=False))

    # --- Thực thi thuật toán ---
    t_vals, x_vals, y_vals = ode_2d_rk2(f, g, t0, x0, y0, h, T)

    # --- In Công thức ---
    console.print("\n[bold yellow]Áp dụng ODE_2D_RK2 Ta có:[/bold yellow]")
    formula = """
    k1_x = f(t, x, y)
    k1_y = g(t, x, y)
    
    x_pred = x + h * k1_x
    y_pred = y + h * k1_y
    
    k2_x = f(t + h, x_pred, y_pred)
    k2_y = g(t + h, x_pred, y_pred)
    
    x_{n+1} = x_n + (h / 2) * (k1_x + k2_x)
    y_{n+1} = y_n + (h / 2) * (k1_y + k2_y)
    """
    console.print(Panel(formula, title="[bold green]Công thức (Heun Method)[/bold green]", expand=False))

    # --- In Bảng giá trị ---
    console.print("\n[bold magenta]Bảng giá trị[/bold magenta]")
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Iter")
    table.add_column("t")
    table.add_column("x")
    table.add_column("y")
    table.add_column("k1_x")
    table.add_column("k1_y")
    table.add_column("k2_x")
    table.add_column("k2_y")

    num_points = len(t_vals)
    
    # Hàm helper để tính hệ số và tạo row cho bảng
    def get_row_data(idx):
        ti, xi, yi = t_vals[idx], x_vals[idx], y_vals[idx]
        
        # Tính lại hệ số k để hiển thị (vì thuật toán gốc không trả về)
        if idx < num_points - 1:
            k1x = f(ti, xi, yi)
            k1y = g(ti, xi, yi)
            
            x_pred = xi + h * k1x
            y_pred = yi + h * k1y
            t_next = ti + h
            
            k2x = f(t_next, x_pred, y_pred)
            k2y = g(t_next, x_pred, y_pred)
            
            return (
                str(idx), 
                f"{ti:.4f}", f"{xi:.6f}", f"{yi:.6f}", 
                f"{k1x:.4f}", f"{k1y:.4f}", f"{k2x:.4f}", f"{k2y:.4f}"
            )
        else:
            return (str(idx), f"{ti:.4f}", f"{xi:.6f}", f"{yi:.6f}", "-", "-", "-", "-")

    # Logic in 5 dòng đầu và 5 dòng cuối
    if num_points <= 10:
        for i in range(num_points):
            table.add_row(*get_row_data(i))
    else:
        for i in range(5):
            table.add_row(*get_row_data(i))
        
        table.add_row("...", "...", "...", "...", "...", "...", "...", "...")
        
        for i in range(num_points - 5, num_points):
            table.add_row(*get_row_data(i))

    console.print(table)

    # --- Xuất file CSV ---
    with open('ODE_2D_RK2.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['t', 'x', 'y'])
        for t, x, y in zip(t_vals, x_vals, y_vals):
            writer.writerow([t, x, y])
    console.print("\n[green]Đã xuất file ODE_2D_RK2.csv[/green]")

    # --- Vẽ đồ thị ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, label='x(t)', marker='o', markersize=4)
    plt.plot(t_vals, y_vals, label='y(t)', marker='s', markersize=4, linestyle='--')
    plt.title(f'Giải hệ ODE bằng RK2 (Heun)\nh={h}, Khoảng [{t0}, {T}]')
    plt.xlabel('Thời gian (t)')
    plt.ylabel('Giá trị nghiệm')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph_ODE_2D_RK2.png')
    console.print("[green]Đã lưu đồ thị graph_ODE_2D_RK2.png[/green]")

if __name__ == "__main__":
    run()