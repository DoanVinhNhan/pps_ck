Dưới đây là file runner Python đáp ứng đầy đủ các yêu cầu của bạn.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ==============================================================================
# INPUT PARAMETERS
# ==============================================================================
# Định nghĩa hệ phương trình vi phân:
# x' = f(t, x, y)
# y' = g(t, x, y)

def f(t, x, y):
    # Ví dụ: x' = -y
    return -y

def g(t, x, y):
    # Ví dụ: y' = x
    return x

t0 = 0.0        # Thời điểm bắt đầu
x0 = 1.0        # Giá trị ban đầu của x
y0 = 0.0        # Giá trị ban đầu của y
h = 0.1         # Bước nhảy thời gian
T = 5.0         # Thời điểm kết thúc

# ==============================================================================
# IMPORT ALGORITHM
# ==============================================================================
try:
    from methods.ODE_2D_Euler_Implicit import solve_ode_2d_implicit_euler
except ImportError:
    # Fallback nếu không tìm thấy module (để code chạy được độc lập khi test)
    def solve_ode_2d_implicit_euler(f, g, t0, x0, y0, h, T):
        t_list = [t0]
        x_list = [x0]
        y_list = [y0]
        num_steps = int(np.round((T - t0) / h))
        t_curr, x_curr, y_curr = t0, x0, y0
        max_iter = 100
        tol = 1e-6
        
        for _ in range(num_steps):
            t_next = t_curr + h
            # Predictor (Euler Explicit)
            x_next = x_curr + h * f(t_curr, x_curr, y_curr)
            y_next = y_curr + h * g(t_curr, x_curr, y_curr)
            
            # Corrector (Fixed-Point Iteration)
            for _ in range(max_iter):
                x_old_iter = x_next
                y_old_iter = y_next
                x_next = x_curr + h * f(t_next, x_next, y_next)
                y_next = y_curr + h * g(t_next, x_next, y_next)
                if abs(x_next - x_old_iter) < tol and abs(y_next - y_old_iter) < tol:
                    break
            
            t_curr, x_curr, y_curr = t_next, x_next, y_next
            t_list.append(t_curr)
            x_list.append(x_curr)
            y_list.append(y_curr)
            
        return {"t": t_list, "x": x_list, "y": y_list}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    console = Console()

    # 1. In tiêu đề Đề bài
    param_text = Text()
    param_text.append(f"Hệ phương trình: x' = f(t,x,y), y' = g(t,x,y)\n", style="bold cyan")
    param_text.append(f"Khoảng thời gian: [{t0}, {T}]\n")
    param_text.append(f"Bước nhảy h: {h}\n")
    param_text.append(f"Điều kiện đầu: x({t0})={x0}, y({t0})={y0}")
    
    console.print(Panel(param_text, title="Đề bài", title_align="left", border_style="blue"))

    # 2. Thực thi thuật toán
    result = solve_ode_2d_implicit_euler(f, g, t0, x0, y0, h, T)
    t_vals = result['t']
    x_vals = result['x']
    y_vals = result['y']

    # 3. In công thức
    console.print("\n[bold yellow]Áp dụng ODE_2D_Euler_Implicit Ta có:[/bold yellow]")
    formula = (
        "Hệ phương trình sai phân (Euler Ẩn):\n"
        "  t_{n+1} = t_n + h\n"
        "  x_{n+1} = x_n + h * f(t_{n+1}, x_{n+1}, y_{n+1})\n"
        "  y_{n+1} = y_n + h * g(t_{n+1}, x_{n+1}, y_{n+1})\n"
        "(Giải hệ bằng phương pháp lặp điểm bất động)"
    )
    console.print(Panel(formula, border_style="green"))

    # 4. In bảng giá trị
    console.print("\n[bold]Bảng giá trị[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Iteration")
    table.add_column("t (Time)")
    table.add_column("x (Value)")
    table.add_column("y (Value)")

    n = len(t_vals)
    if n <= 10:
        for i in range(n):
            table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")
    else:
        # 5 dòng đầu
        for i in range(5):
            table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")
        # Dòng rút gọn
        table.add_row("...", "...", "...", "...")
        # 5 dòng cuối
        for i in range(n - 5, n):
            table.add_row(str(i), f"{t_vals[i]:.4f}", f"{x_vals[i]:.6f}", f"{y_vals[i]:.6f}")

    console.print(table)

    # 5. Xuất file CSV
    df = pd.DataFrame({
        't': t_vals,
        'x': x_vals,
        'y': y_vals
    })
    csv_filename = 'ODE_2D_Euler_Implicit.csv'
    df.to_csv(csv_filename, index=False)
    console.print(f"\n[green]Đã lưu kết quả vào file: {csv_filename}[/green]")

    # 6. Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, label='x(t)', marker='o', markersize=3, linestyle='-')
    plt.plot(t_vals, y_vals, label='y(t)', marker='s', markersize=3, linestyle='--')
    plt.title(f'Nghiệm hệ PTVP bằng Euler Ẩn (h={h})')
    plt.xlabel('Thời gian (t)')
    plt.ylabel('Giá trị nghiệm')
    plt.legend()
    plt.grid(True)
    
    graph_filename = 'graph_ODE_2D_Euler_Implicit.png'
    plt.savefig(graph_filename)
    console.print(f"[green]Đã lưu đồ thị vào file: {graph_filename}[/green]")
    # plt.show() # Bỏ comment nếu muốn hiện cửa sổ plot

if __name__ == "__main__":
    main()