import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.panel import Panel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from methods.ODE.ODE_1D_RK2 import rk2_ode_1d

# ==========================================
# INPUT PARAMETERS
# ==========================================
# Định nghĩa hàm f(t, x) cho phương trình x' = f(t, x)
# Ví dụ: x' = x - t^2 + 1
def f(t, x):
    return x - t**2 + 1

t0 = 0.0    # Thời điểm bắt đầu
x0 = 0.5    # Giá trị ban đầu x(t0)
h = 0.2     # Bước nhảy
T = 2.0     # Thời điểm kết thúc

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    console = Console(record=True)
    import os
    method_name = "ODE_1D_RK2"
    output_dir = os.path.join(os.path.dirname(__file__), '../..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. In Đề bài
    console.print(Panel(
        f"[bold]Hàm số f(t, x):[/bold] x - t^2 + 1\n"
        f"[bold]Khoảng thời gian:[/bold] [{t0}, {T}]\n"
        f"[bold]Bước nhảy (h):[/bold] {h}\n"
        f"[bold]Điều kiện đầu:[/bold] x({t0}) = {x0}",
        title="[bold green]Đề bài[/bold green]",
        expand=False
    ))

    # 2. Gọi thuật toán
    result = rk2_ode_1d(f, t0, x0, h, T)
    t_values = result['t']
    x_values = result['x']

    # 2.1 Info hội tụ
    if "convergence_info" in result:
        info = result["convergence_info"]
        info_text = f"Method Name: {info.get('method_name', 'Unknown')}\n"
        info_text += f"Order: {info.get('approximation_order', 'Unknown')}\n"
        info_text += f"Stability Region: {info.get('stability_region', 'Unknown')}\n"
        info_text += f"Stability Function: {info.get('stability_function', 'Unknown')}"
        console.print(Panel(info_text, title="[bold magenta]Hội tụ & Ổn định[/bold magenta]", expand=False))

    # 3. In Công thức
    console.print("\n[bold cyan]Áp dụng ODE_1D_RK2 Ta có:[/bold cyan]")
    formula = (
        "k₁ = f(tᵢ, xᵢ)\n"
        "k₂ = f(tᵢ + h, xᵢ + h⋅k₁)\n"
        "xᵢ₊₁ = xᵢ + (h/2)⋅(k₁ + k₂)"
    )
    console.print(Panel(formula, title="<CÔNG THỨC>", border_style="cyan", expand=False))

    # 4. Tạo và in Bảng giá trị
    table = Table(title="Bảng giá trị (Heun's Method)")
    table.add_column("Iteration", justify="center", style="cyan")
    table.add_column("t (Biến độc lập)", justify="right")
    table.add_column("x (Biến phụ thuộc)", justify="right", style="green")
    table.add_column("k1", justify="right")
    table.add_column("k2", justify="right")

    # Tái tạo lại các bước trung gian để hiển thị bảng
    rows = []
    n_steps = len(t_values) - 1
    
    for i in range(len(t_values)):
        t_curr = t_values[i]
        x_curr = x_values[i]
        
        # Tính k1, k2 để hiển thị (trừ dòng cuối cùng không cần tính tiếp)
        if i < n_steps:
            k1 = f(t_curr, x_curr)
            x_pred = x_curr + h * k1
            k2 = f(t_curr + h, x_pred)
            k1_str = f"{k1:.6g}"
            k2_str = f"{k2:.6g}"
        else:
            k1_str = "-"
            k2_str = "-"
            
        rows.append([str(i), f"{t_curr:.6g}", f"{x_curr:.6g}", k1_str, k2_str])

    # Logic in rút gọn: 5 dòng đầu + ... + 5 dòng cuối
    total_rows = len(rows)
    if total_rows <= 10:
        for row in rows:
            table.add_row(*row)
    else:
        for row in rows[:5]:
            table.add_row(*row)
        table.add_row("...", "...", "...", "...", "...")
        for row in rows[-5:]:
            table.add_row(*row)

    console.print(table)

    # 5. Xuất file CSV
    # Mapping: t -> x (independent), x -> y (dependent)
    df = pd.DataFrame({
        'x': t_values,
        'y': x_values
    })
    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    df.to_csv(csv_filename, index=False)
    console.print(f"[bold yellow]Đã xuất file: {csv_filename}[/bold yellow]")

    # 6. Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, x_values, marker='o', linestyle='-', color='b', label='RK2 Approximation')
    plt.title(f"Nghiệm gần đúng ODE bằng phương pháp RK2 (h={h})")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
    plt.savefig(img_filename)
    console.print(f"[bold yellow]Đã lưu đồ thị: {img_filename}[/bold yellow]")
    
    # Save Text Report
    txt_filename = os.path.join(output_dir, f"{method_name}.txt")
    console.save_text(txt_filename)
    console.print(f"[bold yellow]Đã lưu báo cáo text vào file: {txt_filename}[/bold yellow]")

if __name__ == "__main__":
    main()