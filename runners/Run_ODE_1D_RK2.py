import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
import math

# ==============================================================================
# 1. CẤU HÌNH ĐẦU VÀO (INPUT CONFIGURATION)
# ==============================================================================

FUNCTION_EXPRESSION = "-2 * x + t"  # Biểu thức f(t, x). Lưu ý: x là hàm cần tìm, t là biến độc lập
T0 = 0.0                            # Thời điểm bắt đầu
X0 = 1.0                            # Giá trị ban đầu x(t0)
H = 0.1                             # Bước nhảy thời gian
T_END = 0.5                         # Thời điểm kết thúc

# Tên file xuất kết quả
OUTPUT_CSV = "result_ODE_1D_RK2.csv"
OUTPUT_IMG = "graph_ODE_1D_RK2.png"

# ==============================================================================
# 2. PHẦN IMPORT VÀ HÀM (MÔ PHỎNG IMPORT TỪ MODULE)
# ==============================================================================

def f_func(t, x):
    """Hàm wrapper để tính giá trị f(t, x) từ biểu thức string."""
    # Cho phép sử dụng các hàm toán học trong biểu thức (sin, cos, exp, etc.)
    return eval(FUNCTION_EXPRESSION, {"__builtins__": None}, 
                {"t": t, "x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt})

# Đây là hàm được yêu cầu import từ methods.ODE_1D_RK2
# Để script chạy độc lập, tôi định nghĩa lại ở đây.
def rk2_ode_1d(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1D (ODE 1D) bằng phương pháp Runge-Kutta bậc 2 (RK2).
    Biến thể sử dụng: Phương pháp Heun (Heun's Method / Improved Euler).
    """
    num_steps = int(np.round((T - t0) / h))
    t_values = np.zeros(num_steps + 1)
    x_values = np.zeros(num_steps + 1)
    
    t_values[0] = t0
    x_values[0] = x0
    
    for i in range(num_steps):
        t_curr = t_values[i]
        x_curr = x_values[i]
        
        k1 = f(t_curr, x_curr)
        
        t_next_pred = t_curr + h
        x_next_pred = x_curr + h * k1
        k2 = f(t_next_pred, x_next_pred)
        
        x_next = x_curr + (h / 2.0) * (k1 + k2)
        
        x_values[i+1] = x_next
        t_values[i+1] = t0 + (i + 1) * h
        
    return {
        "t": t_values.tolist(),
        "x": x_values.tolist()
    }

# ==============================================================================
# 3. PHẦN CHẠY VÀ TRÌNH BÀY KẾT QUẢ (MAIN RUNNER)
# ==============================================================================

def main():
    console = Console()

    # --- 3.1. In Đề bài ---
    console.print(Panel.fit(
        f"[bold yellow]BÀI TOÁN GIẢI PHƯƠNG TRÌNH VI PHÂN THƯỜNG (ODE)[/bold yellow]\n\n"
        f"Phương trình: [cyan]x' = f(t, x) = {FUNCTION_EXPRESSION}[/cyan]\n"
        f"Điều kiện đầu: [cyan]x({T0}) = {X0}[/cyan]\n"
        f"Khoảng tính toán: [{T0}, {T_END}]\n"
        f"Bước nhảy: h = {H}",
        title="ĐỀ BÀI", border_style="blue"
    ))

    # --- 3.2. Trình bày Lời giải chi tiết (Tự luận) ---
    console.print("\n[bold underline]LỜI GIẢI CHI TIẾT:[/bold underline]\n")

    # Lý thuyết
    theory_text = """
    **1. Cơ sở lý thuyết:**
    
    Để giải gần đúng phương trình vi phân đã cho, ta sử dụng phương pháp **Runge-Kutta bậc 2 (biến thể Heun)**.
    Công thức lặp để tính giá trị $x_{n+1}$ từ $x_n$ tại bước thứ $n$ như sau:
    
    Ta có các hệ số độ dốc:
    - $k_1 = f(t_n, x_n)$ : Độ dốc tại đầu khoảng.
    - $k_2 = f(t_n + h, x_n + h \cdot k_1)$ : Độ dốc tại cuối khoảng (dự báo bằng Euler).
    
    Công thức cập nhật nghiệm:
    $$x_{n+1} = x_n + \frac{h}{2}(k_1 + k_2)$$
    
    Trong đó:
    - $t_{n+1} = t_n + h$
    """
    console.print(Markdown(theory_text))

    # Bắt đầu tính toán chi tiết từng bước để hiển thị (Re-calculation for display)
    console.print("[bold]2. Quá trình tính toán cụ thể:[/bold]")
    
    t_curr = T0
    x_curr = X0
    num_steps = int(np.round((T_END - T0) / H))
    
    # Tạo bảng tổng hợp kết quả
    table = Table(title="Bảng tính toán chi tiết các bước lặp (RK2 - Heun)", box=box.ROUNDED)
    table.add_column("Iter (i)", justify="center", style="cyan")
    table.add_column("t_i", justify="right")
    table.add_column("x_i (Current)", justify="right", style="green")
    table.add_column("k1", justify="right")
    table.add_column("x_pred (Euler)", justify="right")
    table.add_column("k2", justify="right")
    table.add_column("x_{i+1} (Next)", justify="right", style="bold magenta")

    # List lưu dữ liệu để xuất CSV
    data_rows = []
    data_rows.append({"Iter": 0, "t": t_curr, "x": x_curr, "k1": None, "k2": None})

    for i in range(num_steps):
        console.print(f"\n[bold yellow]--- Bước lặp thứ {i+1} (t = {t_curr:.4f} -> {t_curr+H:.4f}) ---[/bold yellow]")
        
        # Tính k1
        k1 = f_func(t_curr, x_curr)
        console.print(f"   + Tại ($t_{i}$, $x_{i}$) = ({t_curr:.4f}, {x_curr:.6f}):")
        console.print(f"     Ta tính $k_1 = f({t_curr:.4f}, {x_curr:.6f}) = {k1:.6f}$")
        
        # Tính dự báo cho k2
        t_next = t_curr + H
        x_pred = x_curr + H * k1
        console.print(f"   + Dự báo điểm cuối khoảng (Euler):")
        console.print(f"     $t_{{i+1}} = {t_next:.4f}$")
        console.print(f"     $x_{{pred}} = x_{i} + h \cdot k_1 = {x_curr:.6f} + {H} \cdot {k1:.6f} = {x_pred:.6f}$")
        
        # Tính k2
        k2 = f_func(t_next, x_pred)
        console.print(f"   + Tại ($t_{{i+1}}$, $x_{{pred}}$) = ({t_next:.4f}, {x_pred:.6f}):")
        console.print(f"     Ta tính $k_2 = f({t_next:.4f}, {x_pred:.6f}) = {k2:.6f}$")
        
        # Tính x_next
        x_next = x_curr + (H / 2.0) * (k1 + k2)
        console.print(f"   + Cập nhật nghiệm $x_{{i+1}}$ (Trung bình trọng số):")
        console.print(f"     $x_{{i+1}} = {x_curr:.6f} + \\frac{{{H}}}{{2}}({k1:.6f} + {k2:.6f}) = [bold magenta]{x_next:.6f}[/bold magenta]$")
        
        # Thêm vào bảng hiển thị
        table.add_row(
            str(i), 
            f"{t_curr:.4f}", 
            f"{x_curr:.6f}", 
            f"{k1:.6f}", 
            f"{x_pred:.6f}", 
            f"{k2:.6f}", 
            f"{x_next:.6f}"
        )
        
        # Cập nhật cho vòng lặp sau
        t_curr = t_next
        x_curr = x_next
        
        # Lưu dữ liệu
        data_rows.append({"Iter": i+1, "t": t_curr, "x": x_curr, "k1": k1, "k2": k2})

    # Hiển thị bảng tổng hợp
    console.print("\n")
    console.print(table)

    # --- 3.3. Chạy hàm gốc để verify (Double check) ---
    # Phần này gọi hàm gốc để đảm bảo tính toàn vẹn của logic library
    result = rk2_ode_1d(f_func, T0, X0, H, T_END)
    
    # --- 3.4. Xuất CSV ---
    df = pd.DataFrame(data_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    console.print(f"\n[green]✔ Đã lưu kết quả chi tiết vào file: {OUTPUT_CSV}[/green]")

    # --- 3.5. Vẽ đồ thị ---
    plt.figure(figsize=(10, 6))
    plt.plot(result['t'], result['x'], 'o-', label='RK2 (Heun) Approximation', color='blue', markersize=6)
    
    # Trang trí đồ thị
    plt.title(f"Giải ODE bằng phương pháp RK2: x' = {FUNCTION_EXPRESSION}", fontsize=14)
    plt.xlabel("Thời gian (t)", fontsize=12)
    plt.ylabel("Giá trị x(t)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Thêm text box thông số
    info_text = f"$x_0={X0}$\n$h={H}$"
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(OUTPUT_IMG)
    console.print(f"[green]✔ Đã lưu biểu đồ vào file: {OUTPUT_IMG}[/green]")
    
    # Hiển thị đồ thị (nếu môi trường hỗ trợ)
    # plt.show() 

if __name__ == "__main__":
    main()