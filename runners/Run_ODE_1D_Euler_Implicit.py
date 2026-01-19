import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import math

# ==============================================================================
# 1. CẤU HÌNH ĐẦU VÀO (INPUT CONFIGURATION)
# ==============================================================================
FUNCTION_EXPRESSION = "-2 * x"  # Biểu thức f(t, x)
T0 = 0.0                        # Thời điểm bắt đầu
X0 = 1.0                        # Giá trị ban đầu x(t0)
H = 0.1                         # Bước nhảy thời gian
T_END = 0.5                     # Thời điểm kết thúc (Chọn ngắn để hiển thị đẹp, đề bài là 100)
MAX_ITER_DISPLAY = 2            # Số bước tính toán muốn hiển thị chi tiết lời giải (tự luận)

# ==============================================================================
# 2. PHẦN IMPORT THUẬT TOÁN (GIẢ LẬP MODULE methods.ODE_1D_Euler_Implicit)
# ==============================================================================

def ode_1d_implicit_euler(f, t0, x0, h, T):
    """
    Thuật toán Euler Ẩn (Backward Euler) giải ODE 1D.
    """
    # Xác định số bước lặp
    num_steps = int(np.ceil((T - t0) / h))
    
    # Khởi tạo danh sách kết quả
    t_values = [t0]
    x_values = [x0]
    
    # Các tham số cho vòng lặp giải phương trình ẩn
    max_iter = 100      
    tolerance = 1e-6    
    
    t_current = t0
    x_current = x0
    
    for _ in range(num_steps):
        t_next = t_current + h
        
        # Bước 1: Dự báo (Predictor) bằng Euler hiện
        x_next_guess = x_current + h * f(t_current, x_current)
        
        # Bước 2: Giải phương trình ẩn bằng lặp đơn
        for _ in range(max_iter):
            x_next_new = x_current + h * f(t_next, x_next_guess)
            
            if abs(x_next_new - x_next_guess) < tolerance:
                x_next_guess = x_next_new
                break
            
            x_next_guess = x_next_new
        
        # Cập nhật
        x_current = x_next_guess
        t_current = t_next
        
        # Lưu kết quả
        t_values.append(t_current)
        x_values.append(x_current)
        
    return t_values, x_values

# ==============================================================================
# 3. PHẦN CHẠY VÀ TRÌNH BÀY KẾT QUẢ (MAIN RUNNER)
# ==============================================================================

def f_func(t, x):
    """Hàm wrapper để đánh giá biểu thức nhập vào."""
    return eval(FUNCTION_EXPRESSION, {"x": x, "t": t, "np": np, "math": math})

def main():
    console = Console(record=True)
    import os
    method_name = "ODE_1D_Euler_Implicit"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', method_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- 3.1. HIỂN THỊ ĐỀ BÀI ---
    console.print(Panel.fit(
        f"[bold cyan]ĐỀ BÀI:[/bold cyan]\n"
        f"Giải phương trình vi phân: x' = {FUNCTION_EXPRESSION}\n"
        f"Điều kiện ban đầu: x({T0}) = {X0}\n"
        f"Bước nhảy h = {H}\n"
        f"Miền tính toán: [{T0}, {T_END}]",
        title="BÀI TOÁN GIÁ TRỊ BAN ĐẦU (IVP)",
        border_style="blue"
    ))

    # --- 3.2. TRÌNH BÀY LỜI GIẢI TỰ LUẬN CHI TIẾT ---
    console.print("\n[bold underline]LỜI GIẢI CHI TIẾT (PHƯƠNG PHÁP EULER ẨN):[/bold underline]\n")
    
    console.print("Ta sử dụng phương pháp Euler Ẩn (Backward Euler) với công thức tổng quát:")
    console.print("    [yellow]x_{i+1} = x_i + h * f(t_{i+1}, x_{i+1})[/yellow]")
    console.print(f"Thay số liệu vào, ta có phương trình ẩn tại mỗi bước:")
    console.print(f"    x_{{i+1}} = x_i + {H} * ({FUNCTION_EXPRESSION.replace('x', 'x_{i+1}')})")
    console.print("Do x_{i+1} xuất hiện ở cả hai vế, ta sử dụng phương pháp lặp điểm bất động để tìm nghiệm gần đúng.")
    console.print("Quy trình tại mỗi bước i:")
    console.print("  1. Dự báo giá trị khởi tạo [italic]x^{(0)}[/italic] bằng Euler hiện: [green]x^{(0)} = x_i + h * f(t_i, x_i)[/green]")
    console.print("  2. Lặp hiệu chỉnh: [green]x^{(k+1)} = x_i + h * f(t_{i+1}, x^{(k)})[/green] cho đến khi hội tụ (|x^{(k+1)} - x^{(k)}| < epsilon).")
    console.print("-" * 60)

    # Mô phỏng lại logic để in chi tiết từng bước (chỉ in số bước quy định)
    t_curr = T0
    x_curr = X0
    
    for step in range(MAX_ITER_DISPLAY):
        t_next = t_curr + H
        console.print(f"\n[bold magenta]Tính toán tại bước i = {step}: Từ t_{step}={t_curr:.2f} sang t_{step+1}={t_next:.2f}[/bold magenta]")
        
        # Tính giá trị hàm tại điểm cũ
        f_curr = f_func(t_curr, x_curr)
        console.print(f"Ta có: f(t_{step}, x_{step}) = {FUNCTION_EXPRESSION.replace('x', f'{x_curr:.4f}')} = {f_curr:.4f}")
        
        # Dự báo
        x_guess = x_curr + H * f_curr
        console.print(f"Dự báo giá trị khởi tạo (Euler hiện):")
        console.print(f"    x_{{next}}^(0) = {x_curr:.4f} + {H} * ({f_curr:.4f}) = [bold]{x_guess:.6f}[/bold]")
        
        # Bảng lặp
        table_iter = Table(title=f"Bảng quá trình lặp tìm nghiệm ẩn x_{step+1}", box=box.SIMPLE)
        table_iter.add_column("Lần lặp (k)", justify="center")
        table_iter.add_column("x^(k) (Dự đoán)", justify="right")
        table_iter.add_column("f(t_next, x^(k))", justify="right")
        table_iter.add_column("x^(k+1) (Tính mới)", justify="right")
        table_iter.add_column("Sai số |x^(k+1) - x^(k)|", justify="right")

        # Thực hiện lặp (Logic giống hệt hàm chính)
        max_iter_inner = 10
        tol = 1e-6
        x_k = x_guess
        
        for k in range(max_iter_inner):
            f_next_k = f_func(t_next, x_k)
            x_k_new = x_curr + H * f_next_k
            diff = abs(x_k_new - x_k)
            
            table_iter.add_row(
                str(k), 
                f"{x_k:.6f}", 
                f"{f_next_k:.6f}", 
                f"[green]{x_k_new:.6f}[/green]", 
                f"{diff:.2e}"
            )
            
            if diff < tol:
                x_k = x_k_new
                break
            x_k = x_k_new
            
        console.print(table_iter)
        console.print(f"Vì sai số < {tol}, ta chấp nhận nghiệm: [bold yellow]x_{step+1} = {x_k:.6f}[/bold yellow]")
        
        # Cập nhật cho vòng sau
        t_curr = t_next
        x_curr = x_k

    console.print("\n[italic]... (Các bước tiếp theo thực hiện tương tự) ...[/italic]")

    # --- 3.3. CHẠY TOÀN BỘ THUẬT TOÁN VÀ HIỂN THỊ KẾT QUẢ CUỐI ---
    t_res, x_res = ode_1d_implicit_euler(f_func, T0, X0, H, T_END)

    # Tạo bảng kết quả tổng hợp
    table_res = Table(title="BẢNG KẾT QUẢ TỔNG HỢP", box=box.ROUNDED)
    table_res.add_column("Bước (i)", justify="center", style="cyan")
    table_res.add_column("Thời gian (t)", justify="center")
    table_res.add_column("Nghiệm x(t)", justify="center", style="green")

    for i, (t, x) in enumerate(zip(t_res, x_res)):
        table_res.add_row(str(i), f"{t:.2f}", f"{x:.6f}")

    console.print("\n")
    console.print(table_res)

    # --- 3.4. XUẤT CSV ---
    df = pd.DataFrame({'t': t_res, 'x': x_res})
    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    df.to_csv(csv_filename, index=False)
    console.print(f"\n[bold green]✔ Đã lưu kết quả vào file: {csv_filename}[/bold green]")

    # --- 3.5. VẼ ĐỒ THỊ ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_res, x_res, marker='o', linestyle='-', color='b', label='Euler Ẩn')
    
    # Vẽ thêm đường nghiệm chính xác (nếu biết) để so sánh - Bài này y' = -2y => y = e^(-2t)
    t_exact = np.linspace(T0, T_END, 100)
    x_exact = X0 * np.exp(-2 * t_exact)
    plt.plot(t_exact, x_exact, linestyle='--', color='r', label='Nghiệm chính xác (Analytical)', alpha=0.7)

    plt.title(f"Giải ODE bằng Euler Ẩn: x' = {FUNCTION_EXPRESSION}, h={H}")
    plt.xlabel("Thời gian (t)")
    plt.ylabel("Giá trị nghiệm x(t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    img_filename = os.path.join(output_dir, 'graph_ODE_1D_Euler_Implicit.png')
    plt.savefig(img_filename)
    console.print(f"[bold green]✔ Đã lưu đồ thị vào file: {img_filename}[/bold green]")
    # plt.show() # Bỏ comment nếu muốn hiện cửa sổ đồ thị

if __name__ == "__main__":
    main()