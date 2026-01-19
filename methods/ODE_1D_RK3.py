import numpy as np

def solve_ode_1d_rk3(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1 chiều (ODE 1D) bằng phương pháp Runge-Kutta bậc 3 (RK3).
    Sử dụng công thức Kutta cổ điển (tương tự quy tắc Simpson).

    Bài toán: x' = f(t, x), x(t0) = x0

    Input:
        f  : Hàm số f(t, x) (callable).
        t0 : Thời điểm bắt đầu (float).
        x0 : Giá trị ban đầu x(t0) (float).
        h  : Bước nhảy thời gian (float).
        T  : Thời điểm kết thúc (float).

    Output:
        Dictionary chứa:
            - 't': List các mốc thời gian t_i.
            - 'x': List các giá trị nghiệm x_i tương ứng.
    """
    
    # Khởi tạo danh sách lưu kết quả
    t_values = [t0]
    x_values = [x0]
    
    t_curr = t0
    x_curr = x0
    
    # Vòng lặp tính toán cho đến khi đạt thời điểm T
    # Sử dụng 1e-9 để xử lý sai số dấu phẩy động
    while t_curr < T - 1e-9:
        # Đảm bảo bước nhảy cuối cùng không vượt quá T
        if t_curr + h > T:
            h = T - t_curr
            
        # Tính các hệ số k của RK3 (Công thức Kutta's 3rd order)
        # k1 = h * f(t, x)
        k1 = h * f(t_curr, x_curr)
        
        # k2 = h * f(t + h/2, x + k1/2)
        k2 = h * f(t_curr + h/2, x_curr + k1/2)
        
        # k3 = h * f(t + h, x - k1 + 2*k2)
        k3 = h * f(t_curr + h, x_curr - k1 + 2*k2)
        
        # Cập nhật giá trị x tiếp theo
        # x_{n+1} = x_n + (1/6) * (k1 + 4*k2 + k3)
        x_next = x_curr + (k1 + 4*k2 + k3) / 6.0
        
        # Cập nhật thời gian
        t_next = t_curr + h
        
        # Lưu kết quả
        t_values.append(t_next)
        x_values.append(x_next)
        
        # Cập nhật biến hiện tại cho vòng lặp sau
        t_curr = t_next
        x_curr = x_next
        
    return {
        "t": t_values,
        "x": x_values
    }