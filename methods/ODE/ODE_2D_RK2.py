import numpy as np

def ode_2d_rk2(f, g, t0, x0, y0, h, T):
    """
    Giải hệ phương trình vi phân thường (2D ODE) bằng phương pháp Runge-Kutta bậc 2 (RK2 - Phương pháp Heun).
    
    Hệ phương trình:
        dx/dt = f(t, x, y)
        dy/dt = g(t, x, y)

    INPUT:
        f  : Hàm f(t, x, y) tương ứng với x'.
        g  : Hàm g(t, x, y) tương ứng với y'.
        t0 : Thời điểm đầu.
        x0 : Giá trị ban đầu x(t0).
        y0 : Giá trị ban đầu y(t0).
        h  : Bước nhảy thời gian.
        T  : Thời điểm kết thúc.

    OUTPUT:
        t_values : List các giá trị thời gian t_i.
        x_values : List các giá trị x_i tương ứng.
        y_values : List các giá trị y_i tương ứng.
    """
    
    # Tính số bước lặp
    # Sử dụng np.round để xử lý sai số dấu phẩy động khi chia
    num_steps = int(np.round((T - t0) / h))
    
    # Khởi tạo các danh sách lưu kết quả
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    
    # Gán giá trị hiện tại
    t = t0
    x = x0
    y = y0
    
    for _ in range(num_steps):
        # --- Bước 1: Tính hệ số K1 (độ dốc tại đầu khoảng) ---
        k1_x = f(t, x, y)
        k1_y = g(t, x, y)
        
        # --- Bước 2: Tính hệ số K2 (độ dốc tại cuối khoảng - Phương pháp Heun) ---
        # Dự báo giá trị tại t + h
        t_next_pred = t + h
        x_pred = x + h * k1_x
        y_pred = y + h * k1_y
        
        k2_x = f(t_next_pred, x_pred, y_pred)
        k2_y = g(t_next_pred, x_pred, y_pred)
        
        # --- Bước 3: Cập nhật giá trị tiếp theo (Trung bình cộng độ dốc) ---
        x_next = x + (h / 2.0) * (k1_x + k2_x)
        y_next = y + (h / 2.0) * (k1_y + k2_y)
        t_next = t + h
        
        # Lưu kết quả vào danh sách
        t_values.append(t_next)
        x_values.append(x_next)
        y_values.append(y_next)
        
        # Cập nhật biến cho vòng lặp sau
        t = t_next
        x = x_next
        y = y_next
        
    return t_values, x_values, y_values