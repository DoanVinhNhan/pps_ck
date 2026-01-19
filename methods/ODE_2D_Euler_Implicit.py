import numpy as np

def solve_ode_2d_implicit_euler(f, g, t0, x0, y0, h, T):
    """
    Giải hệ phương trình vi phân thường 2D bằng phương pháp Euler Ẩn (Implicit Euler).
    Hệ phương trình:
        x' = f(t, x, y)
        y' = g(t, x, y)
    
    Phương pháp:
        t_{n+1} = t_n + h
        x_{n+1} = x_n + h * f(t_{n+1}, x_{n+1}, y_{n+1})
        y_{n+1} = y_n + h * g(t_{n+1}, x_{n+1}, y_{n+1})
    
    Do vế phải chứa x_{n+1}, y_{n+1} nên cần giải hệ phương trình đại số.
    Hàm này sử dụng phương pháp Lặp điểm bất động (Fixed-Point Iteration) để tìm nghiệm ẩn.

    Args:
        f (callable): Hàm f(t, x, y).
        g (callable): Hàm g(t, x, y).
        t0 (float): Thời điểm bắt đầu.
        x0 (float): Giá trị ban đầu của x tại t0.
        y0 (float): Giá trị ban đầu của y tại t0.
        h (float): Bước nhảy thời gian.
        T (float): Thời điểm kết thúc.

    Returns:
        dict: Dictionary chứa các list kết quả:
            - 't': List các mốc thời gian.
            - 'x': List các giá trị x tương ứng.
            - 'y': List các giá trị y tương ứng.
    """
    
    # Khởi tạo danh sách lưu kết quả
    t_list = [t0]
    x_list = [x0]
    y_list = [y0]
    
    # Tính số bước lặp
    # Sử dụng round để tránh lỗi làm tròn số thực
    num_steps = int(np.round((T - t0) / h))
    
    t_curr = t0
    x_curr = x0
    y_curr = y0
    
    # Các tham số cho vòng lặp giải phương trình ẩn (Fixed-Point Iteration)
    max_iter = 100      # Số lần lặp tối đa
    tol = 1e-6          # Sai số cho phép để hội tụ
    
    for _ in range(num_steps):
        t_next = t_curr + h
        
        # --- BƯỚC DỰ BÁO (PREDICTOR) ---
        # Sử dụng Euler hiện để lấy giá trị khởi tạo cho vòng lặp
        x_next = x_curr + h * f(t_curr, x_curr, y_curr)
        y_next = y_curr + h * g(t_curr, x_curr, y_curr)
        
        # --- BƯỚC HIỆU CHỈNH (CORRECTOR) ---
        # Giải phương trình ẩn: u = u_old + h * F(t_new, u)
        for _ in range(max_iter):
            x_old_iter = x_next
            y_old_iter = y_next
            
            # Tính lại giá trị dựa trên công thức Euler ẩn
            x_next = x_curr + h * f(t_next, x_next, y_next)
            y_next = y_curr + h * g(t_next, x_next, y_next)
            
            # Kiểm tra điều kiện hội tụ
            if abs(x_next - x_old_iter) < tol and abs(y_next - y_old_iter) < tol:
                break
        
        # Cập nhật giá trị hiện tại
        t_curr = t_next
        x_curr = x_next
        y_curr = y_next
        
        # Lưu kết quả
        t_list.append(t_curr)
        x_list.append(x_curr)
        y_list.append(y_curr)
        
    return {
        "t": t_list,
        "x": x_list,
        "y": y_list
    }