import numpy as np

def ode_1d_implicit_euler(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1D (ODE 1D) bằng phương pháp Euler Ẩn (Backward Euler).
    Phương trình: x' = f(t, x)
    Công thức: x_{i+1} = x_i + h * f(t_{i+1}, x_{i+1})
    
    Do x_{i+1} xuất hiện ở cả hai vế, ta sử dụng phương pháp lặp điểm bất động (Fixed-point iteration)
    để tìm nghiệm gần đúng tại mỗi bước.

    Parameters:
    -----------
    f : callable
        Hàm số f(t, x) tương ứng với x' = f(t, x).
    t0 : float
        Thời điểm bắt đầu.
    x0 : float
        Giá trị ban đầu x(t0).
    h : float
        Bước nhảy thời gian.
    T : float
        Thời điểm kết thúc.

    Returns:
    --------
    t_values : list
        Danh sách các mốc thời gian t_i.
    x_values : list
        Danh sách các giá trị nghiệm x_i tương ứng.
    """
    
    # Xác định số bước lặp
    num_steps = int(np.ceil((T - t0) / h))
    
    # Khởi tạo danh sách kết quả
    t_values = [t0]
    x_values = [x0]
    
    # Các tham số cho vòng lặp giải phương trình ẩn
    max_iter = 100      # Số lần lặp tối đa để tìm nghiệm ẩn
    tolerance = 1e-6    # Sai số cho phép của nghiệm ẩn
    
    t_current = t0
    x_current = x0
    
    for _ in range(num_steps):
        t_next = t_current + h
        
        # Bước 1: Dự báo giá trị ban đầu cho x_{i+1} (Predictor)
        # Sử dụng Euler hiện để có giá trị khởi tạo tốt cho vòng lặp
        x_next_guess = x_current + h * f(t_current, x_current)
        
        # Bước 2: Giải phương trình ẩn x_{i+1} = x_i + h * f(t_{i+1}, x_{i+1})
        # Sử dụng phương pháp lặp đơn (Fixed-point iteration)
        for _ in range(max_iter):
            # Tính giá trị mới dựa trên công thức Euler ẩn
            x_next_new = x_current + h * f(t_next, x_next_guess)
            
            # Kiểm tra hội tụ
            if abs(x_next_new - x_next_guess) < tolerance:
                x_next_guess = x_next_new
                break
            
            x_next_guess = x_next_new
        
        # Cập nhật giá trị cho bước tiếp theo
        x_current = x_next_guess
        t_current = t_next
        
        # Lưu kết quả
        t_values.append(t_current)
        x_values.append(x_current)
        
    return {
        "t": t_values,
        "x": x_values,
        "convergence_info": {
            "method_name": "Euler Implicit (Backward Euler)",
            "approximation_order": "O(h)",
            "stability_region": "|1 - z| >= 1 (Miền ổn định tuyệt đối là bên ngoài hình tròn mở tâm (1, 0) bán kính 1)",
            "stability_function": "R(z) = 1 / (1 - z)",
            "unconditionally_stable": True
        }
    }