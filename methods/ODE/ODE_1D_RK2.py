import numpy as np

def rk2_ode_1d(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1D (ODE 1D) bằng phương pháp Runge-Kutta bậc 2 (RK2).
    Biến thể sử dụng: Phương pháp Heun (Heun's Method / Improved Euler).
    Bài toán: x' = f(t, x) với điều kiện đầu x(t0) = x0.

    Parameters:
    -----------
    f : callable
        Hàm số f(t, x) biểu diễn vế phải của phương trình vi phân.
        Nhận vào 2 tham số (t, x) và trả về đạo hàm dx/dt.
    t0 : float
        Thời điểm bắt đầu.
    x0 : float
        Giá trị ban đầu của x tại t0.
    h : float
        Bước nhảy thời gian (step size).
    T : float
        Thời điểm kết thúc.

    Returns:
    --------
    dict
        Dictionary chứa kết quả tính toán:
        - 't': List các mốc thời gian [t0, t1, ..., tn].
        - 'x': List các giá trị nghiệm gần đúng [x0, x1, ..., xn].
    """
    
    # Tính số bước lặp
    # Sử dụng round để xử lý sai số dấu phẩy động khi chia
    num_steps = int(np.round((T - t0) / h))
    
    # Khởi tạo mảng lưu trữ kết quả
    t_values = np.zeros(num_steps + 1)
    x_values = np.zeros(num_steps + 1)
    
    # Gán giá trị ban đầu
    t_values[0] = t0
    x_values[0] = x0
    
    # Vòng lặp tính toán
    for i in range(num_steps):
        t_curr = t_values[i]
        x_curr = x_values[i]
        
        # Tính k1: Độ dốc tại đầu khoảng (t_n, x_n)
        k1 = f(t_curr, x_curr)
        
        # Tính k2: Độ dốc tại cuối khoảng dự báo (t_n + h, x_n + h*k1)
        # Đây là đặc trưng của phương pháp Heun (alpha = 1)
        t_next_pred = t_curr + h
        x_next_pred = x_curr + h * k1
        k2 = f(t_next_pred, x_next_pred)
        
        # Cập nhật giá trị x tiếp theo bằng trung bình trọng số của k1 và k2
        x_next = x_curr + (h / 2.0) * (k1 + k2)
        
        # Lưu kết quả vào mảng
        x_values[i+1] = x_next
        # Tính t dựa trên chỉ số để tránh tích lũy sai số làm tròn
        t_values[i+1] = t0 + (i + 1) * h
        
    return {
        "t": t_values.tolist(),
        "x": x_values.tolist(),
        "convergence_info": {
            "method_name": "Runge-Kutta Order 2 (Heun's Method)",
            "approximation_order": "O(h^2)",
            "stability_region": "|1 + z + z^2/2| <= 1",
            "stability_function": "R(z) = 1 + z + 0.5*z^2",
            "conditionally_stable": True
        }
    }