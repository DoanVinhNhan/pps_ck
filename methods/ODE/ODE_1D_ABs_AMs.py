import numpy as np

def solve_ode_1d_ab_am(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1D (ODE) bằng phương pháp 
    Adams-Bashforth-Moulton (Dự báo - Hiệu chỉnh) bậc 4.
    
    Phương pháp này yêu cầu 4 giá trị khởi tạo. 3 giá trị đầu tiên 
    sẽ được tính bằng phương pháp Runge-Kutta bậc 4 (RK4).

    Input:
        f: Hàm số f(t, x) tương ứng với x' = f(t, x).
        t0: Thời điểm bắt đầu.
        x0: Giá trị ban đầu x(t0).
        h: Bước nhảy thời gian.
        T: Thời điểm kết thúc.

    Output:
        t_values: List các mốc thời gian.
        x_values: List các giá trị nghiệm x tương ứng.
    """
    
    # Xác định số bước lặp
    num_steps = int(np.ceil((T - t0) / h))
    
    # Khởi tạo mảng lưu trữ kết quả
    t_values = np.zeros(num_steps + 1)
    x_values = np.zeros(num_steps + 1)
    
    # Gán giá trị ban đầu
    t_values[0] = t0
    x_values[0] = x0
    
    # --- Giai đoạn 1: Khởi động bằng RK4 (3 bước đầu tiên) ---
    # Cần có x0, x1, x2, x3 để bắt đầu AB4-AM4 tại bước thứ 4 (tính x4)
    
    for i in range(min(3, num_steps)):
        ti = t_values[i]
        xi = x_values[i]
        
        k1 = h * f(ti, xi)
        k2 = h * f(ti + 0.5 * h, xi + 0.5 * k1)
        k3 = h * f(ti + 0.5 * h, xi + 0.5 * k2)
        k4 = h * f(ti + h, xi + k3)
        
        x_values[i+1] = xi + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t_values[i+1] = t0 + (i + 1) * h

    # --- Giai đoạn 2: Adams-Bashforth-Moulton (AB4-AM4) ---
    # Dự báo (Predictor): Adams-Bashforth 4
    # Hiệu chỉnh (Corrector): Adams-Moulton 4
    
    for i in range(3, num_steps):
        # Lấy các giá trị t và x quá khứ
        t_n = t_values[i]
        t_n_1 = t_values[i-1]
        t_n_2 = t_values[i-2]
        t_n_3 = t_values[i-3]
        
        x_n = x_values[i]
        x_n_1 = x_values[i-1]
        x_n_2 = x_values[i-2]
        x_n_3 = x_values[i-3]
        
        # Tính giá trị hàm f tại các điểm quá khứ
        f_n = f(t_n, x_n)
        f_n_1 = f(t_n_1, x_n_1)
        f_n_2 = f(t_n_2, x_n_2)
        f_n_3 = f(t_n_3, x_n_3)
        
        # BƯỚC DỰ BÁO (Predictor - AB4)
        # x_{n+1}^* = x_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        x_next_pred = x_n + (h / 24.0) * (55 * f_n - 59 * f_n_1 + 37 * f_n_2 - 9 * f_n_3)
        
        t_next = t0 + (i + 1) * h
        
        # Tính f tại điểm dự báo
        f_next_pred = f(t_next, x_next_pred)
        
        # BƯỚC HIỆU CHỈNH (Corrector - AM4)
        # x_{n+1} = x_n + h/24 * (9*f_{n+1}^* + 19*f_n - 5*f_{n-1} + f_{n-2})
        x_next_corr = x_n + (h / 24.0) * (9 * f_next_pred + 19 * f_n - 5 * f_n_1 + f_n_2)
        
        # Lưu kết quả
        x_values[i+1] = x_next_corr
        t_values[i+1] = t_next

    return {
        "t": t_values.tolist(),
        "x": x_values.tolist(),
        "convergence_info": {
            "method_name": "Adams-Bashforth-Moulton (Predictor-Corrector) Order 4",
            "approximation_order": "O(h^4)",
            "stability_region": "Miền ổn định giới hạn, nằm bên trái trục ảo, lớn hơn AB4 nhưng nhỏ hơn AM4 (Implicit).",
            "stability_function": "Phức tạp, phụ thuộc vào đa thức đặc trưng của AB4 và AM4.",
            "conditionally_stable": True
        }
    }