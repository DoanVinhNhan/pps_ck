import numpy as np

def solve_ode_2d_ab_am(f, g, t0, x0, y0, h, T, s=4):
    """
    Giải hệ phương trình vi phân thường 2D bằng phương pháp Adams-Bashforth-Moulton (Dự báo - Hiệu chỉnh).
    
    Phương pháp:
    1. Khởi tạo s-1 giá trị đầu tiên bằng RK4.
    2. Dự báo (Predictor) bằng Adams-Bashforth s-step.
    3. Hiệu chỉnh (Corrector) bằng Adams-Moulton s-step.
    
    Input:
        f: Hàm f(t, x, y) tương ứng x' = f(t, x, y).
        g: Hàm g(t, x, y) tương ứng y' = g(t, x, y).
        t0: Thời điểm đầu.
        x0: Giá trị ban đầu x(t0).
        y0: Giá trị ban đầu y(t0).
        h: Bước nhảy.
        T: Thời điểm kết thúc.
        s: Số bước (bậc) của phương pháp (mặc định là 4). Hỗ trợ s=2, 3, 4.
        
    Output:
        Dictionary chứa:
        - 't': List các mốc thời gian.
        - 'x': List các giá trị x tương ứng.
        - 'y': List các giá trị y tương ứng.
    """
    
    # Xác định số bước
    num_steps = int(np.ceil((T - t0) / h))
    
    # Khởi tạo mảng lưu trữ
    t_values = np.zeros(num_steps + 1)
    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    
    t_values[0] = t0
    x_values[0] = x0
    y_values[0] = y0
    
    # --- GIAI ĐOẠN 1: KHỞI TẠO (BOOTSTRAP) BẰNG RK4 ---
    # Cần s giá trị đầu tiên (từ index 0 đến s-1) để bắt đầu ABs
    
    # Nếu tổng số bước nhỏ hơn s, chỉ chạy RK4
    limit_rk4 = min(s - 1, num_steps)
    
    for i in range(limit_rk4):
        ti = t_values[i]
        xi = x_values[i]
        yi = y_values[i]
        
        # Tính k cho x
        k1_x = h * f(ti, xi, yi)
        k1_y = h * g(ti, xi, yi)
        
        k2_x = h * f(ti + 0.5*h, xi + 0.5*k1_x, yi + 0.5*k1_y)
        k2_y = h * g(ti + 0.5*h, xi + 0.5*k1_x, yi + 0.5*k1_y)
        
        k3_x = h * f(ti + 0.5*h, xi + 0.5*k2_x, yi + 0.5*k2_y)
        k3_y = h * g(ti + 0.5*h, xi + 0.5*k2_x, yi + 0.5*k2_y)
        
        k4_x = h * f(ti + h, xi + k3_x, yi + k3_y)
        k4_y = h * g(ti + h, xi + k3_x, yi + k3_y)
        
        x_values[i+1] = xi + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
        y_values[i+1] = yi + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
        t_values[i+1] = t0 + (i + 1) * h

    # --- CẤU HÌNH HỆ SỐ CHO ABs VÀ AMs ---
    # Hệ số sắp xếp từ xa nhất (n-s+1) đến gần nhất (n)
    if s == 2:
        # AB2: y_{n+1} = y_n + h/2 * (3f_n - 1f_{n-1})
        ab_coeffs = np.array([-1, 3]) / 2.0
        # AM2: y_{n+1} = y_n + h/2 * (1f_{n+1} + 1f_n) -> Trapezoidal
        am_coeffs = np.array([1, 1]) / 2.0 
        
    elif s == 3:
        # AB3: y_{n+1} = y_n + h/12 * (23f_n - 16f_{n-1} + 5f_{n-2})
        ab_coeffs = np.array([5, -16, 23]) / 12.0
        # AM3: y_{n+1} = y_n + h/12 * (5f_{n+1} + 8f_n - 1f_{n-1})
        am_coeffs = np.array([-1, 8, 5]) / 12.0
        
    elif s == 4:
        # AB4: y_{n+1} = y_n + h/24 * (55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})
        ab_coeffs = np.array([-9, 37, -59, 55]) / 24.0
        # AM4: y_{n+1} = y_n + h/24 * (9f_{n+1} + 19f_n - 5f_{n-1} + 1f_{n-2})
        am_coeffs = np.array([1, -5, 19, 9]) / 24.0
    else:
        raise ValueError("Chỉ hỗ trợ s = 2, 3, 4")

    # --- GIAI ĐOẠN 2: DỰ BÁO - HIỆU CHỈNH (AB-AM) ---
    # Bắt đầu từ bước s-1 để tính bước s
    for i in range(s - 1, num_steps):
        # 1. Chuẩn bị dữ liệu lịch sử (f_hist, g_hist)
        # Lấy s điểm từ i-(s-1) đến i
        f_hist = []
        g_hist = []
        for k in range(s):
            idx = i - (s - 1) + k
            val_f = f(t_values[idx], x_values[idx], y_values[idx])
            val_g = g(t_values[idx], x_values[idx], y_values[idx])
            f_hist.append(val_f)
            g_hist.append(val_g)
        
        f_hist = np.array(f_hist)
        g_hist = np.array(g_hist)
        
        # 2. Dự báo (Predictor - Adams-Bashforth)
        # x_{n+1}^P = x_n + h * sum(beta_j * f_{n-j})
        x_pred = x_values[i] + h * np.dot(ab_coeffs, f_hist)
        y_pred = y_values[i] + h * np.dot(ab_coeffs, g_hist)
        
        t_next = t_values[i] + h
        
        # 3. Tính đạo hàm tại điểm dự báo
        f_next_pred = f(t_next, x_pred, y_pred)
        g_next_pred = g(t_next, x_pred, y_pred)
        
        # 4. Hiệu chỉnh (Corrector - Adams-Moulton)
        # Cần lịch sử mới cho AM: [f_{n-(s-2)}, ..., f_n, f_{n+1}^P]
        # Lấy s-1 phần tử cuối của lịch sử cũ + giá trị dự báo mới
        f_hist_am = np.append(f_hist[1:], f_next_pred)
        g_hist_am = np.append(g_hist[1:], g_next_pred)
        
        x_corr = x_values[i] + h * np.dot(am_coeffs, f_hist_am)
        y_corr = y_values[i] + h * np.dot(am_coeffs, g_hist_am)
        
        # 5. Lưu kết quả
        t_values[i+1] = t_next
        x_values[i+1] = x_corr
        y_values[i+1] = y_corr

    return {
        "t": t_values.tolist(),
        "x": x_values.tolist(),
        "y": y_values.tolist()
    }