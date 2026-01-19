import numpy as np

def ode_2d_rk3(f, g, t0, x0, y0, h, T):
    """
    Giải hệ phương trình vi phân thường cấp 1 (2 phương trình) bằng phương pháp Runge-Kutta bậc 3 (RK3).
    Hệ phương trình:
        dx/dt = f(t, x, y)
        dy/dt = g(t, x, y)

    Input:
        f: Hàm f(t, x, y) (callable).
        g: Hàm g(t, x, y) (callable).
        t0: Thời điểm bắt đầu (float).
        x0: Giá trị ban đầu của x tại t0 (float).
        y0: Giá trị ban đầu của y tại t0 (float).
        h: Bước nhảy thời gian (float).
        T: Thời điểm kết thúc (float).

    Output:
        Dictionary chứa các list giá trị:
        {
            "t": [t0, t1, ...],
            "x": [x0, x1, ...],
            "y": [y0, y1, ...]
        }
    """
    # Khởi tạo các danh sách lưu kết quả
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    # Gán giá trị hiện tại
    t_curr = t0
    x_curr = x0
    y_curr = y0

    # Tính số bước lặp
    # Sử dụng round để tránh lỗi làm tròn số thực khi chia
    num_steps = int(np.round((T - t0) / h))

    for _ in range(num_steps):
        # Tính các hệ số k (cho x) và l (cho y) theo công thức RK3 (Kutta's 3rd order)
        
        # Bước 1: Tại t_n
        k1 = h * f(t_curr, x_curr, y_curr)
        l1 = h * g(t_curr, x_curr, y_curr)

        # Bước 2: Tại t_n + h/2
        k2 = h * f(t_curr + h/2, x_curr + k1/2, y_curr + l1/2)
        l2 = h * g(t_curr + h/2, x_curr + k1/2, y_curr + l1/2)

        # Bước 3: Tại t_n + h (sử dụng k1, k2, l1, l2 để dự đoán điểm cuối)
        # Công thức: x_curr - k1 + 2*k2
        k3 = h * f(t_curr + h, x_curr - k1 + 2*k2, y_curr - l1 + 2*l2)
        l3 = h * g(t_curr + h, x_curr - k1 + 2*k2, y_curr - l1 + 2*l2)

        # Cập nhật giá trị tiếp theo
        x_next = x_curr + (1.0/6.0) * (k1 + 4*k2 + k3)
        y_next = y_curr + (1.0/6.0) * (l1 + 4*l2 + l3)
        t_next = t_curr + h

        # Lưu kết quả
        t_values.append(t_next)
        x_values.append(x_next)
        y_values.append(y_next)

        # Cập nhật biến cho vòng lặp sau
        t_curr = t_next
        x_curr = x_next
        y_curr = y_next

    return {
        "t": t_values,
        "x": x_values,
        "y": y_values
    }