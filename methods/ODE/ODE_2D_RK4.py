import numpy as np

def solve_ode_2d_rk4(f, g, t0, x0, y0, h, T):
    """
    Giải hệ phương trình vi phân thường (ODE) 2 chiều bằng phương pháp Runge-Kutta bậc 4 (RK4).
    Hệ phương trình:
        dx/dt = f(t, x, y)
        dy/dt = g(t, x, y)

    Args:
        f (callable): Hàm f(t, x, y) tương ứng với dx/dt.
        g (callable): Hàm g(t, x, y) tương ứng với dy/dt.
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
    # Tính số bước lặp
    # Sử dụng np.ceil để đảm bảo đi hết khoảng [t0, T]
    num_steps = int(np.ceil((T - t0) / h))

    # Khởi tạo các list lưu trữ kết quả
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    # Gán giá trị hiện tại
    t = t0
    x = x0
    y = y0

    for _ in range(num_steps):
        # Tính các hệ số k1
        k1_x = h * f(t, x, y)
        k1_y = h * g(t, x, y)

        # Tính các hệ số k2 (tại t + h/2)
        k2_x = h * f(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y)
        k2_y = h * g(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y)

        # Tính các hệ số k3 (tại t + h/2)
        k3_x = h * f(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y)
        k3_y = h * g(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y)

        # Tính các hệ số k4 (tại t + h)
        k4_x = h * f(t + h, x + k3_x, y + k3_y)
        k4_y = h * g(t + h, x + k3_x, y + k3_y)

        # Cập nhật giá trị tiếp theo
        x_next = x + (1.0 / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_next = y + (1.0 / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        
        # Cập nhật thời gian
        t = t + h

        # Cập nhật biến trạng thái cho vòng lặp sau
        x = x_next
        y = y_next

        # Lưu kết quả
        t_values.append(t)
        x_values.append(x)
        y_values.append(y)

    return {
        "t": t_values,
        "x": x_values,
        "y": y_values
    }