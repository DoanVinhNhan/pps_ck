import numpy as np

def solve_ode_1d_rk4(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1 chiều (1D ODE) bằng phương pháp Runge-Kutta bậc 4 (RK4).
    Bài toán: x' = f(t, x) với điều kiện ban đầu x(t0) = x0.

    Args:
        f (callable): Hàm f(t, x) trả về đạo hàm dx/dt.
        t0 (float): Thời điểm bắt đầu.
        x0 (float): Giá trị ban đầu của x tại t0.
        h (float): Bước nhảy thời gian (step size).
        T (float): Thời điểm kết thúc.

    Returns:
        dict: Dictionary chứa hai list:
            - 't': Danh sách các mốc thời gian t_i.
            - 'x': Danh sách các giá trị nghiệm x_i tương ứng.
    """
    # Khởi tạo danh sách lưu trữ kết quả với giá trị ban đầu
    t_values = [t0]
    x_values = [x0]

    # Gán các biến tạm thời
    t_current = t0
    x_current = x0

    # Tính số bước lặp dự kiến
    # Sử dụng np.ceil để đảm bảo phủ kín khoảng [t0, T]
    num_steps = int(np.ceil((T - t0) / h))

    for _ in range(num_steps):
        # Tính toán 4 hệ số K của phương pháp RK4
        # k1: Độ dốc tại đầu khoảng
        k1 = h * f(t_current, x_current)
        
        # k2: Độ dốc tại điểm giữa khoảng (dựa trên k1)
        k2 = h * f(t_current + 0.5 * h, x_current + 0.5 * k1)
        
        # k3: Độ dốc tại điểm giữa khoảng (dựa trên k2)
        k3 = h * f(t_current + 0.5 * h, x_current + 0.5 * k2)
        
        # k4: Độ dốc tại cuối khoảng (dựa trên k3)
        k4 = h * f(t_current + h, x_current + k3)

        # Cập nhật giá trị x tiếp theo dựa trên trung bình có trọng số của các k
        x_next = x_current + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
        # Cập nhật thời gian t tiếp theo
        t_next = t_current + h

        # Lưu kết quả vào danh sách
        t_values.append(t_next)
        x_values.append(x_next)

        # Cập nhật biến hiện tại cho vòng lặp sau
        t_current = t_next
        x_current = x_next
        
        # Kiểm tra điều kiện dừng nếu t vượt quá T (do sai số làm tròn số thực)
        if t_current >= T:
            break

    return {
        "t": t_values,
        "x": x_values,
        "convergence_info": {
            "method_name": "Runge-Kutta Order 4 (Classic RK4)",
            "approximation_order": "O(h^4)",
            "stability_region": "|1 + z + z^2/2 + z^3/6 + z^4/24| <= 1",
            "stability_function": "R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24",
            "conditionally_stable": True
        }
    }