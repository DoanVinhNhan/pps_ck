import numpy as np

def solve_ode_euler_forward_1d(f, t0, x0, h, T):
    """
    Giải phương trình vi phân thường 1D (ODE 1D) bằng phương pháp Euler Hiện (Forward Euler).
    Bài toán: x' = f(t, x) với điều kiện ban đầu x(t0) = x0.
    
    Công thức: x_{i+1} = x_i + h * f(t_i, x_i)

    Args:
        f (callable): Hàm f(t, x) biểu diễn vế phải của phương trình vi phân.
        t0 (float): Thời điểm bắt đầu.
        x0 (float): Giá trị ban đầu x tại t0.
        h (float): Bước nhảy thời gian (step size).
        T (float): Thời điểm kết thúc (để xác định khoảng tính toán [t0, T]).

    Returns:
        dict: Dictionary chứa kết quả tính toán gồm:
            - 't': List các mốc thời gian [t0, t1, ..., tn].
            - 'x': List các giá trị nghiệm gần đúng [x0, x1, ..., xn].
    """
    # Tính số bước lặp cần thiết
    # Sử dụng np.ceil để đảm bảo đi hết khoảng [t0, T]
    num_steps = int(np.ceil((T - t0) / h))

    # Khởi tạo danh sách lưu trữ kết quả
    t_values = [t0]
    x_values = [x0]

    # Gán giá trị hiện tại
    t_curr = t0
    x_curr = x0

    # Vòng lặp tính toán theo từng bước
    for _ in range(num_steps):
        # Tính đạo hàm tại điểm hiện tại
        slope = f(t_curr, x_curr)
        
        # Cập nhật giá trị x theo công thức Euler hiện
        x_next = x_curr + h * slope
        
        # Cập nhật thời gian
        t_next = t_curr + h
        
        # Lưu kết quả vào danh sách
        t_values.append(t_next)
        x_values.append(x_next)
        
        # Cập nhật biến cho vòng lặp tiếp theo
        t_curr = t_next
        x_curr = x_next

    return {
        "t": t_values,
        "x": x_values,
        "convergence_info": {
            "method_name": "Euler Explicit (Forward Euler)",
            "approximation_order": "O(h)",
            "stability_region": "|1 + z| <= 1 (Hình tròn tâm (-1, 0) bán kính 1 trên mặt phẳng phức)",
            "stability_function": "R(z) = 1 + z",
            "conditionally_stable": True
        }
    }