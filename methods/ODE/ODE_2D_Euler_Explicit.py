import numpy as np

def euler_forward_2d(f, g, t0, x0, y0, h, T):
    """
    Giải hệ phương trình vi phân thường 2D (hoặc ODE bậc 2 đưa về hệ)
    bằng phương pháp Euler Hiện (Forward Euler).

    Hệ phương trình:
        dx/dt = f(t, x, y)
        dy/dt = g(t, x, y)

    Input:
        f: Hàm f(t, x, y) (callable), tương ứng với x'.
        g: Hàm g(t, x, y) (callable), tương ứng với y'.
        t0: Thời điểm bắt đầu.
        x0: Giá trị ban đầu x(t0).
        y0: Giá trị ban đầu y(t0).
        h: Bước nhảy thời gian.
        T: Thời điểm kết thúc.

    Output:
        Dictionary chứa các list giá trị:
        - 't': List các mốc thời gian [t0, t1, ..., tn].
        - 'x': List các giá trị x tương ứng [x0, x1, ..., xn].
        - 'y': List các giá trị y tương ứng [y0, y1, ..., yn].
    """
    
    # Tính số bước lặp cần thiết
    # Sử dụng np.ceil để đảm bảo đi hết khoảng [t0, T]
    num_steps = int(np.ceil((T - t0) / h))

    # Khởi tạo các list lưu trữ kết quả với giá trị ban đầu
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    # Khởi tạo các biến trạng thái hiện tại
    t_curr = t0
    x_curr = x0
    y_curr = y0

    for i in range(num_steps):
        # 1. Tính giá trị đạo hàm tại điểm hiện tại (t_n, x_n, y_n)
        val_f = f(t_curr, x_curr, y_curr)
        val_g = g(t_curr, x_curr, y_curr)

        # 2. Áp dụng công thức Euler hiện:
        # x_{n+1} = x_n + h * f(t_n, x_n, y_n)
        # y_{n+1} = y_n + h * g(t_n, x_n, y_n)
        x_next = x_curr + h * val_f
        y_next = y_curr + h * val_g

        # 3. Cập nhật thời gian
        # Tính t dựa trên index để giảm thiểu sai số làm tròn số thực cộng dồn
        t_next = t0 + (i + 1) * h

        # 4. Lưu kết quả vào list
        t_values.append(t_next)
        x_values.append(x_next)
        y_values.append(y_next)

        # 5. Cập nhật biến trạng thái cho vòng lặp tiếp theo
        t_curr = t_next
        x_curr = x_next
        y_curr = y_next

    return {
        "t": t_values,
        "x": x_values,
        "y": y_values
    }