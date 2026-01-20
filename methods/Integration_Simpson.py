import numpy as np

def simpson_integration(x_nodes, y_nodes, a, b, g=None, epsilon=None):
    """
    Thực hiện tính tích phân Simpson (Simpson's Rule) cho dữ liệu rời rạc hoặc hàm số
    với khả năng nội suy và đánh giá sai số theo nguyên lý Runge.

    Phương pháp:
        I ≈ (h/3) * (y_0 + 4*sigma_1 + 2*sigma_2 + y_2n)

    Args:
        x_nodes (list/array): Danh sách các mốc x (dữ liệu rời rạc).
        y_nodes (list/array): Danh sách các giá trị f(x) tương ứng.
        a (float): Cận dưới tích phân.
        b (float): Cận trên tích phân.
        g (callable, optional): Hàm g(f_val, x_val) biểu diễn hàm dưới dấu tích phân. 
                                Mặc định là None (tích phân f(x)).
        epsilon (float, optional): Sai số cho phép. Nếu được cung cấp, thuật toán sẽ 
                                   lặp và chia đôi bước lưới h cho đến khi thỏa mãn sai số.

    Returns:
        dict: Kết quả bao gồm giá trị tích phân, sai số ước lượng, các giá trị trung gian
              và nhật ký quá trình tính toán.
    """
    
    # 1. Chuẩn bị dữ liệu và hàm
    x_data = np.array(x_nodes, dtype=float)
    y_data = np.array(y_nodes, dtype=float)
    
    # Đảm bảo dữ liệu được sắp xếp
    sort_indices = np.argsort(x_data)
    x_data = x_data[sort_indices]
    y_data = y_data[sort_indices]

    if g is None:
        func_g = lambda f, x: f
    else:
        func_g = g

    computation_log = []
    
    # Hàm nội suy cục bộ theo yêu cầu: 6 điểm
    def get_interpolated_f(x_query, x_arr, y_arr):
        """
        Nội suy giá trị f tại x_query sử dụng 6 điểm lân cận.
        - Vùng biên trái: Newton Forward (chọn 6 điểm đầu).
        - Vùng biên phải: Newton Backward (chọn 6 điểm cuối).
        - Vùng giữa: Lagrange trung tâm (6 điểm quanh x).
        """
        n_points = len(x_arr)
        if n_points < 6:
            raise ValueError("Cần ít nhất 6 điểm dữ liệu để thực hiện nội suy theo yêu cầu.")
        
        # Tìm vị trí của x_query trong mảng x_arr
        # idx là vị trí mà x_arr[idx-1] <= x_query < x_arr[idx]
        idx = np.searchsorted(x_arr, x_query)
        
        # Xác định cửa sổ 6 điểm (indices)
        if idx <= 3:
            # Vùng biên trái: Không đủ điểm bên trái để lấy trung tâm -> Lấy 6 điểm đầu
            # Tương đương logic Newton Forward (mở rộng về phía trước từ x0)
            window_indices = range(0, 6)
        elif idx >= n_points - 3:
            # Vùng biên phải: Không đủ điểm bên phải -> Lấy 6 điểm cuối
            # Tương đương logic Newton Backward (mở rộng về phía sau từ xn)
            window_indices = range(n_points - 6, n_points)
        else:
            # Vùng giữa: Lấy 3 điểm trái, 3 điểm phải quanh vị trí tìm được
            # Tương đương Lagrange trung tâm
            window_indices = range(idx - 3, idx + 3)
            
        # Lấy dữ liệu cục bộ
        x_loc = x_arr[window_indices]
        y_loc = y_arr[window_indices]
        
        # Tính toán đa thức nội suy Lagrange trên cửa sổ đã chọn
        # (Lưu ý: Đa thức nội suy Newton và Lagrange trên cùng một tập điểm là duy nhất)
        res = 0.0
        for i in range(6):
            term = y_loc[i]
            for j in range(6):
                if i != j:
                    term *= (x_query - x_loc[j]) / (x_loc[i] - x_loc[j])
            res += term
        return res

    def calculate_integrand(x_val):
        # Tính f(x) qua nội suy
        f_val = get_interpolated_f(x_val, x_data, y_data)
        # Tính g(f(x), x)
        return func_g(f_val, x_val)

    computation_log.append(f"Bắt đầu tính tích phân Simpson trên đoạn [{a}, {b}].")

    # 2. Xác định số khoảng chia N ban đầu
    # Nếu không có epsilon, ta tính N dựa trên mật độ dữ liệu gốc trong khoảng [a, b]
    # Simpson yêu cầu N chẵn (số khoảng chia).
    if epsilon is None:
        points_in_range = np.sum((x_data >= a) & (x_data <= b))
        N = max(2, int(points_in_range))
        if N % 2 != 0:
            N += 1
        computation_log.append(f"Không có yêu cầu epsilon. Chọn số khoảng chia N = {N} dựa trên dữ liệu đầu vào.")
    else:
        # Nếu có epsilon, vẫn bắt đầu với số khoảng chia dựa trên mật độ dữ liệu
        # nhưng đảm bảo N chẵn để phù hợp Simpson
        points_in_range = np.sum((x_data >= a) & (x_data <= b))
        # Initial N should be based on data points count - 1 (intervals), approx same density
        # Minimum 2 intervals
        n_intervals = max(2, int(points_in_range) - 1)
        
        N = n_intervals
        if N % 2 != 0:
            N += 1
            computation_log.append(f"  -> N ban đầu = {n_intervals} (số khoảng lẻ).")
            computation_log.append(f"  -> Điều chỉnh N = {N} (tăng 1) để thỏa mãn điều kiện Simpson (N chẵn).")
            computation_log.append(f"  -> Bước h thay đổi từ {(b-a)/n_intervals:.6f} thành {(b-a)/N:.6f}.")
            computation_log.append(f"  -> Cảnh báo: Thực hiện nội suy để lấy các giá trị mới không có trong bảng dữ liệu!")
            
        computation_log.append(f"Có yêu cầu epsilon = {epsilon}. Bắt đầu lặp với N = {N} (dựa trên mật độ dữ liệu).")

    current_result = 0.0
    previous_result = None
    error_est = float('inf')
    

    intermediate_vars = {
        "iteration_history": []
    }

    # 3. Vòng lặp tính toán (Iterative process)
    iteration = 0
    max_iterations = 20 # Tránh lặp vô hạn

    while True:
        iteration += 1
        h = (b - a) / N
        
        # Tạo lưới điểm tích phân
        x_grid = np.linspace(a, b, N + 1)
        
        # Tính giá trị hàm số tại các nút lưới
        y_grid = []
        for val in x_grid:
            y_grid.append(calculate_integrand(val))
        y_grid = np.array(y_grid)

        # Tính các tổng sigma
        # y_0 và y_{2n} (y_N)
        y_start = y_grid[0]
        y_end = y_grid[-1]
        
        # Sigma 1: chỉ số lẻ (1, 3, ..., N-1)
        sigma_1 = np.sum(y_grid[1:N:2])
        
        # Sigma 2: chỉ số chẵn (2, 4, ..., N-2)
        # Lưu ý: Python slice end là exclusive, nên N-1 để loại bỏ y_N (đã tính ở y_end)
        # Nếu N=2, range(2, 2, 2) là rỗng -> đúng
        sigma_2 = np.sum(y_grid[2:N:2])
        
        # Công thức Simpson: I ≈ (h / 3) * (y_0 + 4 * sigma_1 + 2 * sigma_2 + y_{2n})
        current_result = (h / 3) * (y_start + 4 * sigma_1 + 2 * sigma_2 + y_end)
        
        log_entry = (f"Lần lặp {iteration}: N={N}, h={h:.6f}, "
                     f"Sigma1={sigma_1:.5f}, Sigma2={sigma_2:.5f}, I={current_result:.8f}")
        computation_log.append(log_entry)
        
        # Prepare history entry (error updated later)
        history_entry = {
            "iter": iteration,
            "N": N,
            "h": h,
            "result": current_result,
            "error": None
        }

        # Lưu giá trị trung gian của lần lặp cuối cùng (hoặc quan trọng nhất)
        intermediate_vars.update({
            "N": N,
            "h": h,
            "y_start": y_start,
            "y_end": y_end,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "grid_samples": list(zip(x_grid, y_grid))
        })
        
        # Detailed table logic
        # We only need it for the first iteration as requested
        if iteration == 1:
            detailed_table = []
            current_h = h
            num_points = len(x_grid)
            
            common_factor = current_h / 3.0
            
            for i in range(num_points):
                x_val = x_grid[i]
                g_val = y_grid[i] 
                f_val = get_interpolated_f(x_val, x_data, y_data)
                
                # Raw Weight logic (Simpson: 1, 4, 2, ..., 4, 1)
                base_coeff = 0
                if i == 0 or i == num_points - 1:
                    base_coeff = 1
                elif i % 2 != 0: # Odd index
                    base_coeff = 4
                else: # Even index
                    base_coeff = 2
                
                term = base_coeff * g_val
                
                detailed_table.append({
                    "i": i,
                    "x": x_val,
                    "f": f_val,
                    "g": g_val,
                    "C": base_coeff, # Raw weight
                    "term": term
                })
                
            intermediate_vars["initial_detailed_table"] = detailed_table
            intermediate_vars["initial_common_factor"] = common_factor

        # Add to history list
        intermediate_vars["iteration_history"].append(history_entry)

        # Kiểm tra điều kiện dừng
        if epsilon is None:
            # Chế độ tính 1 lần
            error_est = 0.0 # Không đánh giá sai số lặp
            computation_log.append("Hoàn thành tính toán (chế độ không lặp).")
            break
        else:
            if previous_result is not None:
                # Đánh giá sai số theo Runge cho Simpson (O(h^4)): |I_h - I_{2h}| / 15
                # Ở đây current là I_h (mịn hơn), previous là I_{2h} (thô hơn)
                # Chú ý: trong code này mình tăng N gấp đôi mỗi lần, nghĩa là bước h giảm một nửa.
                # current là I_{h/2} (so với bước trước), previous là I_h
                # Công thức Runge: |I_{min} - I_{tho}| / (2^p - 1)
                runge_error = abs(current_result - previous_result) / 15.0
                error_est = runge_error
                
                # Update error in current history entry
                history_entry["error"] = runge_error
                
                computation_log.append(f"  -> Sai số Runge ước lượng: {runge_error:.8e}")
                
                if runge_error < epsilon:
                    computation_log.append(f"Đã đạt độ chính xác yêu cầu (< {epsilon}). Dừng.")
                    break
            
            if iteration >= max_iterations:
                computation_log.append("Đã đạt số lần lặp tối đa. Dừng.")
                break
            
            # Chuẩn bị cho vòng lặp sau: Tăng gấp đôi số khoảng chia
            previous_result = current_result
            N *= 2

    return {
        "result": current_result,
        "h": h, # Return final h step size
        "error_estimate": error_est,
        "error_estimate": error_est,
        "intermediate_values": intermediate_vars,
        "computation_process": computation_log
    }