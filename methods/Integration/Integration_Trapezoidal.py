import numpy as np

def trapezoidal_integration(x_nodes, y_nodes, a, b, g=None, epsilon=None):
    """
    Thực hiện tính tích phân hình thang cho hàm hợp g(f(x), x) dựa trên dữ liệu rời rạc của f(x).
    
    Parameters:
    x_nodes (array-like): Các mốc x của dữ liệu f(x).
    y_nodes (array-like): Các giá trị f(x) tương ứng.
    a (float): Cận dưới tích phân.
    b (float): Cận trên tích phân.
    g (callable, optional): Hàm g(f, x) cần tích phân. Mặc định là g(f, x) = f.
    epsilon (float, optional): Sai số cho phép. Nếu được cung cấp, thuật toán sẽ lặp và chia đôi bước lưới.
    
    Returns:
    dict: Kết quả bao gồm giá trị tích phân, sai số ước lượng, các giá trị trung gian và quy trình tính toán.
    """
    
    # --- Helper Functions for Interpolation ---
    
    def _divided_diff(x, y):
        """Tính bảng tỷ hiệu Newton."""
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
        return coef[0, :]

    def _newton_poly(coef, x_data, x):
        """Tính giá trị đa thức Newton tại x."""
        n = len(x_data) - 1
        p = coef[n]
        for k in range(1, n + 1):
            p = coef[n - k] + (x - x_data[n - k]) * p
        return p

    def _lagrange_interpolation(x, x_data, y_data):
        """Nội suy Lagrange."""
        total = 0.0
        n = len(x_data)
        for i in range(n):
            xi, yi = x_data[i], y_data[i]
            term = yi
            for j in range(n):
                if i != j:
                    term *= (x - x_data[j]) / (xi - x_data[j])
            total += term
        return total

    def _interpolate_f(target_x, all_x, all_y):
        """
        Chiến lược nội suy theo yêu cầu:
        - Main: Lagrange 6 điểm.
        - Left Boundary: Newton Forward (nếu không đủ điểm bên trái).
        - Right Boundary: Newton Backward (nếu không đủ điểm bên phải).
        """
        # Tìm vị trí của target_x trong all_x
        # idx là chỉ số sao cho all_x[idx] <= target_x < all_x[idx+1]
        idx = np.searchsorted(all_x, target_x)
        
        # Xử lý biên chính xác
        if idx < len(all_x) and np.isclose(all_x[idx], target_x):
            return all_y[idx]
        
        idx = idx - 1 # Đưa về chỉ số bên trái
        idx = max(0, min(idx, len(all_x) - 2))
        
        N = len(all_x)
        n_points = 6
        
        # Xác định chiến lược
        if N < n_points:
            # Không đủ điểm cho chiến lược 6 điểm, dùng Lagrange trên toàn bộ điểm
            return _lagrange_interpolation(target_x, all_x, all_y)
        
        # Chỉ số trung tâm lý tưởng cho Lagrange
        # Muốn target_x nằm giữa, tức là khoảng [idx-2, idx+3]
        start_idx = idx - 2
        end_idx = start_idx + n_points
        
        if start_idx < 0:
            # Vùng biên trái: Dùng Newton Forward trên 6 điểm đầu tiên
            x_local = all_x[0:n_points]
            y_local = all_y[0:n_points]
            coef = _divided_diff(x_local, y_local)
            return _newton_poly(coef, x_local, target_x)
            
        elif end_idx > N:
            # Vùng biên phải: Dùng Newton Backward trên 6 điểm cuối cùng
            # Newton Backward thực chất là Newton trên các nút xếp ngược hoặc xuôi đều ra cùng đa thức
            # Để đúng bản chất "Backward", ta dùng các nút cuối
            x_local = all_x[N-n_points:N]
            y_local = all_y[N-n_points:N]
            coef = _divided_diff(x_local, y_local)
            return _newton_poly(coef, x_local, target_x)
            
        else:
            # Vùng giữa: Dùng Lagrange 6 điểm
            x_local = all_x[start_idx:end_idx]
            y_local = all_y[start_idx:end_idx]
            return _lagrange_interpolation(target_x, x_local, y_local)

    # --- Initialization ---
    process_log = []
    process_log.append(f"Bắt đầu phương pháp Tích phân Hình thang.")
    
    # Chuẩn hóa dữ liệu đầu vào
    x_nodes = np.array(x_nodes, dtype=float)
    y_nodes = np.array(y_nodes, dtype=float)
    
    # Sắp xếp dữ liệu theo x
    sort_idx = np.argsort(x_nodes)
    x_nodes = x_nodes[sort_idx]
    y_nodes = y_nodes[sort_idx]
    
    if g is None:
        g = lambda f_val, x_val: f_val
        process_log.append("Hàm g chưa được cung cấp, mặc định g(f, x) = f(x).")
    else:
        process_log.append("Sử dụng hàm g(f, x) tùy chỉnh.")

    # Xác định n ban đầu
    # Theo yêu cầu: n_initial = (Số điểm x_nodes nằm trong [a, b]) - 1
    points_in_interval = x_nodes[(x_nodes >= a - 1e-12) & (x_nodes <= b + 1e-12)]
    n_count = len(points_in_interval)
    
    if n_count < 2:
        n_curr = 1
        process_log.append(f"Cảnh báo: Không đủ điểm dữ liệu trong [{a}, {b}]. Gán n_initial = 1.")
    else:
        n_curr = n_count - 1
    
    process_log.append(f"Số khoảng chia ban đầu n = {n_curr} (dựa trên {n_count} điểm dữ liệu trong đoạn tích phân).")

    # --- Calculation Loop ---
    
    iteration = 0
    max_iterations = 20 # Giới hạn tránh lặp vô hạn
    final_result = 0.0
    error_est = float('inf')
    
    intermediate = {
        "h_history": [],
        "n_history": [],
        "I_history": [],
        "errors": [],
        "iteration_history": [] # Standardized history list
    }
    
    while True:
        iteration += 1
        h = (b - a) / n_curr
        
        # Tạo lưới tích phân cho bước hiện tại
        # Lưu ý: Lưới này độc lập với x_nodes ban đầu
        integration_grid_x = np.linspace(a, b, n_curr + 1)
        
        # Tính giá trị hàm cần tích phân (integrand) tại các nút lưới
        integrand_values = []
        f_interpolated_values = []
        
        for x_val in integration_grid_x:
            # 1. Nội suy f(x) từ dữ liệu gốc
            f_val = _interpolate_f(x_val, x_nodes, y_nodes)
            f_interpolated_values.append(f_val)
            
            # 2. Tính g(f(x), x)
            g_val = g(f_val, x_val)
            integrand_values.append(g_val)
        
        integrand_values = np.array(integrand_values)
        
        # Tính tổng hình thang: I = (h/2) * (y0 + 2*sum(yi) + yn)
        sum_inner = np.sum(integrand_values[1:-1]) if n_curr > 1 else 0
        I_curr = (h / 2.0) * (integrand_values[0] + 2 * sum_inner + integrand_values[-1])
        
        # Lưu thông tin
        intermediate["h_history"].append(h)
        intermediate["n_history"].append(n_curr)
        intermediate["I_history"].append(I_curr)

        # For first iteration (no error available yet)
        if iteration == 1:
            intermediate["iteration_history"].append({
                "iter": 1,
                "N": n_curr,
                "h": h,
                "result": I_curr,
                "error": None
            })
        
        process_log.append(f"Lần lặp {iteration}: n = {n_curr}, h = {h:.6f}, I = {I_curr:.9f}")
        
        # Kiểm tra điều kiện dừng
        if epsilon is None:
            final_result = I_curr
            
            # Đánh giá sai số lưới thưa
            sparse_k = None
            for k in range(2, int(n_curr / 2) + 2):
                if n_curr % k == 0:
                    sparse_k = k
                    break
            
            if sparse_k:
                n_sparse = n_curr // sparse_k
                process_log.append(f"Chế độ tính 1 lần: Đánh giá sai số với lưới thưa k={sparse_k} (N_sparse={n_sparse}).")
                
                # Tạo lưới thưa
                # integrand_values has n_curr + 1 points
                g_sparse = integrand_values[::sparse_k]
                
                # Trapezoidal formula on sparse grid
                # I = (h_sparse/2) * (y0 + 2*sum(yi) + yn)
                sum_inner_sp = np.sum(g_sparse[1:-1]) if n_sparse > 1 else 0
                h_sparse = h * sparse_k
                
                I_sparse = (h_sparse / 2.0) * (g_sparse[0] + 2 * sum_inner_sp + g_sparse[-1])
                
                # Trapezoidal error O(h^2)
                runge_denom = (sparse_k ** 2) - 1.0
                runge_error = abs(I_curr - I_sparse) / runge_denom
                
                error_est = runge_error
                if intermediate["iteration_history"]:
                     intermediate["iteration_history"][-1]["error"] = error_est

                process_log.append(f"  -> I_dense (N={n_curr}) = {I_curr:.9f}")
                process_log.append(f"  -> I_sparse (N={n_sparse}, k={sparse_k}) = {I_sparse:.9f}")
                process_log.append(f"  -> Sai số ước lượng (|I - I_sp| / {int(runge_denom)}): {error_est:.9e}")
            else:
                error_est = 0.0
                process_log.append("Hoàn thành. Không tìm được lưới thưa phù hợp để đánh giá sai số.")
            break
        
        if iteration > 1:
            I_prev = intermediate["I_history"][-2]
            
            # Đánh giá sai số theo nguyên lý Runge cho hình thang (O(h^2))
            # Runge estimate: |I_2n - I_n| / 3
            # Tuy nhiên yêu cầu Output 4 ghi: |Rn| <= ... và input yêu cầu dùng |I_h - I_{h/2}|
            runge_diff = abs(I_curr - I_prev)
            current_error = runge_diff / 3.0
            intermediate["errors"].append(current_error)
            
            # Add to standardized history
            intermediate["iteration_history"].append({
                "iter": len(intermediate["iteration_history"]) + 1,
                "N": n_curr,
                "h": h,
                "result": I_curr,
                "error": current_error
            })
            
            process_log.append(f"  -> Sai số Runge ước lượng: {current_error:.9e} (Diff: {runge_diff:.9e})")
            
            if current_error < epsilon:
                final_result = I_curr
                error_est = current_error
                process_log.append(f"Đã đạt độ chính xác yêu cầu (< {epsilon}). Kết thúc.")
                break
        
        # Generate detailed table for this iteration (if first iteration)
        if iteration == 1:
            detailed_table = []
            current_h = h
            current_x_grid = integration_grid_x
            current_f_vals = f_interpolated_values
            current_g_vals = integrand_values
            
            common_factor = current_h / 2.0
            
            for i in range(len(current_x_grid)):
                x_val = current_x_grid[i]
                f_val = current_f_vals[i]
                g_val = current_g_vals[i]
                
                # Determine RAW weight (Trapezoidal: 1, 2, ..., 2, 1)
                if i == 0 or i == len(current_x_grid) - 1:
                    raw_weight = 1.0
                else:
                    raw_weight = 2.0
                    
                term = raw_weight * g_val
                
                detailed_table.append({
                    "i": i,
                    "x": x_val,
                    "f": f_val,
                    "g": g_val,
                    "C": raw_weight, # Raw Weight
                    "term": term     # Raw Term
                })
            intermediate["initial_detailed_table"] = detailed_table
            intermediate["initial_common_factor"] = common_factor
        
        if iteration >= max_iterations:
            final_result = I_curr
            error_est = abs(I_curr - intermediate["I_history"][-2]) / 3.0
            process_log.append(f"Đạt giới hạn số lần lặp ({max_iterations}). Kết thúc.")
            break
            
        # Chuẩn bị cho vòng lặp sau
        n_curr *= 2
    
    intermediate["final_grid_x"] = integration_grid_x.tolist()
    intermediate["final_interpolated_f"] = f_interpolated_values
    intermediate["final_integrand_g"] = integrand_values.tolist()
    
    return {
        "result": final_result,
        "h": h, # Return final h step size
        "error_estimate": error_est,
        "error_estimate": error_est,
        "intermediate_values": intermediate,
        "computation_process": process_log
    }