import math
import numpy as np

def newton_forward_interpolation(x_query, x_data, y_data):
    """
    Thực hiện nội suy Newton tiến (Newton Forward) cho dữ liệu cách đều (hoặc xấp xỉ).
    Sử dụng bảng sai phân hữu hạn.
    """
    n = len(x_data)
    # Tính bảng sai phân
    diff_table = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y_data[i]
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
    
    # Tính u = (x - x0) / h
    h = x_data[1] - x_data[0]
    u = (x_query - x_data[0]) / h
    
    y_val = diff_table[0][0]
    u_term = 1
    factorial = 1
    
    for i in range(1, n):
        u_term *= (u - (i - 1))
        factorial *= i
        y_val += (u_term * diff_table[0][i]) / factorial
        
    return y_val

def newton_backward_interpolation(x_query, x_data, y_data):
    """
    Thực hiện nội suy Newton lùi (Newton Backward).
    """
    n = len(x_data)
    # Tính bảng sai phân
    diff_table = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y_data[i]
        
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff_table[i][j] = diff_table[i][j-1] - diff_table[i-1][j-1]
            
    # Tính u = (x - xn) / h
    h = x_data[1] - x_data[0]
    u = (x_query - x_data[n-1]) / h
    
    y_val = diff_table[n-1][0]
    u_term = 1
    factorial = 1
    
    for i in range(1, n):
        u_term *= (u + (i - 1))
        factorial *= i
        y_val += (u_term * diff_table[n-1][i]) / factorial
        
    return y_val

def lagrange_interpolation(x_query, x_data, y_data):
    """
    Thực hiện nội suy Lagrange 6 điểm.
    """
    n = len(x_data)
    total = 0.0
    for i in range(n):
        xi = x_data[i]
        yi = y_data[i]
        
        term = yi
        for j in range(n):
            if i != j:
                xj = x_data[j]
                term *= (x_query - xj) / (xi - xj)
        total += term
    return total

def get_interpolated_value(x, x_nodes, y_nodes, process_log):
    """
    Hàm điều phối chiến lược nội suy theo yêu cầu:
    - Main: Lagrange 6 điểm.
    - Biên trái: Newton Forward.
    - Biên phải: Newton Backward.
    """
    # Tìm vị trí của x trong x_nodes để chọn 6 điểm lân cận
    # Giả sử x_nodes đã sắp xếp tăng dần
    
    n_points = 6
    total_nodes = len(x_nodes)
    
    # Tìm index i sao cho x_nodes[i] <= x < x_nodes[i+1]
    idx = -1
    for i in range(total_nodes - 1):
        if x_nodes[i] <= x <= x_nodes[i+1]:
            idx = i
            break
            
    # Xử lý trường hợp x nằm ngoài phạm vi (extrapolation nhẹ hoặc đúng biên)
    if idx == -1:
        if x < x_nodes[0]: idx = 0
        else: idx = total_nodes - 2

    # Xác định cửa sổ 6 điểm
    # Cố gắng lấy trung tâm quanh idx
    start_idx = idx - (n_points // 2) + 1
    end_idx = start_idx + n_points
    
    method_used = ""
    local_x = []
    local_y = []
    
    if start_idx < 0:
        # Không đủ điểm bên trái -> Biên trái -> Newton Forward
        start_idx = 0
        end_idx = n_points
        method_used = "Newton Forward (Left Boundary)"
        if end_idx > total_nodes: end_idx = total_nodes # Fallback nếu dữ liệu quá ít
        
        local_x = x_nodes[start_idx:end_idx]
        local_y = y_nodes[start_idx:end_idx]
        val = newton_forward_interpolation(x, local_x, local_y)
        
    elif end_idx > total_nodes:
        # Không đủ điểm bên phải -> Biên phải -> Newton Backward
        end_idx = total_nodes
        start_idx = total_nodes - n_points
        if start_idx < 0: start_idx = 0
        method_used = "Newton Backward (Right Boundary)"
        
        local_x = x_nodes[start_idx:end_idx]
        local_y = y_nodes[start_idx:end_idx]
        val = newton_backward_interpolation(x, local_x, local_y)
        
    else:
        # Đủ điểm hai bên -> Vùng giữa -> Lagrange
        method_used = "Lagrange 6-point"
        local_x = x_nodes[start_idx:end_idx]
        local_y = y_nodes[start_idx:end_idx]
        val = lagrange_interpolation(x, local_x, local_y)

    return val

def newton_cotes_integration(x_nodes, y_nodes, g=None, a=None, b=None, epsilon=None, degree=4):
    """
    Thực hiện tính tích phân Newton-Cotes (độ chính xác bậc n=4, 5, 6) cho hàm g(f(x), x).
    
    Args:
        x_nodes (list): Danh sách các giá trị x rời rạc.
        y_nodes (list): Danh sách các giá trị f(x) tương ứng.
        g (callable, optional): Hàm g(f, x). Mặc định là f.
        a (float): Cận dưới.
        b (float): Cận trên.
        epsilon (float, optional): Sai số cho phép để lặp.
        degree (int): Bậc của công thức Newton-Cotes (4: Boole, 5, 6).
        
    Returns:
        dict: Kết quả, sai số ước lượng, giá trị trung gian, nhật ký tính toán.
    """
    # 1. Khởi tạo và Kiểm tra đầu vào
    process_log = []
    process_log.append(f"Bắt đầu phương pháp Newton-Cotes (Degree n={degree}).")
    
    if degree not in [4, 5, 6]:
        raise ValueError("Degree phải là 4, 5 hoặc 6.")
        
    if g is None:
        g = lambda f, x: f
        process_log.append("Hàm g mặc định: g(f, x) = f.")
    
    # Lọc và sắp xếp dữ liệu đầu vào (để đảm bảo tính nhất quán cho nội suy)
    data = sorted(zip(x_nodes, y_nodes))
    x_nodes_sorted = [k[0] for k in data]
    y_nodes_sorted = [k[1] for k in data]
    
    # Xác định số khoảng chia ban đầu
    # Đếm số điểm x_nodes nằm trong [a, b]
    points_in_range = [x for x in x_nodes_sorted if a <= x <= b]
    count_points = len(points_in_range)
    
    if count_points < 2:
        # Nếu ít hơn 2 điểm, mặc định lấy một số lượng tối thiểu để chạy được nội suy
        n_initial = degree * 2 
        process_log.append(f"Cảnh báo: Quá ít điểm dữ liệu trong [{a}, {b}]. Gán n_initial = {n_initial}.")
    else:
        n_initial = count_points - 1
        process_log.append(f"Số điểm dữ liệu gốc trong khoảng tích phân: {count_points}. n_derived = {n_initial}.")

    # Ràng buộc Newton-Cotes Composite: Tổng số khoảng chia N phải chia hết cho degree (n)
    if n_initial % degree != 0:
        adjustment = degree - (n_initial % degree)
        n_old = n_initial
        n_initial += adjustment
        process_log.append(f"  -> N ban đầu = {n_old} (không chia hết cho {degree}).")
        process_log.append(f"  -> Điều chỉnh N = {n_initial} để thỏa mãn điều kiện Newton-Cotes (chia hết cho {degree}).")
        process_log.append(f"  -> Bước h thay đổi từ {(b-a)/n_old:.6f} thành {(b-a)/n_initial:.6f}.")
        process_log.append(f"  -> Cảnh báo: Thực hiện nội suy để lấy các giá trị mới không có trong bảng dữ liệu!")
    else:
        process_log.append(f"N ban đầu = {n_initial} đã thỏa mãn chia hết cho {degree}.")
    
    if n_initial == 0: n_initial = degree # Fallback an toàn

    # Cấu hình trọng số Newton-Cotes (Cotes numbers)
    # Công thức: I = C * h * sum(w_i * y_i)
    if degree == 4:
        # Boole's Rule: 2h/45 * (7, 32, 12, 32, 7)
        weights_pattern = [7, 32, 12, 32, 7]
        multiplier_factor = 2.0 / 45.0
    elif degree == 5:
        # 5h/288 * (19, 75, 50, 50, 75, 19)
        weights_pattern = [19, 75, 50, 50, 75, 19]
        multiplier_factor = 5.0 / 288.0
    elif degree == 6:
        # h/140 * (41, 216, 27, 272, 27, 216, 41)
        weights_pattern = [41, 216, 27, 272, 27, 216, 41]
        multiplier_factor = 1.0 / 140.0
        
    process_log.append(f"Sử dụng mẫu trọng số bậc {degree}: {weights_pattern} với hệ số nhân {multiplier_factor}*h.")

    current_N = n_initial
    prev_result = None
    error_est = float('inf')
    
    intermediate = {
        "iterations": [],
        "iteration_history": [] # Standardized
    }

    # Vòng lặp tính toán (Refinement)
    max_iterations = 20 if epsilon else 1
    iteration_count = 0
    
    final_result = 0.0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        h = (b - a) / current_N
        
        process_log.append(f"--- Lần lặp {iteration_count}: Số khoảng N={current_N}, Bước nhảy h={h:.6f} ---")
        
        # Tạo lưới điểm tích phân
        integration_x = [a + i * h for i in range(current_N + 1)]
        integration_x[-1] = b # Đảm bảo chính xác cận cuối
        
        # Tính giá trị hàm tại các điểm tích phân (sử dụng nội suy)
        integration_y = []
        integrand_values = []
        
        for x_val in integration_x:
            # 1. Nội suy f(x)
            f_val = get_interpolated_value(x_val, x_nodes_sorted, y_nodes_sorted, process_log)
            integration_y.append(f_val)
            # 2. Tính g(f(x), x)
            g_val = g(f_val, x_val)
            integrand_values.append(g_val)
            
        # Tính tổng Newton-Cotes Composite
        # Chia N khoảng thành (N/degree) khối, mỗi khối có (degree) khoảng nhỏ -> (degree + 1) điểm
        num_blocks = current_N // degree
        total_sum = 0.0
        
        # Calculate Global Effective Weights for Detailed Table
        # Initialize global weights array
        num_grid_points = len(integration_x)
        global_weights = np.zeros(num_grid_points)
        
        idx = 0
        for block in range(num_blocks):
            # Khối bắt đầu từ index: block * degree
            block_sum = 0.0
            for w_idx, weight in enumerate(weights_pattern):
                global_idx = (block * degree) + w_idx
                y_curr = integrand_values[global_idx]
                
                # Accumulate coefficient contribution
                # Total Integral = multiplier * h * sum(coeff * y)
                # So effective C_i = multiplier * h * coeff
                # Store just the coeff part first, or full C? Let's store full C.
                effective_weight_contribution = multiplier_factor * h * weight
                global_weights[global_idx] += effective_weight_contribution
                
                block_sum += weight * y_curr
            
            total_sum += block_sum
            
        current_result = multiplier_factor * h * total_sum
        
        if iteration_count == 1:
             # Generate Detailed Table with RAW weights
            num_grid_points = len(integration_x)
            global_raw_weights = np.zeros(num_grid_points)
            
            common_factor = multiplier_factor * h
            
            # Logic similar to calculation loop but summing Raw Weights
            for block in range(num_blocks):
                for w_idx, weight in enumerate(weights_pattern):
                    global_idx = (block * degree) + w_idx
                    global_raw_weights[global_idx] += weight # Sum raw integer weights
            
            detailed_table = []
            for i in range(num_grid_points):
                 x_val = integration_x[i]
                 g_val = integrand_values[i]
                 f_val = integration_y[i]
                 
                 raw_weight = global_raw_weights[i]
                 term = raw_weight * g_val
                 
                 detailed_table.append({
                    "i": i,
                    "x": x_val,
                    "f": f_val,
                    "g": g_val,
                    "C": raw_weight,
                    "term": term
                })
            
            intermediate["initial_detailed_table"] = detailed_table
            intermediate["initial_common_factor"] = common_factor
        
        # Lưu thông tin trung gian
        iter_info = {
            "N": current_N,
            "h": h,
            "integral_value": current_result,
            "sample_points_count": len(integration_x)
        }
        intermediate["iterations"].append(iter_info)
        
        # Standard history entry
        history_entry = {
            "iter": iteration_count,
            "N": current_N,
            "h": h,
            "result": current_result,
            "error": None
        }
        
        process_log.append(f"Kết quả tích phân tạm thời: {current_result:.9f}")

        # Đánh giá sai số
        should_break = False
        if prev_result is not None:
            # Nguyên lý Runge: Error ~ |I_h - I_2h| / (2^n - 1) ? 
            # Với Newton-Cotes, thường dùng đơn giản |I_moi - I_cu| để ước lượng hội tụ
            # Hoặc Runge chính xác hơn phụ thuộc bậc sai số O(h^k).
            # Ở đây dùng hiệu số tuyệt đối đơn giản cho tiêu chuẩn dừng epsilon.
            error_est = abs(current_result - prev_result)
            history_entry["error"] = error_est
            process_log.append(f"Sai số ước lượng (Runge/Diff): {error_est:.9e}")
            
            if epsilon and error_est < epsilon:
                process_log.append(f"Đã đạt độ chính xác epsilon ({epsilon}). Dừng.")
                final_result = current_result
                should_break = True
        
        intermediate["iteration_history"].append(history_entry)
        
        if should_break:
            break
        
        prev_result = current_result
        final_result = current_result
        
        if epsilon:
            current_N *= 2 # Gấp đôi số khoảng chia
        else:
            # Single pass mode - Add sparse grid error estimation
            sparse_k = None
            # Scan for k
            for k in range(2, int(current_N / degree) + 2):
                if current_N % k == 0:
                    n_sparse = current_N // k
                    if n_sparse % degree == 0:
                        sparse_k = k
                        break
            
            if sparse_k:
                n_sparse = current_N // sparse_k
                process_log.append(f"Chế độ tính 1 lần: Đánh giá sai số với lưới thưa k={sparse_k} (N_sparse={n_sparse}).")
                
                # Slicing
                integrand_sparse = integrand_values[::sparse_k]
                h_sparse = h * sparse_k
                
                # Calculate I_sparse
                num_blocks_sp = n_sparse // degree
                total_sum_sp = 0.0
                
                for block in range(num_blocks_sp):
                    block_sum = 0.0
                    for w_idx, weight in enumerate(weights_pattern):
                        global_idx = (block * degree) + w_idx
                        y_curr = integrand_sparse[global_idx]
                        block_sum += weight * y_curr
                    total_sum_sp += block_sum
                    
                I_sparse = multiplier_factor * h_sparse * total_sum_sp
                
                # Determine Runge power
                if degree == 4: p_nc = 6
                elif degree == 5: p_nc = 6 # Often cited as 6 for N=5? Actually degree 5 is usually same order as degree 4? No, Newton-Cotes n=5 is less common. Let's assume 6.
                elif degree == 6: p_nc = 8
                else: p_nc = degree # Fallback
                
                runge_denom = (sparse_k ** p_nc) - 1.0
                error_est = abs(current_result - I_sparse) / runge_denom
                
                history_entry["error"] = error_est
                
                process_log.append(f"  -> I_dense (N={current_N}) = {current_result:.9f}")
                process_log.append(f"  -> I_sparse (N={n_sparse}, k={sparse_k}) = {I_sparse:.9f}")
                process_log.append(f"  -> Sai số ước lượng (|I - I_sp| / {int(runge_denom)}, p={p_nc}): {error_est:.9e}")

            else:
                 process_log.append(f"Không tìm được lưới thưa phù hợp (chia hết cho degree={degree}) để đánh giá sai số.")
                 
            break
            
    return {
        "result": final_result,
        "h": h, # Return final h step size
        "error_estimate": error_est,
        "intermediate_values": intermediate,
        "computation_process": process_log
    }