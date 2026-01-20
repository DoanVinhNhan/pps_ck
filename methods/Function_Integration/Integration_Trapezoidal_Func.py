import numpy as np

def trapezoidal_integration_func(f, a, b, epsilon=None, N_start=1):
    """
    Thực hiện tính tích phân Hình Thang (Trapezoidal Rule) cho hàm số f(x).

    Phương pháp:
        I ≈ (h/2) * (f(a) + 2*sum(f(x_i)) + f(b))

    Args:
        f (callable): Hàm cần tính tích phân f(x).
        a (float): Cận dưới.
        b (float): Cận trên.
        epsilon (float, optional): Sai số cho phép.
        N_start (int): Số khoảng chia ban đầu.

    Returns:
        dict: Kết quả tính toán.
    """
    
    computation_log = []
    computation_log.append(f"Bắt đầu tính tích phân Hình Thang (Function Mode) trên đoạn [{a}, {b}].")

    N = max(1, N_start)
    computation_log.append(f"Sử dụng N={N} ban đầu.")

    current_result = 0.0
    previous_result = None
    error_est = float('inf')
    
    intermediate_vars = {
        "iteration_history": []
    }

    iteration = 0
    max_iterations = 20

    while True:
        iteration += 1
        h = (b - a) / N
        
        x_grid = np.linspace(a, b, N + 1)
        y_grid = f(x_grid)
        
        y_start = y_grid[0]
        y_end = y_grid[-1]
        
        # Tổng các điểm giữa
        sigma = np.sum(y_grid[1:-1])
        
        # Công thức Hình Thang
        current_result = (h / 2) * (y_start + 2 * sigma + y_end)
        
        log_entry = (f"Lần lặp {iteration}: N={N}, h={h:.6f}, I={current_result:.8f}")
        computation_log.append(log_entry)
        
        history_entry = {
            "iter": iteration,
            "N": N,
            "h": h,
            "result": current_result,
            "error": None
        }
        
        # Detailed table (First Iteration)
        if iteration == 1:
            detailed_table = []
            common_factor = h / 2.0
            num_points = len(x_grid)
            
            for i in range(num_points):
                coeff = 2
                if i == 0 or i == num_points - 1:
                    coeff = 1
                
                g_val = y_grid[i]
                term = coeff * g_val
                
                detailed_table.append({
                    "i": i,
                    "x": x_grid[i],
                    "f": g_val,
                    "g": g_val,
                    "C": coeff,
                    "term": term
                })
            
            intermediate_vars["initial_detailed_table"] = detailed_table
            intermediate_vars["initial_common_factor"] = common_factor

        intermediate_vars["iteration_history"].append(history_entry)

        if epsilon is None:
            computation_log.append("Mode One-shot. Dừng.")
            break
        else:
            if previous_result is not None:
                # Runge error for Trapezoidal (O(h^2)): |I_h - I_{2h}| / 3
                runge_error = abs(current_result - previous_result) / 3.0
                error_est = runge_error
                history_entry["error"] = error_est
                
                computation_log.append(f"  -> Sai số Runge ước lượng: {runge_error:.8e}")
                
                if runge_error < epsilon:
                    computation_log.append(f"Đã đạt độ chính xác yêu cầu (< {epsilon}). Dừng.")
                    break
            
            if iteration >= max_iterations:
                computation_log.append("Đã đạt số lần lặp tối đa. Dừng.")
                break
            
            previous_result = current_result
            N *= 2

    return {
        "result": current_result,
        "h": h,
        "error_estimate": error_est if epsilon is not None else 0.0,
        "intermediate_values": intermediate_vars,
        "computation_process": computation_log
    }
