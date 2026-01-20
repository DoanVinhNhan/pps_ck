import numpy as np

def simpson_integration_func(f, a, b, epsilon=None, N_start=2):
    """
    Thực hiện tính tích phân Simpson (Simpson's Rule) cho hàm số f(x).

    Phương pháp:
        I ≈ (h/3) * (f(a) + 4*sigma_1 + 2*sigma_2 + f(b))

    Args:
        f (callable): Hàm cần tính tích phân f(x).
        a (float): Cận dưới tích phân.
        b (float): Cận trên tích phân.
        epsilon (float, optional): Sai số cho phép. Nếu được cung cấp, thuật toán sẽ 
                                   lặp và chia đôi bước lưới h cho đến khi thỏa mãn sai số.
        N_start (int): Số khoảng chia ban đầu.

    Returns:
        dict: Kết quả bao gồm giá trị tích phân, sai số ước lượng, các giá trị trung gian
              và nhật ký quá trình tính toán.
    """
    
    computation_log = []
    
    computation_log.append(f"Bắt đầu tính tích phân Simpson (Function Mode) trên đoạn [{a}, {b}].")

    # 1. Xác định số khoảng chia N ban đầu
    N = N_start
    if N % 2 != 0:
        N += 1
        computation_log.append(f"N_start={N_start} là số lẻ. Điều chỉnh N={N} (chặn) để thỏa mãn Simpson.")
    else:
        computation_log.append(f"Sử dụng N={N} ban đầu.")

    current_result = 0.0
    previous_result = None
    error_est = float('inf')
    
    intermediate_vars = {
        "iteration_history": []
    }

    # 2. Vòng lặp tính toán (Iterative process)
    iteration = 0
    max_iterations = 20 # Tránh lặp vô hạn

    while True:
        iteration += 1
        h = (b - a) / N
        
        # Tạo lưới điểm tích phân
        x_grid = np.linspace(a, b, N + 1)
        
        # Tính giá trị hàm số tại các nút lưới
        y_grid = f(x_grid)
        
        # Tính các tổng sigma
        # y_0 và y_{2n} (y_N)
        y_start = y_grid[0]
        y_end = y_grid[-1]
        
        # Sigma 1: chỉ số lẻ (1, 3, ..., N-1)
        sigma_1 = np.sum(y_grid[1:N:2])
        
        # Sigma 2: chỉ số chẵn (2, 4, ..., N-2)
        sigma_2 = np.sum(y_grid[2:N:2])
        
        # Công thức Simpson: I ≈ (h / 3) * (y_0 + 4 * sigma_1 + 2 * sigma_2 + y_{2n})
        current_result = (h / 3) * (y_start + 4 * sigma_1 + 2 * sigma_2 + y_end)
        
        log_entry = (f"Lần lặp {iteration}: N={N}, h={h:.6f}, "
                     f"Sigma1={sigma_1:.5f}, Sigma2={sigma_2:.5f}, I={current_result:.8f}")
        computation_log.append(log_entry)
        
        history_entry = {
            "iter": iteration,
            "N": N,
            "h": h,
            "result": current_result,
            "error": None
        }

        # Lưu giá trị trung gian của lần lặp cuối cùng
        intermediate_vars.update({
            "N": N,
            "h": h,
            "y_start": y_start,
            "y_end": y_end,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
        })
        
        # Detailed table logic (First Iteration)
        if iteration == 1:
            detailed_table = []
            current_h = h
            num_points = len(x_grid)
            common_factor = current_h / 3.0
            
            for i in range(num_points):
                x_val = x_grid[i]
                g_val = y_grid[i] # g(x) = f(x)
                
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
                    "f": g_val, # Assume g=f
                    "g": g_val,
                    "C": base_coeff,
                    "term": term
                })
                
            intermediate_vars["initial_detailed_table"] = detailed_table
            intermediate_vars["initial_common_factor"] = common_factor

        # Add to history list
        intermediate_vars["iteration_history"].append(history_entry)

        # Kiểm tra điều kiện dừng
        if epsilon is None:
            # Nếu không có epsilon, chạy 1 lần rồi dừng (Mode One-shot)
            computation_log.append("Mode One-shot (không lặp). Dừng.")
            break
        else:
            if previous_result is not None:
                # Đánh giá sai số Runge (Simpson O(h^4)): |I_h - I_{2h}| / 15
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
            
            # Chuẩn bị cho vòng lặp sau
            previous_result = current_result
            N *= 2

    return {
        "result": current_result,
        "h": h,
        "error_estimate": error_est if epsilon is not None else 0.0,
        "intermediate_values": intermediate_vars,
        "computation_process": computation_log
    }
