import numpy as np

def newton_cotes_integration_func(f, a, b, degree=6, epsilon=None, N_start=None):
    """
    Thực hiện tính tích phân Newton-Cotes (composite) bậc 'degree' cho hàm số f(x).
    
    Args:
        f (callable): Hàm f(x).
        degree (int): Bậc của đa thức nội suy Newton-Cotes (số khoảng chia nhỏ trong mỗi đoạn lớn).
                      Hỗ trợ degree = 1 (Trapz), 2 (Simpson), 3 (3/8), 4 (Boole), 5, 6.
        N_start (int): Số lượng segment ghép ban đầu. (Tổng số khoảng chia = N_segments * degree).
    """

    computation_log = []
    
    # Định nghĩa trọng số Cotes (C_i) và số chia (Divisor D) cho các bậc
    # CT: I ≈ (N*h / D) * sum(w_i * y_i) ?? Không, thường là h * const * sum...
    # Chuẩn: I ≈ factor * h * sum(w_i * y_i)
    
    cotes_weights = {
        1: [1, 1], # Trapezoidal, factor = 1/2
        2: [1, 4, 1], # Simpson 1/3, factor = 1/3
        3: [1, 3, 3, 1], # Simpson 3/8, factor = 3/8
        4: [7, 32, 12, 32, 7], # Boole, factor = 2/45 * 2? No calc below.
        5: [19, 75, 50, 50, 75, 19], 
        6: [41, 216, 27, 272, 27, 216, 41]
    }
    
    # Factor K where Integral = K * h * Sum(weights * y)
    # Note: Traditional forms:
    # n=1: h/2 * ... -> K=1/2
    # n=2: h/3 * ... -> K=1/3
    # n=3: 3h/8 * ... -> K=3/8
    # n=4: 2h/45 * ... (Wait, standard is (b-a) * ... let's use standard h coefficients)
    # Let's standardize: I_local = h * (Const) * Sum(W_i * y_i)
    
    cotes_factors = {
        1: 1/2.0,
        2: 1/3.0,
        3: 3/8.0,
        4: 2/45.0, 
        5: 5/288.0,
        6: 1/140.0
    }
    
    if degree not in cotes_weights:
        raise ValueError(f"Degree {degree} not supported.")
        
    weights_pattern = np.array(cotes_weights[degree])
    factor = cotes_factors[degree]
    
    # Runge Error Power P (Error ~ O(h^P))
    # n even -> P = n + 1 ? (Simpson n=2 -> P=4 not 3. Wait. Order of Accuracy.)
    # n=1 (Trapz) -> O(h^2). P=2.
    # n=2 (Simp) -> O(h^4). P=4.
    # n=3 (3/8) -> O(h^4). P=4.
    # n=4 (Boole) -> O(h^6). P=6.
    # n=5 -> O(h^6). P=6.
    # n=6 -> O(h^8). P=8.
    
    if degree == 1: p_runge = 2
    elif degree == 2: p_runge = 4
    elif degree == 3: p_runge = 4
    elif degree == 4: p_runge = 6
    elif degree == 5: p_runge = 6
    elif degree == 6: p_runge = 8
    else: p_runge = degree + 1 # Fallback
    
    runge_denom = (2.0 ** p_runge) - 1.0

    # N_segments: number of composite intervals. Each interval has 'degree' sub-intervals.
    # Total points = N_segments * degree + 1
    M_segments = max(1, N_start) if N_start else 1
    
    computation_log.append(f"Bắt đầu Newton-Cotes Func (Degree={degree}). M_segments_start={M_segments}.")
    
    current_result = 0.0
    previous_result = None
    error_est = float('inf')
    
    intermediate_vars = {
        "iteration_history": []
    }
    
    iteration = 0
    max_iterations = 15
    
    while True:
        iteration += 1
        
        # Total sub-intervals N_total
        N_total = M_segments * degree
        h = (b - a) / N_total
        
        x_grid = np.linspace(a, b, N_total + 1)
        y_grid = f(x_grid)
        
        # Composite Rule calculation
        # We can iterate over segments
        total_integral = 0.0
        
        # Vectorized approach for efficiency?
        # Construct full weights vector
        # Pattern repeats. Overlap at boundaries?
        # Composite definition: Sum of integrals over each M segment.
        # For seg k (0..M-1): points from k*degree to (k+1)*degree
        
        # Calculate Integral
        # Iterate segments
        s_sum = 0.0
        # Detailed weights for table
        full_weights = np.zeros(N_total + 1)
        
        for k in range(M_segments):
            start_idx = k * degree
            end_idx = (k + 1) * degree
            # Indices: start_idx, start_idx+1, ..., end_idx
            segment_y = y_grid[start_idx : end_idx + 1]
            
            # Add to full weights (accumulate at boundaries)
            full_weights[start_idx : end_idx + 1] += weights_pattern
            
            # Correction: The loop adds the same boundary point twice?
            # Standard composite:
            # I = Sum( I_local )
            # I_local = h * factor * Sum(w_i * y_loc_i)
            # So yes, we just sum them up.
            
            dot_prod = np.dot(weights_pattern, segment_y)
            s_sum += dot_prod
            
        current_result = factor * h * s_sum
        
        # Adjust full_weights to be the effective coefficient C for I = CommonFactor * Sum(C*y)
        # Here CommonFactor = factor * h
        # Note: Boundary points between segments are added twice in s_sum logic implicitly? 
        # Yes, dot_prod sums them.
        # But wait, did I double count?
        # k=0: uses y[0]...y[d]
        # k=1: uses y[d]...y[2d]
        # So y[d] is multiplied by w[d] in seg0 and w[0] in seg1.
        # This is correct for composite rules.
        
        log_entry = f"Iter {iteration}: M_seg={M_segments}, N_total={N_total}, h={h:.6f}, I={current_result:.8f}"
        computation_log.append(log_entry)
        
        history_entry = {
            "iter": iteration,
            "N": N_total, # Total intervals
            "h": h,
            "result": current_result,
            "error": None
        }

        # Detailed Table (Iter 1)
        if iteration == 1:
            detailed = []
            common_val = factor * h
            for i in range(len(x_grid)):
                # We calculated full_weights manually above? 
                # Re-calculate correct weight for this index
                # Actually, let's look at full_weights array constructed in loop?
                # No, the loop didn't construct an array, just summed.
                # Let's reconstruct global weights.
                
                # Global weight W_i
                w_i = 0
                # Point i belongs to which segment(s)?
                # internal points of segment k: i in (k*d, (k+1)*d)
                # boundaries: i = k*d
                
                # Simple logic: Initialize zeros
                # For each segment, add local weights
                pass 
            
            # Let's do it properly
            W_global = np.zeros(N_total + 1)
            for k in range(M_segments):
                s = k * degree
                W_global[s : s + degree + 1] += weights_pattern
            
            # But wait, do we subtract overlap? No, we SUM overlap.
            # Example Trapezoidal (d=1, w=[1,1]). 
            # Seg0: y0, y1 -> adds 1 to W[0], 1 to W[1]
            # Seg1: y1, y2 -> adds 1 to W[1], 1 to W[2]
            # Result W: [1, 2, 1]. Correct.
            
            for i in range(len(x_grid)):
                term = W_global[i] * y_grid[i]
                detailed.append({
                    "i": i,
                    "x": x_grid[i],
                    "f": y_grid[i],
                    "g": y_grid[i],
                    "C": W_global[i],
                    "term": term
                })
            
            intermediate_vars["initial_detailed_table"] = detailed
            intermediate_vars["initial_common_factor"] = common_val

        intermediate_vars["iteration_history"].append(history_entry)

        if epsilon is None:
            break
        else:
            if previous_result is not None:
                runge = abs(current_result - previous_result) / runge_denom
                error_est = runge
                history_entry["error"] = runge
                computation_log.append(f"  -> Sai số Runge: {runge:.8e}")
                
                if runge < epsilon:
                    computation_log.append("Converged.")
                    break
            
            if iteration >= max_iterations:
                break
                
            previous_result = current_result
            M_segments *= 2 # Double number of segments
    
    return {
        "result": current_result,
        "h": h,
        "error_estimate": error_est if epsilon is not None else 0.0,
        "intermediate_values": intermediate_vars,
        "computation_process": computation_log
    }
