import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Giải hệ phương trình ma trận 3 đường chéo (Tridiagonal Matrix Algorithm - TDMA).
    Hệ: a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
    
    Args:
        a (array): Đường chéo dưới (độ dài N, a[0] không dùng).
        b (array): Đường chéo chính (độ dài N).
        c (array): Đường chéo trên (độ dài N, c[N-1] không dùng).
        d (array): Vế phải (độ dài N).
        
    Returns:
        x (array): Nghiệm của hệ.
    """
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    # Khử xuôi (Forward Elimination)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i-1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
    # Thế ngược (Backward Substitution)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

def solve_bvp_fdm(p_func, q_func, f_func, a, b, N, bc_a, bc_b):
    """
    Giải bài toán biên (BVP) phương trình vi phân cấp 2 bằng phương pháp Sai phân hữu hạn (Finite Difference Method).
    Phương trình: [p(x)u'(x)]' - q(x)u(x) = -f(x)
    Tương đương: p(x)u'' + p'(x)u' - q(x)u = -f(x)
    
    Điều kiện biên tổng quát: alpha * u + beta * u' = gamma
    
    Args:
        p_func (callable): Hàm p(x).
        q_func (callable): Hàm q(x).
        f_func (callable): Hàm f(x) (vế phải).
        a (float): Cận trái.
        b (float): Cận phải.
        N (int): Số khoảng chia.
        bc_a (tuple): (alpha, beta, gamma) tại x = a.
        bc_b (tuple): (alpha, beta, gamma) tại x = b.
        
    Returns:
        dict: Kết quả gồm x_nodes, u_values (nghiệm), và log.
    """
    
    process_log = []
    
    h = (b - a) / N
    x_nodes = np.linspace(a, b, N + 1)
    
    process_log.append(f"Miền tính toán: [{a}, {b}], N={N}, h={h:.6f}")
    process_log.append(f"Điều kiện biên trái (x={a}): {bc_a}")
    process_log.append(f"Điều kiện biên phải (x={b}): {bc_b}")
    
    # Khởi tạo ma trận hệ số (N+1 phương trình cho N+1 ẩn u_0 ... u_N)
    # Tuy nhiên, ta dùng indexing 0..N, nên kích thước là N+1.
    dim = N + 1
    diag_main = np.zeros(dim)   # b_i
    diag_lower = np.zeros(dim)  # a_i
    diag_upper = np.zeros(dim)  # c_i
    rhs = np.zeros(dim)         # d_i
    
    # --- 1. Thiết lập phương trình tại các nút trong (i = 1 ... N-1) ---
    # Phương trình sai phân dạng bảo toàn (Conservative/Self-adjoint Form):
    # (1/h^2) * [ p_{i+1/2}(u_{i+1} - u_i) - p_{i-1/2}(u_i - u_{i-1}) ] - q_i*u_i = -f_i
    
    for i in range(1, N):
        xi = x_nodes[i]
        
        # Các điểm giữa
        x_plus_half = xi + h/2
        x_minus_half = xi - h/2
        
        p_plus = p_func(x_plus_half)
        p_minus = p_func(x_minus_half)
        qi = q_func(xi)
        fi = f_func(xi)
        
        # Hệ số cho u_{i-1}, u_i, u_{i+1}
        # A_i * u_{i-1} + B_i * u_i + C_i * u_{i+1} = D_i
        
        # Từ công thức:
        # term1 = p_plus * (u_{i+1} - u_i) / h^2
        # term2 = p_minus * (u_i - u_{i-1}) / h^2
        # term3 = -q_i * u_i
        # rhs = -f_i
        
        # Nhóm lại: 
        # u_{i-1}: p_minus / h^2
        # u_i:    -(p_plus + p_minus) / h^2 - q_i
        # u_{i+1}: p_plus / h^2
        
        # Nhân tất cả với h^2 để đơn giản hóa:
        # u_{i-1} * p_minus  +  u_i * (-(p_plus + p_minus) - q_i * h^2)  +  u_{i+1} * p_plus  =  -f_i * h^2
        
        diag_lower[i] = p_minus
        diag_main[i] = -(p_plus + p_minus + qi * h**2)
        diag_upper[i] = p_plus
        rhs[i] = -fi * h**2
        
    # --- 2. Xử lý Điều kiện biên ---
    # Tổng quát: alpha * u + beta * u' = gamma
    # Xấp xỉ u':
    # - Tại trái (Forward): u'(a) ≈ (u_1 - u_0) / h  (Sai số O(h)) -> Có thể dùng O(h^2) 3 điểm nếu cần cx cao hơn
    # - Tại phải (Backward): u'(b) ≈ (u_N - u_{N-1}) / h
    
    # Boundary Left (i=0)
    alpha_a, beta_a, gamma_a = bc_a
    if abs(beta_a) < 1e-12: # Dirichlet: u(a) = gamma / alpha
        # Phương trình: 1 * u_0 = gamma / alpha
        diag_main[0] = 1.0
        diag_upper[0] = 0.0
        rhs[0] = gamma_a / alpha_a
    else: # Neumann hoặc Robin
        # alpha * u_0 + beta * (u_1 - u_0)/h = gamma
        # u_0 * (alpha - beta/h) + u_1 * (beta/h) = gamma
        diag_main[0] = alpha_a - beta_a / h
        diag_upper[0] = beta_a / h
        rhs[0] = gamma_a
        
    # Boundary Right (i=N)
    alpha_b, beta_b, gamma_b = bc_b
    if abs(beta_b) < 1e-12: # Dirichlet: u(b) = gamma / alpha
        diag_main[N] = 1.0
        diag_lower[N] = 0.0
        rhs[N] = gamma_b / alpha_b
    else: # Neumann hoặc Robin
        # alpha * u_N + beta * (u_N - u_{N-1})/h = gamma
        # u_{N-1} * (-beta/h) + u_N * (alpha + beta/h) = gamma
        diag_lower[N] = -beta_b / h
        diag_main[N] = alpha_b + beta_b / h
        rhs[N] = gamma_b
        
    # --- 3. Giải hệ phương trình ---
    try:
        u_solution = thomas_algorithm(diag_lower, diag_main, diag_upper, rhs)
        process_log.append("Giải hệ phương trình thành công (Thomas Algorithm).")
    except Exception as e:
        process_log.append(f"Lỗi khi giải hệ phương trình: {e}")
        u_solution = np.zeros(dim)
        
    return {
        "x_nodes": x_nodes,
        "u_values": u_solution,
        "log": process_log,
        "h": h
    }
