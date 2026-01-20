
import numpy as np

def solve_general_least_squares(x, y, basis_funcs, basis_names):
    # First Pass: Solve with full basis to identify significant terms
    basis_funcs = np.array(basis_funcs) # Allow indexing
    basis_names = np.array(basis_names)
    
    n_funcs = len(basis_funcs)
    col_vectors = []
    
    # Build design matrix A (Full)
    for func in basis_funcs:
        col = func(x)
        if np.isscalar(col):
            col = np.full(n, col)
        col_vectors.append(col)
        
    A = np.vstack(col_vectors).T
    
    # Initial Solve
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    
    # Identify significant terms
    # Threshold: absolute value > 1e-5 AND relative to max coeff > 0.001 (0.1%)
    max_c = np.max(np.abs(coeffs)) if len(coeffs) > 0 else 1.0
    threshold = 1e-5
    significant_idx = []
    
    for i, c in enumerate(coeffs):
        if abs(c) > threshold and (abs(c) / max_c) > 0.001:
            significant_idx.append(i)
            
    if not significant_idx:
        # Fallback: keep largest term if everything is small? 
        # Or just intercept if available. 
        significant_idx = [np.argmax(np.abs(coeffs))]
        
    # Second Pass: Re-fit with only significant terms
    final_basis_funcs = basis_funcs[significant_idx]
    final_basis_names = basis_names[significant_idx]
    
    # Build reduced design matrix A_reduced
    col_vectors_reduced = []
    for func in final_basis_funcs:
        col = func(x)
        if np.isscalar(col):
            col = np.full(n, col)
        col_vectors_reduced.append(col)
        
    A_reduced = np.vstack(col_vectors_reduced).T
    
    # Final Solve
    final_coeffs, _, _, _ = np.linalg.lstsq(A_reduced, y, rcond=None)
    
    # Calculate Matrices for display
    AtA = A_reduced.T @ A_reduced
    AtY = A_reduced.T @ y
    
    # Calculate fitted y
    fitted_y = np.dot(A_reduced, final_coeffs)
    
    # Build clean equation string
    terms = []
    for i, coeff in enumerate(final_coeffs):
        name = final_basis_names[i]
        val_str = f"{coeff:.4f}"
        if name == "1":
            terms.append(val_str)
        else:
            if "e^" in name or "/" in name:
                 terms.append(f"{val_str}{name}") # 0.1234e^(-x)
            else:
                 terms.append(f"{val_str}{name}") # 0.1234x
                
    if not terms:
        equation_str = "y = 0"
    else:
        equation_str = "y = " + " + ".join(terms).replace("+ -", "- ")
        
    # Create predict function closure using FINAL coeffs and FINAL basis
    def predict(x_new):
        x_new = np.array(x_new)
        y_pred = np.zeros_like(x_new, dtype=float)
        for i, func in enumerate(final_basis_funcs):
            val = func(x_new)
            if np.isscalar(val):
                y_pred += final_coeffs[i] * val
            else:
                y_pred += final_coeffs[i] * val
        return y_pred

    # Return extra matrices
    matrices = {
        'A': A_reduced,
        'AtA': AtA,
        'AtY': AtY,
        'basis_names': final_basis_names
    }

    return final_coeffs, fitted_y, equation_str, predict, matrices

def least_squares_regression(x, y, case):
    """
    Perform Least Squares Regression.
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    result = {}
    
    # Definitions of standard basis functions
    basis_1 = lambda x: np.ones_like(x)
    basis_x = lambda x: x
    basis_x2 = lambda x: x**2
    basis_cos = lambda x: np.cos(x)
    basis_sin = lambda x: np.sin(x)
    
    if case == 'general':
        # Expanded Basis for better accuracy
        # Polynomials: 1, x, x^2, x^3, x^4, x^5
        # Trigs: cos, sin
        # Others: 1/x, 1/x^2, sqrt(x), ln(x), exp(-x)
        # Note: 1/x and ln(x) require x > 0. The CSV data starts at 1.0, so it's safe.
        
        basis_funcs = [
            basis_1, basis_x, basis_x2, 
            lambda x: x**3, lambda x: x**4, lambda x: x**5,
            basis_cos, basis_sin,
            lambda x: 1/(x+1e-9), lambda x: 1/(x**2+1e-9), # Add epsilon just in case
            lambda x: np.log(np.abs(x) + 1e-9),
            lambda x: np.exp(-x)
        ]
        basis_names = [
            "1", "x", "x^2", 
            "x^3", "x^4", "x^5",
            "cos(x)", "sin(x)",
            "1/x", "1/x^2",
            "ln(x)", "e^(-x)"
        ]
        
        coeffs, fitted_y, equation_str, predict, matrices = solve_general_least_squares(x, y, basis_funcs, basis_names)
        result['coeffs'] = coeffs
        result['predict'] = predict
        result['matrices'] = matrices

    elif case == 'linear':
        basis_funcs = [basis_x, basis_1]
        basis_names = ["x", ""] 
        coeffs, fitted_y, equation_str, predict = solve_general_least_squares(x, y, basis_funcs, basis_names)
        result['coeffs'] = {'a': coeffs[0], 'b': coeffs[1]}
        result['predict'] = predict
        
    elif case == 'quadratic':
        basis_funcs = [basis_x2, basis_x, basis_1]
        basis_names = ["x^2", "x", ""]
        coeffs, fitted_y, equation_str, predict = solve_general_least_squares(x, y, basis_funcs, basis_names)
        result['coeffs'] = {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}
        result['predict'] = predict

    elif case == 'trig':
        basis_funcs = [basis_1, basis_cos, basis_sin]
        basis_names = ["", "cos(x)", "sin(x)"]
        coeffs, fitted_y, equation_str, predict = solve_general_least_squares(x, y, basis_funcs, basis_names)
        result['coeffs'] = {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}
        result['predict'] = predict

    elif case == 'exponential':
        # y = a * e^(bx)
        if np.any(y <= 0):
            raise ValueError("y values must be positive for exponential regression")
        y_log = np.log(y)
        basis_funcs = [basis_x, basis_1]
        basis_names = ["x", "1"]
        coeffs, _, _, _ = solve_general_least_squares(x, y_log, basis_funcs, basis_names)
        
        b = coeffs[0]
        a = np.exp(coeffs[1])
        fitted_y = a * np.exp(b * x)
        equation_str = f"y = {a:.4f} * e^({b:.4f}x)"
        result['coeffs'] = {'a': a, 'b': b}
        result['predict'] = lambda x_in: a * np.exp(b * x_in)

    elif case == 'power':
        # y = a * x^b
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("x and y positive for power")
        x_log = np.log(x)
        y_log = np.log(y)
        A_mat = np.vstack([x_log, np.ones(n)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y_log, rcond=None)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        
        fitted_y = a * np.power(x, b)
        equation_str = f"y = {a:.4f} * x^{b:.4f}"
        result['coeffs'] = {'a': a, 'b': b}
        result['predict'] = lambda x_in: a * np.power(x_in, b) if np.all(x_in > 0) else np.nan

    else:
        raise ValueError(f"Unknown case: {case}")
        
    residuals = y - fitted_y
    ssr = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ssr / sst) if sst != 0 else 0
    rmse = np.sqrt(ssr / n)
    
    result['fitted_y'] = fitted_y
    result['equation'] = equation_str
    result['metrics'] = {
        'SSR (Sum Squared Residuals)': ssr,
        'RMSE (Root Mean Squared Error)': rmse,
        'R^2': r_squared
    }
    
    return result
