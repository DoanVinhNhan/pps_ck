import numpy as np
from scipy.interpolate import make_interp_spline

def spline_interpolation(x, y, degree=3, x_query=None):
    """
    Perform Spline Interpolation of a given degree.
    
    Args:
        x (np.array): Independent variable data (must be sorted).
        y (np.array): Dependent variable data.
        degree (int): Degree of spline (2 for Quadratic, 3 for Cubic, 4 for Quartic).
        x_query (np.array, optional): Points to evaluate the spline at. 
                                      If None, generates a dense grid for plotting.
    
    Returns:
        dict: Contains 'model' (spline object), 'x_query', 'y_query', 'degree'.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Sort x if needed (Spline requires sorted x)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    # Create Spline
    # k=degree. 
    spline_model = make_interp_spline(x, y, k=degree)
    
    # Evaluation
    if x_query is None:
        # Create a dense grid for visualization
        x_query = np.linspace(x.min(), x.max(), 500)
    
    y_query = spline_model(x_query)
    
    return {
        'model': spline_model,
        'x_query': x_query,
        'y_query': y_query,
        'degree': degree,
        'original_x': x,
        'original_y': y
    }
