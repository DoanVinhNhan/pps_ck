
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'pps_ck')))

from methods.Integration_Simpson import simpson_integration
from methods.Integration_Trapezoidal import trapezoidal_integration
from methods.Integration_NewtonCotes import newton_cotes_integration

def test_simpson():
    print("\n--- Testing Simpson ---")
    # x^4, exact integral [0, 2] is 32/5 = 6.4
    # N = 12. Divisible by k=2 (N/k=6 even), k=3 (N/k=4 even). Should pick k=2.
    a, b = 0.0, 2.0
    N = 12
    x = np.linspace(a, b, N + 1)
    y = x**4
    
    res = simpson_integration(x, y, a, b, epsilon=None)
    err = res.get('error_estimate')
    logs = res.get('computation_process', [])
    
    print(f"Result: {res['result']}")
    print(f"Error Est: {err}")
    # Check logs for sparse grid info
    for line in logs:
        if "lưới thưa" in line:
            print(f"Log: {line}")
        if "Sai số ước lượng" in line:
            print(f"Log: {line}")
            
    assert err is not None and err != 0.0, "Simpson error should be estimated"

def test_trapezoidal():
    print("\n--- Testing Trapezoidal ---")
    # N = 10. Divisible by k=2, 5. Should pick k=2.
    a, b = 0.0, 2.0
    N = 10
    x = np.linspace(a, b, N + 1)
    y = x**2
    
    res = trapezoidal_integration(x, y, a, b, epsilon=None)
    err = res.get('error_estimate')
    logs = res.get('computation_process', [])
    
    print(f"Result: {res['result']}")
    print(f"Error Est: {err}")
    for line in logs:
        if "lưới thưa" in line:
            print(f"Log: {line}")
            
    assert err is not None and err != 0.0, "Trapezoidal error should be estimated"

def test_newton_cotes():
    print("\n--- Testing Newton-Cotes (n=4) ---")
    # Degree 4. N = 12.
    # k=2 -> N/k = 6. 6 % 4 != 0.
    # k=3 -> N/k = 4. 4 % 4 == 0. Should pick k=3.
    # Use e^x which is not a polynomial, so error should be non-zero
    a, b = 0.0, 3.0
    N = 12
    x = np.linspace(a, b, N + 1)
    y = np.exp(x)
    
    res = newton_cotes_integration(x, y, g=None, a=a, b=b, epsilon=None, degree=4)
    err = res.get('error_estimate')
    logs = res.get('computation_process', [])
    
    print(f"Result: {res['result']}")
    print(f"Error Est: {err}")
    for line in logs:
        if "lưới thưa" in line:
            print(f"Log: {line}")
            
    assert err is not None and err != 0.0, "Newton-Cotes error should be estimated"

if __name__ == "__main__":
    try:
        test_simpson()
        test_trapezoidal()
        test_newton_cotes()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
