import os
import json
import google.generativeai as genai
import textwrap

# --- Configuration ---
METHODS_JSON_PATH = "methods.json"
OUTPUT_DIR = "methods"
API_KEY = "AIzaSyBRcWLKIx-oGdc9VU0PNUvUtXsgVxx85n0"

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=API_KEY)

# --- Target Methods ---
TARGET_METHODS = [
    "Tích phân Hình thang (Trapezoidal Rule for Integration)",
    "Tích phân Simpson (Simpson's Rule)",
    "Tích phân Newton-Cotes"
]

FILE_MAPPING = {
    "Tích phân Hình thang (Trapezoidal Rule for Integration)": "Integration_Trapezoidal.py",
    "Tích phân Simpson (Simpson's Rule)": "Integration_Simpson.py",
    "Tích phân Newton-Cotes": "Integration_NewtonCotes.py"
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are an expert Python Numerical Analyst.
Based on the following JSON description of a Numerical Integration method, write a robust, production-ready Python file.

### Method Description (INCOMPLETE, FOR REFERENCE ONLY):
{method_json}

### Specific Requirements (CRITICAL):
1.  **Goal**: Implement the algorithm to calculate the integral $\\int_{{a}}^{{b}} g(f(x), x) dx$.
2.  **Inputs**:
    *   `x_nodes`, `y_nodes`: Lists/Arrays representing the discrete values of f(x).
    *   `g`: A callable python function `g(f_val, x_val)` representing the integrand.
        *   Default `g` (if None) is `lambda f, x: f` (i.e. integrate f(x) directly).
    *   `a` (lower bound), `b` (upper bound).
    *   `epsilon` (optional): If provided, perform iterative calculation (halving step size h) until error < epsilon.
    *   **Specially for Newton-Cotes** (SKIP THIS FOR TRAPEZOIDAL AND SIMPSON): Add degree `n` (int) as an input parameter (default=4, allowed=[4,5,6]).
3.  **Data Handling & Interpolation (STRICT)**:
    *   The input `f(x)` is **discrete data** (`x_nodes`, `y_nodes`).
    *   **INITIALIZATION CONSTRAINT**: The initial number of intervals `n_initial` (or `n_current` at start) **MUST** be equal to `(Number of x_nodes points within [a, b]) - 1`. Do NOT default to 4 or 1.
        *   **For Simpson's Rule**: If the determined `n_initial` is odd, increase it by 1 to make it even (required for Simpson).
    *   To support `epsilon` (iterative refinement with h/2), you MUST implement the following **Interpolation Strategy**:
        *   USE NODE IN h GRID to interpolate f(x) at query point x.
        *   DO NOT USE NEW NODE IN h/2 GRID to interpolate f(x) at query point x.
        *   **Main Region**: Use **6-point Lagrange Interpolation** utilizing the 6 nearest nodes around the query point $x$.
        *   **Left Boundary**: If there are not enough points to the left for centered Lagrange, use **Newton Forward Interpolation**.
        *   **Right Boundary**: If there are not enough points to the right, use **Newton Backward Interpolation**.
    *   Calculate integrand values as `val = g(interpolated_f(curr_x), curr_x)`.
    *   Length of x_nodes and y_nodes between a and b is initial number of divisions + 1: N.
4.  **Error Estimation**:
    *   Evaluate error using Runge principle: $|I_h - I_{{h/2}}|$.
5.  **Outputs**:
    *   **CRITICAL**: Return a dictionary containing **ALL** possible values:
        *   `result`: The final integral value.
        *   `error_estimate`: The calculated error.
        *   `intermediate_values`: A dictionary containing **EVERY** variable calculated (coefficients, h, selected points, etc.).
        *   `computation_process`: A list of strings logging detailed steps.
6.  **Code Structure Constraints (STRICT)**:
    *   **OUTPUT ONLY THE FUNCTION DEFINITION(S)**.
    *   **DO NOT** include any `if __name__ == "__main__":` block.
    *   **DO NOT** include example usage or drivers.
    *   **DO NOT** include any code outside the functions.
    *   Docstrings in Vietnamese.

### Output Format:
Return ONLY the raw Python code. Do not include markdown fencing.
"""

def generate_code(method_data):
    model = genai.GenerativeModel('gemini-3-pro-preview')
    prompt = PROMPT_TEMPLATE.format(method_json=json.dumps(method_data, indent=2, ensure_ascii=False))
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating code for {method_data.get('method_name')}: {e}")
        return None

def clean_code(code_text):
    if not code_text: return ""
    # Strip markdown code blocks
    if code_text.startswith("```"):
        lines = code_text.splitlines()
        # Remove first line if it starts with ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it starts with ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return code_text

def main():
    if not os.path.exists(METHODS_JSON_PATH):
        print(f"File not found: {METHODS_JSON_PATH}")
        return

    with open(METHODS_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for item in data:
        name = item.get("method_name")
        if name in TARGET_METHODS:
            print(f"Generating code for: {name}...")
            raw_code = generate_code(item)
            cleaned_code = clean_code(raw_code)
            
            if cleaned_code:
                filename = FILE_MAPPING.get(name, f"{name.replace(' ', '_')}.py")
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as out_f:
                    out_f.write(cleaned_code)
                print(f"Saved to {filepath}")
            else:
                print(f"Failed to generate code for {name}")

if __name__ == "__main__":
    main()
