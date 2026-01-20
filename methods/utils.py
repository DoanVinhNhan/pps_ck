import pandas as pd
import numpy as np
import os
from rich.console import Console

console = Console()

def load_data(filepath, x_col='x', y_col='y', transpose=False):
    """
    Load data from a CSV file.

    Parameters:
    - filepath: Path to the CSV file.
    - x_col: Name or index of the column/row for x.
    - y_col: Name or index of the column/row for y.
    - transpose: If True, treats rows as columns (reads data horizontally).

    Returns:
    - x_nodes (np.array): Array of x values.
    - y_nodes (np.array): Array of y values.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        if transpose:
            # Read header=None to treat all as data first, then select rows
            df = pd.read_csv(filepath, header=None)
            # Assuming x_col and y_col are indices (int) when transposing, or we try to match first element
            # But usually if rows, we expect indices.
            # Let's try to interpret x_col as index if int
            
            # Helper to get row by index or string match
            def get_row(identifier):
                if isinstance(identifier, int):
                    return df.iloc[identifier, :].values
                else:
                    # Try to find row starting with identifier
                    for idx, row in df.iterrows():
                        if str(row[0]) == identifier:
                             # Return rest of row? Or including first? Usually header is skipped
                             return row[1:].values
                    raise ValueError(f"Could not find row with identifier '{identifier}'")

            # For simplicity in this specific task context where user might say "Row 0 is x", "Row 1 is y"
            x_nodes = get_row(x_col)
            y_nodes = get_row(y_col)

            # Convert to float, filtering out NaNs
            x_nodes = pd.to_numeric(x_nodes, errors='coerce')
            y_nodes = pd.to_numeric(y_nodes, errors='coerce')
            
            mask = ~np.isnan(x_nodes) & ~np.isnan(y_nodes)
            x_nodes = x_nodes[mask]
            y_nodes = y_nodes[mask]

        else:
            # Normal column reading
            df = pd.read_csv(filepath)
            
            # If x_col is not in columns, try to see if it's an index
            if x_col not in df.columns:
                # If int, use iloc
                if isinstance(x_col, int):
                    x_vals = df.iloc[:, x_col]
                else:
                    # Try to match stripping whitespace
                    found = False
                    for c in df.columns:
                        if c.strip() == x_col.strip():
                            x_vals = df[c]
                            found = True
                            break
                    if not found:
                         # Fallback: maybe the user passed an index as string "0"
                        try:
                            idx = int(x_col)
                            x_vals = df.iloc[:, idx]
                        except:
                            raise ValueError(f"Column '{x_col}' not found in CSV.")
            else:
                x_vals = df[x_col]

            if y_col not in df.columns:
                # Simular logic for y
                if isinstance(y_col, int):
                    y_vals = df.iloc[:, y_col]
                else:
                    found = False
                    for c in df.columns:
                        if c.strip() == y_col.strip():
                            y_vals = df[c]
                            found = True
                            break
                    if not found:
                        try:
                            idx = int(y_col)
                            y_vals = df.iloc[:, idx]
                        except:
                            raise ValueError(f"Column '{y_col}' not found in CSV.")
            else:
                y_vals = df[y_col]

            x_nodes = pd.to_numeric(x_vals, errors='coerce').values
            y_nodes = pd.to_numeric(y_vals, errors='coerce').values

            # Remove NaNs
            mask = ~np.isnan(x_nodes) & ~np.isnan(y_nodes)
            x_nodes = x_nodes[mask]
            y_nodes = y_nodes[mask]

    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {e}")
        raise e

    # Validation: Check for duplicate x
    if len(x_nodes) != len(np.unique(x_nodes)):
        console.print("[bold red]CẢNH BÁO: Dữ liệu chứa các giá trị x bị trùng lặp![/bold red]")
        # Optional: Setup strategy to handle? Taking mean? 
        # For now, just warn as requested.

    # Validation: Check sorted
    if not np.all(np.diff(x_nodes) >= 0):
        console.print("[bold yellow]Cảnh báo: Dữ liệu x không được sắp xếp tăng dần. Đang sắp xếp lại...[/bold yellow]")
        sorted_indices = np.argsort(x_nodes)
        x_nodes = x_nodes[sorted_indices]
        y_nodes = y_nodes[sorted_indices]

    return x_nodes, y_nodes
