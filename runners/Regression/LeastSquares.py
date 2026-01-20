import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from methods.Regression.LeastSquares import least_squares_regression

# ==========================================
# INPUT PARAMETERS
# ==========================================
# Use CSV if provided, else use default data
# Path relative to this script: ../../20251GKtest2.csv
INPUT_CSV = os.path.join(os.path.dirname(__file__), '../../20251GKtest2.csv')
X_COL = "x_i"
Y_COL = "y_i"

# Default Dataset (Example)
# x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
# y = np.array([2.1, 3.9, 8.2, 16.5, 31.8, 65.0], dtype=float) # Exponetial-ish
# Let's use a dataset that might fit multiple loosely for demonstration
DEFAULT_X = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
DEFAULT_Y = [1.0, 1.8, 3.3, 6.0, 11.0, 20.0]

CASES = ['general']

def main():
    console = Console(record=True)
    method_name = "LeastSquares_Regression"
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'output', method_name))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    if INPUT_CSV and os.path.exists(INPUT_CSV):
        try:
            df_in = pd.read_csv(INPUT_CSV)
            if X_COL in df_in.columns and Y_COL in df_in.columns:
                x = df_in[X_COL].values
                y = df_in[Y_COL].values
                console.print(f"[green]Loaded data from {INPUT_CSV}[/green]")
            else:
                console.print(f"[red]CSV must contain '{X_COL}' and '{Y_COL}' columns. Using default data.[/red]")
                x = np.array(DEFAULT_X)
                y = np.array(DEFAULT_Y)
        except Exception as e:
            console.print(f"[red]Error reading CSV: {e}. Using default data.[/red]")
            x = np.array(DEFAULT_X)
            y = np.array(DEFAULT_Y)
    else:
        console.print("[yellow]No CSV provided or found. Using default data.[/yellow]")
        x = np.array(DEFAULT_X)
        y = np.array(DEFAULT_Y)

    # Display Input Data
    console.print(Panel(
        f"x: {x}\ny: {y}",
        title="[bold green]Input Data[/bold green]",
        expand=False
    ))

    # 2. Run Methods
    results = {}
    
    # Store fitted y for CSV export
    export_data = {'x': x, 'y_original': y}
    
    # Helper for table
    table = Table(title="Regression Results Comparison")
    table.add_column("Case", style="cyan")
    table.add_column("Equation", style="magenta")
    table.add_column("R^2", justify="right", style="green")
    table.add_column("RMSE", justify="right", style="red")

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', label='Original Data', zorder=5)

    for case in CASES:
        try:
            # Handle potential errors (e.g. log of negative numbers)
            res = least_squares_regression(x, y, case)
            results[case] = res
            
            # Add to table
            metrics = res['metrics']
            table.add_row(
                case.capitalize(),
                res['equation'],
                f"{metrics['R^2']:.6g}",
                f"{metrics['RMSE (Root Mean Squared Error)']:.6g}"
            )
            
            # Add to export data
            export_data[f'y_{case}'] = res['fitted_y']
            
            # Plot
            # For smooth plotting, generate dense x
            x_dense = np.linspace(x.min(), x.max(), 100)
            
            # Re-evaluate equation on dense grid ? 
            # Simplified: just plot the fitted points or re-calc if possible.
            # To be accurate with the equation, let's just plot the fitted_y connected (linear interp of fit)
            # or better re-calculate using coeffs.
            
            # Re-calculating for smooth plot
            # Plot using the returned predict function
            if 'predict' in res:
                # Handle potential domain errors in predict if needed (e.g. log(negative))
                # For plotting dense grid, ensure input is safe for specific methods if needed
                # But our generic predict should handle it or return meaningful values
                if case == 'power':
                     # Filter for power
                     valid_idx = x_dense > 0
                     x_plot = x_dense[valid_idx]
                     y_plot = res['predict'](x_plot)
                     plt.plot(x_plot, y_plot, label=f"{case} (R^2={metrics['R^2']:.6g})")
                else:
                     y_dense = res['predict'](x_dense)
                     plt.plot(x_dense, y_dense, label=f"{case} (R^2={metrics['R^2']:.6g})")
            elif case == 'general':
                # Fallback if predict not in res (should be there)
                pass

        except Exception as e:
            console.print(f"[red]Failed to run {case}: {e}[/red]")
            export_data[f'y_{case}'] = [np.nan] * len(x)
            table.add_row(case.capitalize(), "Failed", "-", "-")

    # 3. Print Summary
    console.print(table)

    # Print Matrices for General Case
    if 'general' in results and 'matrices' in results['general']:
        m = results['general']['matrices']
        basis_names = m['basis_names']
        AtA = m['AtA']
        AtY = m['AtY']
        
        # Display Normal Matrix (AtA)
        mat_table = Table(title="Ma trận Phương trình Chuẩn (A^T * A)")
        # Add columns
        mat_table.add_column("Basis", style="bold cyan")
        for name in basis_names:
            mat_table.add_column(name, justify="right")
        
        # Add rows
        for i, row in enumerate(AtA):
            row_str = [f"{val:.6g}" for val in row]
            mat_table.add_row(basis_names[i], *row_str)
            
        console.print(mat_table)
        
        # Display RHS Vector (AtY)
        rhs_table = Table(title="Vectơ Vế phải (A^T * Y)")
        rhs_table.add_column("Basis", style="bold cyan")
        rhs_table.add_column("Value", justify="right")
        
        for i, val in enumerate(AtY):
            rhs_table.add_row(basis_names[i], f"{val:.6g}")
            
        console.print(rhs_table)
    
    # 4. Save CSV
    df_out = pd.DataFrame(export_data)
    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    df_out.to_csv(csv_filename, index=False)
    console.print(f"[bold yellow]Đã xuất file kết quả: {csv_filename}[/bold yellow]")

    # 5. Save Plot
    plt.title("Least Squares Regression Comparison")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
    plt.savefig(img_filename)
    console.print(f"[bold yellow]Đã lưu đồ thị: {img_filename}[/bold yellow]")

    # 6. Save Report
    txt_filename = os.path.join(output_dir, f"{method_name}.txt")
    console.save_text(txt_filename)
    console.print(f"[bold yellow]Đã lưu báo cáo text: {txt_filename}[/bold yellow]")

if __name__ == "__main__":
    main()
