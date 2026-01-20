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
from methods.Interpolation.Spline import spline_interpolation

# ==========================================
# INPUT PARAMETERS
# ==========================================
INPUT_CSV = None

# Default Dataset (Runge function example or simple curve)
DEFAULT_X = [0, 1, 2, 3, 4, 5]
DEFAULT_Y = [2.1, 7.7, 13.6, 27.2, 40.9, 61.1]

DEGREES = [2, 3, 4]

def main():
    console = Console(record=True)
    method_name = "Spline_Interpolation"
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'output', method_name))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    if INPUT_CSV and os.path.exists(INPUT_CSV):
        try:
            df_in = pd.read_csv(INPUT_CSV)
            if 'x' in df_in.columns and 'y' in df_in.columns:
                x = df_in['x'].values
                y = df_in['y'].values
                console.print(f"[green]Loaded data from {INPUT_CSV}[/green]")
            else:
                console.print(f"[red]CSV must contain 'x' and 'y' columns. Using default data.[/red]")
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
    # Generate dense grid for evaluation and plotting
    x_dense = np.linspace(x.min(), x.max(), 200)
    
    export_df = pd.DataFrame({'x_query': x_dense})
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Control Points', zorder=5)

    table = Table(title="Spline Interpolation Information")
    table.add_column("Degree", justify="center", style="cyan")
    table.add_column("Type", justify="center", style="magenta")
    table.add_column("Status", justify="center", style="green")

    for deg in DEGREES:
        try:
            # Check if enough points (Need at least k+1 points)
            if len(x) < deg + 1:
                console.print(f"[red]Not enough points for degree {deg} (Need {deg+1}, have {len(x)})[/red]")
                table.add_row(str(deg), "-", "Insufficient Data")
                continue
                
            res = spline_interpolation(x, y, degree=deg, x_query=x_dense)
            y_dense = res['y_query']
            
            col_name = f'y_spline_deg{deg}'
            export_df[col_name] = y_dense
            
            label_map = {2: "Quadratic (Bậc 2)", 3: "Cubic (Bậc 3)", 4: "Quartic (Bậc 4)"}
            label = label_map.get(deg, f"Degree {deg}")
            
            plt.plot(x_dense, y_dense, label=label)
            table.add_row(str(deg), label, "Success")
            
        except Exception as e:
            console.print(f"[red]Error for degree {deg}: {e}[/red]")
            table.add_row(str(deg), str(e), "Failed")

    console.print(table)

    # 3. Save CSV
    csv_filename = os.path.join(output_dir, f"{method_name}.csv")
    export_df.to_csv(csv_filename, index=False)
    console.print(f"[bold yellow]Đã xuất file kết quả nội suy (dense grid): {csv_filename}[/bold yellow]")

    # 4. Save Plot
    plt.title("Spline Interpolation Comparison")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_filename = os.path.join(output_dir, f"graph_{method_name}.png")
    plt.savefig(img_filename)
    console.print(f"[bold yellow]Đã lưu đồ thị: {img_filename}[/bold yellow]")
    
    # 5. Save Report
    txt_filename = os.path.join(output_dir, f"{method_name}.txt")
    console.save_text(txt_filename)
    console.print(f"[bold yellow]Đã lưu báo cáo text: {txt_filename}[/bold yellow]")

if __name__ == "__main__":
    main()
