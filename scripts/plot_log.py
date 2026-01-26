import argparse
import os
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print("This script requires pandas and matplotlib.")
    print(f"Error: {e}")
    print("Install them with: pip install pandas matplotlib")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV log file.")
    parser.add_argument("--file", type=str, help="Path to the CSV file (e.g., log/2023-10-27_..._metrics.csv)")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (optional). If not set, shows the plot.")
    parser.add_argument("--ma", type=int, default=10, help="Window size for moving average (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    # Load data
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if dataframe is empty
    if df.empty:
        print("Error: The CSV file is empty.")
        return

    # Statistics
    print("=" * 60)
    print(f"STATISTICS: {os.path.basename(args.file)}")
    print("=" * 60)
    print(f"Total Epochs: {len(df)}")
    
    if "Ret" in df.columns:
        max_ret = df['Ret'].max()
        max_ret_epoch = df.loc[df['Ret'].idxmax(), 'Epoch']
        last_10_percent = max(1, int(len(df) * 0.1))
        avg_last = df['Ret'].tail(last_10_percent).mean()
        
        print(f"Max Average Return:       {max_ret:.4f} (Epoch {max_ret_epoch})")
        print(f"Mean Return (Last {last_10_percent} eps): {avg_last:.4f}")
    
    print("-" * 60)
    # Print summary of all numeric columns
    print(df.describe().transpose()[['mean', 'std', 'min', 'max']])
    print("=" * 60)

    # Plotting
    # Filter columns to plot (exclude Epoch and Time usually, but keep them if needed)
    plot_cols = [col for col in ["Ret", "Len", "LossPi", "LossV", "Entropy", "KL"] if col in df.columns]
    
    if not plot_cols:
        print("No recognized metrics to plot.")
        return

    n_metrics = len(plot_cols)
    cols = 2
    rows = (n_metrics + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(plot_cols):
        ax = axes[i]
        ax.plot(df["Epoch"], df[metric], label="Raw", alpha=0.4, color='gray')
        
        # Rolling average for smoother plots
        if len(df) > args.ma:
            rolling = df[metric].rolling(window=args.ma).mean()
            ax.plot(df["Epoch"], rolling, label=f"MA ({args.ma})", linewidth=2)
        
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f"Plot saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()