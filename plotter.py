import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_data(data_file, min_cluster_size, min_area):
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found.")
        return

    df = pd.read_csv(data_file)
    if df.empty:
        print("Data file is empty.")
        return

    total_iterations = df['iteration'].max()

    # Remove placeholder entries with -1 values
    df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)]

    if df_valid.empty:
        print("No valid data to plot.")
        return

    S_values = df_valid['S_prime'].values
    N_values = df_valid['N_prime'].values.astype(int)

    # Adjust axes ranges and bins
    min_N = min_cluster_size - 1
    max_N = N_values.max() + 1
    max_S = np.percentile(S_values, 99)

    bins_N = np.arange(min_N, max_N + 1, 1)
    bins_S = np.linspace(min_area, max_S, 50)

    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(
        S_values, N_values,
        bins=[bins_S, bins_N],
        density=False
    )

    # Normalize frequency per 1000 iterations
    hist /= (total_iterations / 1000)

    # Plot the normalized histogram
    X, Y = np.meshgrid(xedges, yedges)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, hist.T, cmap='viridis')
    plt.xlabel("Cluster Area (S')")
    plt.ylabel("Number of Points in Cluster (N')")
    plt.title(f"Distribution of Cluster Area vs. Number of Points\n(Data from {data_file})")
    cbar = plt.colorbar(label='Average Frequency per 1000 Iterations')
    plt.tight_layout()
    plt.show()

def main():
    # Parameters should match those used in simulator.py
    N = 10000
    radius = 100
    eps = 1.6
    min_samples = 10
    min_cluster_size = 10
    min_area = 0.5

    # Data filename (same as in simulator.py)
    data_file = f'simulation_data_N{N}_radius{radius}_eps{eps}.csv'

    plot_data(data_file, min_cluster_size, min_area)

if __name__ == "__main__":
    main()
