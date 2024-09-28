import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


def load_and_aggregate_data(eps_values, data_dir, lambda0):
    # Dictionary to store data for each eps
    data_dict = {}

    for eps in eps_values:
        data_file = os.path.join(data_dir, f'simulation_data_N10000_radius100_eps{eps:.2f}.csv')
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping eps = {eps:.2f}")
            continue

        # Load the data
        df = pd.read_csv(data_file)
        # Filter valid clusters
        df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
        if df_valid.empty:
            continue
        # Compute lambda_prime and ratio
        df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
        df_valid['ratio'] = df_valid['lambda_prime'] / lambda0
        # Store the ratio data
        data_dict[eps] = df_valid['ratio'].values

    return data_dict


def generate_qq_plot_across_eps(data_dict, subsample_size=5000):
    # Prepare data for Q-Q plots
    eps_list = sorted(data_dict.keys())
    quantiles_list = []
    eps_values_list = []
    theoretical_quantiles = None

    for eps in eps_list:
        data = data_dict[eps]
        # Subsample if necessary
        if len(data) > subsample_size:
            data_sample = np.random.choice(data, size=subsample_size, replace=False)
        else:
            data_sample = data
        # Sort data
        data_sample.sort()
        # Compute theoretical quantiles
        percs = np.linspace(0, 100, len(data_sample))
        qn_data = np.percentile(data_sample, percs)
        qn_theoretical = stats.norm.ppf(percs / 100)
        if theoretical_quantiles is None:
            theoretical_quantiles = qn_theoretical
        # Store quantiles
        quantiles_list.append(qn_data)
        eps_values_list.append(np.full_like(qn_data, eps))

    # Convert lists to arrays
    quantiles_array = np.concatenate(quantiles_list)
    eps_array = np.concatenate(eps_values_list)
    theoretical_quantiles = np.tile(theoretical_quantiles, len(eps_list))

    # Create a DataFrame for plotting
    qq_df = pd.DataFrame({
        'eps': eps_array,
        'sample_quantiles': quantiles_array,
        'theoretical_quantiles': theoretical_quantiles
    })

    # Create a 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(qq_df['theoretical_quantiles'], qq_df['eps'], qq_df['sample_quantiles'],
                   c=qq_df['eps'], cmap='viridis', marker='o', s=1)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Epsilon (eps)')
    ax.set_zlabel('Sample Quantiles')
    fig.colorbar(p, ax=ax, label='Epsilon (eps)')
    plt.title('3D Q-Q Plot Across Epsilon Values')
    plt.show()


def generate_2d_density_plot(data_dict):
    # Prepare data for 2D density plot
    eps_list = sorted(data_dict.keys())
    eps_values = []
    ratio_values = []

    for eps in eps_list:
        data = data_dict[eps]
        eps_values.extend([eps] * len(data))
        ratio_values.extend(data)

    # Create a DataFrame
    density_df = pd.DataFrame({
        'eps': eps_values,
        'ratio': ratio_values
    })

    # Create a pivot table for the heatmap
    # We'll bin the data for both eps and ratio
    eps_bins = np.linspace(min(eps_list), max(eps_list), 100)
    ratio_bins = np.linspace(min(ratio_values), max(ratio_values), 100)

    density_df['eps_bin'] = pd.cut(density_df['eps'], bins=eps_bins, labels=eps_bins[:-1])
    density_df['ratio_bin'] = pd.cut(density_df['ratio'], bins=ratio_bins, labels=ratio_bins[:-1])

    # Compute counts
    heatmap_data = density_df.pivot_table(index='ratio_bin', columns='eps_bin', aggfunc='size', fill_value=0)

    # Convert bins to numeric
    heatmap_data.index = heatmap_data.index.astype(float)
    heatmap_data.columns = heatmap_data.columns.astype(float)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', norm=None)
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Density Ratio (lambda\' / lambda0)')
    plt.title('2D Density Plot of Ratio Across Epsilon Values')
    plt.show()


def generate_3d_surface_plot(data_dict):
    # Prepare data for 3D surface plot
    eps_list = sorted(data_dict.keys())
    eps_bins = np.linspace(min(eps_list), max(eps_list), 50)
    ratio_values = np.concatenate(list(data_dict.values()))
    ratio_bins = np.linspace(min(ratio_values), max(ratio_values), 50)

    eps_indices = []
    ratio_indices = []
    counts = []

    for i, eps in enumerate(eps_bins[:-1]):
        eps_min = eps_bins[i]
        eps_max = eps_bins[i + 1]
        eps_center = (eps_min + eps_max) / 2
        if eps_center in data_dict:
            data = data_dict[eps_center]
            hist, ratio_edges = np.histogram(data, bins=ratio_bins)
            eps_indices.extend([eps_center] * len(hist))
            ratio_centers = (ratio_bins[:-1] + ratio_bins[1:]) / 2
            ratio_indices.extend(ratio_centers)
            counts.extend(hist)

    # Create a 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    eps_grid, ratio_grid = np.meshgrid(eps_bins[:-1], ratio_bins[:-1])
    counts_grid = np.zeros_like(eps_grid)

    for e, r, c in zip(eps_indices, ratio_indices, counts):
        e_idx = np.searchsorted(eps_bins, e) - 1
        r_idx = np.searchsorted(ratio_bins, r) - 1
        counts_grid[r_idx, e_idx] = c

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(eps_grid, ratio_grid, counts_grid, cmap='viridis')
    ax.set_xlabel('Epsilon (eps)')
    ax.set_ylabel('Density Ratio (lambda\' / lambda0)')
    ax.set_zlabel('Frequency')
    fig.colorbar(surf, ax=ax, label='Frequency')
    plt.title('3D Surface Plot of Ratio Across Epsilon Values')
    plt.show()


def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area

    # Constants
    S0 = np.pi * radius ** 2  # Total area of the circle
    lambda0 = N / S0  # Initial point density

    # Eps values to analyze
    eps_values = np.arange(0.90, 1.40, 0.01).round(2)  # Ensure rounding consistency

    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Load and aggregate data
    data_dict = load_and_aggregate_data(eps_values, data_dir, lambda0)

    # Generate Q-Q plot across eps
    generate_qq_plot_across_eps(data_dict)

    # Generate 2D density plot
    generate_2d_density_plot(data_dict)

    # Generate 3D surface plot
    generate_3d_surface_plot(data_dict)


if __name__ == "__main__":
    main()
