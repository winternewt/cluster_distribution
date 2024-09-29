import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(eps_values, data_dir, N, radius):
    # Load data for specified eps values
    data_dict = {}
    lambda0 = N / (np.pi * radius**2)  # Initial point density
    for eps in eps_values:
        data_file = os.path.join(data_dir, f'simulation_data_N{N}_radius{radius}_eps{eps:.2f}.csv')
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping eps = {eps:.2f}")
            continue
        # Load the data
        df = pd.read_csv(data_file)
        # Filter valid clusters
        df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
        if df_valid.empty or len(df_valid) < 2000:
            print(f"Not enough valid clusters for eps = {eps:.2f}. Skipping.")
            continue
        # Compute lambda_prime and density ratio
        df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
        df_valid['density_ratio'] = df_valid['lambda_prime'] / lambda0
        data_dict[eps] = df_valid
        print(f"Data loaded for eps = {eps:.2f}, total clusters: {len(df_valid)}")
    return data_dict

def fit_gamma_mixture(data):
    # Fit a mixture of Gamma distributions for each N_prime
    N_prime_values = data['N_prime'].unique()
    total_clusters = len(data)
    mixture_weights = []
    gamma_params = []
    for N in N_prime_values:
        subset = data[data['N_prime'] == N]
        density_ratio = subset['density_ratio'].values
        if len(density_ratio) < 5:
            continue
        params = stats.gamma.fit(density_ratio, floc=0)
        gamma_params.append((N, params))
        weight = len(density_ratio) / total_clusters
        mixture_weights.append((N, weight))
    return gamma_params, mixture_weights

def mixture_pdf(x, gamma_params, mixture_weights):
    # Compute the PDF of the mixture of Gamma distributions
    pdf = np.zeros_like(x)
    for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
        pdf += weight * stats.gamma.pdf(x, *params)
    return pdf

def plot_overlapping_histograms_with_fits(data_dict, eps_values):
    plt.figure(figsize=(12, 8))

    # Prepare colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_values)))

    # X-axis for PDF plotting
    # Determine the combined data range
    all_density_ratios = np.concatenate([data['density_ratio'].values for data in data_dict.values()])
    x_min = 0
    x_max = np.percentile(all_density_ratios, 99.99) # Use 99th percentile to avoid extreme tails
    #x_max = np.max(all_density_ratios)
    x = np.linspace(x_min, x_max, 1000)

    # Plot histograms and fitted curves
    for idx, eps in enumerate(eps_values):
        if eps not in data_dict:
            continue
        data = data_dict[eps]
        density_ratio = data['density_ratio'].values

        # Plot histogram
        plt.hist(density_ratio, bins=100, range=(x_min, x_max), density=True, histtype='stepfilled',
                 alpha=0.3, label=f'eps = {eps:.2f}', color=colors[idx], edgecolor='none')

        # Fit mixture model
        gamma_params, mixture_weights = fit_gamma_mixture(data)
        y = mixture_pdf(x, gamma_params, mixture_weights)

        # Plot fitted curve
        plt.plot(x, y, color=colors[idx], linewidth=2)

    plt.xlabel('Density Ratio (lambda\' / lambda0)')
    plt.ylabel('Density')
    plt.title('Overlapping Histograms and Mixture Model Fits for Different Eps Values')
    # Adjust legend to prevent clutter
    # Show legend for selected eps values (e.g., every 0.05)
    selected_eps_indices = [i for i, eps in enumerate(eps_values) if (eps * 100) % 5 == 0]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[i] for i in selected_eps_indices],
               [labels[i] for i in selected_eps_indices],
               loc='upper right', fontsize='small', ncol=2)
    plt.show()

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area

    # Eps values from 1.10 to 1.40 with increment of 0.01
    eps_values = np.arange(1.10, 1.41, 0.05).round(2)
    data_dir = './simdata/v2/'  # Directory where data files are stored

    # Load data
    data_dict = load_data(eps_values, data_dir, N, radius)

    # Plot overlapping histograms with fits
    plot_overlapping_histograms_with_fits(data_dict, eps_values)

if __name__ == "__main__":
    main()
