import numpy as np
import pandas as pd
import os

def load_data(eps, data_dir, N, radius):
    """
    Load simulation data for a specific epsilon value.

    Parameters:
    - eps: The epsilon value.
    - data_dir: Directory where data files are stored.
    - N: Number of points in the simulation.
    - radius: Radius of the circular area.

    Returns:
    - density_ratio: Numpy array of density ratio values.
    """
    data_file = os.path.join(data_dir, f'simulation_data_N{N}_radius{int(radius)}_eps{eps:.2f}.csv')
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Skipping eps = {eps:.2f}")
        return None

    # Load the data
    df = pd.read_csv(data_file)

    # Filter valid clusters
    df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
    if df_valid.empty or len(df_valid) < 2000:
        print(f"Not enough valid clusters for eps = {eps:.2f}. Skipping.")
        return None

    # Compute lambda_prime and density ratio
    lambda0 = N / (np.pi * radius**2)  # Initial point density
    df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
    density_ratio = df_valid['lambda_prime'].values / lambda0

    return density_ratio


def sample_data(density_ratio, sample_size=100000, random_seed=42):
    """
    Sample data for computational efficiency.

    Parameters:
    - density_ratio: Numpy array of density ratio values.
    - sample_size: Maximum number of samples to draw.
    - random_seed: Seed for reproducibility.

    Returns:
    - ratio_sample: Numpy array of sampled density ratio values.
    """
    if len(density_ratio) == 0:
        return None

    sample_size = min(sample_size, len(density_ratio))
    np.random.seed(random_seed)
    ratio_sample = np.random.choice(density_ratio, size=sample_size, replace=False)
    return ratio_sample

def load_fit_parameters(output_dir, eps):
    """
    Load precomputed fit parameters from CSV files for a specific eps.

    Parameters:
    - output_dir: Directory where fit CSV files are stored.
    - eps: The epsilon value.

    Returns:
    - regular_fit: Pandas Series containing regular fit parameters and statistics.
    - mixture_fit: Pandas Series containing mixture fit parameters and statistics (if exists).
    """
    regular_fit_filename = f'regular_fit_eps{eps:.2f}.csv'
    mixture_fit_filename = f'mixture_fit_eps{eps:.2f}.csv'
    regular_fit_path = os.path.join(output_dir, regular_fit_filename)
    mixture_fit_path = os.path.join(output_dir, mixture_fit_filename)

    if not os.path.exists(regular_fit_path):
        print(f"Regular fit file {regular_fit_path} not found for eps = {eps:.2f}. Skipping.")
        regular_fit = None
    else:
        regular_fit = pd.read_csv(regular_fit_path).iloc[0]

    if not os.path.exists(mixture_fit_path):
        mixture_fit = None
    else:
        mixture_fit = pd.read_csv(mixture_fit_path).iloc[0]

    return regular_fit, mixture_fit

def load_density_ratio(eps, data_dir, N, radius, sample_size=100000, random_seed=42):
    """
    Load and sample density_ratio data for plotting.

    Parameters:
    - eps: The epsilon value.
    - data_dir: Directory where data files are stored.
    - N: Number of points in the simulation.
    - radius: Radius of the circular area.
    - sample_size: Number of data points to sample.
    - random_seed: Seed for reproducibility.

    Returns:
    - density_ratio_sample: Numpy array of sampled density ratio values.
    """
    data_file = os.path.join(data_dir, f'simulation_data_N{N}_radius{int(radius)}_eps{eps:.2f}.csv')
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found for eps = {eps:.2f}. Skipping.")
        return None

    # Load the data
    df = pd.read_csv(data_file)

    # Filter valid clusters
    df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
    if df_valid.empty or len(df_valid) < 2000:
        print(f"Not enough valid clusters for eps = {eps:.2f}. Skipping.")
        return None

    # Compute lambda_prime and density ratio
    lambda0 = N / (np.pi * radius**2)  # Initial point density
    df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
    density_ratio = df_valid['lambda_prime'].values / lambda0

    # Sample data
    sample_size = min(sample_size, len(density_ratio))
    np.random.seed(random_seed)
    density_ratio_sample = np.random.choice(density_ratio, size=sample_size, replace=False)

    return density_ratio_sample