import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(eps_values, data_dir, N, radius):
    # Load data for all eps values
    data_dict = {}
    for eps in eps_values:
        data_file = os.path.join(data_dir, f'simulation_data_N{N}_radius{radius}_eps{eps:.2f}.csv')
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping eps = {eps:.2f}")
            continue
        # Load the data
        df = pd.read_csv(data_file)
        # Filter valid clusters
        df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
        if df_valid.empty:
            print(f"No valid clusters found for eps = {eps:.2f}.")
            continue
        # Compute lambda_prime and ratio
        df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
        data_dict[eps] = df_valid
        print(f"Data loaded for eps = {eps:.2f}, total clusters: {len(df_valid)}")
    return data_dict

def fit_gamma_by_N_prime(data, lambda0):
    # For each N_prime, fit a Gamma distribution to the lambda_prime data
    N_prime_values = data['N_prime'].unique()
    total_clusters = len(data)
    mixture_weights = []
    gamma_params = []
    for N in N_prime_values:
        subset = data[data['N_prime'] == N]
        lambda_prime = subset['lambda_prime'].values
        # Skip if not enough data points
        if len(lambda_prime) < 5:
            continue
        # Fit Gamma distribution
        params = stats.gamma.fit(lambda_prime, floc=0)  # Fix location to zero
        gamma_params.append((N, params))
        # Calculate weight
        weight = len(lambda_prime) / total_clusters
        mixture_weights.append((N, weight))
    return gamma_params, mixture_weights

def mixture_gamma_pdf(x, gamma_params, mixture_weights):
    # Compute the PDF of the mixture of Gamma distributions
    pdf = np.zeros_like(x)
    for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
        pdf += weight * stats.gamma.pdf(x, *params)
    return pdf

def compute_goodness_of_fit_mixture(data, gamma_params, mixture_weights):
    # Prepare data
    lambda_prime = data['lambda_prime'].values
    # Compute empirical CDF
    ecdf = np.sort(lambda_prime)
    cdf_empirical = np.arange(1, len(ecdf)+1) / len(ecdf)
    # Compute theoretical CDF of the mixture
    cdf_mixture = np.zeros_like(ecdf)
    for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
        cdf_mixture += weight * stats.gamma.cdf(ecdf, *params)
    # Perform Kolmogorov-Smirnov test
    D = np.max(np.abs(cdf_empirical - cdf_mixture))
    n = len(ecdf)
    ks_stat = D * np.sqrt(n)
    # For large n, the p-value can be approximated
    ks_p_value = stats.kstest(lambda_prime, lambda x: mixture_cdf(x, gamma_params, mixture_weights))[1]
    # Compute log-likelihood
    log_likelihood = np.sum(np.log(mixture_gamma_pdf(lambda_prime, gamma_params, mixture_weights)))
    # Number of parameters: sum over individual Gamma distributions
    k = len(gamma_params) * 2  # Each Gamma has shape and scale parameters
    # Compute AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return ks_stat, ks_p_value, aic, bic

def mixture_cdf(x, gamma_params, mixture_weights):
    # Compute the CDF of the mixture at x
    cdf = 0.0
    for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
        cdf += weight * stats.gamma.cdf(x, *params)
    return cdf

def fit_gamma_per_eps(data, lambda0):
    # Fit a Gamma distribution to the lambda_prime data for each eps
    lambda_prime = data['lambda_prime'].values
    params = stats.gamma.fit(lambda_prime, floc=0)  # Fix location to zero
    # Compute goodness-of-fit
    D, p_value = stats.kstest(lambda_prime, 'gamma', args=params)
    log_likelihood = np.sum(np.log(stats.gamma.pdf(lambda_prime, *params)))
    k = 2  # Shape and scale parameters
    n = len(lambda_prime)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return params, D, p_value, aic, bic

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    lambda0 = N / (np.pi * radius**2)  # Initial point density

    # Eps values to include in the analysis
    eps_values = np.arange(0.92, 1.41, 0.01).round(2)
    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Load data
    data_dict = load_data(eps_values, data_dir, N, radius)

    # Part a): For all eps, fit mixture of Gammas by N_prime
    mixture_results = []
    for eps in eps_values:
        if eps not in data_dict:
            continue
        data = data_dict[eps]
        gamma_params, mixture_weights = fit_gamma_by_N_prime(data, lambda0)
        # Compute goodness-of-fit for the mixture model
        ks_stat, ks_p_value, aic, bic = compute_goodness_of_fit_mixture(data, gamma_params, mixture_weights)
        mixture_results.append({
            'eps': eps,
            'ks_stat': ks_stat,
            'ks_p_value': ks_p_value,
            'aic': aic,
            'bic': bic,
            'gamma_params': gamma_params,
            'mixture_weights': mixture_weights
        })
        print(f"Mixture model for eps = {eps:.2f}: KS Stat = {ks_stat:.5f}, p-value = {ks_p_value:.5f}, AIC = {aic:.2f}, BIC = {bic:.2f}")

    # Part b): For each eps, fit a single Gamma distribution
    gamma_results = []
    for eps in eps_values:
        if eps not in data_dict:
            continue
        data = data_dict[eps]
        params, D, p_value, aic, bic = fit_gamma_per_eps(data, lambda0)
        gamma_results.append({
            'eps': eps,
            'params': params,
            'ks_stat': D,
            'ks_p_value': p_value,
            'aic': aic,
            'bic': bic
        })
        print(f"Gamma fit for eps = {eps:.2f}: KS Stat = {D:.5f}, p-value = {p_value:.5f}, AIC = {aic:.2f}, BIC = {bic:.2f}")

    # Analyze variance of gamma parameters across eps
    shapes = [res['params'][0] for res in gamma_results]
    scales = [res['params'][2] for res in gamma_results]
    shape_variance = np.var(shapes)
    scale_variance = np.var(scales)
    print(f"\nVariance of Gamma shape parameters across eps: {shape_variance:.5f}")
    print(f"Variance of Gamma scale parameters across eps: {scale_variance:.5f}")

    # Optionally, plot the parameters
    eps_list = [res['eps'] for res in gamma_results]
    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, shapes, label='Shape Parameter')
    plt.plot(eps_list, scales, label='Scale Parameter')
    plt.xlabel('Eps')
    plt.ylabel('Parameter Value')
    plt.title('Gamma Parameters Across Eps')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
