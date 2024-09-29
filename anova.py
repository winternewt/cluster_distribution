import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from scipy.stats import f_oneway
from scipy.stats import linregress


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
        if df_valid.empty or len(df_valid) < 2000:
            print(f"Not enough valid clusters for eps = {eps:.2f}. Skipping.")
            continue
        # Compute lambda_prime
        df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
        data_dict[eps] = df_valid
        print(f"Data loaded for eps = {eps:.2f}, total clusters: {len(df_valid)}")
    return data_dict

def fit_gamma_all(data):
    # Fit a Gamma distribution to all lambda_prime data
    lambda_prime = data['lambda_prime'].values
    params = stats.gamma.fit(lambda_prime, floc=0)  # Fix location to zero
    # Goodness-of-fit
    D, p_value = stats.kstest(lambda_prime, 'gamma', args=params)
    log_likelihood = np.sum(np.log(stats.gamma.pdf(lambda_prime, *params)))
    k = 2  # Number of parameters: shape and scale
    n = len(lambda_prime)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return params, D, p_value, aic, bic

def fit_gamma_N10(data):
    # Fit a Gamma distribution to lambda_prime data where N_prime = 10
    data_N10 = data[data['N_prime'] == 10]
    if len(data_N10) < 2000:
        print("Not enough data for N_prime = 10")
        return None, None, None, None, None
    lambda_prime = data_N10['lambda_prime'].values
    params = stats.gamma.fit(lambda_prime, floc=0)
    # Goodness-of-fit
    D, p_value = stats.kstest(lambda_prime, 'gamma', args=params)
    log_likelihood = np.sum(np.log(stats.gamma.pdf(lambda_prime, *params)))
    k = 2
    n = len(lambda_prime)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return params, D, p_value, aic, bic

def fit_gamma_mixture(data):
    # Fit a mixture of Gamma distributions for each N_prime
    N_prime_values = data['N_prime'].unique()
    total_clusters = len(data)
    mixture_weights = []
    gamma_params = []
    for N in N_prime_values:
        subset = data[data['N_prime'] == N]
        lambda_prime = subset['lambda_prime'].values
        if len(lambda_prime) < 5:
            continue
        params = stats.gamma.fit(lambda_prime, floc=0)
        gamma_params.append((N, params))
        weight = len(lambda_prime) / total_clusters
        mixture_weights.append((N, weight))
    # Goodness-of-fit
    lambda_prime_all = data['lambda_prime'].values
    # Compute mixture PDF
    def mixture_pdf(x):
        pdf = np.zeros_like(x)
        for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
            pdf += weight * stats.gamma.pdf(x, *params)
        return pdf
    # Compute mixture CDF for KS test
    def mixture_cdf(x):
        cdf = 0.0
        for (N, params), (N_w, weight) in zip(gamma_params, mixture_weights):
            cdf += weight * stats.gamma.cdf(x, *params)
        return cdf
    D, p_value = stats.kstest(lambda_prime_all, mixture_cdf)
    # Log-likelihood
    log_likelihood = np.sum(np.log(mixture_pdf(lambda_prime_all)))
    k = len(gamma_params) * 2  # Each Gamma has shape and scale parameters
    n = len(lambda_prime_all)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return gamma_params, mixture_weights, D, p_value, aic, bic

def perform_anova(parameter_list, eps_list):
    # Perform ANOVA on the parameter list across eps
    # parameter_list is a list of parameter values for each eps
    # eps_list is the corresponding list of eps values
    # Group parameters by eps
    unique_eps = sorted(set(eps_list))
    groups = []
    for eps in unique_eps:
        indices = [i for i, e in enumerate(eps_list) if e == eps]
        params = [parameter_list[i] for i in indices]
        groups.append(params)
    # Perform ANOVA
    F_statistic, p_value = f_oneway(*groups)
    return F_statistic, p_value

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    lambda0 = N / (np.pi * radius**2)  # Initial point density

    # Eps values to include in the analysis
    eps_values = np.arange(1.10, 1.41, 0.01).round(2)
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Load data
    data_dict = load_data(eps_values, data_dir, N, radius)

    # Initialize lists to store results
    eps_list = []
    method_a_params = []
    method_b_params = []
    method_c_params = []
    method_a_stats = []
    method_b_stats = []
    method_c_stats = []

    for eps in eps_values:
        if eps not in data_dict:
            continue
        data = data_dict[eps]
        eps_list.append(eps)

        # Method a): Fit Gamma to all data
        params_a, D_a, p_a, aic_a, bic_a = fit_gamma_all(data)
        method_a_params.append(params_a)
        method_a_stats.append({'D': D_a, 'p_value': p_a, 'AIC': aic_a, 'BIC': bic_a})
        print(f"Method a) eps={eps:.2f}: KS Stat={D_a:.5f}, p-value={p_a:.5f}, AIC={aic_a:.2f}, BIC={bic_a:.2f}")

        # Method b): Fit mixture of Gammas by N_prime
        gamma_params_b, mixture_weights_b, D_b, p_b, aic_b, bic_b = fit_gamma_mixture(data)
        method_b_params.append((gamma_params_b, mixture_weights_b))
        method_b_stats.append({'D': D_b, 'p_value': p_b, 'AIC': aic_b, 'BIC': bic_b})
        print(f"Method b) eps={eps:.2f}: KS Stat={D_b:.5f}, p-value={p_b:.5f}, AIC={aic_b:.2f}, BIC={bic_b:.2f}")

        # Method c): Fit Gamma to data where N_prime = 10
        params_c, D_c, p_c, aic_c, bic_c = fit_gamma_N10(data)
        if params_c is not None:
            method_c_params.append(params_c)
            method_c_stats.append({'D': D_c, 'p_value': p_c, 'AIC': aic_c, 'BIC': bic_c})
            print(f"Method c) eps={eps:.2f}: KS Stat={D_c:.5f}, p-value={p_c:.5f}, AIC={aic_c:.2f}, BIC={bic_c:.2f}")
        else:
            method_c_params.append((np.nan, np.nan, np.nan))
            method_c_stats.append({'D': np.nan, 'p_value': np.nan, 'AIC': np.nan, 'BIC': np.nan})
            print(f"Method c) eps={eps:.2f}: Not enough data for N_prime = 10")

    # Convert parameter lists to arrays for ANOVA
    shape_params_a = [params[0] for params in method_a_params]
    scale_params_a = [params[2] for params in method_a_params]

    shape_params_c = [params[0] for params in method_c_params if not np.isnan(params[0])]
    scale_params_c = [params[2] for params in method_c_params if not np.isnan(params[2])]
    eps_list_c = [eps_list[i] for i, params in enumerate(method_c_params) if not np.isnan(params[0])]

    # Perform ANOVA on shape and scale parameters for Method a)
    F_shape_a, p_shape_a = perform_anova(shape_params_a, eps_list)
    F_scale_a, p_scale_a = perform_anova(scale_params_a, eps_list)
    print(f"\nANOVA for Method a) Shape Parameter: F={F_shape_a:.5f}, p-value={p_shape_a:.5f}")
    print(f"ANOVA for Method a) Scale Parameter: F={F_scale_a:.5f}, p-value={p_scale_a:.5f}")

    eps_array = np.array(eps_list)
    shape_params_a = np.array([params[0] for params in method_a_params])
    scale_params_a = np.array([params[2] for params in method_a_params])

    slope, intercept, r_value, p_value, std_err = linregress(eps_array, scale_params_a)
    print(f"Method A Scale Parameter Regression: slope={slope:.5f}, p-value={p_value:.5f}")
    slope, intercept, r_value, p_value, std_err = linregress(eps_array, shape_params_a)
    print(f"Method A Shape Parameter Regression: slope={slope:.5f}, p-value={p_value:.5f}")

    # Perform ANOVA on shape and scale parameters for Method c)
    F_shape_c, p_shape_c = perform_anova(shape_params_c, eps_list_c)
    F_scale_c, p_scale_c = perform_anova(scale_params_c, eps_list_c)
    print(f"\nANOVA for Method c) Shape Parameter: F={F_shape_c:.5f}, p-value={p_shape_c:.5f}")
    print(f"ANOVA for Method c) Scale Parameter: F={F_scale_c:.5f}, p-value={p_scale_c:.5f}")

    eps_array_c = np.array(eps_list_c)
    shape_params_c = np.array([params[0] for params in method_c_params if not np.isnan(params[0])])
    scale_params_c = np.array([params[2] for params in method_c_params if not np.isnan(params[2])])

    slope, intercept, r_value, p_value, std_err = linregress(eps_array_c, scale_params_c)
    print(f"Method C Scale Parameter Regression: slope={slope:.5f}, p-value={p_value:.5f}")
    slope, intercept, r_value, p_value, std_err = linregress(eps_array_c, shape_params_c)
    print(f"Method C Shape Parameter Regression: slope={slope:.5f}, p-value={p_value:.5f}")

    # Note: For Method b), since parameters are per N_prime, ANOVA across eps is not directly applicable

    # Optionally, plot the parameters
    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, shape_params_a, label='Method a) Shape Parameter')
    plt.xlabel('Eps')
    plt.ylabel('Shape Parameter')
    plt.title('Gamma Shape Parameter Across Eps (Method a)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, scale_params_a, label='Method a) Scale Parameter')
    plt.xlabel('Eps')
    plt.ylabel('Scale Parameter')
    plt.title('Gamma Scale Parameter Across Eps (Method a)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
