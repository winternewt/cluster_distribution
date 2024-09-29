import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma, lognorm, weibull_min, betaprime
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(eps_value, data_dir, N, radius):
    # Load data for a specific eps value
    lambda0 = N / (np.pi * radius ** 2)  # Initial point density
    data_file = os.path.join(data_dir, f'simulation_data_N{N}_radius{radius}_eps{eps_value:.2f}.csv')
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found for eps = {eps_value:.2f}.")
        return None, None
    # Load the data
    df = pd.read_csv(data_file)
    # Filter valid clusters
    df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)].copy()
    if df_valid.empty:
        print(f"No valid clusters found for eps = {eps_value:.2f}.")
        return None, None
    # Compute lambda_prime and ratio
    df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
    df_valid['ratio'] = df_valid['lambda_prime'] / lambda0
    # Filter out ratios less than 5
    df_valid = df_valid[df_valid['ratio'] >= 5]
    if df_valid.empty:
        print(f"No data with ratio >= 5 for eps = {eps_value:.2f}.")
        return None, None
    ratio_data = df_valid['ratio'].values
    return df_valid, ratio_data

def fit_distributions(ratio_sample, distributions, floc_value):
    params = {}
    ks_results = {}
    aic_values = {}
    bic_values = {}

    # Fit each distribution
    for name, distribution in distributions.items():
        print(f"Fitting {name} distribution with floc={floc_value}...")
        # Estimate parameters with specified floc
        try:
            if distribution == lognorm:
                # For lognorm, we need to pass data > 0
                shape, loc, scale = distribution.fit(ratio_sample, floc=floc_value)
                params[name] = (shape, loc, scale)
            else:
                params[name] = distribution.fit(ratio_sample, floc=floc_value)
            # Perform K-S test
            ks_stat, ks_p = stats.kstest(ratio_sample, distribution.cdf, args=params[name])
            ks_results[name] = {'Statistic': ks_stat, 'p-value': ks_p}
            # Compute AIC and BIC
            log_likelihood = np.sum(np.log(distribution.pdf(ratio_sample, *params[name])))
            k = len(params[name])  # Number of parameters
            n = len(ratio_sample)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            aic_values[name] = aic
            bic_values[name] = bic
            print(f"{name} distribution fitted.")
        except Exception as e:
            print(f"Error fitting {name} distribution: {e}")
            params[name] = None
            ks_results[name] = {'Statistic': np.nan, 'p-value': np.nan}
            aic_values[name] = np.inf
            bic_values[name] = np.inf

    # Collect all results
    results = {
        'params': params,
        'ks_results': ks_results,
        'aic_values': aic_values,
        'bic_values': bic_values
    }
    return results

def plot_fitted_distributions(ratio_sample, results, distributions, eps_value, floc_value):
    plt.figure(figsize=(12, 8))
    # Plot histogram
    plt.hist(ratio_sample, bins=100, density=True, alpha=0.6, color='g', label='Data')
    x = np.linspace(min(ratio_sample), max(ratio_sample), 1000)
    # Plot fitted distributions
    for name, params in results['params'].items():
        if params is not None:
            dist = distributions[name]
            y = dist.pdf(x, *params)
            plt.plot(x, y, label=f'{name} (floc={floc_value})')
    plt.xlabel('Density Ratio (lambda\' / lambda0)')
    plt.ylabel('Density')
    plt.title(f'Fitted Distributions to Density Ratio Data for eps = {eps_value:.2f}')
    plt.legend()
    plt.show()

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area

    # Eps values to include in the analysis (from 1.10 to 1.40 in steps of 0.05)
    eps_values = np.arange(1.10, 1.41, 0.05).round(2)
    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Define distributions to fit
    distributions = {
        'Gamma': gamma,
        'Log-Normal': lognorm,
        'Weibull': weibull_min,
        'Beta-Prime': betaprime
        # EMG distribution is not directly available in scipy.stats; we'll handle it separately
    }

    # Loop over each eps value
    for eps_value in eps_values:
        print(f"\nProcessing eps = {eps_value:.2f}")
        df_valid, ratio_data = load_data(eps_value, data_dir, N, radius)
        if ratio_data is None:
            continue

        # For computational efficiency, sample the data
        sample_size = min(100000, len(ratio_data))  # Use up to 100,000 data points
        np.random.seed(42)
        ratio_sample = np.random.choice(ratio_data, size=sample_size, replace=False)

        # Fit distributions with floc=0 and floc=5
        for floc_value in [0, 5]:
            results = fit_distributions(ratio_sample, distributions, floc_value)

            # Plot the fitted distributions
            plot_fitted_distributions(ratio_sample, results, distributions, eps_value, floc_value)

            # Summarize results
            print(f"\nGoodness-of-Fit Test Results for eps = {eps_value:.2f} with floc = {floc_value}:")
            print("Kolmogorov-Smirnov Test:")
            for name, res in results['ks_results'].items():
                if not np.isnan(res['Statistic']):
                    print(f"{name}: Statistic = {res['Statistic']:.5f}, p-value = {res['p-value']:.5f}")
                else:
                    print(f"{name}: Could not compute K-S test.")

            print("\nAIC and BIC Values:")
            for name in results['params']:
                aic = results['aic_values'][name]
                bic = results['bic_values'][name]
                if not np.isinf(aic):
                    print(f"{name}: AIC = {aic:.2f}, BIC = {bic:.2f}")
                else:
                    print(f"{name}: AIC and BIC not available.")

            # Determine the best fit based on AIC
            best_fit_aic = min(results['aic_values'], key=results['aic_values'].get)
            print(f"\nBest fit based on AIC: {best_fit_aic}")

            # Determine the best fit based on BIC
            best_fit_bic = min(results['bic_values'], key=results['bic_values'].get)
            print(f"Best fit based on BIC: {best_fit_bic}")

            # Determine the best fit based on K-S test p-value
            ks_p_values = {k: v['p-value'] for k, v in results['ks_results'].items() if not np.isnan(v['p-value'])}
            if ks_p_values:
                best_fit_ks = max(ks_p_values, key=ks_p_values.get)
                print(f"Best fit based on K-S test p-value: {best_fit_ks}")
            else:
                print("No valid K-S test results to determine best fit.")

            # If EMG distribution is to be included, handle it separately
            # Since EMG is not available in scipy.stats, we need to define it or use an alternative method
            # For simplicity, EMG fitting is omitted here but can be added with custom implementation

if __name__ == "__main__":
    main()
