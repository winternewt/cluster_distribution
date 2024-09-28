import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gamma, expon, lognorm, weibull_min
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def merge_data(eps_values, data_dir, N, radius):
    # Initialize an empty DataFrame to store combined data
    combined_df = pd.DataFrame()

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
        # Append to combined DataFrame
        combined_df = pd.concat([combined_df, df_valid], ignore_index=True)
        print(f"Merged data for eps = {eps:.2f}, total clusters so far: {len(combined_df)}")
    return combined_df


def fit_distributions(data, lambda0, distributions):
    # Compute lambda_prime and ratio
    data['lambda_prime'] = data['N_prime'] / data['S_prime']
    data['ratio'] = data['lambda_prime'] / lambda0
    ratio_data = data['ratio'].values

    params = {}
    ks_results = {}
    ad_results = {}
    aic_values = {}
    bic_values = {}

    # For computational efficiency, we might sample the data
    sample_size = min(100000, len(ratio_data))  # Use up to 100,000 data points
    np.random.seed(42)
    ratio_sample = np.random.choice(ratio_data, size=sample_size, replace=False)

    # Fit each distribution
    for name, distribution in distributions.items():
        print(f"Fitting {name} distribution...")
        # Estimate parameters
        if distribution == lognorm:
            # For lognorm, we need to pass data > 0
            shape, loc, scale = distribution.fit(ratio_sample, floc=0)
            params[name] = (shape, loc, scale)
        else:
            params[name] = distribution.fit(ratio_sample)
        # Perform K-S test
        ks_stat, ks_p = stats.kstest(ratio_sample, distribution.cdf, args=params[name])
        ks_results[name] = {'Statistic': ks_stat, 'p-value': ks_p}

        # Perform Anderson-Darling test only if supported
        supported_dists = ['norm', 'expon', 'logistic', 'gumbel_l', 'gumbel_r', 'extreme1', 'weibull_min']
        if distribution.name in supported_dists:
            ad_stat = stats.anderson(ratio_sample, dist=distribution.name).statistic
            ad_results[name] = ad_stat
        else:
            ad_results[name] = np.nan  # Indicate that the test was not performed

        # Compute AIC and BIC
        log_likelihood = np.sum(np.log(distribution.pdf(ratio_sample, *params[name])))
        k = len(params[name])  # Number of parameters
        n = len(ratio_sample)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        aic_values[name] = aic
        bic_values[name] = bic
        print(f"{name} distribution fitted.")

    # Collect all results
    results = {
        'params': params,
        'ks_results': ks_results,
        'ad_results': ad_results,
        'aic_values': aic_values,
        'bic_values': bic_values
    }
    return results, ratio_sample


def plot_fitted_distributions(ratio_sample, results, distributions):
    plt.figure(figsize=(12, 8))
    # Plot histogram
    sns.histplot(ratio_sample, bins=100, kde=False, stat='density', label='Data', color='skyblue')
    x = np.linspace(min(ratio_sample), max(ratio_sample), 1000)
    # Plot fitted distributions
    for name in results['params']:
        dist = distributions[name]
        y = dist.pdf(x, *results['params'][name])
        plt.plot(x, y, label=name)
    plt.xlabel('Density Ratio (lambda\' / lambda0)')
    plt.ylabel('Density')
    plt.title('Fitted Distributions to Density Ratio Data')
    plt.legend()
    plt.show()


def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    lambda0 = N / (np.pi * radius ** 2)  # Initial point density

    # Eps values to include in the analysis
    eps_values = np.arange(0.90, 1.41, 0.01).round(2)
    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Merge data
    combined_data = merge_data(eps_values, data_dir, N, radius)
    print(f"Total clusters after merging: {len(combined_data)}")

    # Define distributions to fit
    distributions = {
        'Gamma': gamma,
        'Exponential': expon,
        'Log-Normal': lognorm,
        'Weibull': weibull_min
    }

    # Fit distributions to the ratio data
    results, ratio_sample = fit_distributions(combined_data, lambda0, distributions)

    # Plot the fitted distributions
    plot_fitted_distributions(ratio_sample, results, distributions)

    # Summarize results
    print("\nGoodness-of-Fit Test Results:")
    print("Kolmogorov-Smirnov Test:")
    for name, res in results['ks_results'].items():
        print(f"{name}: Statistic = {res['Statistic']:.5f}, p-value = {res['p-value']:.5f}")

    print("\nAnderson-Darling Test Statistics:")
    for name, stat in results['ad_results'].items():
        if not np.isnan(stat):
            print(f"{name}: Statistic = {stat:.5f}")
        else:
            print(f"{name}: Anderson-Darling test not performed")

    print("\nAIC and BIC Values:")
    for name in results['params']:
        print(f"{name}: AIC = {results['aic_values'][name]:.2f}, BIC = {results['bic_values'][name]:.2f}")

    # Determine the best fit based on AIC
    best_fit_aic = min(results['aic_values'], key=results['aic_values'].get)
    print(f"\nBest fit based on AIC: {best_fit_aic}")

    # Determine the best fit based on BIC
    best_fit_bic = min(results['bic_values'], key=results['bic_values'].get)
    print(f"Best fit based on BIC: {best_fit_bic}")

    # Determine the best fit based on K-S test p-value
    best_fit_ks = max(results['ks_results'], key=lambda k: results['ks_results'][k]['p-value'])
    print(f"Best fit based on K-S test p-value: {best_fit_ks}")

    # Determine the best fit based on Anderson-Darling test statistic (if available)
    ad_stats = {name: stat for name, stat in results['ad_results'].items() if not np.isnan(stat)}
    if ad_stats:
        best_fit_ad = min(ad_stats, key=ad_stats.get)
        print(f"Best fit based on Anderson-Darling test statistic: {best_fit_ad}")
    else:
        print("No Anderson-Darling test results available to determine best fit.")


if __name__ == "__main__":
    main()
