import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import argparse
from modules.simv2_data import load_data, sample_data
from modules.beta_stats import betaprime_mixture_pdf, negative_log_likelihood
from modules.common_stats import compute_aic_bic, compute_ks_statistic
from scipy.optimize import minimize

def merge_data(eps_values, data_dir, N, radius):
    """
    Merge data from multiple eps values.
    """
    merged_density_ratio = []
    eps_list = []
    for eps in eps_values:
        print(f"Loading data for eps = {eps:.2f}")
        density_ratio = load_data(eps, data_dir, N, radius)
        if density_ratio is not None:
            merged_density_ratio.extend(density_ratio)
            eps_list.extend([eps] * len(density_ratio))
    return np.array(merged_density_ratio), np.array(eps_list)

def fit_mixture_model(data, eps_list, regression_params):
    """
    Fit a mixture of Beta-Prime distributions to the merged data.
    """
    unique_eps = np.unique(eps_list)
    num_components = len(unique_eps)
    print(f"Number of components in the mixture: {num_components}")

    # Generate initial parameters for each component based on regression
    initial_params = []
    bounds = []
    for eps in unique_eps:
        # Get initial guesses from regression
        a_slope, a_intercept = regression_params['a']
        b_slope, b_intercept = regression_params['b']
        loc_slope, loc_intercept = regression_params['loc']
        scale_slope, scale_intercept = regression_params['scale']

        a_guess = a_slope * eps + a_intercept
        b_guess = b_slope * eps + b_intercept
        loc_guess = loc_slope * eps + loc_intercept
        scale_guess = scale_slope * eps + scale_intercept

        # Ensure initial guesses are within valid ranges
        a_guess = max(a_guess, 1e-6)
        b_guess = max(b_guess, 1e-6)
        scale_guess = max(scale_guess, 1e-6)

        # Append parameters for this component
        initial_params.extend([a_guess, b_guess, loc_guess, scale_guess])

        # Bounds for this component
        bounds.extend([
            (1e-6, None),  # a > 0
            (1e-6, None),  # b > 0
            (None, None),  # loc
            (1e-6, None)   # scale > 0
        ])

    # Mixing proportions (weights)
    # Initialize with equal weights
    weights = [1.0 / num_components] * num_components
    initial_params.extend(weights)
    bounds.extend([(0, 1)] * num_components)

    # Constraints: weights sum to 1
    constraints = {
        'type': 'eq',
        'fun': lambda params: np.sum(params[-num_components:]) - 1
    }   # Define the negative log-likelihood function for the mixture
    def mixture_negative_log_likelihood(params):
        num_params_per_component = 4  # a, b, loc, scale
        total_pdf = np.zeros_like(data)
        for i in range(num_components):
            idx = i * num_params_per_component
            a = params[idx]
            b = params[idx + 1]
            loc = params[idx + 2]
            scale = params[idx + 3]
            weight = params[-num_components + i]
            pdf = stats.betaprime.pdf(data, a, b, loc=loc, scale=scale)
            total_pdf += weight * pdf
        total_pdf = np.where(total_pdf == 0, 1e-10, total_pdf)
        return -np.sum(np.log(total_pdf))

    # Perform optimization
    result = minimize(
        mixture_negative_log_likelihood,
        initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        fitted_params = result.x
        return fitted_params, unique_eps, True
    else:
        print("Optimization failed:", result.message)
        return None, None, False

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Model Merged Data Using Beta-Prime Mixture')
    parser.add_argument('--data_dir', type=str, default='./simdata/v2/', help='Directory where data files are stored.')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Directory to save output files.')
    parser.add_argument('--eps_values', type=float, nargs='+', default=[1.10, 1.20, 1.30, 1.40], help='List of eps values to merge.')
    parser.add_argument('--N', type=int, default=10000, help='Number of points in the simulation.')
    parser.add_argument('--radius', type=float, default=100.0, help='Radius of the circular area.')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    eps_values = args.eps_values
    N = args.N
    radius = args.radius

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Try to read regression parameters from file
    regression_params_file = os.path.join(output_dir, 'regression_params.csv')
    if os.path.exists(regression_params_file):
        regression_params_df = pd.read_csv(regression_params_file)
        regression_params = {
            'a': (regression_params_df.loc[regression_params_df['parameter'] == 'a', 'slope'].values[0],
                  regression_params_df.loc[regression_params_df['parameter'] == 'a', 'intercept'].values[0]),
            'b': (regression_params_df.loc[regression_params_df['parameter'] == 'b', 'slope'].values[0],
                  regression_params_df.loc[regression_params_df['parameter'] == 'b', 'intercept'].values[0]),
            'loc': (regression_params_df.loc[regression_params_df['parameter'] == 'loc', 'slope'].values[0],
                    regression_params_df.loc[regression_params_df['parameter'] == 'loc', 'intercept'].values[0]),
            'scale': (regression_params_df.loc[regression_params_df['parameter'] == 'scale', 'slope'].values[0],
                      regression_params_df.loc[regression_params_df['parameter'] == 'scale', 'intercept'].values[0]),
        }
        print("Regression parameters loaded from file.")
    else:
        print("Regression parameters file not found. Please ensure it exists.")
        return

    # Merge data from specified eps values
    merged_data, eps_list = merge_data(eps_values, data_dir, N, radius)
    if len(merged_data) == 0:
        print("No data available after merging.")
        return

    # Sample data for computational efficiency
    sample_size = min(500000, len(merged_data))
    np.random.seed(42)
    sample_indices = np.random.choice(len(merged_data), size=sample_size, replace=False)
    merged_sample = merged_data[sample_indices]
    eps_sample = eps_list[sample_indices]

    # Fit mixture model to the merged data
    fitted_params, unique_eps, success = fit_mixture_model(merged_sample, eps_sample, regression_params)
    if success:
        num_components = len(unique_eps)
        num_params_per_component = 4  # a, b, loc, scale

        # Extract fitted parameters for each component
        components = []
        for i in range(num_components):
            idx = i * num_params_per_component
            a = fitted_params[idx]
            b = fitted_params[idx + 1]
            loc = fitted_params[idx + 2]
            scale = fitted_params[idx + 3]
            weight = fitted_params[-num_components + i]
            components.append({
                'eps': unique_eps[i],
                'a': a,
                'b': b,
                'loc': loc,
                'scale': scale,
                'weight': weight
            })

        # Compute goodness-of-fit statistics
        log_likelihood = -negative_log_likelihood(fitted_params)
        k = len(fitted_params)
        n = len(merged_sample)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Compute KS statistic
        def mixture_cdf(x):
            total_cdf = np.zeros_like(x)
            for i in range(num_components):
                idx = i * num_params_per_component
                a = fitted_params[idx]
                b = fitted_params[idx + 1]
                loc = fitted_params[idx + 2]
                scale = fitted_params[idx + 3]
                weight = fitted_params[-num_components + i]
                cdf = stats.betaprime.cdf(x, a, b, loc=loc, scale=scale)
                total_cdf += weight * cdf
            return total_cdf

        ks_stat, ks_pval = stats.kstest(merged_sample, mixture_cdf)

        # Output results
        print("\nMixture Model Fit Results:")
        for comp in components:
            print(f"Component for eps = {comp['eps']:.2f}:")
            print(f"  a = {comp['a']:.4f}, b = {comp['b']:.4f}, loc = {comp['loc']:.4f}, scale = {comp['scale']:.4f}, weight = {comp['weight']:.4f}")
        print(f"\nGoodness-of-Fit Statistics:")
        print(f"  Log-Likelihood: {log_likelihood:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  KS Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")

        # Save results to CSV
        mixture_results_df = pd.DataFrame(components)
        mixture_results_df.to_csv(os.path.join(output_dir, 'merged_mixture_fit.csv'), index=False)
    else:
        print("Mixture model fitting failed.")

if __name__ == "__main__":
    main()
