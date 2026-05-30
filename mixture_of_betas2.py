import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import argparse
from modules.simv2_data import load_data, sample_data
from modules.common_stats import compute_aic_bic, compute_ks_statistic
from modules.beta_stats import negative_log_likelihood_eps, betaprime_cdf_eps
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Modeling Merged Data from Multiple Epsilons')
    parser.add_argument('--data_dir', type=str, default='./simdata/v2/', help='Directory where data files are stored.')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Directory to save output files.')
    parser.add_argument('--eps_start', type=float, default=1.10, help='Starting value of eps.')
    parser.add_argument('--eps_end', type=float, default=1.40, help='Ending value of eps.')
    parser.add_argument('--eps_step', type=float, default=0.01, help='Step size for eps.')
    parser.add_argument('--N', type=int, default=10000, help='Number of points in the simulation.')
    parser.add_argument('--radius', type=float, default=100.0, help='Radius of the circular area.')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_step = args.eps_step
    N = args.N
    radius = args.radius

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define eps values
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step).round(2)

    # Initialize lists to store merged data
    merged_data = []
    merged_eps = []

    for eps in eps_values:
        print(f"Processing eps = {eps:.2f}")
        # Load data
        density_ratio = load_data(eps, data_dir, N, radius)
        if density_ratio is None:
            continue

        # Sample data
        ratio_sample = sample_data(density_ratio, sample_size=100000, random_seed=42)
        if ratio_sample is None:
            print(f"No data to sample for eps = {eps:.2f}. Skipping.")
            continue

        # Append to merged data
        merged_data.extend(ratio_sample)
        merged_eps.extend([eps] * len(ratio_sample))

    # Convert to numpy arrays
    merged_data = np.array(merged_data)
    merged_eps = np.array(merged_eps)

    # Initial regression parameters (from previous regression results or default values)
    initial_params = [
        50.0,    # a_slope
        -14.0,   # a_intercept
        3.5,     # b_slope
        5.8,     # b_intercept
        -11.5,   # loc_slope
        20.0,    # loc_intercept
        -4.6,    # scale_slope
        7.8      # scale_intercept
    ]

    # Bounds for parameters
    bounds = [
        (1e-6, None),  # a_slope
        (None, None),  # a_intercept
        (1e-6, None),  # b_slope
        (None, None),  # b_intercept
        (None, None),  # loc_slope
        (None, None),  # loc_intercept
        (1e-6, None),  # scale_slope
        (None, None)   # scale_intercept
    ]

    # Optimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood_eps,
        initial_params,
        args=(merged_data, merged_eps),
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        fitted_params = result.x
        print("\nOptimization successful.")
        print("Fitted regression parameters:")
        param_names = [
            'a_slope', 'a_intercept', 'b_slope', 'b_intercept',
            'loc_slope', 'loc_intercept', 'scale_slope', 'scale_intercept'
        ]
        params_dict = dict(zip(param_names, fitted_params))
        for name, value in params_dict.items():
            print(f"{name}: {value:.4f}")

        # Compute AIC and BIC
        nll = negative_log_likelihood_eps(fitted_params, merged_data, merged_eps)
        k = len(fitted_params)
        n = len(merged_data)
        aic = 2 * k + 2 * nll
        bic = k * np.log(n) + 2 * nll
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")

        # Compute KS statistic
        model_cdf_values = betaprime_cdf_eps(merged_data, merged_eps, params_dict)

        # Sort the data and model CDF values
        sorted_indices = np.argsort(merged_data)
        sorted_data = merged_data[sorted_indices]
        sorted_model_cdf = model_cdf_values[sorted_indices]

        # Compute the empirical CDF
        n = len(sorted_data)
        empirical_cdf = np.arange(1, n + 1) / n

        # Compute the KS statistic
        ks_statistic = np.max(np.abs(empirical_cdf - sorted_model_cdf))
        print(f"KS Statistic: {ks_statistic:.4f}")

        # Save the fitted parameters
        fitted_params_df = pd.DataFrame({
            'parameter': param_names,
            'value': fitted_params
        })
        fitted_params_df.to_csv(os.path.join(output_dir, 'merged_fit_params.csv'), index=False)

        # Optionally, plot the fit
        # (Plotting code remains the same)
    else:
        print("Optimization failed:", result.message)



if __name__ == "__main__":
    main()
