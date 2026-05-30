import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import argparse
from modules.simv2_data import load_data, sample_data
from modules.beta_stats import betaprime_mixture_pdf, betaprime_mixture_cdf, negative_log_likelihood, perform_linear_regression
from modules.common_stats import compute_aic_bic, compute_ks_statistic
from scipy.optimize import minimize


def fit_betaprime_mixture(data):
    """
    Fit a mixture of two Beta-Prime distributions to the data.

    Parameters:
    - data: Array of density_ratio values.

    Returns:
    - params: Fitted parameters for the mixture model.
    - success: Boolean indicating if the fitting was successful.
    """
    # Initial guesses for parameters
    a1, b1, loc1, scale1 = 2, 2, np.min(data), 1
    a2, b2, loc2, scale2 = 2, 2, np.min(data), 1
    alpha = 0.5  # Mixing proportion

    initial_params = [a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha]

    # Bounds to ensure parameters are positive and mixing proportion between 0 and 1
    bounds = [(1e-3, None), (1e-3, None), (0, None), (1e-3, None),
              (1e-3, None), (1e-3, None), (0, None), (1e-3, None),
              (0, 1)]

    try:
        result = minimize(negative_log_likelihood, initial_params, args=(data,),
                          method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x, True
        else:
            print(f"Mixture fit failed: {result.message}")
            return None, False
    except Exception as e:
        print(f"Mixture fit exception: {e}")
        return None, False

def fit_betaprime_distribution(data, eps, regression_params):
    """
    Fit the Beta-Prime distribution to the data, estimating all parameters including floc.

    Parameters:
    - data: Array of density_ratio values.

    Returns:
    - params: Fitted parameters for the Beta-Prime distribution.
    - success: Boolean indicating if the fitting was successful.
    """

    # Extract regression coefficients
    a_slope, a_intercept = regression_params['a']
    b_slope, b_intercept = regression_params['b']
    loc_slope, loc_intercept = regression_params['loc']
    scale_slope, scale_intercept = regression_params['scale']

    # Compute initial guesses based on eps
    a_guess = a_slope * eps + a_intercept
    b_guess = b_slope * eps + b_intercept
    loc_guess = loc_slope * eps + loc_intercept
    scale_guess = scale_slope * eps + scale_intercept

    # Ensure initial guesses are within valid ranges
    a_guess = max(a_guess, 1e-6)
    b_guess = max(b_guess, 1e-6)
    scale_guess = max(scale_guess, 1e-6)


    # Initial parameter vector
    initial_params = [a_guess, b_guess, loc_guess, scale_guess]

    # Bounds: a > 0, b > 0, scale > 0
    bounds = [(1e-6, None), (1e-6, None), (None, None), (1e-6, None)]

    # Perform optimization
    result = minimize(negative_log_likelihood, initial_params, args=(data,),
                      bounds=bounds)

    if result.success:
        fitted_params = result.x
        return fitted_params, True
    else:
        print("Optimization failed:", result.message)
        return None, False

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Statistics Computation for Beta-Prime Fits')
    parser.add_argument('--data_dir', type=str, default='./simdata/v2/', help='Directory where data files are stored. Default: ./simdata/v2/')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Directory to save output CSV files. Default: ./results/')
    parser.add_argument('--eps_start', type=float, default=1.10, help='Starting value of eps. Default: 1.10')
    parser.add_argument('--eps_end', type=float, default=1.40, help='Ending value of eps. Default: 1.40')
    parser.add_argument('--eps_step', type=float, default=0.01, help='Step size for eps. Default: 0.05')
    parser.add_argument('--iterations', type=int, default=1, help='Number of subsampling iterations. Default: 1')
    parser.add_argument('--N', type=int, default=10000, help='Number of points in the simulation. Default: 10000')
    parser.add_argument('--radius', type=float, default=100.0, help='Radius of the circular area. Default: 100.0')
    parser.add_argument('--mix', action='store_true', help='Include mixture fits.')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_step = args.eps_step
    iterations = args.iterations
    N = args.N
    radius = args.radius
    include_mix = args.mix

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define eps values based on start, end, and step
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step).round(2)

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
        # Initialize with default or hardcoded values
        regression_params = {
            'a': (50.0568, -14.1770),
            'b': (3.5925, 5.8322),
            'loc': (-11.5939, 20.1187),
            'scale': (-4.6409, 7.8420)
        }
        print("Using default regression parameters.")

    # Initialize lists to store fit results
    regression_data = {
        'eps': [],
        'a': [],
        'b': [],
        'loc': [],
        'scale': []
    }

    for eps in eps_values:
        print(f"\nProcessing eps = {eps:.2f}")
        # Load data
        density_ratio = load_data(eps, data_dir, N, radius)
        if density_ratio is None:
            continue

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")
            # Sample data
            ratio_sample = sample_data(density_ratio, sample_size=100000, random_seed=42 + iteration)
            if ratio_sample is None:
                print(f"No data to sample for eps = {eps:.2f}. Skipping.")
                continue

            # ---- Regular Fit with Initial Guesses ----
            regular_params, regular_success = fit_betaprime_distribution(ratio_sample, eps, regression_params)
            if regular_success:
                a, b, fitted_loc, fitted_scale = regular_params
                # Compute statistics
                aic_regular, bic_regular = compute_aic_bic(ratio_sample, stats.betaprime, regular_params)
                ks_regular, p_regular = compute_ks_statistic(ratio_sample, stats.betaprime, regular_params)
                print(f"\nRegular Fit for eps = {eps:.2f}: a={a:.4f}, b={b:.4f}, loc={fitted_loc:.4f}, scale={fitted_scale:.4f}")
                print(f"  AIC: {aic_regular:.2f}, BIC: {bic_regular:.2f}, KS Statistic: {ks_regular:.4f}, p-value: {p_regular:.4f}")

                # Save regular fit results
                regular_fit_df = pd.DataFrame([{
                    'eps': eps,
                    'iteration': iteration + 1,
                    'a': a,
                    'b': b,
                    'loc': fitted_loc,
                    'scale': fitted_scale,
                    'AIC': aic_regular,
                    'BIC': bic_regular,
                    'KS_stat': ks_regular,
                    'KS_pval': p_regular
                }])
                regular_fit_filename = f'regular_fit.csv'
                regular_fit_path = os.path.join(output_dir, regular_fit_filename)
                # Append to CSV
                regular_fit_df.to_csv(regular_fit_path, mode='a', index=False, header=not os.path.exists(regular_fit_path))

                # Collect parameters for regression
                regression_data['eps'].append(eps)
                regression_data['a'].append(a)
                regression_data['b'].append(b)
                regression_data['loc'].append(fitted_loc)
                regression_data['scale'].append(fitted_scale)
            else:
                print(f"Regular Fit failed for eps = {eps:.2f}")

            # ---- Mixture Fit (if selected) ----
            if include_mix:
                # Similar code for mixture fit, appending results to CSV

                mixture_params, mixture_success = fit_betaprime_mixture(ratio_sample)
                if mixture_success:
                    a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha = mixture_params
                    # Compute log-likelihood
                    pdf_mixture = betaprime_mixture_pdf(ratio_sample, mixture_params)
                    log_likelihood_mixture = np.sum(np.log(pdf_mixture + 1e-10))  # Add small value to avoid log(0)
                    k = len(mixture_params)  # Number of parameters
                    n = len(ratio_sample)
                    aic_mixture = 2 * k - 2 * log_likelihood_mixture
                    bic_mixture = k * np.log(n) - 2 * log_likelihood_mixture

                    # Compute KS statistic (use the mixture CDF, not the PDF)
                    cdf_fitted = lambda x_val: betaprime_mixture_cdf(x_val, mixture_params)
                    ks_mixture, p_mixture = stats.kstest(ratio_sample, cdf_fitted)

                    print(f"\nMixture Fit for eps = {eps:.2f}:")
                    print(f"  Component 1: a={a1:.4f}, b={b1:.4f}, loc={loc1:.4f}, scale={scale1:.4f}")
                    print(f"  Component 2: a={a2:.4f}, b={b2:.4f}, loc={loc2:.4f}, scale={scale2:.4f}")
                    print(f"  Mixing Proportion (alpha): {alpha:.4f}")
                    print(
                        f"  AIC: {aic_mixture:.2f}, BIC: {bic_mixture:.2f}, KS Statistic: {ks_mixture:.4f}, p-value: {p_mixture:.4f}")

                    # Save mixture fit results
                    mixture_fit_df = pd.DataFrame([{
                        'eps': eps,
                        'a1': a1,
                        'b1': b1,
                        'loc1': loc1,
                        'scale1': scale1,
                        'a2': a2,
                        'b2': b2,
                        'loc2': loc2,
                        'scale2': scale2,
                        'alpha': alpha,
                        'AIC': aic_mixture,
                        'BIC': bic_mixture,
                        'KS_stat': ks_mixture,
                        'KS_pval': p_mixture
                    }])
                    mixture_fit_filename = f'mixture_fit_eps{eps:.2f}.csv'
                    mixture_fit_path = os.path.join(output_dir, mixture_fit_filename)
                    mixture_fit_df.to_csv(mixture_fit_path, index=False)
                else:
                    print(f"Mixture Fit failed for eps = {eps:.2f}")

    # After all eps values and iterations, perform linear regression
    if len(regression_data['eps']) >= 2:
        print("\nPerforming Linear Regression on Regular Fit Parameters...")
        regression_params = perform_linear_regression(regression_data, output_dir, return_params=True)
    else:
        print("\nNot enough data points for linear regression.")

if __name__ == "__main__":
    main()