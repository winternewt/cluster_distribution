import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants for floc linear dependence on eps
SLOPE = -9.71
INTERCEPT = 17.1674

def load_data(eps_values, data_dir, N, radius):
    """
    Load simulation data for specified epsilon values.

    Parameters:
    - eps_values: List or array of epsilon values.
    - data_dir: Directory where data files are stored.
    - N: Number of points in the simulation.
    - radius: Radius of the circular area.

    Returns:
    - data_dict: Dictionary with epsilon as keys and corresponding DataFrames as values.
    """
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

def betaprime_mixture_pdf(x, params):
    """
    Compute the PDF of a mixture of two Beta-Prime distributions.

    Parameters:
    - x: Points at which to evaluate the PDF.
    - params: List containing parameters for the two Beta-Prime distributions and mixing proportion.
              params = [a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha]
              where alpha is the mixing proportion for the first component.

    Returns:
    - pdf: The computed mixture PDF evaluated at points x.
    """
    a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha = params
    pdf1 = stats.betaprime.pdf(x, a1, b1, loc=loc1, scale=scale1)
    pdf2 = stats.betaprime.pdf(x, a2, b2, loc=loc2, scale=scale2)
    return alpha * pdf1 + (1 - alpha) * pdf2

def negative_log_likelihood(params, data):
    """
    Compute the negative log-likelihood for the Beta-Prime mixture model.

    Parameters:
    - params: Parameters for the mixture model.
    - data: The density_ratio data.

    Returns:
    - nll: Negative log-likelihood value.
    """
    pdf = betaprime_mixture_pdf(data, params)
    # To avoid log(0), set a minimum value
    pdf = np.where(pdf == 0, 1e-10, pdf)
    return -np.sum(np.log(pdf))

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

def fit_regular_betaprime(data, floc):
    """
    Fit a single Beta-Prime distribution to the data.

    Parameters:
    - data: Array of density_ratio values.
    - floc: The location parameter based on eps.

    Returns:
    - params: Fitted parameters for the Beta-Prime distribution.
    - success: Boolean indicating if the fitting was successful.
    """
    try:
        params = stats.betaprime.fit(data, floc=floc)
        return params, True
    except Exception as e:
        print(f"Regular fit exception: {e}")
        return None, False

def compute_aic_bic(data, distribution, params):
    """
    Compute AIC and BIC for a given distribution fit.

    Parameters:
    - data: Array of density_ratio values.
    - distribution: Scipy.stats distribution object.
    - params: Parameters of the fitted distribution.

    Returns:
    - aic: Akaike Information Criterion.
    - bic: Bayesian Information Criterion.
    """
    log_likelihood = np.sum(distribution.logpdf(data, *params))
    k = len(params)  # Number of parameters
    n = len(data)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

def compute_ks_statistic(data, distribution, params):
    """
    Compute the Kolmogorov-Smirnov statistic for the fit.

    Parameters:
    - data: Array of density_ratio values.
    - distribution: Scipy.stats distribution object.
    - params: Parameters of the fitted distribution.

    Returns:
    - ks_stat: KS statistic.
    - p_value: p-value of the KS test.
    """
    cdf_fitted = lambda x: distribution.cdf(x, *params)
    ks_stat, p_value = stats.kstest(data, cdf_fitted)
    return ks_stat, p_value

def mixture_pdf_plot(x, params):
    """
    Compute the PDF of the Beta-Prime mixture for plotting.

    Parameters:
    - x: Points at which to evaluate the PDF.
    - params: Fitted parameters of the mixture model.

    Returns:
    - pdf: PDF values at x.
    """
    a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha = params
    pdf1 = stats.betaprime.pdf(x, a1, b1, loc=loc1, scale=scale1)
    pdf2 = stats.betaprime.pdf(x, a2, b2, loc=loc2, scale=scale2)
    return alpha * pdf1 + (1 - alpha) * pdf2

def plot_overlapping_histograms_with_fits(data_dict, eps_values):
    """
    Plot overlapping histograms of density ratios with fitted Beta-Prime models.

    Parameters:
    - data_dict: Dictionary with epsilon as keys and corresponding DataFrames as values.
    - eps_values: List or array of epsilon values.

    Returns:
    - regression_data: Dictionary containing parameters for regression.
    """
    plt.figure(figsize=(14, 10))

    # Prepare colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_values)))

    # Compute floc for each eps to determine x_min
    floc_values = SLOPE * eps_values + INTERCEPT
    min_floc = np.min(floc_values)
    print(f"Minimum floc across all eps: {min_floc:.4f}")

    # X-axis for PDF plotting
    # Determine the combined data range
    all_density_ratios = np.concatenate([data['density_ratio'].values for data in data_dict.values()])
    x_min = min_floc
    x_max = np.percentile(all_density_ratios, 99.99)  # Use 99.99th percentile to avoid extreme tails
    x = np.linspace(x_min, x_max, 1000)

    # Containers for regression
    regression_data = {
        'eps': [],
        'a': [],
        'b': [],
        'loc': [],
        'scale': []
    }

    # Containers for legend handles
    legend_handles = []

    # Plot histograms and fitted curves
    for idx, eps in enumerate(eps_values):
        if eps not in data_dict:
            continue
        data = data_dict[eps]
        density_ratio = data['density_ratio'].values

        # Plot histogram
        plt.hist(density_ratio, bins=100, range=(x_min, x_max), density=True, histtype='stepfilled',
                 alpha=0.3, color=colors[idx], edgecolor='none')

        # Regular Fit
        floc = SLOPE * eps + INTERCEPT  # floc depends linearly on eps
        regular_params, regular_success = fit_regular_betaprime(density_ratio, floc)
        if regular_success:
            a, b, fitted_loc, fitted_scale = regular_params
            # Compute statistics
            aic_regular, bic_regular = compute_aic_bic(density_ratio, stats.betaprime, regular_params)
            ks_regular, p_regular = compute_ks_statistic(density_ratio, stats.betaprime, regular_params)
            print(f"Regular Fit for eps = {eps:.2f}: a={a:.4f}, b={b:.4f}, loc={fitted_loc:.4f}, scale={fitted_scale:.4f}")
            print(f"  AIC: {aic_regular:.2f}, BIC: {bic_regular:.2f}, KS Statistic: {ks_regular:.4f}, p-value: {p_regular:.4f}")

            # Plot regular fit
            y_regular = stats.betaprime.pdf(x, a, b, loc=fitted_loc, scale=fitted_scale)
            plt.plot(x, y_regular, color=colors[idx], linewidth=2, linestyle='-', label=f'eps={eps:.2f} Regular Fit')

            # Collect parameters for regression
            regression_data['eps'].append(eps)
            regression_data['a'].append(a)
            regression_data['b'].append(b)
            regression_data['loc'].append(fitted_loc)
            regression_data['scale'].append(fitted_scale)

            # Add to legend handles
            legend_handles.append(plt.Line2D([0], [0], color=colors[idx], linestyle='-', linewidth=2, label=f'eps={eps:.2f} Regular Fit'))
        else:
            print(f"Regular Fit failed for eps = {eps:.2f}")

        # Mixture Fit
        mixture_params, mixture_success = fit_betaprime_mixture(density_ratio)
        if mixture_success:
            # Compute statistics for mixture
            a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha = mixture_params
            # Compute log-likelihood
            pdf_mixture = betaprime_mixture_pdf(density_ratio, mixture_params)
            log_likelihood_mixture = np.sum(np.log(pdf_mixture + 1e-10))  # Add small value to avoid log(0)
            k = len(mixture_params)  # Number of parameters
            n = len(density_ratio)
            aic_mixture = 2 * k - 2 * log_likelihood_mixture
            bic_mixture = k * np.log(n) - 2 * log_likelihood_mixture

            # Compute KS statistic (mixture CDF = alpha*F1 + (1-alpha)*F2, NOT the PDF)
            cdf_fitted = lambda x_val: (alpha * stats.betaprime.cdf(x_val, a1, b1, loc=loc1, scale=scale1)
                                        + (1 - alpha) * stats.betaprime.cdf(x_val, a2, b2, loc=loc2, scale=scale2))
            ks_mixture, p_mixture = stats.kstest(density_ratio, cdf_fitted)

            print(f"Mixture Fit for eps = {eps:.2f}:")
            print(f"  Component 1: a={a1:.4f}, b={b1:.4f}, loc={loc1:.4f}, scale={scale1:.4f}")
            print(f"  Component 2: a={a2:.4f}, b={b2:.4f}, loc={loc2:.4f}, scale={scale2:.4f}")
            print(f"  Mixing Proportion (alpha): {alpha:.4f}")
            print(f"  AIC: {aic_mixture:.2f}, BIC: {bic_mixture:.2f}, KS Statistic: {ks_mixture:.4f}, p-value: {p_mixture:.4f}")

            # Plot mixture fit
            y_mixture = mixture_pdf_plot(x, mixture_params)
            plt.plot(x, y_mixture, color=colors[idx], linewidth=2, linestyle='--', label=f'eps={eps:.2f} Mixture Fit')

            # Add to legend handles
            legend_handles.append(plt.Line2D([0], [0], color=colors[idx], linestyle='--', linewidth=2, label=f'eps={eps:.2f} Mixture Fit'))
        else:
            print(f"Mixture Fit failed for eps = {eps:.2f}")

    plt.xlabel('Density Ratio (lambda\' / lambda0)')
    plt.ylabel('Density')
    plt.title('Overlapping Histograms and Beta-Prime Fit Models for Different Eps Values')

    # Create a unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)

    plt.xlim(left=x_min)  # Ensure the x-axis starts at min_floc
    plt.tight_layout()
    plt.show()

    return regression_data

def perform_linear_regression(regression_data):
    """
    Perform linear regression between eps and each Beta-Prime parameter.

    Parameters:
    - regression_data: Dictionary containing lists of eps and corresponding parameters.

    Returns:
    - None. Prints the regression results.
    """
    parameters = ['a', 'b', 'loc', 'scale']
    print("\nLinear Regression Results (Parameter vs Eps):\n")
    for param in parameters:
        x = regression_data['eps']
        y = regression_data[param]
        if len(x) < 2:
            print(f"Not enough data points for regression of {param} vs eps.")
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f"Parameter: {param}")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Standard Error: {std_err:.4f}\n")

def main():
    """
    Main function to execute the data loading, fitting, plotting, and regression.
    """
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area

    # Eps values from 1.10 to 1.40 with increment of 0.05
    eps_values = np.arange(1.10, 1.41, 0.05).round(2)
    data_dir = './simdata/v2/'  # Directory where data files are stored

    # Load data
    data_dict = load_data(eps_values, data_dir, N, radius)

    # Plot overlapping histograms with fits and collect regression data
    regression_data = plot_overlapping_histograms_with_fits(data_dict, eps_values)

    # Perform linear regression
    perform_linear_regression(regression_data)

if __name__ == "__main__":
    main()
