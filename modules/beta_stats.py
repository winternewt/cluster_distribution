import numpy as np
from scipy import stats
import pandas as pd
import os

def betaprime_mixture_pdf(x, params):
    """
    Compute the PDF of a mixture of Beta-Prime distributions with auto-detected number of components.

    Parameters:
    - x: array-like
        Points at which to evaluate the PDF.
    - params: list or array-like
        Parameters for the Beta-Prime mixture model.
        For each component, provide [a, b, loc, scale].
        After all component parameters, provide (N-1) mixing proportions.
        Total length should be 5*N - 1, where N is the number of components.

    Returns:
    - pdf: array-like
        The computed mixture PDF evaluated at points x.
    """
    total_params = len(params)

    # Determine the number of components (N)
    # Each component has 4 parameters, and there are (N-1) mixing proportions
    # Thus, total_params = 4*N + (N-1) => 5*N - 1 => N = (total_params + 1) / 5
    if (total_params + 1) % 5 != 0:
        raise ValueError("Incorrect number of parameters. It should satisfy 5*N - 1 = len(params).")

    N = (total_params + 1) // 5
    if N < 1:
        raise ValueError("Number of components must be at least 1.")

    # Extract component parameters
    component_params = []
    for i in range(N):
        idx = 5 * i
        a = params[idx]
        b = params[idx + 1]
        loc = params[idx + 2]
        scale = params[idx + 3]
        component_params.append({'a': a, 'b': b, 'loc': loc, 'scale': scale})

    # Extract mixing proportions
    if N == 1:
        mixing_proportions = [1.0]
    else:
        mixing_params = params[4 * N:]
        if len(mixing_params) != N - 1:
            raise ValueError(f"Expected {N - 1} mixing proportions, got {len(mixing_params)}.")

        # Ensure mixing proportions are between 0 and 1
        mixing_params = np.array(mixing_params)
        if np.any(mixing_params < 0) or np.any(mixing_params > 1):
            raise ValueError("Mixing proportions must be between 0 and 1.")

        last_proportion = 1.0 - np.sum(mixing_params)
        if last_proportion < 0:
            raise ValueError("Sum of mixing proportions exceeds 1.")

        mixing_proportions = list(mixing_params) + [last_proportion]

    # Compute the mixture PDF
    pdf = np.zeros_like(x, dtype=float)
    for comp, alpha in zip(component_params, mixing_proportions):
        pdf += alpha * stats.betaprime.pdf(x, comp['a'], comp['b'], loc=comp['loc'], scale=comp['scale'])

    return pdf


def negative_log_likelihood(params, data):
    """
    Compute the negative log-likelihood for the Beta-Prime distribution or mixture model with auto-detected number of components.

    Parameters:
    - params: list or array-like
        Parameters for the Beta-Prime distribution or mixture model.
        For each component, provide [a, b, loc, scale].
        After all component parameters, provide (N-1) mixing proportions.
        Total length should be 5*N - 1, where N is the number of components.
    - data: array-like
        The density ratio data.

    Returns:
    - nll: float
        Negative log-likelihood value.
    """
    total_params = len(params)

    # Determine the number of components (N)
    # Each component has 4 parameters, and there are (N-1) mixing proportions
    # Thus, total_params = 5*N -1 => N = (total_params +1) /5
    if (total_params + 1) % 5 != 0:
        raise ValueError("Incorrect number of parameters. It should satisfy 5*N - 1 = len(params).")

    N = (total_params + 1) // 5
    if N < 1:
        return np.inf  # Invalid number of components

    # Extract component parameters and mixing proportions
    try:
        pdf = betaprime_mixture_pdf(data, params)
    except ValueError as e:
        # If parameter extraction fails (e.g., invalid number of parameters), return infinity
        return np.inf

    # To avoid log(0), set a minimum value
    epsilon = 1e-10
    pdf = np.maximum(pdf, epsilon)

    return -np.sum(np.log(pdf))

def betaprime_pdfcdf_eps(x, eps, params):
    """
    Compute the PDF of the Beta-Prime distribution with parameters dependent on eps.

    Parameters:
    - x: array-like
        Points at which to evaluate the PDF.
    - eps: array-like
        Eps values corresponding to each point in x.
    - params: dict
        Dictionary containing regression parameters for 'a', 'b', 'loc', 'scale'.

    Returns:
    - pdf: array-like
        The computed PDF evaluated at points x.
    - cdf: array-like
        The computed PDF evaluated at points x.
    """
    # Calculate parameters based on eps
    a = params['a_slope'] * eps + params['a_intercept']
    b = params['b_slope'] * eps + params['b_intercept']
    loc = params['loc_slope'] * eps + params['loc_intercept']
    scale = params['scale_slope'] * eps + params['scale_intercept']

    # Ensure parameters are within valid ranges
    a = np.maximum(a, 1e-6)
    b = np.maximum(b, 1e-6)
    scale = np.maximum(scale, 1e-6)

    pdf = stats.betaprime.pdf(x, a, b, loc=loc, scale=scale)
    cdf = stats.betaprime.cdf(x, a, b, loc=loc, scale=scale)

    return pdf, cdf

def betaprime_pdf_eps(x, eps, params):
    """
    Compute the PDF of the Beta-Prime distribution with parameters dependent on eps.

    Parameters:
    - x: array-like
        Points at which to evaluate the PDF.
    - eps: array-like
        Eps values corresponding to each point in x.
    - params: dict
        Dictionary containing regression parameters for 'a', 'b', 'loc', 'scale'.

    Returns:
    - pdf: array-like
        The computed PDF evaluated at points x.
    """

    pdf, _ = betaprime_pdfcdf_eps(x, eps, params)
    return pdf

def betaprime_cdf_eps(x, eps, params):
    """
    Compute the PDF of the Beta-Prime distribution with parameters dependent on eps.

    Parameters:
    - x: array-like
        Points at which to evaluate the PDF.
    - eps: array-like
        Eps values corresponding to each point in x.
    - params: dict
        Dictionary containing regression parameters for 'a', 'b', 'loc', 'scale'.

    Returns:
    - cdf: array-like
        The computed PDF evaluated at points x.
    """
    _, cdf = betaprime_pdfcdf_eps(x, eps, params)
    return cdf


def negative_log_likelihood_eps(params_array, x, eps):
    """
    Compute the negative log-likelihood for the Beta-Prime distribution with parameters dependent on eps.
    """
    params = {
        'a_slope': params_array[0],
        'a_intercept': params_array[1],
        'b_slope': params_array[2],
        'b_intercept': params_array[3],
        'loc_slope': params_array[4],
        'loc_intercept': params_array[5],
        'scale_slope': params_array[6],
        'scale_intercept': params_array[7]
    }

    pdf = betaprime_pdf_eps(x, eps, params)

    # To avoid log(0), set a minimum value
    pdf = np.maximum(pdf, 1e-10)

    return -np.sum(np.log(pdf))


def perform_linear_regression(regression_data, output_dir, return_params=False):
    """
    Perform linear regression between eps and each Beta-Prime parameter.

    Parameters:
    - regression_data: Dictionary containing lists of eps and corresponding parameters.
    - output_dir: Directory to save regression plots and parameters.
    - return_params: Boolean indicating whether to return the regression parameters.

    Returns:
    - regression_params: Dictionary of regression coefficients if return_params is True.
    """
    import matplotlib.pyplot as plt
    parameters = ['a', 'b', 'loc', 'scale']
    regression_params = {}
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

        # Save regression parameters
        regression_params[param] = (slope, intercept)

        # Generate plot
        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, np.array(x) * slope + intercept, 'r', label='Fitted line')
        plt.xlabel('eps')
        plt.ylabel(param)
        plt.title(f'Linear Regression of {param} vs eps')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{param}_vs_eps.png'))
        plt.close()

    # Save regression parameters to file
    regression_params_df = pd.DataFrame({
        'parameter': parameters,
        'slope': [regression_params[param][0] for param in parameters],
        'intercept': [regression_params[param][1] for param in parameters]
    })
    regression_params_df.to_csv(os.path.join(output_dir, 'regression_params.csv'), index=False)

    if return_params:
        return regression_params

