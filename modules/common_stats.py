import numpy as np
import scipy.stats as stats

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
    # Define the CDF function based on fitted parameters
    cdf_fitted = lambda x_val: distribution.cdf(x_val, *params)
    ks_stat, p_value = stats.kstest(data, cdf_fitted)
    return ks_stat, p_value

