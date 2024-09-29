import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import lognorm, betaprime
import warnings
import matplotlib.pyplot as plt  # For optional plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    results = []
    # Fit each distribution
    for name, distribution in distributions.items():
        # Estimate parameters with specified floc
        try:
            params = distribution.fit(ratio_sample, floc=floc_value)
            # Perform K-S test
            ks_stat, ks_p = stats.kstest(ratio_sample, distribution.cdf, args=params)
            # Compute AIC and BIC
            log_likelihood = np.sum(distribution.logpdf(ratio_sample, *params))
            k = len(params)  # Number of parameters
            n = len(ratio_sample)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            results.append({
                'distribution': name,
                'floc': floc_value,
                'params': params,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'aic': aic,
                'bic': bic
            })
        except Exception as e:
            # Handle exceptions, e.g., if the distribution cannot be fitted with the given floc
            results.append({
                'distribution': name,
                'floc': floc_value,
                'params': None,
                'ks_stat': np.nan,
                'ks_p': np.nan,
                'aic': np.inf,
                'bic': np.inf
            })
    return results

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area

    # Eps values to include in the analysis (from 1.10 to 1.40 in steps of 0.05)
    eps_values = np.arange(1.10, 1.42, 0.01).round(2)
    # floc values from 4.0 to 6.4 in steps of 0.1
    slope=-9.71
    inter=17.1674
#    r2=0.9945
#    RMSE=0.0667
#    floc_values = np.arange(4.0, 6.5, 0.1).round(1)
    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Define distributions to fit
    distributions = {
        # 'Log-Normal': lognorm,  # Uncomment if you wish to include Log-Normal
        'Beta-Prime': betaprime
    }

    # Initialize a list to store all results
    all_results = []

    # Loop over each eps value
    for eps_value in eps_values:
        print(f"\nProcessing eps = {eps_value:.2f}")
        df_valid, ratio_data = load_data(eps_value, data_dir, N, radius)
        if ratio_data is None:
            continue
        floc_est=slope*eps_value+inter
        floc_values = np.arange(floc_est - 0.12, floc_est + 0.12, 0.005).round(3)
        # For computational efficiency, sample the data
        sample_size = min(100000, len(ratio_data))  # Use up to 100,000 data points
        np.random.seed(42)
        ratio_sample = np.random.choice(ratio_data, size=sample_size, replace=False)

        # Loop over floc values
        for floc_value in floc_values:
            # Fit distributions
            fit_results = fit_distributions(ratio_sample, distributions, floc_value)
            # Add eps and floc information to the results
            for res in fit_results:
                res['eps'] = eps_value
                all_results.append(res)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # For each eps value, determine the best fit
    summary_results = []
    for eps_value in eps_values:
        eps_results = results_df[results_df['eps'] == eps_value]
        for dist_name in distributions.keys():
            dist_results = eps_results[eps_results['distribution'] == dist_name]
            if dist_results.empty:
                continue  # Skip if no results for this distribution
            # Find the best floc value based on AIC
            valid_dist_results = dist_results.dropna(subset=['aic'])
            if valid_dist_results.empty:
                continue  # Skip if no valid AIC values
            best_fit = valid_dist_results.loc[valid_dist_results['aic'].idxmin()]
            summary_results.append({
                'eps': eps_value,
                'distribution': dist_name,
                'best_floc': best_fit['floc'],
                'params': best_fit['params'],
                'aic': best_fit['aic'],
                'bic': best_fit['bic'],
                'ks_stat': best_fit['ks_stat'],
                'ks_p': best_fit['ks_p']
            })

    # Convert summary results to a DataFrame
    summary_df = pd.DataFrame(summary_results)

    # Now, for each eps, find the best overall fit
    best_overall_results = []
    for eps_value in eps_values:
        eps_summary = summary_df[summary_df['eps'] == eps_value]
        if eps_summary.empty:
            continue  # Skip if no summary for this eps
        # Find the best distribution based on AIC
        best_fit = eps_summary.loc[eps_summary['aic'].idxmin()]
        best_overall_results.append(best_fit)

    # Convert best overall results to a DataFrame
    best_overall_df = pd.DataFrame(best_overall_results)

    # Print the results
    print("\nBest Fit Results for Each eps:")
    for res in best_overall_results:
        print(f"\neps = {res['eps']:.2f}")
        print(f"Best Distribution: {res['distribution']}")
        print(f"Best floc: {res['best_floc']}")
        print(f"Fit Parameters: {res['params']}")
        print(f"AIC: {res['aic']:.2f}")
        print(f"BIC: {res['bic']:.2f}")
        print(f"KS Statistic: {res['ks_stat']:.5f}")
        print(f"KS p-value: {res['ks_p']:.5f}")

    # ================== Linear Regression Analysis ==================

    # Ensure there are enough data points
    if len(best_overall_df) < 2:
        print("\nNot enough data points to perform linear regression.")
        return

    # Extract eps and best_floc values
    X = best_overall_df['eps'].values.reshape(-1, 1)  # Independent variable
    y = best_overall_df['best_floc'].values  # Dependent variable

    # Perform linear regression using sklearn
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate residuals and standard error
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = r2  # Already calculated
    mse = ss_res / len(y)
    rmse = np.sqrt(mse)

    # Alternatively, using scipy.stats.linregress
    # slope, intercept, r_value, p_value, std_err = stats.linregress(best_overall_df['eps'], best_overall_df['best_floc'])
    # r_squared = r_value**2

    # Print regression results
    print("\nLinear Regression: floc = slope * eps + intercept")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

    # Optionally, plot the regression
    plt.figure(figsize=(8,6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='red', label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Best floc')
    plt.title('Linear Regression of floc vs eps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================== End of Linear Regression Analysis ==================

    # Optionally, save the results to a CSV file
    # summary_df.to_csv('distribution_fitting_summary.csv', index=False)
    # best_overall_df.to_csv('best_overall_fits.csv', index=False)

if __name__ == "__main__":
    main()
