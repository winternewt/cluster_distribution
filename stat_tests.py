import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Function to perform normality tests on a continuous variable
def perform_normality_tests(data_sample):
    results = {}
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data_sample)
    results['Shapiro-Wilk'] = {'Statistic': shapiro_stat, 'p-value': shapiro_p}
    # Anderson-Darling Test
    anderson_result = stats.anderson(data_sample, dist='norm')
    results['Anderson-Darling'] = anderson_result
    # Kolmogorov-Smirnov Test against normal distribution
    ks_stat, ks_p = stats.kstest(data_sample, 'norm', args=(np.mean(data_sample), np.std(data_sample)))
    results['Kolmogorov-Smirnov'] = {'Statistic': ks_stat, 'p-value': ks_p}
    return results

# Function to perform Poisson goodness-of-fit test on a discrete variable
def perform_poisson_tests(data_counts):
    results = {}
    # Estimate lambda (mean of counts)
    lambda_estimate = np.mean(data_counts)
    # Observed frequencies
    count_values, count_frequencies = np.unique(data_counts, return_counts=True)
    # Expected frequencies under Poisson distribution
    # Calculate Poisson probabilities for the observed count values
    expected_probs = stats.poisson.pmf(count_values, mu=lambda_estimate)
    # Expected frequencies
    expected_frequencies = expected_probs * len(data_counts)
    # Adjust expected frequencies to ensure sums match
    expected_frequencies *= (count_frequencies.sum() / expected_frequencies.sum())
    # Chi-squared test
    chi_stat, chi_p = stats.chisquare(f_obs=count_frequencies, f_exp=expected_frequencies)
    results['Chi-squared'] = {'Statistic': chi_stat, 'p-value': chi_p}
    # Kolmogorov-Smirnov Test (Not ideal for discrete distributions, but included for completeness)
    # Note: K-S test may not be appropriate here due to the discrete nature of the data
    # Alternative tests like the Cramér–von Mises test may be considered
    ks_stat, ks_p = stats.kstest(data_counts, 'poisson', args=(lambda_estimate,))
    results['Kolmogorov-Smirnov'] = {'Statistic': ks_stat, 'p-value': ks_p}
    return results

def analyze_eps_values(eps_values, data_dir, lambda0, S0, min_cluster_size, subsample_size=5000, batch_size=10):
    # Lists to store results
    all_results = []
    summary_data = []

    # Create directories for plots
    ratio_plot_dir = os.path.join(data_dir, 'ratio_plots')
    N_prime_plot_dir = os.path.join(data_dir, 'N_prime_plots')
    os.makedirs(ratio_plot_dir, exist_ok=True)
    os.makedirs(N_prime_plot_dir, exist_ok=True)

    # Process eps values in batches
    for batch_start in range(0, len(eps_values), batch_size):
        batch_eps = eps_values[batch_start:batch_start + batch_size]

        # Initialize figures for batch plots
        fig_ratio, axes_ratio = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        fig_N_prime, axes_N_prime = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

        for idx, eps in enumerate(batch_eps):
            print(f"\nAnalyzing eps = {eps:.2f}")
            data_file = os.path.join(data_dir, f'simulation_data_N10000_radius100_eps{eps:.2f}.csv')
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

            # Compute lambda_prime and ratio
            df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
            df_valid['ratio'] = df_valid['lambda_prime'] / lambda0

            # Analyze 'ratio' (continuous variable)
            # Subsample if necessary
            ratio_data = df_valid['ratio']
            if len(ratio_data) > subsample_size:
                ratio_sample = ratio_data.sample(n=subsample_size, random_state=42)
            else:
                ratio_sample = ratio_data

            # Perform normality tests on 'ratio'
            ratio_tests = perform_normality_tests(ratio_sample)

            # Histogram and KDE for 'ratio'
            ax_ratio = axes_ratio.flatten()[idx]
            sns.histplot(ratio_data, bins=50, kde=True, ax=ax_ratio)
            ax_ratio.set_xlabel('Density Ratio (lambda\' / lambda0)')
            ax_ratio.set_ylabel('Frequency')
            ax_ratio.set_title(f'eps = {eps:.2f}')

            # Analyze 'N_prime' (discrete variable)
            N_prime_data = df_valid['N_prime']

            # Perform Poisson goodness-of-fit test on 'N_prime'
            N_prime_tests = perform_poisson_tests(N_prime_data)

            # Histogram for 'N_prime'
            ax_N_prime = axes_N_prime.flatten()[idx]
            bins = np.arange(N_prime_data.min(), N_prime_data.max() + 2) - 0.5
            sns.histplot(N_prime_data, bins=bins, stat='probability', ax=ax_N_prime)
            ax_N_prime.set_xlabel('Number of Points in Cluster (N\')')
            ax_N_prime.set_ylabel('Probability')
            ax_N_prime.set_title(f'eps = {eps:.2f}')

            # Collect results
            result = {
                'eps': eps,
                'ratio_tests': ratio_tests,
                'N_prime_tests': N_prime_tests,
                'lambda_prime_mean': df_valid['lambda_prime'].mean(),
                'lambda_prime_max': df_valid['lambda_prime'].max(),
                'N_prime_mean': N_prime_data.mean(),
                'N_prime_max': N_prime_data.max(),
                'num_clusters': len(df_valid)
            }
            all_results.append(result)

            summary_data.append({
                'eps': eps,
                'lambda_prime_mean': df_valid['lambda_prime'].mean(),
                'lambda_prime_max': df_valid['lambda_prime'].max(),
                'N_prime_mean': N_prime_data.mean(),
                'N_prime_max': N_prime_data.max(),
                'num_clusters': len(df_valid),
                'ratio_shapiro_p': ratio_tests['Shapiro-Wilk']['p-value'],
                'ratio_ks_p': ratio_tests['Kolmogorov-Smirnov']['p-value'],
                'N_prime_chi_p': N_prime_tests['Chi-squared']['p-value'],
                'N_prime_ks_p': N_prime_tests['Kolmogorov-Smirnov']['p-value']
            })

        # Adjust layout and save batch plots
        fig_ratio.tight_layout()
        fig_N_prime.tight_layout()
        ratio_batch_file = os.path.join(ratio_plot_dir, f'ratio_plots_batch_{batch_start // batch_size + 1}.png')
        N_prime_batch_file = os.path.join(N_prime_plot_dir, f'N_prime_plots_batch_{batch_start // batch_size + 1}.png')
        fig_ratio.savefig(ratio_batch_file)
        fig_N_prime.savefig(N_prime_batch_file)
        plt.close(fig_ratio)
        plt.close(fig_N_prime)
        print(f"Saved batch {batch_start // batch_size + 1} plots.")

    # Save summary data
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(data_dir, 'analysis_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nAnalysis completed. Summary saved to '{summary_csv}'.")

    return all_results

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    min_samples = 10  # DBSCAN min_samples parameter
    min_cluster_size = 10  # Minimum number of points in a cluster
    min_area = 0.5  # Minimum cluster area

    # Constants
    S0 = np.pi * radius**2  # Total area of the circle
    lambda0 = N / S0  # Initial point density

    # Eps values to analyze
    eps_values = np.arange(0.90, 1.40, 0.01).round(2)  # Ensure rounding consistency

    # Directory where data files are stored
    data_dir = './simdata/v2/'  # Adjust if data files are in a different directory

    # Analyze the data and generate plots
    all_results = analyze_eps_values(
        eps_values=eps_values,
        data_dir=data_dir,
        lambda0=lambda0,
        S0=S0,
        min_cluster_size=min_cluster_size,
        subsample_size=5000,
        batch_size=10  # Number of eps values per batch
    )

if __name__ == "__main__":
    main()
