import numpy as np

import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt
from modules.simv2_data import load_fit_parameters, load_density_ratio

# Constants for floc linear dependence on eps
SLOPE = -11.4372
INTERCEPT = 19.8668



def plot_regular_fit(x, params, color, label):
    """
    Plot the regular Beta-Prime fit.

    Parameters:
    - x: Points at which to evaluate the PDF.
    - params: Fitted parameters of the regular Beta-Prime distribution.
    - color: Color for the plot.
    - label: Label for the plot.
    """
    a, b, loc, scale = params
    y = stats.betaprime.pdf(x, a, b, loc=loc, scale=scale)
    plt.plot(x, y, color=color, linewidth=2, linestyle='-', label=label)

def plot_mixture_fit(x, params, color, label):
    """
    Plot the mixture Beta-Prime fit.

    Parameters:
    - x: Points at which to evaluate the PDF.
    - params: Fitted parameters of the mixture Beta-Prime distribution.
    - color: Color for the plot.
    - label: Label for the plot.
    """
    a1, b1, loc1, scale1, a2, b2, loc2, scale2, alpha = params
    y = alpha * stats.betaprime.pdf(x, a1, b1, loc=loc1, scale=scale1) + \
        (1 - alpha) * stats.betaprime.pdf(x, a2, b2, loc=loc2, scale=scale2)
    plt.plot(x, y, color=color, linewidth=2, linestyle='--', label=label)

def plot_overlapping_histograms_with_fits(data_dir, output_dir, eps_start, eps_end, eps_step, mix=False, N=10000, radius=100.0):
    """
    Plot overlapping histograms of density ratios with fitted Beta-Prime models.

    Parameters:
    - data_dir: Directory where data files are stored.
    - output_dir: Directory where fit CSV files are stored.
    - eps_start: Starting value of eps.
    - eps_end: Ending value of eps.
    - eps_step: Step size for eps.
    - mix: Boolean flag to include mixture fits.
    - N: Number of points in the simulation.
    - radius: Radius of the circular area.
    """
    # Define eps values
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step).round(2)

    # Initialize plot
    plt.figure(figsize=(14, 10))

    # Prepare colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_values)))

    # Compute floc for each eps to determine x_min
    floc_values = SLOPE * eps_values + INTERCEPT
    min_floc = np.min(floc_values)
    print(f"Minimum floc across all eps: {min_floc:.4f}")

    # Determine x_max by aggregating data
    all_density_ratios = []
    for eps in eps_values:
        density_ratio_sample = load_density_ratio(eps, data_dir, N, radius)
        if density_ratio_sample is not None:
            all_density_ratios.extend(density_ratio_sample)
    if not all_density_ratios:
        print("No data available to determine x_max. Exiting.")
        return
    x_min = min_floc
    x_max = np.percentile(all_density_ratios, 99.99)  # Use 99.99th percentile to avoid extreme tails
    x = np.linspace(x_min, x_max, 1000)

    # Containers for legend handles to avoid duplicates
    legend_entries = {}

    for idx, eps in enumerate(eps_values):
        color = colors[idx]
        regular_fit, mixture_fit = load_fit_parameters(output_dir, eps)

        # Load and sample data
        density_ratio_sample = load_density_ratio(eps, data_dir, N, radius)
        if density_ratio_sample is None:
            continue

        # Plot histogram
        plt.hist(density_ratio_sample, bins=100, range=(x_min, x_max), density=True,
                 histtype='stepfilled', alpha=0.3, color=color, edgecolor='none',
                 label=f'eps = {eps:.2f}')

        # Plot Regular Fit
        if regular_fit is not None:
            a = regular_fit['a']
            b = regular_fit['b']
            loc = regular_fit['loc']
            scale = regular_fit['scale']
            label_regular = f'eps={eps:.2f} Regular Fit'
            plt.plot(x, stats.betaprime.pdf(x, a, b, loc=loc, scale=scale),
                     color=color, linewidth=2, linestyle='-', label=label_regular)
            legend_entries[label_regular] = plt.Line2D([0], [0], color=color, linestyle='-', linewidth=2)

        # Plot Mixture Fit if mix is True
        if mix and mixture_fit is not None:
            a1 = mixture_fit['a1']
            b1 = mixture_fit['b1']
            loc1 = mixture_fit['loc1']
            scale1 = mixture_fit['scale1']
            a2 = mixture_fit['a2']
            b2 = mixture_fit['b2']
            loc2 = mixture_fit['loc2']
            scale2 = mixture_fit['scale2']
            alpha_mix = mixture_fit['alpha']
            label_mixture = f'eps={eps:.2f} Mixture Fit'
            y_mixture = alpha_mix * stats.betaprime.pdf(x, a1, b1, loc=loc1, scale=scale1) + \
                        (1 - alpha_mix) * stats.betaprime.pdf(x, a2, b2, loc=loc2, scale=scale2)
            plt.plot(x, y_mixture, color=color, linewidth=2, linestyle='--', label=label_mixture)
            legend_entries[label_mixture] = plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2)

    plt.xlabel('Density Ratio (lambda\' / lambda0)')
    plt.ylabel('Density')
    plt.title('Overlapping Histograms and Beta-Prime Fit Models for Different Eps Values')

    # Create unique legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize='small', ncol=2)

    plt.xlim(left=x_min)  # Ensure the x-axis starts at min_floc
    plt.tight_layout()
    plt.show()

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Plot Beta-Prime Fits')
    parser.add_argument('--data_dir', type=str, default='./simdata/v2/', help='Directory where data files are stored. Default: ./simdata/v2/')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Directory where fit CSV files are stored. Default: ./results/')
    parser.add_argument('--eps_start', type=float, default=1.10, help='Starting value of eps. Default: 1.10')
    parser.add_argument('--eps_end', type=float, default=1.40, help='Ending value of eps. Default: 1.40')
    parser.add_argument('--eps_step', type=float, default=0.05, help='Step size for eps. Default: 0.05')
    parser.add_argument('--mix', action='store_true', help='Include mixture fits in the plots.')
    parser.add_argument('--N', type=int, default=10000, help='Number of points in the simulation. Default: 10000')
    parser.add_argument('--radius', type=float, default=100.0, help='Radius of the circular area. Default: 100.0')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_step = args.eps_step
    mix = args.mix
    N = args.N
    radius = args.radius

    # Plot fits
    plot_overlapping_histograms_with_fits(data_dir, output_dir, eps_start, eps_end, eps_step, mix, N, radius)

if __name__ == "__main__":
    main()
