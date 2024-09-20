import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

def generate_points_in_circle(N, radius, rng):
    r = radius * np.sqrt(rng.uniform(size=N))
    theta = 2 * np.pi * rng.uniform(size=N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def compute_cluster_area(cluster_points):
    if len(cluster_points) < 3:
        return 0  # Area is zero if less than 3 points
    hull = ConvexHull(cluster_points)
    return hull.volume  # 'volume' returns area in 2D

def extract_cluster_properties(points, labels, min_cluster_size, min_area):
    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # Skip noise
        cluster_points = points[labels == label]
        N_prime = len(cluster_points)
        if N_prime < min_cluster_size:
            continue  # Skip small clusters
        S_prime = compute_cluster_area(cluster_points)
        if S_prime < min_area:
            continue  # Skip clusters with small area
        clusters.append({'S_prime': S_prime, 'N_prime': N_prime})
    return clusters

def simulate_once(args):
    N, radius, eps, min_samples, min_cluster_size, min_area, iteration, seed = args
    rng = np.random.default_rng(seed)
    points = generate_points_in_circle(N, radius, rng)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = extract_cluster_properties(points, labels, min_cluster_size, min_area)
    if not clusters:
        return [{'S_prime': -1, 'N_prime': -1, 'iteration': iteration}]
    else:
        data = [{'S_prime': cluster['S_prime'], 'N_prime': cluster['N_prime'], 'iteration': iteration} for cluster in clusters]
        return data

def simulate_parallel(N, radius, eps, min_samples, min_cluster_size, min_area, total_iterations, n_processes, data_file):
    iteration = 0
    clusters_found = 0
    start_time = time.time()
    rng = np.random.default_rng()
    if os.path.exists(data_file):
        print(f"Resuming simulation from existing data file: {data_file}")
        df_existing = pd.read_csv(data_file)
        iteration = df_existing['iteration'].max()
        clusters_found = len(df_existing[df_existing['S_prime'] != -1])
        print(f"Starting from iteration {iteration + 1}")
    else:
        print("Starting new simulation.")

    try:
        while iteration < total_iterations:
            batch_size = min(1000, total_iterations - iteration)
            seeds = rng.integers(0, 1e9, size=batch_size)
            args_list = []
            for i in range(batch_size):
                iter_num = iteration + i + 1
                seed = seeds[i]
                args_list.append((N, radius, eps, min_samples, min_cluster_size, min_area, iter_num, seed))
            with Pool(processes=n_processes) as pool:
                results = pool.map(simulate_once, args_list)
            # Flatten the list of lists
            data_batch = [item for sublist in results for item in sublist]
            # Count clusters found in this batch
            valid_clusters = [d for d in data_batch if d['S_prime'] != -1]
            clusters_found += len(valid_clusters)
            # Save data
            df = pd.DataFrame(data_batch)
            if iteration == 0 and not os.path.exists(data_file):
                df.to_csv(data_file, index=False, mode='w')
            else:
                df.to_csv(data_file, index=False, mode='a', header=False)
            iteration += batch_size
            avg_time_per_iter = (time.time() - start_time) / iteration
            print(f"Iteration: {iteration}, Clusters found so far: {clusters_found}, Avg Time per Iteration: {avg_time_per_iter:.5f} seconds")

            # Termination criterion: If clusters_found is zero after total_iterations, return 0
            if iteration >= total_iterations and clusters_found == 0:
                return 0
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print(f"Total iterations completed: {iteration}")
        print(f"Total simulation time: {total_time:.2f} seconds")
        print(f"Data saved to {data_file}")

    return clusters_found

def plot_experiment_results(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['eps'], results_df['avg_ratio'], label='Average Ratio')
    plt.plot(results_df['eps'], results_df['max_ratio'], label='Maximum Ratio')
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Ratio (lambda\'/lambda0 * S\'/S0)')
    plt.title('Signal-to-Noise Ratio vs. Epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    min_samples = 10  # DBSCAN min_samples parameter
    min_cluster_size = 10  # Minimum number of points in a cluster
    min_area = 0.5  # Minimum cluster area
    total_iterations_per_eps = 1000  # Number of iterations to run per eps
    n_processes = 16  # Number of processes to use for multiprocessing

    # Constants
    S0 = np.pi * radius**2  # Total area of the circle
    lambda0 = N / S0  # Initial point density

    # Eps values to iterate over
    eps_values = np.arange(1.0, 10.0, 0.01)  # Adjust the upper limit as needed

    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=['eps', 'avg_ratio', 'max_ratio', 'max_S_prime', 'clusters_found'])

    for eps in eps_values:
        print(f"\nStarting simulations for eps = {eps:.2f}")

        data_file = f'simulation_data_N{N}_radius{radius}_eps{eps:.2f}.csv'

        # Run simulations for the current eps
        clusters_found = simulate_parallel(
            N, radius, eps, min_samples, min_cluster_size, min_area,
            total_iterations_per_eps, n_processes, data_file
        )

        # Termination criteria: If 1000 iterations yielded 0 clusters
        if clusters_found == 0:
            print(f"No clusters found for eps = {eps:.2f}. Increasing eps.")
            continue

        # Read the data
        df = pd.read_csv(data_file)
        # Filter out placeholder entries
        df_valid = df[(df['S_prime'] != -1) & (df['N_prime'] != -1)]

        # Compute lambda' and ratios
        df_valid['lambda_prime'] = df_valid['N_prime'] / df_valid['S_prime']
        df_valid['ratio'] = (df_valid['lambda_prime'] / lambda0) * (df_valid['S_prime'] / S0)

        # Check if the largest S' exceeds 1/4 * pi * r^2
        max_S_prime = df_valid['S_prime'].max()
        if max_S_prime > (0.25 * S0):
            print(f"Max S' ({max_S_prime}) exceeds 1/4 * S0 for eps = {eps:.2f}. Stopping eps increase.")
            break

        # Compute average and maximum ratio
        avg_ratio = df_valid['ratio'].mean()
        max_ratio = df_valid['ratio'].max()

        # Store the results
        results_df = results_df.append({
            'eps': eps,
            'avg_ratio': avg_ratio,
            'max_ratio': max_ratio,
            'max_S_prime': max_S_prime,
            'clusters_found': clusters_found
        }, ignore_index=True)

    # Save the results
    results_df.to_csv('simdata/eps_experiment_results.csv', index=False)
    print("\nExperiment completed. Results saved to 'eps_experiment_results.csv'.")

    # Plot the results
    plot_experiment_results(results_df)

if __name__ == "__main__":
    main()
