import numpy as np
import pandas as pd
import os
import time
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
    unique_labels = set(labels)
 #   print(f"Iteration {iteration}: Unique labels {unique_labels}")
    clusters = extract_cluster_properties(points, labels, min_cluster_size, min_area)
    if not clusters:
#        print(f"Iteration {iteration}: No clusters found after filtering.")
        return [{'S_prime': -1, 'N_prime': -1, 'iteration': iteration}]
    else:
#        print(f"Iteration {iteration}: {len(clusters)} clusters found.")
        data = [{'S_prime': cluster['S_prime'], 'N_prime': cluster['N_prime'], 'iteration': iteration} for cluster in clusters]
        return data

def simulate_parallel(N, radius, eps, min_samples, min_cluster_size, min_area, total_iterations, n_processes, data_file):
    iteration = 0
    start_time = time.time()
    rng = np.random.default_rng()
    if os.path.exists(data_file):
        print(f"Resuming simulation from existing data file: {data_file}")
        df_existing = pd.read_csv(data_file)
        iteration = df_existing['iteration'].max()
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
            # Save data
            df = pd.DataFrame(data_batch)
            if iteration == 0 and not os.path.exists(data_file):
                df.to_csv(data_file, index=False, mode='w')
            else:
                df.to_csv(data_file, index=False, mode='a', header=False)
            iteration += batch_size
            avg_time_per_iter = (time.time() - start_time) / iteration
            print(f"Iteration: {iteration}, Avg Time per Iteration: {avg_time_per_iter:.5f} seconds")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print(f"Total iterations completed: {iteration}")
        print(f"Total simulation time: {total_time:.2f} seconds")
        print(f"Data saved to {data_file}")

def main():
    # Simulation parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    eps = 1.6  # DBSCAN epsilon parameter
    min_samples = 10  # DBSCAN min_samples parameter
    min_cluster_size = min_samples  # Minimum number of points in a cluster
    min_area = 0.5  # Minimum cluster area
    total_iterations = 100000  # Total number of iterations to run
    n_processes = 16  # Number of processes to use for multiprocessing

    # Create a unique data filename based on parameters
    data_file = f'simulation_data_N{N}_radius{radius}_eps{eps}.csv'

    simulate_parallel(N, radius, eps, min_samples, min_cluster_size, min_area, total_iterations, n_processes, data_file)

if __name__ == "__main__":
    main()
